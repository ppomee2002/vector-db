"""공통 실행기.

    /workspace/jaeeun/venv/bin/python main.py --run runs/b0_smoke.yaml [--set c2.impl=bc1_kim ...]

순서:
 1. config.yaml(계약) + 실행 overlay + --set 합치기, 어댑터 이름(c1/c2/c3.impl) 확인
 2. ChunkMeta 로드(길이 1,000,000, config에 적힌 md5 검사), 질문 로드, 고정 seed 순서 생성
 3. Milvus 사용 중 검사(서버 메트릭만, 다른 사람 컬렉션 API 호출 없음) → 사용 중이면 멈추고 보고(exit 3)
 4. 허용 목록(run.milvus_allowed_load)의 컬렉션만 load → C3.run_many(검색 → C2 → C1) → finally release
 5. results.jsonl(QueryResult, 본문 없음), eval.json(정답 문자열 포함률), run_meta.json, summary.json 저장
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from interfaces import STAGE_KEYS, QueryResult  # noqa: E402
from modules import registry  # noqa: E402
from modules.chunk_meta import FileChunkMetaLoader  # noqa: E402
from modules.config_loader import load_config, settings_from  # noqa: E402
from modules.evaluate import answer_hit, hit_rate, read_chunks_jsonl  # noqa: E402
from modules.pipeline import Pipeline  # noqa: E402
from modules.queries import QueryLoader  # noqa: E402
from modules.result_checks import check_all, percentiles  # noqa: E402
from modules.run_meta import KST, MilvusGuard, now_kst, scrape, server_activity, summarize_metrics, swap_used  # noqa: E402
from modules.search_milvus import MilvusSearchAdapter  # noqa: E402

RUN_KEYS = ("name", "query_set", "n_queries", "max_queries", "warmup_s", "duration_s", "evaluate", "text_check_n",
            "verify_search_doc_idx", "milvus_allowed_load", "preflight_sample_s", "metrics_url", "out_dir")
# 기본값은 config.yaml의 run 블록에 있다. 실행별 값은 runs/*.yaml overlay나 --set으로 덮어쓴다.


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def _jsonable(o):
    if isinstance(o, tuple):
        return list(o)
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(type(o).__name__)


async def _aclose_all(objs):
    for o in objs:
        f = getattr(o, "aclose", None)
        if f is not None:
            try:
                await f()
            except Exception as e:  # noqa: BLE001
                print(f"[warn] aclose 실패 {type(o).__name__}: {e}", file=sys.stderr)


async def execute(cfg, run, settings, meta, ql, keys, adapters, texts_store, meta_out):
    c1, c2, c3 = adapters["c1"], adapters["c2"], adapters["c3"]
    search = MilvusSearchAdapter(meta, cfg, queries=ql)   # 생성자 규약 (meta, cfg)
    meta_out["search"] = search.describe()

    def sink(key, texts):
        texts_store[key] = dict(texts)

    pipe = Pipeline(settings, search, c2, c1, text_sink=sink if run["evaluate"] else None)
    try:
        # D9: search_topk > context_topk 일 때 상위 10개 일치 확인 (B0는 해당 없음)
        chk = cfg["ann"].get("c2_compare", {}).get("check_top10_consistency")
        if chk and settings.search_topk > settings.context_topk:
            n_same, sample = 0, keys[: cfg["experiment"]["smoke_queries"]]
            for k in sample:
                a = await search.search(k, topk=settings.search_topk, ef=settings.ann_ef)
                b = await search.search(k, topk=settings.context_topk, ef=settings.ann_ef)
                n_same += [h.chunk_id for h in a[: settings.context_topk]] == [h.chunk_id for h in b]
            meta_out["top10_consistency"] = {"n": len(sample), "n_same": n_same}
        else:
            meta_out["top10_consistency"] = f"n/a (search_topk={settings.search_topk}, context_topk={settings.context_topk})"

        kw = dict(mode=settings.scheduler_mode, clients=settings.clients, cap=settings.concurrency_cap,
                  warmup_s=float(run["warmup_s"]), duration_s=float(run["duration_s"]),
                  max_queries=None if run["max_queries"] is None else int(run["max_queries"]))   # C3 계약
        meta_out["c3_call"] = kw
        m0 = summarize_metrics(scrape(run["metrics_url"]))
        t0 = time.perf_counter()
        results = await c3.run_many(keys, pipe.run_one, **kw)
        wall = time.perf_counter() - t0
        m1 = summarize_metrics(scrape(run["metrics_url"]))
        meta_out["wall_s"] = wall
        my = (search.db, search.collection)
        meta_out["milvus_metrics_delta"] = {
            "note": "서버 전체 값. 질문 단위로 쓰지 않음(D11)",
            "cache_cell_access_miss_bytes": m1["cache_miss_bytes"] - m0["cache_miss_bytes"],
            "my_search_requests": m1["req_total"].get((*my, "Search"), 0) - m0["req_total"].get((*my, "Search"), 0),
            "milvus_cpu_s": (m1["cpu_s"] or 0) - (m0["cpu_s"] or 0),
        }
        return results
    finally:
        await _aclose_all([search, c1, c2, c3])


def summarize(results: list[QueryResult], settings, run, wall_s) -> dict:
    meas = [r for r in results if not r.is_warmup]
    ok = [r for r in meas if not r.error]
    st = {k: percentiles([r.stage_ms[k] for r in ok if k in r.stage_ms]) for k in STAGE_KEYS
          if any(k in r.stage_ms for r in ok)}
    dur = float(run["duration_s"])
    time_based = run["max_queries"] is None
    fs = [r.fetch_stats for r in ok if r.fetch_stats]
    return {
        "n_results": len(results), "n_warmup": len(results) - len(meas), "n_measured": len(meas),
        "n_error": sum(1 for r in meas if r.error),
        "error_rate": (sum(1 for r in meas if r.error) / len(meas)) if meas else None,
        "qps": (len(meas) / dur) if time_based else (len(meas) / wall_s if wall_s else None),
        "qps_basis": "duration_s" if time_based else "wall_s (max_queries 개수 기반 실행)",
        "stage_ms": st,
        "fetched_chunks_mean": (sum(len(r.fetched_ids) for r in ok) / len(ok)) if ok else None,
        "fetch_n_requests_mean": (sum(f.n_requests for f in fs) / len(fs)) if fs else None,
        "fetched_bytes_mean": (sum(f.bytes_returned for f in fs) / len(fs)) if fs else None,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="실행 overlay yaml (runs/*.yaml)")
    ap.add_argument("--set", action="append", default=[], help="a.b.c=값 (여러 번 가능)")
    ap.add_argument("--list", action="store_true", help="등록된 어댑터 이름 출력 후 종료")
    args = ap.parse_args(argv)

    if args.list:
        print(json.dumps(registry.available(), ensure_ascii=False, indent=1))
        errs = registry.discover()
        for f, e in errs.items():
            print(f"[import 실패] modules/{f}.py: {e.strip().splitlines()[-1]}")
        return 0

    signal.signal(signal.SIGTERM, _raise_interrupt)
    cfg, overlay = load_config(args.run, args.set)
    run = dict(cfg.get("run") or {})
    missing = [k for k in RUN_KEYS if k not in run]
    if missing:
        raise KeyError(f"config run 블록에 키가 없음: {missing}")
    run["warmup_s"] = cfg["c3"]["warmup_s"] if run["warmup_s"] is None else run["warmup_s"]
    run["duration_s"] = cfg["c3"]["duration_s"] if run["duration_s"] is None else run["duration_s"]
    cfg["run"] = run
    settings = settings_from(cfg)
    impls = registry.impl_names(cfg)

    ts = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    out = ROOT / run["out_dir"] / f"{run['name']}_{ts}"
    out.mkdir(parents=True, exist_ok=False)
    meta_out: dict = {"measured_at_kst": {"start": now_kst()}, "argv": sys.argv, "pid": os.getpid(),
                      "settings": asdict(settings), "impl": impls, "overlay": overlay, "config_merged": cfg,
                      "milvus_reset_log": "not collected (관리자 스위치 조회는 ssh 필요, CHECK.md 참고)"}

    def dump_meta():
        (out / "run_meta.json").write_text(json.dumps(meta_out, ensure_ascii=False, indent=1, default=_jsonable))

    print(f"[{now_kst()}] out={out}")
    print(f"  impl={impls}  search={settings.search_collection}  topk={settings.search_topk}/{settings.context_topk} ef={settings.ann_ef}")

    # 2. ChunkMeta, 질문
    t = time.perf_counter()
    loader = FileChunkMetaLoader(cfg, check_md5=True)
    meta = loader.load(layouts=(settings.layout,))
    meta_out["md5_check"] = loader.report
    meta_out["chunk_meta_load_s"] = round(time.perf_counter() - t, 1)
    ql = QueryLoader(cfg)
    qd = ql.load(run["query_set"])
    keys = ql.ordered_keys(run["query_set"], cfg["experiment"]["seed"], run["n_queries"])
    meta_out["queries"] = {"query_set": run["query_set"], "seed": cfg["experiment"]["seed"],
                           "n_keys": len(keys), "keys_head": keys[:20], "info": qd.info}
    print(f"  ChunkMeta ok ({meta_out['chunk_meta_load_s']}s), keys={len(keys)} head={keys[:3]}")

    # 어댑터 생성 (모든 어댑터 생성자 규약: (meta, cfg))
    adapters = {k: registry.build(k, impls[k], meta, cfg) for k in ("c1", "c2", "c3")}
    meta_out["adapters"] = {k: (a.describe() if hasattr(a, "describe") else type(a).__name__)
                            for k, a in adapters.items()}
    # 실제 Fetch 출처 (C1 계약 fetch_source). Settings.fetch_collection과 다르면 둘 다 남긴다.
    try:
        src = adapters["c1"].fetch_source()
    except Exception as e:  # noqa: BLE001
        src = f"unknown (fetch_source 실패: {type(e).__name__}: {e})"
    fc = settings.fetch_collection
    same = src == f"milvus:{fc}"
    meta_out["fetch_source"] = {"c1_impl": impls["c1"], "source": src, "settings_fetch_collection": fc,
                                "matches_settings": same}
    if not same:
        meta_out["fetch_source"]["note"] = ("실제 본문 출처가 Settings.fetch_collection(config milvus.fetch)과 다름. "
                                            "결과 해석은 source 기준")

    # 3. Milvus 사용 중 검사 → load
    targets = [settings.search_collection]
    for a in adapters.values():
        if hasattr(a, "milvus_collections"):
            targets += [c for c in a.milvus_collections() if c not in targets]
    guard = MilvusGuard(cfg["milvus"]["uri"], run["metrics_url"], run["milvus_allowed_load"])
    meta_out["milvus_targets"] = targets
    not_allowed = [c for c in targets if c not in guard.allowed]
    if not_allowed:
        meta_out["aborted"] = f"load 허용 목록에 없는 컬렉션: {not_allowed} (run.milvus_allowed_load)"
        dump_meta()
        print("[abort]", meta_out["aborted"])
        return 2
    free, act = guard.preflight(targets, sample_s=float(run["preflight_sample_s"]))
    meta_out["load_state"] = {"before": act["my_load_state"], "server_loaded_before": act["loaded_collection_num"]}
    meta_out["other_user_activity"] = {"before": act}
    meta_out["swap_used"] = {"before": act["swap"]}
    if not free:
        meta_out["aborted"] = "Milvus 사용 중: " + " / ".join(act["busy_reasons"])
        dump_meta()
        print("[abort]", meta_out["aborted"])
        return 3
    print(f"  preflight ok: server loaded={act['loaded_collection_num']}, data req=0, mine={act['my_load_state']}")

    texts_store: dict = {}
    results: list[QueryResult] = []
    rc = 0
    try:
        for c in targets:
            guard.load(c)
        meta_out["load_state"]["after_load"] = {c: guard.load_state(c) for c in targets}
        print(f"  loaded: {guard.log}")
        results = asyncio.run(execute(cfg, run, settings, meta, ql, keys, adapters, texts_store, meta_out))
        try:
            act2 = server_activity(run["metrics_url"], 5.0,
                                   exclude={tuple(c.split("/")) for c in targets})
            meta_out["other_user_activity"]["after"] = act2
            meta_out["swap_used"]["after"] = act2["swap"]
            meta_out["load_state"]["server_loaded_after_run"] = act2["loaded_collection_num"]
        except Exception as e:  # noqa: BLE001
            meta_out.setdefault("warnings", []).append(f"사후 활동 조회 실패: {e}")
    except BaseException as e:  # KeyboardInterrupt/SIGTERM 포함
        meta_out["exception"] = f"{type(e).__name__}: {e}"
        rc = 1
        print("[error]", meta_out["exception"], file=sys.stderr)
    finally:
        rel_errs = guard.release_all()
        meta_out["milvus_ops"] = guard.log
        try:
            meta_out["load_state"]["after_release"] = {c: guard.load_state(c) for c in targets}
        except Exception as e:  # noqa: BLE001
            meta_out["load_state"]["after_release"] = f"조회 실패: {e}"
        if rel_errs:
            meta_out["release_errors"] = rel_errs
            rc = rc or 4
        meta_out["measured_at_kst"]["end"] = now_kst()
        dump_meta()
        print(f"  released: {[l for l in guard.log if l['op'] != 'load']}")

    # 5. 결과 저장 (본문 없음)
    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False, default=_jsonable) + "\n")

    checks = check_all(results, search_topk=settings.search_topk, context_topk=settings.context_topk,
                       policy=settings.neighbor_policy)
    summary = summarize(results, settings, run, meta_out.get("wall_s"))
    summary["checks"] = {k: {"n_violation": len(v), "examples": v[:5]} for k, v in checks.items()}

    # 평가: 정답 문자열 포함률 + 원문 대조(샘플). 본문은 여기서만 쓰고 버린다.
    if run["evaluate"] and texts_store:
        per_q = []
        for key, texts in texts_store.items():
            h = answer_hit(texts.keys(), texts, meta.doc_idx, qd.answers.get(key[1]))
            per_q.append({"query_set": key[0], "q_idx": key[1], "hit": h, "n_chunks": len(texts)})
        hr = hit_rate(q["hit"] for q in per_q)
        import random
        all_ids = sorted({(k, c) for k, t in texts_store.items() for c in t})
        rnd = random.Random(cfg["experiment"]["seed"])
        sample = rnd.sample(all_ids, min(int(run["text_check_n"]), len(all_ids)))
        orig = read_chunks_jsonl(cfg["dataset"]["files"]["chunks"]["path"], {c for _, c in sample})
        mism = [c for k, c in sample if texts_store[k][c] != orig[c]]
        text_check = {"n_checked": len(sample), "n_mismatch": len(mism), "mismatch_examples": mism[:5],
                      "source": cfg["dataset"]["files"]["chunks"]["path"]}
        (out / "eval.json").write_text(json.dumps(
            {"definition": "정답 문자열 포함률 (modules/evaluate.py, 값은 참고용)", "answer_hit": hr,
             "text_check": text_check, "per_query": per_q}, ensure_ascii=False, indent=1))
        summary["answer_hit"] = hr
        summary["text_check"] = text_check
    texts_store.clear()
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=_jsonable))

    print(json.dumps({k: summary[k] for k in ("n_results", "n_measured", "n_error", "qps", "answer_hit", "text_check")
                      if k in summary}, ensure_ascii=False, default=_jsonable))
    print("  checks:", {k: v["n_violation"] for k, v in summary["checks"].items()})
    print(f"[{now_kst()}] done → {out}")
    if any(v for k, v in checks.items()):
        rc = rc or 5
    return rc


if __name__ == "__main__":
    sys.exit(main())
