"""공통 실행기.

    /workspace/jaeeun/venv/bin/python main.py --run runs/b0_smoke.yaml [--set c2.impl=bc1_kim ...]

순서:
 1. config.yaml(계약) + 실행 overlay + --set 합치기, 어댑터 이름(c1/c2/c3.impl) 확인
 2. ChunkMeta 로드(길이 1,000,000, config에 적힌 md5 검사), 질문 로드, 고정 seed 순서 생성
 3. 예약 시간대(run.reserved_window) 확인, 내 실행 잠금(locks/milvus_run.lock), Milvus 사용 중 검사
    (서버 메트릭 + 허용 목록 컬렉션의 load 상태만. 대상 컬렉션 요청도 남의 것으로 봄) → 안 되면 멈추고 보고
 4. 원래 NotLoad였던 허용 목록 컬렉션만 load → C3.run_many(검색 → C2 → C1) → finally: release 직전 다시 확인 후
    내가 load한 것만 release (원래 Loaded였던 것은 절대 release 안 함, 확인 불가면 release 안 하고 수동 명령 보고)
 5. results.jsonl(QueryResult, 본문 없음), eval.json(정답 문자열 포함률), run_meta.json, summary.json 저장

종료 코드: 0 정상 / 1 예외 / 2 허용 목록 밖 / 3 Milvus 사용 중 / 4 release 실패 / 5 결과 검사 위반
          / 6 자동 release 보류(수동 명령 보고) / 7 예약 시간대 밖·잠금·실행 설정 거부
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
from modules.run_meta import (KST, MilvusGuard, RunLock, check_reserved_window, now_kst, scrape,  # noqa: E402
                              server_activity, summarize_metrics)
from modules.search_milvus import MilvusSearchAdapter  # noqa: E402

RUN_KEYS = ("name", "purpose", "query_set", "n_queries", "max_queries", "warmup_s", "duration_s", "evaluate",
            "text_check_n", "verify_search_doc_idx", "milvus_allowed_load", "allow_preloaded_targets",
            "reserved_window", "preflight_sample_s", "release_check_s", "metrics_url", "out_dir")
LOCK_FILE = ROOT / "locks" / "milvus_run.lock"   # 내 실행끼리만 겹치지 않게 (팀 공통 잠금 아님, 미결)
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
    done_at: dict[int, float] = {}     # id(QueryResult) → run_many 시작부터 완료까지(초). strict_qps용
    t_start = [0.0]

    async def run_one_timed(key):
        r = await pipe.run_one(key)
        done_at[id(r)] = time.perf_counter() - t_start[0]
        return r

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
        time_based = kw["max_queries"] is None
        snaps: dict = {}

        async def snap(name, delay):
            await asyncio.sleep(delay)
            try:
                snaps[name] = await asyncio.to_thread(lambda: summarize_metrics(scrape(run["metrics_url"])))
            except Exception as e:  # noqa: BLE001
                snaps[name] = {"error": str(e)}

        m0 = summarize_metrics(scrape(run["metrics_url"]))
        t0 = time.perf_counter()
        t_start[0] = t0
        # 시간 기반이면 측정 구간(warmup_s ~ warmup_s+duration_s)의 시작·끝에서 메트릭을 따로 찍는다
        tasks = ([asyncio.create_task(snap("meas_start", kw["warmup_s"])),
                  asyncio.create_task(snap("meas_end", kw["warmup_s"] + kw["duration_s"]))] if time_based else [])
        try:
            results = await c3.run_many(keys, run_one_timed, **kw)
        finally:
            wall = time.perf_counter() - t0
            if tasks:   # 측정 구간 끝 스냅샷이 run_many 종료와 거의 같은 시각이라, 취소하지 말고 끝날 때까지 잠깐 기다린다
                left = max(0.0, kw["warmup_s"] + kw["duration_s"] - wall) + 6.0
                done, pending = await asyncio.wait(tasks, timeout=left)
                for tk in pending:
                    tk.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        m1 = summarize_metrics(scrape(run["metrics_url"]))
        meta_out["wall_s"] = wall
        meta_out["_done_at"] = [done_at.get(id(r)) for r in results]   # summarize()에서 쓰고 지움
        if time_based and all(isinstance(snaps.get(k), dict) and "error" not in snaps[k] for k in ("meas_start", "meas_end")):
            a, b, basis = snaps["meas_start"], snaps["meas_end"], "측정 구간 (warmup_s ~ warmup_s+duration_s)"
        else:
            a, b = m0, m1
            basis = "run_many 전체" + ("" if not time_based else " (측정 구간 메트릭을 못 찍음)")
        meta_out["storage_read_delta"] = {
            "basis": basis,
            "cache_cell_access_miss_bytes": b["cache_miss_bytes"] - a["cache_miss_bytes"],
            "by_type_location": {k: v - a["cache_miss_by_loc"].get(k, 0.0) for k, v in b["cache_miss_by_loc"].items()
                                 if v - a["cache_miss_by_loc"].get(k, 0.0) != 0},
            "milvus_restarted": a.get("start_time") != b.get("start_time"),
        }
        my = (search.db, search.collection)
        meta_out["milvus_metrics_delta"] = {
            "note": "서버 전체 값. 질문 단위로 쓰지 않음(D11)",
            "cache_cell_access_miss_bytes": m1["cache_miss_bytes"] - m0["cache_miss_bytes"],
            "note2": "run_many 전체 구간(예열 포함). 측정 구간 값은 storage_read_delta",
            "my_search_requests": m1["req_total"].get((*my, "Search"), 0) - m0["req_total"].get((*my, "Search"), 0),
            "milvus_cpu_s": (m1["cpu_s"] or 0) - (m0["cpu_s"] or 0),
        }
        return results
    finally:
        await _aclose_all([search, c1, c2, c3])


def summarize(results: list[QueryResult], settings, run, wall_s, done_at=None) -> dict:
    """qps는 기존 C3 집계와 같다(시간 기반: 예열 뒤 끝난 결과 수 / duration_s. 측정 구간이 끝난 뒤 끝난 꼬리 결과 포함).
    strict_qps는 측정 구간 [warmup_s, warmup_s+duration_s] 안에서 끝난 결과만 센다(같은 분모). 정의: DESIGN.md 4절."""
    meas = [r for r in results if not r.is_warmup]
    ok = [r for r in meas if not r.error]
    st = {k: percentiles([r.stage_ms[k] for r in ok if k in r.stage_ms]) for k in STAGE_KEYS
          if any(k in r.stage_ms for r in ok)}
    dur = float(run["duration_s"])
    time_based = run["max_queries"] is None
    fs = [r.fetch_stats for r in ok if r.fetch_stats]
    strict = {"strict_qps": None, "n_strict": None, "n_tail_after_window": None,
              "strict_qps_basis": "시간 기반 실행에서만 (max_queries 개수 기반이면 None)"}
    if time_based and done_at is not None and len(done_at) == len(results):
        t_hi = float(run["warmup_s"]) + dur
        inside = [r for r, t in zip(results, done_at) if not r.is_warmup and t is not None and t <= t_hi]
        strict = {"strict_qps": len(inside) / dur if dur else None, "n_strict": len(inside),
                  "n_tail_after_window": len(meas) - len(inside),
                  "strict_qps_basis": "측정 구간 [warmup_s, warmup_s+duration_s] 안에서 끝난 결과 수 / duration_s"}
    return {
        "n_results": len(results), "n_warmup": len(results) - len(meas), "n_measured": len(meas),
        "n_error": sum(1 for r in meas if r.error),
        "error_rate": (sum(1 for r in meas if r.error) / len(meas)) if meas else None,
        "qps": (len(meas) / dur) if time_based else (len(meas) / wall_s if wall_s else None),
        "qps_basis": "duration_s" if time_based else "wall_s (max_queries 개수 기반 실행)",
        **strict,
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
    cache_cond = cfg["experiment"]["cache_condition"]

    ts = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    out = ROOT / run["out_dir"] / f"{run['name']}_{ts}"
    out.mkdir(parents=True, exist_ok=False)
    meta_out: dict = {"measured_at_kst": {"start": now_kst()}, "argv": sys.argv, "pid": os.getpid(),
                      "settings": asdict(settings), "impl": impls, "overlay": overlay, "config_merged": cfg,
                      "milvus_reset_log": "not collected (관리자 스위치 조회는 ssh 필요, CHECK.md 참고)"}

    def dump_meta():
        (out / "run_meta.json").write_text(json.dumps(meta_out, ensure_ascii=False, indent=1, default=_jsonable))

    def refuse(msg, rc):
        meta_out["aborted"] = msg
        meta_out["measured_at_kst"]["end"] = now_kst()
        dump_meta()
        print("[abort]", msg)
        return rc

    print(f"[{now_kst()}] out={out}")
    print(f"  impl={impls}  search={settings.search_collection}  topk={settings.search_topk}/{settings.context_topk} "
          f"ef={settings.ann_ef}  purpose={run['purpose']}  cache_condition={cache_cond}")

    # 1-1. 실행 설정 거부 조건
    if run["purpose"] not in ("measure", "correctness"):
        return refuse(f"run.purpose는 measure 또는 correctness: {run['purpose']!r}", 7)
    if cache_cond not in ("warm", "cold", "cache_constrained"):
        return refuse(f"experiment.cache_condition은 warm / cold / cache_constrained: {cache_cond!r}", 7)
    if run["allow_preloaded_targets"] and run["purpose"] != "correctness":
        return refuse("run.allow_preloaded_targets는 정확성 확인(purpose=correctness)에만 켤 수 있다", 7)
    if run["purpose"] == "correctness" and run["max_queries"] is None:
        return refuse("purpose=correctness는 개수 기반(run.max_queries 정수)으로만 실행한다", 7)
    # 1-2. 예약 시간대 (측정 실행, cold/cache_constrained는 필수)
    ok, win = check_reserved_window(run, cache_cond)
    meta_out["reserved_window"] = win
    if not ok:
        return refuse("예약 시간대 확인 실패: " + win["reason"], 7)
    # 1-3. 내 실행끼리 잠금
    lock = RunLock(LOCK_FILE)
    try:
        lock.acquire({"run": run["name"], "out": str(out)})
    except BlockingIOError as e:
        return refuse(str(e), 7)
    meta_out["run_lock"] = {"path": str(LOCK_FILE), "scope": "내 실행끼리만(팀 공통 잠금 아님, 미결)"}
    try:
        return _main_locked(cfg, run, settings, impls, cache_cond, out, meta_out, dump_meta, refuse)
    finally:
        lock.release()


def _main_locked(cfg, run, settings, impls, cache_cond, out, meta_out, dump_meta, refuse) -> int:
    # 2. ChunkMeta, 질문. md5는 일반 실행에서 크기·수정 시각만(cache/md5_cache.json), 바뀐 파일만 다시 계산.
    #    전체 md5는 별도 명령(verify_md5.py, 하루 한 번). cold/cache_constrained에서는 다시 계산도 하지 않고 중단.
    t = time.perf_counter()
    loader = FileChunkMetaLoader(cfg, md5_mode="quick", cache_condition=cache_cond)
    # 검색 어댑터(modules/search_milvus.py)는 항상 chunk_of_pk["seq"]로 검색 결과 pk를 chunk_id로 바꾸고,
    # C1은 c1.layout(seq/doc/rand)의 매핑을 쓴다 → 둘 다 로드한다(순서 유지, 중복 제거).
    # PR #3(지현) 지적: 예전에는 (settings.layout,)만 로드해서 c1.layout=doc/rand면 검색 단계에서 KeyError('seq').
    meta = loader.load(layouts=tuple(dict.fromkeys(("seq", settings.layout))))
    meta_out["md5_check"] = loader.report
    meta_out["chunk_meta_load_s"] = round(time.perf_counter() - t, 1)
    ql = QueryLoader(cfg)
    qd = ql.load(run["query_set"])
    keys = ql.ordered_keys(run["query_set"], cfg["experiment"]["seed"], run["n_queries"])
    meta_out["queries"] = {"query_set": run["query_set"], "seed": cfg["experiment"]["seed"],
                           "n_keys": len(keys), "keys_head": keys[:20], "info": qd.info}
    print(f"  ChunkMeta ok ({meta_out['chunk_meta_load_s']}s, md5 {loader.report.get('mode')}), "
          f"keys={len(keys)} head={keys[:3]}")

    # 어댑터 생성 (모든 어댑터 생성자 규약: (meta, cfg)). 설정 누락(예: c1.file_ref_threads)이면 실행 거부
    try:
        adapters = {k: registry.build(k, impls[k], meta, cfg) for k in ("c1", "c2", "c3")}
    except (ValueError, KeyError, NotImplementedError) as e:
        return refuse(f"어댑터 생성 거부: {type(e).__name__}: {e}", 7)
    meta_out["adapters"] = {k: (a.describe() if hasattr(a, "describe") else type(a).__name__)
                            for k, a in adapters.items()}
    meta_out["c1_threads"] = getattr(adapters["c1"], "threads", "n/a (스레드 풀을 쓰지 않는 C1)")
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

    # 3. Milvus 사용 중 검사 → load (원래 NotLoad인 것만)
    targets = [settings.search_collection]
    for a in adapters.values():
        if hasattr(a, "milvus_collections"):
            targets += [c for c in a.milvus_collections() if c not in targets]
    guard = MilvusGuard(cfg["milvus"]["uri"], run["metrics_url"], run["milvus_allowed_load"],
                        allow_preloaded=bool(run["allow_preloaded_targets"]))
    meta_out["milvus_targets"] = targets
    not_allowed = [c for c in targets if c not in guard.allowed]
    if not_allowed:
        return refuse(f"load 허용 목록에 없는 컬렉션: {not_allowed} (run.milvus_allowed_load)", 2)
    free, act = guard.preflight(targets, sample_s=float(run["preflight_sample_s"]))
    meta_out["load_state"] = {"before": act["target_load_state"],
                              "server_loaded_before": act.get("loaded_collection_num"),
                              "allow_preloaded_targets": guard.allow_preloaded}
    meta_out["other_user_activity"] = {"before": act}
    meta_out["swap_used"] = {"before": act.get("swap")}
    if not free:
        meta_out["milvus_ops"] = guard.log
        return refuse("Milvus 사용 중 또는 확인 불가: " + " / ".join(act["busy_reasons"]), 3)
    print(f"  preflight ok: server loaded={act.get('loaded_collection_num')}, data req=0, "
          f"before={act['target_load_state']}")

    texts_store: dict = {}
    results: list[QueryResult] = []
    rc = 0
    try:
        guard.load_needed()
        meta_out["load_state"]["after_load"] = {c: guard.load_state(c) for c in targets}
        print(f"  load: {[l for l in guard.log if l['op'] in ('load', 'skip_load')]}")
        results = asyncio.run(execute(cfg, run, settings, meta, ql, keys, adapters, texts_store, meta_out))
    except BaseException as e:  # KeyboardInterrupt/SIGTERM 포함
        meta_out["exception"] = f"{type(e).__name__}: {e}"
        rc = 1
        print("[error]", meta_out["exception"], file=sys.stderr)
    finally:
        # release 직전 확인 표본(run.release_check_s) 동안 대상 컬렉션 요청이 있으면 release하지 않는다.
        # 이 표본은 내 클라이언트가 모두 닫힌 뒤(execute의 finally)라 내 요청은 섞이지 않는다.
        rel = guard.release_safe(sample_s=float(run["release_check_s"]))
        meta_out["release_report"] = rel
        meta_out["milvus_ops"] = guard.log
        try:
            meta_out["load_state"]["after_release"] = {c: guard.load_state(c) for c in targets}
        except Exception as e:  # noqa: BLE001
            meta_out["load_state"]["after_release"] = f"조회 실패: {e}"
        try:
            act2 = server_activity(run["metrics_url"], 1.0)
            meta_out["other_user_activity"]["after"] = act2
            meta_out["swap_used"]["after"] = act2["swap"]
            meta_out["load_state"]["server_loaded_after_run"] = act2["loaded_collection_num"]
        except Exception as e:  # noqa: BLE001
            meta_out.setdefault("warnings", []).append(f"사후 활동 조회 실패: {e}")
        failed = [r for r in rel if r["action"] == "release_failed"]
        manual = [r for r in rel if r["action"] == "not_released" and r.get("manual_cmd")]
        if failed:
            meta_out["release_errors"] = failed
            rc = rc or 4
        if manual:
            meta_out["release_skipped"] = manual
            rc = rc or 6
        meta_out["measured_at_kst"]["end"] = now_kst()
        dump_meta()
        for r in rel:
            print(f"  release: {r['collection']} → {r['action']} ({r['reason']})")
            if r.get("manual_cmd"):
                print(f"    수동 release 명령: {r['manual_cmd']}")

    # 캐시 조건 근거 (실제 캐시 제어는 관리자 설정·실행 스크립트 담당. 여기서는 기록과 유효성 판정만)
    srd = meta_out.get("storage_read_delta") or {}
    io_bytes = srd.get("cache_cell_access_miss_bytes")
    min_b = float(cfg["experiment"]["io_valid_min_bytes"])
    ev = {"applied": cache_cond, "admin_setting": cfg["experiment"].get("admin_cache_setting"),
          "storage_read_delta_bytes": io_bytes, "storage_read_basis": srd.get("basis"),
          "storage_read_by_type_location": srd.get("by_type_location"), "io_valid_min_bytes": min_b}
    if cache_cond in ("cold", "cache_constrained"):
        ev["valid"] = io_bytes is not None and io_bytes > min_b and not srd.get("milvus_restarted")
        ev["verdict"] = ("유효 (측정 구간 저장소 읽기 > 기준)" if ev["valid"]
                         else "실제 I/O 없음 → C1 효과 근거로 쓰지 않음")
    else:
        ev["valid"] = None
        ev["verdict"] = "warm: 저장소 읽기 유효성 판정 대상 아님(기록만)"
    meta_out["cache_condition_evidence"] = ev
    done_at = meta_out.pop("_done_at", None)
    dump_meta()

    # 5. 결과 저장 (본문 없음)
    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False, default=_jsonable) + "\n")

    checks = check_all(results, search_topk=settings.search_topk, context_topk=settings.context_topk,
                       policy=settings.neighbor_policy)
    summary = summarize(results, settings, run, meta_out.get("wall_s"), done_at)
    summary["cache_condition"] = {k: ev[k] for k in ("applied", "valid", "verdict", "storage_read_delta_bytes")}
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

    print(json.dumps({k: summary[k] for k in ("n_results", "n_measured", "n_error", "qps", "strict_qps", "answer_hit",
                                              "text_check")
                      if k in summary}, ensure_ascii=False, default=_jsonable))
    print("  checks:", {k: v["n_violation"] for k, v in summary["checks"].items()})
    print(f"[{now_kst()}] done → {out}")
    if any(v for k, v in checks.items()):
        rc = rc or 5
    return rc


if __name__ == "__main__":
    sys.exit(main())
