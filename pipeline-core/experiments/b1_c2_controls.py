"""B1 대조군과 비교 분석 (실험 계층, D8). PLAN_integ.md B1.

    /workspace/jaeeun/venv/bin/python experiments/b1_c2_controls.py --run runs/b1_c2_compare.yaml \
        --none results/b1_none_<시각> --bc1 results/b1_BC1_<시각> --all results/b1_all_<시각>

하는 일
1. main.py로 돌린 none/BC1/all 세 결과(results.jsonl)를 읽고, 세 실행의 hits(search_topk 40)가 같은지 확인한다.
2. 대조군을 만든다(C2 모듈이 아니라 여기서).
   - ANN Top-(10+k): 질문마다 k = BC1이 추가한 이웃 수, ANN 1~(10+k)위(hits 순위).
   - R3: Top-10 ∪ (그 질문의 ±1 이웃 후보 = all 실행의 neighbor_ids에서 무작위 k개), seed 0~19.
3. MilvusGuard 규칙(사용 중 검사 → 원래 NotLoad인 것만 load → release 직전 확인 후 내가 load한 것만 release)으로
   - search_topk 10으로 전 질문을 다시 검색해 search_topk 40 상위 10개와 같은지 확인(D9)
   - C1 milvus_query_ref로 질문마다 Top-10 ∪ all 이웃 ∪ ANN 11~(10+k)위 본문을 한 번에 읽는다(serial).
4. 다섯 조건의 정답 문자열 포함률(R3는 seed 20개 평균), 추가 chunk 수, 토큰 수(GPT-2), 바이트, paired bootstrap 95% CI.
   none/BC1/all은 main.py eval.json의 질문별 판정과 일치하는지 확인한다.
결과: results/b1_controls_<시각>/ (summary.json, per_query.csv, run_meta.json). 본문은 저장하지 않는다.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules import registry  # noqa: E402
from modules.chunk_meta import FileChunkMetaLoader  # noqa: E402
from modules.config_loader import load_config  # noqa: E402
from modules.evaluate import answer_hit, norm_answers  # noqa: E402
from modules.queries import QueryLoader  # noqa: E402
from modules.run_meta import KST, MilvusGuard, RunLock, check_reserved_window, now_kst  # noqa: E402
from modules.search_milvus import MilvusSearchAdapter  # noqa: E402

TOKENS = "/workspace/jaeeun/sel_fetch_c2x/results/chunk_tokens_gpt2.npy"   # c2x와 같은 GPT-2 BPE 토큰 수
N_SEEDS = 20
N_BOOT = 10_000
BOOT_SEED = 12345
LOCK_FILE = ROOT / "locks" / "milvus_run.lock"


def read_run(d: Path):
    rows = [json.loads(l) for l in open(d / "results.jsonl", encoding="utf-8")]
    ev = json.load(open(d / "eval.json", encoding="utf-8"))
    hit = {e["q_idx"]: e["hit"] for e in ev["per_query"]}
    return {r["q_idx"]: r for r in rows}, hit, json.load(open(d / "run_meta.json", encoding="utf-8"))


def is_short(A):   # c2x c2x_exp12.is_short와 같음 (A = 정규화·공백 패딩된 정답)
    return any((len(a.split()) <= 2) or a.strip().replace(" ", "").isdigit() for a in A)


def boot_ci(diff: np.ndarray, rng: np.random.Generator):
    n = len(diff)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    m = diff[idx].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return {"mean_pp": float(diff.mean() * 100), "ci95_pp": [float(lo * 100), float(hi * 100)], "n": int(n)}


async def fetch_and_search(cfg, meta, ql, keys, want, check_top10):
    c1 = registry.build("c1", "milvus_query_ref", meta, cfg)
    search = MilvusSearchAdapter(meta, cfg, queries=ql)
    texts, top10, stats = {}, {}, {"n_requests": 0, "bytes": 0, "fetch_s": 0.0}
    try:
        for i, k in enumerate(keys):
            q = k[1]
            t = time.perf_counter()
            tx, st = await c1.fetch_text(sorted(want[q]), mode="batch", batch_impl="query_pk_in")
            stats["fetch_s"] += time.perf_counter() - t
            stats["n_requests"] += st.n_requests
            stats["bytes"] += st.bytes_returned
            missing = set(want[q]) - set(tx)
            if missing:
                raise RuntimeError(f"q {q}: 본문 {len(missing)}개 없음 {sorted(missing)[:5]}")
            texts[q] = dict(tx)
            if check_top10:
                hs = await search.search(k, topk=10, ef=int(cfg["ann"]["ef"]))
                top10[q] = [h.chunk_id for h in hs]
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(keys)}", flush=True)
    finally:
        for o in (search, c1):
            f = getattr(o, "aclose", None)
            if f:
                await f()
    return texts, top10, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--none", required=True)
    ap.add_argument("--bc1", required=True)
    ap.add_argument("--all", required=True)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)

    cfg, overlay = load_config(a.run, a.set)
    run = cfg["run"]
    ts = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    out = ROOT / run["out_dir"] / f"b1_controls_{ts}"
    out.mkdir(parents=True)
    rm = {"measured_at_kst": {"start": now_kst()}, "argv": sys.argv, "inputs": {"none": a.none, "BC1": a.bc1, "all": a.all},
          "overlay": overlay}

    def dump():
        (out / "run_meta.json").write_text(json.dumps(rm, ensure_ascii=False, indent=1, default=str))

    ok, win = check_reserved_window(run, cfg["experiment"]["cache_condition"])
    rm["reserved_window"] = win
    if not ok:
        rm["aborted"] = win.get("reason"); dump(); print("[abort]", win.get("reason")); return 7

    R = {}
    for name, d in (("none", a.none), ("BC1", a.bc1), ("all", a.all)):
        R[name] = read_run(ROOT / d if not Path(d).is_absolute() else Path(d))
    qs = list(R["BC1"][0].keys())
    # 1. 같은 ANN 결과인지
    same_hits = sum(1 for q in qs if [h["chunk_id"] for h in R["none"][0][q]["hits"]]
                    == [h["chunk_id"] for h in R["BC1"][0][q]["hits"]] == [h["chunk_id"] for h in R["all"][0][q]["hits"]])
    rm["same_hits_across_runs"] = {"n": len(qs), "n_same": same_hits}
    errs = sum(1 for n in R for r in R[n][0].values() if r["error"])
    rm["pipeline_errors"] = errs
    print(f"queries {len(qs)}, hits same {same_hits}, pipeline errors {errs}")

    meta = FileChunkMetaLoader(cfg, md5_mode="quick", cache_condition="warm").load(layouts=("seq",))
    ql = QueryLoader(cfg)
    qd = ql.load(run["query_set"])
    ntok = np.load(TOKENS)
    if len(ntok) != meta.n_chunks:
        raise SystemExit("토큰 배열 길이 불일치")

    # 2. 대조군 구성
    sets = {}       # q → {cond: set(chunk)} (R3는 seed별 list)
    want = {}
    for q in qs:
        b = R["BC1"][0][q]
        top = [h["chunk_id"] for h in b["hits"]]
        ctx = b["context_ids"]
        k = len(b["neighbor_ids"])
        cand = sorted(R["all"][0][q]["neighbor_ids"])
        if not set(b["neighbor_ids"]) <= set(cand):
            raise SystemExit(f"q {q}: BC1 이웃이 all 이웃의 부분집합이 아님")
        if 10 + k > len(top):
            raise SystemExit(f"q {q}: 10+k={10 + k} > search_topk {len(top)}")
        r3 = []
        for s in range(N_SEEDS):
            rng = random.Random(s * 1_000_003 + q)
            r3.append(set(ctx) | set(rng.sample(cand, k)))
        sets[q] = {"none": set(ctx), "BC1": set(ctx) | set(b["neighbor_ids"]), "all": set(ctx) | set(cand),
                   "ANN": set(top[: 10 + k]), "R3": r3, "k": k}
        want[q] = sets[q]["all"] | sets[q]["ANN"]
    keys = [(run["query_set"], q) for q in qs]

    # 3. Milvus (Guard)
    lock = RunLock(LOCK_FILE)
    try:
        lock.acquire({"run": "b1_controls", "out": str(out)})
    except BlockingIOError as e:
        rm["aborted"] = str(e); dump(); print("[abort]", e); return 7
    try:
        targets = [f"{cfg['milvus']['search']['database']}/{cfg['milvus']['search']['collection']}"]
        guard = MilvusGuard(cfg["milvus"]["uri"], run["metrics_url"], run["milvus_allowed_load"],
                            allow_preloaded=bool(run["allow_preloaded_targets"]))
        free, act = guard.preflight(targets, sample_s=float(run["preflight_sample_s"]))
        rm["preflight"] = {"ok": free, "reasons": act["busy_reasons"], "state": act["target_load_state"],
                           "server_loaded": act.get("loaded_collection_num")}
        if not free:
            rm["milvus_ops"] = guard.log; rm["aborted"] = "Milvus 사용 중"; dump()
            print("[abort] Milvus 사용 중:", act["busy_reasons"]); return 3
        print(f"  preflight ok {act['target_load_state']}")
        rc = 0
        try:
            guard.load_needed()
            texts, top10, fst = asyncio.run(fetch_and_search(cfg, meta, ql, keys, want, True))
        except BaseException as e:  # noqa: BLE001
            rm["exception"] = f"{type(e).__name__}: {e}"; rc = 1
            print("[error]", rm["exception"])
        finally:
            rel = guard.release_safe(sample_s=float(run["release_check_s"]))
            rm["release_report"] = rel
            rm["milvus_ops"] = guard.log
            for r in rel:
                print(f"  release: {r['collection']} → {r['action']} ({r['reason']})")
                if r["action"] != "released" and r.get("manual_cmd"):
                    rc = rc or 6
        if rc:
            dump(); return rc
    finally:
        lock.release()
    rm["fetch_stats"] = fst
    rm["top10_consistency_full"] = {
        "n": len(qs),
        "n_same_order": sum(1 for q in qs if top10[q] == R["BC1"][0][q]["context_ids"]),
        "n_same_set": sum(1 for q in qs if set(top10[q]) == set(R["BC1"][0][q]["context_ids"])),
        "set_overlap_mean": float(np.mean([len(set(top10[q]) & set(R["BC1"][0][q]["context_ids"])) / 10 for q in qs])),
        "note": "search_topk 10 (ef 64) 재검색 vs search_topk 40의 상위 10개",
    }

    # 4. 평가
    doc = meta.doc_idx
    conds = ["none", "BC1", "all", "ANN", "R3"]
    rows = []
    mism = {c: 0 for c in ("none", "BC1", "all")}
    for q in qs:
        A = qd.answers.get(q)
        nA = norm_answers(A)
        if not nA:
            continue
        tx = texts[q]
        row = {"q_idx": q, "k": sets[q]["k"], "short": int(is_short(nA))}
        for c in conds:
            if c == "R3":
                hs = [answer_hit(s, tx, doc, A) for s in sets[q]["R3"]]
                row["hit_R3"] = float(np.mean(hs))
                row["extra_R3"] = sets[q]["k"]
                row["tok_R3"] = float(np.mean([int(ntok[sorted(s)].sum()) for s in sets[q]["R3"]]))
                row["bytes_R3"] = float(np.mean([sum(len(tx[c_].encode("utf-8")) for c_ in s) for s in sets[q]["R3"]]))
            else:
                s = sets[q][c]
                row[f"hit_{c}"] = float(answer_hit(s, tx, doc, A))
                row[f"extra_{c}"] = len(s) - 10
                row[f"tok_{c}"] = int(ntok[sorted(s)].sum())
                row[f"bytes_{c}"] = sum(len(tx[c_].encode("utf-8")) for c_ in s)
        for c in mism:
            if bool(row[f"hit_{c}"]) != bool(R[c][1].get(q)):
                mism[c] += 1
        rows.append(row)
    rm["eval_mismatch_vs_main"] = mism

    with open(out / "per_query.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    rng = np.random.default_rng(BOOT_SEED)
    summ = {"definition": {"hit": "정답 문자열 포함률(modules/evaluate.py). R3는 질문별 seed 20개 평균",
                           "tokens": "GPT-2 BPE (" + TOKENS + ")", "short": "정답 별칭 중 하나라도 2단어 이하 또는 숫자",
                           "bootstrap": f"질문 단위 paired, {N_BOOT}회, seed {BOOT_SEED}"},
            "top10_consistency_full": rm["top10_consistency_full"], "same_hits_across_runs": rm["same_hits_across_runs"],
            "eval_mismatch_vs_main": mism, "subsets": {}}
    pairs = [("BC1", "none"), ("BC1", "ANN"), ("BC1", "R3"), ("all", "BC1"), ("ANN", "none"), ("R3", "none"), ("all", "ANN")]
    for sub, keep in (("all", lambda r: True), ("no_short", lambda r: not r["short"])):
        rs = [r for r in rows if keep(r)]
        S = {"n": len(rs), "conditions": {}, "paired": {}}
        for c in conds:
            S["conditions"][c] = {
                "hit_rate": float(np.mean([r[f"hit_{c}"] for r in rs])),
                "extra_chunks_mean": float(np.mean([r[f"extra_{c}"] for r in rs])),
                "tokens_mean": float(np.mean([r[f"tok_{c}"] for r in rs])),
                "bytes_mean": float(np.mean([r[f"bytes_{c}"] for r in rs])),
            }
        if sub == "all":
            S["R3_seed_hit_rates"] = [float(np.mean([answer_hit(sets[r["q_idx"]]["R3"][s], texts[r["q_idx"]], doc,
                                                                qd.answers.get(r["q_idx"])) for r in rs]))
                                      for s in range(N_SEEDS)]
        for x, y in pairs:
            d = np.array([r[f"hit_{x}"] - r[f"hit_{y}"] for r in rs])
            S["paired"][f"{x}-{y}"] = boot_ci(d, rng)
        summ["subsets"][sub] = S
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    rm["measured_at_kst"]["end"] = now_kst()
    dump()
    for sub in summ["subsets"]:
        S = summ["subsets"][sub]
        print(f"[{sub}] n={S['n']}")
        for c, v in S["conditions"].items():
            print(f"  {c:5s} hit {v['hit_rate']:.4f}  extra {v['extra_chunks_mean']:.2f}  tok {v['tokens_mean']:.0f}")
        for p, v in S["paired"].items():
            print(f"  {p:9s} {v['mean_pp']:+.2f}pp [{v['ci95_pp'][0]:+.2f}, {v['ci95_pp'][1]:+.2f}]")
    print("top10 full:", rm["top10_consistency_full"], "mismatch vs main:", mism)
    print("done →", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
