#!/usr/bin/env python3
"""C2 미리 읽기(prefetch) 비용과 이득을 재는 도구. bench_v2.py의 클래스를 그대로 가져다 쓴다.

검색 결과 top-k가 확정된 뒤 같은 문서의 앞뒤 조각(이웃)을
  미리 읽기: 결과와 같은 호출에서 같이 읽는다           -> 추가 비용 = B - A
  필요할 때 읽기: 나중에 이웃만 따로 한 번 더 부른다     -> 비용 = N (필요할 때만)
로 나눠 본다. 요청을 하나씩 순서대로 보내고, 측정 종류를 요청마다 A, B, N, T 순으로 돌아가며 바꾼다.
종류마다 다른 질의를 쓰므로 앞 측정이 읽어 둔 캐시가 다음 측정을 돕지 않는다.

  S: 검색(ID만 받음)                    T: 검색 + 결과 본문을 같이 받음(output_fields=text)
  A: 결과 top-k 본문만 읽기 (호출 1회)   B: 결과 + 이웃 본문을 한 번에 읽기 (호출 1회)
  N: 이웃 본문만 읽기 (호출 1회)
손익분기 p* = (B - A) / N : 이웃이 필요한 요청 비율이 p*보다 크면 미리 읽는 쪽이 평균 읽기 시간이 짧다
(ID로 검색한 뒤 본문을 따로 읽는 경로 기준).

usage: BENCH_DATA=$ROOT/data_nq BENCH_COLL=nq python bench_pf.py --cond pf_warm --rep 1 --out results_pf_nq.csv
       python bench_pf.py --backend mock --selftest      (서버 없이 도구 점검)
"""
import argparse
import sys
import time

import numpy as np

import bench_v2 as bv

PF_COLS = ["ts_utc", "ts_kst", "version", "layout", "cond", "rep", "topk", "expand", "reps_per_kind", "skipped",
           "res_pks_mean", "ctx_pks_mean", "nb_pks_mean",
           "S_p50", "T_p50", "A_p50", "B_p50", "N_p50",
           "S_p95", "T_p95", "A_p95", "B_p95", "N_p95",
           "prefetch_extra_ms", "ondemand_ms", "breakeven_p",
           "eager_ids_ms", "lazy_text_noneed_ms", "lazy_text_need_ms"]
KINDS = "ABNT"


def run_pf(B, M, a):
    rng = np.random.default_rng(a.seed)
    t = {k: [] for k in "STABN"}
    npk = {"res": [], "ctx": [], "nb": []}
    total = a.warm_q + a.reps * len(KINDS)
    done = skipped = 0
    while done < total:
        qi = int(rng.integers(0, len(M.qv)))
        kind = KINDS[done % len(KINDS)]
        keep = done >= a.warm_q
        if kind == "T":
            t0 = time.perf_counter(); B.search_text(M.qv[qi], a.topk); t1 = time.perf_counter()
            if keep:
                t["T"].append((t1 - t0) * 1e3)
            done += 1
            continue
        t0 = time.perf_counter(); ids = B.search_ids(M.qv[qi], a.topk); t1 = time.perf_counter()
        res = sorted(set(ids))
        ctx = M.context(ids, a.expand)
        nb = sorted(set(ctx) - set(res))
        if not nb:  # 결과가 모두 한 조각짜리 문서라 이웃이 없으면 이 질의는 건너뛴다
            skipped += 1
            continue
        target = {"A": res, "B": ctx, "N": nb}[kind]
        t2 = time.perf_counter(); got = B.fetch(target); t3 = time.perf_counter()
        if len(got) != len(target):
            raise SystemExit(f"bad fetch ({kind}): got {len(got)} rows for {len(target)} pks")
        if keep:
            t["S"].append((t1 - t0) * 1e3)
            t[kind].append((t3 - t2) * 1e3)
            npk["res"].append(len(res)); npk["ctx"].append(len(ctx)); npk["nb"].append(len(nb))
        done += 1

    p50 = {k: bv.pctl(v, 50) for k, v in t.items()}
    p95 = {k: bv.pctl(v, 95) for k, v in t.items()}
    extra = p50["B"] - p50["A"]
    row = dict(version=B.version, layout=a.layout, cond=a.cond, rep=a.rep, topk=a.topk, expand=a.expand,
               reps_per_kind=a.reps, skipped=skipped,
               res_pks_mean=round(float(np.mean(npk["res"])), 1), ctx_pks_mean=round(float(np.mean(npk["ctx"])), 1),
               nb_pks_mean=round(float(np.mean(npk["nb"])), 1),
               prefetch_extra_ms=round(extra, 3), ondemand_ms=p50["N"],
               breakeven_p=round(max(extra, 0.0) / p50["N"], 3),
               eager_ids_ms=round(p50["S"] + p50["B"], 3),
               lazy_text_noneed_ms=p50["T"], lazy_text_need_ms=round(p50["T"] + p50["N"], 3))
    for k in "STABN":
        row[f"{k}_p50"], row[f"{k}_p95"] = p50[k], p95[k]
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layout", default="seq"); p.add_argument("--topk", type=int, default=10)
    p.add_argument("--expand", type=int, default=1); p.add_argument("--reps", type=int, default=300)
    p.add_argument("--warm-q", dest="warm_q", type=int, default=50)
    p.add_argument("--cond", default="pf_warm"); p.add_argument("--rep", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--backend", default="milvus", choices=["milvus", "mock"])
    p.add_argument("--out", default="results_pf_nq.csv"); p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    M = bv.Mapper(a.layout)
    B = bv.MockBackend(M.n) if a.backend == "mock" else bv.MilvusBackend(a.layout)
    if a.selftest:
        a.reps, a.warm_q = 20, 4
        r = run_pf(B, M, a)
        ok = all(r[f"{k}_p50"] > 0 for k in "STABN")
        print(r); print("SELFTEST", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    a.seed = a.seed * 1000 + a.rep  # 회차마다 다른 질의
    r = run_pf(B, M, a)
    bv.write_row(a.out, PF_COLS, r)
    print({k: r[k] for k in ("cond", "rep", "S_p50", "T_p50", "A_p50", "B_p50", "N_p50",
                             "prefetch_extra_ms", "breakeven_p", "skipped")})


if __name__ == "__main__":
    main()
