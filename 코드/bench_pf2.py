#!/usr/bin/env python3
"""C2 미리 읽기 검증 v2: 이웃이 필요한 요청 비율 p를 바꿔 가며 세 가지 읽기 정책을 실측한다.
bench_v2.py의 클래스를 그대로 가져다 쓴다.

검색 결과 top-k가 확정된 뒤, 같은 문서의 앞뒤 expand개 조각을 "이웃"이라 한다.
요청마다 이웃이 필요한지를 확률 p로 정한다 (앱이 결과를 본 뒤 더 읽을지 정하는 상황을 흉내 냄).
  lazy_id    필요할 때 읽기 (ID 구조)   : 검색(ID) -> 결과 본문 읽기 -> 필요하면 이웃만 한 번 더 읽기
  prefetch   미리 읽기 (Ours)           : 검색(ID) -> 결과와 이웃을 항상 한 번에 읽기
  lazy_text  필요할 때 읽기 (본문 반환) : 검색이 결과 본문까지 반환 -> 필요하면 이웃만 한 번 더 읽기

모드
  --mode seq   요청을 하나씩 순서대로 보낸다. 같은 질의, 같은 필요 여부로 세 정책을 번갈아(순서 회전) 재서
               p마다 평균 요청 시간을 비교한다. 한 번 실행에 --ps의 모든 p를 잰다.
  --mode load  동시 요청 n개(기본 64)를 계속 걸고, 요청 전체의 동시 수 상한 cap(기본 8, bench_v2의 m7과 같은 방식)
               아래에서 정책 하나, p 하나를 한 구간(예열 warm초 + 측정 dur초) 잰다.

usage:
  BENCH_DATA=$ROOT/data_nq BENCH_COLL=nq python bench_pf2.py --mode seq --rep 1 --out results_pf2_seq_nq.csv
  BENCH_DATA=$ROOT/data_nq BENCH_COLL=nq python bench_pf2.py --mode load --policy prefetch --p 0.1 --rep 1 --out results_pf2_load_nq.csv
  python bench_pf2.py --backend mock --selftest      (서버 없이 도구 점검)
"""
import argparse
import asyncio
import os
import random
import sys
import threading
import time

import numpy as np

import bench_v2 as bv

POLICIES = ("lazy_id", "prefetch", "lazy_text")
SEQ_COLS = ["ts_utc", "ts_kst", "version", "layout", "cond", "rep", "mode", "policy", "p", "topk", "expand",
            "reqs", "need_share", "mean_ms", "p50_ms", "p95_ms", "calls_per_req", "fetched_pks_per_req"]
LOAD_COLS = ["ts_utc", "ts_kst", "version", "layout", "cond", "rep", "mode", "policy", "p", "n", "cap", "topk",
             "expand", "dur_s", "completed", "errors", "need_share", "qps", "mean_ms", "p50_ms", "p95_ms", "p99_ms",
             "search_calls_per_req", "fetch_calls_per_req", "fetched_pks_per_req", "r_iops", "client_cores"]


def neighbours(M, ids, expand):
    return sorted(set(M.context(ids, expand)) - set(ids))


def check(got, want, what):
    if len(got) != len(want):
        raise RuntimeError(f"bad fetch ({what}): got {len(got)} rows for {len(want)} pks")


# ---------------------------------------------------------------- one request, sync
def text_ids(B, vec, k):
    if isinstance(B, bv.MockBackend):  # mock: 검색 지연만 흉내
        time.sleep(0.0002)
        return B.search_ids(vec, k)
    return [bv.hit_pk(h) for h in B.search_text(vec, k)[0]]


def one_sync(B, M, a, pol, qi, need):
    vec = M.qv[qi]
    c0 = B.snap()
    t0 = time.perf_counter()
    if pol == "lazy_text":
        ids = text_ids(B, vec, a.topk)
    else:
        ids = B.search_ids(vec, a.topk)
    res = sorted(set(ids))
    if pol == "prefetch":
        ctx = M.context(ids, a.expand); check(B.fetch(ctx), ctx, "prefetch")
    else:
        if pol == "lazy_id":
            check(B.fetch(res), res, "lazy_id results")
        if need:
            nb = neighbours(M, ids, a.expand)
            if nb:
                check(B.fetch(nb), nb, f"{pol} neighbours")
    ms = (time.perf_counter() - t0) * 1e3
    c1 = B.snap()
    return ms, (c1["search"] - c0["search"]) + (c1["fetch"] - c0["fetch"]), c1["pks"] - c0["pks"]


def run_seq(B, M, a):
    rng = np.random.default_rng(a.seed)
    for _ in range(a.warm_q):  # 예열 (기록하지 않음)
        qi = int(rng.integers(0, len(M.qv)))
        ids = B.search_ids(M.qv[qi], a.topk); B.fetch(M.context(ids, a.expand))
    rows = []
    for p in a.ps:
        lat = {x: [] for x in POLICIES}; calls = {x: [] for x in POLICIES}; pks = {x: [] for x in POLICIES}
        needs = 0
        for i in range(a.reps):
            qi = int(rng.integers(0, len(M.qv))); need = bool(rng.random() < p); needs += need
            k = i % len(POLICIES)
            for pol in POLICIES[k:] + POLICIES[:k]:  # 정책 순서를 요청마다 돌린다
                ms, nc, npk = one_sync(B, M, a, pol, qi, need)
                lat[pol].append(ms); calls[pol].append(nc); pks[pol].append(npk)
        for pol in POLICIES:
            rows.append(dict(version=B.version, layout=a.layout, cond=a.cond, rep=a.rep, mode="seq", policy=pol, p=p,
                             topk=a.topk, expand=a.expand, reqs=len(lat[pol]), need_share=round(needs / a.reps, 3),
                             mean_ms=round(float(np.mean(lat[pol])), 3), p50_ms=bv.pctl(lat[pol], 50),
                             p95_ms=bv.pctl(lat[pol], 95), calls_per_req=round(float(np.mean(calls[pol])), 3),
                             fetched_pks_per_req=round(float(np.mean(pks[pol])), 2)))
        print(f"p={p}", {x["policy"]: x["mean_ms"] for x in rows[-len(POLICIES):]}, flush=True)
    return rows


# ---------------------------------------------------------------- load: n concurrent clients, request cap
def run_load(B, M, a):
    pol, p = a.policy, a.p
    t_begin = time.perf_counter(); t_warm = t_begin + a.warm; t_stop = t_warm + a.dur
    snaps, errs, first_err = {}, [0], []
    max_err = int(os.environ.get("BENCH_MAX_ERR", "20"))

    def monitor():
        time.sleep(max(0, t_warm - time.perf_counter()))
        snaps.update(t0=time.perf_counter(), cpu0=time.process_time(), d0=bv.disk_snap(a.dev), c0=B.snap())
        time.sleep(max(0, t_stop - time.perf_counter()))
        snaps.update(t1=time.perf_counter(), cpu1=time.process_time(), d1=bv.disk_snap(a.dev), c1=B.snap())

    mon = threading.Thread(target=monitor, daemon=True); mon.start()

    async def main():
        await B.aopen()
        rsem = asyncio.Semaphore(a.cap) if a.cap > 0 else None

        async def atext_ids(vec, k):
            if isinstance(B, bv.MockBackend):
                await asyncio.sleep(0.0002)
                return await B.asearch_ids(vec, k)
            r = await B.ac.search(B.name, [vec], limit=k, search_params=B.sp, output_fields=["text"])
            B.count(search=1)
            return [bv.hit_pk(h) for h in r[0]]

        async def one(qi, need):
            vec = M.qv[qi]
            ids = await (atext_ids(vec, a.topk) if pol == "lazy_text" else B.asearch_ids(vec, a.topk))
            if pol == "prefetch":
                ctx = M.context(ids, a.expand); check(await B.afetch(ctx), ctx, "prefetch")
                return
            if pol == "lazy_id":
                res = sorted(set(ids)); check(await B.afetch(res), res, "lazy_id results")
            if need:
                nb = neighbours(M, ids, a.expand)
                if nb:
                    check(await B.afetch(nb), nb, f"{pol} neighbours")

        async def client(cid):
            rng, out = random.Random(a.seed * 100000 + a.rep * 1000 + cid), []
            while time.perf_counter() < t_stop and errs[0] <= max_err:
                qi = rng.randrange(len(M.qv)); need = rng.random() < p
                t = time.perf_counter(); ok = 1
                try:
                    if rsem:
                        async with rsem:
                            await one(qi, need)
                    else:
                        await one(qi, need)
                except Exception as e:
                    ok = 0; errs[0] += 1
                    if not first_err:
                        first_err.append(repr(e)); print("첫 오류:", repr(e), file=sys.stderr)
                    await asyncio.sleep(0.2)
                d = time.perf_counter(); out.append((d, (d - t) * 1e3, ok, need))
            return out

        res = await asyncio.gather(*[client(i) for i in range(a.n)])
        await B.aclose()
        return [r for o in res for r in o]

    recs = asyncio.run(main())
    mon.join(timeout=5)
    if errs[0] > max_err:
        raise SystemExit(f"HALT: 오류 {errs[0]}회 ({first_err[0] if first_err else ''})")
    done = [r for r in recs if r[0] >= t_warm and r[2] == 1]
    errors = sum(1 for r in recs if r[0] >= t_warm and r[2] == 0)
    row = dict(version=B.version, layout=a.layout, cond=a.cond, rep=a.rep, mode="load", policy=pol, p=p, n=a.n,
               cap=a.cap, topk=a.topk, expand=a.expand, completed=len(done), errors=errors)
    if "t1" in snaps and done:
        dur = snaps["t1"] - snaps["t0"]; lat = np.array([r[1] for r in done]); nd = len(done)
        c0, c1 = snaps["c0"], snaps["c1"]; d = {k: c1[k] - c0[k] for k in c1}
        row.update(dur_s=round(dur, 2), need_share=round(sum(1 for r in done if r[3]) / nd, 3),
                   qps=round(nd / dur, 1), mean_ms=round(float(lat.mean()), 2),
                   p50_ms=round(float(np.percentile(lat, 50)), 2), p95_ms=round(float(np.percentile(lat, 95)), 2),
                   p99_ms=round(float(np.percentile(lat, 99)), 2),
                   search_calls_per_req=round(d["search"] / nd, 3), fetch_calls_per_req=round(d["fetch"] / nd, 3),
                   fetched_pks_per_req=round(d["pks"] / nd, 2),
                   client_cores=round((snaps["cpu1"] - snaps["cpu0"]) / dur, 2))
        if snaps.get("d0") and snaps.get("d1"):
            row["r_iops"] = round((snaps["d1"]["r"] - snaps["d0"]["r"]) / dur)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["seq", "load"], default="seq")
    ap.add_argument("--layout", default="seq"); ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--expand", type=int, default=1)
    ap.add_argument("--ps", default="0,0.05,0.1,0.2,0.5,1", help="seq: 잴 p 목록")
    ap.add_argument("--reps", type=int, default=300, help="seq: p마다 질의 수")
    ap.add_argument("--warm-q", dest="warm_q", type=int, default=50)
    ap.add_argument("--policy", choices=POLICIES, default="prefetch", help="load: 정책")
    ap.add_argument("--p", type=float, default=0.1, help="load: 이웃이 필요한 비율")
    ap.add_argument("--n", type=int, default=64); ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--dur", type=float, default=20); ap.add_argument("--warm", type=float, default=5)
    ap.add_argument("--dev", default="loop1")
    ap.add_argument("--cond", default="base"); ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--backend", default="milvus", choices=["milvus", "mock"])
    ap.add_argument("--out", default=None); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    a.ps = [float(x) for x in a.ps.split(",")]
    M = bv.Mapper(a.layout)
    B = bv.MockBackend(M.n) if a.backend == "mock" else bv.MilvusBackend(a.layout)
    if a.selftest:
        a.reps, a.warm_q, a.ps = 12, 3, [0.0, 1.0]
        rows = run_seq(B, M, a)
        r = {(x["policy"], x["p"]): x for x in rows}
        ok = (r[("lazy_id", 0.0)]["calls_per_req"] == 2 and r[("lazy_id", 1.0)]["calls_per_req"] == 3
              and r[("prefetch", 0.0)]["calls_per_req"] == 2 and r[("lazy_text", 0.0)]["calls_per_req"] == 1
              and r[("lazy_text", 1.0)]["calls_per_req"] == 2)
        a.dur, a.warm, a.n, a.policy, a.p = 2, 0.5, 8, "lazy_id", 1.0
        lr = run_load(B, M, a)
        ok = ok and lr.get("qps", 0) > 0 and lr["errors"] == 0 and abs(lr["fetch_calls_per_req"] - 2) < 0.05
        print("seq calls:", {k: v["calls_per_req"] for k, v in r.items()}, "load:", {k: lr.get(k) for k in ("qps", "fetch_calls_per_req", "errors")})
        print("SELFTEST", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    a.seed = a.seed * 1000 + a.rep  # 회차마다 다른 질의
    if a.mode == "seq":
        rows = run_seq(B, M, a)
        for r in rows:
            bv.write_row(a.out or "results_pf2_seq_nq.csv", SEQ_COLS, r)
    else:
        r = run_load(B, M, a)
        bv.write_row(a.out or "results_pf2_load_nq.csv", LOAD_COLS, r)
        print({k: r.get(k) for k in ("policy", "p", "qps", "mean_ms", "p99_ms", "fetch_calls_per_req", "need_share", "errors")}, flush=True)


if __name__ == "__main__":
    main()
