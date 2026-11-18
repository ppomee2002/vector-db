#!/usr/bin/env python3
"""C2 v3: 실제 질문으로 이웃 선택 규칙을 검증한다. bench_v2.py의 클래스와 pf3_rules.py의 규칙을 쓴다.

요청 1건 = 검색 top-10 -> (정책에 따라) 결과와 고른 이웃 읽기. 이웃 = 같은 문서의 앞뒤 한 조각.
정책 (모든 규칙은 정답을 보지 않는다. 검색 결과가 확정된 뒤 고른다)
  none       결과만 읽기                               : 검색(ID) -> 결과 읽기
  all        이웃 전부 같이 읽기 (기존 방식)          : 검색(ID) -> 결과 + 이웃 전부 한 번에
  rA rB rC1 rC3  규칙이 고른 이웃만 같이 읽기           : 검색(ID) -> 결과 + 규칙이 고른 이웃 한 번에
  lazy_need  [참고: 정답을 아는 상한] 필요한 질문만 나중에 : 검색(ID) -> 결과 읽기 -> 필요하면 이웃 전부 한 번 더
  text_need  [참고: 정답을 아는 상한] 본문 반환 + 나중에   : 검색이 본문까지 반환 -> 필요하면 이웃 전부 한 번 더
  "필요" = analyze_pf3.py가 만든 need_<ds>.json (정답 문자열이 이웃까지 읽어야 들어오는 질문). 실제 시스템은 이걸 미리 알 수 없다.

모드
  --mode hits  실제 질문을 전부 검색해 top-10 ID와 점수를 저장 (품질 분석용, 시간 측정 아님).
               이때 결과와 이웃 전부를 한 번 읽어 둬서, 뒤의 시간 측정에서 모든 정책이 같은 캐시 상태로 시작하게 한다
  --mode seq   요청을 하나씩. 같은 질문으로 정책들을 번갈아(순서 회전) 잰다
  --mode load  동시 요청 n(64), 요청 전체의 동시 수 상한 cap(8, bench_v2의 m7과 같은 방식). 정책 하나 구간 잼

usage:
  BENCH_DATA=$ROOT/data_nq BENCH_COLL=nq python bench_pf3.py --mode hits --ds nq --gold $ROOT/pf3_out
  BENCH_DATA=$ROOT/data_nq BENCH_COLL=nq python bench_pf3.py --mode seq --rep 1 --gold $ROOT/pf3_out
  BENCH_DATA=$ROOT/data_nq BENCH_COLL=nq python bench_pf3.py --mode load --policy rB --rep 1 --gold $ROOT/pf3_out
  python bench_pf3.py --backend mock --selftest --gold <폴더>
"""
import argparse
import asyncio
import json
import os
import random
import sys
import threading
import time

import numpy as np

import bench_v2 as bv
from pf3_rules import select

POLICIES = ("none", "all", "rA", "rB", "rC1", "rC3", "lazy_need", "text_need")
RULE_OF = {"none": "none", "all": "all", "rA": "A", "rB": "B", "rC1": "C1", "rC3": "C3"}
SEQ_COLS = ["ts_utc", "ts_kst", "version", "layout", "ds", "cond", "rep", "mode", "policy", "need_src", "reqs",
            "need_share", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "calls_per_req", "fetched_pks_per_req"]
RAW_COLS = ["rep", "i", "qi", "need", "policy", "ms", "calls", "pks"]
LOAD_COLS = ["ts_utc", "ts_kst", "version", "layout", "ds", "cond", "rep", "mode", "policy", "need_src", "n", "cap",
             "topk", "dur_s", "completed", "errors", "need_share", "qps", "mean_ms", "p50_ms", "p95_ms", "p99_ms",
             "search_calls_per_req", "fetch_calls_per_req", "fetched_pks_per_req", "r_iops", "client_cores"]


def check(got, want, what):
    if len(got) != len(want):
        raise RuntimeError(f"bad fetch ({what}): got {len(got)} rows for {len(want)} pks")


def hit_dist(h):
    for get in (lambda: h["distance"], lambda: h.distance, lambda: h["score"]):
        try:
            v = get()
            if v is not None:
                return float(v)
        except Exception:
            pass
    return 0.0


class Ctx:
    def __init__(self, a, M):
        self.M, self.k = M, a.topk
        self.flags = np.load(f"{a.gold}/{a.ds}_flags.npy")
        if len(self.flags) != M.n:
            raise SystemExit(f"flags 길이 {len(self.flags)} != 조각 수 {M.n}")
        self.need, self.src = {}, "none"
        p = f"{a.gold}/need_{a.ds}.json"
        if os.path.exists(p):
            d = json.load(open(p))
            if d.get("source") not in (None, "none") and d.get("qi"):
                self.need = dict(zip(d["qi"], d["need"]))
                self.src = d["source"]
        if self.need:
            self.qset = sorted(self.need)
        else:
            self.qset = list(range(real_queries(a)))
        self.policies = POLICIES if self.need else POLICIES[:6]

    def plan(self, ids, rule):
        M = self.M
        R = [int(M.chunk_of_pk[p]) for p in ids]
        sel = select(rule, R, M.doc, self.flags)
        return sorted(set(int(p) for p in ids) | {int(M.pk_of_chunk[c]) for c in sel})

    def nbs(self, ids):
        M = self.M
        R = [int(M.chunk_of_pk[p]) for p in ids]
        return [int(M.pk_of_chunk[c]) for c in select("all", R, M.doc)]


def real_queries(a):
    try:
        return int(json.load(open(os.path.join(bv.DATA, "meta.json")))["n_real_queries"])
    except Exception:
        return 1000


def text_ids(B, vec, k):
    if isinstance(B, bv.MockBackend):
        time.sleep(0.0002)
        return B.search_ids(vec, k)
    return [bv.hit_pk(h) for h in B.search_text(vec, k)[0]]


def one_sync(B, C, pol, qi):
    vec = C.M.qv[qi]
    c0 = B.snap()
    t0 = time.perf_counter()
    ids = text_ids(B, vec, C.k) if pol == "text_need" else B.search_ids(vec, C.k)
    if pol in RULE_OF:
        pks = C.plan(ids, RULE_OF[pol])
        check(B.fetch(pks), pks, pol)
    else:
        if pol == "lazy_need":
            res = sorted(set(ids))
            check(B.fetch(res), res, "lazy_need results")
        if C.need.get(qi, 0):
            nb = C.nbs(ids)
            if nb:
                check(B.fetch(nb), nb, f"{pol} neighbours")
    ms = (time.perf_counter() - t0) * 1e3
    c1 = B.snap()
    return ms, (c1["search"] - c0["search"]) + (c1["fetch"] - c0["fetch"]), c1["pks"] - c0["pks"]


def run_hits(B, M, a):
    gpath = f"{a.gold}/{a.ds}_gold.jsonl"
    if os.path.exists(gpath):
        qis = [json.loads(line)["qi"] for line in open(gpath, encoding="utf-8")]
    else:
        qis = list(range(real_queries(a)))
    out = a.out or f"{a.gold}/hits_{a.ds}.jsonl"
    t0 = time.time()
    pkcheck = {}
    with open(out, "w") as f:
        for j, qi in enumerate(qis):
            if isinstance(B, bv.MockBackend):
                ids, dist = B.search_ids(M.qv[qi], a.topk), [0.0] * a.topk
            else:
                r = B.c0.search(B.name, [M.qv[qi]], limit=a.topk, search_params=B.sp, output_fields=["pk"])
                ids, dist = [bv.hit_pk(h) for h in r[0]], [round(hit_dist(h), 5) for h in r[0]]
            if a.warm_ctx:  # 결과와 이웃 전부를 한 번 읽어 둔다: 뒤의 시간 측정에서 모든 정책이 같은 (따뜻한) 캐시 상태에서 시작
                got = B.fetch(M.context(ids, 1))
                if j < 20 and not isinstance(B, bv.MockBackend):
                    pkcheck.update({str(p): t for p, t in got.items()})
            f.write(json.dumps({"qi": int(qi), "ids": [int(x) for x in ids], "dist": dist}) + "\n")
    if pkcheck:
        json.dump(pkcheck, open(f"{a.gold}/hits_{a.ds}_pkcheck.json", "w"), ensure_ascii=False)
    print(f"hits {a.ds}: {len(qis)} queries in {time.time() - t0:.1f}s -> {out}", flush=True)


def run_seq(B, C, a):
    rng = np.random.default_rng(a.seed)
    M = C.M
    for _ in range(a.warm_q):
        qi = int(rng.choice(C.qset))
        ids = B.search_ids(M.qv[qi], a.topk)
        B.fetch(C.plan(ids, "all"))
    qs = rng.choice(C.qset, size=min(a.reqs, len(C.qset)), replace=len(C.qset) < a.reqs) if a.reqs > 0 else []
    pols = C.policies
    lat = {x: [] for x in pols}; calls = {x: [] for x in pols}; pks = {x: [] for x in pols}
    raw = []
    for i, qi in enumerate(qs):
        qi = int(qi)
        k = i % len(pols)
        for pol in pols[k:] + pols[:k]:
            ms, nc, npk = one_sync(B, C, pol, qi)
            lat[pol].append(ms); calls[pol].append(nc); pks[pol].append(npk)
            raw.append(dict(rep=a.rep, i=i, qi=qi, need=C.need.get(qi, ""), policy=pol, ms=round(ms, 3), calls=nc, pks=npk))
    ns = round(float(np.mean([C.need.get(int(q), 0) for q in qs])), 4) if len(qs) else 0
    rows = []
    for pol in pols:
        rows.append(dict(version=B.version, layout=a.layout, ds=a.ds, cond=a.cond, rep=a.rep, mode="seq", policy=pol,
                         need_src=C.src, reqs=len(lat[pol]), need_share=ns, mean_ms=round(float(np.mean(lat[pol])), 3),
                         p50_ms=bv.pctl(lat[pol], 50), p95_ms=bv.pctl(lat[pol], 95), p99_ms=bv.pctl(lat[pol], 99),
                         calls_per_req=round(float(np.mean(calls[pol])), 3),
                         fetched_pks_per_req=round(float(np.mean(pks[pol])), 2)))
    print("seq", {r["policy"]: r["mean_ms"] for r in rows}, flush=True)
    return rows, raw


def run_load(B, C, a):
    pol = a.policy
    M = C.M
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
            ids = await (atext_ids(vec, a.topk) if pol == "text_need" else B.asearch_ids(vec, a.topk))
            if pol in RULE_OF:
                pks = C.plan(ids, RULE_OF[pol]); check(await B.afetch(pks), pks, pol)
                return
            if pol == "lazy_need":
                res = sorted(set(ids)); check(await B.afetch(res), res, "lazy_need results")
            if need:
                nb = C.nbs(ids)
                if nb:
                    check(await B.afetch(nb), nb, f"{pol} neighbours")

        async def client(cid):
            rng, out = random.Random(a.seed * 100000 + a.rep * 1000 + cid), []
            while time.perf_counter() < t_stop and errs[0] <= max_err:
                qi = C.qset[rng.randrange(len(C.qset))]; need = C.need.get(qi, 0)
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
    row = dict(version=B.version, layout=a.layout, ds=a.ds, cond=a.cond, rep=a.rep, mode="load", policy=pol,
               need_src=C.src, n=a.n, cap=a.cap, topk=a.topk, completed=len(done), errors=errors)
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
    ap.add_argument("--mode", choices=["hits", "seq", "load"], default="seq")
    ap.add_argument("--ds", default="nq")
    ap.add_argument("--gold", default="/workspace/junlim/pf3_out", help="gold_map.py, analyze_pf3.py 출력 폴더")
    ap.add_argument("--layout", default="seq"); ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--reqs", type=int, default=400, help="seq: 회차마다 질문 수 (질문마다 정책 전부)")
    ap.add_argument("--warm-q", dest="warm_q", type=int, default=50)
    ap.add_argument("--policy", choices=POLICIES, default="all", help="load: 정책")
    ap.add_argument("--n", type=int, default=64); ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--dur", type=float, default=20); ap.add_argument("--warm", type=float, default=5)
    ap.add_argument("--dev", default="loop1")
    ap.add_argument("--cond", default="base"); ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--backend", default="milvus", choices=["milvus", "mock"])
    ap.add_argument("--warm-ctx", dest="warm_ctx", type=int, default=1, help="hits: 결과+이웃을 한 번 읽어 캐시를 데워 둠")
    ap.add_argument("--out", default=None); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    M = bv.Mapper(a.layout)
    B = bv.MockBackend(M.n) if a.backend == "mock" else bv.MilvusBackend(a.layout)
    if a.selftest:
        C = Ctx(a, M)
        if not C.need:  # 점검용 가짜 필요 여부
            C.qset = list(range(min(200, len(M.qv))))
            C.need = {q: int(q % 3 == 0) for q in C.qset}
            C.policies, C.src = POLICIES, "fake"
        a.reqs, a.warm_q = 16, 3
        rows, raw = run_seq(B, C, a)
        r = {x["policy"]: x for x in rows}
        ok = all(r[p]["calls_per_req"] == 2 for p in ("none", "all", "rA", "rB", "rC1", "rC3"))
        ok = ok and abs(r["lazy_need"]["calls_per_req"] - r["text_need"]["calls_per_req"] - 1) < 0.15
        ok = ok and r["none"]["fetched_pks_per_req"] <= r["rC1"]["fetched_pks_per_req"] <= r["all"]["fetched_pks_per_req"]
        a.dur, a.warm, a.n, a.policy = 1.5, 0.3, 8, "lazy_need"
        lr = run_load(B, C, a)
        ok = ok and lr.get("qps", 0) > 0 and lr["errors"] == 0
        print("seq calls:", {k: v["calls_per_req"] for k, v in r.items()}, "load:", {k: lr.get(k) for k in ("qps", "fetch_calls_per_req", "errors")})
        print("SELFTEST", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    if a.mode == "hits":
        run_hits(B, M, a)
        return
    C = Ctx(a, M)
    a.seed = a.seed * 1000 + a.rep
    if a.mode == "seq":
        rows, raw = run_seq(B, C, a)
        for r in rows:
            bv.write_row(a.out or f"{a.gold}/results_pf3_seq_{a.ds}.csv", SEQ_COLS, r)
        rp = f"{a.gold}/results_pf3_seq_raw_{a.ds}.csv"
        new = not os.path.exists(rp)
        with open(rp, "a", encoding="utf-8") as f:
            if new:
                f.write(",".join(RAW_COLS) + "\n")
            for x in raw:
                f.write(",".join(str(x[c]) for c in RAW_COLS) + "\n")
    else:
        if a.policy not in C.policies:
            print(f"skip {a.policy}: 필요 여부 파일이 없어 이 정책은 잴 수 없음", flush=True)
            return
        r = run_load(B, C, a)
        bv.write_row(a.out or f"{a.gold}/results_pf3_load_{a.ds}.csv", LOAD_COLS, r)
        print({k: r.get(k) for k in ("policy", "qps", "mean_ms", "p99_ms", "fetch_calls_per_req", "fetched_pks_per_req", "need_share", "errors")}, flush=True)


if __name__ == "__main__":
    main()
