#!/usr/bin/env python3
"""C3 13주차: 파이프라인의 검색 설정(top-10 / top-100)이 바뀌면 m7(요청 전체 동시 수 상한)의 알맞은 상한 k가 달라지는가.
bench_v2.py(58d771b9)의 MilvusBackend, Mapper, MockBackend를 가져다 쓴다. bench_v2.py는 고치지 않는다.

검색 설정 (모든 설정에서 본문 조회 대상은 같다: 질문마다 미리 저장한 PK 목록)
  A  top-10,  ef=64   지난주 m7과 같은 검색
  B  top-10,  ef=100  A 대비 ef만 올림
  C  top-100, ef=100  B 대비 top-k만 올림 (ef를 고정한 상태에서 top-k를 바꾼 전체 영향)
  ef=100: top-100을 검색하려면 ef가 limit 이상이어야 하므로 그 최솟값.

요청 1건 (m7과 같은 흐름) = [상한 k 대기] -> 벡터 검색(설정 A/B/C) -> 미리 저장한 PK 목록 본문 조회 1회.
  상한은 asyncio.Semaphore(k)로 요청 전체(검색+조회)를 감싼다. bench_v2.py m7과 같은 방식.
  PK 목록 = 설정 A로 검색한 top-10 + 같은 문서의 앞뒤 1조각(expand=1). 지난주 m7이 읽은 것과 같은 범위.
  PK 목록을 만드는 검색(prep)은 측정 전에 따로 하고 측정 시간에 넣지 않는다.

모드
  --mode prep  질문(기본: query_vecs 전체에서 고정 시드로 10,000개)마다 설정 A로 검색해 top-10과 PK 목록을 저장한다. 같은 목록을 한 번 읽어 캐시를 데운다.
               ef=64, limit=100 검색이 거부되는지도 한 번 확인해 기록한다(ef >= limit 조건 확인용).
  --mode warmup  저장한 PK 목록을 한 번씩 읽어 캐시를 데운다 (컬렉션을 다시 올린 세션에서 측정 전에).
  --mode seq   요청을 하나씩. 같은 질문으로 설정 A/B/C를 번갈아(순서 회전) 검색 시간과 조회 시간을 잰다.
  --mode load  동시 요청 n, 상한 k, 설정 하나로 한 구간을 잰다.

usage (서버, /workspace/junlim에서. 보통은 run_c3a.sh가 부른다)
  BENCH_DATA=$ROOT/data_nq BENCH_COLL=nq python bench_c3a.py --mode prep --outdir $ROOT/c3a_out
  BENCH_DATA=$ROOT/data_nq BENCH_COLL=nq python bench_c3a.py --mode seq --rep 1 --outdir $ROOT/c3a_out
  BENCH_DATA=$ROOT/data_nq BENCH_COLL=nq python bench_c3a.py --mode load --setting C --cap 12 --phase explore --rep 1 --outdir $ROOT/c3a_out
  python bench_c3a.py --backend mock --selftest --outdir /tmp/c3a_selftest
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

SETTINGS = {"A": dict(limit=10, ef=64), "B": dict(limit=10, ef=100), "C": dict(limit=100, ef=100)}
LOAD_COLS = ["ts_utc", "ts_kst", "version", "layout", "ds", "phase", "rep", "slot", "setting", "limit", "ef", "n",
             "cap", "dur_s", "completed", "errors", "qps", "mean_ms", "p50_ms", "p95_ms", "p99_ms",
             "wait_ms_mean", "wait_ms_p50", "wait_ms_p99", "search_ms_mean", "search_ms_p50", "search_ms_p99",
             "fetch_ms_mean", "fetch_ms_p50", "fetch_ms_p99", "hits_per_req", "fetched_pks_per_req",
             "top10_same", "client_cores", "disk_dev", "r_iops", "read_mbps"]
SEQ_COLS = ["ts_utc", "ts_kst", "version", "ds", "rep", "setting", "limit", "ef", "reqs", "search_ms_mean",
            "search_ms_p50", "search_ms_p95", "fetch_ms_mean", "fetch_ms_p50", "total_ms_mean", "top10_same"]
RAW_COLS = ["rep", "i", "qi", "setting", "search_ms", "fetch_ms", "hits", "top10_same"]
# 11/22 추가: 구간마다 Milvus CPU(코어)와 서버 부하 평균을 따로 기록 (탐색 세션 CSV의 칸 구성을 바꾸지 않으려고 별도 파일)
MCPU_COLS = ["ts_utc", "ts_kst", "phase", "rep", "slot", "setting", "cap", "qps", "milvus_cores", "loadavg1_start", "loadavg1_end"]


def milvus_cpu_seconds(B):
    """Milvus 메트릭(localhost:9091)의 process_cpu_seconds_total 합. check_busy.sh와 같은 방식. 못 읽으면 None."""
    if isinstance(B, bv.MockBackend):
        return None
    try:
        import urllib.request
        txt = urllib.request.urlopen("http://localhost:9091/metrics", timeout=5).read().decode()
    except Exception:
        return None
    vals = [float(l.split()[-1]) for l in txt.splitlines() if l.startswith("process_cpu_seconds_total")]
    return sum(vals) if vals else None


def loadavg1():
    try:
        return float(open("/proc/loadavg").read().split()[0])
    except Exception:
        return None


def sp(ef):
    return {"metric_type": "IP", "params": {"ef": ef}}


def ids_of(res):
    return [bv.hit_pk(h) for h in res[0]]


# ---------- 검색 (설정마다 limit, ef만 다르다) ----------
def search_sync(B, vec, s):
    if isinstance(B, bv.MockBackend):
        time.sleep(0.003 + 0.00002 * s["limit"] + 0.00001 * (s["ef"] - 64))
        return [int(x) for x in np.random.randint(0, B.n, s["limit"])]
    r = B.c0.search(B.name, [vec], limit=s["limit"], search_params=sp(s["ef"]), output_fields=["pk"])
    return ids_of(r)


async def search_async(B, vec, s):
    if isinstance(B, bv.MockBackend):
        await asyncio.sleep(0.003 + 0.00002 * s["limit"] + 0.00001 * (s["ef"] - 64))
        return [int(x) for x in np.random.randint(0, B.n, s["limit"])]
    r = await B.ac.search(B.name, [vec], limit=s["limit"], search_params=sp(s["ef"]), output_fields=["pk"])
    return ids_of(r)


def fetch_sync(B, pks):
    got = B.fetch(pks, "none")
    if len(got) != len(pks):
        raise RuntimeError(f"bad fetch: got {len(got)} rows for {len(pks)} pks")


async def fetch_async(B, pks):
    got = await B.afetch(pks, "none")   # m7과 같은 조회 경로 (merge 없음, PK 목록 한 번)
    if len(got) != len(pks):
        raise RuntimeError(f"bad fetch: got {len(got)} rows for {len(pks)} pks")


# ---------- 미리 저장한 PK 목록 ----------
class Fixed:
    def __init__(self, outdir, ds):
        z = np.load(f"{outdir}/fixed_{ds}.npz")
        self.qid = z["qid"]                          # 쓰는 질문의 query_vecs 번호
        self.top10 = z["top10"]                      # (질문 수, 10)
        off, flat = z["ctx_off"], z["ctx_flat"]      # 질문 i의 PK 목록 = flat[off[i]:off[i+1]]
        self.ctx = [flat[off[i]:off[i + 1]].tolist() for i in range(len(off) - 1)]
        self.top10_set = [set(t.tolist()) for t in self.top10]
        self.nq = len(self.ctx)

    def same(self, j, ids):
        """검색 결과 앞 10개가 미리 저장한 top-10과 몇 개 같은가 (0~1). 설정 A/B/C의 검색 결과 차이 확인용."""
        return len(self.top10_set[j].intersection(ids[:10])) / 10.0


def pick_queries(M, a):
    """쓸 질문을 고른다. all: query_vecs 전체(실제 질문 + 가짜 질문)에서 고정 시드로 nsub개 (지난주 m7은 전체에서 무작위로 뽑았음).
    real: 실제 NQ 질문만 (meta.json의 n_real_queries개, query_vecs 앞부분)."""
    if a.qset == "real":
        meta = json.load(open(f"{bv.DATA}/meta.json"))
        return np.arange(int(meta["n_real_queries"]))
    n = len(M.qv)
    if a.nsub >= n:
        return np.arange(n)
    return np.sort(np.random.default_rng(7).choice(n, size=a.nsub, replace=False))


def run_prep(B, M, a):
    os.makedirs(a.outdir, exist_ok=True)
    path = f"{a.outdir}/fixed_{a.ds}.npz"
    if os.path.exists(path):
        raise SystemExit(f"{path}가 이미 있음. 같은 목록을 계속 써야 하므로 덮어쓰지 않는다.")
    t0 = time.time()
    qids = pick_queries(M, a)
    print(f"prep: {len(qids)} queries (qset={a.qset}, query_vecs {len(M.qv)})", flush=True)
    top10, ctx_off, ctx_flat = [], [0], []
    for j, qi in enumerate(qids):
        ids = search_sync(B, M.qv[qi], SETTINGS["A"])
        ctx = M.context(ids, a.expand)
        fetch_sync(B, ctx)                      # 목록을 한 번 읽어 캐시를 데운다 (모든 설정이 같은 캐시 상태에서 시작)
        top10.append(ids[:10]); ctx_flat += ctx; ctx_off.append(len(ctx_flat))
        if (j + 1) % 1000 == 0:
            print(f"  prep {j + 1}/{len(qids)} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(path, qid=np.asarray(qids, dtype=np.int64), top10=np.array(top10, dtype=np.int64),
             ctx_off=np.array(ctx_off, dtype=np.int64), ctx_flat=np.array(ctx_flat, dtype=np.int64))
    sizes = np.diff(ctx_off)
    info = dict(ds=a.ds, qset=a.qset, queries=len(qids), query_vecs=len(M.qv), expand=a.expand, setting=SETTINGS["A"],
                pks_mean=round(float(sizes.mean()), 2),
                pks_min=int(sizes.min()), pks_max=int(sizes.max()), seconds=round(time.time() - t0, 1), version=B.version)
    # ef < limit 검색이 거부되는지 확인 (ef=100으로 정한 근거 확인용, 측정과 무관)
    try:
        r = search_sync(B, M.qv[int(qids[0])], dict(limit=100, ef=64))
        info["ef64_limit100"] = f"accepted, {len(r)} hits"
    except Exception as e:
        info["ef64_limit100"] = f"rejected: {repr(e)[:300]}"
    json.dump(info, open(f"{a.outdir}/fixed_{a.ds}_info.json", "w"), ensure_ascii=False, indent=1)
    print("prep", info, flush=True)


def run_warmup(B, F):
    """저장한 PK 목록을 한 번씩 읽어 캐시를 데운다 (컬렉션을 다시 올린 세션에서 prep과 같은 캐시 상태를 만들기 위함)."""
    t0 = time.time()
    for j, pks in enumerate(F.ctx):
        fetch_sync(B, pks)
        if (j + 1) % 2000 == 0:
            print(f"  warmup {j + 1}/{F.nq} ({time.time() - t0:.0f}s)", flush=True)
    print(f"warmup: {F.nq} lists in {time.time() - t0:.1f}s", flush=True)


def pctl(x, q):
    return round(float(np.percentile(x, q)), 3) if len(x) else ""


def run_seq(B, M, F, a):
    """요청을 하나씩. 같은 질문에 설정 A/B/C를 번갈아 적용(질문마다 순서 회전). 대기열이 없으므로 순수한 호출 비용."""
    rng = np.random.default_rng(a.seed)
    order = ["A", "B", "C"]
    per = {s: dict(S=[], R=[], same=[]) for s in order}
    raw = []
    for i in range(a.warm_q + a.reqs):
        j = int(rng.integers(0, F.nq)); qi = int(F.qid[j])
        rot = order[i % 3:] + order[:i % 3]
        for s in rot:
            t0 = time.perf_counter(); ids = search_sync(B, M.qv[qi], SETTINGS[s]); t1 = time.perf_counter()
            fetch_sync(B, F.ctx[j]); t2 = time.perf_counter()
            if i >= a.warm_q:
                sm, fm, same = (t1 - t0) * 1e3, (t2 - t1) * 1e3, F.same(j, ids)
                per[s]["S"].append(sm); per[s]["R"].append(fm); per[s]["same"].append(same)
                raw.append(dict(rep=a.rep, i=i - a.warm_q, qi=qi, setting=s, search_ms=round(sm, 3),
                                fetch_ms=round(fm, 3), hits=len(ids), top10_same=same))
    rows = []
    for s in order:
        S, R = per[s]["S"], per[s]["R"]
        rows.append(dict(version=B.version, ds=a.ds, rep=a.rep, setting=s, limit=SETTINGS[s]["limit"], ef=SETTINGS[s]["ef"],
                         reqs=len(S), search_ms_mean=round(float(np.mean(S)), 3), search_ms_p50=pctl(S, 50),
                         search_ms_p95=pctl(S, 95), fetch_ms_mean=round(float(np.mean(R)), 3), fetch_ms_p50=pctl(R, 50),
                         total_ms_mean=round(float(np.mean(S) + np.mean(R)), 3),
                         top10_same=round(float(np.mean(per[s]["same"])), 4)))
    return rows, raw


def run_load(B, M, F, a):
    s = SETTINGS[a.setting]
    t_begin = time.perf_counter(); t_warm = t_begin + a.warm; t_stop = t_warm + a.dur
    snaps = {}

    def monitor():
        time.sleep(max(0, t_warm - time.perf_counter()))
        snaps.update(t0=time.perf_counter(), cpu0=time.process_time(), d0=bv.disk_snap(a.dev),
                     m0=milvus_cpu_seconds(B), la0=loadavg1())
        time.sleep(max(0, t_stop - time.perf_counter()))
        snaps.update(t1=time.perf_counter(), cpu1=time.process_time(), d1=bv.disk_snap(a.dev),
                     m1=milvus_cpu_seconds(B), la1=loadavg1())

    mon = threading.Thread(target=monitor, daemon=True); mon.start()
    max_err = int(os.environ.get("BENCH_MAX_ERR", "20"))
    errc, halt, first_err = [0], [False], []

    def note_err(e):
        errc[0] += 1
        if not first_err:
            first_err.append(repr(e)); print("첫 오류:", repr(e), file=sys.stderr)
        if errc[0] > max_err and not halt[0]:
            halt[0] = True; print(f"HALT: 오류 {errc[0]}회, 이 구간을 멈춤", file=sys.stderr)

    async def main():
        await B.aopen()
        rsem = asyncio.Semaphore(a.cap)   # bench_v2 m7과 같은 요청 전체 상한

        async def one(j):
            vec, pks = M.qv[int(F.qid[j])], F.ctx[j]
            t0 = time.perf_counter()
            async with rsem:
                t1 = time.perf_counter()
                ids = await search_async(B, vec, s)
                t2 = time.perf_counter()
                await fetch_async(B, pks)
                t3 = time.perf_counter()
            return (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, len(ids), len(pks), F.same(j, ids)

        async def client(cid):
            rng, out = random.Random(a.seed * 100000 + a.rep * 1000 + cid), []
            while time.perf_counter() < t_stop and not halt[0]:
                j = rng.randrange(F.nq); t = time.perf_counter()
                try:
                    st = await one(j); ok = 1
                except Exception as e:
                    st, ok = None, 0; note_err(e); await asyncio.sleep(0.2)
                d = time.perf_counter(); out.append((d, (d - t) * 1e3, ok, st))
            return out

        res = await asyncio.gather(*[client(i) for i in range(a.n)])
        await B.aclose()
        return [r for o in res for r in o]

    recs = asyncio.run(main())
    mon.join(timeout=5)
    if halt[0]:
        snaps.pop("t1", None)
    done = [r for r in recs if r[0] >= t_warm and r[2] == 1]
    errors = sum(1 for r in recs if r[0] >= t_warm and r[2] == 0)
    row = dict(version=B.version, layout=a.layout, ds=a.ds, phase=a.phase, rep=a.rep, slot=a.slot, setting=a.setting,
               limit=s["limit"], ef=s["ef"], n=a.n, cap=a.cap, completed=len(done), errors=errors, disk_dev=a.dev)
    if "t1" in snaps and done:
        dur = snaps["t1"] - snaps["t0"]
        lat = np.array([r[1] for r in done]); st = np.array([r[3] for r in done], dtype=float)
        W, S, R = st[:, 0], st[:, 1], st[:, 2]
        row.update(dur_s=round(dur, 2), qps=round(len(done) / dur, 1), mean_ms=round(float(lat.mean()), 2),
                   p50_ms=pctl(lat, 50), p95_ms=pctl(lat, 95), p99_ms=pctl(lat, 99),
                   wait_ms_mean=round(float(W.mean()), 3), wait_ms_p50=pctl(W, 50), wait_ms_p99=pctl(W, 99),
                   search_ms_mean=round(float(S.mean()), 3), search_ms_p50=pctl(S, 50), search_ms_p99=pctl(S, 99),
                   fetch_ms_mean=round(float(R.mean()), 3), fetch_ms_p50=pctl(R, 50), fetch_ms_p99=pctl(R, 99),
                   hits_per_req=round(float(st[:, 3].mean()), 2), fetched_pks_per_req=round(float(st[:, 4].mean()), 2),
                   top10_same=round(float(st[:, 5].mean()), 4),
                   client_cores=round((snaps["cpu1"] - snaps["cpu0"]) / dur, 2))
        if snaps.get("m0") is not None and snaps.get("m1") is not None:
            row["milvus_cores"] = round((snaps["m1"] - snaps["m0"]) / dur, 2)
        row["loadavg1_start"], row["loadavg1_end"] = snaps.get("la0"), snaps.get("la1")
        if snaps["d0"] and snaps["d1"]:
            x0, x1 = snaps["d0"], snaps["d1"]
            row.update(r_iops=round((x1["r"] - x0["r"]) / dur), read_mbps=round((x1["sec"] - x0["sec"]) * 512 / dur / 1e6, 1))
    return row


def make_mock_fixed(M, a):
    """selftest용: 서버 없이 PK 목록 파일을 만든다."""
    os.makedirs(a.outdir, exist_ok=True)
    rng = np.random.default_rng(0)
    qids = np.arange(min(300, len(M.qv)))
    top10, off, flat = [], [0], []
    for qi in qids:
        ids = [int(x) for x in rng.integers(0, M.n, 10)]
        ctx = M.context(ids, a.expand); top10.append(ids); flat += ctx; off.append(len(flat))
    np.savez(f"{a.outdir}/fixed_{a.ds}.npz", qid=qids, top10=np.array(top10), ctx_off=np.array(off), ctx_flat=np.array(flat))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prep", "warmup", "seq", "load"], default="load")
    ap.add_argument("--ds", default="nq"); ap.add_argument("--layout", default="seq")
    ap.add_argument("--outdir", default="/workspace/junlim/c3a_out")
    ap.add_argument("--setting", choices=sorted(SETTINGS), default="A")
    ap.add_argument("--cap", type=int, default=12); ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--dur", type=float, default=20); ap.add_argument("--warm", type=float, default=5)
    ap.add_argument("--expand", type=int, default=1)
    ap.add_argument("--qset", choices=["all", "real"], default="all", help="prep: 쓸 질문 집합")
    ap.add_argument("--nsub", type=int, default=10000, help="prep, qset=all: 전체에서 고정 시드로 뽑을 질문 수")
    ap.add_argument("--phase", default="explore", help="explore | confirm | anchor (결과 CSV의 phase 칸)")
    ap.add_argument("--rep", type=int, default=1); ap.add_argument("--slot", type=int, default=0, help="회차 안의 순서")
    ap.add_argument("--reqs", type=int, default=400); ap.add_argument("--warm-q", dest="warm_q", type=int, default=30)
    ap.add_argument("--seed", type=int, default=13); ap.add_argument("--dev", default="loop1")
    ap.add_argument("--backend", choices=["milvus", "mock"], default="milvus")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    M = bv.Mapper(a.layout)
    B = bv.MockBackend(M.n) if a.backend == "mock" else bv.MilvusBackend(a.layout)

    if a.selftest:
        make_mock_fixed(M, a); F = Fixed(a.outdir, a.ds)
        a.reqs, a.warm_q = 12, 2
        rows, raw = run_seq(B, M, F, a)
        for r in rows:
            bv.write_row(f"{a.outdir}/selftest_seq.csv", SEQ_COLS, r)
        ok = all(r["reqs"] == 12 for r in rows) and len(raw) == 36
        a.dur, a.warm, a.n = 1.5, 0.3, 16
        for s in sorted(SETTINGS):
            for k in (6, 12):
                a.setting, a.cap = s, k
                r = run_load(B, M, F, a); bv.write_row(f"{a.outdir}/selftest_load.csv", LOAD_COLS, r)
                good = r["errors"] == 0 and r["completed"] > 0 and r["hits_per_req"] == SETTINGS[s]["limit"]
                ok &= good
                print(f"{s} k={k} {'OK ' if good else 'FAIL'} qps={r.get('qps')} wait={r.get('wait_ms_mean')} "
                      f"search={r.get('search_ms_mean')} fetch={r.get('fetch_ms_mean')} hits={r.get('hits_per_req')}")
        print("SELFTEST", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)

    if a.mode == "prep":
        run_prep(B, M, a); return
    F = Fixed(a.outdir, a.ds)
    a.seed = a.seed * 1000 + a.rep
    if a.mode == "warmup":
        run_warmup(B, F); return
    if a.mode == "seq":
        rows, raw = run_seq(B, M, F, a)
        for r in rows:
            bv.write_row(f"{a.outdir}/results_c3a_seq_{a.ds}.csv", SEQ_COLS, r); print(r)
        rp = f"{a.outdir}/results_c3a_seq_raw_{a.ds}.csv"
        new = not os.path.exists(rp)
        with open(rp, "a", encoding="utf-8") as f:
            if new:
                f.write(",".join(RAW_COLS) + "\n")
            for r in raw:
                f.write(",".join(str(r[c]) for c in RAW_COLS) + "\n")
        return
    r = run_load(B, M, F, a)
    bv.write_row(f"{a.outdir}/results_c3a_load_{a.ds}.csv", LOAD_COLS, r)
    bv.write_row(f"{a.outdir}/results_c3a_mcpu_{a.ds}.csv", MCPU_COLS, r)
    print({k: r.get(k) for k in ("phase", "rep", "setting", "cap", "qps", "p99_ms", "wait_ms_mean", "search_ms_mean",
                                 "fetch_ms_mean", "errors", "r_iops", "milvus_cores", "loadavg1_end")}, flush=True)
    if r.get("qps") in (None, "") or not r.get("completed"):
        print("FAILED RUN: 처리량을 잴 수 없음", file=sys.stderr); sys.exit(3)


if __name__ == "__main__":
    main()
