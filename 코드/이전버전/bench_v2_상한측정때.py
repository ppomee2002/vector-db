#!/usr/bin/env python3
"""Milvus 위에서 실행 방식별 처리량·지연을 재는 도구.
요청 1건 = 벡터 검색(top-k ID) -> 이웃 chunk까지 문맥 확장 -> 원문 조회. 방식마다 실행 방법만 다르다.
  m0 DB 기본 경로(search +text 한 번, 이웃 확장 없음: 참고용)   m1 동기(worker N개)   m2 스레드 증설
  m3 비동기 I/O   m4 비동기 + 쿼리 안 병합(PK 연속 구간)   m5 비동기 + 쿼리 간 병합   m6 비동기 + 동시 읽기 상한
"""
import argparse, asyncio, csv, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

DATA = os.environ.get("BENCH_DATA", "/workspace/junlim/data")
COLL = os.environ.get("BENCH_COLL", "fiqa")
CHUNK_LIMIT = 512  # query 한 번에 넣는 PK 수 상한
PRESET = {
    "m0": dict(mode="pool", merge="none"), "m1": dict(mode="pool", merge="none"),
    "m2": dict(mode="threads", merge="none"), "m3": dict(mode="async", merge="none"),
    "m4": dict(mode="async", merge="range"), "m5": dict(mode="async", merge="xq"),
    "m6": dict(mode="async", merge="none", cap=True),
    "m7": dict(mode="async", merge="none", rcap=True),  # 요청 전체(검색+읽기)의 동시 수 상한 = --cap
    "m8": dict(mode="async", merge="xq", rcap=True),    # m7 + 쿼리 간 병합
}
COLS = ["ts_utc", "ts_kst", "version", "layout", "cond", "rep", "method", "mode", "n", "workers", "cap",
        "window_ms", "maxb", "topk", "expand", "dur_s", "completed", "errors", "bad", "qps", "p50_ms", "p95_ms",
        "p99_ms", "search_calls_per_req", "fetch_calls_per_req", "req_pks_per_req", "fetched_pks_per_req",
        "text_chars_per_req", "client_cores", "disk_dev", "r_iops", "read_mbps", "r_await_ms", "util_pct",
        "avg_qd", "n_chunks"]
SR_COLS = ["ts_utc", "ts_kst", "version", "layout", "cond", "topk", "expand", "reps", "ctx_pks_mean",
           "S_ids_p50", "S_ids_p95", "R_fetch_p50", "R_fetch_p95", "floor_p50", "R_net_p50",
           "search_text_p50", "gain_max"]


def now_str():
    t = time.time()
    return (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t)),
            time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t + 9 * 3600)))


def f_in(pks):
    return "pk in [" + ",".join(map(str, pks)) + "]"


def f_range(pks):
    pks = sorted(pks)
    rs, a, b = [], pks[0], pks[0]
    for p in pks[1:]:
        if p == b + 1:
            b = p
        else:
            rs.append((a, b)); a = b = p
    rs.append((a, b))
    return " or ".join(f"pk == {x}" if x == y else f"(pk >= {x} and pk <= {y})" for x, y in rs)


def hit_pk(h):
    """Get the PK from one search hit; the location differs by pymilvus version, so try several."""
    for get in (lambda: h["id"], lambda: h.id, lambda: h["pk"], lambda: h.entity["pk"], lambda: h["entity"]["pk"]):
        try:
            v = get()
            if v is not None:
                return int(v)
        except Exception:
            pass
    keys = list(h.keys()) if hasattr(h, "keys") else dir(h)
    raise KeyError(f"no pk in hit: type={type(h)} keys={keys}")


class Mapper:
    def __init__(self, layout):
        pk_of_chunk = np.load(f"{DATA}/pk_{layout}.npy")
        self.n = len(pk_of_chunk)
        self.pk_of_chunk = pk_of_chunk
        self.chunk_of_pk = np.empty(self.n, dtype=np.int64)
        self.chunk_of_pk[pk_of_chunk] = np.arange(self.n)
        if os.path.exists(f"{DATA}/doc_idx.npy"):
            self.doc = np.load(f"{DATA}/doc_idx.npy")
        else:
            with open(f"{DATA}/chunks.jsonl", encoding="utf-8") as f:
                self.doc = np.array([json.loads(line)["doc_idx"] for line in f])
        self.qv = np.load(f"{DATA}/query_vecs.npy")

    def context(self, pks, expand):
        out = set()
        for p in pks:
            k = int(self.chunk_of_pk[p]); out.add(int(p))
            for d in range(1, expand + 1):
                for kk in (k - d, k + d):
                    if 0 <= kk < self.n and self.doc[kk] == self.doc[k]:
                        out.add(int(self.pk_of_chunk[kk]))
        return sorted(out)


class Base:
    def __init__(self):
        self.lock = threading.Lock()
        self.c = dict(search=0, fetch=0, pks=0, chars=0, req_pks=0)

    def count(self, **kw):
        with self.lock:
            for k, v in kw.items():
                self.c[k] += v

    def snap(self):
        with self.lock:
            return dict(self.c)


class MilvusBackend(Base):
    def __init__(self, layout, ef=64):
        super().__init__()
        from pymilvus import MilvusClient, __version__
        self.uri, self.db, self.name, self.ac = "http://localhost:19530", "junlim", f"{COLL}_{layout}", None
        self.c0 = MilvusClient(uri=self.uri, db_name=self.db)
        self.sp = {"metric_type": "IP", "params": {"ef": ef}}
        self.version = f"pymilvus {__version__} / server {self.c0.get_server_version()}"

    def search_ids(self, vec, k):
        r = self.c0.search(self.name, [vec], limit=k, search_params=self.sp, output_fields=["pk"])
        self.count(search=1)
        return [hit_pk(h) for h in r[0]]

    def search_text(self, vec, k):
        r = self.c0.search(self.name, [vec], limit=k, search_params=self.sp, output_fields=["text"])
        self.count(search=1, fetch=0)
        return r

    def _parts(self, pks, merge):
        for i in range(0, len(pks), CHUNK_LIMIT):
            part = pks[i:i + CHUNK_LIMIT]
            yield (f_range(part) if merge == "range" else f_in(part)), len(part)

    def fetch(self, pks, merge="none"):
        got = {}
        for filt, m in self._parts(pks, merge):
            rows = self.c0.query(self.name, filter=filt, output_fields=["pk", "text"])
            got.update({int(r["pk"]): r["text"] for r in rows})
            self.count(fetch=1, pks=m, chars=sum(len(r["text"]) for r in rows))
        return got

    async def aopen(self):
        from pymilvus import AsyncMilvusClient
        self.ac = AsyncMilvusClient(uri=self.uri, db_name=self.db)

    async def aclose(self):
        try:
            await self.ac.close()
        except Exception:
            pass

    async def asearch_ids(self, vec, k):
        r = await self.ac.search(self.name, [vec], limit=k, search_params=self.sp, output_fields=["pk"])
        self.count(search=1)
        return [hit_pk(h) for h in r[0]]

    async def afetch(self, pks, merge="none"):
        got = {}
        for filt, m in self._parts(pks, merge):
            rows = await self.ac.query(self.name, filter=filt, output_fields=["pk", "text"])
            got.update({int(r["pk"]): r["text"] for r in rows})
            self.count(fetch=1, pks=m, chars=sum(len(r["text"]) for r in rows))
        return got


class MockBackend(Base):  # 도구 점검용: 서버 없이 지연만 흉내 낸다
    version = "mock"

    def __init__(self, n):
        super().__init__(); self.n = n

    def search_ids(self, vec, k):
        time.sleep(0.003); self.count(search=1)
        return [int(x) for x in np.random.randint(0, self.n, k)]

    def search_text(self, vec, k):
        time.sleep(0.004); self.count(search=1)

    def fetch(self, pks, merge="none"):
        time.sleep(0.004 + 0.0002 * len(pks)); self.count(fetch=1, pks=len(pks), chars=len(pks) * 400)
        return {p: "x" * 400 for p in pks}

    async def aopen(self): pass
    async def aclose(self): pass

    async def asearch_ids(self, vec, k):
        await asyncio.sleep(0.003); self.count(search=1)
        return [int(x) for x in np.random.randint(0, self.n, k)]

    async def afetch(self, pks, merge="none"):
        await asyncio.sleep(0.004 + 0.0002 * len(pks)); self.count(fetch=1, pks=len(pks), chars=len(pks) * 400)
        return {p: "x" * 400 for p in pks}


class XBatcher:
    """짧은 시간(window) 동안 모인 요청들의 PK를 합쳐 중복을 없애고 한 번에 조회한 뒤 요청별로 나눠 준다."""
    def __init__(self, fetch, window, maxb):
        self.fetch, self.window, self.maxb, self.pending, self.timer = fetch, window, maxb, [], None

    async def get(self, pks):
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self.pending.append((pks, fut))
        if len(self.pending) >= self.maxb:
            self._flush()
        elif self.timer is None:
            self.timer = loop.call_later(self.window, self._flush)
        return await fut

    def _flush(self):
        if self.timer:
            self.timer.cancel(); self.timer = None
        batch, self.pending = self.pending, []
        if batch:
            asyncio.ensure_future(self._run(batch))

    async def _run(self, batch):
        union = sorted({p for pks, _ in batch for p in pks})
        try:
            rows = await self.fetch(union)
            for pks, fut in batch:
                fut.set_result({p: rows[p] for p in pks if p in rows})
        except Exception as e:
            for _, fut in batch:
                if not fut.done():
                    fut.set_exception(e)


def disk_snap(dev):
    try:
        with open("/proc/diskstats") as f:
            for line in f:
                p = line.split()
                if len(p) > 13 and p[2] == dev:
                    return dict(r=int(p[3]), sec=int(p[5]), rms=int(p[6]), io=int(p[12]), wq=int(p[13]))
    except Exception:
        pass
    return None


def handle_sync(B, M, a, qi):
    if a.method == "m0":
        B.search_text(M.qv[qi], a.topk)
        return 1
    ids = B.search_ids(M.qv[qi], a.topk)
    ctx = M.context(ids, a.expand)
    B.count(req_pks=len(ctx))
    return int(len(B.fetch(ctx, PRESET[a.method]["merge"])) == len(ctx))


def run_load(B, M, a):
    pre = PRESET[a.method]
    t_begin = time.perf_counter(); t_warm = t_begin + a.warm; t_stop = t_warm + a.dur
    snaps = {}

    def monitor():
        time.sleep(max(0, t_warm - time.perf_counter()))
        snaps.update(t0=time.perf_counter(), cpu0=time.process_time(), d0=disk_snap(a.dev), c0=B.snap())
        time.sleep(max(0, t_stop - time.perf_counter()))
        snaps.update(t1=time.perf_counter(), cpu1=time.process_time(), d1=disk_snap(a.dev), c1=B.snap())

    mon = threading.Thread(target=monitor, daemon=True); mon.start()
    recs, first_err = [], []

    def note_err(e):
        if not first_err:
            first_err.append(repr(e)); print("첫 오류:", repr(e), file=sys.stderr)

    if pre["mode"] in ("pool", "threads"):
        pool = ThreadPoolExecutor(max_workers=a.workers) if pre["mode"] == "pool" else None
        outs = [[] for _ in range(a.n)]

        def client(cid):
            rng, out = random.Random(a.seed * 1000 + cid), outs[cid]
            while time.perf_counter() < t_stop:
                qi = rng.randrange(len(M.qv)); t = time.perf_counter()
                try:
                    ok = pool.submit(handle_sync, B, M, a, qi).result() if pool else handle_sync(B, M, a, qi)
                except Exception as e:
                    ok = -1; note_err(e); time.sleep(0.05)
                d = time.perf_counter(); out.append((d, (d - t) * 1e3, ok))

        ths = [threading.Thread(target=client, args=(i,)) for i in range(a.n)]
        [t.start() for t in ths]; [t.join() for t in ths]
        if pool:
            pool.shutdown()
        recs = [r for o in outs for r in o]
    else:
        async def main():
            await B.aopen()
            sem = asyncio.Semaphore(a.cap) if pre.get("cap") else None
            rsem = asyncio.Semaphore(a.cap) if pre.get("rcap") else None
            bat = XBatcher(lambda p: B.afetch(p, "none"), a.window_ms / 1e3, a.maxb) if pre["merge"] == "xq" else None

            async def fx(pks):
                if bat:
                    return await bat.get(pks)
                if sem:
                    async with sem:
                        return await B.afetch(pks, pre["merge"])
                return await B.afetch(pks, pre["merge"])

            async def one(qi):
                ids = await B.asearch_ids(M.qv[qi], a.topk)
                ctx = M.context(ids, a.expand); B.count(req_pks=len(ctx))
                return int(len(await fx(ctx)) == len(ctx))

            async def client(cid):
                rng, out = random.Random(a.seed * 1000 + cid), []
                while time.perf_counter() < t_stop:
                    qi = rng.randrange(len(M.qv)); t = time.perf_counter()
                    try:
                        if rsem:
                            async with rsem:
                                ok = await one(qi)
                        else:
                            ok = await one(qi)
                    except Exception as e:
                        ok = -1; note_err(e); await asyncio.sleep(0.05)
                    d = time.perf_counter(); out.append((d, (d - t) * 1e3, ok))
                return out

            res = await asyncio.gather(*[client(i) for i in range(a.n)])
            await B.aclose()
            return [r for o in res for r in o]
        recs = asyncio.run(main())

    mon.join(timeout=5)
    done = [r for r in recs if r[0] >= t_warm and r[2] != -1]
    errors = sum(1 for r in recs if r[0] >= t_warm and r[2] == -1)
    bad = sum(1 for r in done if r[2] == 0)
    row = dict(layout=a.layout, cond=a.cond, rep=a.rep, method=a.method, mode=pre["mode"], n=a.n,
               workers=a.workers if pre["mode"] == "pool" else "", cap=a.cap if (pre.get("cap") or pre.get("rcap")) else "",
               window_ms=a.window_ms if pre["merge"] == "xq" else "", maxb=a.maxb if pre["merge"] == "xq" else "",
               topk=a.topk, expand=a.expand if a.method != "m0" else 0, completed=len(done), errors=errors,
               bad=bad, version=B.version, disk_dev=a.dev, n_chunks=M.n)
    if "t1" in snaps and done:
        dur = snaps["t1"] - snaps["t0"]; lat = np.array([r[1] for r in done]); c0, c1 = snaps["c0"], snaps["c1"]
        d = {k: c1[k] - c0[k] for k in c1}; nd = max(len(done), 1)
        row.update(dur_s=round(dur, 2), qps=round(len(done) / dur, 1), p50_ms=round(float(np.percentile(lat, 50)), 2),
                   p95_ms=round(float(np.percentile(lat, 95)), 2), p99_ms=round(float(np.percentile(lat, 99)), 2),
                   search_calls_per_req=round(d["search"] / nd, 3), fetch_calls_per_req=round(d["fetch"] / nd, 3),
                   req_pks_per_req=round(d["req_pks"] / nd, 2), fetched_pks_per_req=round(d["pks"] / nd, 2),
                   text_chars_per_req=round(d["chars"] / nd), client_cores=round((snaps["cpu1"] - snaps["cpu0"]) / dur, 2))
        if snaps["d0"] and snaps["d1"]:
            x0, x1 = snaps["d0"], snaps["d1"]; dr = x1["r"] - x0["r"]
            row.update(r_iops=round(dr / dur), read_mbps=round((x1["sec"] - x0["sec"]) * 512 / dur / 1e6, 1),
                       r_await_ms=round((x1["rms"] - x0["rms"]) / dr, 3) if dr else 0,
                       util_pct=round((x1["io"] - x0["io"]) / (dur * 10), 1), avg_qd=round((x1["wq"] - x0["wq"]) / (dur * 1000), 2))
    return row


def pctl(x, q):
    return round(float(np.percentile(x, q)), 3)


def run_sr(B, M, a):
    rng = np.random.default_rng(a.seed)
    S, R, F, T, C = [], [], [], [], []
    for i in range(a.warm_q + a.reps):
        qi = int(rng.integers(0, len(M.qv)))
        t0 = time.perf_counter(); ids = B.search_ids(M.qv[qi], a.topk); t1 = time.perf_counter()
        ctx = M.context(ids, a.expand); t2 = time.perf_counter()
        B.fetch(ctx); t3 = time.perf_counter()
        B.fetch([ids[0]]); t4 = time.perf_counter()
        B.search_text(M.qv[qi], a.topk); t5 = time.perf_counter()
        if i >= a.warm_q:
            S.append((t1 - t0) * 1e3); R.append((t3 - t2) * 1e3); F.append((t4 - t3) * 1e3)
            T.append((t5 - t4) * 1e3); C.append(len(ctx))
    s, r = pctl(S, 50), pctl(R, 50)
    return dict(layout=a.layout, cond=a.cond, topk=a.topk, expand=a.expand, reps=a.reps, version=B.version,
                ctx_pks_mean=round(float(np.mean(C)), 1), S_ids_p50=s, S_ids_p95=pctl(S, 95), R_fetch_p50=r,
                R_fetch_p95=pctl(R, 95), floor_p50=pctl(F, 50), R_net_p50=round(r - pctl(F, 50), 3),
                search_text_p50=pctl(T, 50), gain_max=round((s + r) / max(s, r), 3))


def write_row(path, cols, row):
    row = dict(row); row["ts_utc"], row["ts_kst"] = now_str()
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in cols})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="m3", choices=sorted(PRESET)); p.add_argument("--n", type=int, default=8)
    p.add_argument("--dur", type=float, default=20); p.add_argument("--warm", type=float, default=5)
    p.add_argument("--layout", default="seq"); p.add_argument("--topk", type=int, default=10)
    p.add_argument("--expand", type=int, default=1); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cap", type=int, default=8); p.add_argument("--window-ms", dest="window_ms", type=float, default=2)
    p.add_argument("--maxb", type=int, default=32); p.add_argument("--dev", default="loop1")
    p.add_argument("--cond", default="base"); p.add_argument("--rep", type=int, default=1)
    p.add_argument("--seed", type=int, default=1); p.add_argument("--backend", default="milvus", choices=["milvus", "mock"])
    p.add_argument("--out", default="/workspace/junlim/results.csv")
    p.add_argument("--sr", action="store_true"); p.add_argument("--reps", type=int, default=300)
    p.add_argument("--warm-q", dest="warm_q", type=int, default=50); p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    M = Mapper(a.layout)
    B = MockBackend(M.n) if a.backend == "mock" else MilvusBackend(a.layout)
    if a.selftest:
        bad = 0
        for m in sorted(PRESET):
            a.method, a.n, a.warm, a.dur = m, 4, 1, 2
            r = run_load(B, M, a); write_row(a.out.replace(".csv", "_selftest.csv"), COLS, r)
            ok = r["errors"] == 0 and r["bad"] == 0 and r["completed"] > 0
            bad += not ok
            print(f"{m} {'OK ' if ok else 'FAIL'} completed={r['completed']} qps={r.get('qps')} p95={r.get('p95_ms')} "
                  f"fetch/req={r.get('fetch_calls_per_req')} errors={r['errors']} bad={r['bad']}")
        r = run_sr(B, M, argparse.Namespace(**{**vars(a), "reps": 20, "warm_q": 5})); print("sr", r)
        print("SELFTEST", "PASS" if not bad else "FAIL"); return
    if a.sr:
        r = run_sr(B, M, a); write_row(a.out.replace(".csv", "_sr.csv"), SR_COLS, r); print(r); return
    r = run_load(B, M, a); write_row(a.out, COLS, r)
    print({k: r.get(k) for k in ("method", "n", "qps", "p50_ms", "p95_ms", "errors", "bad", "fetch_calls_per_req", "client_cores", "r_iops")})


if __name__ == "__main__":
    main()
