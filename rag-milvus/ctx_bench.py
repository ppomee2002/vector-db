#!/usr/bin/env python3
"""Same-budget comparison of context-fetch layers in front of a text store (NQ).

The VectorDB search is not re-run: each request sleeps for the search latency Milvus measured
without text (--search-from), then fetches the text of its top-10 chunks (+-w neighbours of the
same article) through one of these layers, all with the same cache budget (bytes of cached text):

  cell2m   LRU of 2 MB cells, one read per missing cell (how Milvus tiered storage caches text)
  chunk    LRU of chunks, one read per missing chunk, issued in parallel (simple chunk cache)
  block    LRU of 4 KB blocks in pk (= article) order, missing blocks of one request merged into
           contiguous ranges, one read per range
  group    block + ranges of one request less than --gap apart fetched as one read, and blocks
           already being fetched by another request are waited on instead of read again

Text store backends: local  = one file on NVMe read with O_DIRECT (no page cache)
                     s3     = 64 MB objects on SeaweedFS S3, ranged GETs (~33 ms per GET regardless of size)
Requests: closed loop, --concurrency workers; trace generated exactly like bench.py.
"""
import argparse, collections, json, mmap, os, threading, time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
D = ROOT / "datasets/ctx/nq"
SLICE = 64 << 20
BLK = 4096
CELL = 2 << 20


class Local:
    def __init__(self):
        self.fd = os.open(D / "text.bin", os.O_RDONLY | os.O_DIRECT)
        self.reqs = self.bytes = 0; self.lk = threading.Lock()

    def read(self, a, b):
        s, e = a // BLK * BLK, -(-b // BLK) * BLK
        buf = mmap.mmap(-1, e - s)
        n = os.preadv(self.fd, [buf], s)
        with self.lk:
            self.reqs += 1; self.bytes += e - s
        return bytes(buf[a - s:min(b, s + n) - s])


class S3:
    """Ranged GETs over plain HTTP (urllib3; boto3 request signing capped the client at ~800 GET/s)."""
    def __init__(self, url):
        import urllib3
        self.url, self.pm = url, urllib3.PoolManager(maxsize=2048, block=False, retries=urllib3.Retry(3))
        self.reqs = self.bytes = 0; self.lk = threading.Lock()

    def read(self, a, b):
        out = []
        while a < b:
            i = a // SLICE; e = min(b, (i + 1) * SLICE)
            r = self.pm.request("GET", f"{self.url}/ctxstore/text-{i}",
                                headers={"Range": f"bytes={a - i * SLICE}-{e - i * SLICE - 1}"})
            assert r.status == 206 and len(r.data) == e - a, (r.status, len(r.data), e - a)
            out.append(r.data)
            with self.lk:
                self.reqs += 1; self.bytes += e - a
            a = e
        return b"".join(out)


class LRU:
    def __init__(self, cap):
        self.cap, self.used, self.d, self.lk = cap, 0, collections.OrderedDict(), threading.Lock()
        self.hit = self.miss = 0

    def get_many(self, keys):
        out = {}
        with self.lk:
            for k in keys:
                v = self.d.get(k)
                if v is not None:
                    self.d.move_to_end(k); out[k] = v; self.hit += 1
                else:
                    self.miss += 1
        return out

    def put(self, k, v):
        with self.lk:
            if k in self.d:
                return
            self.d[k] = v; self.used += len(v)
            while self.used > self.cap:
                self.used -= len(self.d.popitem(last=False)[1])


class Layer:
    def __init__(self, a, store, off):
        self.a, self.st, self.off = a, store, off
        self.cache = LRU(a.budget * int(off[-1]))
        self.pool = ThreadPoolExecutor(1024)
        self.inflight, self.ilk = {}, threading.Lock()
        self.size = int(off[-1])

    # --- unit helpers -------------------------------------------------------------------------
    def blocks(self, pks, g):
        s = set()
        for p in pks:
            s.update(range(int(self.off[p]) // g, (int(self.off[p + 1]) - 1) // g + 1))
        return sorted(s)

    def assemble(self, pks, blk, g):
        out = []
        for p in pks:
            a, b = int(self.off[p]), int(self.off[p + 1])
            out.append(b"".join(blk[i][max(a, i * g) - i * g:min(b, (i + 1) * g) - i * g]
                                for i in range(a // g, (b - 1) // g + 1)))
        return out

    def fetch_ranges(self, ranges, g):
        """ranges = [(first_block, last_block)] -> {block: bytes}, one storage read per range, in parallel."""
        def one(r):
            a, b = r[0] * g, min((r[1] + 1) * g, self.size)
            data = self.st.read(a, b)
            return {i: data[(i - r[0]) * g:(i - r[0] + 1) * g] for i in range(r[0], r[1] + 1)}
        out = {}
        for d in self.pool.map(one, ranges):
            out.update(d)
        return out

    @staticmethod
    def runs(blocks, gap_blocks):
        r = []
        for b in blocks:
            if r and b - r[-1][1] <= gap_blocks + 1:
                r[-1][1] = b
            else:
                r.append([b, b])
        return r

    # --- policies -----------------------------------------------------------------------------
    def get(self, pks):
        p = self.a.policy
        if p == "chunk":
            got = self.cache.get_many(pks)
            miss = [x for x in pks if x not in got]
            for x, v in zip(miss, self.pool.map(lambda x: self.st.read(int(self.off[x]), int(self.off[x + 1])), miss)):
                got[x] = v; self.cache.put(x, v)
            return [got[x] for x in pks]
        g = CELL if p == "cell2m" else BLK
        need = self.blocks(pks, g)
        got = self.cache.get_many(need)
        miss = [b for b in need if b not in got]
        if p == "cell2m":
            new = self.fetch_ranges([(b, b) for b in miss], g)
        elif p == "block":
            new = self.fetch_ranges(self.runs(miss, 0), g)
        else:  # group: coalesce small gaps + share in-flight reads
            mine, wait = [], []
            with self.ilk:
                for b in miss:
                    f = self.inflight.get(b)
                    if f is None:
                        self.inflight[b] = Future(); mine.append(b)
                    else:
                        wait.append((b, f))
            new = self.fetch_ranges(self.runs(mine, self.a.gap // BLK), g) if mine else {}
            with self.ilk:
                for b in mine:
                    self.inflight.pop(b).set_result(new[b])
            for b, f in wait:
                new[b] = f.result()
        for b, v in new.items():
            if b in miss:
                self.cache.put(b, v)
        got.update(new)
        return self.assemble(pks, got, g)


def make_trace(nq, requests, warmup, kind, alpha=1.2, seed=20261103):  # identical to bench.py
    rng = np.random.default_rng(seed)
    total = requests + warmup
    if kind == "natural":
        return np.concatenate([rng.permutation(nq) for _ in range(-(-total // nq))])[:total]
    ranks = np.arange(1, nq + 1, dtype=float) ** -alpha
    return rng.permutation(nq)[rng.choice(nq, size=total, p=ranks / ranks.sum())]


def expand(pks, w, doc, n):
    if w == 0:
        return list(dict.fromkeys(pks))
    out = []
    for p in pks:
        for q in range(max(0, p - w), min(n, p + w + 1)):
            if doc[q] == doc[p]:
                out.append(q)
    return list(dict.fromkeys(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["cell2m", "chunk", "block", "group"], required=True)
    ap.add_argument("--backend", choices=["local", "s3"], required=True)
    ap.add_argument("--budget", type=float, required=True, help="cache bytes / total text bytes")
    ap.add_argument("--trace", choices=["natural", "zipf"], required=True)
    ap.add_argument("--w", type=int, default=0)
    ap.add_argument("--gap", type=int, default=65536)
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--search-from", default="results/trend-nq-mmap/nq-B-b0.05-zipf-c32-notext.jsonl")
    ap.add_argument("--s3", default="http://127.0.0.1:8334")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, default=ROOT / "results/ctx-bench")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    off, doc, topk = np.load(D / "off.npy"), np.load(D / "doc.npy"), np.load(D / "topk.npy")
    n = len(off) - 1
    tr = make_trace(topk.shape[0], a.requests, a.warmup, a.trace)
    reqs = [expand(topk[q].tolist(), a.w, doc, n) for q in tr]
    srch = [json.loads(l)["us"] for l in open(ROOT / a.search_from)]
    store = Local() if a.backend == "local" else S3(a.s3)
    L = Layer(a, store, off)
    case = f"{a.backend}{a.tag}-{a.policy}-b{a.budget}-{a.trace}-w{a.w}"

    def run(lo, hi, record):
        ctr, lk, rows = [lo], threading.Lock(), []
        def worker():
            while True:
                with lk:
                    i = ctr[0]; ctr[0] += 1
                if i >= hi:
                    return
                t0 = time.perf_counter()
                time.sleep(srch[i % len(srch)] / 1e6)
                t1 = time.perf_counter()
                txt = L.get(reqs[i])
                t2 = time.perf_counter()
                if record:
                    rows.append((t2 - t0, t2 - t1, sum(len(x) for x in txt)))
        ts = [threading.Thread(target=worker) for _ in range(a.concurrency)]
        for t in ts: t.start()
        for t in ts: t.join()
        return rows

    run(0, a.warmup, False)
    r0, b0, h0, m0 = store.reqs, store.bytes, L.cache.hit, L.cache.miss
    t0 = time.time()
    rows = run(a.warmup, a.warmup + a.requests, True)
    sec = time.time() - t0
    tot = np.array([r[0] for r in rows]) * 1e3; fet = np.array([r[1] for r in rows]) * 1e3
    ret = sum(r[2] for r in rows)
    s = dict(case=case, backend=a.backend, tag=a.tag, policy=a.policy, budget=a.budget, trace=a.trace, w=a.w, gap=a.gap,
             concurrency=a.concurrency, requests=len(rows), seconds=sec, qps=len(rows) / sec,
             p50_ms=float(np.percentile(tot, 50)), p99_ms=float(np.percentile(tot, 99)),
             fetch_p50_ms=float(np.percentile(fet, 50)), fetch_p99_ms=float(np.percentile(fet, 99)),
             fetch_mean_ms=float(fet.mean()),
             returned_bytes=int(ret), storage_bytes=store.bytes - b0, storage_reqs=store.reqs - r0,
             unit_hit_ratio=(L.cache.hit - h0) / max(1, L.cache.hit - h0 + L.cache.miss - m0),
             cache_cap=L.cache.cap, cache_used=L.cache.used)
    s["amplification"] = s["storage_bytes"] / ret
    s["reqs_per_query"] = s["storage_reqs"] / len(rows)
    (a.out / f"{case}.json").write_text(json.dumps(s, indent=1) + "\n")
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()
                      if k in ("case", "qps", "p50_ms", "p99_ms", "fetch_p99_ms", "amplification", "reqs_per_query")}), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
