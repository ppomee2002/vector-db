#!/usr/bin/env python3
"""Measure FAISS single-thread search cost, to calibrate m3bench's search CPU cost.

usage:
  python calibrate_faiss.py --data synth   --cpus 0-6 --out calib_faiss_synth.csv
  python calibrate_faiss.py --data nytimes --cpus 0-6 --out calib_faiss_real.csv --cache ~/m3data/ann

--data synth   : the same clustered synthetic data m3bench uses (500k x 384). Easy for ANN
                 (well separated clusters) -> a LOWER bound on realistic search cost.
--data nytimes : a real embedding benchmark (ann-benchmarks NYTimes-256-angular, 290k x 256,
                 known to be hard). Its time is scaled to 500k x 384:
                 x (384/256) for the dimension and x ln(500k)/ln(290k) for HNSW's log-n hops.

Index: HNSW-Flat (M=32, efConstruction=64) [+ IVF-Flat for synth]. Build is multi-threaded
on the given CPUs; queries run one at a time on ONE CPU. Reported per efSearch/nprobe:
ms/query and recall@10. The target is the smallest setting reaching --recall.
"""
import argparse
import math
import os
import sys
import time
import urllib.request

import numpy as np

ANN_URL = "https://ann-benchmarks.com/nytimes-256-angular.hdf5"


def cpulist(s):
    out = []
    for p in s.split(","):
        if "-" in p:
            a, b = p.split("-")
            out += range(int(a), int(b) + 1)
        elif p.strip():
            out.append(int(p))
    return out


def synth(a, rng):
    d = a.dim
    cent = rng.uniform(-1, 1, (a.clusters, d)).astype("float32")
    assign = rng.integers(0, a.clusters, a.nvec)
    X = np.empty((a.nvec, d), dtype="float32")
    for s in range(0, a.nvec, 50000):
        e = min(s + 50000, a.nvec)
        X[s:e] = cent[assign[s:e]] + 0.35 * rng.uniform(-1, 1, (e - s, d)).astype("float32")
    qc = rng.integers(0, a.clusters, a.queries)
    Q = (cent[qc] + 0.35 * rng.uniform(-1, 1, (a.queries, d))).astype("float32")
    return X, Q, None


HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
UA = {"User-Agent": "Mozilla/5.0 (m3bench calibration)"}  # the site answers 403 to Python's default User-Agent


def remote_size(url):
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers.get("Content-Length") or 0)


def valid_hdf5(path, size):
    if not os.path.exists(path) or (size and os.path.getsize(path) != size):
        return False
    with open(path, "rb") as f:
        return f.read(8) == HDF5_MAGIC


def download(url, path, tries=30):
    """Download with resume (HTTP Range) until the size matches; verify the HDF5 signature.
    A partial or broken file is always removed, never left behind."""
    size = remote_size(url)
    if valid_hdf5(path, size):
        return
    if os.path.exists(path):
        os.remove(path)  # our own broken file from an earlier attempt
    tmp = path + ".part"
    print(f"downloading {url} ({size / 1e6:.0f} MB) -> {path}", flush=True)
    try:
        done, last = 0, -1
        for attempt in range(tries):
            hdr = dict(UA)
            if done:
                hdr["Range"] = f"bytes={done}-"
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=60) as r:
                    if done and r.status != 206:  # server ignored Range: start over
                        done = 0
                    with open(tmp, "ab" if done else "wb") as out:
                        while True:
                            buf = r.read(1 << 20)
                            if not buf:
                                break
                            out.write(buf)
                            done += len(buf)
                            if done // (50 << 20) != last:
                                last = done // (50 << 20)
                                print(f"  {done / 1e6:.0f} / {size / 1e6:.0f} MB", flush=True)
            except Exception as e:  # connection reset etc. -> resume
                print(f"  interrupted at {done / 1e6:.1f} MB ({type(e).__name__}); resuming", flush=True)
            if size and done >= size:
                break
            time.sleep(min(2 * (attempt + 1), 10))
        if not valid_hdf5(tmp, size):
            with open(tmp, "rb") as f:
                head = f.read(200)
            raise RuntimeError(f"download incomplete or not HDF5 ({os.path.getsize(tmp)} of {size} bytes); "
                               f"starts with {head[:120]!r}")
        os.rename(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def nytimes(a):
    import h5py
    os.makedirs(a.cache, exist_ok=True)
    path = os.path.join(a.cache, "nytimes-256-angular.hdf5")
    download(ANN_URL, path)
    with h5py.File(path, "r") as f:
        X = np.asarray(f["train"], dtype="float32")
        Q = np.asarray(f["test"][: a.queries], dtype="float32")
        gt = np.asarray(f["neighbors"][: a.queries, : a.k])
    # angular distance == L2 on unit vectors (same ranking)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12
    return X, Q, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synth", "nytimes"], default="synth")
    ap.add_argument("--cpus", default="")
    ap.add_argument("--out", default="calib_faiss.csv")
    ap.add_argument("--cache", default=os.path.expanduser("~/m3data/ann"))
    ap.add_argument("--nvec", type=int, default=500000)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--clusters", type=int, default=2048)
    ap.add_argument("--nlist", type=int, default=2048)
    ap.add_argument("--hnsw-m", type=int, default=32)
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--recall", type=float, default=0.95)
    a = ap.parse_args()
    cpus = cpulist(a.cpus) if a.cpus else sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, cpus)
    import faiss  # after pinning, so OpenMP threads inherit the mask

    rng = np.random.default_rng(1)
    t0 = time.time()
    X, Q, gt = synth(a, rng) if a.data == "synth" else nytimes(a)
    n, d = X.shape
    scale = 1.0 if a.data == "synth" else (a.dim / d) * (math.log(a.nvec) / math.log(n))
    print(f"[{a.data}] data {n}x{d}, {len(Q)} queries, loaded in {time.time() - t0:.1f}s; time scale to "
          f"{a.nvec}x{a.dim}: x{scale:.3f}", flush=True)

    faiss.omp_set_num_threads(len(cpus))
    bad = ~np.isfinite(X).all(axis=1) | (np.abs(X).sum(axis=1) == 0)
    if bad.any():  # NaN / all-zero rows would make distances meaningless
        print(f"dropping {int(bad.sum())} invalid vectors (NaN or all zeros)", flush=True)
        X = np.ascontiguousarray(X[~bad])
        n = len(X)
    X = np.ascontiguousarray(X, dtype="float32")
    Q = np.ascontiguousarray(Q, dtype="float32")
    # ground truth = exact search with the same metric we use for the index (never trust a file's list)
    flat = faiss.IndexFlatL2(d)
    flat.add(X)
    _, exact = flat.search(Q, a.k)
    del flat
    if gt is not None:
        agree = float(np.mean([len(set(exact[i]) & set(int(v) for v in gt[i][: a.k])) / a.k for i in range(len(Q))]))
        print(f"file ground truth vs exact search on these vectors: {agree:.3f} overlap (using exact search)", flush=True)
    gt = exact
    rows = []

    def timed(index, label, knob, value):
        faiss.omp_set_num_threads(1)
        for i in range(min(20, len(Q))):
            index.search(Q[i:i + 1], a.k)
        ids = np.empty((len(Q), a.k), dtype="int64")
        t = time.perf_counter()
        for i in range(len(Q)):
            _, I = index.search(Q[i:i + 1], a.k)
            ids[i] = I[0]
        ms = (time.perf_counter() - t) / len(Q) * 1000
        rec = float(np.mean([len(set(ids[i]) & set(gt[i][: a.k])) / a.k for i in range(len(Q))]))
        rows.append((label, knob, value, ms, rec, ms * scale))
        print(f"{label:10s} {knob}={value:<4d} {ms:8.4f} ms/query (scaled {ms * scale:7.4f})  recall@{a.k}={rec:.3f}", flush=True)

    if a.data == "synth":
        faiss.omp_set_num_threads(len(cpus))
        t0 = time.time()
        quant = faiss.IndexFlatL2(d)
        ivf = faiss.IndexIVFFlat(quant, d, a.nlist)
        ivf.train(X[rng.choice(n, min(n, 40 * a.nlist), replace=False)])
        ivf.add(X)
        print(f"IVF-Flat nlist={a.nlist} built in {time.time() - t0:.1f}s", flush=True)
        for npb in [1, 2, 4, 8, 16, 32, 64]:
            ivf.nprobe = npb
            timed(ivf, "ivf_flat", "nprobe", npb)
        del ivf, quant

    faiss.omp_set_num_threads(len(cpus))
    t0 = time.time()
    hnsw = faiss.IndexHNSWFlat(d, a.hnsw_m)
    hnsw.hnsw.efConstruction = 64
    hnsw.add(X)
    print(f"HNSW M={a.hnsw_m} built in {time.time() - t0:.1f}s", flush=True)
    for ef in [16, 32, 64, 128, 256, 512, 1024, 2048]:
        hnsw.hnsw.efSearch = ef
        timed(hnsw, "hnsw_flat", "efSearch", ef)
        if rows[-1][4] >= 0.995:
            break

    with open(a.out, "w") as f:
        f.write("data,index,knob,value,ms_per_query,recall,ms_scaled\n")
        for r in rows:
            f.write(f"{a.data},{r[0]},{r[1]},{r[2]},{r[3]:.5f},{r[4]:.4f},{r[5]:.5f}\n")
    ok = sorted((r for r in rows if r[0] == "hnsw_flat" and r[4] >= a.recall), key=lambda r: r[2])
    r = ok[0] if ok else max((r for r in rows if r[0] == "hnsw_flat"), key=lambda r: r[4])
    print(f"TARGET {a.data} hnsw efSearch={r[2]} {r[5]:.4f} ms (scaled) recall={r[4]:.3f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
