#!/usr/bin/env python3
"""Closed-loop RAG retrieval load against Qdrant; same traces/outputs as bench.py (Milvus).

Storage reads = container cgroup io.stat rbytes (payload served from disk via page cache).
"""
import argparse, json, subprocess, time
from multiprocessing import Process, Value, Queue
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def task_pid(name="rm-qgran"):
    for line in subprocess.run(["sudo", "-n", "k3s", "ctr", "task", "ls"], capture_output=True, text=True).stdout.splitlines():
        f = line.split()
        if f and f[0] == name:
            return int(f[1])
    raise SystemExit(f"{name} not running")


def proc_counters(pid):
    st = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    base = Path("/sys/fs/cgroup" + Path(f"/proc/{pid}/cgroup").read_text().strip().split("::", 1)[1])
    io = {}
    for line in (base / "io.stat").read_text().splitlines():
        for kv in line.split()[1:]:
            k, v = kv.split("=")
            io[k] = io.get(k, 0) + int(v)
    mem = dict(l.split() for l in (base / "memory.stat").read_text().splitlines())
    return dict(majflt=int(st[9]), minflt=int(st[7]), io_rbytes=io.get("rbytes", 0), io_rios=io.get("rios", 0),
                mem_file=int(mem.get("file", 0)), mem_anon=int(mem.get("anon", 0)),
                mem_current=int((base / "memory.current").read_text()), cgroup=str(base))


def worker(a, trace, counter, q):
    from qdrant_client import QdrantClient, models
    c = QdrantClient(url=a.url, prefer_grpc=True, timeout=120)
    qv = np.load(ROOT / f"datasets/emb/{a.dataset}/queries.npy", mmap_mode="r")
    sp = models.SearchParams(hnsw_ef=a.ef)
    payload = ["text"] if a.text else False
    rows = []
    try:
        while True:
            with counter.get_lock():
                i = counter.value
                counter.value += 1
            if i >= len(trace):
                break
            qi = int(trace[i])
            t0 = time.perf_counter()
            r = c.query_points(a.dataset, query=qv[qi].tolist(), limit=a.topk, search_params=sp, with_payload=payload).points
            dt = (time.perf_counter() - t0) * 1e6
            rows.append((i, qi, dt, [int(h.id) for h in r], sum(len((h.payload or {}).get("text", "")) for h in r)))
    except BaseException as e:
        q.put(("error", repr(e)))
        raise
    q.put(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--url", default="http://127.0.0.1:6333")
    p.add_argument("--trace", choices=["natural", "zipf"], default="natural")
    p.add_argument("--alpha", type=float, default=1.2)
    p.add_argument("--requests", type=int, default=0)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--ef", type=int, default=64)
    p.add_argument("--text", type=int, default=1)
    p.add_argument("--seed", type=int, default=20261103)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    nq = np.load(ROOT / f"datasets/emb/{a.dataset}/queries.npy", mmap_mode="r").shape[0]
    rng = np.random.default_rng(a.seed)  # same seed/procedure as bench.py → identical traces
    total = (a.requests or nq) + a.warmup
    if a.trace == "natural":
        trace = np.concatenate([rng.permutation(nq) for _ in range(-(-total // nq))])[:total]
    else:
        ranks = np.arange(1, nq + 1, dtype=float) ** -a.alpha
        trace = rng.permutation(nq)[rng.choice(nq, size=total, p=ranks / ranks.sum())]

    def run(tr, conc):
        counter, q = Value("q", 0), Queue()
        ps = [Process(target=worker, args=(a, tr, counter, q)) for _ in range(conc)]
        for x in ps: x.start()
        rows = []
        for _ in ps:
            got = q.get()
            if isinstance(got, tuple) and got and got[0] == "error":
                for x in ps: x.terminate()
                raise SystemExit("worker failed: " + got[1])
            rows += got
        for x in ps: x.join()
        return rows

    if a.warmup:
        run(trace[:a.warmup], min(a.concurrency, 16))
    pid = task_pid()
    p0, t0 = proc_counters(pid), time.time()
    rows = run(trace[a.warmup:], a.concurrency)
    sec = time.time() - t0
    p1 = proc_counters(pid)
    rows.sort()
    with (a.out / f"{a.case}.jsonl").open("w") as f:
        for i, qi, dt, pks, tb in rows:
            f.write(json.dumps({"i": i, "q": qi, "us": round(dt, 1), "pks": pks, "text_bytes": tb}) + "\n")
    lat = np.array([r[2] for r in rows])
    dp = {k: p1[k] - p0[k] for k in ("majflt", "minflt", "io_rbytes", "io_rios")}
    summ = dict(engine="qdrant", case=a.case, dataset=a.dataset, trace=a.trace, alpha=a.alpha if a.trace == "zipf" else None,
                requests=len(rows), concurrency=a.concurrency, topk=a.topk, ef=a.ef, text=a.text, seconds=sec,
                qps=len(rows) / sec, p50_ms=float(np.percentile(lat, 50)) / 1e3, p95_ms=float(np.percentile(lat, 95)) / 1e3,
                p99_ms=float(np.percentile(lat, 99)) / 1e3, returned_text_bytes=int(sum(r[4] for r in rows)),
                storage_read_bytes=dp["io_rbytes"], proc_delta=dp,
                proc_after={k: p1[k] for k in ("mem_file", "mem_anon", "mem_current", "cgroup")})
    (a.out / f"{a.case}.json").write_text(json.dumps(summ, indent=2) + "\n")
    print(json.dumps({k: summ[k] for k in ("case", "qps", "p50_ms", "p95_ms", "p99_ms", "storage_read_bytes")}
                     | {"majflt": dp["majflt"], "mem_file": p1["mem_file"]}), flush=True)


if __name__ == "__main__":
    main()
