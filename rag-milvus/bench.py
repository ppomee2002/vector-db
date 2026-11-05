#!/usr/bin/env python3
"""Closed-loop RAG retrieval load against Milvus with server-side counter deltas.

Each request: search(top-k) with or without output_fields=["text"].
Writes <out>/<case>.jsonl (per request) and <out>/<case>.json (summary + deltas).
"""
import argparse, json, os, re, subprocess, time, urllib.request
from multiprocessing import Process, Value, Queue
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def metrics():
    """Sum Prometheus samples by metric name (labels collapsed)."""
    out = {}
    for line in urllib.request.urlopen("http://127.0.0.1:9091/metrics", timeout=10).read().decode().splitlines():
        if not line or line[0] == "#":
            continue
        m = re.match(r"([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)", line)
        if m and (m.group(1).startswith(("internal_cache", "internal_mmap", "internal_storage", "milvus_querynode_sq",
                                         "milvus_proxy_sq", "process_"))):
            try:
                v = float(m.group(3))
            except ValueError:
                continue
            out[m.group(1)] = out.get(m.group(1), 0.0) + v
            dt = re.search(r'data_type="([a-z_]+)"', m.group(2) or "")
            loc = re.search(r'location="([a-z]+)"', m.group(2) or "")
            if dt and loc and loc.group(1) == "memory":  # keep per-type split (e.g. scalar_field vs vector_index)
                k = f"{m.group(1)}:{dt.group(1)}"
                out[k] = out.get(k, 0.0) + v
    return out


def milvus_pid():
    for line in subprocess.run(["sudo", "-n", "k3s", "ctr", "task", "ls"], capture_output=True, text=True).stdout.splitlines():
        f = line.split()
        if f and f[0] == "rm-milvus":
            return int(f[1])
    raise SystemExit("rm-milvus not running")


def proc_counters(pid):
    st = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    cg = Path(f"/proc/{pid}/cgroup").read_text().strip().split("::", 1)[1]
    base = Path("/sys/fs/cgroup" + cg)
    io = {}
    for line in (base / "io.stat").read_text().splitlines():
        for kv in line.split()[1:]:
            k, v = kv.split("=")
            io[k] = io.get(k, 0) + int(v)
    mem = dict(l.split() for l in (base / "memory.stat").read_text().splitlines())
    return dict(majflt=int(st[9]), minflt=int(st[7]), io_rbytes=io.get("rbytes", 0), io_rios=io.get("rios", 0),
                mem_file=int(mem.get("file", 0)), mem_anon=int(mem.get("anon", 0)),
                mem_current=int((base / "memory.current").read_text()), cgroup=str(base))


def worker(wid, a, trace, counter, q):
    from pymilvus import MilvusClient
    c = MilvusClient(a.uri, timeout=120)
    qv = np.load(ROOT / f"datasets/emb/{a.dataset}/queries.npy", mmap_mode="r")
    params = {"metric_type": "IP", "params": {"ef": a.ef}}
    of = ["text"] if a.text else []
    rows = []
    try:
        loop(c, a, trace, counter, qv, params, of, rows)
    except BaseException as e:  # never leave the parent blocked on q.get()
        q.put(("error", repr(e)))
        raise
    q.put(rows)


def loop(c, a, trace, counter, qv, params, of, rows):
    while True:
        with counter.get_lock():
            i = counter.value
            counter.value += 1
        if i >= len(trace):
            break
        qi = int(trace[i])
        t0 = time.perf_counter()
        r = c.search(a.dataset, data=[qv[qi].tolist()], limit=a.topk, search_params=params, output_fields=of,
                     anns_field="vec")
        dt = (time.perf_counter() - t0) * 1e6
        rows.append((i, qi, dt, [h["pk"] for h in r[0]], sum(len(h["entity"].get("text", "")) for h in r[0])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--uri", default="http://127.0.0.1:19530")
    p.add_argument("--trace", choices=["natural", "zipf"], default="natural")
    p.add_argument("--alpha", type=float, default=1.2)
    p.add_argument("--requests", type=int, default=0, help="0 = number of queries")
    p.add_argument("--warmup", type=int, default=0, help="requests run before measurement (same trace prefix)")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--ef", type=int, default=64)
    p.add_argument("--text", type=int, default=1, help="1 = output_fields=[text]")
    p.add_argument("--seed", type=int, default=20261103)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    nq = np.load(ROOT / f"datasets/emb/{a.dataset}/queries.npy", mmap_mode="r").shape[0]
    rng = np.random.default_rng(a.seed)
    total = (a.requests or nq) + a.warmup
    if a.trace == "natural":
        trace = np.concatenate([rng.permutation(nq) for _ in range(-(-total // nq))])[:total]
    else:
        ranks = np.arange(1, nq + 1, dtype=float) ** -a.alpha
        trace = rng.permutation(nq)[rng.choice(nq, size=total, p=ranks / ranks.sum())]

    def run(tr, conc):
        counter, q = Value("q", 0), Queue()
        ps = [Process(target=worker, args=(w, a, tr, counter, q)) for w in range(conc)]
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
    pid = milvus_pid()
    m0, p0, t0 = metrics(), proc_counters(pid), time.time()
    rows = run(trace[a.warmup:], a.concurrency)
    sec = time.time() - t0
    m1, p1 = metrics(), proc_counters(pid)
    rows.sort()
    with (a.out / f"{a.case}.jsonl").open("w") as f:
        for i, qi, dt, pks, tb in rows:
            f.write(json.dumps({"i": i, "q": qi, "us": round(dt, 1), "pks": pks, "text_bytes": tb}) + "\n")
    lat = np.array([r[2] for r in rows])
    dm = {k: m1.get(k, 0) - m0.get(k, 0) for k in sorted(set(m0) | set(m1))}
    dp = {k: p1[k] - p0[k] for k in p0 if k not in ("cgroup", "mem_file", "mem_anon", "mem_current")}
    hit, miss = dm.get("internal_cache_cell_access_hit_bytes_total", 0), dm.get("internal_cache_cell_access_miss_bytes_total", 0)
    summ = dict(case=a.case, dataset=a.dataset, trace=a.trace, alpha=a.alpha if a.trace == "zipf" else None,
                requests=len(rows), concurrency=a.concurrency, topk=a.topk, ef=a.ef, text=a.text, seconds=sec,
                qps=len(rows) / sec, p50_ms=float(np.percentile(lat, 50)) / 1e3, p95_ms=float(np.percentile(lat, 95)) / 1e3,
                p99_ms=float(np.percentile(lat, 99)) / 1e3, returned_text_bytes=int(sum(r[4] for r in rows)),
                cache_hit_bytes=hit, cache_miss_bytes=miss,
                cache_byte_hit_ratio=(hit / (hit + miss)) if hit + miss else None,
                proc_delta=dp, proc_after={k: p1[k] for k in ("mem_file", "mem_anon", "mem_current", "cgroup")},
                metrics_delta={k: v for k, v in dm.items() if v}, metrics_after={k: v for k, v in m1.items() if k.startswith("internal_cache")})
    (a.out / f"{a.case}.json").write_text(json.dumps(summ, indent=2) + "\n")
    print(json.dumps({k: summ[k] for k in ("case", "qps", "p50_ms", "p95_ms", "p99_ms", "cache_byte_hit_ratio")}
                     | {"io_rbytes": dp.get("io_rbytes"), "majflt": dp.get("majflt")}), flush=True)


if __name__ == "__main__":
    main()
