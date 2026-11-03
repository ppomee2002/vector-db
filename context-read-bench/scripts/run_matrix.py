#!/usr/bin/env python3
"""Python only orchestrates processes; timed retrieval runs entirely in C++."""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--binary", type=Path, default=Path(__file__).resolve().parents[1]/"build/context_bench")
p.add_argument("--data", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--traces", default="unique,zipf")
p.add_argument("--concurrency", default="1,4,16,32,64,128")
p.add_argument("--io", default="direct")
p.add_argument("--repeats", type=int, default=3)
p.add_argument("--warmup", type=int, default=1024)
p.add_argument("--cache-pages", type=int, default=4096)
p.add_argument("--topk", type=int, default=10)
p.add_argument("--ef-search", type=int, default=32)
p.add_argument("--device", default="", help="Confirmed backing device; device counters include all host activity")
p.add_argument("--observe-devices", default="", help="Comma-separated devices to observe without asserting backing-device identity")
p.add_argument("--buffered-state", choices=["warm", "drop-file"], default="warm")
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
cases = [(rep, trace, io, c, mode) for rep in range(a.repeats)
         for trace in a.traces.split(",") for io in a.io.split(",")
         for c in map(int, a.concurrency.split(","))
         for mode in ("none", "shared", "lru", "lru_shared")]
random.Random(20261103).shuffle(cases)
manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
manifest.update(platform=platform.platform(), cpu_count=os.cpu_count(), order=cases,
                load_start=os.getloadavg(), binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest())
for name in ("cpu.max", "memory.max", "io.max"):
    path = Path("/sys/fs/cgroup")/name
    manifest[name] = path.read_text() if path.exists() else None
manifest["input_sha256"] = {}
for name in ["index.faiss", "queries.fvecs", "chunks.tsv"] + [f"{t}.trace" for t in a.traces.split(",")]:
    manifest["input_sha256"][name] = hashlib.sha256((a.data/name).read_bytes()).hexdigest()
(a.out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
rows = []
reference = {}
for rep, trace, io, c, mode in cases:
    case = f"{trace}-{io}-c{c}-{mode}-r{rep}"
    if io == "buffered":
        with (a.data/"context.bin").open("rb") as f:
            if a.buffered_state == "warm":
                for _ in iter(lambda: f.read(4*1024*1024), b""):
                    pass
            else:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    cmd = [str(a.binary), "run", "--index", str(a.data/"index.faiss"),
           "--queries", str(a.data/"queries.fvecs"), "--mapping", str(a.data/"chunks.tsv"),
           "--context", str(a.data/"context.bin"), "--trace", str(a.data/f"{trace}.trace"),
           "--mode", mode, "--io", io, "--concurrency", str(c),
           "--warmup", str(a.warmup), "--cache-pages", str(a.cache_pages),
           "--topk", str(a.topk), "--ef-search", str(a.ef_search),
           "--latencies", str(a.out/f"{case}.latencies.csv")]
    if a.device:
        cmd += ["--device", a.device]
    if a.observe_devices:
        cmd += ["--observe-devices", a.observe_devices]
    print(case, flush=True)
    with (a.out/f"{case}.stderr").open("w") as err:
        result = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=err, check=True)
    row = json.loads(result.stdout)
    assert row["logical_pages"] == row["pread_calls"]+row["cache_hits"]+row["joined_reads"]
    assert mode in ("shared", "lru_shared") or row["joined_reads"] == 0
    assert mode in ("lru", "lru_shared") or row["cache_hits"] == 0
    # Verify reconstructed bytes per trace position, independent of completion order.
    with (a.out/f"{case}.latencies.csv").open() as f:
        content = [(r["query_id"], r["digest"]) for r in csv.DictReader(f)]
    if trace not in reference:
        reference[trace] = content
    assert content == reference[trace], f"ANN/reconstruction mismatch in {case}"
    row.update(repeat=rep, trace=trace, cache_pages=a.cache_pages,
               buffered_state=a.buffered_state if io=="buffered" else "not_applicable")
    (a.out/f"{case}.json").write_text(json.dumps(row, indent=2)+"\n")
    rows.append(row)
    with (a.out/"summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
(a.out/"COMPLETE").write_text(f"{len(rows)} cases; all per-query digests match\n")
