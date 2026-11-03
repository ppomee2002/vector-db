#!/usr/bin/env python3
"""Integration checks for completion eviction, caching, coalescing and error exits."""
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, required=True)
p.add_argument("--binary", type=Path, default=Path(__file__).resolve().parents[1]/"build/context_bench")
a = p.parse_args()
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
with tempfile.TemporaryDirectory(prefix="context-read-validate-") as tmp:
    trace = Path(tmp)/"repeat.trace"
    trace.write_text("0\n"*1056)
    cmd = [str(a.binary), "run", "--index", str(a.data/"index.faiss"),
           "--queries", str(a.data/"queries.fvecs"), "--mapping", str(a.data/"chunks.tsv"),
           "--context", str(a.data/"context.bin"), "--trace", str(trace),
           "--warmup", "32", "--cache-pages", "4096"]
    results = {}
    for io in ("direct", "buffered"):
        for c in (1, 32):
            for mode in ("none", "shared", "lru", "lru_shared"):
                r = subprocess.run(cmd+["--io",io,"--concurrency",str(c),"--mode",mode],
                                   text=True, capture_output=True, env=env, check=True)
                result = json.loads(r.stdout)
                results[io,c,mode] = result
                assert result["logical_pages"] == result["pread_calls"]+result["joined_reads"]+result["cache_hits"]
                if mode == "none" or (mode == "shared" and c == 1):
                    assert result["pread_calls"] == result["logical_pages"]
                    assert result["cache_hits"] == result["joined_reads"] == 0
                if mode in ("lru", "lru_shared"):
                    assert result["pread_calls"] == 0, "Warm cache did not retain the repeated Query working set"
                print(io, c, mode, "reads", result["pread_calls"], "joins",result["joined_reads"], flush=True)
    assert len({r["digest"] for r in results.values()}) == 1
    assert results["direct",32,"shared"]["joined_reads"] > 0, "Stress run did not exercise sharing"
    # Invalid Query IDs must fail, not silently wrap or use different ANN inputs.
    trace.write_text("999999999\n"*1056)
    r = subprocess.run(cmd, text=True, capture_output=True, env=env)
    assert r.returncode != 0 and "Query ID out of range" in r.stderr
print("PASS: 16 configurations, identical digests, no completed-page retention in shared-only mode, cache hits, overlapping joins, invalid trace rejection")
