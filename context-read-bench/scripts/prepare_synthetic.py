#!/usr/bin/env python3
"""Offline fixture only. Does not model a real corpus or QASPER."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
p.add_argument("--vectors", type=int, default=131072)
p.add_argument("--dimension", type=int, default=64)
p.add_argument("--queries", type=int, default=17408)
p.add_argument("--seed", type=int, default=20261103)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
assert a.vectors > 10 and a.queries > 1024 and a.dimension > 0
rng = np.random.default_rng(a.seed)

def vectors(name, count):
    x = rng.normal(size=(count, a.dimension)).astype("<f4")
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    raw = np.empty((count, a.dimension + 1), dtype="<f4")
    raw[:, 0] = np.array([a.dimension], dtype="<i4").view("<f4")[0]
    raw[:, 1:] = x
    with (a.out / name).open("xb") as f:
        raw.tofile(f)

vectors("base.fvecs", a.vectors)
vectors("queries.fvecs", a.queries)
permutation = rng.permutation(a.vectors)
with (a.out / "chunks.tsv").open("x") as f:
    for slot in permutation:
        f.write(f"{int(slot)*8192+512}\t6144\n")
# Fully write every byte: sparse/unwritten extents would invalidate SSD reads.
with (a.out / "context.bin").open("xb") as f:
    remaining = a.vectors * 8192
    while remaining:
        n = min(1024*1024, remaining)
        f.write(rng.bytes(n))
        remaining -= n
    f.flush()
    os.fsync(f.fileno())
unique = rng.permutation(a.queries)
bank = min(4096, a.queries)
ranks = np.arange(1, bank+1, dtype=float)**-1.2
zipf = rng.choice(bank, size=a.queries, p=ranks/ranks.sum())
for name, trace in [("unique", unique), ("zipf", zipf)]:
    with (a.out / f"{name}.trace").open("x") as f:
        f.writelines(f"{int(i)}\n" for i in trace)
manifest = dict(kind="SYNTHETIC_IMPLEMENTATION_PILOT", seed=a.seed,
                vectors=a.vectors, dimension=a.dimension, queries=a.queries,
                chunk_bytes=6144, allocation_stride=8192, zipf_alpha=1.2,
                zipf_query_bank=bank, same_paper=False, hashes={})
for path in a.out.iterdir():
    if path.is_file():
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(4*1024*1024), b""):
                h.update(block)
        manifest["hashes"][path.name] = h.hexdigest()
(a.out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
print(json.dumps(manifest, indent=2))
