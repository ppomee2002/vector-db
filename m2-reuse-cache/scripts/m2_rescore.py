#!/usr/bin/env python3
"""Rescoring reads for the 'quantized vectors in RAM, original vectors on disk' setup.
Per query and segment: the top-K (K = limit x oversampling) of the HNSW-visited nodes by exact score are re-read
from the original vectors (Qdrant rescoring). Adds rn/rp (+_off) to <dir>/trace_r.npz and checks that the
returned context chunks are among the rescored nodes (validates the H/P id mapping)."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
D, SEGDIR, K = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 10
T = dict(np.load(D / "trace.npz")); LAY = json.loads((D / "layout.json").read_text())
PG, DIM = 4096, 384
qv = np.load(Path.home() / "rag-milvus/datasets/emb/nq/queries.npy").astype(np.float32)
M = {}
for s, L in LAY["segments"].items():
    raw = subprocess.run(["sudo", "-n", "cat", str(SEGDIR / s / "vector_storage/matrix.dat")], capture_output=True).stdout
    M[L["idx"]] = (np.frombuffer(raw, dtype=np.float32, offset=4, count=L["points"] * DIM).reshape(-1, DIM), L)
rn, rp, ok, tot = [], [], 0, 0
for q in range(len(T["qid"])):
    vn = T["vn"][T["vn_off"][q]:T["vn_off"][q + 1]]
    seg, off = vn >> 22, vn & ((1 << 22) - 1)
    x = qv[T["qid"][q]]; nodes, pages = [], []
    for sidx in np.unique(seg):
        o = off[seg == sidx]; X, L = M[int(sidx)]
        sc = X[o] @ x; top = o[np.argsort(-sc)[:K]]
        nodes += [int(sidx) << 22 | int(i) for i in top]
        for i in top:
            a = 4 + int(i) * DIM * 4
            pages += range(L["vec_base"] + a // PG, L["vec_base"] + (a + DIM * 4 - 1) // PG + 1)
    cn = set(T["cn"][T["cn_off"][q]:T["cn_off"][q + 1]].tolist())
    ok += len(cn & set(nodes)); tot += len(cn)
    rn.append(np.array(nodes, np.int64)); rp.append(np.array(pages, np.int64))
print(f"context chunks among rescored nodes: {ok}/{tot}; rescored vectors/query {np.mean([len(a) for a in rn]):.0f}, pages/query {np.mean([len(a) for a in rp]):.0f}")
for k, a in (("rn", rn), ("rp", rp)):
    T[k] = np.concatenate(a); T[k + "_off"] = np.concatenate([[0], np.cumsum([len(x) for x in a])])
np.savez(D / "trace.npz", **T)
