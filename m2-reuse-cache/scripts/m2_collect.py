#!/usr/bin/env python3
"""Run every NQ query once (concurrency 1) against the M2-trace Qdrant and save the client view.
The server-side trace (H/P lines) is grouped per query offline: one query = a block of H lines then its P lines."""
import json, sys, time
from pathlib import Path
import numpy as np
from qdrant_client import QdrantClient, models
ds, out, n = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0
qv = np.load(Path.home() / f"rag-milvus/datasets/emb/{ds}/queries.npy")
c = QdrantClient(url="http://127.0.0.1:6333", prefer_grpc=True, timeout=120)
order = list(range(len(qv)))[: n or None]
t0 = time.time()
with out.open("w") as f:
    for qi in order:
        r = c.query_points(ds, query=qv[qi].tolist(), limit=10, search_params=models.SearchParams(hnsw_ef=64),
                           with_payload=["text"]).points
        f.write(json.dumps({"q": qi, "pks": [int(h.id) for h in r], "tl": [len(h.payload["text"]) for h in r]}) + "\n")
print(f"{len(order)} queries in {time.time()-t0:.0f}s", flush=True)
