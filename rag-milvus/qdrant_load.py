#!/usr/bin/env python3
"""Load a BEIR dataset into Qdrant: vectors + HNSW in RAM, payload (doc_id, text) on disk.

Point id == corpus row == datasets/emb/<ds>/corpus.npy row (same as the Milvus pk).
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--url", default="http://127.0.0.1:6333")
    p.add_argument("--batch", type=int, default=2000)
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--drop", action="store_true")
    a = p.parse_args()

    c = QdrantClient(url=a.url, prefer_grpc=True, timeout=600)
    name = a.dataset
    if c.collection_exists(name):
        if not a.drop:
            raise SystemExit(f"{name} exists; pass --drop")
        c.delete_collection(name)

    vec = np.load(ROOT / f"datasets/emb/{name}/corpus.npy", mmap_mode="r")
    t = pq.read_table(next((ROOT / f"datasets/raw/hf/{name}").glob("corpus-*.parquet")), columns=["_id", "title", "text"]).to_pydict()
    texts = [(ti + " " + x).strip() if ti else x for ti, x in zip(t["title"], t["text"])]
    assert len(texts) == vec.shape[0]
    n = len(texts)

    c.create_collection(
        name,
        vectors_config=models.VectorParams(size=vec.shape[1], distance=models.Distance.DOT, on_disk=False),
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128, on_disk=False),
        on_disk_payload=True,
        shard_number=1,
    )
    t0 = time.time()
    c.upload_collection(name, vectors=vec, payload=({"doc_id": str(i), "text": s} for i, s in zip(t["_id"], texts)),
                        ids=range(n), batch_size=a.batch, parallel=a.parallel, wait=True)
    t_ins = time.time() - t0
    t1 = time.time()
    while True:  # wait for the optimizer to finish building HNSW
        info = c.get_collection(name)
        if info.status == models.CollectionStatus.GREEN and (info.indexed_vectors_count or 0) >= n * 0.99:
            break
        time.sleep(5)
    meta = dict(dataset=name, rows=n, text_bytes=sum(len(s.encode()) for s in texts), insert_s=t_ins,
                index_s=time.time() - t1, index="HNSW m16 ef_construct128 DOT", on_disk_payload=True,
                points=info.points_count, segments=info.segments_count)
    (ROOT / f"datasets/emb/{name}/qdrant_load.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":  # qdrant-client parallel upload spawns workers that re-import this module
    main()
