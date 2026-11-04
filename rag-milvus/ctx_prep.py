#!/usr/bin/env python3
"""Prepare the external text store and request traces for ctx_bench.py (NQ).

  datasets/ctx/nq/text.bin       text of every chunk (same string Milvus stores), UTF-8, pk order
  datasets/ctx/nq/off.npy        byte offsets (n+1)
  datasets/ctx/nq/doc.npy        article id per chunk (contiguous passages sharing a title)
  datasets/ctx/nq/topk.npy       top-10 pks per query, faiss HNSW M16 efC128 ef64 (same index params as Milvus)
  S3 bucket ctxstore, objects text-<i> of 64 MB slices of text.bin (--s3)
"""
import argparse, os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
SLICE = 64 << 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3", default="http://127.0.0.1:8334")
    a = ap.parse_args()
    out = ROOT / "datasets/ctx/nq"; out.mkdir(parents=True, exist_ok=True)
    if not (out / "off.npy").exists():
        t = pq.read_table(next((ROOT / "datasets/raw/hf/nq").glob("corpus-*.parquet")), columns=["title", "text"]).to_pydict()
        texts = [(ti + " " + x).strip() if ti else x for ti, x in zip(t["title"], t["text"])]
        enc = [s.encode() for s in texts]
        off = np.concatenate([[0], np.cumsum([len(e) for e in enc])]).astype(np.int64)
        with open(out / "text.bin", "wb") as f:
            for e in enc:
                f.write(e)
        d = np.zeros(len(texts), dtype=np.int64)
        for i in range(1, len(texts)):
            d[i] = d[i - 1] + (t["title"][i] != t["title"][i - 1])
        np.save(out / "doc.npy", d); np.save(out / "off.npy", off)
        print("text.bin", off[-1], "bytes", len(texts), "chunks", int(d[-1]) + 1, "articles", flush=True)
    if not (out / "topk.npy").exists():
        import faiss
        faiss.omp_set_num_threads(64)
        idx = faiss.read_index(str(ROOT / "results/ann-vs-context/nq.hnsw"))
        idx.hnsw.efSearch = 64
        Q = np.load(ROOT / "datasets/emb/nq/queries.npy")
        _, I = idx.search(Q, 10)
        np.save(out / "topk.npy", I.astype(np.int64))
        print("topk", I.shape, flush=True)
    import boto3
    from botocore.config import Config
    s3 = boto3.client("s3", endpoint_url=a.s3, aws_access_key_id="x", aws_secret_access_key="x", region_name="us-east-1",
                      config=Config(s3={"addressing_style": "path"}))
    try:
        s3.create_bucket(Bucket="ctxstore")
    except Exception as e:
        print("bucket:", type(e).__name__)
    have = {o["Key"]: o["Size"] for p in s3.get_paginator("list_objects_v2").paginate(Bucket="ctxstore") for o in p.get("Contents", [])}
    size = os.path.getsize(out / "text.bin")
    with open(out / "text.bin", "rb") as f:
        for i in range(-(-size // SLICE)):
            n = min(SLICE, size - i * SLICE)
            if have.get(f"text-{i}") == n:
                continue
            f.seek(i * SLICE)
            s3.put_object(Bucket="ctxstore", Key=f"text-{i}", Body=f.read(n))
            print("put", i, n, flush=True)
    print("s3 ok", -(-size // SLICE), "objects")


if __name__ == "__main__":
    main()
