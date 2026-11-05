#!/usr/bin/env python3
"""Create and load a BEIR collection into Milvus: pk, doc_id, text, vec (HNSW, IP).

Row order = corpus parquet order = datasets/emb/<ds>/corpus.npy order, so pk == row.
"""
import argparse, json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from pymilvus import MilvusClient, DataType

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--uri", default="http://127.0.0.1:19530")
p.add_argument("--batch", type=int, default=5000)
p.add_argument("--threads", type=int, default=8)
p.add_argument("--drop", action="store_true")
a = p.parse_args()

c = MilvusClient(a.uri, timeout=600)
name = a.dataset
if c.has_collection(name):
    if not a.drop:
        raise SystemExit(f"{name} exists; pass --drop to recreate")
    c.drop_collection(name)

vec = np.load(ROOT / f"datasets/emb/{name}/corpus.npy", mmap_mode="r")
t = pq.read_table(next((ROOT / f"datasets/raw/hf/{name}").glob("corpus-*.parquet")), columns=["_id", "title", "text"]).to_pydict()
texts = [(ti + " " + x).strip() if ti else x for ti, x in zip(t["title"], t["text"])]
ids = list(map(str, t["_id"]))
assert len(texts) == vec.shape[0], (len(texts), vec.shape)
n = len(texts)
maxlen = max(len(s.encode()) for s in texts)

schema = c.create_schema(auto_id=False)
schema.add_field("pk", DataType.INT64, is_primary=True)
schema.add_field("doc_id", DataType.VARCHAR, max_length=256)
schema.add_field("text", DataType.VARCHAR, max_length=min(65535, max(maxlen + 16, 256)))
schema.add_field("vec", DataType.FLOAT_VECTOR, dim=vec.shape[1])
c.create_collection(name, schema=schema, shards_num=1)

t0 = time.time()
def put(s):
    e = min(s + a.batch, n)
    c.insert(name, [{"pk": i, "doc_id": ids[i], "text": texts[i], "vec": vec[i].tolist()} for i in range(s, e)])
    return e - s
done = 0
with ThreadPoolExecutor(a.threads) as ex:
    for k in ex.map(put, range(0, n, a.batch)):
        done += k
        if done % (a.batch * 50) < a.batch or done == n:
            print(f"insert {done}/{n} {done/(time.time()-t0):.0f} rows/s", flush=True)
c.flush(name)
t_ins = time.time() - t0

ip = c.prepare_index_params()
ip.add_index("vec", index_type="HNSW", metric_type="IP", params={"M": 16, "efConstruction": 128})
t1 = time.time()
c.create_index(name, ip, sync=True)
t_idx = time.time() - t1
t2 = time.time()
c.load_collection(name)
t_load = time.time() - t2
text_bytes = sum(len(s.encode()) for s in texts)
meta = dict(dataset=name, rows=n, text_bytes=text_bytes, vec_bytes=int(vec.nbytes),
            insert_s=t_ins, index_s=t_idx, load_s=t_load, index="HNSW M16 efC128 IP")
(ROOT / f"datasets/emb/{name}/milvus_load.json").write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps(meta))
