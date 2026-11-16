"""12주차 Milvus: 배치 순서대로 PK를 매겨 적재하고, 지난주와 같은 방식으로 측정. jihyeon database의 w12m_ 컬렉션만 다룸."""
import argparse, csv, json, os, sys, time, numpy as np
sys.path.insert(0, os.path.expanduser("~/m4-exp/scripts"))
from pymilvus import MilvusClient, DataType, connections, utility
from common import load_data, expand
from milvus_common import URI, metrics, delta, pick
W = "/workspace/jihyeon/w12q"; DB = "jihyeon"; P = "w12m_"; R = f"{W}/results/milvus"
ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["load", "measure", "drop"]); ap.add_argument("--layout")
ap.add_argument("--qs", default="unseen"); ap.add_argument("--window", type=int, default=0); ap.add_argument("--nq", type=int, default=100)
a = ap.parse_args()
root = MilvusClient(URI, timeout=1800)
if DB not in root.list_databases(): root.create_database(DB); print("created database", DB)
c = MilvusClient(URI, db_name=DB, timeout=1800)
if a.cmd == "drop":
    for n in c.list_collections():
        if n.startswith(P): c.drop_collection(n); print("dropped", DB, n)
    sys.exit()
chunks, topk, train, test, bounds = load_data(f"{W}/data_{a.qs}"); N = len(chunks); name = P + a.layout
if a.cmd == "load":
    emb = np.load(f"{W}/data_unseen/emb.npy"); order = np.load(f"{W}/orders/order_w12_{a.layout}.npy"); assert len(order) == N
    pk_of = np.empty(N, np.int64); pk_of[order] = np.arange(N); np.save(f"{W}/orders/mpk_{a.layout}.npy", pk_of)
    if c.has_collection(name): c.drop_collection(name)
    s = c.create_schema(auto_id=False)
    s.add_field("pk", DataType.INT64, is_primary=True); s.add_field("vec", DataType.FLOAT_VECTOR, dim=emb.shape[1])
    s.add_field("text", DataType.VARCHAR, max_length=65535); s.add_field("title", DataType.VARCHAR, max_length=1024)
    s.add_field("doc", DataType.INT64); s.add_field("pos", DataType.INT64)
    ip = c.prepare_index_params(); ip.add_index("vec", index_type="FLAT", metric_type="COSINE")
    c.create_collection(name, schema=s, index_params=ip, properties={"warmup.scalarField": "disable"})
    t0 = time.time()
    for i in range(0, N, 5000):
        ids = order[i:i + 5000]
        c.insert(name, [{"pk": int(pk_of[j]), "vec": emb[j].tolist(), "text": chunks[j]["text"], "title": chunks[j]["title"][:1000],
                         "doc": chunks[j]["doc"], "pos": chunks[j]["pos"]} for j in ids])
    c.flush(name); connections.connect(uri=URI, db_name=DB); utility.wait_for_index_building_complete(name)
    c.release_collection(name)  # 측정 전까지 메모리에서 내림
    print(f"[mload] {DB}.{name}: rows={N} time={time.time()-t0:.0f}s", flush=True); sys.exit()
pk_of = np.load(f"{W}/orders/mpk_{a.layout}.npy"); nbytes = np.array([len(x["text"].encode()) for x in chunks])
def reload(): c.release_collection(name); c.load_collection(name); time.sleep(1)
reload(); rows = []; T0 = time.time()
for q in test[: a.nq]:
    ids = expand(topk[q], bounds, a.window); reload()
    m0 = metrics(); t = time.perf_counter()
    r = c.get(name, ids=[int(pk_of[i]) for i in ids], output_fields=["text"])
    lat = time.perf_counter() - t; time.sleep(0.3); d = delta(m0, metrics()); assert len(r) == len(ids)
    rows.append(dict(qs=a.qs, layout=a.layout, window=a.window, qid=int(q), n_ctx=len(ids), need_bytes=int(nbytes[ids].sum()), lat_ms=lat * 1e3,
                     miss_bytes_all=pick(d, "internal_cache_cell_access_miss_bytes_total"), cell_loads=pick(d, "internal_cache_load_latency_microseconds_count")))
c.release_collection(name)
with open(f"{R}/mq_{a.qs}_{a.layout}_w{a.window}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
s = {k: float(np.mean([x[k] for x in rows])) for k in ("n_ctx", "need_bytes", "lat_ms", "miss_bytes_all", "cell_loads")}
s["p50_ms"] = float(np.median([x["lat_ms"] for x in rows])); s.update(qs=a.qs, layout=a.layout, window=a.window, nq=len(rows))
open(f"{R}/summary.jsonl", "a").write(json.dumps(s) + "\n")
print(f"[mmeasure] {a.qs} {a.layout:5s} w={a.window}: miss={s['miss_bytes_all']/1e6:.1f}MB cells={s['cell_loads']:.1f} lat={s['lat_ms']:.1f}ms ({time.time()-T0:.0f}s)", flush=True)
