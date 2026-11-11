"""Milvus에 배치별 컬렉션(m4m_<layout>) 적재. pk = 그 배치 순서에서의 위치 -> pk 순서 = 배치 순서.
  doc / interleaved / stream8 / coaccess / oracle : Qdrant 실험과 같은 순서 파일(order_*.npy) 사용
  doc_rpk : 문서 순서로 넣되 pk는 무작위 (Milvus가 pk 순으로 재정렬하는지 확인용)"""
import argparse, json, os, time, numpy as np
from pymilvus import MilvusClient, DataType, connections, utility
from common import load_data
from milvus_common import URI
ap = argparse.ArgumentParser(); ap.add_argument("--layout", required=True); ap.add_argument("--batch", type=int, default=5000)
a = ap.parse_args()
D = os.environ.get("M4_DATA", os.path.expanduser("~/m4-exp/data"))
chunks, topk, train, test, bounds = load_data(D); emb = np.load(os.path.join(D, "emb.npy")); N = len(chunks)
base = "doc" if a.layout == "doc_rpk" else a.layout
order = np.load(os.path.join(D, f"order_{base}.npy"))
pk_of = np.empty(N, np.int64)
if a.layout == "doc_rpk": pk_of[order] = np.random.default_rng(7).permutation(N)
else: pk_of[order] = np.arange(N)
np.save(os.path.join(D, f"mpk_{a.layout}.npy"), pk_of)  # chunk id -> pk
name = f"m4m_{a.layout}"; c = MilvusClient(URI, timeout=1800)
if c.has_collection(name): c.drop_collection(name)  # 우리가 만든 m4m_ 컬렉션만
s = c.create_schema(auto_id=False)
s.add_field("pk", DataType.INT64, is_primary=True); s.add_field("vec", DataType.FLOAT_VECTOR, dim=emb.shape[1])
s.add_field("text", DataType.VARCHAR, max_length=65535); s.add_field("title", DataType.VARCHAR, max_length=1024)
s.add_field("doc", DataType.INT64); s.add_field("pos", DataType.INT64)
ip = c.prepare_index_params(); ip.add_index("vec", index_type="FLAT", metric_type="COSINE")
c.create_collection(name, schema=s, index_params=ip, properties={"warmup.scalarField": "disable"})  # text는 필요할 때만 저장소에서 가져오기 (우리 컬렉션에만 적용)
t0 = time.time()
for i in range(0, N, a.batch):
    ids = order[i:i + a.batch]
    c.insert(name, [{"pk": int(pk_of[j]), "vec": emb[j].tolist(), "text": chunks[j]["text"], "title": chunks[j]["title"][:1000],
                     "doc": chunks[j]["doc"], "pos": chunks[j]["pos"]} for j in ids])
c.flush(name); connections.connect(uri=URI); utility.wait_for_index_building_complete(name)  # sealed + 인덱스 완료까지 대기
print(f"[mload] {name}: rows={N} time={time.time()-t0:.0f}s", flush=True)
