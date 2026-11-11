"""1단계: Milvus가 M4 측정에 맞는 설정인지 확인 (작은 컬렉션 m4m_probe, 끝나면 삭제)
- 불러오기(load) 직후 text가 메모리에 다 올라오면 -> warmup 켜짐 (필요할 때 저장소에서 가져오는 구조 아님)
- 조회할 때 cell miss bytes가 늘면 -> 필요할 때 가져오는 구조 (M4 측정 가능)"""
import os, time, numpy as np
from pymilvus import MilvusClient, DataType
from common import load_data
from milvus_common import URI, metrics, delta, pick
D = os.environ.get("M4_DATA", os.path.expanduser("~/m4-exp/data"))
chunks, topk, train, test, bounds = load_data(D); emb = np.load(os.path.join(D, "emb.npy"))
N = 20000; name = "m4m_probe"
c = MilvusClient(URI, timeout=600)
if c.has_collection(name): c.drop_collection(name)
s = c.create_schema(auto_id=False)
s.add_field("pk", DataType.INT64, is_primary=True); s.add_field("vec", DataType.FLOAT_VECTOR, dim=emb.shape[1])
s.add_field("text", DataType.VARCHAR, max_length=65535)
ip = c.prepare_index_params(); ip.add_index("vec", index_type="FLAT", metric_type="COSINE")
c.create_collection(name, schema=s, index_params=ip)
for i in range(0, N, 5000):
    c.insert(name, [{"pk": j, "vec": emb[j].tolist(), "text": chunks[j]["text"]} for j in range(i, min(N, i + 5000))])
c.flush(name); c.release_collection(name)
m0 = metrics(); c.load_collection(name); time.sleep(3); m1 = metrics()
ld = delta(m0, m1)
print("load 직후 scalar_field loaded_bytes 증가:", int(pick(ld, "internal_cache_loaded_bytes", "scalar_field")))
r = c.get(name, ids=list(range(100, 110)), output_fields=["text"]); time.sleep(1); m2 = metrics()
qd = delta(m1, m2)
print("조회 10건: miss bytes", int(pick(qd, "internal_cache_cell_access_miss_bytes_total")),
      "/ hit bytes", int(pick(qd, "internal_cache_cell_access_hit_bytes_total")),
      "/ cell 로드", int(pick(qd, "internal_cache_load_latency_microseconds_count")),
      "/ 저장소 요청", int(pick(qd, "internal_storage_op_count")))
print("text 실제 크기(10건):", sum(len(x["text"].encode()) for x in r))
c.release_collection(name); c.drop_collection(name); print("probe 삭제 완료")
