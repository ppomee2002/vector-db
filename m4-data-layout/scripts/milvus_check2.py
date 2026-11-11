"""확인 2: 인덱스 완료 + sealed segment 전환을 기다린 뒤 캐시 측정값이 움직이는지 확인 (m4m_probe2, 끝나면 삭제)"""
import os, time, urllib.request, numpy as np
from pymilvus import MilvusClient, DataType, connections, utility
from common import load_data
from milvus_common import URI, MET, metrics, delta, pick
D = os.environ.get("M4_DATA", os.path.expanduser("~/m4-exp/data"))
chunks, topk, train, test, bounds = load_data(D); emb = np.load(os.path.join(D, "emb.npy"))
N = 20000; name = "m4m_probe2"
c = MilvusClient(URI, timeout=600); connections.connect(uri=URI)
if c.has_collection(name): c.drop_collection(name)
s = c.create_schema(auto_id=False)
s.add_field("pk", DataType.INT64, is_primary=True); s.add_field("vec", DataType.FLOAT_VECTOR, dim=emb.shape[1])
s.add_field("text", DataType.VARCHAR, max_length=65535)
c.create_collection(name, schema=s, properties={"warmup.scalarField": "disable"})  # 우리 컬렉션에만 warmup 끄기
for i in range(0, N, 5000):
    c.insert(name, [{"pk": j, "vec": emb[j].tolist(), "text": chunks[j]["text"]} for j in range(i, min(N, i + 5000))])
c.flush(name)
ip = c.prepare_index_params(); ip.add_index("vec", index_type="FLAT", metric_type="COSINE")
c.create_index(name, ip); utility.wait_for_index_building_complete(name)
print("index 완료")
m0 = metrics(); c.load_collection(name)
t0 = time.time()
while True:
    try: c.search(name, data=[emb[0].tolist()], limit=1); break
    except Exception:
        if time.time() - t0 > 300: raise
        time.sleep(3)
time.sleep(10)
for sg in utility.get_query_segment_info(name):
    print(f"segment {sg.segmentID}: state={sg.state} rows={sg.num_rows}")
m1 = metrics(); ld = delta(m0, m1)
print("load 후 loaded_bytes 증가(종류별):", {t: int(v) for (k, t), v in ld.items() if k == "internal_cache_loaded_bytes" and v})
r = c.get(name, ids=list(range(100, 110)), output_fields=["text"]); time.sleep(2); m2 = metrics(); qd = delta(m1, m2)
print("조회 10건: miss", int(pick(qd, "internal_cache_cell_access_miss_bytes_total")), "/ hit", int(pick(qd, "internal_cache_cell_access_hit_bytes_total")),
      "/ cell 로드", int(pick(qd, "internal_cache_load_latency_microseconds_count")), "/ 저장소 요청", int(pick(qd, "internal_storage_op_count")))
nz = [l for l in urllib.request.urlopen(MET).read().decode().splitlines() if l.startswith("internal_cache") and not l.endswith(" 0")]
print("0이 아닌 캐시 측정값:", len(nz)); [print("  ", l[:160]) for l in nz[:15]]
c.release_collection(name); c.drop_collection(name); print("probe2 삭제 완료")
