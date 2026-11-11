"""Step 2: 같은 chunk들을 삽입 순서만 바꿔 Qdrant 컬렉션(m4_<layout>)에 적재
 - doc        : 문서 순서대로 (일괄 적재, Expansion에 유리한 배치)
 - interleaved: 여러 문서가 섞여 들어옴 (스트리밍/병렬 적재 상황)
 - coaccess   : train 쿼리에서 함께 검색된 chunk(+앞뒤 chunk)를 붙여서 적재 (RAG 패턴 이용)
point id = chunk id 로 모든 배치에서 동일. payload는 디스크(on_disk_payload).
"""
import argparse, os, time
import numpy as np
from qdrant_client import QdrantClient, models
from common import load_data, expand

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.environ.get("M4_DATA", os.path.expanduser("~/m4-exp/data")))
ap.add_argument("--layout", required=True, help="doc | interleaved | coaccess | oracle | stream<N>")
ap.add_argument("--port", type=int, default=6343)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--window", type=int, default=1)
a = ap.parse_args()

chunks, topk, train, test, bounds = load_data(a.data)
# "<layout>_1seg": 같은 배치를 세그먼트 1개로 저장 (Qdrant가 데이터를 여러 세그먼트로 나누는 영향 확인)
#   "<layout>_b<N>": 한 번에 N개씩 적재 (예: doc_b1 = 1개씩 순서대로) -> 적재 묶음 크기가 저장 순서에 주는 영향 확인
import re
_m = re.match(r"^(.*?)(_1seg)?(?:_b(\d+))?$", a.layout)
base, one_seg = _m.group(1), bool(_m.group(2))
if _m.group(3): a.batch = int(_m.group(3))
emb = np.load(os.path.join(a.data, "emb.npy")); N = len(chunks)
if base == "doc":
    order = np.arange(N)
elif base == "interleaved":
    order = np.random.default_rng(1).permutation(N)
elif base.startswith("stream"):
    # N개 문서 스트림이 동시에 적재되는 상황: 문서들을 N개 큐로 나누고 chunk를 라운드로빈으로 섞음
    ns = int(base[6:]); docs = {}
    for j in range(N): docs.setdefault(chunks[j]["doc"], []).append(j)
    queues = [[] for _ in range(ns)]
    for i, d in enumerate(sorted(docs)): queues[i % ns].extend(docs[d])
    order = [x for grp in __import__("itertools").zip_longest(*queues) for x in grp if x is not None]
    order = np.array(order)
else:
    # coaccess: train 쿼리로 학습(현실적) / oracle: 측정할 test 쿼리로 학습(상한)
    qset = test if base == "oracle" else train
    freq = np.bincount(topk[qset].ravel(), minlength=N); seen = np.zeros(N, bool); order = []
    for q in sorted(qset, key=lambda q: -freq[topk[q]].sum()):
        for j in expand(topk[q], bounds, a.window):
            if not seen[j]: seen[j] = True; order.append(j)
    order = np.array(order + [j for j in range(N) if not seen[j]])
np.save(os.path.join(a.data, f"order_{a.layout}.npy"), order)

c = QdrantClient(host="127.0.0.1", port=a.port, timeout=600)
name = f"m4_{a.layout}"
if c.collection_exists(name): c.delete_collection(name)  # 우리가 만든 m4_ 컬렉션만
c.create_collection(name,
    vectors_config=models.VectorParams(size=emb.shape[1], distance=models.Distance.COSINE, on_disk=False),
    on_disk_payload=True,
    optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0, **({"default_segment_number": 1} if one_seg else {})))
t0 = time.time()
for i in range(0, N, a.batch):
    ids = order[i:i + a.batch]
    c.upsert(name, wait=True, points=models.Batch(ids=ids.tolist(), vectors=emb[ids].tolist(),
             payloads=[{k: chunks[j][k] for k in ("text", "title", "doc", "pos")} for j in ids]))
while c.get_collection(name).status != models.CollectionStatus.GREEN: time.sleep(2)
info = c.get_collection(name)
print(f"[load] {name}: points={info.points_count} segments={info.segments_count} time={time.time()-t0:.0f}s")
