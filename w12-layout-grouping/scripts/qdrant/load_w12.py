"""배치 순서 파일대로 Qdrant에 적재 (세그먼트 1개, 1개씩 순서대로). 지난주 load.py와 같은 설정."""
import argparse, os, sys, time, numpy as np
sys.path.insert(0, os.path.expanduser("~/m4-exp/scripts"))
from qdrant_client import QdrantClient, models
from common import load_data
ap = argparse.ArgumentParser(); ap.add_argument("--layout", required=True); ap.add_argument("--port", type=int, default=6343); a = ap.parse_args()
W = "/workspace/jihyeon/w12q"; assert a.layout.startswith("w12_")
chunks, topk, train, test, bounds = load_data(f"{W}/data_unseen")
emb = np.load(f"{W}/data_unseen/emb.npy"); order = np.load(f"{W}/orders/order_{a.layout}.npy"); N = len(chunks); assert len(order) == N
c = QdrantClient(host="127.0.0.1", port=a.port, timeout=600); name = f"m4_{a.layout}"
if c.collection_exists(name): c.delete_collection(name)  # w12_ 로 시작하는 우리 컬렉션만
c.create_collection(name, vectors_config=models.VectorParams(size=emb.shape[1], distance=models.Distance.COSINE, on_disk=False),
    on_disk_payload=True, optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0, default_segment_number=1))
t0 = time.time()
for n, j in enumerate(order.tolist()):
    c.upsert(name, wait=True, points=models.Batch(ids=[j], vectors=[emb[j].tolist()],
             payloads=[{k: chunks[j][k] for k in ("text", "title", "doc", "pos")}]))
    if (n + 1) % 25000 == 0: print(f"  {name}: {n+1:,}/{N:,}  {time.time()-t0:.0f}s", flush=True)
while c.get_collection(name).status != models.CollectionStatus.GREEN: time.sleep(2)
info = c.get_collection(name)
print(f"[load] {name}: points={info.points_count} segments={info.segments_count} time={time.time()-t0:.0f}s", flush=True)
