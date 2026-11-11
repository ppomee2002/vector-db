"""Milvus에서 쿼리마다 top-k(+앞뒤 ±w) Context(text)를 가져오며 저장소 읽기 측정.
mode cold_each: 쿼리마다 release+load로 캐시를 비움 (쿼리 단위 cold) / warm: 처음에만 비움.
측정: 저장소에서 새로 가져온 bytes(miss), cell 로드 수, 저장소 요청 수, 필요한 text bytes, 지연시간."""
import argparse, csv, json, os, time, numpy as np
from pymilvus import MilvusClient
from common import load_data, expand
from milvus_common import URI, metrics, delta, pick
ap = argparse.ArgumentParser()
ap.add_argument("--layout", required=True); ap.add_argument("--window", type=int, default=0)
ap.add_argument("--mode", default="cold_each", choices=["cold_each", "warm"]); ap.add_argument("--nq", type=int, default=200)
ap.add_argument("--out", default=os.path.expanduser("~/m4-exp/results_milvus"))
a = ap.parse_args()
D = os.environ.get("M4_DATA", os.path.expanduser("~/m4-exp/data"))
chunks, topk, train, test, bounds = load_data(D); pk_of = np.load(os.path.join(D, f"mpk_{a.layout}.npy"))
nbytes = np.array([len(x["text"].encode()) for x in chunks])
name = f"m4m_{a.layout}"; c = MilvusClient(URI, timeout=600)
def reload():
    c.release_collection(name); c.load_collection(name); time.sleep(1)
reload(); rows = []
for q in test[: a.nq]:
    ids = expand(topk[q], bounds, a.window)
    if a.mode == "cold_each": reload()
    m0 = metrics(); t = time.perf_counter()
    r = c.get(name, ids=[int(pk_of[i]) for i in ids], output_fields=["text"])
    lat = time.perf_counter() - t; time.sleep(0.3); d = delta(m0, metrics())
    assert len(r) == len(ids)
    rows.append(dict(layout=a.layout, window=a.window, mode=a.mode, qid=int(q), n_ctx=len(ids), need_bytes=int(nbytes[ids].sum()),
                     lat_ms=lat * 1e3, miss_bytes=pick(d, "internal_cache_cell_access_miss_bytes_total", "scalar_field"),
                     miss_bytes_all=pick(d, "internal_cache_cell_access_miss_bytes_total"),
                     cell_loads=pick(d, "internal_cache_load_latency_microseconds_count"),
                     storage_ops=pick(d, "internal_storage_op_count"), storage_bytes=pick(d, "internal_storage_kv_size_sum")))
os.makedirs(a.out, exist_ok=True)
with open(f"{a.out}/mq_{a.layout}_w{a.window}_{a.mode}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
s = {k: float(np.mean([x[k] for x in rows])) for k in ("n_ctx", "need_bytes", "lat_ms", "miss_bytes", "miss_bytes_all", "cell_loads", "storage_ops", "storage_bytes")}
s.update(layout=a.layout, window=a.window, mode=a.mode, amp=s["miss_bytes_all"] / s["need_bytes"] if s["need_bytes"] else 0)
open(f"{a.out}/summary.jsonl", "a").write(json.dumps(s) + "\n")
print(f"[mmeasure] {a.layout:12s} w={a.window} {a.mode}: miss={s['miss_bytes_all']/1024:.0f}KB cells={s['cell_loads']:.1f} "
      f"ops={s['storage_ops']:.1f} need={s['need_bytes']/1024:.1f}KB amp={s['amp']:.0f}x lat={s['lat_ms']:.1f}ms", flush=True)
