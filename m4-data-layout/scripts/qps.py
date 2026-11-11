"""처리량(QPS) 측정 - 캐시가 정말 비어 있는 상태에서 (Qdrant 재시작 직후 우리 파일만 evict)
conc개의 스레드가 동시에 retrieve. 실제로 디스크를 읽었는지 total_read_mb로 확인."""
import argparse, json, os, threading, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from qdrant_client import QdrantClient
from common import load_data, expand

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.environ.get("M4_DATA", os.path.expanduser("~/m4-exp/data")))
ap.add_argument("--layout", required=True); ap.add_argument("--window", type=int, default=0)
ap.add_argument("--conc", type=int, default=1); ap.add_argument("--rep", type=int, default=0)
ap.add_argument("--nq", type=int, default=500); ap.add_argument("--port", type=int, default=6343)
ap.add_argument("--out", default=os.path.expanduser("~/m4-exp/results2"))
a = ap.parse_args()
D = os.path.expanduser("~/m4-exp"); pid = int(open(f"{D}/qdrant.pid").read())
cd = os.environ.get("M4_STORAGE", f"{D}/qdrant_storage") + f"/collections/m4_{a.layout}"
files = [os.path.join(r, f) for r, _, fs in os.walk(cd) for f in fs]
def evict():
    for p in files:
        fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
def proc():
    io = dict(l.split(": ") for l in open(f"/proc/{pid}/io").read().splitlines())
    st = open(f"/proc/{pid}/stat").read(); st = st[st.rfind(")") + 2:].split()
    return int(io["read_bytes"]), int(st[9])
chunks, topk, train, test, bounds = load_data(a.data)
name = f"m4_{a.layout}"; tl = threading.local()
def cli():
    if not hasattr(tl, "c"): tl.c = QdrantClient(host="127.0.0.1", port=a.port, timeout=120)
    return tl.c
reqs = [expand(topk[q], bounds, a.window) for q in test[: a.nq]]
def one(ids):
    c = cli(); t = time.perf_counter(); c.retrieve(name, ids=ids, with_payload=True, with_vectors=False)
    return time.perf_counter() - t
with ThreadPoolExecutor(a.conc) as ex: list(ex.map(lambda _: cli().get_collections(), range(a.conc * 2)))  # 연결만 준비
evict(); rb0, mf0 = proc(); t0 = time.perf_counter()
with ThreadPoolExecutor(a.conc) as ex: lats = np.array(list(ex.map(one, reqs)))
el = time.perf_counter() - t0; rb1, mf1 = proc()
r = dict(layout=a.layout, window=a.window, conc=a.conc, rep=a.rep, qps=len(reqs) / el,
         p50_ms=float(np.median(lats) * 1e3), p95_ms=float(np.percentile(lats, 95) * 1e3),
         total_read_mb=(rb1 - rb0) / 1e6, majflt=mf1 - mf0)
os.makedirs(a.out, exist_ok=True)
open(f"{a.out}/qps.jsonl", "a").write(json.dumps(r) + "\n")
print(f"[qps] {a.layout:11s} w={a.window} C={a.conc:2d} rep={a.rep} qps={r['qps']:7.1f} p50={r['p50_ms']:.2f} p95={r['p95_ms']:.2f} read={r['total_read_mb']:.1f}MB majflt={r['majflt']}", flush=True)
