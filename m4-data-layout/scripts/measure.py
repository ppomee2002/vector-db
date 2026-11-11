"""Step 3: 실제 Qdrant에서 Context 읽기(retrieve)의 I/O 측정
쿼리마다: top-k(+Expansion ±w) chunk의 payload를 retrieve 하면서
 - lat_ms       : retrieve 지연시간
 - read_bytes   : Qdrant 프로세스가 실제 저장장치에서 읽은 bytes (/proc/<pid>/io)
 - majflt       : major page fault 수 (mmap으로 디스크에서 읽은 횟수)
 - pages_*      : 이번 쿼리로 새로 캐시에 올라온 페이지 수 (mincore, 파일 종류별)
 - need_bytes   : 실제로 필요한 Context bytes
mode=cold_each : 쿼리마다 '우리 컬렉션 파일만' 캐시에서 내림(posix_fadvise) -> 쿼리 단위 cold
mode=warm      : 시작 시 한 번만 내림 -> 캐시가 차가는 실제 운영 상황 (배치 간 캐시 효율 비교, QPS)
drop_caches/sudo 사용 안 함. 다른 사용자 파일·캐시는 건드리지 않음.
"""
import argparse, csv, ctypes, json, mmap, os, time
import numpy as np
from qdrant_client import QdrantClient
from common import load_data, expand

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.environ.get("M4_DATA", os.path.expanduser("~/m4-exp/data")))
ap.add_argument("--layout", required=True)
ap.add_argument("--window", type=int, default=0)
ap.add_argument("--mode", choices=["cold_each", "warm"], default="cold_each")
ap.add_argument("--nq", type=int, default=500)
ap.add_argument("--port", type=int, default=6343)
ap.add_argument("--out", default=os.path.expanduser("~/m4-exp/results"))
a = ap.parse_args()
D = os.path.expanduser("~/m4-exp")
pid = int(open(f"{D}/qdrant.pid").read())
coll_dir = os.environ.get("M4_STORAGE", f"{D}/qdrant_storage") + f"/collections/m4_{a.layout}"
files = [os.path.join(r, f) for r, _, fs in os.walk(coll_dir) for f in fs if os.path.getsize(os.path.join(r, f)) > 0]
def kind(p):
    b = os.path.basename(p)
    if "payload" in p:  # payload 파일을 데이터 페이지와 위치정보(tracker) 등으로 구분
        return "pdata" if b.startswith("page_") else ("ptracker" if b.startswith("tracker") else "pmeta")
    return "vector" if "vector" in p else "other"

libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
PS = os.sysconf("SC_PAGE_SIZE"); FAIL = ctypes.c_void_p(-1).value

def resident(p):
    sz = os.path.getsize(p); fd = os.open(p, os.O_RDONLY)
    addr = libc.mmap(None, sz, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0); os.close(fd)
    if addr in (None, FAIL): return np.zeros(0, np.uint8)
    n = (sz + PS - 1) // PS; vec = (ctypes.c_ubyte * n)()
    libc.mincore(addr, sz, vec); libc.munmap(addr, sz)
    return np.frombuffer(vec, np.uint8) & 1

def snapshot(): return {p: resident(p) for p in files}
def evict():
    for p in files:
        fd = os.open(p, os.O_RDONLY); os.fsync(fd); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
def proc():
    io = dict(l.split(": ") for l in open(f"/proc/{pid}/io").read().splitlines())
    st = open(f"/proc/{pid}/stat").read(); st = st[st.rfind(")") + 2:].split()
    return int(io["read_bytes"]), int(st[9])

chunks, topk, train, test, bounds = load_data(a.data)
nbytes = np.array([len(c["text"].encode()) + len(c["title"].encode()) for c in chunks])
c = QdrantClient(host="127.0.0.1", port=a.port, timeout=120)
name = f"m4_{a.layout}"
c.retrieve(name, ids=[0], with_payload=True)  # 연결 워밍업
evict()
rows = []; T0 = time.perf_counter()
for q in test[: a.nq]:
    ids = expand(topk[q], bounds, a.window)
    if a.mode == "cold_each": evict()
    s0 = snapshot(); rb0, mf0 = proc()
    t = time.perf_counter(); res = c.retrieve(name, ids=ids, with_payload=True, with_vectors=False); lat = time.perf_counter() - t
    rb1, mf1 = proc(); s1 = snapshot()
    assert len(res) == len(ids)
    pg = {"pdata": 0, "ptracker": 0, "pmeta": 0, "vector": 0, "other": 0}
    for p in files:
        x0, x1 = s0[p], s1[p]; m = min(len(x0), len(x1))
        pg[kind(p)] += int(((x1[:m] == 1) & (x0[:m] == 0)).sum())
    rows.append(dict(layout=a.layout, window=a.window, mode=a.mode, qid=int(q), n_ctx=len(ids),
                     need_bytes=int(nbytes[ids].sum()), lat_ms=lat * 1e3, read_bytes=rb1 - rb0, majflt=mf1 - mf0,
                     pages_payload=pg["pdata"] + pg["ptracker"] + pg["pmeta"], pages_pdata=pg["pdata"], pages_ptracker=pg["ptracker"],
                     pages_pmeta=pg["pmeta"], pages_vector=pg["vector"], pages_other=pg["other"]))
wall = time.perf_counter() - T0
os.makedirs(a.out, exist_ok=True)
fn = f"{a.out}/q_{a.layout}_w{a.window}_{a.mode}.csv"
with open(fn, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
r = {k: float(np.mean([x[k] for x in rows])) for k in ("n_ctx", "need_bytes", "lat_ms", "read_bytes", "majflt", "pages_payload", "pages_pdata", "pages_ptracker", "pages_pmeta")}
r["p95_ms"] = float(np.percentile([x["lat_ms"] for x in rows], 95))
print(f"[measure] {a.layout:11s} w={a.window} {a.mode:9s} lat={r['lat_ms']:.2f}ms p95={r['p95_ms']:.2f} "
      f"read={r['read_bytes']/1024:.1f}KB need={r['need_bytes']/1024:.1f}KB pages_payload={r['pages_payload']:.1f} majflt={r['majflt']:.1f} "
      f"files={len(files)} (note: wall incl. measuring overhead {wall:.0f}s)", flush=True)

# 측정 오버헤드 없이 순수 처리량(QPS): 캐시를 비운 상태에서 같은 쿼리들을 연속 실행
if a.mode == "warm":
    evict(); rb0, mf0 = proc(); t = time.perf_counter()
    for q in test[: a.nq]: c.retrieve(name, ids=expand(topk[q], bounds, a.window), with_payload=True, with_vectors=False)
    el = time.perf_counter() - t; rb1, mf1 = proc()
    r.update(qps=a.nq / el, total_read_mb=(rb1 - rb0) / 1e6, total_majflt=mf1 - mf0)
    print(f"[qps] {a.layout:11s} w={a.window} qps={r['qps']:.1f} total_read={r['total_read_mb']:.1f}MB majflt={r['total_majflt']}", flush=True)
r.update(layout=a.layout, window=a.window, mode=a.mode)
with open(f"{a.out}/summary.jsonl", "a") as f: f.write(json.dumps(r) + "\n")
