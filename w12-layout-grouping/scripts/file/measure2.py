import numpy as np, os, mmap, time, csv, random
L = "/workspace/jihyeon/layouts"; E = "/workspace/jihyeon/emb"; R = "/workspace/jihyeon/results"; B = 4096; N = 2681468; REPS = 3
LAYS = ["D", "R", "K", "L", "P16", "P64", "P256", "P64s"]
def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
ln = np.load(f"{L}/len.npy").astype(np.int64); doc = np.load(f"{L}/doc.npy")
tops = {"seen": np.load(f"{E}/q_train_top20_ids.npy")[np.load(f"{L}/qset_seen.npy")],
        "unseen": np.load(f"{E}/q_test_top20_ids.npy")[np.load(f"{L}/qset_unseen.npy")]}
offs = {x: np.load(f"{L}/off_{x}.npy") for x in LAYS}
src = {x: np.memmap(f"{L}/{x}.bin", dtype=np.uint8, mode="r") for x in LAYS}
for p in np.random.default_rng(1).choice(N, 1000, replace=False):
    ref = src["D"][offs["D"][p]:offs["D"][p] + ln[p]].tobytes()
    for x in LAYS: assert src[x][offs[x][p]:offs[x][p] + ln[p]].tobytes() == ref, (x, p)
del src; log("layout check OK")
fds = {x: os.open(f"{L}/{x}.bin", os.O_RDONLY | os.O_DIRECT) for x in LAYS}
mv = memoryview(mmap.mmap(-1, 64 << 20))
def need(top, w):
    out, seen = [], set()
    for p in top:
        for q in range(p - w, p + w + 1):
            if 0 <= q < N and doc[q] == doc[p] and q not in seen: seen.add(q); out.append(q)
    return out
def plan(lay, pids, G):  # 블록 간격이 G 이하면 한 번에 읽음 (G=0: 붙은 블록만)
    off = offs[lay]
    bl = sorted({b for p in pids for b in range(off[p] // B, (off[p] + ln[p] - 1) // B + 1)}); ext = []; s = pv = bl[0]
    for b in bl[1:]:
        if b <= pv + 1 + G: pv = b; continue
        ext.append((s, pv)); s = pv = b
    ext.append((s, pv)); return ext
def disk():
    for l in open("/proc/diskstats"):
        f = l.split()
        if f[2] == "loop1": return int(f[3]), int(f[5]) * 512
conds = [(lay, w, qs, k, G) for lay in LAYS for w in (0, 1) for qs in ("seen", "unseen") for k in (5, 10) for G in (0, 2)]
fq = open(f"{R}/per_query2.csv", "w", newline=""); wq = csv.writer(fq)
wq.writerow(["run_id", "layout", "neighbor", "query_set", "k", "gap", "rep", "query_idx", "n_passages", "useful_bytes", "n_reads", "bytes_read", "latency_us", "timestamp"])
fr = open(f"{R}/per_run2.csv", "w", newline=""); wr = csv.writer(fr)
wr.writerow(["run_id", "layout", "neighbor", "query_set", "k", "gap", "rep", "n_queries", "app_reads", "app_bytes", "disk_reads", "disk_bytes", "wall_s", "start_ts", "end_ts"])
rid = 0
for rep in range(REPS):
    order = conds[:]; random.Random(rep).shuffle(order)
    for lay, w, qs, k, G in order:
        rid += 1; fd = fds[lay]; st = now(); d0 = disk(); t0 = time.perf_counter(); ar = ab = 0
        for qi, top in enumerate(tops[qs][:, :k]):
            pids = need(top, w); ext = plan(lay, pids, G)
            t = time.perf_counter()
            for a, e in ext: os.preadv(fd, [mv[:(e - a + 1) * B]], a * B)
            lat = (time.perf_counter() - t) * 1e6
            nb = sum(e - a + 1 for a, e in ext); ar += len(ext); ab += nb * B
            wq.writerow([rid, lay, w, qs, k, G, rep, qi, len(pids), int(ln[pids].sum()), len(ext), nb * B, round(lat, 1), now()])
        d1 = disk(); wall = time.perf_counter() - t0
        wr.writerow([rid, lay, w, qs, k, G, rep, len(tops[qs]), ar, ab, d1[0] - d0[0], d1[1] - d0[1], round(wall, 2), st, now()])
        fq.flush(); fr.flush()
        if rid % 20 == 0: log(f"run {rid}/{len(conds)*REPS}")
log("ALL DONE")
