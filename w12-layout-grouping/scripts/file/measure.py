import numpy as np, os, mmap, time, csv, json, random
L = "/workspace/jihyeon/layouts"; E = "/workspace/jihyeon/emb"; D = "/workspace/jihyeon/data/nq"; R = "/workspace/jihyeon/results"
os.makedirs(R, exist_ok=True); B = 4096; N = 2681468; REPS = 5
def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

ln = np.load(f"{L}/len.npy").astype(np.int64)
if not os.path.exists(f"{L}/doc.npy"):
    doc = np.empty(N, np.int32); prev, k = None, -1
    with open(f"{D}/corpus.jsonl") as f:
        for n, line in enumerate(f):
            t = json.loads(line)["title"]
            if t != prev: k += 1; prev = t
            doc[n] = k
    np.save(f"{L}/doc.npy", doc)
doc = np.load(f"{L}/doc.npy")
tops = {"seen": np.load(f"{E}/q_train_top20_ids.npy")[np.load(f"{L}/qset_seen.npy")],
        "unseen": np.load(f"{E}/q_test_top20_ids.npy")[np.load(f"{L}/qset_unseen.npy")]}
logged = np.load(f"{L}/logged_L.npy")
LAYS = "DRKL"
offs = {x: np.load(f"{L}/off_{x}.npy") for x in LAYS}

# 0) 레이아웃 검증: 1000개 passage가 모든 파일에서 원본(D)과 같은지
src = {x: np.memmap(f"{L}/{x}.bin", dtype=np.uint8, mode="r") for x in LAYS}
for p in np.random.default_rng(1).choice(N, 1000, replace=False):
    ref = src["D"][offs["D"][p]:offs["D"][p] + ln[p]].tobytes()
    for x in "RKL": assert src[x][offs[x][p]:offs[x][p] + ln[p]].tobytes() == ref, (x, p)
del src; log("layout check OK")

fds = {x: os.open(f"{L}/{x}.bin", os.O_RDONLY | os.O_DIRECT) for x in LAYS}
mv = memoryview(mmap.mmap(-1, 64 << 20))

def need(top, w):  # 순위 순서대로, 같은 문서 안의 ±w 이웃 포함, 중복 제거
    out, seen = [], set()
    for p in top:
        for q in range(p - w, p + w + 1):
            if 0 <= q < N and doc[q] == doc[p] and q not in seen: seen.add(q); out.append(q)
    return out

def plan(lay, pids, mode):
    off = offs[lay]
    rng_ = [(off[p] // B, (off[p] + ln[p] - 1) // B) for p in pids]
    if mode == "indiv": return rng_
    bl = sorted({b for a, e in rng_ for b in range(a, e + 1)}); ext = []; s = pv = bl[0]
    for b in bl[1:]:
        if b == pv + 1: pv = b; continue
        ext.append((s, pv)); s = pv = b
    ext.append((s, pv)); return ext

def disk():
    for l in open("/proc/diskstats"):
        f = l.split()
        if f[2] == "loop1": return int(f[3]), int(f[5]) * 512

conds = [("exp1", lay, w, qs, k, "merge") for lay in LAYS for w in (0, 1) for qs in ("seen", "unseen") for k in (5, 10)]
conds += [("exp2", "D", 1, qs, k, "indiv") for qs in ("seen", "unseen") for k in (5, 10)]

# idle 기준치 30초
r0, b0 = disk(); time.sleep(30); r1, b1 = disk()
with open(f"{R}/idle_baseline.csv", "w", newline="") as f:
    csv.writer(f).writerows([["window_s", "disk_reads", "disk_bytes", "measured_at"], [30, r1 - r0, b1 - b0, now()]])
log(f"idle 30s: reads={r1-r0}, bytes={b1-b0}")

for p in need(tops["unseen"][0, :10], 1): os.preadv(fds["D"], [mv[:B]], offs["D"][p] // B * B)  # warm-up

QH = ["run_id", "exp", "layout", "neighbor", "query_set", "k", "mode", "rep", "query_idx", "n_passages",
      "useful_bytes", "n_reads", "bytes_read", "blocks", "latency_us", "frac_in_log", "timestamp"]
RH = ["run_id", "exp", "layout", "neighbor", "query_set", "k", "mode", "rep", "n_queries",
      "app_reads", "app_bytes", "disk_reads", "disk_bytes", "wall_s", "start_ts", "end_ts"]
fq = open(f"{R}/per_query.csv", "w", newline=""); wq = csv.writer(fq); wq.writerow(QH)
fr = open(f"{R}/per_run.csv", "w", newline=""); wr = csv.writer(fr); wr.writerow(RH)
rid = 0
for rep in range(REPS):
    order = conds[:]; random.Random(rep).shuffle(order)
    for exp, lay, w, qs, k, mode in order:
        rid += 1; fd = fds[lay]; st = now(); d0 = disk(); t0 = time.perf_counter(); ar = ab = 0
        for qi, top in enumerate(tops[qs][:, :k]):
            pids = need(top, w); ext = plan(lay, pids, mode)
            t = time.perf_counter()
            for a, e in ext: os.preadv(fd, [mv[:(e - a + 1) * B]], a * B)
            lat = (time.perf_counter() - t) * 1e6
            nb = sum(e - a + 1 for a, e in ext); ar += len(ext); ab += nb * B
            wq.writerow([rid, exp, lay, w, qs, k, mode, rep, qi, len(pids), int(ln[pids].sum()),
                         len(ext), nb * B, nb, round(lat, 1), round(float(logged[top].mean()), 3), now()])
        d1 = disk(); wall = time.perf_counter() - t0
        wr.writerow([rid, exp, lay, w, qs, k, mode, rep, len(tops[qs]), ar, ab, d1[0] - d0[0], d1[1] - d0[1], round(wall, 2), st, now()])
        fq.flush(); fr.flush()
        log(f"run {rid}/{len(conds)*REPS} {exp} {lay} ±{w} {qs} k={k} {mode}: {wall:.1f}s")
log("ALL DONE")
