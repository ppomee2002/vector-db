#!/usr/bin/env python3
"""Experiment 2: same total cache budget, unified vs split (ANN | context) page caches, trace-driven.

Request traces over the traced distinct queries: natural (permutations, as qgran_bench) and zipf (alpha).
Per access stream (4 KB pages): per query = ANN link pages, ANN vector pages, then context pages.
Policies (budget = frac of all ANN + context pages):
  unified LRU (Mattson stack distances -> exact miss curve for every size), split LRU (best ratio on 1% grid),
  unified perfect-LFU (counts survive eviction, LRU tie-break), split LFU (ratio grid 5%), OPT (Belady + bypass).
Misses counted after the warm-up requests. Writes <dir>/sim.json."""
import json, sys, time
from pathlib import Path
import numpy as np
from numba import njit

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("nq"); WARM = 2000; REQ = 10000
T = dict(np.load(D / "trace.npz")); LAY = json.loads((D / "layout.json").read_text())
NP = LAY["total_pages"]; nq = len(T["qid"])
VEC_PAGES = sum(-(-L["vec_bytes"] // LAY["page"]) for L in LAY["segments"].values())
LINK_PAGES = LAY["ann_pages"] - VEC_PAGES
# what lives on disk (and so in the cache) per ANN storage setup
SCEN = {"S1": (("lp", "vp"), LAY["ann_pages"]),            # graph + full vectors on disk (traced as is)
        "S2": (("lp", "rp"), LAY["ann_pages"]),            # quantized vectors in RAM; graph + originals on disk (rescoring)
        "S3": (("rp",), VEC_PAGES)}                        # graph + quantized in RAM; originals on disk (rescoring only)
ANN_KEYS, ANN_SIZE = SCEN["S1"]
FRACS = [0.01, 0.02, 0.05, 0.10, 0.20]

def trace(kind, alpha=1.2, seed=20261103):
    rng = np.random.default_rng(seed); total = WARM + REQ
    if kind == "natural":
        return np.concatenate([rng.permutation(nq) for _ in range(-(-total // nq))])[:total]
    ranks = np.arange(1, nq + 1, dtype=float) ** -alpha
    return rng.permutation(nq)[rng.choice(nq, size=total, p=ranks / ranks.sum())]

def build(tr):
    parts, kinds, qstart = [], [], []
    n = 0
    for j, q in enumerate(tr):
        if j == WARM: qstart.append(n)
        for k, kind in tuple((k, 0) for k in ANN_KEYS) + (("cp", 1),):
            a = T[k][T[k + "_off"][q]:T[k + "_off"][q + 1]]
            parts.append(a); kinds.append(np.full(len(a), kind, np.int8)); n += len(a)
    return np.concatenate(parts).astype(np.int32), np.concatenate(kinds), qstart[0]

@njit(cache=True)
def stack_dist(s, npages):
    """LRU stack distance per access (-1 = first access)."""
    n = len(s); bit = np.zeros(n + 1, np.int32); last = np.full(npages, -1, np.int64)
    out = np.empty(n, np.int64)
    for t in range(n):
        p = s[t]; l = last[p]
        if l < 0:
            out[t] = -1
        else:
            # marks in (l, t) = distinct pages touched since
            c = 0; i = t
            while i > 0: c += bit[i]; i -= i & -i
            i = l + 1
            while i > 0: c -= bit[i]; i -= i & -i
            out[t] = c
            i = l + 1
            while i <= n: bit[i] -= 1; i += i & -i
        i = t + 1
        while i <= n: bit[i] += 1; i += i & -i
        last[p] = t
    return out

def miss_curve(sd, start, maxc):
    """misses (accesses >= start) for every cache size 0..maxc."""
    x = sd[start:]
    cold = int((x < 0).sum())
    h = np.bincount(np.minimum(x[x >= 0], maxc), minlength=maxc + 1)
    # hit at size C iff sd < C
    hits = np.concatenate([[0], np.cumsum(h)[:-1]])
    return cold + (len(x) - cold) - hits  # array index = C

@njit(cache=True)
def _push(hk, hp, n, k, p):
    i = n; hk[i] = k; hp[i] = p
    while i > 0:
        j = (i - 1) >> 1
        if hk[j] <= hk[i]: break
        hk[i], hk[j] = hk[j], hk[i]; hp[i], hp[j] = hp[j], hp[i]; i = j
    return n + 1

@njit(cache=True)
def _pop(hk, hp, n):
    k, p = hk[0], hp[0]; n -= 1
    hk[0] = hk[n]; hp[0] = hp[n]; i = 0
    while True:
        a = 2 * i + 1; b = a + 1; m = i
        if a < n and hk[a] < hk[m]: m = a
        if b < n and hk[b] < hk[m]: m = b
        if m == i: break
        hk[i], hk[m] = hk[m], hk[i]; hp[i], hp[m] = hp[m], hp[i]; i = m
    return k, p, n

@njit(cache=True)
def _rebuild(hk, hp, res, cur, npages):
    n = 0
    for p in range(npages):
        if res[p]: n = _push(hk, hp, n, cur[p], p)
    return n

@njit(cache=True)
def lfu(s, start, cap, npages):
    if cap <= 0: return len(s) - start
    cnt = np.zeros(npages, np.int64); cur = np.zeros(npages, np.int64); res = np.zeros(npages, np.bool_)
    H = 4 * cap + 1024; hk = np.empty(H, np.int64); hp = np.empty(H, np.int32); n = 0; size = 0; miss = 0
    for t in range(len(s)):
        p = s[t]; cnt[p] += 1; k = (cnt[p] << 36) | t; cur[p] = k
        if not res[p]:
            if t >= start: miss += 1
            if size == cap:
                while True:
                    kk, pp, n = _pop(hk, hp, n)
                    if res[pp] and cur[pp] == kk: break
                res[pp] = False; size -= 1
            res[p] = True; size += 1
        if n == H: n = _rebuild(hk, hp, res, cur, npages)
        n = _push(hk, hp, n, k, p)
    return miss

@njit(cache=True)
def next_use(s, npages):
    n = len(s); nx = np.empty(n, np.int64); seen = np.full(npages, 1 << 62, np.int64)
    for t in range(n - 1, -1, -1):
        nx[t] = seen[s[t]]; seen[s[t]] = t
    return nx

@njit(cache=True)
def opt(s, nx, start, cap, npages):
    """Belady with bypass; heap on -next_use (max next use on top)."""
    if cap <= 0: return len(s) - start
    cur = np.zeros(npages, np.int64); res = np.zeros(npages, np.bool_)
    H = 4 * cap + 1024; hk = np.empty(H, np.int64); hp = np.empty(H, np.int32); n = 0; size = 0; miss = 0
    INF = 1 << 62
    for t in range(len(s)):
        p = s[t]; v = nx[t]
        if res[p]:
            if v >= INF:
                res[p] = False; size -= 1  # never used again: free the slot
            else:
                cur[p] = -v
                if n == H: n = _rebuild(hk, hp, res, cur, npages)
                n = _push(hk, hp, n, -v, p)
            continue
        if t >= start: miss += 1
        if v >= INF: continue
        if size == cap:
            while True:  # peek the valid farthest
                kk, pp = hk[0], hp[0]
                if res[pp] and cur[pp] == kk: break
                kk, pp, n = _pop(hk, hp, n)
            if -kk <= v: continue  # bypass: the new page is needed later than everything cached
            kk, pp, n = _pop(hk, hp, n); res[pp] = False; size -= 1
        res[p] = True; size += 1; cur[p] = -v
        if n == H: n = _rebuild(hk, hp, res, cur, npages)
        n = _push(hk, hp, n, -v, p)
    return miss

def run(name, tr):
    t0 = time.time()
    s, kind, start = build(tr)
    A = s[kind == 0]; C = s[kind == 1]
    startA = int((kind[:start] == 0).sum()); startC = start - startA
    ann_acc, ctx_acc = len(A) - startA, len(C) - startC
    B = {f: int(f * (ANN_SIZE + LAY["ctx_pages"])) for f in FRACS}; maxB = max(B.values())
    sdU, sdA, sdC = stack_dist(s, NP), stack_dist(A, NP), stack_dist(C, NP)
    mU, mA, mC = (miss_curve(x, st, maxB) for x, st in ((sdU, start), (sdA, startA), (sdC, startC)))
    # unified LRU split by kind
    xU, kU = sdU[start:], kind[start:]
    nx = next_use(s, NP)
    out = dict(trace=name, requests=REQ, accesses_per_query=dict(ann=ann_acc / REQ, ctx=ctx_acc / REQ), budgets={})
    for f, b in B.items():
        lru_missU = int(mU[b]); lru_ann = int(((xU < 0) | (xU >= b))[kU == 0].sum())
        grid = np.arange(0, 101) / 100
        lru_split = [int(mA[int(round(r * b))] + mC[b - int(round(r * b))]) for r in grid]
        rg = np.arange(0, 21) / 20
        lfu_split = []
        for r in rg:
            ba = int(round(r * b)); lfu_split.append(int(lfu(A, startA, ba, NP) + lfu(C, startC, b - ba, NP)))
        lfuU = int(lfu(s, start, b, NP)); optU = int(opt(s, nx, start, b, NP))
        ib, jb = int(np.argmin(lru_split)), int(np.argmin(lfu_split))
        res = dict(pages=b, lru_unified=lru_missU, lru_unified_ann=lru_ann, lru_unified_ctx=lru_missU - lru_ann,
                   lfu_unified=lfuU, opt=optU,
                   lru_split50=lru_split[50], lru_best=lru_split[ib], lru_best_ratio=float(grid[ib]),
                   lru_best_ann=int(mA[int(round(grid[ib] * b))]), lru_best_ctx=int(mC[b - int(round(grid[ib] * b))]),
                   lfu_split50=lfu_split[10], lfu_best=lfu_split[jb], lfu_best_ratio=float(rg[jb]),
                   lru_split_curve=lru_split, lfu_split_curve=lfu_split)
        out["budgets"][str(f)] = res
        print(sc, name, f, {k: (round(v / REQ, 1) if isinstance(v, int) and k != "pages" else v) for k, v in res.items() if "curve" not in k}, flush=True)
    # LRU unified ANN/ctx hit ratio curves for the report (coarse)
    out["ann_only_lru_curve"] = [int(mA[int(c)]) for c in np.linspace(0, maxB, 41)]
    out["ctx_only_lru_curve"] = [int(mC[int(c)]) for c in np.linspace(0, maxB, 41)]
    out["curve_pages"] = [int(c) for c in np.linspace(0, maxB, 41)]
    out["seconds"] = time.time() - t0
    return out

if __name__ == "__main__":
    sc = sys.argv[2]; ANN_KEYS, ANN_SIZE = SCEN[sc]
    traces = sys.argv[3].split(",") if len(sys.argv) > 3 else ["natural", "zipf1.2", "zipf0.9"]
    res = {"scenario": sc, "ann_keys": ANN_KEYS, "data_pages": dict(ann=ANN_SIZE, ctx=LAY["ctx_pages"])}
    for name in traces:
        tr = trace("natural") if name == "natural" else trace("zipf", float(name[4:]))
        res[name] = run(name, tr)
        (D / f"sim-{sc}.json").write_text(json.dumps(res))
    print("SIM_DONE")
