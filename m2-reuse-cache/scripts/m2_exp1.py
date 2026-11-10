#!/usr/bin/env python3
"""Experiment 1: ANN access vs context access per chunk (and per 4 KB page), same queries.
Frequencies = number of requests (in the trace) that touch the item; per request an item counts once.
Writes <dir>/exp1.json (+ 2-D log histogram for the scatter figure)."""
import json, sys
from pathlib import Path
import numpy as np
sys.argv += [] ; D = Path(sys.argv[1])
sys.path.insert(0, str(Path(__file__).parent))
from m2_sim import trace, T, nq  # noqa: E402  (same request traces)

def rankdata(x):
    o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); xs = x[o]
    b = np.concatenate([[True], xs[1:] != xs[:-1], [True]]); idx = np.flatnonzero(b)
    for a, c in zip(idx[:-1], idx[1:]): r[o[a:c]] = (a + c - 1) / 2
    return r

def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b); ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))

def per_query_sets(key):
    return [np.unique(T[key][T[key + "_off"][q]:T[key + "_off"][q + 1]]) for q in range(nq)]

CN, CP = per_query_sets("cn"), per_query_sets("cp")
STREAMS = {"traverse": (per_query_sets("vn"), per_query_sets("vp")), "rescore": (per_query_sets("rn"), per_query_sets("rp"))}

def freqs(sets, w):
    ids = np.concatenate(sets); ww = np.concatenate([np.full(len(s), w[q]) for q, s in enumerate(sets)])
    u, inv = np.unique(ids, return_inverse=True)
    return u, np.bincount(inv, weights=ww)

def gini(f):
    f = np.sort(f); n = len(f); return float((2 * np.arange(1, n + 1) - n - 1).dot(f) / (n * f.sum()))

def cover(f, share):
    """fraction of accessed items holding `share` of the accesses."""
    c = np.cumsum(np.sort(f)[::-1]); return float((np.searchsorted(c, share * c[-1]) + 1) / len(f))

def analyse(name, w, ann):
    VN, VP = STREAMS[ann]
    ua, fa = freqs(VN, w); uc, fc = freqs(CN, w)
    U = np.union1d(ua, uc)
    A = np.zeros(len(U)); A[np.searchsorted(U, ua)] = fa
    C = np.zeros(len(U)); C[np.searchsorted(U, uc)] = fc
    res = dict(trace=name, ann=ann, requests=int(w.sum()), ann_items=len(ua), ctx_items=len(uc), union=len(U),
               ctx_items_never_in_ann=int(((C > 0) & (A == 0)).sum()))
    for p in (0.01, 0.10):
        ha = set(U[np.argsort(-A, kind="stable")[:max(1, int(p * len(ua)))]])
        hc = set(U[np.argsort(-C, kind="stable")[:max(1, int(p * len(uc)))]])
        inter = len(ha & hc)
        hc_arr = np.isin(U, list(hc)); ha_arr = np.isin(U, list(ha))
        res[f"top{int(p*100)}"] = dict(ann_hot=len(ha), ctx_hot=len(hc), overlap=inter,
            overlap_of_ctx_hot=inter / len(hc), jaccard=inter / len(ha | hc),
            ctx_accesses_in_ann_hot=float(C[ha_arr].sum() / C.sum()),
            ann_accesses_in_ctx_hot=float(A[hc_arr].sum() / A.sum()))
    res["spearman_union"] = spearman(A, C)
    both = (A > 0) & (C > 0); res["spearman_both"] = spearman(A[both], C[both]); res["n_both"] = int(both.sum())
    res["gini"] = dict(ann=gini(fa), ctx=gini(fc))
    res["cover80"] = dict(ann=cover(fa, 0.8), ctx=cover(fc, 0.8))
    # page level (what a page cache sees)
    _, pa = freqs(VP, w); _, pc = freqs(CP, w)
    res["page"] = dict(ann_pages=len(pa), ctx_pages=len(pc), gini_ann=gini(pa), gini_ctx=gini(pc),
                       cover80_ann=cover(pa, 0.8), cover80_ctx=cover(pc, 0.8),
                       mean_ann=float(pa.mean()), mean_ctx=float(pc.mean()))
    # rank-frequency (log-spaced ranks) for pages
    res["rankfreq"] = {k: [[int(r), float(np.sort(f)[::-1][r - 1])] for r in
                       np.unique(np.logspace(0, np.log10(len(f)), 60).astype(int))] for k, f in (("ann_page", pa), ("ctx_page", pc), ("ann_chunk", fa), ("ctx_chunk", fc))}
    # 2-D log histogram over the union: x = ANN freq, y = ctx freq (0 kept in its own bin)
    edges = np.concatenate([[0, 1], np.logspace(0, np.log10(max(A.max(), C.max()) + 1), 25)[1:]])
    H, _, _ = np.histogram2d(A, C, bins=[edges, edges])
    res["hist2d"] = dict(edges=edges.tolist(), counts=H.astype(int).tolist())
    # sample of points for a scatter (all ctx items + random ANN-only items)
    rng = np.random.default_rng(0)
    idx = np.concatenate([np.flatnonzero(C > 0), rng.choice(np.flatnonzero(C == 0), min(20000, int((C == 0).sum())), replace=False)])
    res["sample"] = dict(ann=A[idx].tolist(), ctx=C[idx].tolist())
    # reuse distance in requests between consecutive touches of the same page (ANN vs context)
    tr = trace_of[name]
    for kind, sets in (("ann", VP), ("ctx", CP)):
        last = {}; d = []
        for j, q in enumerate(tr):
            for p in sets[q]:
                l = last.get(p)
                if l is not None: d.append(j - l)
                last[p] = j
        d = np.array(d); tot = sum(len(sets[q]) for q in tr)
        res[f"reuse_{kind}"] = dict(reused_share=len(d) / tot,
            cdf=[[int(x), float((d <= x).sum() / tot)] for x in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 12000]])
    print(name, json.dumps({k: v for k, v in res.items() if k not in ("hist2d", "sample", "rankfreq")}), flush=True)
    return res

trace_of = {"distinct": np.arange(nq), "natural": trace("natural"), "zipf1.2": trace("zipf", 1.2)}
out = {}
import os
for name, tr in trace_of.items():
    if os.environ.get("ONLY") and name != os.environ["ONLY"]: continue
    for ann in STREAMS:
        out[f"{name}/{ann}"] = analyse(name, np.bincount(tr, minlength=nq).astype(float), ann)
(D / ("exp1.json" if not os.environ.get("ONLY") else "exp1-test.json")).write_text(json.dumps(out))
print("EXP1_DONE")
