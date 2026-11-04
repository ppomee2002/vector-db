#!/usr/bin/env python3
"""ANN search access vs RAG context access, measured on the same queries.

Builds a faiss HNSW (M=16, efConstruction=128, inner product; same parameters as the Milvus runs),
replays HNSW search in Python with ef=64 to log every node whose vector is read (distance computed)
and every node whose neighbour list is read (expanded), and compares that stream with the context
stream (the top-10 chunks whose text a RAG pipeline fetches).

Outputs results/ann-vs-context/<ds>.json and ANN_VS_CONTEXT.md.
"""
import argparse, collections, heapq, json, os, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
G = {}


def build(ds, threads):
    import faiss
    faiss.omp_set_num_threads(threads)
    X = np.ascontiguousarray(np.load(ROOT / f"datasets/emb/{ds}/corpus.npy"))
    f = ROOT / f"results/ann-vs-context/{ds}.hnsw"
    if f.exists():
        idx = faiss.read_index(str(f))
    else:
        idx = faiss.IndexHNSWFlat(X.shape[1], 16, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = 128
        t0 = time.time(); idx.add(X); print(f"{ds}: HNSW built in {time.time() - t0:.0f}s", flush=True)
        faiss.write_index(idx, str(f))
    h = idx.hnsw
    return dict(X=X, idx=idx, nb=faiss.vector_to_array(h.neighbors), off=faiss.vector_to_array(h.offsets),
                cum=faiss.vector_to_array(h.cum_nneighbor_per_level), lev=faiss.vector_to_array(h.levels),
                ep=int(h.entry_point), maxl=int(h.max_level))


def neigh(i, l):
    g = G
    s = int(g["off"][i]) + int(g["cum"][l]); e = int(g["off"][i]) + int(g["cum"][l + 1])
    n = g["nb"][s:e]
    return n[n >= 0]


def search(qi, ef=64, k=10):
    g = G
    q = g["Q"][qi]
    X = g["X"]
    visited, expanded = [], []
    cur = g["ep"]; d = float(X[cur] @ q); visited.append(cur)
    for l in range(g["maxl"], 0, -1):  # greedy descent on upper levels
        changed = True
        while changed:
            changed = False
            n = neigh(cur, l); expanded.append((cur, l))
            if len(n) == 0:
                continue
            ds = X[n] @ q; visited.extend(n.tolist())
            j = int(np.argmax(ds))
            if ds[j] > d:
                cur, d, changed = int(n[j]), float(ds[j]), True
    seen = {cur}
    cand = [(-d, cur)]; top = [(d, cur)]
    while cand:
        nd, c = heapq.heappop(cand)
        if -nd < top[0][0] and len(top) >= ef:
            break
        n = neigh(c, 0); expanded.append((c, 0))
        n = np.array([x for x in n.tolist() if x not in seen], dtype=np.int64)
        if len(n) == 0:
            continue
        seen.update(n.tolist())
        ds = X[n] @ q; visited.extend(n.tolist())
        for x, dx in zip(n.tolist(), ds.tolist()):
            if len(top) < ef or dx > top[0][0]:
                heapq.heappush(cand, (-dx, x)); heapq.heappush(top, (dx, x))
                if len(top) > ef:
                    heapq.heappop(top)
    res = [x for _, x in sorted(top, reverse=True)[:k]]
    return visited, expanded, res


def lru_hit(stream, cap):
    od, hit = collections.OrderedDict(), 0
    for x in stream:
        if x in od:
            od.move_to_end(x); hit += 1
        else:
            od[x] = None
            if len(od) > cap:
                od.popitem(last=False)
    return hit / len(stream)


def top_share(stream, n_items, frac):
    c = np.array(sorted(collections.Counter(stream).values(), reverse=True))
    return c[:max(1, int(n_items * frac))].sum() / c.sum()


def reuse(per_query):
    seen, hit, tot = set(), 0, 0
    for items in per_query:
        s = set(items)
        hit += len(s & seen); tot += len(s); seen |= s
    return hit / tot


def blocks_per_query(per_query, off, gsz):
    """Average distinct blocks of size gsz touched per query, and bytes touched / bytes needed."""
    nb, touched, need = 0, 0, 0
    for items in per_query:
        b = set()
        for x in set(items):
            b.update(range(int(off[x]) // gsz, (int(off[x + 1]) - 1) // gsz + 1))
            need += int(off[x + 1] - off[x])
        nb += len(b); touched += len(b) * gsz
    return nb / len(per_query), touched / need


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="fiqa,nq")
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    out = ROOT / "results/ann-vs-context"; out.mkdir(parents=True, exist_ok=True)
    md = ["# ANN search access vs RAG context access (same queries)", "",
          "faiss HNSW M=16 efC=128 IP, search replayed in Python with ef=64, k=10. Distinct queries in the natural (permutation) order.",
          "ANN stream = nodes whose vector is read (distance computed). Context stream = top-10 chunks whose text is fetched.", ""]
    summary = {}
    for ds in a.datasets.split(","):
        global G
        G = build(ds, a.threads)
        G["Q"] = np.load(ROOT / f"datasets/emb/{ds}/queries.npy")
        nq, N = G["Q"].shape[0], G["X"].shape[0]
        order = np.random.default_rng(20261103).permutation(nq)[:a.queries].tolist()
        with Pool(min(a.threads, 32)) as p:
            R = p.map(search, order, chunksize=8)
        V = [r[0] for r in R]; E = [r[1] for r in R]; T = [r[2] for r in R]
        # sanity: agreement with faiss' own search at ef=64
        G["idx"].hnsw.efSearch = 64
        _, I = G["idx"].search(G["Q"][order[:200]], 10)
        agree = np.mean([len(set(T[i]) & set(I[i].tolist())) / 10 for i in range(200)])
        L = np.load(ROOT / f"datasets/emb/{ds}/text_len.npy").astype(np.int64)
        toff = np.concatenate([[0], np.cumsum(L)])
        voff = np.arange(N + 1, dtype=np.int64) * G["X"].shape[1] * 4  # vectors stored in pk order
        lev = G["lev"]
        vs = [x for v in V for x in v]; cs = [x for t in T for x in t]
        upper = np.mean([lev[x] > 1 for x in vs])  # faiss levels: 1 = level 0 only
        s = dict(dataset=ds, corpus=N, queries=len(order), faiss_agreement=float(agree),
                 ann_vectors_per_query=float(np.mean([len(v) for v in V])),
                 ann_lists_per_query=float(np.mean([len(e) for e in E])),
                 ann_bytes_per_query=float(np.mean([len(v) for v in V]) * G["X"].shape[1] * 4 + np.mean([len(e) for e in E]) * 32 * 4),
                 ctx_chunks_per_query=10.0, ctx_bytes_per_query=float(np.mean([L[t].sum() for t in T])),
                 ann_upper_level_share=float(upper),
                 ann_top1pct_share=float(top_share(vs, N, 0.01)), ctx_top1pct_share=float(top_share(cs, N, 0.01)),
                 ann_top10pct_share=float(top_share(vs, N, 0.10)), ctx_top10pct_share=float(top_share(cs, N, 0.10)),
                 ann_reuse=float(reuse(V)), ctx_reuse=float(reuse(T)),
                 lru={}, blocks={})
        for frac in (0.001, 0.01, 0.05, 0.10):
            cap = max(1, int(N * frac))
            s["lru"][str(frac)] = dict(ann=lru_hit(vs, cap), ctx=lru_hit(cs, cap))
        for name, gsz in (("4KB", 4096), ("2MB", 2 << 20)):
            s["blocks"][name] = dict(ann=blocks_per_query(V, voff, gsz), ctx=blocks_per_query(T, toff, gsz))
        summary[ds] = s
        (out / f"{ds}.json").write_text(json.dumps(s, indent=1) + "\n")
        print(json.dumps(s), flush=True)
    for ds, s in summary.items():
        pc = lambda x: f"{x:.1%}"
        md += [f"## {ds} ({s['corpus']:,} chunks, {s['queries']} queries; replay agrees with faiss top-10: {pc(s['faiss_agreement'])})", "",
               "| | ANN search (HNSW) | RAG context fetch |", "|---|---:|---:|",
               f"| items read per query | {s['ann_vectors_per_query']:.0f} vectors + {s['ann_lists_per_query']:.0f} neighbour lists | 10 chunks |",
               f"| bytes read per query | {s['ann_bytes_per_query'] / 1e3:.0f} KB | {s['ctx_bytes_per_query'] / 1e3:.1f} KB |",
               f"| accesses on top 1% items | {pc(s['ann_top1pct_share'])} | {pc(s['ctx_top1pct_share'])} |",
               f"| accesses on top 10% items | {pc(s['ann_top10pct_share'])} | {pc(s['ctx_top10pct_share'])} |",
               f"| items already used by an earlier query | {pc(s['ann_reuse'])} | {pc(s['ctx_reuse'])} |",
               f"| reads on upper HNSW levels | {pc(s['ann_upper_level_share'])} | - |"]
        for frac, h in s["lru"].items():
            md.append(f"| LRU hit ratio, cache = {float(frac):.1%} of items | {pc(h['ann'])} | {pc(h['ctx'])} |")
        for name, b in s["blocks"].items():
            md.append(f"| {name} blocks touched per query (pk-order layout), touched/needed bytes | {b['ann'][0]:.0f}, {b['ann'][1]:.1f}x | {b['ctx'][0]:.1f}, {b['ctx'][1]:.1f}x |")
        md.append("")
    (out / "ANN_VS_CONTEXT.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
