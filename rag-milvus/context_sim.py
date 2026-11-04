#!/usr/bin/env python3
"""Trace-driven what-if analysis of RAG context reads, replaying measured Milvus results.

Uses the per-query top-k pks recorded by bench.py (results/<run>/*.jsonl), in dispatch order.
  (1) granularity: LRU cache of the same byte budget at chunk / 4 KB / 64 KB / 2 MB blocks
      (blocks = contiguous byte ranges of the corpus in pk order; 2 MB ~ a Milvus text cell)
  (2) expansion (NQ): add +-w neighbouring chunks of the same document; bytes requested
      without and with intra-query range merge, and LRU reads at 4 KB blocks
  (3) cross-query overlap: closed loop with C workers keeps ~C consecutive requests in flight;
      share of block requests in a window of N consecutive requests that another request in the
      same window also needs (= reads a shared read could remove when nothing is cached)
Writes <out>/CONTEXT_SIM.md and context_sim.json.
"""
import argparse, collections, json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
GRAN = {"chunk": 0, "4KB": 4096, "64KB": 65536, "2MB": 2 << 20}


def load_case(path):
    rows = [json.loads(l) for l in open(path)]
    rows.sort(key=lambda r: r["i"])
    return [r["pks"] for r in rows]


class Corpus:
    def __init__(self, ds):
        self.L = np.load(ROOT / f"datasets/emb/{ds}/text_len.npy").astype(np.int64)
        self.off = np.concatenate([[0], np.cumsum(self.L)])
        self.total = int(self.off[-1])
        self.doc = None
        if ds == "nq":  # passages of one Wikipedia article are contiguous and share a title
            t = pq.read_table(next((ROOT / f"datasets/raw/hf/{ds}").glob("corpus-*.parquet")), columns=["title"]).column(0).to_pylist()
            d = np.zeros(len(t), dtype=np.int64)
            for i in range(1, len(t)):
                d[i] = d[i - 1] + (t[i] != t[i - 1])
            self.doc = d

    def blocks(self, pks, g):
        """Blocks touched by a set of chunks, in first-touch order."""
        if g == 0:
            return list(dict.fromkeys(pks))
        out = {}
        for p in pks:
            for b in range(int(self.off[p]) // g, (int(self.off[p + 1]) - 1) // g + 1):
                out[b] = None
        return list(out)

    def bsize(self, b, g):
        return int(self.L[b]) if g == 0 else g

    def expand(self, pks, w):
        if w == 0 or self.doc is None:
            return list(pks)
        out = []
        n = len(self.L)
        for p in pks:
            for q in range(max(0, p - w), min(n, p + w + 1)):
                if self.doc[q] == self.doc[p]:
                    out.append(q)
        return out


def lru_read(C, queries, g, cap):
    od, used, miss = collections.OrderedDict(), 0, 0
    for pks in queries:
        for b in C.blocks(pks, g):
            if b in od:
                od.move_to_end(b); continue
            s = C.bsize(b, g)
            miss += s
            od[b] = s; used += s
            while used > cap:
                used -= od.popitem(last=False)[1]
    return miss


def overlap(C, queries, g, n):
    """Share of block requests (bytes) in windows of n consecutive queries needed by >1 query."""
    tot = dup = 0
    for s in range(0, len(queries) - n + 1, n):
        cnt = collections.Counter()
        for pks in queries[s:s + n]:
            cnt.update(C.blocks(pks, g))
        for b, k in cnt.items():
            sz = C.bsize(b, g)
            tot += k * sz; dup += (k - 1) * sz
    return dup / tot if tot else 0.0


CORP = {}


def corpus(ds):
    if ds not in CORP:
        CORP[ds] = Corpus(ds)
    return CORP[ds]


def job(spec):
    kind, ds, path, *rest = spec
    C = corpus(ds)
    Q = load_case(path)
    if kind == "gran":
        budget, = rest
        cap = budget * C.total
        returned = sum(int(C.L[p]) for q in Q for p in q)
        return dict(kind=kind, ds=ds, case=Path(path).stem, budget=budget, returned=returned,
                    **{g: lru_read(C, Q, GRAN[g], cap) for g in GRAN})
    if kind == "expand":
        w, = rest
        E = [C.expand(q, w) for q in Q]
        no_merge = sum(int(C.L[p]) for q in E for p in q)
        merged = sum(int(C.L[p]) for q in E for p in set(q))
        base = sum(int(C.L[p]) for q in Q for p in q)
        r = dict(kind=kind, ds=ds, case=Path(path).stem, w=w, base=base, no_merge=no_merge, merged=merged)
        for budget in (0.2, 0.05):
            r[f"lru4k_b{budget}"] = lru_read(C, E, 4096, budget * C.total)
            r[f"lru2m_b{budget}"] = lru_read(C, E, 2 << 20, budget * C.total)
        return r
    if kind == "overlap":
        w, = rest
        E = [C.expand(q, w) for q in Q]
        return dict(kind=kind, ds=ds, case=Path(path).stem, w=w,
                    **{f"{g}_n{n}": overlap(C, E, GRAN[g], n) for g in ("chunk", "4KB", "2MB") for n in (8, 32, 128)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/context-sim")
    ap.add_argument("--procs", type=int, default=48)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    R = ROOT / "results"
    specs = []
    for ds, run, budgets in (("fiqa", "trend-fiqa", (0.5, 0.2, 0.1)), ("nq", "trend-nq", (0.5, 0.2))):
        for b in budgets:
            for t in ("natural", "zipf"):
                specs.append(("gran", ds, str(R / run / f"{ds}-C-b{b}-{t}-c32.jsonl"), b))
        for t in ("natural", "zipf"):
            for c in (32, 128):
                p = str(R / run / f"{ds}-C-b0.5-{t}-c{c}.jsonl")
                for w in ((0, 1, 2) if ds == "nq" else (0,)):
                    specs.append(("overlap", ds, p, w))
    for t in ("natural", "zipf"):
        for w in (0, 1, 2):
            specs.append(("expand", "nq", str(R / "trend-nq" / f"nq-C-b0.5-{t}-c32.jsonl"), w))
    corpus("fiqa"); corpus("nq")  # build once, shared with forked workers
    with Pool(min(a.procs, len(specs))) as pool:
        res = pool.map(job, specs, chunksize=1)
    (a.out / "context_sim.json").write_text(json.dumps(res, indent=1) + "\n")

    milvus = {}
    for run in ("trend-fiqa", "trend-nq"):
        for r in json.load(open(R / run / "analysis.json")):
            milvus[r["case"]] = r.get("milvus_text_miss")
    mb = lambda x: f"{x / 1e6:.1f}"
    L = ["# Context-read what-if (trace replay of measured Milvus results)", "",
         "Simulation on the recorded top-10 results, not a measurement. Blocks = contiguous byte ranges in pk order.", "",
         "## (1) Cache granularity: storage text read (MB) at the same budget, LRU, C=32", "",
         "| dataset | budget | trace | returned | chunk | 4 KB | 64 KB | 2 MB | Milvus measured | 2 MB / chunk |",
         "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in res:
        if r["kind"] != "gran":
            continue
        m = milvus.get(r["case"])
        L.append(f"| {r['ds']} | {r['budget']:.0%} | {r['case'].split('-')[3]} | {mb(r['returned'])} | {mb(r['chunk'])} | {mb(r['4KB'])} | "
                 f"{mb(r['64KB'])} | {mb(r['2MB'])} | {mb(m) if m is not None else '-'} | {r['2MB'] / max(r['chunk'], 1):.0f}x |")
    L += ["", "## (2) Neighbour-chunk expansion (NQ, +-w chunks of the same article, C=32)", "",
          "| trace | w | top-10 only MB | requested, no merge MB | after intra-query merge MB | merge saving | LRU 4 KB @20% | LRU 2 MB @20% | LRU 4 KB @5% | LRU 2 MB @5% |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in res:
        if r["kind"] != "expand":
            continue
        L.append(f"| {r['case'].split('-')[3]} | {r['w']} | {mb(r['base'])} | {mb(r['no_merge'])} | {mb(r['merged'])} | "
                 f"{1 - r['merged'] / r['no_merge']:.0%} | {mb(r['lru4k_b0.2'])} | {mb(r['lru2m_b0.2'])} | {mb(r['lru4k_b0.05'])} | {mb(r['lru2m_b0.05'])} |")
    L += ["", "## (3) Cross-query overlap: share of block bytes in a window of N consecutive requests also needed by another request in the window", "",
          "N ~ requests in flight (closed loop: C workers keep ~C in flight). This is the read volume a shared read could remove with no cache.", "",
          "| dataset | trace | C | w | chunk N=8 | chunk N=32 | chunk N=128 | 4 KB N=32 | 4 KB N=128 | 2 MB N=32 | 2 MB N=128 |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    pct = lambda x: f"{x:.1%}"
    for r in res:
        if r["kind"] != "overlap":
            continue
        f = r["case"].split("-")
        L.append(f"| {r['ds']} | {f[3]} | {f[4][1:]} | {r['w']} | {pct(r['chunk_n8'])} | {pct(r['chunk_n32'])} | {pct(r['chunk_n128'])} | "
                 f"{pct(r['4KB_n32'])} | {pct(r['4KB_n128'])} | {pct(r['2MB_n32'])} | {pct(r['2MB_n128'])} |")
    (a.out / "CONTEXT_SIM.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
