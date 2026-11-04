#!/usr/bin/env python3
"""Aggregate results/ctx-bench/rep*/ into CTX_BENCH.md (median over repetitions) + summary.json."""
import collections, json, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R = ROOT / "results/ctx-bench"
POL = ["cell2m", "chunk", "block", "group"]
KEYS = ["qps", "p50_ms", "p99_ms", "fetch_mean_ms", "fetch_p99_ms", "amplification", "reqs_per_query", "storage_bytes"]

runs = collections.defaultdict(list)
for f in sorted(R.glob("rep*/*.json")):
    s = json.loads(f.read_text())
    runs[(s["backend"], s["budget"], s["trace"], s["w"], s["policy"])].append(s)
med = {k: {m: statistics.median(x[m] for x in v) for m in KEYS} | {"n": len(v),
       "qps_min": min(x["qps"] for x in v), "qps_max": max(x["qps"] for x in v)} for k, v in runs.items()}

# Milvus measured at the same budget (NQ, C=32, text included / search only)
mil = {}
for run, cfg in (("trend-nq-low", "C"), ("trend-nq", "C"), ("trend-nq-mmap", "B")):
    for f in (ROOT / "results" / run).glob("nq-*-c32*.json"):
        s = json.loads(f.read_text())
        mil[f.stem] = s

L = ["# Same-budget context-layer comparison (NQ, 2,000 requests after 300 warmup, C=32, median of reps)", "",
     "Each request sleeps for the Milvus search latency measured without text (mmap, 5%, zipf), then fetches the text",
     "of its top-10 chunks (w=1: plus +-1 neighbour of the same article) through the layer. Cache budget = share of the",
     "1.33 GB text. local = NVMe with O_DIRECT; s3 = SeaweedFS ranged GETs (~33 ms each).", "",
     "cell2m = 2 MB cell LRU (Milvus tiered); chunk = chunk LRU, one read per missing chunk; block = 4 KB block LRU +",
     "intra-request range merge; group = block + coalesce gaps < 64 KB + share in-flight reads.", ""]
for be in ("local", "s3"):
    L += [f"## backend = {be}", "",
          "| budget | trace | w | layer | QPS | p50 ms | p99 ms | fetch mean ms | fetch p99 ms | read amp | storage reads/query | reps |",
          "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for b in (0.05, 0.2):
        for t in ("natural", "zipf"):
            for w in (0, 1):
                for p in POL:
                    m = med.get((be, b, t, w, p))
                    if not m:
                        continue
                    L.append(f"| {b:.0%} | {t} | {w} | {p} | {m['qps']:.0f} | {m['p50_ms']:.0f} | {m['p99_ms']:.0f} | "
                             f"{m['fetch_mean_ms']:.1f} | {m['fetch_p99_ms']:.0f} | {m['amplification']:.1f}x | {m['reqs_per_query']:.2f} | {m['n']} |")
    L.append("")

L += ["## Proposed (block / group) vs best simple layer (chunk), same condition", "",
      "| backend | budget | trace | w | best of block/group | QPS gain | p99 reduction | fetch-mean reduction | storage reads/query reduction |",
      "|---|---:|---|---:|---|---:|---:|---:|---:|"]
summary = []
for (be, b, t, w, p), m in sorted(med.items()):
    if p != "chunk":
        continue
    c = m
    best = min((med[k] for k in [(be, b, t, w, "block"), (be, b, t, w, "group")] if k in med), key=lambda x: x["p99_ms"], default=None)
    if best is None:
        continue
    name = "block" if med.get((be, b, t, w, "block")) is best else "group"
    row = dict(backend=be, budget=b, trace=t, w=w, best=name, qps_gain=best["qps"] / c["qps"], p99_ratio=c["p99_ms"] / best["p99_ms"],
               fetch_ratio=c["fetch_mean_ms"] / max(best["fetch_mean_ms"], 1e-9), reqs_ratio=c["reqs_per_query"] / max(best["reqs_per_query"], 1e-9))
    summary.append(row)
    L.append(f"| {be} | {b:.0%} | {t} | {w} | {name} | {row['qps_gain']:.2f}x | {row['p99_ratio']:.2f}x | {row['fetch_ratio']:.1f}x | {row['reqs_ratio']:.1f}x |")
L.append("")
L += ["## Milvus measured (NQ, C=32) for reference", "", "| case | QPS | p50 ms | p99 ms |", "|---|---:|---:|---:|"]
for k in sorted(mil):
    if "b0.05" in k or "b0.2" in k:
        s = mil[k]
        L.append(f"| {k} | {s['qps']:.0f} | {s['p50_ms']:.0f} | {s['p99_ms']:.0f} |")
(R / "CTX_BENCH.md").write_text("\n".join(L) + "\n")
(R / "summary.json").write_text(json.dumps(dict(median={"|".join(map(str, k)): v for k, v in med.items()}, vs_chunk=summary), indent=1) + "\n")
print("\n".join(L))
