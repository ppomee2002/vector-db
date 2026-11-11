"""M4 figures (M1 slide style): Qdrant pages/query by load method, Milvus MB/query and latency by layout."""
import json, os
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(B, "figs")
LIGHT, DARK, GREY = "#8fb4ee", "#2a6fdb", "#9aa0a6"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.8, "axes.axisbelow": True})
def bars(ax, labels, v0, v1, fmt):
    x = range(len(labels)); w = 0.36
    b0 = ax.bar([i - w/2 - 0.01 for i in x], v0, w, color=LIGHT, label="top-k only (±0)")
    b1 = ax.bar([i + w/2 + 0.01 for i in x], v1, w, color=DARK, label="top-k + neighbors (±1)")
    for bb, vals in ((b0, v0), (b1, v1)):
        for r, v in zip(bb, vals): ax.text(r.get_x() + r.get_width()/2, r.get_height(), fmt(v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels); ax.set_ylim(0, max(max(v0), max(v1)) * 1.22)
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=2)

# 1) Qdrant: data pages per query by load method (NQ, 150k chunks)
rows = {}
for f in ["results3/summary.jsonl", "results4/summary.jsonl"]:
    for l in open(os.path.join(B, f)):
        r = json.loads(l); rows[(r["layout"], r["window"])] = r
L = [("interleaved", "random\norder"), ("doc", "doc order\n(default)"), ("doc_b1", "doc order\n1-by-1"),
     ("doc_1seg_b16", "1 segment\nbatch 16"), ("doc_1seg_b1", "1 segment\n1-by-1")]
v0 = [rows[(k, 0)]["pages_pdata"] for k, _ in L]; v1 = [rows[(k, 1)]["pages_pdata"] for k, _ in L]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
bars(ax, [n for _, n in L], v0, v1, lambda v: f"{v:.1f}")
ax.axhline(6.72, color=GREY, ls="--", lw=1.2); ax.text(4.45, 6.72 + 0.3, "ideal contiguous (±1)", color="#5f6368", fontsize=8.5, ha="right")
ax.set_ylabel("payload pages / query (4KB)")
ax.set_title(f"Qdrant: pages per query by load method (NQ, 150k chunks)\n±1: {v1[1]:.1f} → {v1[-1]:.1f} pages ({(v1[-1]/v1[1]-1)*100:.0f}%) when doc order is kept on disk", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "m4_qdrant_pages.png"), dpi=200); plt.close(fig)

# 2) Milvus: MB fetched from object storage per query, and latency
R = os.path.join(B, "results_milvus")
LM = [("interleaved", "random order"), ("doc", "doc order\n(PK = doc order)"), ("doc_rpk", "doc order\n(random PK)")]
mb = {(k, w): pd.read_csv(f"{R}/mq_{k}_w{w}_cold_each.csv").miss_bytes_all.mean() / 1e6 for k, _ in LM for w in (0, 1)}
lat = {(k, w): pd.read_csv(f"{R}/mq_{k}_w{w}_cold_each.csv").lat_ms.median() for k, _ in LM for w in (0, 1)}
fig, ax = plt.subplots(figsize=(6.4, 4.2))
bars(ax, [n for _, n in LM], [mb[(k, 0)] for k, _ in LM], [mb[(k, 1)] for k, _ in LM], lambda v: f"{v:.1f}")
ax.set_ylabel("MB fetched from object storage / query")
ax.set_title(f"Milvus: data fetched per query (NQ, 150k chunks, cell ≈ 4 MB)\n±1: {mb[('interleaved',1)]:.0f} → {mb[('doc',1)]:.0f} MB ({(mb[('doc',1)]/mb[('interleaved',1)]-1)*100:.0f}%), needed text ≈ 10 KB", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "m4_milvus_bytes.png"), dpi=200); plt.close(fig)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
bars(ax, [n for _, n in LM], [lat[(k, 0)] for k, _ in LM], [lat[(k, 1)] for k, _ in LM], lambda v: f"{v:.0f}")
ax.set_ylabel("median latency / query (ms, cold cache)")
ax.set_title(f"Milvus: Context retrieval latency (NQ, 150k chunks)\n±1: {lat[('interleaved',1)]:.0f} → {lat[('doc',1)]:.0f} ms ({lat[('interleaved',1)]/lat[('doc',1)]:.1f}x faster) with doc-ordered PKs", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "m4_milvus_latency.png"), dpi=200); plt.close(fig)
print("saved", os.listdir(OUT))
