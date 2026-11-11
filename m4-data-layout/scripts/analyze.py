"""Step 4: 결과 요약표 + 그래프"""
import json, os, sys
R = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/m4-exp/results")
rows = [json.loads(l) for l in open(f"{R}/summary.jsonl")]
hdr = f"{'layout':12s}{'w':>3s}{'mode':>11s}{'lat_ms':>9s}{'p95_ms':>9s}{'read_KB':>10s}{'need_KB':>9s}{'amp':>7s}{'pages':>8s}{'majflt':>8s}{'QPS':>8s}"
out = [hdr]
for r in sorted(rows, key=lambda r: (r["mode"], r["window"], r["layout"])):
    amp = r["read_bytes"] / r["need_bytes"] if r["need_bytes"] else 0
    out.append(f"{r['layout']:12s}{r['window']:3d}{r['mode']:>11s}{r['lat_ms']:9.2f}{r['p95_ms']:9.2f}{r['read_bytes']/1024:10.1f}"
               f"{r['need_bytes']/1024:9.1f}{amp:7.1f}{r['pages_payload']:8.1f}{r['majflt']:8.1f}{r.get('qps', float('nan')):8.1f}")
txt = "\n".join(out); print(txt); open(f"{R}/summary.txt", "w").write(txt + "\n")
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cold = [r for r in rows if r["mode"] == "cold_each"]
    L = ["doc", "interleaved", "coaccess"]; W = sorted({r["window"] for r in cold})
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for i, (key, lab) in enumerate([("read_bytes", "read KB / query"), ("pages_payload", "payload pages / query"), ("lat_ms", "latency ms / query")]):
        for j, w in enumerate(W):
            v = [next((r[key] for r in cold if r["layout"] == l and r["window"] == w), 0) for l in L]
            v = [x / 1024 for x in v] if key == "read_bytes" else v
            ax[i].bar([k + j * 0.4 for k in range(len(L))], v, 0.4, label=f"window ±{w}")
        ax[i].set_xticks([k + 0.2 for k in range(len(L))]); ax[i].set_xticklabels(L); ax[i].set_title(lab); ax[i].legend()
    plt.tight_layout(); plt.savefig(f"{R}/m4_cold.png", dpi=150); print("saved m4_cold.png")
except Exception as e: print("plot skipped:", e)
