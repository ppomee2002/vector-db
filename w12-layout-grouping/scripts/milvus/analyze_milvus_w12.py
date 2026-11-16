import json
R = "/workspace/jihyeon/w12q/results/milvus"; L = ["rand", "doc", "P64", "P64x", "HN"]; S = {}
for l in open(f"{R}/summary.jsonl"): r = json.loads(l); S[(r["qs"], r["window"], r["layout"])] = r
cols = [("seen", 0), ("unseen", 0), ("seen", 1), ("unseen", 1)]; out = []
for key, title, f in [("miss_bytes_all", "쿼리당 저장소에서 읽은 양 (MB)", 1e-6), ("cell_loads", "쿼리당 cell 로드 수", 1), ("p50_ms", "쿼리당 지연 p50 (ms)", 1)]:
    out.append(f"\n== {title} =="); out.append(f"{'layout':8s}" + "".join(f"{'±%d %s' % (w, qs):>12s}" for qs, w in cols))
    for l in L: out.append(f"{l:8s}" + "".join(f"{S[(qs, w, l)][key] * f:12.2f}" if (qs, w, l) in S else f"{'-':>12s}" for qs, w in cols))
print("\n".join(out)); open(f"{R}/summary_milvus.txt", "w").write("\n".join(out) + "\n")
