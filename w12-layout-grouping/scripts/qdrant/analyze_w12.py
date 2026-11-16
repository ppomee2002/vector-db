import json
W = "/workspace/jihyeon/w12q/results"; L = ["w12_rand", "w12_doc", "w12_P64", "w12_P64x", "w12_HN"]; R = {}
for qs in ("seen", "unseen"):
    for l in open(f"{W}/{qs}/summary.jsonl"):
        r = json.loads(l); R[(qs, r["window"], r["layout"])] = r
cols = [("seen", 0), ("unseen", 0), ("seen", 1), ("unseen", 1)]; out = []
for key, title, f in [("pages_pdata", "쿼리당 데이터 페이지 수", 1), ("read_bytes", "쿼리당 읽은 양 (KB)", 1 / 1024), ("lat_ms", "쿼리당 지연 평균 (ms)", 1), ("n_ctx", "읽은 chunk 수", 1)]:
    out.append(f"\n== {title} =="); out.append(f"{'layout':8s}" + "".join(f"{'±%d %s' % (w, qs):>12s}" for qs, w in cols))
    for l in L: out.append(f"{l[4:]:8s}" + "".join(f"{R[(qs, w, l)][key] * f:12.2f}" if (qs, w, l) in R else f"{'-':>12s}" for qs, w in cols))
print("\n".join(out)); open(f"{W}/summary_w12.txt", "w").write("\n".join(out) + "\n")
