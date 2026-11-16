import pandas as pd, sys
pd.set_option("display.width", 200); R = "/workspace/jihyeon/results"
q = pd.read_csv(f"{R}/per_query3.csv"); w = int(sys.argv[1]); order = ["D", "K", "P64", "P64w", "HC", "HN"]
for qs in ("seen", "unseen"):
    e = q[(q.neighbor == w) & (q.query_set == qs)]
    for col, fn, title in [("n_reads", "mean", "read 횟수 평균"), ("latency_us", "median", "지연 p50 (us)")]:
        t = e.pivot_table(index="layout", columns="k", values=col, aggfunc=fn).loc[order]
        print(f"\n== ±{w}, {qs}: {title} (열 = k) ==\n" + t.to_string(float_format=lambda x: f"{x:.1f}"))
