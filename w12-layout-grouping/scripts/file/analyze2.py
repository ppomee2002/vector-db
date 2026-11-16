import pandas as pd, sys
pd.set_option("display.width", 200); R = "/workspace/jihyeon/results"
q = pd.read_csv(f"{R}/per_query2.csv"); q["kb"] = q.bytes_read / 1024
e = q[(q.gap == int(sys.argv[1])) & (q.k == 10)]; order = ["D", "R", "K", "L", "P16", "P64", "P256", "P64s"]
for col, fn, title in [("n_reads", "mean", "read 횟수 평균"), ("latency_us", "median", "지연 p50 (us)"), ("kb", "mean", "읽은 양 KB 평균")]:
    t = e.pivot_table(index="layout", columns=["neighbor", "query_set"], values=col, aggfunc=fn).loc[order]
    print(f"\n== k=10, gap={sys.argv[1]}: {title} ==\n" + t.to_string(float_format=lambda x: f"{x:.1f}"))
