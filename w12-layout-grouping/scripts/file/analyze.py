import pandas as pd, sys, io
pd.set_option("display.width", 200)
R = "/workspace/jihyeon/results"
q = pd.read_csv(f"{R}/per_query.csv"); r = pd.read_csv(f"{R}/per_run.csv")
q["kb"] = q.bytes_read / 1024; e = q[q.exp == "exp1"]; f2 = lambda x: f"{x:.2f}"; f0 = lambda x: f"{x:.0f}"
def tab(col, fn, title, fmt):
    t = e.pivot_table(index=["k", "neighbor", "query_set"], columns="layout", values=col, aggfunc=fn)[list("DRKL")]
    return f"\n== {title} ==\n" + t.to_string(float_format=fmt)
p1 = [tab("n_reads", "mean", "실험1: 쿼리당 read 횟수 (평균)", f2),
      tab("latency_us", "median", "실험1: 쿼리당 지연 p50 (us)", f0)]
d = q[q.layout == "D"].copy()
d["읽기"] = d.apply(lambda x: "±0 묶음" if x.neighbor == 0 else ("±1 개별" if x["mode"] == "indiv" else "±1 묶음"), axis=1)
g = d.groupby(["k", "query_set", "읽기"]).agg(passages=("n_passages", "mean"), reads=("n_reads", "mean"), KB=("kb", "mean"),
      p50_us=("latency_us", "median"), p99_us=("latency_us", lambda s: s.quantile(.99)))
r["rr"] = r.disk_reads / r.app_reads; r["br"] = r.disk_bytes / r.app_bytes
idle = pd.read_csv(f"{R}/idle_baseline.csv").iloc[0]
io_txt = (f"\n== iostat 대조 (디스크/앱 비율, 1.00이면 일치) ==\n"
          f"전체 read 횟수: 앱 {r.app_reads.sum():,} / 디스크 {r.disk_reads.sum():,} = {r.disk_reads.sum()/r.app_reads.sum():.3f}\n"
          f"전체 바이트:    앱 {r.app_bytes.sum():,} / 디스크 {r.disk_bytes.sum():,} = {r.disk_bytes.sum()/r.app_bytes.sum():.3f}\n"
          f"run별 횟수 비율 min/중앙/max: {r.rr.min():.3f} / {r.rr.median():.3f} / {r.rr.max():.3f}   바이트 비율: {r.br.min():.3f} / {r.br.median():.3f} / {r.br.max():.3f}\n"
          f"idle 30초 동안 다른 읽기: {idle.disk_reads}회, {idle.disk_bytes} bytes")
inlog = e[e.k == 10].groupby("query_set").frac_in_log.mean()
p2 = ["\n== 실험2: 문서 순(D) 배치에서 이웃 읽기 ==\n" + g.to_string(float_format=f2), io_txt,
      f"\ntop-10 중 로그에 있던 passage 비율: seen {inlog['seen']:.1%}, unseen {inlog['unseen']:.1%}"]
p3 = [tab("kb", "mean", "실험1: 쿼리당 읽은 양 (KB, 평균)", f2), tab("latency_us", lambda s: s.quantile(.99), "실험1: 지연 p99 (us)", f0)]
open(f"{R}/summary.txt", "w").write("\n".join(p1 + p2 + p3) + "\n")
print("\n".join({"1": p1, "2": p2, "3": p3}[sys.argv[1]]))
