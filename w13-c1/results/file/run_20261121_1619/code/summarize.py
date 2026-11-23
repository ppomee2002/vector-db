# summarize.py: run_c1.py 결과를 표로 정리  김지현 2024134018
# 사용: python summarize.py <결과 폴더>
# 출력: 같은 폴더에 summary.csv (조건별), summary.md (발표용 표)
# 값: 조건마다 회차별 값을 먼저 구하고, 회차 사이 중앙값을 쓴다. [최소~최대]는 회차 범위.
import os, sys, json
import numpy as np
import pandas as pd
from common import safe_path, open_read

NAMES = {"top10": "top-10", "bc1": "top-10 + B+C1 이웃", "all1": "top-10 ±1 전부", "ann30": "ANN 후보 30", "ann100": "ANN 후보 100"}


def per_rep(pq):
    g = pq.groupby(["exp", "layout", "merge", "gap", "read_set", "rep"])
    r = g.agg(n_queries=("q_idx", "size"), chunks_q=("n_chunks", "mean"), reads_q=("n_reads", "mean"),
              kb_read_q=("bytes_read", lambda x: x.mean() / 1024), kb_useful_q=("useful_bytes", lambda x: x.mean() / 1024),
              p50_us=("latency_us", "median"), p99_us=("latency_us", lambda x: np.percentile(x, 99)),
              mean_us=("latency_us", "mean")).reset_index()
    r["amplification"] = r.kb_read_q / r.kb_useful_q
    return r


def main():
    R = sys.argv[1]
    pq = pd.read_csv(os.path.join(R, "per_query.csv.gz"))
    pr = pd.read_csv(os.path.join(R, "per_run.csv"))
    r = per_rep(pq)
    keys = ["exp", "layout", "merge", "gap", "read_set"]
    cols = ["chunks_q", "reads_q", "kb_read_q", "kb_useful_q", "amplification", "p50_us", "p99_us", "mean_us"]
    s = r.groupby(keys)[cols].median().reset_index()
    for c in ("reads_q", "p50_us", "p99_us"):
        s[c + "_min"] = r.groupby(keys)[c].min().values
        s[c + "_max"] = r.groupby(keys)[c].max().values
    s["reps"] = r.groupby(keys).size().values
    if pr.disk_reads.notna().all() and (pr.app_reads > 0).all():
        pr["disk_ratio"] = pr.disk_reads / pr.app_reads
        s = s.merge(pr.groupby(keys).disk_ratio.median().reset_index(), on=keys, how="left")
    s.to_csv(safe_path(os.path.join(R, "summary.csv")), index=False)

    env = json.load(open_read(os.path.join(R, "env.json"), encoding="utf-8"))
    rs = json.load(open_read(os.path.join(R, "read_sets.json"), encoding="utf-8"))
    L = [f"# C1 실험 결과 ({os.path.basename(R)})", "",
         f"- 시작 {env.get('started_at_kst')} / 끝 {env.get('finished_at_kst')} (KST), 서버 {env.get('host')}, 장치 {env.get('device')}, O_DIRECT {env.get('o_direct')}",
         f"- 쿼리 {env.get('n_queries')}개, 조건별 {env.get('reps')}회, 단일 스레드, 4KB 블록. 값은 회차 중앙값, [ ]는 회차 범위",
         f"- 측정 전 장치 상태: {env.get('quiet_check_before')}", ""]
    if "disk_ratio" in s:
        L.append(f"- 디스크 읽기 / 앱 읽기 비율: {s.disk_ratio.min():.3f} ~ {s.disk_ratio.max():.3f} (1.0 근처면 다른 사용자 영향 없음)")
        L.append("")
    L += ["## 읽는 범위", "", "| 읽는 범위 | 쿼리당 chunk | 정답 chunk 포함률 |", "|---|---|---|"]
    for k, v in rs.items():
        L.append(f"| {NAMES[k]} | {v['mean_chunks_per_query']} | {v.get('gold_chunk_hit_rate', '-')} |")
    L.append("")
    L.append("정답 chunk 포함률은 qrels_chunk.tsv(정답 문단 기준)로 계산했다. 준이 C2 발표 숫자(정답 문자열 기준 2,254개)와는 기준이 다르다.")
    L.append("")

    def fmt(x, lo, hi, d=2):
        return f"{x:.{d}f} [{lo:.{d}f}~{hi:.{d}f}]"

    def table(df, title, base_note=""):
        L.extend([f"## {title}", ""])
        if base_note:
            L.extend([base_note, ""])
        L.append("| 배치 | 읽기 | gap | 읽는 범위 | 읽기 횟수/쿼리 | 읽은 KB/쿼리 | 필요한 KB/쿼리 | 증폭 | p50 us | p99 us | rand 개별 대비 읽기 횟수 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for _, x in df.iterrows():
            base = s[(s.layout == "rand") & (s["merge"] == 0) & (s.gap == 0) & (s.read_set == x.read_set)]
            rel = f"{x.reads_q / base.reads_q.iloc[0]:.2f}x" if len(base) else "-"
            L.append(f"| {x.layout} | {'묶어 읽기' if x['merge'] else '개별'} | {int(x.gap) // 1024}KB | {NAMES[x.read_set]} | "
                     f"{fmt(x.reads_q, x.reads_q_min, x.reads_q_max)} | {x.kb_read_q:.1f} | {x.kb_useful_q:.1f} | {x.amplification:.2f} | "
                     f"{fmt(x.p50_us, x.p50_us_min, x.p50_us_max, 0)} | {fmt(x.p99_us, x.p99_us_min, x.p99_us_max, 0)} | {rel} |")
        L.append("")

    order = {"top10": 0, "bc1": 1, "all1": 2, "ann30": 3, "ann100": 4}
    srt = lambda d: d.assign(_o=d.read_set.map(order)).sort_values(["_o", "layout", "merge", "gap"], ascending=[True, False, True, True])
    if (s.exp == "E1").any():
        table(srt(s[s.exp == "E1"]), "E1 배치 x 읽기 방식")
    if (s.exp == "E2").any():
        e2 = pd.concat([s[(s.exp == "E1") & (s.layout == "doc") & (s["merge"] == 1) & s.read_set.isin(["bc1", "all1"])], s[s.exp == "E2"]])
        table(srt(e2), "E2 병합 간격 (doc 배치, 묶어 읽기)", "gap 0 줄은 E1 값이다.")
    if (s.exp == "E3").any():
        table(srt(s[s.exp == "E3"]), "E3 Rerank 전 후보 읽기")
    with open(safe_path(os.path.join(R, "summary.md")), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
