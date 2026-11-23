# summarize_milvus.py: c1_milvus.py 결과를 표로 정리  김지현 2024134018
# 사용: python summarize_milvus.py <m1_cold_... 또는 m2_warm_... 폴더>
# 출력: 같은 폴더에 summary.md, summary.csv
import os, sys, json
import numpy as np
import pandas as pd
from common import safe_path, open_read

NAMES = {"top10": "top-10", "bc1": "top-10 + B+C1 이웃", "all1": "top-10 ±1 전부", "ann30": "ANN 후보 30", "ann100": "ANN 후보 100"}
MODES = {"one": "한 번에", "split": "따로 (top-10 → 이웃)", "each": "Chunk마다 따로"}
ORDER = {k: i for i, k in enumerate(NAMES)}


def boot_ratio(a, b, n=2000, seed=0):
    """짝지은 쿼리별 값 a(doc), b(rand)에서 평균 비율 mean(a)/mean(b)의 95% bootstrap 구간."""
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(a), (n, len(a)))
    r = a[idx].mean(1) / np.maximum(b[idx].mean(1), 1e-9)
    return a.mean() / max(b.mean(), 1e-9), np.percentile(r, 2.5), np.percentile(r, 97.5)


def m1(R, L):
    d = pd.read_csv(os.path.join(R, "per_query.csv"))
    d["miss_MB"] = d.miss_bytes / 1e6
    g = d.groupby(["layout", "read_set", "mode"])
    s = g.agg(n=("q_idx", "size"), chunks=("n_chunks", "mean"), requests=("n_requests", "mean"), miss_MB=("miss_MB", "mean"),
              miss_MB_p50=("miss_MB", "median"), cell_loads=("cell_loads", "mean"), storage_ops=("storage_ops", "mean"),
              lat_p50=("lat_ms", "median"), lat_mean=("lat_ms", "mean"), lat_p99=("lat_ms", lambda x: np.percentile(x, 99)),
              ok=("ok", "mean")).reset_index()
    s = s.assign(_o=s.read_set.map(ORDER)).sort_values(["_o", "mode", "layout"], ascending=[True, True, False]).drop(columns="_o")
    s.to_csv(safe_path(os.path.join(R, "summary.csv")), index=False)
    L += ["## M1 콜드 조회 (쿼리마다 release/load)", "",
          "| 읽는 범위 | 요청 방식 | 배치 | 쿼리당 Chunk | 요청 수 | 저장소에서 읽은 MB (평균) | cell 로드 | 지연 p50 ms | 지연 p99 ms | 본문 일치 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, x in s.iterrows():
        L.append(f"| {NAMES[x.read_set]} | {MODES[x['mode']]} | {x.layout} | {x.chunks:.1f} | {x.requests:.1f} | {x.miss_MB:.2f} | "
                 f"{x.cell_loads:.1f} | {x.lat_p50:.1f} | {x.lat_p99:.1f} | {x.ok:.0%} |")
    L += ["", "### doc ÷ rand (같은 쿼리끼리 짝지어 비교, 한 번에 요청)", "",
          "| 읽는 범위 | 읽은 MB 비율 [95% 구간] | 지연 비율 [95% 구간] |", "|---|---|---|"]
    one = d[d["mode"] == "one"]
    for rs in NAMES:
        a = one[(one.layout == "doc") & (one.read_set == rs)].sort_values("q_idx")
        b = one[(one.layout == "rand") & (one.read_set == rs)].sort_values("q_idx")
        if len(a) and len(a) == len(b):
            m = boot_ratio(a.miss_MB.values, b.miss_MB.values); t = boot_ratio(a.lat_ms.values, b.lat_ms.values)
            L.append(f"| {NAMES[rs]} | {m[0]:.2f} [{m[1]:.2f}~{m[2]:.2f}] | {t[0]:.2f} [{t[1]:.2f}~{t[2]:.2f}] |")
    L += ["", "### 따로 ÷ 한 번에 (같은 배치, 같은 쿼리)", "", "| 읽는 범위 | 배치 | 읽은 MB 비율 | 지연 비율 |", "|---|---|---|---|"]
    for rs in ("bc1", "all1"):
        for lay in ("doc", "rand"):
            a = d[(d.layout == lay) & (d.read_set == rs) & (d["mode"] == "split")].sort_values("q_idx")
            b = d[(d.layout == lay) & (d.read_set == rs) & (d["mode"] == "one")].sort_values("q_idx")
            if len(a) and len(a) == len(b):
                L.append(f"| {NAMES[rs]} | {lay} | {a.miss_MB.mean() / max(b.miss_MB.mean(), 1e-9):.2f} | {a.lat_ms.mean() / b.lat_ms.mean():.2f} |")
    L += ["", "저장소에서 읽은 MB는 Milvus 메트릭 internal_cache_cell_access_miss_bytes_total(서버 전체 값)의 조회 전후 차이다. "
          "다른 사람이 같은 시간에 Milvus를 쓰면 섞인다.", ""]


def m2(R, L):
    q = pd.read_csv(os.path.join(R, "per_query.csv")); r = pd.read_csv(os.path.join(R, "per_run.csv"))
    pr = q.groupby(["rep", "layout", "read_set", "mode"]).agg(requests=("n_requests", "mean"), p50=("lat_ms", "median"),
                                                               p99=("lat_ms", lambda x: np.percentile(x, 99)), ok=("ok", "mean")).reset_index()
    keys = ["layout", "read_set", "mode"]
    s = pr.groupby(keys).agg(requests=("requests", "median"), p50=("p50", "median"), p50_min=("p50", "min"), p50_max=("p50", "max"),
                             p99=("p99", "median"), ok=("ok", "min")).reset_index()
    s = s.merge(r.groupby(keys).agg(miss_MB_run=("miss_bytes", lambda x: x.median() / 1e6)).reset_index(), on=keys)
    s = s.assign(_o=s.read_set.map(ORDER)).sort_values(["_o", "mode", "layout"], ascending=[True, True, False]).drop(columns="_o")
    s.to_csv(safe_path(os.path.join(R, "summary.csv")), index=False)
    L += ["## M2 웜 조회 (한 번 load, 예열 후 측정)", "",
          "| 읽는 범위 | 요청 방식 | 배치 | 요청 수/쿼리 | 지연 p50 ms [회차 범위] | 지연 p99 ms | 실행당 저장소 읽기 MB | 본문 일치 |",
          "|---|---|---|---|---|---|---|---|"]
    for _, x in s.iterrows():
        L.append(f"| {NAMES[x.read_set]} | {MODES[x['mode']]} | {x.layout} | {x.requests:.1f} | {x.p50:.2f} [{x.p50_min:.2f}~{x.p50_max:.2f}] | "
                 f"{x.p99:.2f} | {x.miss_MB_run:.2f} | {x.ok:.0%} |")
    L += ["", "예열 뒤라 저장소 읽기는 0에 가까워야 한다. 0이 아니면 캐시에서 밀려난 것이다.", ""]


def main():
    R = sys.argv[1]
    env = json.load(open_read(os.path.join(R, "env.json"), encoding="utf-8"))
    rs = json.load(open_read(os.path.join(R, "read_sets.json"), encoding="utf-8"))
    L = [f"# C1 Milvus 실험 결과 ({os.path.basename(R)})", "",
         f"- {env.get('kind')}, database {env.get('db')}, 쿼리 {env.get('n_queries')}개" + (f", {env.get('reps')}회" if env.get("reps") else ""),
         f"- 측정 전 Milvus 상태: {env.get('quiet_check_before')}" + (f" / 측정 후: {env.get('quiet_check_after')}" if env.get("quiet_check_after") else ""),
         "- 벡터는 8차원 가짜 벡터(FLAT)다. 검색 결과는 재은이 ANN·Rerank 파일을 쓰고, 본문 조회만 잰다.", "",
         "| 읽는 범위 | 쿼리당 Chunk | 정답 chunk 포함률 |", "|---|---|---|"]
    for k, v in rs.items():
        L.append(f"| {NAMES[k]} | {v['mean_chunks_per_query']} | {v.get('gold_chunk_hit_rate', '-')} |")
    L.append("")
    (m1 if os.path.basename(R).startswith("m1") else m2)(R, L)
    with open(safe_path(os.path.join(R, "summary.md")), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
