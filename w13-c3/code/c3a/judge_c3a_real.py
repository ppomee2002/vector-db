#!/usr/bin/env python3
"""C3 13주차 실제 질문(QSET=real) 탐색 결과 판정. analyze_c3a.py는 고치지 않고 따로 둔다.

usage: python judge_c3a_real.py --outdir /workspace/junlim/c3a_out_real

RULE (2026-11-22 측정 전에 고정, 결과를 본 뒤 바꾸지 않는다)
  설정(A/B/C)마다
    기준 k   = k=6, k=8 중 탐색 3회 평균 QPS가 높은 쪽 (analyze_c3a.py의 비교용 k와 같은 방식으로 고른다)
    재현     = 3회 모두 같은 회차 안에서 k=12의 QPS < 기준 k의 QPS
    재현 안 됨 = 한 회라도 k=12의 QPS >= 기준 k의 QPS
    판정 불가 = 회차가 3개가 아니거나, 해당 구간에 오류가 있거나 값이 비어 있음
  '재현'은 "fake 질문이 섞인 10,000개 집합에서 본 k=12 손해가 실제 질문만 써도 나타났다"는 뜻이다.
  통계적 입증이 아니며, 실제 질문 3,452개는 한 구간(20초) 안에서 여러 번 반복된다(Milvus 결과 캐시 여부 미확인).
"""
import argparse
import csv
import os
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ds", default="nq")
    a = ap.parse_args()
    path = f"{a.outdir}/results_c3a_load_{a.ds}.csv"
    if not os.path.exists(path):
        raise SystemExit(f"{path} 없음")
    with open(path, encoding="utf-8") as f:
        R = [r for r in csv.DictReader(f) if r["phase"] == "explore"]
    info_path = f"{a.outdir}/fixed_{a.ds}_info.json"
    qline = open(info_path, encoding="utf-8").read().strip() if os.path.exists(info_path) else "(fixed_info 없음)"

    q = defaultdict(dict)   # (setting, cap) -> {rep: qps}
    bad = defaultdict(list)
    for r in R:
        key = (r["setting"], int(r["cap"]))
        v = r.get("qps", "")
        if v in ("", None) or int(r.get("errors") or 0) > 0:
            bad[key].append(r["rep"])
            continue
        q[key][int(r["rep"])] = float(v)

    out = [f"# 실제 질문 탐색 판정 ({a.ds})", "", f"질문 정보: `{qline[:300]}`", "",
           "| 설정 | 기준 k (6/8 중 평균 높은 쪽) | 회차 | 기준 k QPS | k=12 QPS | k=12 < 기준 k | 판정 |",
           "|---|---|---|---|---|---|---|"]
    verdicts = {}
    for s in sorted({k[0] for k in q} | {k[0] for k in bad}):
        g6, g8, g12 = q.get((s, 6), {}), q.get((s, 8), {}), q.get((s, 12), {})
        problems = [f"k={k} 오류/빈 값 회차 {bad[(s, k)]}" for k in (6, 8, 12) if bad.get((s, k))]
        reps = sorted(set(g6) & set(g8) & set(g12))
        if problems or len(reps) != 3:
            verdicts[s] = "판정 불가"
            out.append(f"| {s} | - | {len(reps)} | - | - | - | 판정 불가 ({'; '.join(problems) or '회차 수 ≠ 3'}) |")
            continue
        m6 = sum(g6[r] for r in reps) / 3
        m8 = sum(g8[r] for r in reps) / 3
        kb, gb = (6, g6) if m6 >= m8 else (8, g8)
        lower = [g12[r] < gb[r] for r in reps]
        v = "재현" if all(lower) else "재현 안 됨"
        verdicts[s] = v
        out.append(f"| {s} | {kb} (k6 {m6:.0f} / k8 {m8:.0f}) | {len(reps)} | "
                   f"{', '.join(f'{gb[r]:.0f}' for r in reps)} | {', '.join(f'{g12[r]:.0f}' for r in reps)} | "
                   f"{sum(lower)}/3 | **{v}** |")
    out.append("")
    out.append("요약: " + ", ".join(f"{s} {v}" for s, v in sorted(verdicts.items())))
    out.append("규칙: 설정마다 k=6, 8 중 3회 평균 QPS가 높은 쪽을 기준 k로 두고, 3회 모두 같은 회차에서 k=12 < 기준 k이면 재현 (11/22 측정 전 고정).")
    text = "\n".join(out)
    print(text)
    open(f"{a.outdir}/summary_real_judge_{a.ds}.md", "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
