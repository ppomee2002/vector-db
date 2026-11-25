#!/usr/bin/env python3
"""C3 13주차 결과 정리와 판정. 판정 규칙은 측정 전에 고정했다(아래 RULES). 측정 결과를 보고 바꾸지 않는다.

usage
  python analyze_c3a.py explore --outdir $ROOT/c3a_out   # 탐색 결과 표 + 비교용 k를 ref_k_<ds>.json에 고정 (이미 있으면 덮어쓰지 않음)
  python analyze_c3a.py confirm --outdir $ROOT/c3a_out   # 확인 결과로 설정별 판정
  python analyze_c3a.py seq     --outdir $ROOT/c3a_out   # 요청 하나씩: 설정 간 검색 시간 차이(같은 질문끼리 짝지음)

RULES (2026-11-21 확정)
  비교용 k  = 탐색(3회)에서 평균 QPS가 가장 높은 k (후보 6, 8, 12, 16). 확인 세션 동안 바꾸지 않는다.
             비교용 k가 12면 그 설정은 확인하지 않는다: "탐색에서 k=12가 최고".
  기준 설정 = 설정 A, k=8 (지난주 m7). 확인 세션 처음과 끝에 한 번씩 재서 그 차이를 '같은 설정을 다시 잰 차이'로 쓴다.
  진행      = 다음을 모두 만족
              (1) 확인 4회 모두 같은 회차 안에서 비교용 k의 QPS > k=12의 QPS
              (2) QPS 평균 차이(비교용 k − k=12) > 기준 설정 QPS 차이(|끝 − 처음|)
              (3) p99 평균 차이(비교용 k − k=12) <= 기준 설정 p99 차이(|끝 − 처음|)
  (1)(2)는 만족하고 (3)을 못 만족하면 "맞바꿈"(처리량은 앞서지만 p99 악화)으로 따로 보고한다.
  (3)을 만족해도 "p99 악화가 확인되지 않음"으로만 적고, 두 설정의 p99가 같다고 하지 않는다.
  이 판정은 새 방식을 시도할지 정하는 내부 기준이다. 통계적 입증이나 새 방식의 효과 입증으로 쓰지 않는다.
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

K_CANDIDATES = (6, 8, 12, 16)
ANCHOR = ("A", 8)


def rows(path):
    if not os.path.exists(path):
        raise SystemExit(f"{path} 없음")
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else None


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def explore(a):
    R = [r for r in rows(f"{a.outdir}/results_c3a_load_{a.ds}.csv") if r["phase"] == "explore" and num(r, "qps")]
    by = defaultdict(list)
    for r in R:
        by[(r["setting"], int(r["cap"]))].append(r)
    out, ref = [], {}
    out.append(f"# 탐색 결과 ({a.ds}, 동시 요청 {R[0]['n'] if R else '?'})\n")
    out.append("| 설정 | k | 회차 | QPS 평균 | QPS 회차별 | p99 평균(ms) | 상한 대기 평균(ms) | 검색 평균(ms) | 조회 평균(ms) | top10 같음 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for s in sorted({k[0] for k in by}):
        best = None
        for k in K_CANDIDATES:
            g = by.get((s, k), [])
            if not g:
                continue
            q = [num(r, "qps") for r in g]; mq = mean(q)
            out.append(f"| {s} | {k} | {len(g)} | {mq:.0f} | {', '.join(f'{x:.0f}' for x in q)} | "
                       f"{mean([num(r, 'p99_ms') for r in g]):.0f} | {mean([num(r, 'wait_ms_mean') for r in g]):.2f} | "
                       f"{mean([num(r, 'search_ms_mean') for r in g]):.2f} | {mean([num(r, 'fetch_ms_mean') for r in g]):.2f} | "
                       f"{mean([num(r, 'top10_same') for r in g]):.3f} |")
            if best is None or mq > best[1]:
                best = (k, mq)
        if best:
            ref[s] = best[0]
            out.append(f"| {s} | **비교용 k = {best[0]}** | | | | | | | | |")
    errs = sum(int(r["errors"] or 0) for r in R)
    out.append(f"\n오류 합계 {errs}. 비교용 k = 평균 QPS가 가장 높은 k (사전 고정 규칙).")
    text = "\n".join(out)
    print(text)
    open(f"{a.outdir}/summary_explore_{a.ds}.md", "w", encoding="utf-8").write(text + "\n")
    path = f"{a.outdir}/ref_k_{a.ds}.json"
    if os.path.exists(path):
        old = json.load(open(path))
        print(f"\n{path}가 이미 있어 바꾸지 않음: {old}" + ("" if old == ref else f"  (이번 계산 {ref}와 다름: 확인 전에 정한 값을 유지)"))
    else:
        json.dump(ref, open(path, "w")); print(f"\n비교용 k 고정 -> {path}: {ref}")


def confirm(a):
    ref = json.load(open(f"{a.outdir}/ref_k_{a.ds}.json"))
    R = [r for r in rows(f"{a.outdir}/results_c3a_load_{a.ds}.csv") if r["phase"] in ("confirm", "anchor")]
    anc = sorted([r for r in R if r["phase"] == "anchor" and num(r, "qps")], key=lambda r: r["ts_utc"])
    out = [f"# 확인 결과 ({a.ds})\n", f"비교용 k (탐색 후 고정): {ref}\n"]
    if len(anc) >= 2:
        a0, a1 = anc[0], anc[-1]
        dq_anc = abs(num(a1, "qps") - num(a0, "qps")); dp_anc = abs(num(a1, "p99_ms") - num(a0, "p99_ms"))
        out.append(f"기준 설정(A, k=8) 처음 {num(a0, 'qps'):.0f} QPS / p99 {num(a0, 'p99_ms'):.0f}ms, "
                   f"끝 {num(a1, 'qps'):.0f} / {num(a1, 'p99_ms'):.0f}ms -> 다시 잰 차이 QPS {dq_anc:.1f}, p99 {dp_anc:.1f}ms\n")
    else:
        dq_anc = dp_anc = None
        out.append("기준 설정 측정이 2개 미만이라 (2)(3)을 판정할 수 없음\n")
    conf = [r for r in R if r["phase"] == "confirm" and num(r, "qps")]
    out.append("| 설정 | 비교용 k | 회차 | 비교용 k QPS | k=12 QPS | 비교용 k가 높은 회차 | QPS 평균 차이 | p99 평균 차이(ms) | 판정 |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for s in sorted(ref):
        k = int(ref[s])
        if k == 12:
            out.append(f"| {s} | 12 | - | - | - | - | - | - | 탐색에서 k=12가 최고 (확인 없음) |"); continue
        pr = defaultdict(dict)
        for r in conf:
            if r["setting"] == s and int(r["cap"]) in (k, 12):
                pr[int(r["rep"])][int(r["cap"])] = r
        pairs = [(v[k], v[12]) for _, v in sorted(pr.items()) if k in v and 12 in v]
        if not pairs:
            out.append(f"| {s} | {k} | 0 | | | | | | 확인 측정 없음 |"); continue
        dq = [num(x, "qps") - num(y, "qps") for x, y in pairs]
        dp = [num(x, "p99_ms") - num(y, "p99_ms") for x, y in pairs]
        wins = sum(d > 0 for d in dq)
        c1 = wins == len(pairs) and len(pairs) >= a.reps
        c2 = dq_anc is not None and mean(dq) > dq_anc
        c3 = dp_anc is not None and mean(dp) <= dp_anc
        if c1 and c2 and c3:
            v = "진행: k=12 부족 (p99 악화는 확인되지 않음)"
        elif c1 and c2:
            v = "맞바꿈: 처리량은 앞서나 p99 악화"
        else:
            why = [w for w, c in (("(1) 회차 일관성", c1), ("(2) 다시 잰 차이보다 큼", c2)) if not c]
            v = "k=12 부족이 확인되지 않음 — 미충족: " + ", ".join(why)
        q12 = mean([num(y, "qps") for _, y in pairs])
        out.append(f"| {s} | {k} | {len(pairs)} | {mean([num(x, 'qps') for x, _ in pairs]):.0f} | {q12:.0f} | "
                   f"{wins}/{len(pairs)} | {mean(dq):+.1f} ({mean(dq) / q12 * 100:+.1f}%) | {mean(dp):+.1f} | {v} |")
    out.append("\n이 판정은 새 방식을 시도할지 정하는 내부 기준이다. 통계적 입증이나 새 방식의 효과 입증이 아니다.")
    text = "\n".join(out)
    print(text)
    open(f"{a.outdir}/summary_confirm_{a.ds}.md", "w", encoding="utf-8").write(text + "\n")


def seq(a):
    R = rows(f"{a.outdir}/results_c3a_seq_raw_{a.ds}.csv")
    d = {(r["rep"], r["i"], r["setting"]): r for r in R}
    keys = {(r["rep"], r["i"]) for r in R}
    out = [f"# 요청 하나씩: 검색 시간 (같은 질문끼리 짝지은 차이, {a.ds})\n", "| 비교 | 질문 수 | 검색 시간 차이 평균(ms) | 95% 구간 | 조회 시간 차이 평균(ms) |", "|---|---|---|---|---|"]
    for x, y in (("B", "A"), ("C", "B"), ("C", "A")):
        ds, df = [], []
        for k in keys:
            if (k[0], k[1], x) in d and (k[0], k[1], y) in d:
                ds.append(float(d[(k[0], k[1], x)]["search_ms"]) - float(d[(k[0], k[1], y)]["search_ms"]))
                df.append(float(d[(k[0], k[1], x)]["fetch_ms"]) - float(d[(k[0], k[1], y)]["fetch_ms"]))
        if len(ds) > 1:
            m = mean(ds); sd = math.sqrt(sum((v - m) ** 2 for v in ds) / (len(ds) - 1))
            out.append(f"| {x} − {y} | {len(ds)} | {m:+.3f} | ±{1.96 * sd / math.sqrt(len(ds)):.3f} | {mean(df):+.3f} |")
    for s in ("A", "B", "C"):
        g = [r for r in R if r["setting"] == s]
        if g:
            out.append(f"\n설정 {s}: 검색 평균 {mean([float(r['search_ms']) for r in g]):.3f}ms, 조회 평균 "
                       f"{mean([float(r['fetch_ms']) for r in g]):.3f}ms, top-10 같음 {mean([float(r['top10_same']) for r in g]):.3f}")
    text = "\n".join(out)
    print(text)
    open(f"{a.outdir}/summary_seq_{a.ds}.md", "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["explore", "confirm", "seq"])
    ap.add_argument("--outdir", default="/workspace/junlim/c3a_out"); ap.add_argument("--ds", default="nq")
    ap.add_argument("--reps", type=int, default=4, help="확인 회차 수 (판정 (1)에 필요한 최소 회차)")
    a = ap.parse_args()
    {"explore": explore, "confirm": confirm, "seq": seq}[a.what](a)
