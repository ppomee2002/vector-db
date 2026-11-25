#!/usr/bin/env python3
"""C3 13주차 실제 질문(QSET=real) 탐색 재측정 판정 (run_c3a_real2.sh 결과용). judge_c3a_real.py는 고치지 않고 따로 둔다.

usage: python judge_c3a_real2.py --outdir /workspace/junlim/c3a_out_real2 --waitlog c3a_real2_wait.log --runlog c3a_real2.log

RULE (2026-11-23 측정 전에 고정, 결과를 본 뒤 바꾸지 않는다)
  1. 세션 유효 조건 (모두 만족해야 판정에 쓴다)
     V1 wait 로그의 OVERLAP 기록에 'python'이 들어간 다른 계정 프로세스가 없다.
        (wait_run_pf.sh가 측정 시작 조건으로 보는 것과 같은 기준. claude·셸처럼 python이 아닌 새 프로세스는 기록만 한다)
     V2 wait 로그와 실행 로그에 'Resource temporarily unavailable'(fork 실패)이 없다. 측정 클라이언트가 같은 컨테이너에서 돌기 때문.
     V3 기준 설정(A, k=8) 4번(1회차 앞, 각 회차 뒤)이 모두 있고, 회차마다 앞뒤 기준 설정의 QPS 차이가
        |뒤 − 앞| / max(앞, 뒤) <= 0.15 이다.
        0.15의 근거: 강한 근거는 없다. 1544 탐색에서 A k=8 3회 범위가 5.4%(616~651)였고, 확인하려는 k=12 손해가 약 30% 이상이었다.
        그 사이에서 '확인하려는 차이의 절반을 넘는 상태 변화가 한 회차 안에 있으면 그 회차 비교를 믿지 않는다'로 정했다.
     V4 탐색 36구간이 모두 있고 오류가 0이다.
     하나라도 어기면 "판정 불가(사유)"로 기록하고 판정에 쓰지 않는다. 판정 불가 세션도 결과는 모두 보고한다.
     재측정을 여러 번 하면 처음으로 유효한 세션 하나로 판정한다 (여러 세션 중 고르지 않는다).
  2. 판정 (유효한 세션만, judge_c3a_real.py와 같은 규칙)
     설정마다 k=6, 8 중 3회 평균 QPS가 높은 쪽을 기준 k로 두고, 3회 모두 같은 회차 안에서 k=12 QPS < 기준 k QPS이면 "재현", 아니면 "재현 안 됨".
  3. 상태 표시 (판정을 바꾸지 않음)
     기준 설정 4번의 평균이 631(1544 탐색 A k=8 평균) × 0.85 = 536 미만이면 "느린 상태에서 측정"으로 표시한다.
     이 경우 결과를 1544(질문 전체 집합) 결과와 같은 조건으로 비교하지 않는다. 0.85도 강한 근거는 없다(위 15%와 맞춤).
  이 판정은 내부 기준이며 통계적 입증이 아니다. 실제 질문 3,452개는 한 구간 안에서 여러 번 반복된다(Milvus 결과 캐시 여부 미확인).
"""
import argparse
import csv
import os
import re
from collections import defaultdict

ANCHOR = ("A", 8)
DRIFT_MAX = 0.15
REF_1544_A8 = 631.0
SLOW_RATIO = 0.85
FORK_MSG = "Resource temporarily unavailable"


def overlap_python(waitlog):
    """OVERLAP 블록(제목 줄 다음의 들여쓴 줄들) 중 python이 들어간 줄을 돌려준다."""
    hits, inside = [], False
    for line in open(waitlog, encoding="utf-8", errors="replace"):
        if "OVERLAP:" in line:
            inside = True
            continue
        if inside and line.startswith("  "):
            if "python" in line.lower():
                hits.append(line.strip()[:150])
            continue
        inside = False
    return hits


def has_fork_error(*paths):
    return [p for p in paths if p and os.path.exists(p) and FORK_MSG in open(p, encoding="utf-8", errors="replace").read()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--waitlog", required=True, help="wait_run_pf.sh 출력 (예: c3a_real2_wait.log)")
    ap.add_argument("--runlog", default="", help="run_c3a_real2.sh 로그 (예: c3a_real2.log). 없으면 outdir 안의 c3a_real2.log")
    ap.add_argument("--ds", default="nq")
    a = ap.parse_args()
    runlog = a.runlog or f"{a.outdir}/c3a_real2.log"
    for p in (a.waitlog,):
        if not os.path.exists(p):
            raise SystemExit(f"{p} 없음 (V1·V2를 확인할 수 없어 판정하지 않음)")
    path = f"{a.outdir}/results_c3a_load_{a.ds}.csv"
    if not os.path.exists(path):
        raise SystemExit(f"{path} 없음")
    with open(path, encoding="utf-8") as f:
        R = list(csv.DictReader(f))
    info_path = f"{a.outdir}/fixed_{a.ds}_info.json"
    qline = " ".join(open(info_path, encoding="utf-8").read().split()) if os.path.exists(info_path) else "(fixed_info 없음)"

    out = [f"# 실제 질문 탐색 재측정 판정 ({a.ds})", "", f"질문 정보: {qline[:300]}", "", "## 1. 세션 유효 조건", ""]
    bad = []

    # V1
    py = overlap_python(a.waitlog)
    out.append(f"- V1 겹친 python 작업: {'없음' if not py else f'{len(py)}개'}")
    out += [f"  - {x}" for x in py]
    if py:
        bad.append("V1 측정 중 다른 계정 python 작업")
    n_overlap = sum(1 for l in open(a.waitlog, encoding="utf-8", errors="replace") if "OVERLAP:" in l)
    out.append(f"  - (참고) OVERLAP 기록 수 {n_overlap}. python이 아닌 것은 기록만 한다.")

    # V2
    if not os.path.exists(runlog):
        out.append(f"- V2 실행 로그 {runlog} 없음 → wait 로그만 확인")
    forks = has_fork_error(a.waitlog, runlog)
    out.append(f"- V2 fork 실패 기록: {'없음' if not forks else ', '.join(forks)}")
    if forks:
        bad.append("V2 fork 실패(Resource temporarily unavailable)")

    # V3
    anc = {}
    for r in R:
        if r["phase"] == "anchor" and r["setting"] == ANCHOR[0] and int(r["cap"]) == ANCHOR[1] and r.get("qps"):
            anc[int(r["rep"])] = float(r["qps"])
    reps_expected = sorted({int(r["rep"]) for r in R if r["phase"] == "explore"})
    out.append(f"- V3 기준 설정 {ANCHOR[0]} k={ANCHOR[1]} QPS: " + ", ".join(f"rep{k} {v:.0f}" for k, v in sorted(anc.items())))
    need = [0] + reps_expected
    if any(k not in anc for k in need):
        bad.append(f"V3 기준 설정 누락 (필요 {need}, 있음 {sorted(anc)})")
    else:
        for rep in reps_expected:
            b, e = anc[rep - 1], anc[rep]
            d = abs(e - b) / max(b, e)
            ok = d <= DRIFT_MAX
            out.append(f"  - {rep}회차 앞 {b:.0f} → 뒤 {e:.0f}: 차이 {d * 100:.1f}% ({'통과' if ok else '초과'}, 기준 {DRIFT_MAX * 100:.0f}%)")
            if not ok:
                bad.append(f"V3 {rep}회차 동안 상태 변화 {d * 100:.1f}%")

    # V4
    E = [r for r in R if r["phase"] == "explore"]
    errs = sum(int(r.get("errors") or 0) for r in E)
    empty = sum(1 for r in E if not r.get("qps"))
    out.append(f"- V4 탐색 구간 {len(E)}개 (기대 36), 오류 {errs}, 빈 값 {empty}")
    if len(E) != 36 or errs or empty:
        bad.append("V4 구간 수·오류")

    valid = not bad
    out.append("")
    out.append(f"**세션: {'유효' if valid else '판정 불가: ' +'; '.join(bad)}**")
    if anc:
        am = sum(anc.values()) / len(anc)
        slow = am < REF_1544_A8 * SLOW_RATIO
        out.append(f"상태 표시: 기준 설정 평균 {am:.0f} (1544 A k8 {REF_1544_A8:.0f}의 {am / REF_1544_A8 * 100:.0f}%) → "
                   f"{'느린 상태에서 측정 (1544와 같은 조건으로 비교하지 않음)' if slow else '1544와 비슷한 상태'}")

    # 판정 (유효하지 않으면 계산은 보이되 판정으로 쓰지 않음)
    q = defaultdict(dict)
    for r in E:
        if r.get("qps") and not int(r.get("errors") or 0):
            q[(r["setting"], int(r["cap"]))][int(r["rep"])] = float(r["qps"])
    title = "## 2. 판정" if valid else "## 2. 참고 계산 (세션이 판정 불가라서 판정으로 쓰지 않음)"
    out += ["", title, "",
            "| 설정 | 기준 k (6/8 중 평균 높은 쪽) | 기준 k QPS | k=12 QPS | k=12 < 기준 k | 결과 |",
            "|---|---|---|---|---|---|"]
    verdicts = {}
    for s in sorted({k[0] for k in q}):
        g6, g8, g12 = q.get((s, 6), {}), q.get((s, 8), {}), q.get((s, 12), {})
        reps = sorted(set(g6) & set(g8) & set(g12))
        if len(reps) != 3:
            verdicts[s] = "판정 불가"
            out.append(f"| {s} | - | - | - | - | 판정 불가 (회차 수 {len(reps)}) |")
            continue
        m6, m8 = sum(g6[r] for r in reps) / 3, sum(g8[r] for r in reps) / 3
        kb, gb = (6, g6) if m6 >= m8 else (8, g8)
        lower = [g12[r] < gb[r] for r in reps]
        v = "재현" if all(lower) else "재현 안 됨"
        verdicts[s] = v
        out.append(f"| {s} | {kb} (k6 {m6:.0f} / k8 {m8:.0f}) | {', '.join(f'{gb[r]:.0f}' for r in reps)} | "
                   f"{', '.join(f'{g12[r]:.0f}' for r in reps)} | {sum(lower)}/3 | {'**' + v + '**' if valid else '(참고) ' + v} |")
    out.append("")
    if valid:
        out.append("요약: " + ", ".join(f"{s} {v}" for s, v in sorted(verdicts.items())))
    else:
        out.append("요약: 세션 판정 불가. 위 표는 참고 계산이며 판정에 쓰지 않는다.")
    out.append("규칙: judge_c3a_real2.py 맨 위 (11/23 측정 전 고정).")
    text = "\n".join(out)
    print(text)
    open(f"{a.outdir}/summary_real2_judge_{a.ds}.md", "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
