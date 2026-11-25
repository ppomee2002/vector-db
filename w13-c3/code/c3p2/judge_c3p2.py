"""C3 단독 효과 측정 판정 (설계 v3 고정본, 2026-11-23). 규칙은 측정 전에 고정했고 결과를 본 뒤 바꾸지 않는다.

1단계 정확성 (--stage correctness --prefix c|ca)
  실행 3번 <prefix>_serial / <prefix>_m3 / <prefix>_m7.
  PASS = 세 실행 모두 종료 코드 0, 오류 0(summary n_error), checks 위반 0, text_check 불일치 0(대조 수 > 0).
  기록만: 실행 사이 질문별 hits(chunk_id 순서)·fetched_ids 일치 수, 결과 수.
  evaluate 관련 실패 = 종료 코드가 0이 아니고, 로그의 마지막 traceback에서 마지막 pipeline-core 코드 위치가
    modules/evaluate.py 이거나 main.py의 평가 블록(`if run["evaluate"] and texts_store:` ~ `texts_store.clear()`) 안.
    오류 수, checks 위반, text_check 불일치는 해당하지 않는다(그건 FAIL).
  반환 코드: 0 PASS / 10 evaluate 관련 실패(실행기가 nq_test_ans로 다시 함) / 1 FAIL

2단계 판정 (--stage all)
  세션 유효 (하나라도 어기면 "판정 불가", 결과는 모두 보고)
    V1 측정 중 다른 계정 python 작업이 새로 없음: wait 로그 OVERLAP 블록에 python 없음(로그가 있을 때),
       그리고 각 실행 run_meta other_user_activity의 after에 before에 없던 python 프로세스 없음
    V2 fork 실패('Resource temporarily unavailable') 기록 없음 (실행 로그, wait 로그, 실행별 로그)
    V3 성능 6번 모두 있고 종료 코드 0, 오류 0, checks 위반 0
    V4 각 실행 앞뒤 스위치 status의 OOM 숫자가 늘지 않음. status에서 OOM 숫자를 읽지 못하면 "확인 못 함" → 판정 불가
    V5 (설계 v3.1, 측정 전 추가) 정확성 포함 모든 실행의 앞뒤 status에 스위치 제한 없음.
       제한 = memory.high 또는 memory.max 값이 max가 아님, 또는 io.max: 뒤(같은 줄~'floor for mem:' 전)에 내용이 있음.
       status를 못 읽으면 "확인 못 함" → 판정 불가
  판정 (유효한 세션만, 여유폭 0): 쌍 (p1,p2) (p3,p4) (p5,p6), r_i = m7 qps / m3 qps
    "상한 8 쪽으로 3쌍 일치" = r 3개 모두 > 1 / "제한 없음 쪽으로 3쌍 일치" = 모두 < 1 / 나머지 "방향 일치 안 함"
  보고용 (판정 불변): r 3개와 최솟값(제한 없음 쪽이면 최댓값), 우연 확률 1/4, r 최솟값 < 1.01이면 strict_qps r 3개,
    "흔들림 큼" = max(m3 3회 범위, m7 3회 범위) > min(쌍별 |m7 − m3|) (QPS), p50/p95/p99·queue_wait,
    m3가 먼저인 쌍이 2개(m7 판정에 보수적), 실행마다 nq_seq 새로 load·예열 5초.
  판정은 내부 기준이고 통계적 입증이 아니다. B0(C2 none) 결과는 예비 결과이고 C3 주 결과는 BC1 위(D5).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

FORK_MSG = "Resource temporarily unavailable"
PAIRS = [("p1_m3", "p2_m7"), ("p4_m3", "p3_m7"), ("p5_m3", "p6_m7")]


def read_runs(sess):
    runs = {}
    p = os.path.join(sess, "runs.tsv")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            f = line.rstrip("\n").split("\t")
            if len(f) >= 3:
                runs[f[0]] = {"rc": int(f[1]), "out": f[2]}   # 같은 이름이 여러 번이면 마지막 것
    return runs


def jload(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return None


def out_dir(sess, run):
    """runs.tsv의 결과 폴더. 서버 절대 경로가 PC에서 안 맞으면 SESS/out/<이름> 으로 찾는다."""
    o = run.get("out") or ""
    if o and os.path.isdir(o):
        return o
    alt = os.path.join(sess, "out", os.path.basename(o)) if o else ""
    return alt if alt and os.path.isdir(alt) else o


def eval_related(log_path):
    """마지막 traceback의 마지막 pipeline-core 위치가 evaluate.py 또는 main.py 평가 블록이면 True."""
    try:
        txt = open(log_path, encoding="utf-8", errors="replace").read()
    except OSError:
        return False, "로그 없음"
    i = txt.rfind("Traceback (most recent call last)")
    if i < 0:
        return False, "traceback 없음"
    frames = re.findall(r'File "([^"]+)", line (\d+)', txt[i:])
    frames = [(f, int(n)) for f, n in frames if "pipeline-core" in f.replace("\\", "/")]
    if not frames:
        return False, "pipeline-core 위치 없음"
    f, n = frames[-1]
    fn = f.replace("\\", "/")
    if fn.endswith("modules/evaluate.py"):
        return True, f"{fn}:{n}"
    if fn.endswith("/main.py"):
        try:
            lines = open(f, encoding="utf-8").read().splitlines()
            s = next(k for k, l in enumerate(lines, 1) if 'if run["evaluate"] and texts_store:' in l)
            e = next(k for k, l in enumerate(lines, 1) if k > s and "texts_store.clear()" in l)
        except (OSError, StopIteration):
            return False, f"{fn}:{n} (main.py 평가 블록 위치를 못 찾음)"
        return (s <= n <= e), f"{fn}:{n} (평가 블록 {s}~{e})"
    return False, f"{fn}:{n}"


def read_results(od):
    out = {}
    p = os.path.join(od, "results.jsonl")
    if not os.path.exists(p):
        return None
    for line in open(p, encoding="utf-8"):
        d = json.loads(line)
        key = (d.get("query_set"), d.get("q_idx"))
        out[key] = ([h.get("chunk_id") for h in d.get("hits", [])], d.get("fetched_ids"))
    return out


def stage_correctness(sess, prefix, md):
    runs = read_runs(sess)
    labels = [f"{prefix}_serial", f"{prefix}_m3", f"{prefix}_m7"]
    md.append(f"## 1단계 정확성 ({prefix})")
    fail, evalfail, res = [], [], {}
    for lb in labels:
        r = runs.get(lb)
        if r is None:
            md.append(f"- {lb}: 실행 기록 없음 → FAIL"); fail.append(lb); continue
        od = out_dir(sess, r)
        s = jload(os.path.join(od, "summary.json"))
        if r["rc"] != 0:
            ok, where = eval_related(os.path.join(sess, f"{lb}.log"))
            md.append(f"- {lb}: 종료 코드 {r['rc']}, 예외 위치 {where} → {'evaluate 관련 실패' if ok else 'FAIL'}")
            (evalfail if ok else fail).append(lb)
            continue
        if s is None:
            md.append(f"- {lb}: summary.json 없음 → FAIL"); fail.append(lb); continue
        nerr = s.get("n_error")
        chk = {k: v.get("n_violation", 0) for k, v in (s.get("checks") or {}).items()}
        tc = s.get("text_check") or {}
        good = (nerr == 0 and not any(chk.values()) and tc.get("n_checked", 0) > 0 and tc.get("n_mismatch") == 0)
        md.append(f"- {lb}: 결과 {s.get('n_results')}, 오류 {nerr}, checks 위반 {sum(chk.values())}, "
                  f"text_check {tc.get('n_mismatch')}/{tc.get('n_checked')} → {'통과' if good else 'FAIL'}")
        if not good:
            fail.append(lb)
        res[lb] = read_results(od)
    # 기록만: 실행 사이 일치
    base = res.get(labels[0])
    for lb in labels[1:]:
        other = res.get(lb)
        if base is None or other is None:
            continue
        common = set(base) & set(other)
        nh = sum(1 for k in common if base[k][0] == other[k][0])
        nf = sum(1 for k in common if base[k][1] == other[k][1])
        md.append(f"  - (기록) {labels[0]} vs {lb}: 같은 질문 {len(common)}개 중 hits 같음 {nh}, fetched_ids 같음 {nf}")
    if fail:
        md.append(f"- **1단계 FAIL** ({', '.join(fail)})"); return 1
    if evalfail:
        md.append(f"- **evaluate 관련 실패** ({', '.join(evalfail)}) → C3 결함으로 보지 않음, nq_test_ans로 다시"); return 10
    md.append("- **1단계 PASS**"); return 0


def oom_numbers(path):
    try:
        txt = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return None
    nums = {}
    for k, v in re.findall(r"\b(oom[\w.]*)\b\D{0,3}(\d+)", txt, flags=re.I):
        nums[k.lower()] = int(v)
    return nums or None


def limited(path):
    """설계 v3.1 V5: 스위치 status에 제한이 걸려 있으면 True, 없으면 False, 확인 못 하면 None."""
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return None
    hi = mx = None
    io_i = None
    for k, l in enumerate(lines):
        m = re.search(r"memory\.high\s+(\S+)", l)
        if m:
            hi = m.group(1)
            m2 = re.search(r"memory\.max\s+(\S+)", l)
            mx = m2.group(1) if m2 else None
        if l.startswith("io.max:"):
            io_i = k
    if hi is None or mx is None or io_i is None:
        return None
    io_txt = lines[io_i][len("io.max:"):].strip()
    for l in lines[io_i + 1:]:
        if l.startswith("floor for mem:"):
            break
        io_txt += l.strip()
    return hi != "max" or mx != "max" or bool(io_txt)


def python_procs(act):
    try:
        return {(p["user"], p["comm"]) for p in act.get("other_user_processes", []) if "python" in p["comm"].lower()}
    except (AttributeError, KeyError, TypeError):
        return set()


def stage_all(sess, waitlog, runlog, md):
    runs = read_runs(sess)
    labels = ["p1_m3", "p2_m7", "p3_m7", "p4_m3", "p5_m3", "p6_m7"]
    md.append("## 2단계 성능 (설계 v3)")
    bad = []
    S = {}
    for lb in labels:
        r = runs.get(lb)
        S[lb] = (r, jload(os.path.join(out_dir(sess, r), "summary.json")) if r else None,
                 jload(os.path.join(out_dir(sess, r), "run_meta.json")) if r else None)

    # V1
    v1 = []
    if waitlog and os.path.exists(waitlog):
        inside = False
        for line in open(waitlog, encoding="utf-8", errors="replace"):
            if "OVERLAP:" in line:
                inside = True; continue
            if inside and line.startswith("  "):
                if "python" in line.lower():
                    v1.append("wait 로그: " + line.strip()[:120])
                continue
            inside = False
    else:
        md.append("- V1 참고: wait 로그 없음 → run_meta other_user_activity만 확인")
    for lb in labels:
        meta = S[lb][2] or {}
        act = meta.get("other_user_activity") or {}
        new = python_procs(act.get("after") or {}) - python_procs(act.get("before") or {})
        if new:
            v1.append(f"{lb}: 새 python {sorted(new)}")
    md.append(f"- V1 다른 계정 새 python 작업: {'없음' if not v1 else '; '.join(v1)}")
    if v1:
        bad.append("V1")
    # 참고(판정 불변): 기록기 2초 기록에서 실행 중 처음 보다 새로 나타난 다른 계정 python
    rec = os.path.join(sess, "rec.csv")
    if os.path.exists(rec):
        seen: dict = {}
        for line in open(rec, encoding="utf-8"):
            f = line.rstrip("\n").split(",")
            if len(f) >= 10 and f[2] in labels:
                seen.setdefault(f[2], []).append(set(f[9].split()))
        notes = []
        for lb, rows in seen.items():
            new = set().union(*rows) - rows[0]
            if new:
                notes.append(f"{lb}: {' '.join(sorted(new))}")
        md.append(f"- (참고, 판정 불변) 기록기 2초 기록 중 실행 도중 새로 나타난 다른 계정 python(uid:pid): "
                  f"{'없음' if not notes else '; '.join(notes)}")
    # V2
    paths = [p for p in [runlog, waitlog] if p] + [os.path.join(sess, f"{lb}.log") for lb in labels]
    forks = [p for p in paths if os.path.exists(p) and FORK_MSG in open(p, encoding="utf-8", errors="replace").read()]
    md.append(f"- V2 fork 실패: {'없음' if not forks else ', '.join(forks)}")
    if forks:
        bad.append("V2")
    # V3
    v3 = []
    for lb in labels:
        r, s, _ = S[lb]
        if r is None or s is None:
            v3.append(f"{lb} 없음"); continue
        nchk = sum(v.get("n_violation", 0) for v in (s.get("checks") or {}).values())
        if r["rc"] != 0 or s.get("n_error") or nchk:
            v3.append(f"{lb} rc={r['rc']} 오류={s.get('n_error')} checks={nchk}")
    md.append(f"- V3 6번 실행 정상: {'예' if not v3 else '; '.join(v3)}")
    if v3:
        bad.append("V3")
    # V4
    v4, unread = [], []
    for lb in labels:
        a = oom_numbers(os.path.join(sess, f"{lb}.status_before.txt"))
        b = oom_numbers(os.path.join(sess, f"{lb}.status_after.txt"))
        if a is None or b is None:
            unread.append(lb); continue
        up = [k for k in b if k in a and b[k] > a[k]]
        if up:
            v4.append(f"{lb}: {', '.join(f'{k} {a[k]}→{b[k]}' for k in up)}")
    if unread:
        md.append(f"- V4 OOM: 확인 못 함 (status에서 OOM 숫자를 못 읽음: {', '.join(unread)})"); bad.append("V4(확인 못 함)")
    else:
        md.append(f"- V4 OOM 증가: {'없음' if not v4 else '; '.join(v4)}")
        if v4:
            bad.append("V4")

    # V5 (설계 v3.1): 정확성 포함 모든 실행 앞뒤 status에 스위치 제한 없음
    lim, unk = [], []
    allrun = [lb for lb in read_runs(sess)]
    for lb in allrun:
        for side in ("before", "after"):
            v = limited(os.path.join(sess, f"{lb}.status_{side}.txt"))
            if v is None:
                unk.append(f"{lb}.{side}")
            elif v:
                lim.append(f"{lb}.{side}")
    if unk:
        md.append(f"- V5 스위치 제한: 확인 못 함 ({', '.join(unk)})"); bad.append("V5(확인 못 함)")
    else:
        md.append(f"- V5 스위치 제한이 걸린 status: {'없음' if not lim else ', '.join(lim)} (실행 {len(allrun)}개 앞뒤)")
        if lim:
            bad.append("V5")

    # 표
    md.append("\n| 실행 | qps | strict_qps | total p50 | p95 | p99 | queue_wait p50 | 오류 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for lb in labels:
        s = S[lb][1] or {}
        st = s.get("stage_ms") or {}
        tot, qw = st.get("total") or {}, st.get("queue_wait") or {}
        f = lambda v: "" if v is None else f"{v:.1f}"  # noqa: E731
        md.append(f"| {lb} | {f(s.get('qps'))} | {f(s.get('strict_qps'))} | {f(tot.get('p50'))} | {f(tot.get('p95'))} | "
                  f"{f(tot.get('p99'))} | {f(qw.get('p50'))} | {s.get('n_error', '')} |")

    q = {lb: (S[lb][1] or {}).get("qps") for lb in labels}
    sq = {lb: (S[lb][1] or {}).get("strict_qps") for lb in labels}
    if any(v is None for v in q.values()):
        md.append("\n**판정 불가 (qps 없음)**" + (f" — 위반: {', '.join(bad)}" if bad else ""))
        return 2
    r = [q[b] / q[a] for a, b in PAIRS]
    m3 = [q[a] for a, _ in PAIRS]; m7 = [q[b] for _, b in PAIRS]
    if all(x > 1 for x in r):
        verdict, edge = "상한 8 쪽으로 3쌍 일치", f"r 최솟값 {min(r):.3f}"
    elif all(x < 1 for x in r):
        verdict, edge = "제한 없음 쪽으로 3쌍 일치", f"r 최댓값 {max(r):.3f}"
    else:
        verdict, edge = "방향 일치 안 함", f"r 범위 {min(r):.3f}~{max(r):.3f}"
    md.append(f"\n- 쌍별 r = m7/m3: {', '.join(f'{x:.3f}' for x in r)} ({edge})")
    if min(r) < 1.01:
        rs = [sq[b] / sq[a] if sq[a] and sq[b] else None for a, b in PAIRS]
        md.append(f"- r 최솟값 < 1.01 → strict_qps r: {', '.join('없음' if x is None else f'{x:.3f}' for x in rs)}")
    rng = max(max(m3) - min(m3), max(m7) - min(m7))
    diff = min(abs(b - a) for a, b in zip(m3, m7))
    md.append(f"- 같은 조건 3회 범위 최대 {rng:.1f} QPS, 쌍별 |차이| 최소 {diff:.1f} QPS"
              + (" → **흔들림 큼**" if rng > diff else ""))
    md.append("- 차이가 없어도 3쌍이 우연히 한 방향으로 모일 확률은 양쪽 합쳐 1/4(쌍 독립 가정).")
    md.append("- m3가 먼저인 쌍이 2개: 처리량이 시간에 따라 계속 떨어지면 m3에 약간 유리(m7 판정에 보수적).")
    md.append("- 실행마다 MilvusGuard가 nq_seq를 새로 load, 예열 5초. 측정 중 메모리·CPU 기록은 rec.csv(판정에 안 씀).")
    md.append("- B0(C2 none)면 예비 결과. C3 주 결과는 BC1 위(D5). clients 16은 D4의 보조 조건.")
    if bad:
        md.append(f"\n**판정 불가** (위반: {', '.join(bad)}). 참고 방향: {verdict}")
        return 2
    md.append(f"\n**판정: {verdict}** ({edge})")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sess", required=True)
    ap.add_argument("--stage", choices=["correctness", "all"], default="all")
    ap.add_argument("--prefix", default="c")
    ap.add_argument("--waitlog")
    ap.add_argument("--runlog")
    a = ap.parse_args()
    md = [f"# C3 단독 효과 판정 (judge_c3p2.py, 설계 v3 + v3.1 V5) — {a.sess}"]
    if a.stage == "correctness":
        rc = stage_correctness(a.sess, a.prefix, md)
        name = f"judge_correctness_{a.prefix}.md"
    else:
        pre = "ca" if os.path.exists(os.path.join(a.sess, "ca_serial.log")) else "c"
        stage_correctness(a.sess, pre, md)
        rc = stage_all(a.sess, a.waitlog, a.runlog or os.path.join(a.sess, "run.log"), md)
        name = "judge_all.md"
    text = "\n".join(md) + "\n"
    open(os.path.join(a.sess, name), "w", encoding="utf-8").write(text)
    sys.stdout.buffer.write(text.encode("utf-8"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
