#!/usr/bin/env python3
"""analyze_pf3.py: 실제 질문의 검색 결과(hits)와 정답 근거(gold_map.py)로 이웃 선택 규칙을 채점한다. Milvus는 쓰지 않는다.

질문마다 R = top-10 조각, N = R의 같은 문서 앞뒤 이웃 전부(R 제외), sel(규칙) = 규칙이 고른 이웃.
읽는 조각 = R + sel(규칙). 규칙은 pf3_rules.py에 있고 qrels/정답 문자열을 보지 않는다.

채점 (두 기준을 따로 낸다)
  관련 문단 기준 (qrels, gold_ok가 true일 때만)
    gold_any   정답 문단의 조각을 하나라도 읽었나
    gold_cov   정답 문단 단어 중 읽은 조각에 들어 있는 비율 (질문마다 계산 후 평균)
    gold_full  정답 문단 하나를 통째로 읽었나
  정답 문자열 기준 (NQ만, nq_open 짧은 답)
    ans        읽은 글(이어진 조각은 이어 붙임) 안에 정답 문자열이 있나
  "이웃이 필요함" = top-10만으로는 ans가 없는데 이웃까지 읽으면 생김 (p_ans_need)
  "이웃에 정답이 있음"과 구분하려고, 이웃 조각 자체에 정답 문자열이 있는 질문 중 top-10에 이미 정답이 있던 비율도 낸다.
선택 정확도 (이웃 하나하나 단위, 전체 합산)
  useful_gold(n) = 이웃 n에 정답 문단 단어가 있음
  useful_ans(n)  = R만으로는 정답 문자열이 없고 R + n이면 생김
  precision = 고른 이웃 중 쓸모 있는 것 비율, recall = 쓸모 있는 이웃 중 고른 비율

출력 (--out 폴더)
  pf3_quality_<ds>.csv   부분집합(gold, ans, all) × 규칙별 요약
  pf3_need_<ds>.json     필요 비율 요약
  need_<ds>.json         시간 측정용 질문 목록과 질문별 필요 여부 (bench_pf3.py가 읽음)
  pf3_perq_<ds>.csv      질문별 값,  pf3_nb_<ds>.csv.gz  이웃별 값,  cand_texts_<ds>.jsonl.gz  후보 조각 글(다시 분석용)
usage: python analyze_pf3.py --ds nq --data /workspace/junlim/data_nq --gold /workspace/junlim/pf3_out
       python analyze_pf3.py --selftest
"""
import argparse
import csv
import gzip
import json
import os
import re
import sys
import unicodedata

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pf3_rules import RULES, select  # noqa: E402

WORD = re.compile(r"\w+", re.UNICODE)
QCOLS = ["ds", "subset", "rule", "queries", "sel_q_share", "extra_pks", "fetched_pks", "nb_share", "gold_any", "gold_cov",
         "gold_full", "gold_gain_share", "ans", "need_fixed_share", "prec_gold", "rec_gold", "prec_ans", "rec_ans"]


def toks(s):
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return " ".join(WORD.findall(s.lower()))


def segments(S, doc, tk):
    out, cur = [], []
    for c in sorted(S):
        if cur and c == cur[-1] + 1 and doc[c] == doc[cur[-1]]:
            cur.append(c)
        else:
            if cur:
                out.append(cur)
            cur = [c]
    if cur:
        out.append(cur)
    return [" " + " ".join(tk[c] for c in seg) + " " for seg in out]


def has_ans(S, doc, tk, ans):
    if not ans:
        return False
    segs = segments(S, doc, tk)
    return any(a in s for s in segs for a in ans)


def gold_metrics(S, gold):
    tot = sum(g["words"] for g in gold)
    got = sum(w for g in gold for c, w in g["chunks"] if c in S)
    anyg = any(c in S for g in gold for c, _ in g["chunks"])
    full = any(all(c in S for c, _ in g["chunks"]) for g in gold)
    return anyg, got / max(tot, 1), full


def analyze(ds, recs, hits, doc, flags, tk, gold_ok):
    """recs: {qi: gold rec}, hits: {qi: [chunk,...]}, tk: {chunk: token string}. 반환: rows, need summary, perq, nbrows, need file."""
    per = []
    nbrows = []
    for qi, R in hits.items():
        rec = recs.get(qi, {})
        g_eval = bool(gold_ok and rec.get("status") == "eval")
        gold = [g for g in rec.get("gold", []) if g.get("full")] if g_eval else []
        ans = None
        if rec.get("ans"):
            ans = [" " + toks(x) + " " for x in rec["ans"] if toks(x)]
            ans = ans or None
        a_eval = ans is not None
        Rs = set(R)
        N = select("all", R, doc)
        sel = {r: select(r, R, doc, flags) for r in RULES}
        q = dict(qi=qi, g_eval=g_eval, a_eval=a_eval, n_nb=len(N), sel={r: len(sel[r]) for r in RULES})
        gchunks = {c for g in gold for c, _ in g["chunks"]}
        ans_R = has_ans(Rs, doc, tk, ans) if a_eval else False
        for r in RULES:
            S = Rs | set(sel[r])
            if g_eval:
                q[("gold", r)] = gold_metrics(S, gold)
            if a_eval:
                q[("ans", r)] = has_ans(S, doc, tk, ans)
        # 이웃 하나하나
        q["useful_gold"], q["useful_ans"], q["nb_has_ans"] = set(), set(), False
        rank_of = {}
        for i, c in enumerate(R):
            for nb in (c - 1, c + 1):
                if nb not in Rs:
                    rank_of.setdefault(nb, i + 1)
        for nb in N:
            ug = g_eval and nb in gchunks
            in_chunk = a_eval and any(a in " " + tk[nb] + " " for a in ans)
            ua = a_eval and (not ans_R) and has_ans(Rs | {nb}, doc, tk, ans)
            if ug:
                q["useful_gold"].add(nb)
            if ua:
                q["useful_ans"].add(nb)
            q["nb_has_ans"] |= bool(in_chunk)
            bits = sum(1 << i for i, r in enumerate(RULES) if nb in sel[r])
            nbrows.append(dict(qi=qi, chunk=nb, src_rank=rank_of.get(nb, 0), useful_gold=int(ug) if g_eval else "",
                               ans_in_chunk=int(in_chunk) if a_eval else "", useful_ans=int(ua) if a_eval else "", rules=bits))
        q["selsets"] = sel
        per.append(q)

    rows = []
    for sub in ("gold", "ans", "all"):
        Q = [q for q in per if (sub == "all" or (sub == "gold" and q["g_eval"]) or (sub == "ans" and q["a_eval"]))]
        if not Q:
            continue
        for r in RULES:
            row = dict(ds=ds, subset=sub, rule=r, queries=len(Q))
            row["sel_q_share"] = round(float(np.mean([q["sel"][r] > 0 for q in Q])), 4)
            row["extra_pks"] = round(float(np.mean([q["sel"][r] for q in Q])), 3)
            row["fetched_pks"] = round(float(np.mean([len(set(hits[q["qi"]])) + q["sel"][r] for q in Q])), 3)
            row["nb_share"] = round(sum(q["sel"][r] for q in Q) / max(sum(q["n_nb"] for q in Q), 1), 4)
            G = [q for q in Q if q["g_eval"]]
            if G:
                row["gold_any"] = round(float(np.mean([q[("gold", r)][0] for q in G])), 4)
                row["gold_cov"] = round(float(np.mean([q[("gold", r)][1] for q in G])), 4)
                row["gold_full"] = round(float(np.mean([q[("gold", r)][2] for q in G])), 4)
                den = sum(q[("gold", "all")][1] - q[("gold", "none")][1] for q in G)
                num = sum(q[("gold", r)][1] - q[("gold", "none")][1] for q in G)
                row["gold_gain_share"] = round(num / den, 4) if den > 0 else ""
                sel_u = sum(len(set(q["selsets"][r]) & q["useful_gold"]) for q in G)
                n_sel = sum(q["sel"][r] for q in G)
                n_u = sum(len(q["useful_gold"]) for q in G)
                row["prec_gold"] = round(sel_u / n_sel, 4) if n_sel else ""
                row["rec_gold"] = round(sel_u / n_u, 4) if n_u else ""
            A = [q for q in Q if q["a_eval"]]
            if A:
                row["ans"] = round(float(np.mean([q[("ans", r)] for q in A])), 4)
                need = [q for q in A if q[("ans", "all")] and not q[("ans", "none")]]
                row["need_fixed_share"] = round(float(np.mean([q[("ans", r)] for q in need])), 4) if need else ""
                sel_u = sum(len(set(q["selsets"][r]) & q["useful_ans"]) for q in A)
                n_sel = sum(q["sel"][r] for q in A)
                n_u = sum(len(q["useful_ans"]) for q in A)
                row["prec_ans"] = round(sel_u / n_sel, 4) if n_sel else ""
                row["rec_ans"] = round(sel_u / n_u, 4) if n_u else ""
            rows.append(row)

    G = [q for q in per if q["g_eval"]]
    A = [q for q in per if q["a_eval"]]
    m = lambda xs: round(float(np.mean(xs)), 4) if len(xs) else None  # noqa: E731
    summ = dict(ds=ds, searched=len(per), gold_eval=len(G), ans_eval=len(A))
    if G:
        summ.update(
            gold_any_top10=m([q[("gold", "none")][0] for q in G]), gold_any_all=m([q[("gold", "all")][0] for q in G]),
            gold_cov_top10=m([q[("gold", "none")][1] for q in G]), gold_cov_all=m([q[("gold", "all")][1] for q in G]),
            gold_full_top10=m([q[("gold", "none")][2] for q in G]), gold_full_all=m([q[("gold", "all")][2] for q in G]),
            p_nb_gold=m([bool(q["useful_gold"]) for q in G]),
            p_nb_gold_rescue=m([bool(q["useful_gold"]) and not q[("gold", "none")][0] for q in G]),
            p_nb_gold_complete=m([bool(q["useful_gold"]) and q[("gold", "none")][0] for q in G]))
    if A:
        hasnb = [q for q in A if q["nb_has_ans"]]
        summ.update(
            ans_top10=m([q[("ans", "none")] for q in A]), ans_all=m([q[("ans", "all")] for q in A]),
            p_ans_need=m([q[("ans", "all")] and not q[("ans", "none")] for q in A]),
            p_nb_has_ans=m([q["nb_has_ans"] for q in A]),
            nb_has_ans_but_top10_already=m([q[("ans", "none")] for q in hasnb]) if hasnb else None)
        reach = [q for q in A if q[("ans", "all")]]
        summ["p_ans_need_given_reach"] = m([not q[("ans", "none")] for q in reach]) if reach else None
        GA = [q for q in A if q["g_eval"]]
        if GA:
            summ.update(ans_eval_and_gold_eval=len(GA), p_ans_need_on_gold_eval=m([q[("ans", "all")] and not q[("ans", "none")] for q in GA]))

    if len(A) >= 300:
        src, Q = "answer", A
        flag = lambda q: bool(q[("ans", "all")] and not q[("ans", "none")])  # noqa: E731
    elif len(G) >= 300:
        src, Q = "gold", G
        flag = lambda q: bool(q["useful_gold"])  # noqa: E731
    else:
        src, Q, flag = "none", per, (lambda q: False)
    needf = dict(ds=ds, source=src, qi=[int(q["qi"]) for q in Q], need=[int(flag(q)) for q in Q])
    summ["timing_set"] = dict(source=src, queries=len(Q), need_share=m(needf["need"]))
    return rows, summ, per, nbrows, needf


def write_csv(path, cols, rows, gz=False):
    op = gzip.open if gz else open
    with op(path, "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def main_run(a):
    out = a.out or a.gold
    gold_ok = False
    try:
        gold_ok = bool(json.load(open(f"{a.gold}/gold_ok_{a.ds}.json")).get("ok"))
    except Exception as e:
        print("gold_ok 없음:", repr(e))
    recs = {}
    gpath = f"{a.gold}/{a.ds}_gold.jsonl"
    if os.path.exists(gpath):
        with open(gpath, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                recs[int(r["qi"])] = r
    pk_of_chunk = np.load(f"{a.data}/pk_seq.npy")
    chunk_of_pk = np.empty(len(pk_of_chunk), np.int64)
    chunk_of_pk[pk_of_chunk] = np.arange(len(pk_of_chunk))
    if os.path.exists(f"{a.data}/doc_idx.npy"):
        doc = np.load(f"{a.data}/doc_idx.npy")
    else:
        with open(f"{a.data}/chunks.jsonl", encoding="utf-8") as f:
            doc = np.array([json.loads(line)["doc_idx"] for line in f])
    flags = np.load(f"{a.gold}/{a.ds}_flags.npy")
    if len(flags) != len(doc):
        raise SystemExit(f"flags 길이 {len(flags)} != 조각 수 {len(doc)}")
    hits = {}
    with open(a.hits or f"{a.gold}/hits_{a.ds}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            hits[int(r["qi"])] = [int(chunk_of_pk[p]) for p in r["ids"]]
    cand = set()
    for R in hits.values():
        cand.update(R)
        cand.update(select("all", R, doc))
    raw, tk = {}, {}
    with open(f"{a.data}/chunks.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in cand:
                raw[i] = json.loads(line)["text"]
                tk[i] = toks(raw[i])
    # Milvus에서 받은 글과 chunks.jsonl이 같은지 (pk <-> 조각 번호가 맞는지)
    chk = f"{a.gold}/hits_{a.ds}_pkcheck.json"
    pk_same = pk_diff = 0
    if os.path.exists(chk):
        for p, t in json.load(open(chk)).items():
            c = int(chunk_of_pk[int(p)])
            if c in raw:
                pk_same += raw[c] == t
                pk_diff += raw[c] != t
    rows, summ, per, nbrows, needf = analyze(a.ds, recs, hits, doc, flags, tk, gold_ok)
    summ.update(gold_ok=gold_ok, pk_text_same=pk_same, pk_text_diff=pk_diff)
    write_csv(f"{out}/pf3_quality_{a.ds}.csv", QCOLS, rows)
    json.dump(summ, open(f"{out}/pf3_need_{a.ds}.json", "w"), ensure_ascii=False, indent=1)
    json.dump(needf, open(f"{out}/need_{a.ds}.json", "w"))
    pcols = ["qi", "g_eval", "a_eval", "n_nb"] + [f"sel_{r}" for r in RULES] + \
            [f"{k}_{r}" for k in ("gany", "gcov", "gfull", "ans") for r in ("none", "all")] + ["n_useful_gold", "n_useful_ans", "nb_has_ans"]
    prow = []
    for q in per:
        d = dict(qi=q["qi"], g_eval=int(q["g_eval"]), a_eval=int(q["a_eval"]), n_nb=q["n_nb"],
                 n_useful_gold=len(q["useful_gold"]), n_useful_ans=len(q["useful_ans"]), nb_has_ans=int(q["nb_has_ans"]))
        d.update({f"sel_{r}": q["sel"][r] for r in RULES})
        for r in ("none", "all"):
            if q["g_eval"]:
                g = q[("gold", r)]
                d.update({f"gany_{r}": int(g[0]), f"gcov_{r}": round(g[1], 4), f"gfull_{r}": int(g[2])})
            if q["a_eval"]:
                d[f"ans_{r}"] = int(q[("ans", r)])
        prow.append(d)
    write_csv(f"{out}/pf3_perq_{a.ds}.csv", pcols, prow)
    write_csv(f"{out}/pf3_nb_{a.ds}.csv.gz", ["qi", "chunk", "src_rank", "useful_gold", "ans_in_chunk", "useful_ans", "rules"],
              nbrows, gz=True)
    with gzip.open(f"{out}/cand_texts_{a.ds}.jsonl.gz", "wt", encoding="utf-8") as f:
        for c in sorted(raw):
            f.write(json.dumps({"chunk": c, "doc": int(doc[c]), "text": raw[c]}, ensure_ascii=False) + "\n")
    # 화면 요약
    print(json.dumps(summ, ensure_ascii=False))
    print(f"{'subset':6} {'rule':5} {'q':>5} {'extra':>6} {'gold_cov':>8} {'gold_full':>9} {'ans':>6} {'fixed':>6} {'prec_g':>6} {'rec_g':>6}")
    for r in rows:
        print(f"{r['subset']:6} {r['rule']:5} {r['queries']:>5} {r['extra_pks']:>6} {str(r.get('gold_cov', '')):>8} "
              f"{str(r.get('gold_full', '')):>9} {str(r.get('ans', '')):>6} {str(r.get('need_fixed_share', '')):>6} "
              f"{str(r.get('prec_gold', '')):>6} {str(r.get('rec_gold', '')):>6}")


def selftest():
    # 문서 0: 조각 0~3, 문서 1: 조각 4~6, 문서 2: 조각 7
    doc = np.array([0, 0, 0, 0, 1, 1, 1, 2])
    flags = np.array([2, 1 | 2, 1, 0, 2, 1, 0, 0], np.uint8)
    text = ["alpha beta", "gamma delta", "the answer is", "forty two here", "x y", "z w", "v u", "solo"]
    tk = {i: toks(t) for i, t in enumerate(text)}
    # 질문 0: top이 조각 2. 정답 "forty two"는 이웃 3에만 있음 -> 필요. 정답 문단 = 조각 2(3단어)+3(2단어 겹침)
    # 질문 1: top이 조각 3. 정답이 이미 있음 -> 이웃 2에도 'answer' 없음, 필요 없음
    recs = {0: dict(status="eval", ans=["Forty-two"], gold=[dict(full=True, words=5, chunks=[[2, 3], [3, 2]])]),
            1: dict(status="eval", ans=["forty two"], gold=[dict(full=True, words=4, chunks=[[3, 4]])]),
            2: dict(status="out", ans=None, gold=[])}
    hits = {0: [2, 7], 1: [3, 5], 2: [5]}
    rows, summ, per, nb, needf = analyze("t", recs, hits, doc, flags, tk, True)
    R = {(r["subset"], r["rule"]): r for r in rows}
    ok = True
    ok &= summ["ans_top10"] == 0.5 and summ["ans_all"] == 1.0 and summ["p_ans_need"] == 0.5
    ok &= abs(R[("gold", "none")]["gold_cov"] - (3 / 5 + 1.0) / 2) < 1e-4
    ok &= R[("gold", "all")]["gold_cov"] == 1.0 and R[("gold", "all")]["gold_full"] == 1.0
    ok &= R[("ans", "all")]["need_fixed_share"] == 1.0 and R[("ans", "none")]["need_fixed_share"] == 0.0
    # 규칙 A: 조각 2는 앞이 잘림(bit0) -> 1 선택, 뒤(3)는 안 잘림 -> 질문 0을 못 고침
    ok &= R[("ans", "A")]["need_fixed_share"] == 0.0
    ok &= summ["nb_has_ans_but_top10_already"] == 0.0  # 이웃 자체에 정답이 있는 질문은 0번뿐
    ok &= needf["source"] == "none"  # 300개 미만
    print("analyze selftest", "PASS" if ok else "FAIL", summ, file=sys.stderr)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="nq")
    ap.add_argument("--data", default="/workspace/junlim/data_nq")
    ap.add_argument("--gold", default="/workspace/junlim/pf3_out")
    ap.add_argument("--hits", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    main_run(a)


if __name__ == "__main__":
    main()
