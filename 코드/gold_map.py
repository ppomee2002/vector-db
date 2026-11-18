#!/usr/bin/env python3
"""gold_map.py: 실제 질문의 정답 근거(qrels)를 우리 조각(chunk) 번호로 옮긴다. Milvus는 쓰지 않는다.

NQ
  prep_nq.py와 똑같은 순서와 함수(group_articles, build_chunks)로 mteb/nq 코퍼스를 다시 읽어서
  문단(passage) ID -> (문서 번호, 단어 위치, 걸친 조각들)을 계산한다.
  다시 만든 100만 조각의 글, 문서 번호, 조각 번호가 data_nq/chunks.jsonl과 하나도 다르지 않은지 전부 대조한다.
  정답 문단의 글이 저장된 조각 글 안에 그대로 들어 있는지도 정답 문단마다 따로 확인한다.
  정답 문자열(짧은 답)은 google-research-datasets/nq_open(validation)에서 질문 글이 같은 것을 찾아 붙인다.
FiQA
  data/chunks.jsonl의 문서별 글과 mteb/fiqa 코퍼스의 글을 맞춰 문서 ID -> 조각들을 찾는다(글이 같아야 인정).
  FiQA에는 짧은 답이 없어서 관련 문서(qrels) 기준만 낸다.
조각 경계 표시(flags, 규칙 A용): 조각이 문장 중간에서 시작하는지/끝나는지. 저장된 조각 글로 계산한다(정답과 무관).
qrels와 정답 문자열은 채점에만 쓴다. 이웃을 고르는 규칙은 이것들을 보지 않는다.

출력 (--out, 기본 /workspace/junlim/gold)
  nq_gold.jsonl, fiqa_gold.jsonl   질문마다 {qi, qid, q, ans, n_qrels, gold:[{pid, full, words, chunks:[[조각, 겹친 단어 수],..]}], status}
                                   qi = query_vecs.npy의 행 번호
  nq_flags.npy, fiqa_flags.npy     조각마다 bit0 앞 잘림, bit1 뒤 잘림, bit2 문장 끝으로 끝남
  gold_ok_nq.json, gold_ok_fiqa.json   검증 결과 요약 (ok가 true일 때만 정답 문단 기준 분석을 한다)
  gold_report.txt, gold_samples_nq.txt, gold_samples_fiqa.txt   사람이 읽는 보고와 눈으로 확인할 샘플 30개

usage
  python gold_map.py --probe          (데이터셋 접속만 확인. 몇 초, 서버 부하 거의 없음)
  python gold_map.py                  (전체. 약 10분 예상, CPU 1코어, 메모리 2~3GB)
"""
import argparse
import bisect
import hashlib
import json
import os
import random
import re
import sys
import time

os.environ.setdefault("HF_HOME", "/workspace/junlim/hf")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prep_nq as P  # noqa: E402  (전처리 때 쓴 함수를 그대로 쓴다)

FAKE = os.environ.get("GOLD_FAKE")  # 점검용: 가짜 데이터셋 폴더
TERM = re.compile(r"[.!?][\"'”’)\]]*$")
REPORT = []


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def rep(*a):
    s = " ".join(str(x) for x in a)
    REPORT.append(s)
    log(s)


def hf(name, config, split, streaming=False):
    if FAKE:
        fn = f"{FAKE}/{name.replace('/', '__')}__{config}__{split}.jsonl"

        def gen():
            with open(fn, encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)
        return gen() if streaming else list(gen())
    from datasets import load_dataset
    return load_dataset(name, config, split=split, streaming=streaming)


def fld(r, *names):
    for n in names:
        if n in r and r[n] is not None:
            return r[n]
    return None


def norm_ws(s):
    return " ".join(s.split())


def norm_q(s):
    return " ".join(re.findall(r"\w+", s.lower()))


def md5(path):
    try:
        return hashlib.md5(open(path, "rb").read()).hexdigest()[:8]
    except Exception:
        return "?"


def load_qrels(name, split):
    out = {}
    for r in hf(name, "default", split):
        q, d, s = fld(r, "query-id", "query_id", "qid"), fld(r, "corpus-id", "corpus_id", "doc_id", "pid"), fld(r, "score")
        if q is None or d is None:
            raise KeyError(f"qrels row keys: {list(r.keys())}")
        if s is None or float(s) > 0:
            out.setdefault(str(q), set()).add(str(d))
    return out


def term_flags(texts_iter_doc):
    """texts_iter_doc: (doc, text) 순서대로. 반환: doc 배열, flags 배열."""
    docs, term = [], []
    for d, t in texts_iter_doc:
        docs.append(int(d))
        term.append(bool(TERM.search(t.rstrip())))
    doc = np.array(docs, np.int64)
    term = np.array(term, bool)
    n = len(doc)
    same_prev = np.zeros(n, bool)
    same_prev[1:] = doc[1:] == doc[:-1]
    same_next = np.zeros(n, bool)
    same_next[:-1] = doc[:-1] == doc[1:]
    start_cut = same_prev.copy()
    start_cut[1:] &= ~term[:-1]
    end_cut = same_next & ~term
    flags = start_cut.astype(np.uint8) | (end_cut.astype(np.uint8) << 1) | (term.astype(np.uint8) << 2)
    has_nb = same_prev | same_next
    st = dict(chunks=n, term_share=round(float(term.mean()), 4),
              start_cut_share=round(float(start_cut[same_prev].mean()) if same_prev.any() else 0.0, 4),
              end_cut_share=round(float(end_cut[same_next].mean()) if same_next.any() else 0.0, 4),
              cut_any_share_of_chunks_with_nb=round(float((start_cut | end_cut)[has_nb].mean()) if has_nb.any() else 0.0, 4))
    return doc, flags, st


# ---------------------------------------------------------------- probe
def probe():
    tests = [("mteb/nq", "corpus", "corpus"), ("mteb/nq", "queries", "queries"), ("mteb/nq", "default", "test"),
             ("google-research-datasets/nq_open", "nq_open", "validation"),
             ("mteb/fiqa", "corpus", "corpus"), ("mteb/fiqa", "queries", "queries"), ("mteb/fiqa", "default", "test")]
    bad = 0
    for name, cfg, sp in tests:
        t = time.time()
        try:
            r = next(iter(hf(name, cfg, sp, streaming=True)))
            print(f"OK   {name} {cfg}/{sp} ({time.time() - t:.1f}s) keys={sorted(r.keys())}", flush=True)
        except Exception as e:
            bad += 1
            print(f"FAIL {name} {cfg}/{sp}: {repr(e)[:200]}", flush=True)
    print("PROBE", "PASS" if bad == 0 else f"FAIL ({bad})")
    return bad == 0


# ---------------------------------------------------------------- NQ
class Stream:
    """prep_nq.read_passages와 같은 규칙(빈 글 건너뜀, 단어 수 누적이 want 이상이 되는 줄까지)으로 앞부분을 돌려준다.
    want를 늘려 다시 부르면 이미 읽은 것은 다시 받지 않고 이어서 읽는다."""

    def __init__(self, it):
        self.it, self.rows, self.cum, self.total, self.done, self.t0 = iter(it), [], [], 0, False, time.time()

    def prefix(self, want):
        while self.total < want and not self.done:
            try:
                r = next(self.it)
            except StopIteration:
                self.done = True
                break
            txt = (r.get("text") or "").strip()
            if not txt:
                continue
            self.rows.append((str(fld(r, "_id", "id")), r.get("title") or "", txt))
            self.total += len(txt.split())
            self.cum.append(self.total)
            if len(self.rows) % 100000 == 0:
                log(f"passages read {len(self.rows)}, words {self.total}, {time.time() - self.t0:.0f}s")
        k = bisect.bisect_left(self.cum, want)
        return self.rows[:k + 1]


def run_nq(a, out):
    D = a.nq_data
    meta = json.load(open(f"{D}/meta.json"))
    n_chunks, W, min_tail, n_real = int(meta["n_chunks"]), int(meta["words"]), int(meta["min_tail"]), int(meta["n_real_queries"])
    ok = dict(dataset="nq", ok=False, prep_nq_md5=md5(os.path.join(HERE, "prep_nq.py")), n_chunks=n_chunks, n_real=n_real)
    rep(f"== NQ  data={D} chunks={n_chunks} words={W} min_tail={min_tail} 실제 질문={n_real} prep_nq.py md5={ok['prep_nq_md5']} (전처리 때 cf0ca4d9)")

    # 0) 조각 경계 표시 (저장된 조각 글로 계산, 정답과 무관)
    def it_chunks():
        with open(f"{D}/chunks.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                yield r["doc_idx"], r["text"]
    doc_arr, flags, fst = term_flags(it_chunks())
    np.save(f"{out}/nq_flags.npy", flags)
    ok["flags"] = fst
    rep(f"조각 경계(규칙 A): 문장 끝으로 끝나는 조각 {fst['term_share']*100:.1f}%, 앞 이웃이 있는 조각 중 앞이 잘린 것 "
        f"{fst['start_cut_share']*100:.1f}%, 뒤 이웃이 있는 조각 중 뒤가 잘린 것 {fst['end_cut_share']*100:.1f}%")

    # 1) 질문: prep_nq.read_real_queries와 같은 걸러내기, 같은 순서
    local = []
    with open(f"{D}/queries.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_real:
                break
            local.append(json.loads(line)["text"])
    qrows, qrels, answers = None, {}, {}
    try:
        qrows = [(str(fld(r, "_id", "id")), r["text"].strip()) for r in hf("mteb/nq", "queries", "queries")
                 if (r.get("text") or "").strip()]
        aligned = sum(1 for (_, t), lt in zip(qrows, local) if t == lt)
        ok.update(src_queries=len(qrows), query_aligned=aligned)
        rep(f"질문: 원본 {len(qrows)}개, 우리 실제 질문 {len(local)}개, 같은 자리에 같은 글 {aligned}개")
        qrels = load_qrels("mteb/nq", "test")
        ok["qrels_queries"] = len(qrels)
        rep(f"qrels(test): 질문 {len(qrels)}개, 관련 문단 쌍 {sum(len(v) for v in qrels.values())}개")
    except Exception as e:
        rep("질문/qrels를 읽지 못함:", repr(e)[:300])
    if not a.skip_answers:
        try:
            for r in hf("google-research-datasets/nq_open", "nq_open", "validation"):
                ans = [x for x in (r.get("answer") or []) if str(x).strip()]
                if ans:
                    answers.setdefault(norm_q(r["question"]), ans)
            rep(f"nq_open validation: 정답 문자열이 있는 질문 {len(answers)}개")
        except Exception as e:
            rep("nq_open을 읽지 못함 (정답 문자열 기준은 빠짐):", repr(e)[:300])
    gold_pids = set().union(*qrels.values()) if qrels else set()

    # 2) 코퍼스를 전처리 때와 같은 순서로 다시 읽기
    gmap, mismatch, mism_first, n_checked, struct_bad = {}, 0, [], 0, 0
    corpus_ok = False
    if gold_pids and not a.skip_corpus:
        try:
            S = Stream(hf("mteb/nq", "corpus", "corpus", streaming=True))
            want = int(n_chunks * W * a.oversample)
            while True:
                rows = S.prefix(want)
                arts = P.group_articles([(t, x) for _, t, x in rows])
                texts, doc_idx, chunk_idx = P.build_chunks(arts, n_chunks, W, min_tail)
                if len(texts) >= n_chunks:
                    break
                log(f"only {len(texts)} chunks, reading more (전처리와 같은 규칙)")
                want = int(want * 1.3)
            ok.update(passages_read=len(rows), articles=len(arts), want_words=want)
            rep(f"코퍼스 다시 읽기: 문단 {len(rows)}개, 위키 글 {len(arts)}개, 다시 만든 조각 {len(texts)}개")

            # 문단 -> 위키 글(문서) 안 단어 위치. group_articles와 같은 키 규칙
            order, groups = [], {}
            for i, (pid, title, txt) in enumerate(rows):
                key = title if title else f"__no_title_{i}"
                if key not in groups:
                    groups[key] = []
                    order.append(key)
                groups[key].append((pid, len(txt.split()), txt if pid in gold_pids else None))
            del rows
            n_docs_inc = int(doc_idx[-1]) + 1
            doc_start = np.searchsorted(doc_idx, np.arange(n_docs_inc + 1))
            read_pids = 0
            for d, key in enumerate(order):
                if d < len(arts) and arts[d][0] != key:
                    struct_bad += 1
                plist = groups[key]
                read_pids += len(plist)
                if d >= n_docs_inc:
                    for pid, nw, txt in plist:
                        if pid in gold_pids:
                            gmap[pid] = dict(where="read_not_included")
                    continue
                L = sum(nw for _, nw, _ in plist)
                npc = (L + W - 1) // W
                if npc >= 2 and L - (npc - 1) * W < min_tail:
                    npc -= 1
                if d < n_docs_inc - 1 and doc_start[d] + npc != doc_start[d + 1]:
                    struct_bad += 1
                off = 0
                for pid, nw, txt in plist:
                    if pid in gold_pids:
                        ws, we = off, off + nw
                        ch = []
                        for j in range(min(ws // W, npc - 1), min((we - 1) // W, npc - 1) + 1):
                            cs, ce = j * W, ((j + 1) * W if j < npc - 1 else L)
                            w = min(we, ce) - max(ws, cs)
                            if w > 0:
                                ch.append((int(doc_start[d] + j), int(w)))
                        full = all(c < n_chunks for c, _ in ch)
                        gmap[pid] = dict(where="included" if full else "partial", doc=d, ws=ws, we=we, words=nw,
                                         chunks=[[c, w] for c, w in ch if c < n_chunks], full=full, text=txt)
                    off += nw
            ok["structure_mismatch"] = struct_bad

            # 3) 다시 만든 조각과 chunks.jsonl 전부 대조 + 정답 조각의 저장 글 모으기
            need_c = {c for g in gmap.values() for c, _ in g.get("chunks", [])}
            stored = {}
            with open(f"{D}/chunks.jsonl", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    r = json.loads(line)
                    n_checked += 1
                    if i >= len(texts) or r["text"] != texts[i] or int(r["doc_idx"]) != int(doc_idx[i]) \
                            or int(r.get("chunk_idx", chunk_idx[i])) != int(chunk_idx[i]):
                        mismatch += 1
                        if len(mism_first) < 5:
                            mism_first.append(dict(i=i, stored=r["text"][:80], rebuilt=(texts[i][:80] if i < len(texts) else None)))
                    if i in need_c:
                        stored[i] = r["text"]
            if n_checked != len(texts):
                mismatch += abs(n_checked - len(texts))
            del texts
            ok.update(chunks_checked=n_checked, chunk_mismatch=mismatch, chunk_mismatch_first=mism_first)
            rep(f"조각 대조: {n_checked}개 중 다른 것 {mismatch}개, 문서 경계 계산이 어긋난 곳 {struct_bad}개")
            for m in mism_first:
                rep("  다름:", m)

            # 4) 정답 문단 글이 저장된 조각 글 안에 그대로 있는지
            chk = good = 0
            for pid, g in gmap.items():
                if g.get("full"):
                    chk += 1
                    seg = " ".join(stored[c] for c, _ in g["chunks"])
                    good += norm_ws(g["text"]) in seg
            ok.update(passage_text_checked=chk, passage_text_ok=good)
            rep(f"정답 문단 글 확인: {chk}개 중 저장된 조각 글 안에 그대로 있는 것 {good}개")
            corpus_ok = (mismatch == 0 and struct_bad == 0 and chk > 0 and good == chk)
        except Exception as e:
            rep("코퍼스 다시 읽기 실패 (정답 문단 기준은 빠짐):", repr(e)[:300])

    # 5) 질문별 출력
    where = {}
    for g in gmap.values():
        where[g["where"]] = where.get(g["where"], 0) + 1
    if gold_pids and gmap:
        where["not_read"] = len(gold_pids) - len(gmap)
    n_eval = n_ans = n_both = 0
    recs = []
    for qi in range(len(local)):
        qid, qtext = (qrows[qi] if qrows and qi < len(qrows) else (None, local[qi]))
        aligned = qtext == local[qi]
        golds = qrels.get(qid, set()) if (qid and aligned) else set()
        gl = []
        for pid in sorted(golds):
            g = gmap.get(pid)
            if g and g.get("chunks") is not None and "doc" in g:
                gl.append(dict(pid=pid, full=g["full"], words=g["words"], chunks=g["chunks"]))
        ev = corpus_ok and any(x["full"] for x in gl)
        ans = answers.get(norm_q(local[qi]))
        n_eval += ev
        n_ans += ans is not None
        n_both += ev and ans is not None
        recs.append(dict(qi=qi, qid=qid, q=local[qi], ans=ans, n_qrels=len(golds), gold=gl if corpus_ok else [],
                         status=("eval" if ev else ("misaligned" if not aligned else "out"))))
    with open(f"{out}/nq_gold.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok.update(gold_passages=len(gold_pids), gold_passages_where=where, eval_queries=n_eval, answers_matched=n_ans,
              eval_and_answers=n_both, corpus_ok=corpus_ok,
              ok=bool(corpus_ok and qrows and ok.get("query_aligned") == len(local) and n_eval > 0))
    rep(f"정답 문단 {len(gold_pids)}개의 위치: {where} (included=우리 조각 안에 통째로, partial=일부만, read_not_included=읽었지만 조각 100만 개 밖, not_read=전처리 때 읽지 않은 부분)")
    rep(f"평가 가능한 질문(정답 문단이 통째로 우리 데이터 안에 있음): {n_eval}개 / 실제 질문 {len(local)}개")
    rep(f"정답 문자열이 붙은 질문: {n_ans}개 (평가 가능 + 정답 문자열: {n_both}개)")
    json.dump(ok, open(f"{out}/gold_ok_nq.json", "w"), ensure_ascii=False, indent=1)
    rep(f"NQ 정답 문단 기준 사용: {'가능' if ok['ok'] else '불가 (위 숫자 확인)'}")

    # 6) 눈으로 확인할 샘플 30개
    ev_recs = [r for r in recs if r["status"] == "eval"]
    random.Random(1).shuffle(ev_recs)
    with open(f"{out}/gold_samples_nq.txt", "w", encoding="utf-8") as f:
        f.write("정답 문단(qrels)이 우리 조각 어디에 들어갔는지 눈으로 확인하는 샘플. [조각번호 +겹친단어수] 뒤에 저장된 조각 글.\n")
        f.write("확인할 것: 정답 문단 글이 조각 글들을 이어 붙인 것 안에 그대로 있는가, 질문과 관련 있는가.\n\n")
        for r in ev_recs[:30]:
            f.write(f"### qi={r['qi']} qid={r['qid']}  질문: {r['q']}\n    정답 문자열: {r['ans']}\n")
            for g in r["gold"]:
                if not g["full"]:
                    continue
                gm = gmap[g["pid"]]
                f.write(f"  정답 문단 {g['pid']} (문서 {gm['doc']}, 단어 {gm['ws']}~{gm['we']}, {g['words']}단어)\n")
                f.write(f"    문단 글: {norm_ws(gm['text'])}\n")
                for c, w in g["chunks"]:
                    f.write(f"    [조각 {c} +{w}] {stored.get(c, '?') if corpus_ok else '?'}\n")
            f.write("\n")
    return ok


# ---------------------------------------------------------------- FiQA
def run_fiqa(a, out):
    D = a.fiqa_data
    ok = dict(dataset="fiqa", ok=False)
    rep(f"== FiQA  data={D}")
    texts, docs, cidx = [], [], []
    with open(f"{D}/chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["text"])
            docs.append(int(r["doc_idx"]))
            cidx.append(int(r.get("chunk_idx", 0)))
    doc_arr, flags, fst = term_flags(zip(docs, texts))
    np.save(f"{out}/fiqa_flags.npy", flags)
    ok["flags"] = fst
    rep(f"조각 {len(texts)}개. 조각 경계(규칙 A): 문장 끝으로 끝나는 조각 {fst['term_share']*100:.1f}%, 앞이 잘린 것 "
        f"{fst['start_cut_share']*100:.1f}%, 뒤가 잘린 것 {fst['end_cut_share']*100:.1f}%")
    by_doc = {}
    for i, d in enumerate(docs):
        by_doc.setdefault(d, []).append(i)
    contiguous = all(v == list(range(v[0], v[0] + len(v))) for v in by_doc.values())
    wc = [len(t.split()) for t in texts]
    doc_text = {d: norm_ws(" ".join(texts[i] for i in sorted(v, key=lambda i: cidx[i]))) for d, v in by_doc.items()}
    ok.update(chunks=len(texts), docs=len(by_doc), doc_chunks_contiguous=contiguous)
    try:
        corpus = hf("mteb/fiqa", "corpus", "corpus")
        exact, titled, first = {}, {}, {}
        for r in corpus:
            pid, txt, title = str(fld(r, "_id", "id")), norm_ws(r.get("text") or ""), (r.get("title") or "").strip()
            if not txt:
                continue
            exact.setdefault(txt, []).append(pid)
            if title:
                titled.setdefault(norm_ws(title + " " + txt), []).append(pid)
            first.setdefault(" ".join(txt.split()[:20]), []).append((pid, txt))
        id2docs, how = {}, {"exact": 0, "title+text": 0, "prefix": 0, "none": 0}
        for d, t in doc_text.items():
            ids = exact.get(t)
            k = "exact"
            if not ids:
                ids, k = titled.get(t), "title+text"
            if not ids:
                ids = [pid for pid, full in first.get(" ".join(t.split()[:20]), []) if full.startswith(t)]
                k = "prefix" if ids else "none"
            how[k] += 1
            for pid in ids or []:
                id2docs.setdefault(pid, []).append(d)
        share = 1 - how["none"] / max(len(doc_text), 1)
        ok.update(corpus_docs=len(corpus), doc_match=how, doc_match_share=round(share, 4))
        rep(f"문서 맞추기: 우리 문서 {len(doc_text)}개, 원본 {len(corpus)}개, 방법별 {how} (맞춘 비율 {share*100:.1f}%)")

        local = []
        with open(f"{D}/queries.jsonl", encoding="utf-8") as f:
            for i, line in enumerate(f):
                r = json.loads(line)
                local.append((i, r.get("kind", "real"), r.get("text") or r.get("query") or ""))
        try:
            nqv = np.load(f"{D}/query_vecs.npy", mmap_mode="r").shape[0]
        except Exception:
            nqv = -1
        qmap = {}
        for r in hf("mteb/fiqa", "queries", "queries"):
            qmap.setdefault(norm_ws(r.get("text") or ""), []).append(str(fld(r, "_id", "id")))
        qrels = load_qrels("mteb/fiqa", "test")
        ok.update(local_queries=len(local), query_vecs=nqv, qrels_queries=len(qrels))
        n_eval, recs = 0, []
        for qi, kind, t in local:
            if kind != "real":
                continue
            qids = [q for q in qmap.get(norm_ws(t), []) if q in qrels]
            if not qids:
                continue
            gl = []
            for pid in sorted(set().union(*[qrels[q] for q in qids])):
                for d in id2docs.get(pid, []):
                    ch = [[i, wc[i]] for i in sorted(by_doc[d], key=lambda i: cidx[i])]
                    gl.append(dict(pid=pid, full=True, words=sum(w for _, w in ch), chunks=ch))
            ev = bool(gl) and share >= 0.95
            n_eval += ev
            recs.append(dict(qi=qi, qid=qids[0], q=t, ans=None, n_qrels=len(set().union(*[qrels[q] for q in qids])),
                             gold=gl, status="eval" if ev else "out"))
        with open(f"{out}/fiqa_gold.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        ok.update(test_queries_found=len(recs), eval_queries=n_eval,
                  ok=bool(share >= 0.95 and n_eval > 0 and nqv == len(local)))
        rep(f"질문: queries.jsonl {len(local)}줄(query_vecs {nqv}행), test qrels가 있는 질문 {len(recs)}개, 평가 가능 {n_eval}개")
        ev_recs = [r for r in recs if r["status"] == "eval"]
        random.Random(1).shuffle(ev_recs)
        with open(f"{out}/gold_samples_fiqa.txt", "w", encoding="utf-8") as f:
            for r in ev_recs[:30]:
                f.write(f"### qi={r['qi']} qid={r['qid']}  질문: {r['q']}\n")
                for g in r["gold"][:3]:
                    f.write(f"  관련 문서 {g['pid']} ({g['words']}단어, 조각 {[c for c, _ in g['chunks']]})\n")
                    f.write(f"    {' '.join(texts[c] for c, _ in g['chunks'])[:600]}\n")
                f.write("\n")
    except Exception as e:
        rep("FiQA 원본을 읽지 못함 (FiQA 정답 기준은 빠짐):", repr(e)[:300])
    json.dump(ok, open(f"{out}/gold_ok_fiqa.json", "w"), ensure_ascii=False, indent=1)
    rep(f"FiQA 관련 문서 기준 사용: {'가능' if ok['ok'] else '불가 (위 숫자 확인)'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nq-data", dest="nq_data", default="/workspace/junlim/data_nq")
    ap.add_argument("--fiqa-data", dest="fiqa_data", default="/workspace/junlim/data")
    ap.add_argument("--out", default="/workspace/junlim/gold")
    ap.add_argument("--oversample", type=float, default=1.3, help="prep_nq.py 기본값과 같아야 한다")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--skip-nq", dest="skip_nq", action="store_true")
    ap.add_argument("--skip-fiqa", dest="skip_fiqa", action="store_true")
    ap.add_argument("--skip-answers", dest="skip_answers", action="store_true")
    ap.add_argument("--skip-corpus", dest="skip_corpus", action="store_true")
    a = ap.parse_args()
    if a.probe:
        sys.exit(0 if probe() else 1)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    res = {}
    if not a.skip_nq:
        res["nq"] = run_nq(a, a.out)
    if not a.skip_fiqa:
        try:
            res["fiqa"] = run_fiqa(a, a.out)
        except Exception as e:
            rep("FiQA 단계 실패:", repr(e)[:300])
    rep(f"gold_map 끝 ({time.time() - t0:.0f}s): " + ", ".join(f"{k} ok={v.get('ok')}" for k, v in res.items()))
    with open(f"{a.out}/gold_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT) + "\n")


if __name__ == "__main__":
    main()
