#!/usr/bin/env python3
"""w100_check.py: 새로 자른 데이터(D)가 64단어 데이터(SRC)와 같은 글, 같은 질문인지 확인한다. 아무것도 쓰지 않는다.
  - 문서 수가 같고, 문서마다 단어 수가 같다 (마지막 문서는 64단어 때 중간에서 잘렸을 수 있어 차이를 따로 보여 줌)
  - 문서마다 글의 앞 20단어가 같다
  - 실제 질문(queries.jsonl 앞 n_real_queries줄)의 글과 순서가 같다
사용: python w100_check.py SRC D      (종료 코드 0: 같음, 1: 다름)
"""
import json
import sys


def docs(path):
    w, head = {}, {}
    with open(f"{path}/chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d = int(r["doc_idx"])
            t = r["text"].split()
            w[d] = w.get(d, 0) + len(t)
            if d not in head:
                head[d] = t[:20]
            elif len(head[d]) < 20:
                head[d] += t[:20 - len(head[d])]
    return w, head


def queries(path):
    n = int(json.load(open(f"{path}/meta.json"))["n_real_queries"])
    out = []
    with open(f"{path}/queries.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            out.append(json.loads(line)["text"])
    return out


def main():
    src, d = sys.argv[1], sys.argv[2]
    w0, h0 = docs(src)
    w1, h1 = docs(d)
    ok = True
    print(f"문서 수: {src} {len(w0)}개, {d} {len(w1)}개")
    ok &= len(w0) == len(w1)
    last = max(w0)
    diff_words = [k for k in w0 if k != last and w1.get(k) != w0[k]]
    diff_head = [k for k in w0 if h1.get(k) != h0[k]]
    print(f"단어 수가 다른 문서(마지막 제외): {len(diff_words)}개 {diff_words[:5]}")
    print(f"마지막 문서 {last}: {w0[last]}단어 -> {w1.get(last)}단어")
    print(f"앞 20단어가 다른 문서: {len(diff_head)}개 {diff_head[:5]}")
    print(f"총 단어: {sum(w0.values())} -> {sum(w1.values())}")
    ok &= not diff_words and not diff_head
    q0, q1 = queries(src), queries(d)
    same_q = sum(a == b for a, b in zip(q0, q1))
    print(f"실제 질문: {len(q0)}개 -> {len(q1)}개, 같은 자리에 같은 글 {same_q}개")
    ok &= len(q0) == len(q1) == same_q
    print("CHECK", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
