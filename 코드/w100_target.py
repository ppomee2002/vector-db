#!/usr/bin/env python3
"""w100_target.py: 64단어 데이터(data_nq)와 똑같은 글을 W단어로 다시 자를 때 필요한 값을 계산한다. 아무것도 쓰지 않는다.
  N  = data_nq의 문서들을 W단어(끝 조각이 min_tail보다 짧으면 앞에 붙임)로 자를 때 나오는 Chunk 수
       (prep_nq.build_chunks는 Chunk가 N개가 되는 문서에서 멈추므로, 같은 문서들이 들어간다)
  OV = prep_nq가 읽을 문단 양(want = N * W * OV)을 64단어 때와 똑같이 맞추는 oversample 값
       (want가 다르면 같은 제목의 문단이 더/덜 붙어 문서 글이 달라질 수 있다)
사용: python w100_target.py SRC_DIR W MIN_TAIL WANT0      -> 표준 출력 한 줄: "N OV"  (설명은 표준 오류로)
  WANT0: 64단어 때 읽은 단어 수 (gold_ok_nq.json의 want_words, 11/16 기록 83200000)
"""
import json
import sys


def n_pieces(L, W, MT):
    """prep_nq.chunk_words와 같은 규칙으로 L단어 글의 조각 수"""
    if L <= 0:
        return 0
    npc = (L + W - 1) // W
    if npc >= 2 and L - (npc - 1) * W < MT:
        npc -= 1
    return npc


def main():
    src, W, MT, want0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    words, n_src = {}, 0
    with open(f"{src}/chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d = int(r["doc_idx"])
            words[d] = words.get(d, 0) + len(r["text"].split())
            n_src += 1
    docs = sorted(words)
    assert docs == list(range(len(docs))), "doc_idx가 0부터 연속이 아님"
    N = sum(n_pieces(words[d], W, MT) for d in docs)
    # 마지막 문서가 64단어 때 중간에서 잘렸을 수 있다. 그 문서는 앞부분 n_pieces(잘린 길이)개만 들어간다.
    ov = (want0 + 0.5) / (N * W)
    ov_s = repr(ov)
    want = int(N * W * float(ov_s))
    assert want == want0, f"want 맞추기 실패: {want} != {want0}"
    meta = json.load(open(f"{src}/meta.json"))
    print(f"src={src} 조각 {n_src}개 ({meta.get('words')}단어) 문서 {len(docs)}개, 총 단어 {sum(words.values())}, "
          f"마지막 문서 단어 {words[docs[-1]]} -> {W}단어(끝 조각 기준 {MT}) 조각 {N}개, oversample {ov_s} (want {want})",
          file=sys.stderr)
    print(N, ov_s)


if __name__ == "__main__":
    main()
