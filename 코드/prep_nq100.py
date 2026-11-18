#!/usr/bin/env python3
"""prep_nq100.py: prep_nq.py(cf0ca4d9)를 그대로 쓰되, 임베딩 최대 길이(max_seq_length)만 MAX_SEQ(기본 256)로 바꾼다.
  근거: all-MiniLM-L6-v2 모델 카드 "By default, input text longer than 256 word pieces is truncated."
  prep_nq.py는 128로 두었다(64단어 조각용). 100단어 조각은 끝 조각을 붙이면 최대 149단어라 128 토큰을 넘는 경우가 많아,
  잘리지 않도록 모델 기본값 256을 쓴다. 조각 나누기, 문서 묶기, 질문, 출력 형식은 prep_nq.py와 같다
  (gold_map.py도 prep_nq.py의 group_articles, build_chunks를 그대로 쓰므로 조각 대조가 맞는다).
안전장치: --out을 꼭 줘야 하고, 64단어 데이터 폴더(/workspace/junlim/data_nq)나 meta.json이 이미 있는 폴더에는 쓰지 않는다.
사용: CUDA_VISIBLE_DEVICES=0 MAX_SEQ=256 python prep_nq100.py --out DIR --chunks N --words 100 --min-tail 50 --oversample OV --pseudo-q 0
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep_nq as P  # noqa: E402

MAX_SEQ = int(os.environ.get("MAX_SEQ", "256"))


def out_dir(argv):
    for i, x in enumerate(argv):
        if x == "--out" and i + 1 < len(argv):
            return argv[i + 1]
        if x.startswith("--out="):
            return x.split("=", 1)[1]
    return None


_orig_load_model = P.load_model


def load_model(a):
    m = _orig_load_model(a)
    m.max_seq_length = MAX_SEQ
    P.log(f"max_seq_length = {m.max_seq_length}")
    return m


P.load_model = load_model

if __name__ == "__main__":
    out = out_dir(sys.argv[1:])
    if not out:
        raise SystemExit("--out을 꼭 지정해야 함 (기본값이 64단어 데이터 폴더라서)")
    if os.path.realpath(out) == os.path.realpath("/workspace/junlim/data_nq"):
        raise SystemExit("64단어 데이터 폴더에는 쓰지 않음")
    if os.path.exists(os.path.join(out, "meta.json")):
        raise SystemExit(f"{out}/meta.json이 이미 있음 (끝난 데이터). 지우거나 다른 폴더를 쓸 것")
    P.main()
    mp = os.path.join(out, "meta.json")
    meta = json.load(open(mp))
    meta.update(max_seq_length=MAX_SEQ, prep="prep_nq100.py (prep_nq.py + max_seq_length)")
    json.dump(meta, open(mp, "w"), ensure_ascii=False, indent=1)
    P.log("meta updated:", {k: meta[k] for k in ("n_chunks", "n_docs", "words", "min_tail", "max_seq_length")})
