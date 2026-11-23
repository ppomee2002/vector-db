"""품질 평가 (공통 코어): 정답 문자열 포함률.

정의 (c2x 평가기 / analyze_pf3.segments와 같은 방식):
- 정규화 toks(s): NFD → 결합 문자 제거 → 소문자 → \\w+ 토큰을 공백으로 연결.
- 최종 Context = fetched_ids를 같은 문서 안에서 연속인 것끼리 묶은 구간(segment)들. 구간 본문 = 구간 chunk들의 toks를 이어 붙인 것.
- hit = 정답 문자열(정규화, 앞뒤 공백 패딩) 중 하나가 어떤 구간 본문(앞뒤 공백 패딩)에 들어 있음.
- 평가 대상 = 정규화 후 비어 있지 않은 정답이 하나 이상 있는 질문 (nq_test_ans 2,254개가 이 기준).
본문은 평가할 때만 쓰고 결과 파일에는 넣지 않는다.
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Iterable, Mapping, Sequence

import numpy as np

_WORD = re.compile(r"\w+", re.UNICODE)


def toks(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return " ".join(_WORD.findall(s.lower()))


def norm_answers(ans: Sequence[str] | None) -> list[str]:
    if not ans:
        return []
    return [" " + t + " " for t in (toks(a) for a in ans) if t]


def segments(chunk_ids: Iterable[int], texts: Mapping[int, str], doc_idx: np.ndarray) -> list[str]:
    out, cur = [], []
    for c in sorted(chunk_ids):
        if cur and c == cur[-1] + 1 and doc_idx[c] == doc_idx[cur[-1]]:
            cur.append(c)
        else:
            if cur:
                out.append(cur)
            cur = [c]
    if cur:
        out.append(cur)
    return [" " + " ".join(toks(texts[c]) for c in seg) + " " for seg in out]


def answer_hit(chunk_ids: Iterable[int], texts: Mapping[int, str], doc_idx: np.ndarray,
               answers: Sequence[str] | None) -> bool | None:
    """None = 평가 대상 아님(정답 없음)."""
    A = norm_answers(answers)
    if not A:
        return None
    segs = segments(chunk_ids, texts, doc_idx)
    return any(a in s for s in segs for a in A)


def hit_rate(flags: Iterable[bool | None]) -> dict:
    v = [f for f in flags if f is not None]
    return {"n_eval": len(v), "n_hit": int(sum(v)), "hit_rate": (sum(v) / len(v)) if v else None}


def read_chunks_jsonl(path: str, ids: Iterable[int]) -> dict[int, str]:
    """chunks.jsonl에서 필요한 줄만 읽어 {chunk_id: text}. 원문 대조용(읽기만)."""
    want = set(int(i) for i in ids)
    out: dict[int, str] = {}
    if not want:
        return out
    last = max(want)
    with open(path, "rb") as f:
        for i, line in enumerate(f):
            if i in want:
                out[i] = json.loads(line)["text"]
            if i >= last:
                break
    return out
