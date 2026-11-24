# 원래 C2 담당 팀원의 w12 코드(브랜치 w12-c2-prefetching, 코드/pf3_rules.py, select 함수)를 로직 변경 없이 옮김. 통합용 어댑터 작성: 총괄
"""C2 이웃 선택 어댑터 (등록 이름: c2.impl = bc1_v1).

- 아래 `nbrs`, `_cut_nbrs`, `select`는 pf3_rules.py 원본을 한 줄도 바꾸지 않고 옮긴 것이다
  (원본: `git show origin/w12-c2-prefetching:코드/pf3_rules.py`, /workspace/jaeeun/team_ref).
- 어댑터는 계약(interfaces.C2NeighborAdapter)에 맞춰 감싸기만 한다.
  · 생성자 (meta, cfg): doc_idx, flags는 ChunkMeta에서 읽는다.
  · 정책 none / BC1 / all만 받는다(원본의 A·B·C1·C3 등 내부 규칙은 계약 밖이라 열지 않음).
  · 입력·출력은 chunk_id. PK 변환은 하지 않는다(C1 몫).
  · 반환은 입력 R에 없는 같은 문서 ±1 이웃, chunk_id 오름차순 list[int] (원본 select 반환과 같음).
- 대조군(ANN Top-(10+k), R3)은 이 모듈이 아니라 실험 계층(D8).
"""
from __future__ import annotations

from typing import Sequence

from interfaces import ChunkMeta, NeighborPolicy
from modules.registry import register

POLICIES = ("none", "BC1", "all")


# ---------------------------------------------------------------------------
# pf3_rules.py 원본 (변경 없음) ---------------------------------------------
# ---------------------------------------------------------------------------
RULES = ("none", "A", "A3", "B", "C1", "C3", "BC1", "all")


def nbrs(c, doc):
    out = []
    for kk in (c - 1, c + 1):
        if 0 <= kk < len(doc) and doc[kk] == doc[c]:
            out.append(kk)
    return out


def _cut_nbrs(c, doc, flags):
    out = []
    f = int(flags[c])
    if f & 1 and c - 1 >= 0 and doc[c - 1] == doc[c]:
        out.append(c - 1)
    if f & 2 and c + 1 < len(doc) and doc[c + 1] == doc[c]:
        out.append(c + 1)
    return out


def select(rule, R, doc, flags=None):
    """R: 결과 조각 번호(순위 순). doc: 조각별 문서 번호 배열. flags: 조각 경계 표시 배열(규칙 A만 사용)."""
    Rs = set(int(c) for c in R)
    out = set()
    if rule == "none":
        return []
    if rule in ("A", "A3"):
        if flags is None:
            raise ValueError("rule A needs flags")
        src = R if rule == "A" else R[:3]
        for c in src:
            out.update(_cut_nbrs(int(c), doc, flags))
    elif rule in ("B", "BC1"):
        cnt = {}
        for c in R:
            cnt[int(doc[c])] = cnt.get(int(doc[c]), 0) + 1
        for c in R:
            if cnt[int(doc[c])] >= 2:
                out.update(nbrs(int(c), doc))
        if rule == "BC1" and len(R):
            out.update(nbrs(int(R[0]), doc))
    elif rule in ("C1", "C3"):
        for c in R[:1 if rule == "C1" else 3]:
            out.update(nbrs(int(c), doc))
    elif rule == "all":
        for c in R:
            out.update(nbrs(int(c), doc))
    else:
        raise ValueError(f"unknown rule {rule}")
    return sorted(out - Rs)
# ---------------------------------------------------------------------------
# 원본 끝 ---------------------------------------------------------------------
# ---------------------------------------------------------------------------


@register("c2", "bc1_v1")
class C2BC1:
    """pf3_rules.select를 감싼 C2 어댑터. none / BC1 / all."""

    def __init__(self, meta: ChunkMeta, cfg: dict | None = None):
        self.meta = meta
        self.doc = meta.doc_idx      # chunk_id → 문서 번호 (pf3 select의 doc)
        self.flags = meta.flags      # 규칙 A용. none/BC1/all에서는 쓰지 않지만 원본 인자 그대로 넘김
        self.n = int(meta.n_chunks)
        window = ((cfg or {}).get("c2") or {}).get("neighbor_window", 1)
        if window != 1:
            raise ValueError(f"c2.neighbor_window={window!r}: pf3 규칙은 ±1(expand=1)만 지원")

    def select_neighbors(self, ranked_chunk_ids: Sequence[int], *, policy: NeighborPolicy) -> list[int]:
        if policy not in POLICIES:
            raise ValueError(f"알 수 없는 policy: {policy!r} (none / BC1 / all)")
        R = [int(c) for c in ranked_chunk_ids]   # 입력 복사(원본 목록 수정 안 함), 순위 순 유지
        for c in R:
            if not 0 <= c < self.n:
                raise ValueError(f"chunk_id 범위 밖: {c} (0 ~ {self.n - 1})")
        return [int(x) for x in select(policy, R, self.doc, self.flags)]

    def describe(self) -> dict:
        return {"impl": "bc1_v1",
                "source": "w12-c2-prefetching:코드/pf3_rules.py select (로직 변경 없음)",
                "policies": list(POLICIES), "neighbor_window": 1}
