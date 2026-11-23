"""[reference, 담당자 구현으로 교체] C2 이웃 선택 자리 표시용 어댑터.

- 등록 이름: c2.impl = none_ref
- policy=none만 지원(항상 빈 목록). BC1, all은 NotImplementedError("C2 담당 구현 필요").
- 계약(interfaces.C2NeighborAdapter): 생성자에서 ChunkMeta 주입, 반환은 입력 R에 없는 같은 문서 ±1 이웃,
  chunk_id 오름차순 list[int]. 대조군(ANN Top-(10+k), R3)은 이 모듈이 아니라 실험 계층(D8).
"""
from __future__ import annotations

from typing import Sequence

from interfaces import ChunkMeta, NeighborPolicy
from modules.registry import register


@register("c2", "none_ref")
class C2NoneReference:
    """reference, 담당자 구현으로 교체. none만 지원."""

    def __init__(self, meta: ChunkMeta, cfg: dict | None = None):
        self.meta = meta

    def select_neighbors(self, ranked_chunk_ids: Sequence[int], *, policy: NeighborPolicy) -> list[int]:
        if policy == "none":
            return []
        if policy in ("BC1", "all"):
            raise NotImplementedError("C2 담당 구현 필요")
        raise ValueError(f"알 수 없는 policy: {policy!r}")

    def describe(self) -> dict:
        return {"impl": "none_ref", "note": "reference, 담당자 구현으로 교체 (none만 지원)"}
