"""[reference, 담당자 구현으로 교체] C1 Fetch 자리 표시용 어댑터 (파일 저장소).

- 등록 이름: c1.impl = file_ref
- 저장소: config dataset.file_store.seq (chunk_store_seq.bin + chunk_store_off.npy). seq 레이아웃만 지원.
- 읽기: chunk마다 os.pread 한 번(묶어 읽기·병합 같은 최적화 없음). mode/batch_impl은 받기만 하고
  동작은 같다(range는 D7에 따라 오류).
- PK 변환: ChunkMeta.pk_of_chunk["seq"](항등)로 C1 내부에서만. 반환 dict 키는 chunk_id.
- FetchStats: n_requests(= pread 횟수), n_pks, bytes_returned(= 반환 본문의 UTF-8 바이트 합, 계약 정의)만 채움.
  blocks, device_bytes는 None.
- fetch_source(): "file:<chunk_store_seq.bin 경로>".
- Milvus를 쓰지 않는다. 원본 파일은 읽기 전용으로 연다.
"""
from __future__ import annotations

import asyncio
import os
from typing import Mapping, Sequence

from interfaces import BatchImpl, ChunkMeta, FetchMode, FetchStats
from modules.registry import register


@register("c1", "file_ref")
class C1FileReference:
    """reference, 담당자 구현으로 교체. 파일 저장소에서 chunk별 단순 읽기."""

    LAYOUT = "seq"

    def __init__(self, meta: ChunkMeta, cfg: dict):
        layout = cfg["c1"].get("layout", "seq")
        if layout != self.LAYOUT:
            raise NotImplementedError(f"file_ref는 seq만 지원 (요청 layout={layout}). C1 담당 구현 필요")
        if meta.offset is None or "seq" not in meta.offset or meta.length is None:
            raise ValueError("ChunkMeta에 seq offset/length가 없음 (layouts=('seq',)로 로드해야 함)")
        self.path = cfg["dataset"]["file_store"]["seq"]["data"]
        self.meta = meta
        self.pk_of = meta.pk_of_chunk["seq"]
        self.off = meta.offset["seq"]
        self.length = meta.length
        self.fd = os.open(self.path, os.O_RDONLY)

    def _read(self, chunk_ids: Sequence[int]) -> tuple[dict[int, str], FetchStats]:
        out: dict[int, str] = {}
        nbytes = 0
        for cid in chunk_ids:
            cid = int(cid)
            pk = int(self.pk_of[cid])  # 파일 저장소는 pk(=줄 번호) 순서로 저장됨
            text = os.pread(self.fd, int(self.length[pk]), int(self.off[pk])).decode("utf-8")
            out[cid] = text
            nbytes += len(text.encode("utf-8"))   # 계약: 반환 본문 텍스트의 UTF-8 바이트 합
        return out, FetchStats(n_requests=len(chunk_ids), n_pks=len(chunk_ids), bytes_returned=nbytes)

    async def fetch_text(self, chunk_ids: Sequence[int], *, mode: FetchMode,
                         batch_impl: BatchImpl) -> tuple[Mapping[int, str], FetchStats]:
        if mode == "range":
            raise ValueError("fetch_mode=range는 1차 제외(D7)")
        if mode not in ("batch", "individual"):
            raise ValueError(f"알 수 없는 fetch_mode: {mode!r}")
        ids = list(dict.fromkeys(int(c) for c in chunk_ids))  # 중복 제거, 순서 유지
        return await asyncio.to_thread(self._read, ids)

    def fetch_source(self) -> str:
        return f"file:{self.path}"

    async def aclose(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def describe(self) -> dict:
        return {"impl": "file_ref", "source": self.fetch_source(), "layout": self.LAYOUT,
                "note": "reference, 담당자 구현으로 교체 (chunk별 pread, 최적화 없음)"}
