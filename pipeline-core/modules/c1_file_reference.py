"""[reference, 담당자 구현으로 교체] C1 Fetch 자리 표시용 어댑터 (파일 저장소).

- 등록 이름: c1.impl = file_ref
- 저장소: config dataset.file_store.seq (chunk_store_seq.bin + chunk_store_off.npy). seq 레이아웃만 지원.
- 읽기: chunk마다 os.pread 한 번(묶어 읽기·병합 같은 최적화 없음). mode/batch_impl은 받기만 하고
  동작은 같다(range는 D7에 따라 오류).
- PK 변환: ChunkMeta.pk_of_chunk["seq"](항등)로 C1 내부에서만. 반환 dict 키는 chunk_id.
- FetchStats: n_requests(= pread 횟수), n_pks, bytes_returned(= 반환 본문의 UTF-8 바이트 합, 계약 정의)만 채움.
  blocks, device_bytes는 None.
- fetch_source(): "file:<chunk_store_seq.bin 경로>".
- 스레드 (C3 검토 반영): 기본 스레드 풀(asyncio.to_thread 기본값, min(32, CPU+4))에 의존하지 않는다.
  cfg c1.file_ref_threads(정수 ≥ 1)로 전용 ThreadPoolExecutor를 만든다. 값이 없으면(null) 생성자에서 ValueError → 실행 거부.
  이 스레드 수가 곧 파일 Fetch의 동시 실행 상한이므로 실행 메타(c1_threads, adapters.c1.threads)에 남긴다.
  C3 비교에 file_ref를 쓰지 않는 이유: 스레드 풀 상한이 C3 동시성 제어와 별개로 숨은 상한이 되기 때문(DESIGN.md 4절).
- Milvus를 쓰지 않는다. 원본 파일은 읽기 전용으로 연다.
"""
from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
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
        threads = cfg["c1"].get("file_ref_threads")
        if type(threads) is not int or threads < 1:
            raise ValueError(f"c1.file_ref_threads가 없거나 잘못됨({threads!r}). 암묵적 기본값 없음: 실행 설정에 정수로 적을 것 "
                             f"(B0 정확성 확인은 8)")
        if meta.offset is None or "seq" not in meta.offset or meta.length is None:
            raise ValueError("ChunkMeta에 seq offset/length가 없음 (layouts=('seq',)로 로드해야 함)")
        self.path = cfg["dataset"]["file_store"]["seq"]["data"]
        self.meta = meta
        self.pk_of = meta.pk_of_chunk["seq"]
        self.off = meta.offset["seq"]
        self.length = meta.length
        self.fd = os.open(self.path, os.O_RDONLY)
        self.threads = threads
        self.pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="c1_file_ref")

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
        return await asyncio.get_running_loop().run_in_executor(self.pool, self._read, ids)

    def fetch_source(self) -> str:
        return f"file:{self.path}"

    async def aclose(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def describe(self) -> dict:
        return {"impl": "file_ref", "source": self.fetch_source(), "layout": self.LAYOUT, "threads": self.threads,
                "note": "reference, 담당자 구현으로 교체 (chunk별 pread, 최적화 없음)"}
