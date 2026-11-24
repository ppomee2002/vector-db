"""C1 어댑터 (김지현, C1 담당). 문서 순 배치(Layout) 비교용 본문 Fetch. 계약: interfaces.C1FetchAdapter.

등록 이름: c1.impl = layout_jihyeon

설정 (config 또는 runs/*.yaml overlay)
  c1.layout   : seq / doc / rand          어느 배치에서 읽을지
  c1.backend  : milvus (기본) / file
  c1.fetch_mode, c1.batch_impl            실행기가 fetch_text 인자로 넘김
    - fetch_mode : batch (한 번에) / individual (chunk마다 따로). range는 D7에 따라 오류
    - batch_impl : query_pk_in (query(filter="pk in [...]"), B0와 같음) / get (get(ids), w13 실험과 같음). milvus에서만 의미 있음
  milvus 경로
    - layout seq      → milvus.fetch.{database, collection}            (junlim/nq_seq, 검색 컬렉션과 같음)
    - layout doc/rand → milvus.c1_compare.{database, doc/rand.collection} (jihyeon/w13c1_doc, w13c1_rand)
    - PK 변환: ChunkMeta.pk_of_chunk[layout] (doc·seq 항등, rand = pk_rand.npy). 반환 키는 chunk_id.
    - 컬렉션에 chunk_id 필드가 있으면(w13c1_*) 돌아온 행의 chunk_id로 PK 변환을 한 번 더 확인한다.
    - 요청당 PK 최대 512개 (milvus_query_ref, bench_v2와 같음).
    - load/release, 컬렉션 설정 변경은 하지 않는다. milvus_collections()로 선언만 하고 실행기(MilvusGuard)가 처리.
  file 경로
    - 데이터 파일: dataset.file_store[layout].data, 위치·길이: ChunkMeta.offset[layout], ChunkMeta.length
    - 4KB 블록으로 바꿔 O_DIRECT로 읽음. batch = 붙은 블록을 pread 한 번(묶어 읽기), individual = chunk마다 pread.
    - c1.file_threads (필수, 정수): 전용 스레드 풀 크기. 암묵적 기본 풀을 쓰지 않는다(숨은 동시성 상한 방지).
    - c1.file_direct (기본 true): false면 페이지 캐시 경로(로컬 테스트용).
    - FetchStats.blocks = 읽은 4KB 블록 수, device_bytes = 읽은 바이트(블록 단위).

동시 호출: fetch_text가 동시에 여러 번 불려도 결과가 섞이지 않는다. 호출마다 지역 변수와 자기 버퍼만 쓰고,
          공유하는 것은 읽기 전용 ChunkMeta와 Milvus 클라이언트(내부적으로 동시 요청 지원)뿐이다.
원본 파일은 읽기 전용(O_RDONLY)으로만 연다.
"""
from __future__ import annotations

import asyncio
import mmap
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Mapping, Sequence

from interfaces import BatchImpl, ChunkMeta, FetchMode, FetchStats
from modules.registry import register

CHUNK_LIMIT = 512
BLOCK = 4096


@register("c1", "layout_jihyeon")
class C1LayoutJihyeon:
    def __init__(self, meta: ChunkMeta, cfg: dict, *, client_factory=None):
        c1 = cfg["c1"]
        self.layout = c1.get("layout", "seq")
        self.backend = c1.get("backend", "milvus")
        if self.layout not in ("seq", "doc", "rand"):
            raise ValueError(f"c1.layout은 seq/doc/rand: {self.layout!r}")
        if self.backend not in ("milvus", "file"):
            raise ValueError(f"c1.backend는 milvus/file: {self.backend!r}")
        if self.layout not in meta.pk_of_chunk or self.layout not in meta.chunk_of_pk:
            raise ValueError(f"ChunkMeta에 layout={self.layout} PK 매핑이 없음 (layouts에 넣어 로드해야 함)")
        self.meta = meta
        self.pk_of = meta.pk_of_chunk[self.layout]
        self.chunk_of = meta.chunk_of_pk[self.layout]
        self._client = None
        self._lock: asyncio.Lock | None = None
        self._schema_ok = False
        self._has_chunk_id = False
        self._factory = client_factory
        self._pool = None
        self._fd = None

        if self.backend == "milvus":
            m = cfg["milvus"]
            self.uri = m["uri"]
            if self.layout == "seq":
                self.db, self.collection = m["fetch"]["database"], m["fetch"]["collection"]
                self.text_field = m["fetch"].get("text_field", "text")
                self.pk_field = m["fetch"].get("pk_field", "pk")
            else:
                cc = m["c1_compare"]
                self.db, self.collection = cc["database"], cc[self.layout]["collection"]
                self.text_field = cc.get("text_field", "text")
                self.pk_field = cc.get("pk_field", "pk")
        else:
            threads = c1.get("file_threads")
            if not isinstance(threads, int) or threads < 1:
                raise ValueError("c1.backend=file이면 c1.file_threads(1 이상 정수)를 반드시 적어야 함 (암묵적 기본 풀 금지)")
            if meta.offset is None or self.layout not in meta.offset or meta.length is None:
                raise ValueError(f"ChunkMeta에 layout={self.layout} offset/length가 없음")
            self.threads = threads
            self.direct = bool(c1.get("file_direct", True))
            self.path = cfg["dataset"]["file_store"][self.layout]["data"]
            self.off = meta.offset[self.layout]
            self.length = meta.length
            self._fd = os.open(self.path, os.O_RDONLY | (os.O_DIRECT if self.direct else 0))
            self._pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="c1file")

    # ------------------------------------------------------------ 계약 훅
    @property
    def name(self) -> str:
        return f"{self.db}/{self.collection}" if self.backend == "milvus" else self.path

    def fetch_source(self) -> str:
        return f"milvus:{self.name}" if self.backend == "milvus" else f"file:{self.path}"

    def milvus_collections(self) -> list[str]:
        return [self.name] if self.backend == "milvus" else []

    def describe(self) -> dict:
        d = {"impl": "layout_jihyeon", "layout": self.layout, "backend": self.backend, "source": self.fetch_source()}
        if self.backend == "milvus":
            d.update(text_field=self.text_field, pk_field=self.pk_field, chunk_limit=CHUNK_LIMIT,
                     verify_chunk_id_field=self._has_chunk_id)
        else:
            d.update(threads=self.threads, o_direct=self.direct, block=BLOCK)
        return d

    async def aclose(self):
        if self._client is not None:
            try:
                await self._client.close()
            finally:
                self._client = None
                self._lock = None
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    async def fetch_text(self, chunk_ids: Sequence[int], *, mode: FetchMode,
                         batch_impl: BatchImpl) -> tuple[Mapping[int, str], FetchStats]:
        if mode == "range":
            raise ValueError("fetch_mode=range는 1차 제외(D7)")
        if mode not in ("batch", "individual"):
            raise ValueError(f"알 수 없는 fetch_mode: {mode!r}")
        ids = list(dict.fromkeys(int(c) for c in chunk_ids))
        if not ids:
            return {}, FetchStats(n_requests=0, n_pks=0, bytes_returned=0,
                                  **({"blocks": 0, "device_bytes": 0} if self.backend == "file" else {}))
        if self.backend == "milvus":
            return await self._fetch_milvus(ids, mode, batch_impl)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, self._read_file, ids, mode == "batch")

    # ------------------------------------------------------------ Milvus
    async def _ready(self):
        if self._client is not None and self._schema_ok:
            return self._client
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._client is None:
                if self._factory is not None:
                    self._client = self._factory(uri=self.uri, db_name=self.db)
                else:
                    from pymilvus import AsyncMilvusClient
                    self._client = AsyncMilvusClient(uri=self.uri, db_name=self.db)
            if not self._schema_ok:
                desc = await self._client.describe_collection(self.collection)
                names = [fd["name"] for fd in desc.get("fields", [])]
                for f in (self.text_field, self.pk_field):
                    if f not in names:
                        raise ValueError(f"컬렉션 {self.name}에 필드 '{f}'가 없음 (필드: {names})")
                self._has_chunk_id = "chunk_id" in names
                self._schema_ok = True
        return self._client

    async def _fetch_milvus(self, ids, mode, batch_impl):
        if batch_impl not in ("query_pk_in", "get"):
            raise ValueError(f"batch_impl은 query_pk_in/get: {batch_impl!r}")
        pk_to_cid = {int(self.pk_of[c]): c for c in ids}
        pks = sorted(pk_to_cid)
        parts = [pks[i:i + CHUNK_LIMIT] for i in range(0, len(pks), CHUNK_LIMIT)] if mode == "batch" else [[p] for p in pks]
        cli = await self._ready()
        fields = [self.pk_field, self.text_field] + (["chunk_id"] if self._has_chunk_id else [])
        out: dict[int, str] = {}
        n_req = 0
        for part in parts:
            if batch_impl == "query_pk_in":
                rows = await cli.query(self.collection, filter=f"{self.pk_field} in [{','.join(map(str, part))}]",
                                       output_fields=fields)
            else:
                rows = await cli.get(self.collection, ids=part, output_fields=fields)
            n_req += 1
            for r in rows:
                pk = int(r[self.pk_field])
                cid = pk_to_cid.get(pk)
                if cid is None:
                    continue
                if int(self.chunk_of[pk]) != cid:
                    raise RuntimeError(f"PK 역매핑 불일치 pk {pk}: chunk_of_pk {int(self.chunk_of[pk])} != {cid}")
                if self._has_chunk_id and int(r["chunk_id"]) != cid:
                    raise RuntimeError(f"컬렉션 chunk_id 필드 불일치 pk {pk}: {int(r['chunk_id'])} != {cid}")
                out[cid] = r[self.text_field]
        nbytes = sum(len(t.encode("utf-8")) for t in out.values())
        return out, FetchStats(n_requests=n_req, n_pks=len(pks), bytes_returned=nbytes)

    # ------------------------------------------------------------ 파일
    def _span(self, cid):
        a = int(self.off[cid])
        return a // BLOCK, (a + int(self.length[cid]) - 1) // BLOCK

    def _read_file(self, ids, merge):
        if merge:
            bs = set()
            for c in ids:
                s, e = self._span(c)
                bs.update(range(s, e + 1))
            bl = sorted(bs)
            ext, s, pv = [], bl[0], bl[0]
            for b in bl[1:]:
                if b == pv + 1:
                    pv = b
                    continue
                ext.append((s, pv))
                s = pv = b
            ext.append((s, pv))
        else:
            ext = [self._span(c) for c in ids]
        nblocks = sum(e - s + 1 for s, e in ext)
        buf = mmap.mmap(-1, max(nblocks, 1) * BLOCK)      # 호출마다 자기 버퍼 (페이지 정렬, O_DIRECT 조건)
        try:
            mv = memoryview(buf)
            pos, where = 0, []
            for s, e in ext:
                nb = (e - s + 1) * BLOCK
                os.preadv(self._fd, [mv[pos:pos + nb]], s * BLOCK)
                where.append((s, e, pos))
                pos += nb
            out: dict[int, str] = {}
            for i, c in enumerate(ids):
                a, ln = int(self.off[c]), int(self.length[c])
                if merge:
                    b = a // BLOCK
                    s, e, p = next(w for w in where if w[0] <= b <= w[1])
                else:
                    s, e, p = where[i]
                st = p + a - s * BLOCK
                out[c] = bytes(mv[st:st + ln]).decode("utf-8")
            mv.release()
        finally:
            buf.close()
        nbytes = sum(len(t.encode("utf-8")) for t in out.values())
        return out, FetchStats(n_requests=len(ext), n_pks=len(ids), bytes_returned=nbytes,
                               blocks=nblocks, device_bytes=nblocks * BLOCK)
