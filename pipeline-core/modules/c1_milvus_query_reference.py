"""B0 기본 Fetch(최적화 없음). C1 기여 구현은 C1 담당자 어댑터.

[reference] Milvus에서 본문을 읽는 기본 C1 어댑터.
- 등록 이름: c1.impl = milvus_query_ref
- 컬렉션: cfg milvus.fetch.{database, collection, layout, text_field(기본 text), pk_field(기본 pk)}.
  C3 공식 비교(warm)는 이 어댑터를 쓴다(C1 구현을 기다리지 않아도 됨). C1 Layout 비교는 C1 담당 어댑터.
- 조회: 실제 AsyncMilvusClient의 query(filter="pk in [...]", output_fields=[pk, text]).
  기존 C3 bench_v2.py(MilvusBackend.afetch, CHUNK_LIMIT = 512)와 같이 PK를 최대 512개씩 나눠 순서대로 요청한다.
  - fetch_mode=batch: 정렬한 PK를 512개씩 묶어 요청(질문당 보통 1회).
  - fetch_mode=individual: PK마다 query 1회(순서대로, 동시 요청 없음).
  - fetch_mode=range: 오류(D7). batch_impl은 query_pk_in만 지원(get은 C1 비교 항목 → C1 담당 어댑터).
  묶어 읽기·병합·캐시 같은 최적화는 하지 않는다.
- chunk_id ↔ PK 변환은 이 어댑터 안에서만: pk = ChunkMeta.pk_of_chunk[layout][chunk_id],
  돌아온 pk → chunk_id = ChunkMeta.chunk_of_pk[layout][pk]. 반환 dict 키는 chunk_id.
- 본문은 Milvus text 필드 값 그대로. 요청하지 않은 PK가 오면 버리고, 빠진 chunk_id는 반환 dict에서 빠진다
  (실행기가 오류로 기록).
- FetchStats: n_requests = query 호출 수, n_pks = 요청한 PK 수(중복 제거), bytes_returned = 반환 본문의 UTF-8 바이트 합.
  blocks, device_bytes는 None(Milvus 경로에서는 질문 단위로 얻을 수 없음, D11).
- 첫 fetch 때 describe_collection으로 text·pk 필드를 확인한다. text 필드가 없는 컬렉션(예: jaeeun/nq1m_sel)이면
  명확한 오류(ValueError)를 낸다.
- milvus_collections()로 컬렉션을 선언 → 실행기(MilvusGuard)가 사용 중 검사 후 필요할 때만 load·release.
  이 어댑터는 load/release를 하지 않는다. 컬렉션 설정(mmap·인덱스)도 바꾸지 않는다.
- 생성자 규약 (meta, cfg). AsyncMilvusClient는 실행 중인 이벤트 루프 안에서 처음 쓸 때 만든다(aclose 후 다시 만들 수 있음).
"""
from __future__ import annotations

import asyncio
from typing import Mapping, Sequence

from interfaces import BatchImpl, ChunkMeta, FetchMode, FetchStats
from modules.registry import register

CHUNK_LIMIT = 512   # query 한 번에 넣는 PK 수 상한 (기존 C3 bench_v2.py와 같음)


@register("c1", "milvus_query_ref")
class C1MilvusQueryReference:
    """B0 기본 Fetch(최적화 없음). C1 기여 구현은 C1 담당자 어댑터."""

    def __init__(self, meta: ChunkMeta, cfg: dict, *, client_factory=None):
        f = cfg["milvus"]["fetch"]
        self.uri = cfg["milvus"]["uri"]
        self.db = f["database"]
        self.collection = f["collection"]
        self.text_field = f.get("text_field", "text")
        self.pk_field = f.get("pk_field", "pk")
        self.layout = cfg["c1"].get("layout", f.get("layout", "seq"))
        if f.get("layout", self.layout) != self.layout:
            raise ValueError(f"c1.layout({self.layout}) != milvus.fetch.layout({f.get('layout')})")
        if self.layout not in meta.pk_of_chunk or self.layout not in meta.chunk_of_pk:
            raise ValueError(f"ChunkMeta에 layout={self.layout} PK 매핑이 없음 (layouts에 넣어 로드해야 함)")
        self.pk_of = meta.pk_of_chunk[self.layout]
        self.chunk_of = meta.chunk_of_pk[self.layout]
        self.meta = meta
        self._factory = client_factory          # 테스트용: 가짜 AsyncMilvusClient 주입
        self._client = None
        self._schema_ok = False
        self._lock: asyncio.Lock | None = None

    @property
    def name(self) -> str:
        return f"{self.db}/{self.collection}"

    def milvus_collections(self) -> list[str]:
        return [self.name]

    def fetch_source(self) -> str:
        return f"milvus:{self.name}"

    async def _ready(self):
        if self._client is not None and self._schema_ok:
            return self._client
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:                       # 동시에 첫 호출이 와도 클라이언트·스키마 확인은 한 번
            if self._client is None:
                if self._factory is not None:
                    self._client = self._factory(uri=self.uri, db_name=self.db)
                else:
                    from pymilvus import AsyncMilvusClient
                    self._client = AsyncMilvusClient(uri=self.uri, db_name=self.db)
            if not self._schema_ok:
                desc = await self._client.describe_collection(self.collection)
                names = [fd["name"] for fd in desc.get("fields", [])]
                if self.text_field not in names:
                    raise ValueError(f"컬렉션 {self.name}에 본문 필드 '{self.text_field}'가 없음 (필드: {names}). "
                                     f"본문이 없는 컬렉션은 milvus_query_ref로 읽을 수 없음 "
                                     f"(milvus.fetch를 본문 컬렉션으로 바꾸거나 c1.impl=file_ref)")
                if self.pk_field not in names:
                    raise ValueError(f"컬렉션 {self.name}에 PK 필드 '{self.pk_field}'가 없음 (필드: {names})")
                self._schema_ok = True
        return self._client

    async def fetch_text(self, chunk_ids: Sequence[int], *, mode: FetchMode,
                         batch_impl: BatchImpl) -> tuple[Mapping[int, str], FetchStats]:
        if mode == "range":
            raise ValueError("fetch_mode=range는 1차 제외(D7)")
        if mode not in ("batch", "individual"):
            raise ValueError(f"알 수 없는 fetch_mode: {mode!r}")
        if batch_impl != "query_pk_in":
            raise ValueError(f"milvus_query_ref는 batch_impl=query_pk_in만 지원 (요청: {batch_impl!r}). "
                             f"get(ids)는 C1 비교 항목 → C1 담당 어댑터")
        ids = list(dict.fromkeys(int(c) for c in chunk_ids))     # 중복 제거, 순서 유지
        if not ids:
            return {}, FetchStats(n_requests=0, n_pks=0, bytes_returned=0)
        pk_to_cid = {int(self.pk_of[c]): c for c in ids}
        pks = sorted(pk_to_cid)
        parts = [pks[i:i + CHUNK_LIMIT] for i in range(0, len(pks), CHUNK_LIMIT)] if mode == "batch" else [[p] for p in pks]
        cli = await self._ready()
        out: dict[int, str] = {}
        n_req = 0
        for part in parts:
            rows = await cli.query(self.collection, filter=f"{self.pk_field} in [{','.join(map(str, part))}]",
                                   output_fields=[self.pk_field, self.text_field])
            n_req += 1
            for r in rows:
                pk = int(r[self.pk_field])
                cid = pk_to_cid.get(pk)
                if cid is None:
                    continue                                       # 요청하지 않은 PK는 버림
                if int(self.chunk_of[pk]) != cid:
                    raise RuntimeError(f"PK 역매핑 불일치 pk {pk}: chunk_of_pk {int(self.chunk_of[pk])} != {cid}")
                out[cid] = r[self.text_field]
        nbytes = sum(len(t.encode("utf-8")) for t in out.values())   # 계약: 반환 본문 UTF-8 바이트 합
        return out, FetchStats(n_requests=n_req, n_pks=len(pks), bytes_returned=nbytes)

    async def aclose(self):
        if self._client is not None:
            try:
                await self._client.close()
            finally:
                self._client = None
                self._lock = None

    def describe(self) -> dict:
        return {"impl": "milvus_query_ref", "source": self.fetch_source(), "layout": self.layout,
                "text_field": self.text_field, "pk_field": self.pk_field, "chunk_limit": CHUNK_LIMIT,
                "note": "B0 기본 Fetch(최적화 없음). C1 기여 구현은 C1 담당자 어댑터"}
