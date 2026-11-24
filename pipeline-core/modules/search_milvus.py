"""Search 어댑터 (공통 코어). interfaces.SearchAdapter 구현.

- 컬렉션: config milvus.search.{database, collection}. 회의 후 junlim/nq_seq로 바꿀 때는 이 두 값만 바꾼다.
- 검색: HNSW/IP, limit = search_topk, params.ef = ef (ef >= topk 검사).
- 결과 pk → chunk_id: 검색 컬렉션은 pk == chunk_id(항등)이므로 ChunkMeta.chunk_of_pk["seq"]로 변환하며 항등을 확인한다.
- cfg run.verify_search_doc_idx=true면 doc_idx 필드를 함께 받아 ChunkMeta.doc_idx와 대조한다(ID 대응 검사, 스모크용).
- 생성자 규약 (meta, cfg). queries(QueryLoader)는 선택 키워드 인자: 없으면 cfg로 새로 만든다.
load/release는 이 어댑터가 하지 않는다(main.py가 시간표·사용 중 검사 후 처리).
"""
from __future__ import annotations

from interfaces import ChunkMeta, QueryKey, SearchHit
from modules.queries import QueryLoader


class MilvusSearchAdapter:
    def __init__(self, meta: ChunkMeta, cfg: dict, queries: QueryLoader | None = None):
        s = cfg["milvus"]["search"]
        self.uri = cfg["milvus"]["uri"]
        self.db = s["database"]
        self.collection = s["collection"]
        self.anns_field = s.get("anns_field", "vec")
        self.pk_field = s.get("pk_field", "pk")
        self.meta = meta
        self.queries = queries if queries is not None else QueryLoader(cfg)
        self.verify_doc_idx = bool((cfg.get("run") or {}).get("verify_search_doc_idx", False))
        self._client = None
        self.n_doc_mismatch = 0

    @property
    def name(self) -> str:
        return f"{self.db}/{self.collection}"

    async def _get_client(self):
        if self._client is None:  # grpc aio 채널은 실행 중인 이벤트 루프 안에서 만들어야 함
            from pymilvus import AsyncMilvusClient
            self._client = AsyncMilvusClient(uri=self.uri, db_name=self.db)
        return self._client

    async def search(self, key: QueryKey, *, topk: int, ef: int) -> list[SearchHit]:
        if ef < topk:
            raise ValueError(f"ef({ef}) < topk({topk}) (D9)")
        query_set, q_idx = key
        vec = self.queries.load(query_set).vec(q_idx)
        cli = await self._get_client()
        res = await cli.search(
            collection_name=self.collection,
            data=[vec.astype("float32", copy=False).tolist()],
            anns_field=self.anns_field,
            limit=topk,
            output_fields=["doc_idx"] if self.verify_doc_idx else [],
            search_params={"metric_type": "IP", "params": {"ef": ef}},
        )
        hits = []
        # 배치를 "seq"로 고정하는 이유: 검색 컬렉션(nq_seq, nq1m_sel)은 pk == chunk_id인 문서 순(seq) 배치다.
        # c1.layout(doc/rand)은 C1 본문 컬렉션의 배치일 뿐 검색 컬렉션과 무관하다. 그래서 실행기는 c1.layout과
        # 상관없이 ChunkMeta에 "seq"를 항상 로드한다(main.py, PR #3). 검색 컬렉션을 다른 배치로 바꾸면 여기와 로드를 같이 고칠 것.
        chunk_of = self.meta.chunk_of_pk["seq"]
        for rank, h in enumerate(res[0]):
            pk = int(h["id"] if "id" in h else h[self.pk_field])  # pymilvus 3: PK 필드 이름(pk)으로 옴
            cid = int(chunk_of[pk])
            if cid != pk:
                raise RuntimeError(f"검색 컬렉션 pk {pk} != chunk_id {cid} (계약: 검색 컬렉션은 항등)")
            if self.verify_doc_idx:
                d = h.get("entity", {}).get("doc_idx", h.get("doc_idx"))
                if d is None or int(d) != int(self.meta.doc_idx[cid]):
                    self.n_doc_mismatch += 1
                    raise RuntimeError(f"doc_idx 불일치 chunk {cid}: milvus {d} vs meta {self.meta.doc_idx[cid]}")
            hits.append(SearchHit(chunk_id=cid, rank=rank, score=float(h["distance"])))
        return hits

    async def aclose(self):
        if self._client is not None:
            await self._client.close()
            self._client = None

    def describe(self) -> dict:
        return {"collection": self.name, "uri": self.uri, "anns_field": self.anns_field,
                "verify_doc_idx": self.verify_doc_idx}
