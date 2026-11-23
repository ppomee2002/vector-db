"""c1_milvus_query_reference 단위 테스트 (가짜 AsyncMilvusClient, 실제 서버 호출 없음).

    /workspace/jaeeun/venv/bin/python experiments/test_c1_milvus_query.py

실제 Milvus 경로는 계약 테스트(--allow-load)와 nq_seq B0에서 확인한다. 여기서는 어댑터 로직만 본다:
512개 분할, chunk_id↔PK 변환(rand 순열 포함), 반환 키, bytes_returned(UTF-8), 요청하지 않은 PK 무시,
individual 모드 요청 수, range/get 오류, text 필드 없는 컬렉션 오류, 동시 호출 16개에서 결과가 섞이지 않음.
"""
from __future__ import annotations

import asyncio
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from interfaces import ChunkMeta, FetchStats  # noqa: E402
from modules.c1_milvus_query_reference import C1MilvusQueryReference  # noqa: E402

N = 3000
RESULTS = []


def check(name, cond, msg=""):
    RESULTS.append(cond)
    print(f"{'PASS' if cond else 'FAIL'} {name}" + ("" if cond else f": {msg}"))


def meta_with(layout: str) -> ChunkMeta:
    ident = np.arange(N, dtype=np.int64)
    perm = np.random.default_rng(3).permutation(N).astype(np.int64)     # chunk_id → pk
    inv = np.empty_like(perm)
    inv[perm] = ident
    pk_of = {"seq": ident, "rand": perm}
    chunk_of = {"seq": ident, "rand": inv}
    z = np.zeros(N, dtype=np.int64)
    return ChunkMeta(n_chunks=N, doc_idx=ident // 5, chunk_idx=(ident % 5).astype(np.int32), flags=z.astype(np.uint8),
                     nbytes=z, pk_of_chunk={layout: pk_of[layout]}, chunk_of_pk={layout: chunk_of[layout]})


def text_of_chunk(c: int) -> str:
    return f"chunk {c} — é{'ü' * (c % 4)}"   # 비ASCII 포함 → UTF-8 바이트 ≠ 글자 수


class FakeAsyncClient:
    def __init__(self, pk_to_chunk, fields=("pk", "doc_idx", "vec", "text"), extra_pk=False, **kw):
        self.pk_to_chunk = pk_to_chunk
        self.fields = fields
        self.extra_pk = extra_pk
        self.queries: list[int] = []
        self.closed = False

    async def describe_collection(self, name):
        return {"fields": [{"name": f} for f in self.fields]}

    async def query(self, name, filter, output_fields):
        assert filter.startswith("pk in [") and filter.endswith("]"), filter
        pks = [int(x) for x in filter[len("pk in ["):-1].split(",")]
        self.queries.append(len(pks))
        await asyncio.sleep(random.random() * 0.003)          # 동시 호출이 엇갈리게
        rows = [{"pk": p, "text": text_of_chunk(int(self.pk_to_chunk[p]))} for p in pks]
        random.shuffle(rows)                                   # Milvus는 순서를 보장하지 않음
        if self.extra_pk:
            rows.append({"pk": (pks[0] + 1) % N, "text": "요청하지 않은 행"})
        return rows

    async def close(self):
        self.closed = True


def make(layout="seq", **kw):
    meta = meta_with(layout)
    cfg = {"milvus": {"uri": "fake", "fetch": {"database": "junlim", "collection": "nq_seq", "layout": layout}},
           "c1": {"layout": layout}}
    holder = {}

    def factory(**ckw):
        holder["c"] = FakeAsyncClient(meta.chunk_of_pk[layout], **kw)
        return holder["c"]
    return C1MilvusQueryReference(meta, cfg, client_factory=factory), holder


async def go():
    # 512개 분할 + PK 변환(rand)
    a, h = make("rand")
    ids = random.Random(1).sample(range(N), 1100)
    texts, st = await a.fetch_text(ids, mode="batch", batch_impl="query_pk_in")
    check("512개씩 분할 (1100개 → 512+512+76, n_requests=3)", h["c"].queries == [512, 512, 76] and st.n_requests == 3,
          f"{h['c'].queries} {st}")
    check("반환 키 = 요청 chunk_id (rand PK 변환)", set(texts) == set(ids))
    check("본문 = 해당 chunk의 text", all(texts[c] == text_of_chunk(c) for c in ids))
    check("bytes_returned = UTF-8 바이트 합", st.bytes_returned == sum(len(t.encode()) for t in texts.values())
          and st.bytes_returned != sum(len(t) for t in texts.values()))
    check("FetchStats 형식", isinstance(st, FetchStats) and st.n_pks == 1100 and st.blocks is None)
    check("fetch_source / milvus_collections", a.fetch_source() == "milvus:junlim/nq_seq"
          and a.milvus_collections() == ["junlim/nq_seq"])
    await a.aclose()
    check("aclose가 클라이언트를 닫음", h["c"].closed)
    # individual
    a, h = make("seq")
    texts, st = await a.fetch_text([5, 9, 5, 7], mode="individual", batch_impl="query_pk_in")
    check("individual: PK마다 1회, 중복 제거", h["c"].queries == [1, 1, 1] and st.n_requests == 3 and set(texts) == {5, 7, 9})
    # 빈 요청
    texts, st = await a.fetch_text([], mode="batch", batch_impl="query_pk_in")
    check("빈 요청 → 빈 dict, n_requests 0", texts == {} and st.n_requests == 0)
    # range / get
    for mode, bi, nm in (("range", "query_pk_in", "range 모드 오류(D7)"), ("batch", "get", "get은 오류(C1 담당 항목)")):
        try:
            await a.fetch_text([1], mode=mode, batch_impl=bi)
            check(nm, False, "예외 없음")
        except ValueError:
            check(nm, True)
    await a.aclose()
    # 요청하지 않은 PK 무시
    a, h = make("seq", extra_pk=True)
    texts, st = await a.fetch_text([10, 11, 30], mode="batch", batch_impl="query_pk_in")
    check("요청하지 않은 PK 행은 버림", set(texts) == {10, 11, 30}, f"{sorted(texts)}")
    await a.aclose()
    # text 필드 없는 컬렉션
    a, h = make("seq", fields=("pk", "doc_idx", "vec"))
    try:
        await a.fetch_text([1, 2], mode="batch", batch_impl="query_pk_in")
        check("text 필드 없는 컬렉션 → 명확한 오류", False, "예외 없음")
    except ValueError as e:
        check("text 필드 없는 컬렉션 → 명확한 오류", "본문 필드 'text'가 없음" in str(e), str(e))
    await a.aclose()
    # 동시 호출 16개
    a, h = make("rand")
    rng = random.Random(9)
    pool = rng.sample(range(N), 16 * 40)
    groups = [pool[i * 40:i * 40 + rng.randint(1, 40)] for i in range(16)]
    res = await asyncio.gather(*(a.fetch_text(g, mode="batch", batch_impl="query_pk_in") for g in groups))
    bad = [i for i, (g, (t, s)) in enumerate(zip(groups, res))
           if set(t) != set(g) or any(t[c] != text_of_chunk(c) for c in g) or s.n_pks != len(g)
           or s.bytes_returned != sum(len(x.encode()) for x in t.values())]
    check("동시 호출 16개: 각 결과가 자기 요청과 정확히 일치", not bad, f"섞인 요청 {bad}")
    await a.aclose()


def main() -> int:
    random.seed(0)
    asyncio.run(go())
    n_fail = RESULTS.count(False)
    print(f"\n# 결과: PASS {len(RESULTS) - n_fail} / FAIL {n_fail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
