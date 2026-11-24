"""c1_jihyeon(layout_jihyeon) Milvus 경로 단위 테스트 (가짜 AsyncMilvusClient, 실제 서버 호출 없음). 김지현

    python experiments/test_c1_jihyeon.py

파일 경로는 계약 테스트(experiments/contract_test.py --c1 layout_jihyeon, c1.backend=file)로 확인한다.
여기서는 Milvus 경로 로직만 본다: query_pk_in / get, batch(512개 분할) / individual, seq·doc·rand PK 변환,
컬렉션 chunk_id 필드 역확인, bytes_returned(UTF-8), 요청하지 않은 PK 무시, range 오류, 필드 없는 컬렉션 오류,
동시 호출 16개에서 결과가 섞이지 않음, 컬렉션 선택(seq → milvus.fetch, doc/rand → milvus.c1_compare).
"""
from __future__ import annotations

import asyncio
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from interfaces import ChunkMeta  # noqa: E402
from modules.c1_jihyeon import C1LayoutJihyeon  # noqa: E402

N = 3000
RESULTS = []


def check(name, cond, msg=""):
    RESULTS.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'} {name}" + ("" if cond else f": {msg}"))


IDENT = np.arange(N, dtype=np.int64)
PERM = np.random.default_rng(3).permutation(N).astype(np.int64)
INV = np.empty_like(PERM)
INV[PERM] = IDENT
PK_OF = {"seq": IDENT, "doc": IDENT, "rand": PERM}
CHUNK_OF = {"seq": IDENT, "doc": IDENT, "rand": INV}


def meta_with(layout):
    z = np.zeros(N, dtype=np.int64)
    return ChunkMeta(n_chunks=N, doc_idx=IDENT // 5, chunk_idx=(IDENT % 5).astype(np.int32), flags=z.astype(np.uint8),
                     nbytes=z, pk_of_chunk={layout: PK_OF[layout]}, chunk_of_pk={layout: CHUNK_OF[layout]})


def text_of(c):
    return f"chunk {c} — é{'ü' * (c % 4)}"


class FakeClient:
    def __init__(self, layout, fields=("pk", "chunk_id", "text", "vec"), bad_chunk_id=False, extra_pk=False, **kw):
        self.layout = layout
        self.fields = fields
        self.bad = bad_chunk_id
        self.extra = extra_pk
        self.calls = []          # (kind, n_pks)

    async def describe_collection(self, name):
        return {"fields": [{"name": f} for f in self.fields]}

    def _rows(self, pks, output_fields):
        rows = []
        for p in pks:
            cid = int(CHUNK_OF[self.layout][p])
            r = {"pk": p, "text": text_of(cid)}
            if "chunk_id" in output_fields:
                r["chunk_id"] = cid + (1 if self.bad else 0)
            rows.append(r)
        random.shuffle(rows)
        if self.extra:
            extra = next(p for p in range(N) if p not in set(pks))
            rows.append({"pk": extra, "text": "요청 안 한 행", "chunk_id": int(CHUNK_OF[self.layout][extra])})
        return rows

    async def query(self, name, filter, output_fields):
        assert filter.startswith("pk in [") and filter.endswith("]"), filter
        pks = [int(x) for x in filter[len("pk in ["):-1].split(",")]
        self.calls.append(("query", len(pks)))
        await asyncio.sleep(random.random() * 0.003)
        return self._rows(pks, output_fields)

    async def get(self, name, ids, output_fields):
        self.calls.append(("get", len(ids)))
        await asyncio.sleep(random.random() * 0.003)
        return self._rows([int(x) for x in ids], output_fields)

    async def close(self):
        pass


def make(layout, **kw):
    cfg = {"milvus": {"uri": "fake",
                      "fetch": {"database": "junlim", "collection": "nq_seq", "layout": "seq"},
                      "c1_compare": {"database": "jihyeon", "doc": {"collection": "w13c1_doc"},
                                     "rand": {"collection": "w13c1_rand"}}},
           "c1": {"layout": layout, "backend": "milvus"}}
    holder = {}

    def factory(**k):
        holder["c"] = FakeClient(layout, **kw)
        return holder["c"]

    return C1LayoutJihyeon(meta_with(layout), cfg, client_factory=factory), holder


async def run_all():
    rng = random.Random(1)
    for layout, src in (("seq", "milvus:junlim/nq_seq"), ("doc", "milvus:jihyeon/w13c1_doc"), ("rand", "milvus:jihyeon/w13c1_rand")):
        for bi in ("query_pk_in", "get"):
            ad, h = make(layout)
            check(f"{layout} fetch_source", ad.fetch_source() == src, ad.fetch_source())
            check(f"{layout} milvus_collections", ad.milvus_collections() == [src.split(":", 1)[1]])
            ids = rng.sample(range(N), 1100) + [5, 5]                      # 중복 포함, 512 넘게
            out, st = await ad.fetch_text(ids, mode="batch", batch_impl=bi)
            want = sorted(set(ids))
            check(f"{layout}/{bi} batch keys", sorted(out) == want)
            check(f"{layout}/{bi} batch text", all(out[c] == text_of(c) for c in want))
            check(f"{layout}/{bi} batch n_requests (512 split)", st.n_requests == -(-len(want) // 512) and st.n_pks == len(want), (st.n_requests, st.n_pks, len(want)))
            check(f"{layout}/{bi} batch call kind", all(k == ("query" if bi == "query_pk_in" else "get") for k, _ in h["c"].calls))
            check(f"{layout}/{bi} bytes_returned UTF-8", st.bytes_returned == sum(len(text_of(c).encode()) for c in want))
            check(f"{layout}/{bi} blocks None", st.blocks is None and st.device_bytes is None)
            out, st = await ad.fetch_text([7, 8, 9], mode="individual", batch_impl=bi)
            check(f"{layout}/{bi} individual", sorted(out) == [7, 8, 9] and st.n_requests == 3)
            await ad.aclose()

    ad, _ = make("rand", bad_chunk_id=True)
    try:
        await ad.fetch_text([1, 2], mode="batch", batch_impl="query_pk_in")
        check("chunk_id 필드 불일치 검출", False, "오류 안 남")
    except RuntimeError:
        check("chunk_id 필드 불일치 검출", True)

    ad, _ = make("doc", extra_pk=True)
    out, st = await ad.fetch_text([10, 11], mode="batch", batch_impl="query_pk_in")
    check("요청 안 한 PK 무시", sorted(out) == [10, 11])

    ad, _ = make("doc", fields=("pk", "chunk_id", "vec"))
    try:
        await ad.fetch_text([1], mode="batch", batch_impl="query_pk_in")
        check("text 필드 없는 컬렉션 오류", False)
    except ValueError:
        check("text 필드 없는 컬렉션 오류", True)

    ad, _ = make("doc")
    for bad in (("range", "query_pk_in"), ("batch", "scan")):
        try:
            await ad.fetch_text([1], mode=bad[0], batch_impl=bad[1])
            check(f"{bad} 오류", False)
        except ValueError:
            check(f"{bad} 오류", True)
    out, st = await ad.fetch_text([], mode="batch", batch_impl="get")
    check("빈 요청", out == {} and st.n_requests == 0)

    ad, _ = make("rand")
    groups = [list(range(g * 150, g * 150 + 11 + g)) for g in range(16)]
    res = await asyncio.gather(*[ad.fetch_text(g, mode="batch", batch_impl=["query_pk_in", "get"][i % 2]) for i, g in enumerate(groups)])
    ok = all(sorted(o) == g and all(o[c] == text_of(c) for c in g) and s.n_pks == len(g) for (o, s), g in zip(res, groups))
    check("동시 호출 16개 결과 섞이지 않음", ok)

    try:
        C1LayoutJihyeon(meta_with("doc"), {"c1": {"layout": "doc", "backend": "file"}, "dataset": {}})
        check("file_threads 없으면 거부", False)
    except ValueError:
        check("file_threads 없으면 거부", True)


asyncio.run(run_all())
print(f"# 결과: {sum(RESULTS)}/{len(RESULTS)} PASS")
sys.exit(0 if all(RESULTS) else 1)
