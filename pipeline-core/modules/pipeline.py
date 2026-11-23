"""run_one (공통 코어): 질문 하나를 검색 → C2 → C1 순서로 처리해 QueryResult를 만든다.

- stage_ms는 interfaces.STAGE_KEYS만 쓴다. run_one은 search / c2_select / fetch만 기록하고,
  queue_wait / total / service는 C3가 기록한다. embed는 미리 계산한 벡터라 기록하지 않음(D10).
- context_ids = hits 상위 context_topk의 chunk_id(순위 순). 이웃은 C2가 ID만으로 고른다.
- Fetch는 context ∪ 이웃을 chunk_id 정렬로 한 번에 요청한다. 순위는 hits로만 보존.
- 요청한 chunk_id가 C1 결과에 빠지면 error로 기록(FetchStats는 그대로 남김).
- 본문은 QueryResult에 넣지 않는다. text_sink가 있으면 평가용으로 메모리에만 넘긴다(결과 파일에는 안 씀).
- 예외는 잡아서 QueryResult.error에 기록하고 결과를 돌려준다(C3가 멈추지 않게).
"""
from __future__ import annotations

import time
import traceback
from typing import Callable, Mapping

from interfaces import (STAGE_C2_SELECT, STAGE_FETCH, STAGE_KEYS, STAGE_SEARCH, FetchStats,
                        QueryKey, QueryResult, Settings)

TextSink = Callable[[QueryKey, Mapping[int, str]], None]


class Pipeline:
    def __init__(self, settings: Settings, search, c2, c1, *, text_sink: TextSink | None = None):
        self.s = settings
        self.search = search
        self.c2 = c2
        self.c1 = c1
        self.text_sink = text_sink

    async def run_one(self, key: QueryKey) -> QueryResult:
        query_set, q_idx = key
        r = QueryResult(query_set=query_set, q_idx=int(q_idx), hits=[], context_ids=[],
                        neighbor_ids=[], fetched_ids=[])
        stage = "search"
        try:
            t = time.perf_counter()
            hits = await self.search.search(key, topk=self.s.search_topk, ef=self.s.ann_ef)
            r.stage_ms[STAGE_SEARCH] = (time.perf_counter() - t) * 1e3
            r.hits = list(hits)
            r.context_ids = [h.chunk_id for h in r.hits[: self.s.context_topk]]

            stage = "c2_select"
            t = time.perf_counter()
            neighbors = self.c2.select_neighbors(list(r.context_ids), policy=self.s.neighbor_policy)
            r.stage_ms[STAGE_C2_SELECT] = (time.perf_counter() - t) * 1e3
            r.neighbor_ids = [int(x) for x in neighbors]

            stage = "fetch"
            want = sorted(set(r.context_ids) | set(r.neighbor_ids))
            t = time.perf_counter()
            texts, stats = await self.c1.fetch_text(want, mode=self.s.fetch_mode, batch_impl=self.s.batch_impl)
            r.stage_ms[STAGE_FETCH] = (time.perf_counter() - t) * 1e3
            r.fetch_stats = stats if isinstance(stats, FetchStats) else None
            want_set = set(want)
            r.fetched_ids = sorted(int(k) for k in texts.keys() if int(k) in want_set)
            missing = sorted(want_set - set(r.fetched_ids))
            errs = []
            if r.fetch_stats is None:
                errs.append(f"C1이 FetchStats가 아닌 {type(stats).__name__}를 반환")
            if missing:
                errs.append(f"C1 결과에 요청 chunk_id {len(missing)}개 없음: {missing[:5]}")
            if errs:
                r.error = "; ".join(errs)
            if self.text_sink is not None:
                self.text_sink(key, texts)
        except Exception as e:  # noqa: BLE001
            fr = traceback.extract_tb(e.__traceback__)[-1]
            r.error = f"{stage}: {type(e).__name__}: {e} @ {fr.filename.rsplit('/', 1)[-1]}:{fr.lineno}"
        assert set(r.stage_ms) <= set(STAGE_KEYS)
        return r
