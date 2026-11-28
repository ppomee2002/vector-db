"""C3 측정용 어댑터 (임준, 2026-11-28). c3_m7.py(m7_junlim, 734b71ae)와 같은 실행 구조에 질의별 실제 시각 기록만 더했다.

등록 이름: c3.impl = m3trace_junlim. 팀 pipeline-core/modules/ 사본에 넣어서 쓴다(팀 저장소·c3_m7.py는 고치지 않음).
- 동작은 m7_junlim과 같다: m3 = 제한 없음(clients개 클라이언트), m7 = Semaphore(min(cap, clients))가 run_one 전체를 감쌈.
  질문은 공유 순번으로 고정 순서 순환, 시간 기반/개수 기반 규칙도 같다.
- 추가: 환경 변수 C3_TRACE에 파일 경로가 있으면, run_many가 끝날 때 결과 순서(results.jsonl과 같은 순서)대로 한 줄씩 적는다.
    seq, q_idx, is_warmup, error, t0_epoch(run_many 시작), t_enq, t_acq, t_done (epoch 초),
    search_ms, c2_select_ms, fetch_ms, n_fetched
  단계 시작 시각은 t_acq + 앞 단계 시간의 합으로 근사한다(분석 쪽에서 계산). 기록은 메모리에 모았다가 끝에 한 번에 쓴다
  (측정 중 파일 쓰기 없음).
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Awaitable, Callable, Sequence

from interfaces import (STAGE_QUEUE_WAIT, STAGE_SERVICE, STAGE_TOTAL, ChunkMeta, QueryKey,
                        QueryResult, SchedulerMode)
from modules.registry import register


@register("c3", "m3trace_junlim")
class C3M3TraceJunlim:
    def __init__(self, meta: ChunkMeta | None = None, cfg: dict | None = None):
        self.cfg = cfg or {}

    async def run_many(
        self,
        query_keys: Sequence[QueryKey],
        run_one: Callable[[QueryKey], Awaitable[QueryResult]],
        *,
        mode: SchedulerMode,
        clients: int,
        cap: int,
        warmup_s: float,
        duration_s: float,
        max_queries: int | None = None,
    ) -> list[QueryResult]:
        if mode not in ("m3", "m7"):
            raise ValueError(f"mode는 m3 또는 m7: {mode!r}")
        if clients < 1:
            raise ValueError("clients는 1 이상")
        if mode == "m7" and cap < 1:
            raise ValueError("m7은 cap 1 이상")
        count_based = max_queries is not None
        if count_based and max_queries < 0:
            raise ValueError("max_queries는 0 이상")
        if not query_keys or (count_based and max_queries == 0):
            return []

        keys = list(query_keys)
        sem = asyncio.Semaphore(min(cap, clients)) if mode == "m7" else None
        results: list[QueryResult] = []
        trace: list[tuple] = []
        nxt = [0]
        t0 = time.perf_counter()
        e0 = time.time()                            # perf_counter → epoch 변환 기준
        t_end = t0 + warmup_s + duration_s

        def ep(t: float) -> float:
            return e0 + (t - t0)

        def can_start(now: float) -> bool:
            return nxt[0] < max_queries if count_based else now < t_end

        async def one_client():
            while True:
                t_enq = time.perf_counter()
                if not can_start(t_enq):
                    return
                if sem is not None:
                    await sem.acquire()
                try:
                    t_acq = time.perf_counter()
                    if not can_start(t_acq):
                        return
                    key = keys[nxt[0] % len(keys)]
                    nxt[0] += 1
                    try:
                        r = await run_one(key)
                    except Exception as e:  # noqa: BLE001
                        r = QueryResult(query_set=key[0], q_idx=int(key[1]), hits=[], context_ids=[],
                                        neighbor_ids=[], fetched_ids=[],
                                        error=f"run_one raised {type(e).__name__}: {e}")
                    t_done = time.perf_counter()
                finally:
                    if sem is not None:
                        sem.release()
                qw = (t_acq - t_enq) * 1e3
                total = (t_done - t_enq) * 1e3
                r.stage_ms[STAGE_QUEUE_WAIT] = qw
                r.stage_ms[STAGE_TOTAL] = total
                r.stage_ms[STAGE_SERVICE] = total - qw
                r.is_warmup = False if count_based else (t_done - t0) < warmup_s
                st = r.stage_ms
                trace.append((len(results), int(r.q_idx), int(bool(r.is_warmup)), 1 if r.error else 0,
                              e0, ep(t_enq), ep(t_acq), ep(t_done),
                              st.get("search"), st.get("c2_select"), st.get("fetch"), len(r.fetched_ids or [])))
                results.append(r)

        await asyncio.gather(*[one_client() for _ in range(clients)])
        path = os.environ.get("C3_TRACE")
        if path:
            new = not os.path.exists(path)
            with open(path, "a", encoding="utf-8") as f:
                if new:
                    f.write("seq,q_idx,is_warmup,error,t0_epoch,t_enq,t_acq,t_done,search_ms,c2_select_ms,fetch_ms,n_fetched\n")
                for row in trace:
                    f.write(",".join("" if v is None else (f"{v:.6f}" if isinstance(v, float) else str(v)) for v in row) + "\n")
        return results

    def describe(self) -> dict:
        return {"impl": "m3trace_junlim",
                "note": "m7_junlim과 같은 구조 + 환경 변수 C3_TRACE 파일에 질의별 epoch 시각·단계 시간 기록(run_many 끝에 한 번에)."}
