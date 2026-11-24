"""C3 스케줄러 어댑터 (임준, 2026-11-23). 팀 pipeline-core 계약 v1 후보(interfaces.C3SchedulerAdapter)에 맞춘 구현.

설치: 이 파일을 pipeline-core/modules/ 에 복사하고 config의 c3.impl을 "m7_junlim"으로 (또는 --c3 m7_junlim).
출처: bench_v2.py(58d771b9) m3/m7, bench_c3a.py(554335bb)의 요청 상한과 대기 측정. 두 파일은 고치지 않고 방식만 옮겼다.

동작
- m3: 동시 실행 제한 없음. clients개의 클라이언트가 각자 끝나는 대로 다음 질문을 실행한다(닫힌 부하, 기존 m3와 같음).
- m7: asyncio.Semaphore(min(cap, clients))가 run_one 전체(검색 → C2 → C1 Fetch)를 감싼다. 기존 m7과 같은 구조.
- 질문 순서: query_keys를 공유 순번으로 순서대로 순환한다(D3). m7에서는 semaphore를 잡은 뒤에 다음 질문을 꺼내므로
  run_one이 시작되는 순서가 목록 순서와 같다.
- 시간 기록 (stage_ms, 단위 ms. run_one이 쓴 search/c2_select/fetch는 건드리지 않음)
    t_enq = 클라이언트가 다음 요청을 내려는 시각(큐 진입), t_acq = semaphore를 잡은 시각(m3는 t_enq와 같음), t_done = run_one 완료
    queue_wait = t_acq − t_enq,  total = t_done − t_enq,  service = total − queue_wait
- 시간 기반(max_queries=None): 시작 후 warmup_s 안에 끝난 질문은 is_warmup=True. warmup_s + duration_s가 지나면
  새 질문을 시작하지 않는다(이미 시작한 질문은 끝까지 기다려 결과에 넣는다).
- 개수 기반(max_queries=정수): 정확히 max_queries개를 실행하고 멈춘다. warmup_s/duration_s는 쓰지 않고 모두 is_warmup=False.
- run_one이 예외를 내면(계약상 공통 코어 run_one은 예외를 잡아 error에 넣지만) 스케줄러가 멈추지 않도록
  QueryResult(error=...)를 만들어 결과에 넣는다.
- 측정 중 Milvus CPU, 클라이언트 CPU 같은 구간 기록은 이 어댑터가 하지 않는다(실행기·run_meta 몫).
"""
from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, Sequence

from interfaces import (STAGE_QUEUE_WAIT, STAGE_SERVICE, STAGE_TOTAL, ChunkMeta, QueryKey,
                        QueryResult, SchedulerMode)
from modules.registry import register


@register("c3", "m7_junlim")
class C3M7Junlim:
    """m3(제한 없음) / m7(요청 전체 상한 cap). 기존 bench_v2 m3/m7과 같은 실행 구조."""

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
        nxt = [0]                                  # 다음에 꺼낼 질문 순번 (모든 클라이언트가 공유)
        t0 = time.perf_counter()
        t_end = t0 + warmup_s + duration_s

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
                    if not can_start(t_acq):       # 기다리는 동안 끝났으면 시작하지 않음
                        return
                    key = keys[nxt[0] % len(keys)]
                    nxt[0] += 1
                    try:
                        r = await run_one(key)     # 꺼내자마자 시작(사이에 await 없음) → 시작 순서 = 목록 순서
                    except Exception as e:  # noqa: BLE001 - 스케줄러는 멈추지 않는다
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
                results.append(r)

        await asyncio.gather(*[one_client() for _ in range(clients)])
        return results

    def describe(self) -> dict:
        return {"impl": "m7_junlim",
                "note": "m3 = 제한 없음, m7 = Semaphore(min(cap, clients))가 run_one 전체를 감쌈 (bench_v2 m3/m7과 같은 구조). "
                        "질문은 공유 순번으로 고정 순서 순환, m7은 semaphore 획득 후 질문을 꺼냄."}
