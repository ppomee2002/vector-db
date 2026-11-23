"""[reference, 담당자 구현으로 교체] C3 스케줄러 자리 표시용 어댑터 (동시성 없음).

- 등록 이름: c3.impl = serial_ref
- 질문을 하나씩 순서대로 실행한다. mode/clients/cap은 받기만 하고 무시(실행 메타에 그대로 기록됨).
- 시간 기반 실행(계약): query_keys를 순서대로 순환, 시작 후 warmup_s 안에 끝난 질문은 is_warmup=True,
  warmup_s + duration_s가 지나면 새 질문을 시작하지 않는다.
- stage_ms: queue_wait = 0 (대기열 없음), total = 큐 진입~완료, service = total − queue_wait.
- 개수 기반 실행(계약, max_queries가 None이 아닐 때): query_keys를 순서대로(필요하면 순환) 정확히 max_queries개
  실행하고 멈춘다. warmup_s/duration_s는 쓰지 않고 모든 결과는 is_warmup=False (품질 평가용).
"""
from __future__ import annotations

import time
from typing import Awaitable, Callable, Sequence

from interfaces import (STAGE_QUEUE_WAIT, STAGE_SERVICE, STAGE_TOTAL, ChunkMeta, QueryKey,
                        QueryResult, SchedulerMode)
from modules.registry import register


@register("c3", "serial_ref")
class C3SerialReference:
    """reference, 담당자 구현으로 교체. 동시성 없이 순서대로 실행."""

    def __init__(self, meta: ChunkMeta | None = None, cfg: dict | None = None):
        pass

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
        if not query_keys:
            return []
        results: list[QueryResult] = []
        count_based = max_queries is not None
        if count_based and max_queries < 0:
            raise ValueError("max_queries는 0 이상")
        t0 = time.perf_counter()
        t_end = t0 + warmup_s + duration_s
        i = 0
        while (i < max_queries) if count_based else (time.perf_counter() < t_end):
            key = query_keys[i % len(query_keys)]
            i += 1
            t_enq = time.perf_counter()   # 큐 진입
            t_acq = t_enq                 # 대기열 없음 → 바로 시작
            r = await run_one(key)
            t_done = time.perf_counter()
            qw = (t_acq - t_enq) * 1e3
            total = (t_done - t_enq) * 1e3
            r.stage_ms[STAGE_QUEUE_WAIT] = qw
            r.stage_ms[STAGE_TOTAL] = total
            r.stage_ms[STAGE_SERVICE] = total - qw
            r.is_warmup = False if count_based else (t_done - t0) < warmup_s
            results.append(r)
        return results

    def describe(self) -> dict:
        return {"impl": "serial_ref", "note": "reference, 담당자 구현으로 교체 (동시성 없음)"}
