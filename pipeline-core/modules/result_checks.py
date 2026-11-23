"""QueryResult 공통 검사 (main.py 요약과 experiments/contract_test.py가 같이 씀).

각 함수는 위반 메시지 목록(문자열)을 돌려준다. 빈 목록 = 통과.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from interfaces import STAGE_KEYS, FetchStats, QueryResult

QUERY_SETS = ("nq_test_real", "nq_test_ans", "c2x_test")


def _tag(r: QueryResult) -> str:
    return f"({r.query_set},{r.q_idx})"


def check_key(r: QueryResult) -> list[str]:
    e = []
    if r.query_set not in QUERY_SETS:
        e.append(f"{_tag(r)} query_set이 {QUERY_SETS} 밖")
    if type(r.q_idx) is not int:
        e.append(f"{_tag(r)} q_idx 타입이 int가 아님: {type(r.q_idx).__name__}")
    return e


def check_stage_keys(r: QueryResult) -> list[str]:
    bad = set(r.stage_ms) - set(STAGE_KEYS)
    return [f"{_tag(r)} stage_ms에 STAGE_KEYS 밖의 키 {sorted(bad)}"] if bad else []


def check_hits(r: QueryResult, search_topk: int, context_topk: int) -> list[str]:
    e = []
    if r.error and not r.hits:
        return e
    ranks = [h.rank for h in r.hits]
    if ranks != list(range(len(r.hits))):
        e.append(f"{_tag(r)} hits 순위가 0부터 연속 순서가 아님: {ranks[:12]}")
    if len(r.hits) != search_topk:
        e.append(f"{_tag(r)} hits 개수 {len(r.hits)} != search_topk {search_topk}")
    scores = [h.score for h in r.hits]
    if any(s is None for s in scores):
        e.append(f"{_tag(r)} SearchHit.score가 비어 있음(계약: 항상 채움)")
    elif any(scores[i] < scores[i + 1] - 1e-6 for i in range(len(scores) - 1)):
        e.append(f"{_tag(r)} hits 점수가 순위 순(내림차순)이 아님 → 순위가 섞였을 수 있음")
    ids = [h.chunk_id for h in r.hits]
    if len(set(ids)) != len(ids):
        e.append(f"{_tag(r)} hits에 중복 chunk_id")
    if r.context_ids != ids[:context_topk]:
        e.append(f"{_tag(r)} context_ids가 hits 상위 {context_topk}개(순위 순)와 다름")
    return e


def check_ids(r: QueryResult) -> list[str]:
    e = []
    if r.error:
        return e
    if r.neighbor_ids != sorted(set(r.neighbor_ids)):
        e.append(f"{_tag(r)} neighbor_ids가 chunk_id 오름차순(중복 없음)이 아님")
    if set(r.neighbor_ids) & set(r.context_ids):
        e.append(f"{_tag(r)} neighbor_ids에 context_ids와 겹치는 chunk_id")
    want = sorted(set(r.context_ids) | set(r.neighbor_ids))
    if r.fetched_ids != want:
        e.append(f"{_tag(r)} fetched_ids != sorted(context ∪ neighbor)")
    return e


def check_fetch_stats(r: QueryResult) -> list[str]:
    if r.error:
        return []
    fs = r.fetch_stats
    if not isinstance(fs, FetchStats):
        return [f"{_tag(r)} fetch_stats가 FetchStats가 아님"]
    e = []
    for f in ("n_requests", "n_pks", "bytes_returned"):
        v = getattr(fs, f)
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            e.append(f"{_tag(r)} FetchStats.{f}={v!r} (필수 0 이상 int)")
    return e


def check_c3_timing(r: QueryResult, tol_ms: float = 1e-3) -> list[str]:
    e = []
    miss = [k for k in ("queue_wait", "total", "service") if k not in r.stage_ms]
    if miss:
        return [f"{_tag(r)} stage_ms에 {miss} 없음(C3가 기록해야 함)"]
    qw, tot, svc = (r.stage_ms[k] for k in ("queue_wait", "total", "service"))
    if abs(svc - (tot - qw)) > tol_ms:
        e.append(f"{_tag(r)} service {svc:.4f} != total {tot:.4f} − queue_wait {qw:.4f}")
    if qw < 0 or tot < 0:
        e.append(f"{_tag(r)} queue_wait/total 음수")
    return e


def check_all(results: Sequence[QueryResult], *, search_topk: int, context_topk: int,
              policy: str | None = None, with_c3: bool = True) -> dict[str, list[str]]:
    out = {"key": [], "stage_keys": [], "hits_rank": [], "ids": [], "fetch_stats": [], "c3_timing": [],
           "policy_none": [], "errors": []}
    for r in results:
        out["key"] += check_key(r)
        out["stage_keys"] += check_stage_keys(r)
        out["hits_rank"] += check_hits(r, search_topk, context_topk)
        out["ids"] += check_ids(r)
        out["fetch_stats"] += check_fetch_stats(r)
        if with_c3:
            out["c3_timing"] += check_c3_timing(r)
        if policy == "none" and not r.error and (r.neighbor_ids or r.fetched_ids != sorted(r.context_ids)):
            out["policy_none"].append(f"{_tag(r)} policy none인데 이웃이 있거나 fetched_ids != context_ids")
        if r.error:
            out["errors"].append(f"{_tag(r)} {r.error}")
    return out


def result_from_dict(d: dict) -> QueryResult:
    """results.jsonl 한 줄 → QueryResult (계약 테스트에서 저장된 결과 검사용)."""
    from interfaces import SearchHit
    d = dict(d)
    d["hits"] = [SearchHit(**h) for h in d.get("hits", [])]
    if d.get("fetch_stats") is not None:
        d["fetch_stats"] = FetchStats(**d["fetch_stats"])
    return QueryResult(**d)


def percentiles(xs: Iterable[float], ps=(50, 95, 99)) -> dict:
    import numpy as np
    a = np.asarray(list(xs), dtype=float)
    if a.size == 0:
        return {f"p{p}": None for p in ps}
    return {f"p{p}": float(np.percentile(a, p)) for p in ps}
