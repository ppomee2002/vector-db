"""계약 테스트: 등록한 어댑터가 interfaces.py 계약을 지키는지 검사한다.

    cd /workspace/jaeeun/pipeline-core
    /workspace/jaeeun/venv/bin/python experiments/contract_test.py --c2 <내 c2 이름>
    /workspace/jaeeun/venv/bin/python experiments/contract_test.py --c1 <이름> --c2 <이름> --c3 <이름>
    /workspace/jaeeun/venv/bin/python experiments/contract_test.py --results results/<run>/results.jsonl

옵션
  --run runs/b0_smoke.yaml  config overlay (기본: 없음 → config.yaml + reference 어댑터)
  --only c1,c2,c3,common    일부만 검사
  --require-policies BC1,all  이 정책에서 NotImplementedError면 FAIL (기본: SKIP으로 표시)
  --allow-load              C1이 milvus_collections()를 선언한 경우, run.milvus_allowed_load 안에서만
                            사용 중 검사 후 load하고 끝나면 release (Milvus 쓰는 C1용)

검사 내용 (출력: PASS / FAIL / SKIP / WARN, FAIL은 무엇이 왜 틀렸는지 한 줄)
  C2: list[int] 반환, 입력 R에 없는 것만, chunk_id 오름차순·중복 없음, 같은 문서 ±1만, none이면 빈 목록,
      all이면 ±1 후보 전부, 같은 입력에 같은 출력, 입력 불변. 작은 가짜 ChunkMeta + 실제 ChunkMeta.
  C1: (Mapping[int,str], FetchStats) 반환, 요청한 chunk_id가 모두 키(남는 키 없음), 본문이 chunks.jsonl 원문과 일치,
      FetchStats 필수 필드, 키가 PK가 아니라 chunk_id, range 모드는 오류.
  C3: 모든 결과에 queue_wait/total/service, service = total − queue_wait, is_warmup 구간 구분,
      duration이 지나면 멈춤, 결과 수·질문 키가 실제 실행한 질문과 일치, 고정 순서 순환, 동시 실행 상한.
  공통: 가짜 검색 + 선택한 C1/C2/C3로 run_one 실행 → stage_ms 키 ⊆ STAGE_KEYS, hits 순위 보존,
      (query_set, q_idx) 키, fetched_ids = sorted(context ∪ 이웃). --results가 있으면 저장된 결과도 검사.
Milvus는 --allow-load 없이는 호출하지 않는다.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import inspect
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from interfaces import STAGE_KEYS, ChunkMeta, FetchStats, QueryResult, SearchHit  # noqa: E402
from modules import registry  # noqa: E402
from modules.chunk_meta import FileChunkMetaLoader  # noqa: E402
from modules.config_loader import load_config, settings_from  # noqa: E402
from modules.evaluate import read_chunks_jsonl  # noqa: E402
from modules.pipeline import Pipeline  # noqa: E402
from modules.result_checks import check_all, check_stage_keys, result_from_dict  # noqa: E402


class Report:
    def __init__(self):
        self.rows: list[tuple[str, str, str, str]] = []

    def add(self, status, comp, name, msg=""):
        self.rows.append((status, comp, name, msg))
        line = f"{status:4s} [{comp}] {name}" + (f": {msg}" if msg else "")
        print(line, flush=True)

    def check(self, comp, name, violations: list[str], note: str = ""):
        if violations:
            more = f" (외 {len(violations) - 1}건)" if len(violations) > 1 else ""
            self.add("FAIL", comp, name, violations[0] + more)
        else:
            self.add("PASS", comp, name, note)

    def counts(self):
        return Counter(r[0] for r in self.rows)


# ---------------------------------------------------------------------------
# 가짜 ChunkMeta (작은 문서 집합)
# ---------------------------------------------------------------------------
def fake_meta(doc_sizes=(1, 4, 2, 6, 1, 3, 5, 2), seed=7) -> ChunkMeta:
    doc = np.repeat(np.arange(len(doc_sizes), dtype=np.int64), doc_sizes)
    n = len(doc)
    cidx = np.concatenate([np.arange(s, dtype=np.int32) for s in doc_sizes])
    rng = np.random.default_rng(seed)
    flags = rng.integers(0, 4, n).astype(np.uint8)
    ident = np.arange(n, dtype=np.int64)
    nbytes = np.full(n, 100, dtype=np.int64)
    return ChunkMeta(n_chunks=n, doc_idx=doc, chunk_idx=cidx, flags=flags, nbytes=nbytes,
                     pk_of_chunk={"seq": ident}, chunk_of_pk={"seq": ident},
                     offset={"seq": np.arange(n, dtype=np.int64) * 100}, length=nbytes)


def all_neighbors(R, doc) -> list[int]:
    n = len(doc)
    s = set()
    for r in R:
        for x in (r - 1, r + 1):
            if 0 <= x < n and doc[x] == doc[r]:
                s.add(x)
    return sorted(s - set(R))


def clustered_R(rng: random.Random, n: int, k: int = 10, span: int = 15) -> list[int]:
    a = rng.randrange(n)
    pool = list(range(max(0, a - span), min(n, a + span + 1)))
    R = rng.sample(pool, min(k - 3, len(pool)))
    while len(R) < k:
        x = rng.randrange(n)
        if x not in R:
            R.append(x)
    rng.shuffle(R)  # 순위 순 = 무작위 순서 (정렬되어 있지 않은 입력)
    return R


# ---------------------------------------------------------------------------
# C2
# ---------------------------------------------------------------------------
def test_c2(rep: Report, name: str, cfg: dict, real_meta: ChunkMeta | None, require: set[str]):
    comp = f"C2:{name}"
    fm = fake_meta()
    n = fm.n_chunks
    fake_cases = [[0], [n - 1], [1, 2], [5, 2, 9], [6, 7, 8, 9, 10, 11], [11, 12], [13], [23, 0, 12, 4],
                  list(range(n))]
    suites = [("fake", fm, fake_cases)]
    if real_meta is not None:
        rng = random.Random(0)
        N = real_meta.n_chunks
        d = real_meta.doc_idx
        last_doc = np.where(d == d[-1])[0]
        real_cases = [clustered_R(rng, N) for _ in range(30)]
        real_cases += [[0, 1, 500000], [N - 1, N - 2, 3], [int(last_doc[0]), int(last_doc[0]) - 1, N - 1]]
        suites.append(("real", real_meta, real_cases))

    for suite, meta, cases in suites:
        try:
            adapter = registry.build("c2", name, meta, cfg)
        except Exception as e:  # noqa: BLE001
            rep.add("FAIL", comp, f"{suite} 생성", f"__init__(meta, cfg) 실패: {type(e).__name__}: {e}")
            continue
        doc = meta.doc_idx
        for policy in ("none", "BC1", "all"):
            v = {k: [] for k in ("type", "not_in_R", "sorted", "pm1_same_doc", "none_empty", "all_complete",
                                 "deterministic", "input_unchanged", "exception")}
            skipped = None
            for R in cases:
                R_in = list(R)
                try:
                    out = adapter.select_neighbors(R_in, policy=policy)
                    out2 = adapter.select_neighbors(list(R), policy=policy)
                except NotImplementedError as e:
                    skipped = str(e) or "NotImplementedError"
                    break
                except Exception as e:  # noqa: BLE001
                    v["exception"].append(f"R={R[:6]}: {type(e).__name__}: {e}")
                    continue
                tag = f"R={R[:6]}{'…' if len(R) > 6 else ''}"
                if R_in != list(R):
                    v["input_unchanged"].append(f"{tag}: 입력 목록을 수정함")
                if not isinstance(out, list):
                    v["type"].append(f"{tag}: 반환 타입 {type(out).__name__} (list 필요)")
                    continue
                bad = [x for x in out if type(x) is not int]
                if bad:
                    v["type"].append(f"{tag}: int가 아닌 원소 {type(bad[0]).__name__} (numpy 정수면 int()로 변환)")
                    out = [int(x) for x in out]
                if set(out) & set(R):
                    v["not_in_R"].append(f"{tag}: R에 이미 있는 {sorted(set(out) & set(R))[:5]}")
                if out != sorted(set(out)):
                    v["sorted"].append(f"{tag}: 오름차순·중복 없음 아님 {out[:8]}")
                cand = set(all_neighbors(R, doc)) | set(R)
                far = [x for x in out if x not in cand]
                if far:
                    v["pm1_same_doc"].append(f"{tag}: 같은 문서 ±1이 아닌 {far[:5]}")
                if policy == "none" and out:
                    v["none_empty"].append(f"{tag}: none인데 {out[:5]} 반환")
                if policy == "all" and out != all_neighbors(R, doc):
                    exp = all_neighbors(R, doc)
                    v["all_complete"].append(f"{tag}: all은 ±1 후보 전부여야 함. 빠짐 {sorted(set(exp) - set(out))[:5]}")
                if [int(x) for x in out2] != out:
                    v["deterministic"].append(f"{tag}: 같은 입력에 다른 출력")
            if skipped is not None:
                status = "FAIL" if policy in require else "SKIP"
                rep.add(status, comp, f"{suite}/{policy}", f"NotImplementedError: {skipped}")
                continue
            for k, viol in v.items():
                if k == "none_empty" and policy != "none":
                    continue
                if k == "all_complete" and policy != "all":
                    continue
                rep.check(comp, f"{suite}/{policy} {k}", viol)


# ---------------------------------------------------------------------------
# C1
# ---------------------------------------------------------------------------
async def _c1_calls(adapter, requests, modes):
    out = []
    for mode, bi in modes:
        for ids in requests:
            try:
                t = time.perf_counter()
                res = await adapter.fetch_text(list(ids), mode=mode, batch_impl=bi)
                out.append((mode, bi, ids, res, None, time.perf_counter() - t))
            except Exception as e:  # noqa: BLE001
                out.append((mode, bi, ids, None, e, 0.0))
    try:
        await adapter.fetch_text([0, 1], mode="range", batch_impl=modes[0][1])
        range_err = None
    except Exception as e:  # noqa: BLE001
        range_err = e
    try:
        empty = await adapter.fetch_text([], mode=modes[0][0], batch_impl=modes[0][1])
        empty_err = None
    except Exception as e:  # noqa: BLE001
        empty, empty_err = None, e
    f = getattr(adapter, "aclose", None)
    if f:
        await f()
    return out, range_err, (empty, empty_err)


def test_c1(rep: Report, name: str, cfg: dict, meta: ChunkMeta):
    comp = f"C1:{name}"
    try:
        adapter = registry.build("c1", name, meta, cfg)
    except Exception as e:  # noqa: BLE001
        rep.add("FAIL", comp, "생성", f"__init__(meta, cfg) 실패: {type(e).__name__}: {e}")
        return
    N = meta.n_chunks
    rng = random.Random(1)
    d = meta.doc_idx
    last_doc = [int(x) for x in np.where(d == d[-1])[0]]
    b = int(np.searchsorted(d, d[400000] + 1))  # 문서 경계
    requests = [[0], [N - 1], sorted(rng.sample(range(N), 10)), list(range(b - 15, b + 15)),
                last_doc[-12:], sorted(set(clustered_R(rng, N, 18)))]
    modes = [(cfg["c1"]["fetch_mode"], cfg["c1"]["batch_impl"])]
    if modes[0][0] != "individual":
        modes.append(("individual", cfg["c1"]["batch_impl"]))
    calls, range_err, (empty, empty_err) = asyncio.run(_c1_calls(adapter, requests, modes))
    orig = read_chunks_jsonl(cfg["dataset"]["files"]["chunks"]["path"], {c for r in requests for c in r})
    layout = cfg["c1"]["layout"]
    pk_map = meta.pk_of_chunk.get(layout)
    identity = pk_map is not None and bool(np.array_equal(pk_map, np.arange(N)))

    v = {k: [] for k in ("exception", "return_type", "keys_complete", "keys_no_extra", "key_type", "text_match",
                         "fetchstats_fields", "fetchstats_values", "bytes_returned_utf8")}
    for mode, bi, ids, res, err, sec in calls:
        tag = f"{mode}/{bi} n={len(ids)}"
        if err is not None:
            v["exception"].append(f"{tag}: {type(err).__name__}: {err}")
            continue
        if not (isinstance(res, tuple) and len(res) == 2):
            v["return_type"].append(f"{tag}: (Mapping, FetchStats) 튜플이 아님: {type(res).__name__}")
            continue
        texts, st = res
        if not isinstance(texts, Mapping):
            v["return_type"].append(f"{tag}: 첫 값이 Mapping이 아님: {type(texts).__name__}")
            continue
        if not isinstance(st, FetchStats):
            v["return_type"].append(f"{tag}: 둘째 값이 interfaces.FetchStats가 아님: {type(st).__name__}")
        keys = set(texts.keys())
        if any(type(k) is not int for k in keys):
            v["key_type"].append(f"{tag}: 키 타입 {type(next(iter(keys))).__name__} (int chunk_id 필요)")
        keys = {int(k) for k in keys}
        miss = sorted(set(ids) - keys)
        extra = sorted(keys - set(ids))
        if miss:
            v["keys_complete"].append(f"{tag}: 요청 chunk_id {len(miss)}개가 키에 없음 {miss[:5]}")
        if extra:
            hint = ""
            if not identity and pk_map is not None and set(extra) <= {int(pk_map[c]) for c in ids}:
                hint = " ← PK를 키로 돌려줌(chunk_id로 바꿔야 함)"
            v["keys_no_extra"].append(f"{tag}: 요청하지 않은 키 {extra[:5]}{hint}")
        bad = [c for c in ids if c in texts and texts[c] != orig[c]]
        nonstr = [c for c in ids if c in texts and not isinstance(texts[c], str)]
        if nonstr:
            v["text_match"].append(f"{tag}: 본문이 str가 아님 chunk {nonstr[:3]}")
        elif bad:
            v["text_match"].append(f"{tag}: chunks.jsonl 원문과 다른 본문 chunk {bad[:5]} "
                                   f"(PK↔chunk_id 변환 오류 가능)")
        if isinstance(st, FetchStats):
            for f in ("n_requests", "n_pks", "bytes_returned"):
                x = getattr(st, f)
                if not isinstance(x, int) or isinstance(x, bool) or x < 0:
                    v["fetchstats_fields"].append(f"{tag}: FetchStats.{f}={x!r} (필수, 0 이상 int)")
            all_int = all(isinstance(getattr(st, f), int) for f in ("n_requests", "n_pks", "bytes_returned"))
            if ids and all_int and (st.n_requests < 1 or st.n_pks < len(set(ids)) or st.bytes_returned <= 0):
                v["fetchstats_values"].append(f"{tag}: n_requests={st.n_requests}, n_pks={st.n_pks} "
                                              f"(요청 {len(set(ids))}개), bytes={st.bytes_returned}")
            exp_b = sum(len(t.encode("utf-8")) for t in texts.values() if isinstance(t, str))
            if isinstance(st.bytes_returned, int) and st.bytes_returned != exp_b:
                v["bytes_returned_utf8"].append(f"{tag}: bytes_returned {st.bytes_returned} != 반환 본문 UTF-8 바이트 합 "
                                                f"{exp_b} (계약 정의)")
    for k, viol in v.items():
        note = ""
        if k == "keys_no_extra" and identity:
            note = f"layout={layout}은 pk==chunk_id라 키 구분은 본문 대조(text_match)로 확인"
        rep.check(comp, k, viol, note)
    # 실제 Fetch 출처 (계약: "file:<경로>" 또는 "milvus:<db>/<collection>")
    fs_viol = []
    f = getattr(adapter, "fetch_source", None)
    if f is None:
        fs_viol.append("fetch_source() 메서드가 없음 (계약: file:<경로> 또는 milvus:<db>/<collection>)")
    else:
        try:
            src = f()
            ok = isinstance(src, str) and (
                (src.startswith("file:") and len(src) > 5) or
                (src.startswith("milvus:") and src.count("/") == 1 and all(src[7:].split("/"))))
            if not ok:
                fs_viol.append(f"fetch_source() 형식 오류: {src!r}")
        except Exception as e:  # noqa: BLE001
            fs_viol.append(f"fetch_source() 예외: {type(e).__name__}: {e}")
    rep.check(comp, "fetch_source", fs_viol, "" if fs_viol else f"{f()}")
    rep.check(comp, "range 모드는 오류(D7)", [] if range_err is not None else ["fetch_mode=range에 예외가 없음"])
    if empty_err is not None:
        rep.add("WARN", comp, "빈 요청", f"fetch_text([])가 예외: {type(empty_err).__name__}: {empty_err}")
    elif empty is not None and len(empty[0]) != 0:
        rep.add("WARN", comp, "빈 요청", "fetch_text([])가 빈 dict가 아님")
    ms = sorted(c[5] * 1e3 for c in calls if c[4] is None)
    if ms:
        rep.add("INFO", comp, "fetch 시간", f"{len(ms)}회 중앙값 {ms[len(ms)//2]:.2f} ms (참고)")


# ---------------------------------------------------------------------------
# C3
# ---------------------------------------------------------------------------
def test_c3(rep: Report, name: str, cfg: dict, meta):
    comp = f"C3:{name}"
    try:
        sched = registry.build("c3", name, meta, cfg)
    except Exception as e:  # noqa: BLE001
        rep.add("FAIL", comp, "생성", f"__init__(meta, cfg) 실패: {type(e).__name__}: {e}")
        return
    keys = [("nq_test_ans", q) for q in (5, 3, 9, 1, 7, 2)]
    # (mode, clients, cap, warmup_s, duration_s, max_queries). max_queries가 있으면 개수 기반(계약):
    #  정확히 max_queries개, warmup_s/duration_s 무시, 모두 is_warmup=False.
    #  (m3, …, 0.0, 0.02, 9): duration 20ms보다 오래 걸리게 해서 시간으로 멈추지 않는지, 목록(6개)을 넘어 순환하는지 본다.
    scenarios = [("m3", 4, 2, 0.3, 0.7, None), ("m7", 8, 2, 0.3, 0.7, None),
                 ("m3", 4, 2, 0.0, 0.02, 9), ("m7", 8, 2, 0.3, 0.7, 4)]
    for mode, clients, cap, wu, du, mq in scenarios:
        tag = f"{mode} clients={clients} cap={cap} " + (f"max_queries={mq} (warmup={wu}s duration={du}s 무시)" if mq is not None
                                                         else f"warmup={wu}s duration={du}s")
        st = {"calls": [], "done": [], "inflight": 0, "max_inflight": 0, "t0": None, "done_at": {},
              "start_at": []}

        async def run_one(key):
            st["calls"].append(key)
            st["start_at"].append(time.perf_counter() - st["t0"])
            st["inflight"] += 1
            st["max_inflight"] = max(st["max_inflight"], st["inflight"])
            await asyncio.sleep(0.004 + 0.002 * (key[1] % 3))
            st["inflight"] -= 1
            r = QueryResult(query_set=key[0], q_idx=key[1], hits=[], context_ids=[], neighbor_ids=[],
                            fetched_ids=[], stage_ms={"search": 1.25, "fetch": 2.5})
            st["done"].append(key)
            st["done_at"][id(r)] = time.perf_counter() - st["t0"]
            return r

        async def go():
            st["t0"] = time.perf_counter()
            res = await sched.run_many(keys, run_one, mode=mode, clients=clients, cap=cap,
                                       warmup_s=wu, duration_s=du, max_queries=mq)
            return res, time.perf_counter() - st["t0"]

        try:
            results, wall = asyncio.run(go())
        except Exception as e:  # noqa: BLE001
            rep.add("FAIL", comp, f"{tag} 실행", f"{type(e).__name__}: {e}")
            continue
        v = {k: [] for k in ("return_type", "timing_keys", "service_eq", "total_covers_run_one",
                             "run_one_stages_kept", "stage_keys", "warmup_split", "stops_after_duration",
                             "result_count", "result_keys", "cycle_order", "concurrency_limit", "max_queries_count")}
        if not isinstance(results, list) or any(not isinstance(r, QueryResult) for r in results):
            rep.add("FAIL", comp, f"{tag} return_type", "list[QueryResult]가 아님")
            continue
        for r in results:
            sm = r.stage_ms
            miss = [k for k in ("queue_wait", "total", "service") if k not in sm]
            if miss:
                v["timing_keys"].append(f"({r.query_set},{r.q_idx}) {miss} 없음")
                continue
            if abs(sm["service"] - (sm["total"] - sm["queue_wait"])) > 1e-3:
                v["service_eq"].append(f"service {sm['service']:.4f} != total {sm['total']:.4f} − queue_wait {sm['queue_wait']:.4f}")
            if sm["total"] < 3.5 or sm["service"] < 3.5:
                v["total_covers_run_one"].append(f"total {sm['total']:.2f}ms/service {sm['service']:.2f}ms < run_one 최소 4ms")
            if sm.get("search") != 1.25 or sm.get("fetch") != 2.5:
                v["run_one_stages_kept"].append("run_one이 기록한 search/fetch를 덮어쓰거나 지움")
            v["stage_keys"] += check_stage_keys(r)
            t_done = st["done_at"].get(id(r))
            if mq is not None:
                if r.is_warmup:
                    v["warmup_split"].append("max_queries 실행인데 is_warmup=True (계약: 모두 False)")
            elif t_done is not None:
                if r.is_warmup and t_done > wu + 0.05:
                    v["warmup_split"].append(f"완료 {t_done:.3f}s인데 is_warmup=True (warmup {wu}s)")
                if not r.is_warmup and t_done < wu - 0.05:
                    v["warmup_split"].append(f"완료 {t_done:.3f}s인데 is_warmup=False (warmup {wu}s)")
        nw = sum(r.is_warmup for r in results)
        if mq is None:
            del v["max_queries_count"]
            if nw == 0 or nw == len(results):
                v["warmup_split"].append(f"예열 {nw}/{len(results)}: 예열·측정 구간이 나뉘지 않음")
            if wall > wu + du + 0.3:
                v["stops_after_duration"].append(f"run_many {wall:.2f}s > warmup+duration {wu + du}s + 0.3s")
            late = [t for t in st["start_at"] if t > wu + du + 0.02]
            if late:
                v["stops_after_duration"].append(f"warmup+duration 이후 시작한 질문 {len(late)}개 (첫 {late[0]:.3f}s)")
        else:
            del v["stops_after_duration"]
            if len(results) != mq or len(st["calls"]) != mq:
                v["max_queries_count"].append(f"결과 {len(results)}개 / 실행 {len(st['calls'])}개 != max_queries {mq} "
                                              f"(계약: 정확히 max_queries개, 시간으로 멈추지 않음)")
        if len(results) != len(st["done"]):
            v["result_count"].append(f"결과 {len(results)}개 != 완료된 run_one {len(st['done'])}개")
        if Counter((r.query_set, r.q_idx) for r in results) != Counter(st["done"]):
            v["result_keys"].append("결과의 (query_set, q_idx) 다중집합이 실제 실행한 질문과 다름")
        if any(k not in keys for k in st["calls"]):
            v["result_keys"].append("query_keys에 없는 질문을 실행함")
        exp = [keys[i % len(keys)] for i in range(len(st["calls"]))]
        if st["calls"] != exp:
            i = next(i for i, (a, b) in enumerate(zip(st["calls"], exp)) if a != b)
            v["cycle_order"].append(f"{i}번째 실행 {st['calls'][i]} != 고정 순서 {exp[i]} (목록을 순서대로 순환해야 함)")
        limit = clients if mode == "m3" else min(cap, clients)
        if st["max_inflight"] > limit:
            v["concurrency_limit"].append(f"동시 실행 최대 {st['max_inflight']} > 상한 {limit}")
        for k, viol in v.items():
            rep.check(comp, f"{tag} {k}", viol)
        if len(st["calls"]) > len(st["done"]):
            rep.add("WARN", comp, f"{tag} 취소", f"시작했지만 끝나지 않은 run_one {len(st['calls']) - len(st['done'])}개")
        if st["max_inflight"] == 1 and limit > 1:
            rep.add("WARN", comp, f"{tag} 동시성", "동시 실행 1 (reference serial이면 정상, 담당 구현이면 확인)")
        rep.add("INFO", comp, f"{tag} 요약", f"결과 {len(results)}개(예열 {nw}), 최대 동시 {st['max_inflight']}, {wall:.2f}s")


# ---------------------------------------------------------------------------
# 공통 (가짜 검색 + 선택 어댑터로 run_one)
# ---------------------------------------------------------------------------
class FakeSearch:
    """Milvus 없이 결정적인 hits를 만든다(질문마다 같은 결과, 점수 내림차순)."""

    def __init__(self, n):
        self.n = n

    async def search(self, key, *, topk, ef):
        rng = random.Random(key[1] * 7919 + len(key[0]))
        R = clustered_R(rng, self.n, topk)
        return [SearchHit(chunk_id=int(c), rank=i, score=1.0 - 0.01 * i) for i, c in enumerate(R)]


def test_common(rep: Report, names: dict, cfg: dict, meta: ChunkMeta):
    comp = "공통"
    settings = settings_from(cfg)
    try:
        c1 = registry.build("c1", names["c1"], meta, cfg)
        c2 = registry.build("c2", names["c2"], meta, cfg)
        c3 = registry.build("c3", names["c3"], meta, cfg)
    except Exception as e:  # noqa: BLE001
        rep.add("FAIL", comp, "어댑터 생성", f"{type(e).__name__}: {e}")
        return
    pipe = Pipeline(settings, FakeSearch(meta.n_chunks), c2, c1)
    keys = [("nq_test_ans", q) for q in (11, 4, 27, 8)]

    async def go():
        try:
            return await c3.run_many(keys, pipe.run_one, mode=settings.scheduler_mode, clients=4,
                                     cap=settings.concurrency_cap, warmup_s=0.1, duration_s=0.4)
        finally:
            for a in (c1, c2, c3):
                f = getattr(a, "aclose", None)
                if f:
                    await f()

    try:
        results = asyncio.run(go())
    except Exception as e:  # noqa: BLE001
        rep.add("FAIL", comp, "run_many 실행", f"{type(e).__name__}: {e}")
        return
    rep.add("INFO", comp, "실행", f"가짜 검색 + {names}로 결과 {len(results)}개, policy={settings.neighbor_policy}")
    checks = check_all(results, search_topk=settings.search_topk, context_topk=settings.context_topk,
                       policy=settings.neighbor_policy)
    for k, viol in checks.items():
        rep.check(comp, f"pipeline {k}", viol)
    # 검사기 자체 확인: STAGE_KEYS 밖 키가 실제로 걸리는지
    probe = QueryResult("nq_test_ans", 0, [], [], [], [], stage_ms={"search": 1.0, "fetch_ms": 2.0})
    rep.check(comp, "검사기: STAGE_KEYS 밖 키 탐지", [] if check_stage_keys(probe) else ["'fetch_ms'를 못 잡음"])


def test_results_file(rep: Report, path: str, cfg: dict):
    comp = "결과파일"
    p = Path(path)
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    meta_p = p.parent / "run_meta.json"
    s = json.loads(meta_p.read_text())["settings"] if meta_p.exists() else None
    stopk = s["search_topk"] if s else cfg["ann"]["search_topk"]
    ctopk = s["context_topk"] if s else cfg["ann"]["context_topk"]
    policy = s["neighbor_policy"] if s else cfg["c2"]["policy"]
    bad_text = [i for i, r in enumerate(rows) if any(k in r for k in ("text", "texts", "context_text"))]
    rep.check(comp, "본문이 결과 파일에 없음", [f"{len(bad_text)}행에 본문 필드"] if bad_text else [])
    try:
        results = [result_from_dict(r) for r in rows]
    except TypeError as e:
        rep.add("FAIL", comp, "QueryResult 필드", f"interfaces.QueryResult와 필드가 다름: {e}")
        return
    rep.add("INFO", comp, "읽기", f"{p} {len(results)}행, policy={policy}, topk={stopk}/{ctopk}")
    for k, viol in check_all(results, search_topk=stopk, context_topk=ctopk, policy=policy).items():
        rep.check(comp, k, viol)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--c1")
    ap.add_argument("--c2")
    ap.add_argument("--c3")
    ap.add_argument("--only", default="c1,c2,c3,common")
    ap.add_argument("--require-policies", default="")
    ap.add_argument("--results")
    ap.add_argument("--allow-load", action="store_true")
    args = ap.parse_args(argv)

    cfg, _ = load_config(args.run, args.set)
    cfg.setdefault("run", {})
    names = registry.impl_names(cfg)
    for k in ("c1", "c2", "c3"):
        if getattr(args, k):
            names[k] = getattr(args, k)
            cfg.setdefault(k, {})["impl"] = names[k]
    only = set(args.only.split(","))
    require = {p for p in args.require_policies.split(",") if p}
    rep = Report()
    errs = registry.discover()
    for f, e in errs.items():
        rep.add("WARN", "registry", f"modules/{f}.py import 실패", e.strip().splitlines()[-1])
    print(f"# 어댑터: {names}  (등록된 것: {registry.available()})")

    meta = None
    if only & {"c1", "c2", "common"}:
        t = time.perf_counter()
        meta = FileChunkMetaLoader(cfg, check_md5=False).load(layouts=(cfg["c1"]["layout"],))
        print(f"# 실제 ChunkMeta 로드 {time.perf_counter() - t:.1f}s (md5 생략: main.py 실행 때 검사)")

    guard = None
    try:
        if args.allow_load and only & {"c1", "common"}:
            from modules.run_meta import MilvusGuard
            c1tmp = registry.build("c1", names["c1"], meta, cfg)
            targets = list(getattr(c1tmp, "milvus_collections", lambda: [])())
            if targets:
                run = cfg["run"]
                guard = MilvusGuard(cfg["milvus"]["uri"], run.get("metrics_url", "http://localhost:9091/metrics"),
                                    run.get("milvus_allowed_load", []))
                free, act = guard.preflight(targets, sample_s=10)
                if not free:
                    rep.add("FAIL", "milvus", "사용 중", " / ".join(act["busy_reasons"]))
                    return 1
                for c in targets:
                    guard.load(c)
        if "c2" in only:
            test_c2(rep, names["c2"], cfg, meta, require)
        if "c1" in only:
            test_c1(rep, names["c1"], cfg, meta)
        if "c3" in only:
            test_c3(rep, names["c3"], cfg, meta)
        if "common" in only:
            test_common(rep, names, cfg, meta)
        if args.results:
            test_results_file(rep, args.results, cfg)
    finally:
        if guard is not None:
            print("# release:", guard.release_all() or "ok")

    c = rep.counts()
    print(f"\n# 결과: PASS {c['PASS']} / FAIL {c['FAIL']} / SKIP {c['SKIP']} / WARN {c['WARN']}")
    if c["FAIL"]:
        print("# FAIL 목록:")
        for s, comp, nm, msg in rep.rows:
            if s == "FAIL":
                print(f"  - [{comp}] {nm}: {msg}")
    return 1 if c["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
