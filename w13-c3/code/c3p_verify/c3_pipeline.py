#!/usr/bin/env python3
"""C3 프로토타입: 파이프라인 실행 단계 (13주차, 임준). bench_v2.py(58d771b9)와 bench_c3a.py(554335bb)를 가져다 쓰고 고치지 않는다.

요청 1건 (기존 m7처럼 상한이 요청 전체를 감싼다)
  도착 ─ [C3 요청 상한 k 대기] ─ ANN 검색 ─ 후보 선별 ─ C2 이웃 선별 ─ C1 읽기 계획 ─ [C3 계획 실행] ─ Context 조립 ─ 완료
  - C3의 두 기능을 나눴다
    1) 요청 동시 수 관리: RequestLimiter. ANN 검색 전부터 Context 조립까지 요청 전체를 감싼다 (cap=0이면 무제한 = 기존 m3).
    2) 읽기 계획 실행: execute_plan. C1이 준 계획의 호출을 실행하고, 빠진 PK가 있으면 오류를 낸다.
  - 본문은 그 요청의 ANN 검색 결과로 만든 계획을 읽는다. 13주차 bench_c3a.py는 미리 저장한 PK 목록을 읽었다(검색 결과는 버림). 이 점이 다르다.
  - Context는 조회가 끝난 순서가 아니라 계획 순서(plan.order)로 조립한다.

교체 가능한 단계 (지금은 임시 구현. 팀의 실제 구현이 오면 함수 본문만 바꾼다. 입력·출력 형식은 각 함수 설명 참고)
  select_candidates  임시: ANN 결과 앞 KEEP개(기본 10)를 그대로 남김. 기존 m7과 읽는 범위를 맞추려는 것이고 팀의 후보 선별 방식이 아니다.
  c2_neighbours      임시: 같은 문서 앞뒤 EXPAND조각(기본 1)을 붙임 (bench_v2 Mapper.context, 기존 m7과 같은 범위). 팀의 C2 방식이 아니다.
  c1_plan            임시: Milvus PK 목록 조회 1회(512개 넘으면 bench_v2가 나눠 호출). 실제 C1의 문서 순 저장소·묶어 읽기와 다르다.
                     이 프로토타입의 조회 시간은 "Milvus PK 조회" 시간이며, C1 묶어 읽기 성능으로 쓰면 안 된다.

모드
  selftest  서버 없이(결정적 mock) 검증·부하 경로와 오류 검출을 점검한다. 실제 데이터 파일(BENCH_DATA)은 필요하다.
  verify    정확성 검증. 같은 질문 Q개로
            ① 기준: 요청을 하나씩 두 번(ref, ref2) 실행해 계획(PK 목록)과 Context 해시를 저장. ref와 ref2 차이 = 동시 실행과 무관한 검색 결과 변동.
            ② 같은 계획 입력: ref의 계획을 무제한과 상한 k로 동시 실행(동시 N) → ref와 PK·Context가 모두 같아야 PASS (실행 단계 검증).
            ③ 전체 경로: ANN 검색부터 무제한과 상한 k로 동시 실행 → ref와 같은 개수를 기록 (검색이 같은 결과를 내는지까지 포함).
  warmup    질문 집합 전체를 한 번씩 전체 경로로 실행해 캐시를 데운다.
  load      동시 요청 N, 상한 k(0=무제한)로 한 구간(예열 WARM초 + 측정 DUR초)을 잰다. 상한 대기를 포함한 전체 지연과 단계별 시간을 기록.

usage (서버, /workspace/junlim. bench_v2.py, bench_c3a.py가 같은 폴더에 있어야 한다)
  BENCH_DATA=$ROOT/data_nq BENCH_COLL=nq python c3_pipeline.py --backend mock --mode selftest --outdir /tmp/c3p_selftest
  BENCH_DATA=$ROOT/data_nq BENCH_COLL=nq python c3_pipeline.py --mode verify --outdir $ROOT/c3p_out
  BENCH_DATA=$ROOT/data_nq BENCH_COLL=nq python c3_pipeline.py --mode load --cap 8 --phase test --rep 1 --outdir $ROOT/c3p_out
"""
import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

import bench_v2 as bv
import bench_c3a as c3a

SETTINGS = c3a.SETTINGS          # A top-10/ef64, B top-10/ef100, C top-100/ef100
pc = time.perf_counter

LOAD_COLS = ["ts_utc", "ts_kst", "version", "ds", "qset", "phase", "rep", "slot", "setting", "limit", "ef", "keep", "expand",
             "n", "cap", "dur_s", "completed", "errors", "qps", "mean_ms", "p50_ms", "p95_ms", "p99_ms",
             "wait_ms_mean", "wait_ms_p99", "search_ms_mean", "search_ms_p99", "plan_ms_mean", "plan_ms_p99",
             "fetch_ms_mean", "fetch_ms_p99", "assemble_ms_mean", "assemble_ms_p99",
             "calls_per_req", "pks_per_req", "ctx_chars_per_req", "client_cores", "milvus_cores", "loadavg1_start", "loadavg1_end"]


# =====================================================================
# 교체 가능한 단계 (임시 구현)
# =====================================================================
def select_candidates(ids, keep):
    """후보 선별. 입력: ANN 결과 PK 목록(순위순). 출력: 남길 PK 목록(순위순).
    임시: 앞 keep개. 팀의 후보 선별이 오면 이 함수만 바꾼다."""
    return list(ids[:keep])


def c2_neighbours(M, cand, expand):
    """C2 이웃 선별. 입력: 후보 PK 목록. 출력: 읽을 PK 목록(정렬, 중복 없음).
    임시: 같은 문서 앞뒤 expand조각 (bench_v2 Mapper.context, 기존 m7과 같은 범위)."""
    return M.context(cand, expand)


@dataclass
class ReadPlan:
    order: list          # Context 조립 순서 (PK)
    calls: list          # 실행할 호출 목록. 호출마다 PK 목록 하나 (Milvus PK 조회)


def c1_plan(pks):
    """C1 읽기 계획. 입력: 읽을 PK 목록. 출력: ReadPlan.
    임시: PK 목록 조회 1회. 실제 C1(문서 순 저장소 + 묶어 읽기)과 성능이 다르다."""
    return ReadPlan(order=list(pks), calls=[list(pks)])


# =====================================================================
# C3: 요청 동시 수 관리 + 읽기 계획 실행
# =====================================================================
class RequestLimiter:
    """요청 전체 동시 수 상한. cap <= 0 이면 무제한(기존 m3). 이벤트 루프 안에서 만든다."""

    def __init__(self, cap):
        self.cap = cap
        self.sem = asyncio.Semaphore(cap) if cap > 0 else None

    async def __aenter__(self):
        if self.sem is not None:
            await self.sem.acquire()

    async def __aexit__(self, *exc):
        if self.sem is not None:
            self.sem.release()


def calls_of(plan):
    """계획 실행에 드는 Milvus 조회 호출 수 (bench_v2는 512개마다 나눠 호출)."""
    return sum((len(c) + bv.CHUNK_LIMIT - 1) // bv.CHUNK_LIMIT for c in plan.calls if c)


async def execute_plan(B, plan):
    """읽기 계획 실행. 계획의 PK가 하나라도 빠지면 오류."""
    texts = {}
    for pks in plan.calls:
        if pks:
            texts.update(await B.afetch(pks, "none"))
    missing = [p for p in plan.order if p not in texts]
    if missing:
        raise RuntimeError(f"plan execution missing {len(missing)} of {len(plan.order)} pks (e.g. {missing[:3]})")
    return texts


def assemble(plan, texts):
    """Context 조립: 계획 순서대로 (조회가 끝난 순서와 무관)."""
    return "\n\n".join(texts[p] for p in plan.order)


def h(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


# =====================================================================
# 백엔드: 검색 (bench_c3a와 같은 호출), 결정적 mock
# =====================================================================
class DetMock(bv.MockBackend):
    """도구 점검용 mock. 같은 벡터면 같은 검색 결과, PK마다 다른 본문, 지연과 반환 순서는 무작위(동시 실행에서 순서가 섞이게)."""
    version = "mock-det"

    def _ids(self, vec, k):
        seed = int(hashlib.sha1(np.asarray(vec, dtype=np.float32).tobytes()).hexdigest()[:8], 16)
        return [int(x) for x in np.random.default_rng(seed).choice(self.n, size=k, replace=False)]

    async def asearch_det(self, vec, s):
        await asyncio.sleep(0.003 + random.random() * 0.004); self.count(search=1)
        return self._ids(vec, s["limit"])

    async def afetch(self, pks, merge="none"):
        await asyncio.sleep(0.004 + 0.0002 * len(pks) + random.random() * 0.004)
        self.count(fetch=1, pks=len(pks))
        items = [(int(p), f"text-{int(p)}") for p in pks]; random.shuffle(items)
        return dict(items)


async def ann_search(B, vec, s):
    if isinstance(B, DetMock):
        return await B.asearch_det(vec, s)
    r = await B.ac.search(B.name, [vec], limit=s["limit"], search_params=c3a.sp(s["ef"]), output_fields=["pk"])
    return c3a.ids_of(r)


# =====================================================================
# 요청 1건
# =====================================================================
async def full_request(B, M, a, s, lim, qi):
    """전체 경로. 반환: 단계별 시간(ms)과 결과 확인용 값."""
    vec = M.qv[qi]
    t0 = pc()
    async with lim:
        t1 = pc()
        ids = await ann_search(B, vec, s)
        t2 = pc()
        plan = c1_plan(c2_neighbours(M, select_candidates(ids, a.keep), a.expand))
        t3 = pc()
        texts = await execute_plan(B, plan)
        t4 = pc()
        ctx = assemble(plan, texts)
        t5 = pc()
    ms = lambda x, y: (y - x) * 1e3
    return dict(wait=ms(t0, t1), search=ms(t1, t2), plan=ms(t2, t3), fetch=ms(t3, t4), assemble=ms(t4, t5),
                total=ms(t0, t5), calls=1 + calls_of(plan), pks=len(plan.order), chars=len(ctx),
                plan_obj=plan, key=(tuple(plan.order), h(ctx)))


async def plan_request(B, plan, lim):
    """같은 계획 입력 검증용: 계획 실행과 조립만 (상한 안에서)."""
    async with lim:
        texts = await execute_plan(B, plan)
        ctx = assemble(plan, texts)
    return (tuple(p for p in plan.order if p in texts), h(ctx))


async def pool(n, items, fn):
    """동시 n개 작업자로 items를 한 번씩 처리. 결과 {item: 값 또는 ('ERR', 메시지)}."""
    q = asyncio.Queue()
    for it in items:
        q.put_nowait(it)
    out = {}

    async def worker():
        while True:
            try:
                it = q.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                out[it] = await fn(it)
            except Exception as e:
                out[it] = ("ERR", repr(e)[:200])

    await asyncio.gather(*[worker() for _ in range(n)])
    return out


def question_pool(M, a):
    return [int(x) for x in c3a.pick_queries(M, a)]


# =====================================================================
# verify
# =====================================================================
def run_verify(B, M, a):
    s = SETTINGS[a.setting]
    qs = question_pool(M, a)
    rng = random.Random(7)
    qids = sorted(rng.sample(qs, min(a.vq, len(qs))))

    async def main():
        await B.aopen()
        nolim = RequestLimiter(0)
        t0 = time.time()
        ref, ref2 = {}, {}
        for qi in qids:                       # 하나씩 (동시 실행 없음)
            ref[qi] = await full_request(B, M, a, s, nolim, qi)
        for qi in qids:
            ref2[qi] = await full_request(B, M, a, s, nolim, qi)
        t_seq = time.time() - t0
        res = {}
        for cap in (0, a.cap):
            lim = RequestLimiter(cap)
            res[("plan", cap)] = await pool(a.n, qids, lambda qi: plan_request(B, ref[qi]["plan_obj"], lim))
            res[("full", cap)] = await pool(a.n, qids, lambda qi: full_request(B, M, a, s, lim, qi))
        await B.aclose()
        return ref, ref2, res, t_seq

    ref, ref2, res, t_seq = asyncio.run(main())
    Q = len(qids)
    rk = {qi: ref[qi]["key"] for qi in qids}
    same_ref2 = sum(ref2[qi]["key"] == rk[qi] for qi in qids)
    out = dict(version=B.version, ds=a.ds, qset=a.qset, setting=a.setting, keep=a.keep, expand=a.expand, n=a.n, cap=a.cap,
               questions=Q, seq_seconds=round(t_seq, 1), ref2_same=same_ref2,
               pks_per_req=round(float(np.mean([len(rk[qi][0]) for qi in qids])), 2))
    mism = []
    ok_plan = True
    for (kind, cap), r in res.items():
        errs = [qi for qi in qids if isinstance(r[qi], tuple) and r[qi] and r[qi][0] == "ERR"]
        if kind == "plan":
            same = [qi for qi in qids if qi not in errs and r[qi] == rk[qi]]
        else:
            same = [qi for qi in qids if qi not in errs and r[qi]["key"] == rk[qi]]
        out[f"{kind}_cap{cap}_same"] = len(same)
        out[f"{kind}_cap{cap}_errors"] = len(errs)
        for qi in qids:
            if qi not in same:
                mism.append(dict(kind=kind, cap=cap, qi=qi, error=r[qi][1] if qi in errs else "",
                                 ref2_same=int(ref2[qi]["key"] == rk[qi])))
        if kind == "plan" and (len(same) != Q):
            ok_plan = False
    full_all = all(out[f"full_cap{c}_same"] == Q for c in (0, a.cap))
    out["plan_input_check"] = "PASS" if ok_plan else "FAIL"
    out["full_path_check"] = "같음" if full_all else "확인 필요 (verify_mismatch 파일과 ref2_same 참고)"
    os.makedirs(a.outdir, exist_ok=True)
    json.dump(out, open(f"{a.outdir}/verify_{a.ds}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(f"{a.outdir}/verify_mismatch_{a.ds}.csv", "w", encoding="utf-8") as f:
        f.write("kind,cap,qi,ref2_same,error\n")
        for m in mism:
            f.write(f"{m['kind']},{m['cap']},{m['qi']},{m['ref2_same']},\"{m['error']}\"\n")
    print("verify", json.dumps(out, ensure_ascii=False), flush=True)
    return out


# =====================================================================
# warmup, load
# =====================================================================
def run_warmup(B, M, a):
    s = SETTINGS[a.setting]
    qs = question_pool(M, a)

    async def main():
        await B.aopen()
        lim = RequestLimiter(0); t0 = time.time()
        for i, qi in enumerate(qs):
            await full_request(B, M, a, s, lim, qi)
            if (i + 1) % 1000 == 0:
                print(f"  warmup {i + 1}/{len(qs)} ({time.time() - t0:.0f}s)", flush=True)
        await B.aclose()
        print(f"warmup: {len(qs)} questions in {time.time() - t0:.1f}s", flush=True)

    asyncio.run(main())


def run_load(B, M, a):
    s = SETTINGS[a.setting]
    qs = question_pool(M, a)
    t_begin = pc(); t_warm = t_begin + a.warm; t_stop = t_warm + a.dur
    snaps = {}

    def monitor():
        time.sleep(max(0, t_warm - pc()))
        snaps.update(t0=pc(), cpu0=time.process_time(), m0=c3a.milvus_cpu_seconds(B), la0=c3a.loadavg1())
        time.sleep(max(0, t_stop - pc()))
        snaps.update(t1=pc(), cpu1=time.process_time(), m1=c3a.milvus_cpu_seconds(B), la1=c3a.loadavg1())

    mon = threading.Thread(target=monitor, daemon=True); mon.start()
    max_err = int(os.environ.get("BENCH_MAX_ERR", "20"))
    errc, halt = [0], [False]

    async def main():
        await B.aopen()
        lim = RequestLimiter(a.cap)

        async def client(cid):
            rng, out = random.Random(a.seed * 100000 + a.rep * 1000 + cid), []
            while pc() < t_stop and not halt[0]:
                qi = qs[rng.randrange(len(qs))]
                try:
                    st = await full_request(B, M, a, s, lim, qi); ok = 1
                except Exception as e:
                    st, ok = None, 0; errc[0] += 1
                    if errc[0] == 1:
                        print("첫 오류:", repr(e), file=sys.stderr)
                    if errc[0] > max_err:
                        halt[0] = True
                    await asyncio.sleep(0.2)
                out.append((pc(), ok, st))
            return out

        res = await asyncio.gather(*[client(i) for i in range(a.n)])
        await B.aclose()
        return [r for o in res for r in o]

    recs = asyncio.run(main())
    mon.join(timeout=5)
    if halt[0]:
        snaps.pop("t1", None)
    done = [r[2] for r in recs if r[0] >= t_warm and r[1] == 1]
    errors = sum(1 for r in recs if r[0] >= t_warm and r[1] == 0)
    row = dict(version=B.version, ds=a.ds, qset=a.qset, phase=a.phase, rep=a.rep, slot=a.slot, setting=a.setting,
               limit=s["limit"], ef=s["ef"], keep=a.keep, expand=a.expand, n=a.n, cap=a.cap, completed=len(done), errors=errors)
    if "t1" in snaps and done:
        dur = snaps["t1"] - snaps["t0"]
        col = lambda k: np.array([d[k] for d in done], dtype=float)
        T = col("total")
        row.update(dur_s=round(dur, 2), qps=round(len(done) / dur, 1), mean_ms=round(float(T.mean()), 2),
                   p50_ms=c3a.pctl(T, 50), p95_ms=c3a.pctl(T, 95), p99_ms=c3a.pctl(T, 99),
                   calls_per_req=round(float(col("calls").mean()), 2), pks_per_req=round(float(col("pks").mean()), 2),
                   ctx_chars_per_req=round(float(col("chars").mean()), 1),
                   client_cores=round((snaps["cpu1"] - snaps["cpu0"]) / dur, 2),
                   loadavg1_start=snaps.get("la0"), loadavg1_end=snaps.get("la1"))
        for k in ("wait", "search", "plan", "fetch", "assemble"):
            x = col(k); row[f"{k}_ms_mean"] = round(float(x.mean()), 3); row[f"{k}_ms_p99"] = c3a.pctl(x, 99)
        if snaps.get("m0") is not None and snaps.get("m1") is not None:
            row["milvus_cores"] = round((snaps["m1"] - snaps["m0"]) / dur, 2)
    return row


# =====================================================================
# selftest (mock)
# =====================================================================
def run_selftest(B, M, a):
    ok = True
    a.vq, a.n = 120, 16
    v = run_verify(B, M, a)
    good = (v["plan_input_check"] == "PASS" and v["full_path_check"] == "같음" and v["ref2_same"] == v["questions"])
    print("verify", "OK" if good else "FAIL"); ok &= good

    # 검출 점검 1: 실행 단계가 PK를 빠뜨리면 오류가 나야 한다
    async def drop_one():
        plan = ReadPlan(order=[1, 2, 3], calls=[[1, 2]])
        try:
            await execute_plan(B, plan); return False
        except RuntimeError:
            return True
    good = asyncio.run(drop_one()); print("missing-pk detection", "OK" if good else "FAIL"); ok &= good

    # 검출 점검 2: 끝난 순서로 조립하면(잘못된 구현) 계획 순서 조립과 해시가 달라야 한다 (mock은 반환 순서를 섞는다)
    async def order_check():
        plan = ReadPlan(order=list(range(10, 40)), calls=[list(range(10, 40))])
        texts = await execute_plan(B, plan)
        wrong = "\n\n".join(texts.values())
        return h(assemble(plan, texts)) != h(wrong) and assemble(plan, texts).startswith("text-10\n\ntext-11")
    good = asyncio.run(order_check()); print("assembly-order check", "OK" if good else "FAIL"); ok &= good

    # 부하 경로: 무제한과 상한 k가 둘 다 오류 없이 돌고, 상한일 때만 대기가 생기는지
    a.dur, a.warm = 1.5, 0.3
    rows = {}
    for cap in (0, a.cap):
        aa = argparse.Namespace(**vars(a)); aa.cap = cap
        r = run_load(B, M, aa); rows[cap] = r
        bv.write_row(f"{a.outdir}/selftest_load.csv", LOAD_COLS, r)
        good = r["errors"] == 0 and r["completed"] > 0 and r.get("calls_per_req", 0) >= 2
        print(f"load cap={cap} {'OK ' if good else 'FAIL'} qps={r.get('qps')} wait={r.get('wait_ms_mean')} "
              f"search={r.get('search_ms_mean')} fetch={r.get('fetch_ms_mean')} calls={r.get('calls_per_req')} pks={r.get('pks_per_req')}")
        ok &= good
    good = rows[0].get("wait_ms_mean", 1) < 0.5 and rows[a.cap].get("wait_ms_mean", 0) > rows[0].get("wait_ms_mean", 0)
    print("wait recorded (cap>0 only)", "OK" if good else "FAIL"); ok &= good
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["selftest", "verify", "warmup", "load"], default="load")
    ap.add_argument("--backend", choices=["milvus", "mock"], default="milvus")
    ap.add_argument("--ds", default="nq"); ap.add_argument("--layout", default="seq")
    ap.add_argument("--outdir", default="/workspace/junlim/c3p_out")
    ap.add_argument("--setting", choices=sorted(SETTINGS), default="C", help="ANN 검색 설정. 기본 C(top-100/ef100) = 팀 파이프라인의 top-100")
    ap.add_argument("--keep", type=int, default=10, help="임시 후보 선별: ANN 결과 앞 몇 개를 남길지 (기존 m7 범위 = 10)")
    ap.add_argument("--expand", type=int, default=1, help="임시 C2: 같은 문서 앞뒤 몇 조각 (기존 m7 = 1)")
    ap.add_argument("--qset", choices=["all", "real"], default="real")
    ap.add_argument("--nsub", type=int, default=10000)
    ap.add_argument("--cap", type=int, default=8, help="요청 상한 k. 0 = 무제한. 8은 초기 설정값이며 이 구조에서 최적이라는 근거는 없다")
    ap.add_argument("--n", type=int, default=64); ap.add_argument("--vq", type=int, default=1000, help="verify 질문 수")
    ap.add_argument("--dur", type=float, default=20); ap.add_argument("--warm", type=float, default=5)
    ap.add_argument("--phase", default="test"); ap.add_argument("--rep", type=int, default=1); ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    M = bv.Mapper(a.layout)
    B = DetMock(M.n) if a.backend == "mock" else bv.MilvusBackend(a.layout)
    os.makedirs(a.outdir, exist_ok=True)
    if a.mode == "selftest":
        sys.exit(0 if run_selftest(B, M, a) else 1)
    if a.mode == "verify":
        v = run_verify(B, M, a); sys.exit(0 if v["plan_input_check"] == "PASS" else 2)
    if a.mode == "warmup":
        run_warmup(B, M, a); return
    a.seed = a.seed * 1000 + a.rep
    r = run_load(B, M, a)
    bv.write_row(f"{a.outdir}/results_c3p_load_{a.ds}.csv", LOAD_COLS, r)
    print({k: r.get(k) for k in ("phase", "rep", "cap", "qps", "p99_ms", "wait_ms_mean", "search_ms_mean", "fetch_ms_mean",
                                 "calls_per_req", "pks_per_req", "errors", "client_cores", "milvus_cores")}, flush=True)
    if r.get("qps") in (None, "") or not r.get("completed"):
        print("FAILED RUN: 처리량을 잴 수 없음", file=sys.stderr); sys.exit(3)


if __name__ == "__main__":
    main()
