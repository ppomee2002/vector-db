"""MilvusGuard 단위 테스트 (가짜 Milvus 클라이언트 + 가짜 메트릭, 실제 서버 호출 없음).

    cd <pipeline-core>
    /workspace/jaeeun/venv/bin/python experiments/test_milvus_guard.py

검증 (C3 검토 반영):
  1. 기존 Loaded 컬렉션은 Release되지 않음
  2. 대상 컬렉션에 요청이 있으면 실행 전 검사에서 중단
  3. 대상 컬렉션이 이미 Loaded면 기본 중단, 허용 설정이면 진행하되 load·release 안 함
  4. 원래 NotLoad였던 컬렉션은 Load → 처리 → Release
  5. Release 직전 다른 사용자 요청이 보이면 Release하지 않고 보고(수동 명령 포함)
  6. 예약 시간대 밖이면 실행 거부
  + 메트릭 조회 실패, 다른 컬렉션 load, Milvus 재시작, load 직전 상태 변화, 잠금 파일 중복
출력: 항목마다 PASS/FAIL 한 줄, 끝에 합계. FAIL이 있으면 종료 코드 1.
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.run_meta import KST, MilvusGuard, RunLock, check_reserved_window  # noqa: E402

T = "junlim/nq_seq"
URI = "http://localhost:19530"


class FakeServer:
    def __init__(self, states: dict[str, str]):
        self.states = dict(states)
        self.calls: list[tuple[str, str]] = []


class FakeClient:
    def __init__(self, server: FakeServer, db: str):
        self.s, self.db = server, db

    def get_load_state(self, coll):
        self.s.calls.append(("get_load_state", f"{self.db}/{coll}"))
        return {"state": f"<LoadState: {self.s.states[f'{self.db}/{coll}']}>"}

    def load_collection(self, coll, timeout=None):
        self.s.calls.append(("load", f"{self.db}/{coll}"))
        self.s.states[f"{self.db}/{coll}"] = "Loaded"

    def release_collection(self, coll):
        self.s.calls.append(("release", f"{self.db}/{coll}"))
        self.s.states[f"{self.db}/{coll}"] = "NotLoad"

    def close(self):
        pass


def act(loaded=0, data=None, start=100.0, error=None, restarted=False):
    if error:
        return {"error": error}
    return {"loaded_collection_num": loaded, "req_delta_data": dict(data or {}), "milvus_start_time": start,
            "milvus_restarted_in_sample": restarted}


def make(states, acts, allow_preloaded=False, allowed=(T,)):
    srv = FakeServer(states)
    q = list(acts)

    def activity(sample_s):
        return q.pop(0)
    g = MilvusGuard(URI, "fake://metrics", list(allowed), allow_preloaded=allow_preloaded,
                    client_factory=lambda db: FakeClient(srv, db), activity_fn=activity)
    return g, srv


RESULTS: list[tuple[str, str, str]] = []


def check(name, cond, msg=""):
    RESULTS.append(("PASS" if cond else "FAIL", name, msg))
    print(f"{'PASS' if cond else 'FAIL'} {name}" + ("" if cond else f": {msg}"))


def ops(srv, kind):
    return [c for k, c in srv.calls if k == kind]


def run_flow(g, targets):
    """main.py와 같은 순서: preflight → load_needed → (처리) → release_safe."""
    ok, a = g.preflight(targets, 0.0)
    rep = None
    if ok:
        g.load_needed()
        rep = g.release_safe(0.0)
    return ok, a, rep


def t1_preloaded_never_released():
    g, srv = make({T: "Loaded"}, [act(loaded=1), act(loaded=1)], allow_preloaded=True)
    ok, a, rep = run_flow(g, [T])
    check("1 기존 Loaded 컬렉션은 release 안 함 (허용 설정으로 진행)",
          ok and not ops(srv, "release") and not ops(srv, "load") and srv.states[T] == "Loaded"
          and rep[0]["action"] == "not_released" and rep[0]["manual_cmd"] is None,
          f"ok={ok} calls={srv.calls} rep={rep}")


def t2_target_requests_abort():
    g, srv = make({T: "NotLoad"}, [act(loaded=0, data={f"{T}/Search": 4.0})])
    ok, a, rep = run_flow(g, [T])
    check("2 실행 전 대상 컬렉션 요청이 있으면 중단 (대상 요청도 남의 것)",
          not ok and not ops(srv, "load") and any(T in r for r in a["busy_reasons"]),
          f"ok={ok} reasons={a['busy_reasons']}")
    g, srv = make({T: "NotLoad"}, [act(loaded=0, data={"other/coll/Query": 1.0})])
    ok, a, rep = run_flow(g, [T])
    check("2b 다른 컬렉션 데이터 요청이 있어도 중단", not ok and not ops(srv, "load"), f"{a['busy_reasons']}")


def t3_preloaded_default_abort_and_allowed():
    g, srv = make({T: "Loaded"}, [act(loaded=1)])
    ok, a, rep = run_flow(g, [T])
    check("3a 대상이 이미 Loaded면 기본 중단",
          not ok and not ops(srv, "load") and not ops(srv, "release") and srv.states[T] == "Loaded",
          f"ok={ok} reasons={a['busy_reasons']}")
    check("3a' 중단 이유에 허용 설정 안내", any("allow_preloaded_targets" in r for r in a["busy_reasons"]),
          f"{a['busy_reasons']}")
    g, srv = make({T: "Loaded"}, [act(loaded=1), act(loaded=1)], allow_preloaded=True)
    ok, a, rep = run_flow(g, [T])
    check("3b 허용 설정이면 진행하되 load·release 안 함",
          ok and not ops(srv, "load") and not ops(srv, "release") and srv.states[T] == "Loaded",
          f"ok={ok} calls={srv.calls}")


def t4_notload_load_release():
    g, srv = make({T: "NotLoad"}, [act(loaded=0), act(loaded=1)])
    ok, a, rep = run_flow(g, [T])
    check("4 원래 NotLoad → load → 처리 → release",
          ok and ops(srv, "load") == [T] and ops(srv, "release") == [T] and srv.states[T] == "NotLoad"
          and rep[0]["action"] == "released" and g.initial_state[T] == "NotLoad",
          f"ok={ok} calls={srv.calls} rep={rep}")


def t5_pre_release_requests_skip():
    g, srv = make({T: "NotLoad"}, [act(loaded=0), act(loaded=1, data={f"{T}/Query": 2.0})])
    ok, a, rep = run_flow(g, [T])
    r = rep[0] if rep else {}
    check("5 release 직전 대상 컬렉션에 다른 요청 → release 안 하고 보고",
          ok and ops(srv, "load") == [T] and not ops(srv, "release") and srv.states[T] == "Loaded"
          and r.get("action") == "not_released" and "release_collection('nq_seq')" in (r.get("manual_cmd") or ""),
          f"calls={srv.calls} rep={rep}")
    g, srv = make({T: "NotLoad"}, [act(loaded=0), act(error="URLError timeout")])
    ok, a, rep = run_flow(g, [T])
    check("5b release 직전 메트릭 조회 실패(확인 불가) → release 안 함",
          ok and not ops(srv, "release") and rep[0]["action"] == "not_released" and rep[0]["manual_cmd"],
          f"rep={rep}")
    g, srv = make({T: "NotLoad"}, [act(loaded=0, start=100.0), act(loaded=1, start=200.0)])
    ok, a, rep = run_flow(g, [T])
    check("5c 실행 중 Milvus 재시작(카운터 초기화) → release 안 함",
          ok and not ops(srv, "release") and "재시작" in rep[0]["reason"], f"rep={rep}")
    g, srv = make({T: "NotLoad"}, [act(loaded=0), act(loaded=2, data={"other/coll/Search": 5.0})])
    ok, a, rep = run_flow(g, [T])
    check("5d release 직전 다른 컬렉션 요청만 있으면 내 것은 release (기록만)",
          ok and ops(srv, "release") == [T], f"rep={rep}")


def t6_window():
    now = datetime(2026, 11, 23, 15, 30, tzinfo=KST)
    run = {"purpose": "measure", "reserved_window": "16:00-18:00"}
    ok, info = check_reserved_window(run, "warm", now)
    check("6 예약 시간대 밖이면 측정 실행 거부", not ok and "밖" in info.get("reason", ""), f"{info}")
    ok, _ = check_reserved_window({"purpose": "measure", "reserved_window": "15:00-16:00"}, "warm", now)
    check("6b 시간대 안이면 허용", ok)
    ok, info = check_reserved_window({"purpose": "measure", "reserved_window": None}, "warm", now)
    check("6c 측정 실행인데 시간대 없음 → 거부", not ok, f"{info}")
    ok, _ = check_reserved_window({"purpose": "correctness", "reserved_window": None}, "warm", now)
    check("6d 정확성 확인은 시간대 없이 허용", ok)
    ok, _ = check_reserved_window({"purpose": "correctness", "reserved_window": None}, "cold", now)
    check("6e cold는 정확성 확인이라도 시간대 필수", not ok)
    ok, _ = check_reserved_window({"purpose": "measure", "reserved_window": "23:00-02:00"},
                                  "warm", datetime(2026, 11, 24, 1, 0, tzinfo=KST))
    check("6f 자정을 넘는 시간대", ok)
    ok, _ = check_reserved_window({"purpose": "measure", "reserved_window": "2026-11-23 15:00~2026-11-23 16:00"},
                                  "warm", now)
    check("6g 날짜+시각 시간대 안", ok)
    ok, _ = check_reserved_window({"purpose": "measure", "reserved_window": "2026-11-22 15:00~2026-11-22 16:00"},
                                  "warm", now)
    check("6h 날짜+시각 시간대 밖(어제)", not ok)
    ok, info = check_reserved_window({"purpose": "measure", "reserved_window": "3pm-4pm"}, "warm", now)
    check("6i 형식 오류는 거부", not ok, f"{info}")


def t_extra():
    g, srv = make({T: "NotLoad"}, [act(error="ConnectionRefused")])
    ok, a, rep = run_flow(g, [T])
    check("x1 실행 전 메트릭 조회 실패 → 중단", not ok and not ops(srv, "load"), f"{a['busy_reasons']}")
    g, srv = make({T: "NotLoad"}, [act(loaded=1)])
    ok, a, rep = run_flow(g, [T])
    check("x2 다른 컬렉션이 load되어 있으면 중단", not ok and any("다른 컬렉션" in r for r in a["busy_reasons"]))
    g, srv = make({T: "Loading"}, [act(loaded=1)], allow_preloaded=True)
    ok, a, rep = run_flow(g, [T])
    check("x3 대상이 Loading 중이면 허용 설정이어도 중단", not ok and not ops(srv, "load"))
    # preflight 뒤 load 직전에 다른 사람이 load한 경우 → load 안 하고, release 대상도 아님
    g, srv = make({T: "NotLoad"}, [act(loaded=0), act(loaded=1)])
    ok, a = g.preflight([T], 0.0)
    srv.states[T] = "Loaded"
    try:
        g.load_needed()
        raised = False
    except RuntimeError:
        raised = True
    rep = g.release_safe(0.0)
    check("x4 load 직전 상태가 바뀌면(남이 load) load·release 안 함",
          ok and raised and not ops(srv, "load") and not ops(srv, "release") and srv.states[T] == "Loaded",
          f"calls={srv.calls} rep={rep}")
    g, srv = make({T: "NotLoad"}, [], allowed=("jaeeun/nq1m_sel",))
    try:
        g.preflight([T], 0.0)
        denied = False
    except PermissionError:
        denied = True
    check("x5 허용 목록 밖 컬렉션은 조회도 안 함", denied and not srv.calls, f"calls={srv.calls}")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.lock")
        a1 = RunLock(p)
        a1.acquire({"run": "a"})
        try:
            RunLock(p).acquire({"run": "b"})
            second = True
        except BlockingIOError:
            second = False
        a1.release()
        RunLock(p).acquire()  # 풀린 뒤에는 다시 잡힘
        check("x6 잠금 파일: 내 실행끼리 두 번째는 거부, 풀리면 다시 잡힘", not second)


def main() -> int:
    for f in (t1_preloaded_never_released, t2_target_requests_abort, t3_preloaded_default_abort_and_allowed,
              t4_notload_load_release, t5_pre_release_requests_skip, t6_window, t_extra):
        try:
            f()
        except Exception as e:  # noqa: BLE001
            check(f.__name__, False, f"예외 {type(e).__name__}: {e}")
    n_fail = sum(1 for r in RESULTS if r[0] == "FAIL")
    print(f"\n# 결과: PASS {len(RESULTS) - n_fail} / FAIL {n_fail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
