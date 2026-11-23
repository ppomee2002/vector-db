"""실행 메타 수집과 Milvus load/release 관리 (공통 코어).

원칙 (공유 서버):
- 서버 전체 상태는 메트릭(localhost:9091/metrics)으로만 본다. Milvus API는 run.milvus_allowed_load에 있는
  "<db>/<collection>"에만 호출한다(load 상태 조회, load, release).
  · 로드된 컬렉션 수: milvus_querynode_collection_num
  · 들어오는 요청: milvus_proxy_req_count(status="total")의 표본 구간 차이 (Search/Query/Get/Insert/Load 등 데이터 요청)
  · Milvus 재시작: process_start_time_seconds가 바뀌면 요청 카운터가 초기화되므로 "확인 불가"로 본다.
- MilvusGuard (C3 검토 반영, 2026-11-23):
  1. 실행 전 대상 컬렉션의 load 상태를 기록한다(initial_state).
  2. 실행 전 사용 중 검사에서는 대상 컬렉션으로 들어온 요청도 남의 요청으로 본다(실행 전이므로).
     다른 컬렉션이 load되어 있거나, 데이터 요청이 있거나, 대상이 Loading이거나, 메트릭을 못 읽으면 중단.
  3. 대상 컬렉션이 이미 Loaded면 기본은 중단. run.allow_preloaded_targets=true일 때만 진행하고, 그때도 load·release하지 않는다.
  4. release 대상 = 원래 NotLoad였고 이번 실행이 직접 load한 컬렉션만. 원래 Loaded였던 컬렉션은 절대 자동 release하지 않는다.
  5. release 직전에 다시 표본 구간을 보고, 대상 컬렉션 요청·Milvus 재시작·메트릭 조회 실패가 있으면 release하지 않고
     이유와 수동 release 명령을 보고(release_report)한다.
- 예약 시간대(run.reserved_window, KST)와 내 실행끼리의 잠금 파일(locks/milvus_run.lock)도 여기서 처리한다.
  팀 공통 잠금은 합의 전이라 만들지 않는다(미결).
"""
from __future__ import annotations

import fcntl
import getpass
import json
import os
import re
import subprocess
import time
import urllib.request
from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))
DATA_FUNCS = {"Search", "HybridSearch", "Query", "Get", "Insert", "Upsert", "Delete", "LoadCollection",
              "LoadPartitions", "ReleaseCollection", "ReleasePartitions", "Flush", "CreateIndex", "DropCollection"}
_LINE = re.compile(r'^(\w+)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN)$')
_LBL = re.compile(r'(\w+)="([^"]*)"')


def now_kst() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


def scrape(url: str) -> list[tuple[str, dict, float]]:
    with urllib.request.urlopen(url, timeout=5) as r:
        text = r.read().decode()
    out = []
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _LINE.match(line)
        if not m:
            continue
        labels = dict(_LBL.findall(m.group(2) or ""))
        try:
            out.append((m.group(1), labels, float(m.group(3))))
        except ValueError:
            pass
    return out


def summarize_metrics(ms: list[tuple[str, dict, float]]) -> dict:
    s = {"collection_num": None, "cpu_s": None, "rss_bytes": None, "req_total": {}, "cache_miss_bytes": 0.0,
         "cache_miss_by_loc": {}, "start_time": None}
    for name, lb, v in ms:
        if name == "milvus_querynode_collection_num":
            s["collection_num"] = (s["collection_num"] or 0) + v
        elif name == "process_cpu_seconds_total":
            s["cpu_s"] = v
        elif name == "process_resident_memory_bytes":
            s["rss_bytes"] = v
        elif name == "milvus_proxy_req_count" and lb.get("status") == "total":
            k = (lb.get("db_name", ""), lb.get("collection_name", ""), lb.get("function_name", ""))
            s["req_total"][k] = s["req_total"].get(k, 0.0) + v
        elif name == "process_start_time_seconds":
            s["start_time"] = v
        elif name.endswith("internal_cache_cell_access_miss_bytes_total"):
            s["cache_miss_bytes"] += v
            k = f'{lb.get("data_type", "")}/{lb.get("location", "")}'
            s["cache_miss_by_loc"][k] = s["cache_miss_by_loc"].get(k, 0.0) + v
    return s


def req_delta(a: dict, b: dict, *, exclude: set[tuple[str, str]] = frozenset()) -> dict:
    """구간 요청 수 차이. 다른 사람 요청이 들어왔는지만 판단(컬렉션 이름은 메트릭 라벨일 뿐 조회 아님)."""
    d = {}
    for k, v in b["req_total"].items():
        dv = v - a["req_total"].get(k, 0.0)
        if dv > 0 and (k[0], k[1]) not in exclude:
            d["/".join(k)] = dv
    return d


def swap_used() -> dict:
    with open("/proc/meminfo") as f:
        mi = {l.split(":")[0]: int(l.split()[1]) for l in f}
    return {"swap_total_mib": mi["SwapTotal"] // 1024, "swap_used_mib": (mi["SwapTotal"] - mi["SwapFree"]) // 1024,
            "mem_available_mib": mi["MemAvailable"] // 1024}


def other_user_processes(min_cpu: float = 1.0) -> list[dict]:
    me = getpass.getuser()
    out = subprocess.run(["ps", "-eo", "user:20,pcpu,etime,comm", "--no-headers"],
                         capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        p = line.split(None, 3)
        if len(p) == 4 and p[0] != me and p[0] != "root" and float(p[1]) >= min_cpu:
            rows.append({"user": p[0], "pcpu": float(p[1]), "etime": p[2], "comm": p[3]})
    return rows


def server_activity(metrics_url: str, sample_s: float = 10.0, exclude=frozenset()) -> dict:
    a = summarize_metrics(scrape(metrics_url))
    time.sleep(sample_s)
    b = summarize_metrics(scrape(metrics_url))
    d = req_delta(a, b, exclude=exclude)
    data = {k: v for k, v in d.items() if k.rsplit("/", 1)[-1] in DATA_FUNCS}
    return {
        "at_kst": now_kst(),
        "sample_s": sample_s,
        "milvus_start_time": b["start_time"],
        "milvus_restarted_in_sample": (a["start_time"] != b["start_time"]),
        "loaded_collection_num": b["collection_num"],
        "milvus_cpu_cores": round((b["cpu_s"] - a["cpu_s"]) / sample_s, 3) if a["cpu_s"] is not None else None,
        "milvus_rss_gib": round(b["rss_bytes"] / 2**30, 3) if b["rss_bytes"] else None,
        "req_delta_data": data,
        "req_delta_other": {k: v for k, v in d.items() if k not in data},
        "other_user_processes": other_user_processes(),
        "swap": swap_used(),
    }




# ---------------------------------------------------------------------------
# 예약 시간대 (run.reserved_window, KST)
# ---------------------------------------------------------------------------
_HM = r"\d{1,2}:\d{2}"
_DT = r"\d{4}-\d{2}-\d{2}[ T]\d{1,2}:\d{2}"


def parse_window(spec: str, now: datetime) -> tuple[datetime, datetime]:
    """'HH:MM-HH:MM'(매일, 자정 넘김 가능) 또는 'YYYY-MM-DD HH:MM~YYYY-MM-DD HH:MM'(구분자 ~ 또는 –). 모두 KST."""
    now = now.astimezone(KST)
    m = (re.fullmatch(rf"\s*({_DT}|{_HM})\s*[~–—]\s*({_DT}|{_HM})\s*", spec)
         or re.fullmatch(rf"\s*({_HM})\s*-\s*({_HM})\s*", spec))
    if not m:
        raise ValueError(f"run.reserved_window 형식 오류: {spec!r} (예: '14:00-16:00' 또는 "
                         f"'2026-11-23 14:00~2026-11-23 16:00', KST)")
    a, b = m.group(1), m.group(2)
    if re.fullmatch(_HM, a) and re.fullmatch(_HM, b):
        def at(day, hm):
            h, mi = map(int, hm.split(":"))
            return day.replace(hour=h, minute=mi, second=0, microsecond=0)
        start, end = at(now, a), at(now, b)
        if end <= start:                        # 자정을 넘는 시간대
            if now < end:
                start -= timedelta(days=1)
            else:
                end += timedelta(days=1)
        return start, end
    if re.fullmatch(_DT, a) and re.fullmatch(_DT, b):
        start = datetime.fromisoformat(a.replace(" ", "T")).replace(tzinfo=KST)
        end = datetime.fromisoformat(b.replace(" ", "T")).replace(tzinfo=KST)
        if end <= start:
            raise ValueError(f"run.reserved_window 종료가 시작보다 이르다: {spec!r}")
        return start, end
    raise ValueError(f"run.reserved_window는 둘 다 HH:MM이거나 둘 다 날짜+시각이어야 함: {spec!r}")


def check_reserved_window(run: dict, cache_condition: str, now: datetime | None = None) -> tuple[bool, dict]:
    """측정 실행(run.purpose=measure)과 cold/cache_constrained 실행은 예약 시간대 안에서만 허용.
    정확성 확인(run.purpose=correctness)은 시간대를 적지 않아도 되지만, 적었으면 지킨다."""
    now = (now or datetime.now(KST)).astimezone(KST)
    spec = run.get("reserved_window")
    purpose = run.get("purpose")
    required = purpose == "measure" or cache_condition in ("cold", "cache_constrained")
    info = {"reserved_window": spec, "purpose": purpose, "cache_condition": cache_condition,
            "required": required, "now_kst": now.isoformat(timespec="seconds")}
    if not spec:
        if required:
            info["reason"] = (f"run.reserved_window가 없음: purpose={purpose}, cache_condition={cache_condition}인 실행은 "
                              f"예약 시간대(KST) 안에서만 한다")
            return False, info
        info["status"] = "시간대 없음 (정확성 확인 실행이라 허용)"
        return True, info
    try:
        start, end = parse_window(str(spec), now)
    except ValueError as e:
        info["reason"] = str(e)
        return False, info
    info["window_kst"] = [start.isoformat(timespec="minutes"), end.isoformat(timespec="minutes")]
    if not (start <= now < end):
        info["reason"] = f"예약 시간대 밖: 지금 {now:%Y-%m-%d %H:%M} KST, 시간대 {start:%m-%d %H:%M}~{end:%m-%d %H:%M}"
        return False, info
    info["status"] = "시간대 안"
    return True, info


# ---------------------------------------------------------------------------
# 내 실행끼리 겹치지 않게 하는 잠금 파일 (팀 공통 잠금 아님)
# ---------------------------------------------------------------------------
class RunLock:
    """pipeline-core/locks/milvus_run.lock 에 flock(비차단). 이미 잡혀 있으면 BlockingIOError → 실행 거부.
    프로세스가 죽으면 OS가 잠금을 푼다(파일은 남아도 됨). 팀 공통 잠금은 합의 전이라 만들지 않는다."""

    def __init__(self, path: str | os.PathLike):
        self.path = str(path)
        self.fd: int | None = None

    def acquire(self, note: dict | None = None) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            try:
                holder = os.pread(fd, 4096, 0).decode(errors="replace").strip()
            finally:
                os.close(fd)
            raise BlockingIOError(f"내 다른 실행이 잠금을 잡고 있음 ({self.path}): {holder or '?'}")
        os.ftruncate(fd, 0)
        os.pwrite(fd, json.dumps({"pid": os.getpid(), "at_kst": now_kst(), **(note or {})},
                                 ensure_ascii=False).encode(), 0)
        self.fd = fd

    def release(self) -> None:
        if self.fd is not None:
            try:
                os.ftruncate(self.fd, 0)
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


# ---------------------------------------------------------------------------
# MilvusGuard
# ---------------------------------------------------------------------------
LOADED, NOTLOAD, LOADING = "Loaded", "NotLoad", "Loading"


def norm_state(x) -> str:
    s = str(x.get("state") if isinstance(x, dict) else x)
    for k in (NOTLOAD, LOADING, LOADED):   # "NotLoad"가 "Load"를 포함하므로 순서 중요
        if k in s:
            return k
    return f"Unknown({s})"


def manual_release_cmd(uri: str, full: str) -> str:
    db, coll = full.split("/")
    return (f"/workspace/jaeeun/venv/bin/python -c \"from pymilvus import MilvusClient; "
            f"c = MilvusClient(uri='{uri}', db_name='{db}'); c.release_collection('{coll}'); "
            f"print(c.get_load_state('{coll}'))\"")


def _default_activity(metrics_url: str):
    def f(sample_s: float) -> dict:
        try:
            return server_activity(metrics_url, sample_s, exclude=set())
        except Exception as e:  # noqa: BLE001 - 메트릭을 못 읽으면 "확인 불가"
            return {"error": f"{type(e).__name__}: {e}", "at_kst": now_kst(), "sample_s": sample_s}
    return f


class MilvusGuard:
    """허용 목록에 있는 컬렉션만 조회·load·release. 위 원칙 1~5.

    테스트용 주입: client_factory(db) → load_state/load/release를 가진 클라이언트(MilvusClient와 같은 메서드),
    activity_fn(sample_s) → server_activity와 같은 dict.
    """

    def __init__(self, uri: str, metrics_url: str, allowed: list[str], *, allow_preloaded: bool = False,
                 client_factory=None, activity_fn=None):
        self.uri = uri
        self.metrics_url = metrics_url
        self.allowed = set(allowed)
        self.allow_preloaded = bool(allow_preloaded)
        self._factory = client_factory
        self._activity = activity_fn or _default_activity(metrics_url)
        self.targets: list[str] = []
        self.initial_state: dict[str, str] = {}
        self.loaded_by_me: list[str] = []
        self.release_report: list[dict] = []
        self.start_time = None
        self.log: list[dict] = []

    # -- 클라이언트 --
    def _client(self, db: str):
        if self._factory is not None:
            return self._factory(db)
        from pymilvus import MilvusClient
        return MilvusClient(uri=self.uri, db_name=db)

    def _check_allowed(self, full: str):
        if full not in self.allowed:
            raise PermissionError(f"{full}은 run.milvus_allowed_load에 없음 → 조회·load 하지 않음")

    def load_state(self, full: str) -> str:
        self._check_allowed(full)
        db, coll = full.split("/")
        c = self._client(db)
        try:
            return norm_state(c.get_load_state(coll))
        finally:
            c.close()

    # -- 1. 실행 전 상태 기록 --
    def record_initial(self, targets: list[str]) -> dict[str, str]:
        self.targets = list(targets)
        for t in targets:
            try:
                self.initial_state[t] = self.load_state(t)
            except PermissionError:
                raise
            except Exception as e:  # noqa: BLE001
                self.initial_state[t] = f"Unknown({type(e).__name__}: {e})"
        self.log.append({"op": "initial_state", "state": dict(self.initial_state), "at_kst": now_kst()})
        return dict(self.initial_state)

    # -- 2·3. 실행 전 사용 중 검사 --
    def preflight(self, targets: list[str], sample_s: float = 10.0) -> tuple[bool, dict]:
        """(진행 가능, 근거). 대상 컬렉션 요청도 남의 요청으로 센다."""
        states = self.record_initial(targets)
        act = self._activity(sample_s)
        reasons = []
        if act.get("error"):
            reasons.append(f"메트릭 조회 실패 → 사용 여부를 확인할 수 없음: {act['error']}")
        else:
            self.start_time = act.get("milvus_start_time")
            if act.get("milvus_restarted_in_sample"):
                reasons.append("표본 구간 중 Milvus가 재시작됨 → 사용 여부를 확인할 수 없음")
            n_target_loaded = sum(s in (LOADED, LOADING) for s in states.values())
            others = (act.get("loaded_collection_num") or 0) - n_target_loaded
            if others > 0:
                reasons.append(f"다른 컬렉션 {others:g}개가 load되어 있음 "
                               f"(querynode collection_num={act.get('loaded_collection_num'):g})")
            data = act.get("req_delta_data") or {}
            if data:
                reasons.append(f"{sample_s:g}초 동안 데이터 요청이 들어옴(대상 컬렉션 포함, 실행 전이라 모두 남의 요청): {data}")
        unknown = {t: s for t, s in states.items() if s.startswith("Unknown")}
        if unknown:
            reasons.append(f"대상 컬렉션 load 상태를 확인할 수 없음: {unknown}")
        loading = [t for t, s in states.items() if s == LOADING]
        if loading:
            reasons.append(f"대상 컬렉션이 Loading 중(누군가 load하는 중): {loading}")
        pre = [t for t, s in states.items() if s == LOADED]
        if pre and not self.allow_preloaded:
            reasons.append(f"대상 컬렉션이 이미 Loaded: {pre} → 다른 사람이 쓰는 중일 수 있어 중단. "
                           f"진행하려면 run.allow_preloaded_targets: true를 명시(그때도 load·release하지 않음)")
        act["target_load_state"] = states
        act["my_load_state"] = states          # 이전 키 이름 호환
        act["allow_preloaded_targets"] = self.allow_preloaded
        act["busy_reasons"] = reasons
        self.log.append({"op": "preflight", "ok": not reasons, "reasons": reasons, "at_kst": now_kst()})
        return (not reasons), act

    # -- load: 원래 NotLoad였던 것만 --
    def load_needed(self, timeout_s: float = 600) -> list[str]:
        done = []
        for t in self.targets:
            s0 = self.initial_state.get(t)
            if s0 != NOTLOAD:
                self.log.append({"op": "skip_load", "collection": t, "initial_state": s0, "at_kst": now_kst(),
                                 "note": "원래 NotLoad가 아님 → load하지 않고 release 대상에서도 제외"})
                continue
            self.load(t, timeout_s)
            done.append(t)
        return done

    def load(self, full: str, timeout_s: float = 600):
        self._check_allowed(full)
        now_state = self.load_state(full)      # 검사와 load 사이에 누가 load했는지 다시 확인
        if now_state != NOTLOAD:
            self.log.append({"op": "skip_load", "collection": full, "state_now": now_state, "at_kst": now_kst(),
                             "note": "load 직전 확인에서 NotLoad가 아님(다른 사람이 load) → 내 것이 아니므로 release 안 함"})
            raise RuntimeError(f"{full}: load 직전 상태가 {now_state} (다른 사람이 load한 것으로 보고 중단)")
        db, coll = full.split("/")
        c = self._client(db)
        t = time.perf_counter()
        try:
            self.loaded_by_me.append(full)   # load 호출이 실패·중단돼도 부분 load를 정리 대상으로 남긴다
            c.load_collection(coll, timeout=timeout_s)
        finally:
            c.close()
        self.log.append({"op": "load", "collection": full, "at_kst": now_kst(),
                         "sec": round(time.perf_counter() - t, 1)})

    # -- 4·5. 안전 확인 후 release --
    def release_safe(self, sample_s: float = 10.0) -> list[dict]:
        """release_report 반환. 각 항목: collection, action(released/not_released/release_failed), reason, manual_cmd."""
        rep = []
        for t in self.targets:
            if t not in self.loaded_by_me and self.initial_state.get(t) != NOTLOAD:
                rep.append({"collection": t, "action": "not_released", "initial_state": self.initial_state.get(t),
                            "reason": "실행 전부터 Loaded(또는 확인 불가)였던 컬렉션 → 자동 release 대상 아님, 그대로 둠",
                            "manual_cmd": None})
        cands = list(self.loaded_by_me)
        if cands:
            act = self._activity(sample_s)
            unsafe = []
            if act.get("error"):
                unsafe.append(f"메트릭 조회 실패 → 다른 사용자 요청 여부를 확인할 수 없음: {act['error']}")
            else:
                if act.get("milvus_restarted_in_sample") or (
                        self.start_time is not None and act.get("milvus_start_time") != self.start_time):
                    unsafe.append("실행 중 Milvus가 재시작됨(요청 카운터 초기화) → 다른 사용자 요청 여부를 확인할 수 없음")
            mine = {tuple(c.split("/")) for c in cands}
            seen = {k: v for k, v in (act.get("req_delta_data") or {}).items()
                    if tuple(k.split("/")[:2]) in mine}
            self.log.append({"op": "pre_release_check", "sample_s": sample_s, "at_kst": now_kst(),
                             "target_requests": seen, "unsafe": unsafe,
                             "other_data_requests": {k: v for k, v in (act.get("req_delta_data") or {}).items()
                                                     if k not in seen},
                             "loaded_collection_num": act.get("loaded_collection_num")})
            for full in cands:
                why = list(unsafe)
                hit = {k: v for k, v in seen.items() if k.startswith(full + "/")}
                if hit:
                    why.append(f"release 직전 {sample_s:g}초 동안 대상 컬렉션에 다른 사용자 요청: {hit}")
                if why:
                    rep.append({"collection": full, "action": "not_released", "initial_state": NOTLOAD,
                                "reason": " / ".join(why), "manual_cmd": manual_release_cmd(self.uri, full)})
                    self.log.append({"op": "release_skipped", "collection": full, "at_kst": now_kst(), "reason": why})
                    continue
                db, coll = full.split("/")
                try:
                    c = self._client(db)
                    try:
                        c.release_collection(coll)
                    finally:
                        c.close()
                    self.loaded_by_me.remove(full)
                    self.log.append({"op": "release", "collection": full, "at_kst": now_kst()})
                    rep.append({"collection": full, "action": "released", "initial_state": NOTLOAD,
                                "reason": "원래 NotLoad, 이번 실행이 load, release 직전 다른 요청 없음",
                                "manual_cmd": None})
                except Exception as e:  # noqa: BLE001
                    self.log.append({"op": "release_failed", "collection": full, "at_kst": now_kst(), "error": str(e)})
                    rep.append({"collection": full, "action": "release_failed", "initial_state": NOTLOAD,
                                "reason": f"{type(e).__name__}: {e}", "manual_cmd": manual_release_cmd(self.uri, full)})
        self.release_report = rep
        return rep

    def pending_manual(self) -> list[dict]:
        return [r for r in self.release_report if r.get("manual_cmd")]
