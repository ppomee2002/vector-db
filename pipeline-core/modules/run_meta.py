"""실행 메타 수집과 Milvus load/release 관리 (공통 코어).

원칙 (공유 서버):
- 다른 사람 컬렉션에는 어떤 API도 호출하지 않는다. 서버 전체 상태는 메트릭(localhost:9091/metrics)으로만 본다.
  · 로드된 컬렉션 수: milvus_querynode_collection_num
  · 들어오는 요청: milvus_proxy_req_count(status="total")의 표본 구간 차이 (Search/Query/Get/Insert/Load 등 데이터 요청)
  · Milvus CPU: process_cpu_seconds_total 차이
- load 대상은 run.milvus_allowed_load 목록에 있는 "<db>/<collection>"만. 끝나면 반드시 release (main.py에서 try/finally).
- 로드 전 사용 중이면(다른 컬렉션이 load되어 있거나 데이터 요청이 들어오면) 멈추고 보고.
"""
from __future__ import annotations

import getpass
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
    s = {"collection_num": None, "cpu_s": None, "rss_bytes": None, "req_total": {}, "cache_miss_bytes": 0.0}
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
        elif name.endswith("internal_cache_cell_access_miss_bytes_total"):
            s["cache_miss_bytes"] += v
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
        "loaded_collection_num": b["collection_num"],
        "milvus_cpu_cores": round((b["cpu_s"] - a["cpu_s"]) / sample_s, 3) if a["cpu_s"] is not None else None,
        "milvus_rss_gib": round(b["rss_bytes"] / 2**30, 3) if b["rss_bytes"] else None,
        "req_delta_data": data,
        "req_delta_other": {k: v for k, v in d.items() if k not in data},
        "other_user_processes": other_user_processes(),
        "swap": swap_used(),
    }


class MilvusGuard:
    """허용 목록에 있는 내 컬렉션만 load/release. 다른 컬렉션 API 호출 없음."""

    def __init__(self, uri: str, metrics_url: str, allowed: list[str]):
        self.uri = uri
        self.metrics_url = metrics_url
        self.allowed = set(allowed)
        self.loaded_by_me: list[str] = []
        self.log: list[dict] = []

    def _client(self, db: str):
        from pymilvus import MilvusClient
        return MilvusClient(uri=self.uri, db_name=db)

    def load_state(self, full: str) -> str:
        self._check_allowed(full)
        db, coll = full.split("/")
        c = self._client(db)
        try:
            return str(c.get_load_state(coll).get("state"))
        finally:
            c.close()

    def _check_allowed(self, full: str):
        if full not in self.allowed:
            raise PermissionError(f"{full}은 run.milvus_allowed_load에 없음 → 조회·load 하지 않음")

    def preflight(self, targets: list[str], sample_s: float = 10.0) -> tuple[bool, dict]:
        """사용 중이면 (False, 근거). 내 대상 컬렉션이 이미 load되어 있는 것은 허용."""
        mine = {tuple(t.split("/")) for t in targets}
        act = server_activity(self.metrics_url, sample_s, exclude=set())
        my_states = {t: self.load_state(t) for t in targets}
        n_mine_loaded = sum("Loaded" in s and "NotLoad" not in s for s in my_states.values())
        others_loaded = (act["loaded_collection_num"] or 0) - n_mine_loaded
        foreign_data = {k: v for k, v in act["req_delta_data"].items()
                        if tuple(k.split("/")[:2]) not in mine}
        reasons = []
        if others_loaded > 0:
            reasons.append(f"다른 컬렉션 {others_loaded:g}개가 load되어 있음 (querynode collection_num={act['loaded_collection_num']:g})")
        if foreign_data:
            reasons.append(f"{sample_s:g}초 동안 다른 데이터 요청이 들어옴: {foreign_data}")
        act["my_load_state"] = my_states
        act["busy_reasons"] = reasons
        return (not reasons), act

    def load(self, full: str, timeout_s: float = 600):
        self._check_allowed(full)
        db, coll = full.split("/")
        c = self._client(db)
        t = time.perf_counter()
        try:
            c.load_collection(coll, timeout=timeout_s)
        finally:
            c.close()
        self.loaded_by_me.append(full)
        self.log.append({"op": "load", "collection": full, "at_kst": now_kst(),
                         "sec": round(time.perf_counter() - t, 1)})

    def release_all(self):
        errs = []
        for full in list(self.loaded_by_me):
            db, coll = full.split("/")
            try:
                c = self._client(db)
                try:
                    c.release_collection(coll)
                finally:
                    c.close()
                self.log.append({"op": "release", "collection": full, "at_kst": now_kst()})
                self.loaded_by_me.remove(full)
            except Exception as e:  # noqa: BLE001
                errs.append(f"{full}: {e}")
                self.log.append({"op": "release_failed", "collection": full, "at_kst": now_kst(), "error": str(e)})
        return errs
