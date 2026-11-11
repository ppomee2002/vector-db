"""Milvus 실험 공통: 접속, 측정값(metrics) 읽기. 컬렉션은 m4m_ 로 시작하는 것만 다룸."""
import re, urllib.request
URI = "http://localhost:19530"
MET = "http://localhost:9091/metrics"
KEYS = ["internal_cache_cell_access_miss_bytes_total", "internal_cache_cell_access_hit_bytes_total",
        "internal_cache_load_latency_microseconds_count", "internal_storage_op_count", "internal_storage_kv_size_sum",
        "internal_storage_kv_size_count", "internal_cache_loaded_bytes"]
def metrics():
    """{(이름, data_type): 값} - data_type 꼬리표가 없으면 '' 로 합산"""
    out = {}
    for line in urllib.request.urlopen(MET, timeout=10).read().decode().splitlines():
        if line.startswith("#"): continue
        for k in KEYS:
            if line.startswith(k + "{") or line.startswith(k + " "):
                m = re.search(r'data_type="([^"]+)"', line); dt = m.group(1) if m else ""
                try: v = float(line.rsplit(" ", 1)[1])
                except ValueError: continue
                out[(k, dt)] = out.get((k, dt), 0) + v
    return out
def delta(a, b): return {k: b.get(k, 0) - a.get(k, 0) for k in set(a) | set(b)}
def pick(d, k, dt=None):
    return sum(v for (kk, t), v in d.items() if kk == k and (dt is None or t == dt))
