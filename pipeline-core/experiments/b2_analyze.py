"""B2 집계 (PLAN_integ.md B2): logs/b2_manifest.json의 실행들을 조건별로 모아 평균·A/A 범위·기여 몫·판정을 낸다.

    /workspace/jaeeun/venv/bin/python experiments/b2_analyze.py   → logs/b2_analysis.json + 표 출력
판정: |평균 차이| > max(두 조건의 A/A 범위) 이고 3회차 모두 같은 부호면 "차이 있음", 아니면 "잡음 범위".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NAMES = {"1_B0": "① B0", "2_ALL": "② ALL", "3_noC2": "③ ALL−C2", "4_noC3": "④ ALL−C3", "5_C1rand": "⑤ ALL−C1(rand)"}


def pick(s, m):
    st = s["stage_ms"]
    g = lambda k, p: (st.get(k) or {}).get(p)  # noqa: E731
    return {
        "qps": s["qps"], "strict_qps": s["strict_qps"], "error_rate": s["error_rate"], "n_measured": s["n_measured"],
        "total_p50": g("total", "p50"), "total_p95": g("total", "p95"), "total_p99": g("total", "p99"),
        "queue_wait_p50": g("queue_wait", "p50"), "queue_wait_p95": g("queue_wait", "p95"),
        "search_p50": g("search", "p50"), "search_p95": g("search", "p95"),
        "c2_select_p50": g("c2_select", "p50"), "c2_select_p95": g("c2_select", "p95"),
        "fetch_p50": g("fetch", "p50"), "fetch_p95": g("fetch", "p95"),
        "service_p50": g("service", "p50"),
        "fetched_chunks": s["fetched_chunks_mean"], "fetch_requests": s["fetch_n_requests_mean"],
        "fetched_bytes": s["fetched_bytes_mean"],
        "storage_read_bytes": (m.get("storage_read_delta") or {}).get("cache_cell_access_miss_bytes"),
        "milvus_restarted": (m.get("storage_read_delta") or {}).get("milvus_restarted"),
    }


def main():
    man = json.loads((ROOT / "logs" / "b2_manifest.json").read_text())
    runs = {}
    for d in man:
        if d["rc"] != 0 or not d["out"]:
            continue
        o = Path(d["out"])
        s = json.loads((o / "summary.json").read_text())
        m = json.loads((o / "run_meta.json").read_text())
        runs.setdefault(d["cond"], {})[d["rep"]] = {**pick(s, m), "out": o.name,
                                                     "window": (m.get("reserved_window") or {}).get("status"),
                                                     "release": [(r["collection"], r["action"]) for r in m.get("release_report", [])],
                                                     "load": [(l["collection"], l.get("sec")) for l in m.get("milvus_ops", []) if l["op"] == "load"]}
    keys = ["qps", "strict_qps", "total_p50", "total_p95", "total_p99", "queue_wait_p50", "queue_wait_p95",
            "search_p50", "c2_select_p50", "fetch_p50", "fetch_p95", "fetched_chunks", "fetch_requests", "fetched_bytes",
            "error_rate", "storage_read_bytes"]
    agg = {}
    for c, reps in runs.items():
        a = {}
        for k in keys:
            v = [reps[r][k] for r in sorted(reps) if reps[r][k] is not None]
            if v:
                a[k] = {"mean": float(np.mean(v)), "range": float(max(v) - min(v)),
                        "cv": float(np.std(v) / np.mean(v)) if np.mean(v) else None, "values": v}
        agg[c] = a
    contrib = {}
    for label, x, y in (("전체 (②−①)", "2_ALL", "1_B0"), ("C2 (②−③)", "2_ALL", "3_noC2"),
                        ("C3 (②−④)", "2_ALL", "4_noC3"), ("C1 배치 (②−⑤)", "2_ALL", "5_C1rand")):
        if x not in runs or y not in runs:
            continue
        reps = sorted(set(runs[x]) & set(runs[y]))
        out = {}
        for k in ("qps", "strict_qps", "total_p50", "total_p95", "total_p99", "fetch_p50", "queue_wait_p50"):
            dx = [runs[x][r][k] - runs[y][r][k] for r in reps]
            d = float(np.mean(dx))
            noise = max(agg[x][k]["range"], agg[y][k]["range"])
            same_sign = all(v > 0 for v in dx) or all(v < 0 for v in dx)
            out[k] = {"diff": d, "per_rep": dx, "aa_noise": noise,
                      "rel": d / agg[y][k]["mean"] if agg[y][k]["mean"] else None,
                      "verdict": "차이 있음" if (abs(d) > noise and same_sign and len(reps) == 3) else "잡음 범위"}
        contrib[label] = out
    res = {"runs": runs, "agg": agg, "contrib": contrib}
    (ROOT / "logs" / "b2_analysis.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(f"{'조건':16s} {'QPS':>14s} {'strict':>7s} {'p50':>6s} {'p95':>6s} {'p99':>6s} {'qwait50':>7s} {'srch50':>6s} "
          f"{'c2_50':>6s} {'fetch50':>7s} {'chunks':>6s} {'req':>4s} {'err':>5s} {'storMB':>7s}")
    for c in NAMES:
        if c not in agg:
            continue
        a = agg[c]
        f = lambda k, d=1: f"{a[k]['mean']:.{d}f}" if k in a else "-"  # noqa: E731
        print(f"{NAMES[c]:16s} {a['qps']['mean']:7.1f}±{a['qps']['range']:5.1f} {f('strict_qps')} {f('total_p50')} {f('total_p95')} "
              f"{f('total_p99')} {f('queue_wait_p50', 2)} {f('search_p50')} {f('c2_select_p50', 3)} {f('fetch_p50')} "
              f"{f('fetched_chunks')} {f('fetch_requests', 2)} {f('error_rate', 3)} "
              f"{a['storage_read_bytes']['mean'] / 2**20 if 'storage_read_bytes' in a else float('nan'):7.2f}")
    for label, out in contrib.items():
        print(label, {k: f"{v['diff']:+.2f} (noise {v['aa_noise']:.2f}, {v['verdict']})" for k, v in out.items()})


if __name__ == "__main__":
    main()
