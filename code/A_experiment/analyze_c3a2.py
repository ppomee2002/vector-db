"""C3 14주차 A 실험 분석·판정 (규칙: C3_A실험_규칙.md, 측정 전 고정). 결과 폴더(SESS)를 읽기만 한다.
사용: python analyze_c3a2.py --sess <SESS 폴더> [--out <md>]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as stt
from collections import defaultdict

import numpy as np

CONDS = ("A1", "A2", "A3")
NS = (1, 4, 6, 8, 12, 16)
WARM, DUR = 5.0, 20.0
MIB = 1024 * 1024
R_BLOCK, N_CHUNK = 11750, 1_000_000
STAGES = ("search", "c2_select", "fetch")
L: list[str] = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    L.append(s)


def load_runs(sess):
    runs = []
    with open(os.path.join(sess, "runs.tsv"), encoding="utf-8") as f:
        for line in f:
            lab, rep, cond, n, rc, out, when = line.rstrip("\n").split("\t")
            d = os.path.join(sess, "out", os.path.basename(out)) if out else None
            runs.append(dict(label=lab, rep=int(rep), cond=cond, n=int(n), rc=int(rc), dir=d, when=when))
    return runs


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_trace(path):
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({k: (float(v) if v not in ("", None) else None) for k, v in r.items()})
    except OSError:
        pass
    return rows


def read_rec(sess):
    rows = []
    try:
        with open(os.path.join(sess, "rec.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        pass
    return rows


def window_rate(rec_rows, label, t_lo, t_hi, col):
    pts = [(float(r["t_epoch"]), float(r[col])) for r in rec_rows
           if r.get("label") == label and r.get(col) not in ("", None) and t_lo <= float(r["t_epoch"]) <= t_hi]
    if len(pts) < 2:
        return None
    (ta, va), (tb, vb) = pts[0], pts[-1]
    return (vb - va) / (tb - ta) if tb > ta else None


def disk_delta(rec_rows, label, t_lo, t_hi):
    pts = [r for r in rec_rows if r.get("label") == label and t_lo <= float(r["t_epoch"]) <= t_hi]
    if len(pts) < 2:
        return {}
    def parse(s):
        return {k: int(v) for k, v in (x.split(":") for x in (s or "").split() if ":" in x)}
    a, b = parse(pts[0].get("diskstats_read_sectors")), parse(pts[-1].get("diskstats_read_sectors"))
    return {k: (b[k] - a.get(k, 0)) * 512 for k in b if b[k] - a.get(k, 0) > 0}


def run_stats(run, rec_rows, perm):
    s = read_json(os.path.join(run["dir"], "summary.json")) if run["dir"] else None
    m = read_json(os.path.join(run["dir"], "run_meta.json")) if run["dir"] else None
    tr = read_trace(os.path.join(SESS, "trace", run["label"] + ".csv"))
    v = {"V1": run["rc"] == 0 and s is not None and s.get("n_error", 1) == 0}
    srd = (m or {}).get("storage_read_delta") or {}
    miss = srd.get("cache_cell_access_miss_bytes")
    v["V2"] = miss is not None and miss <= MIB
    v["V3"] = srd.get("milvus_restarted") is False
    out = dict(run, valid=all(v.values()), checks=v, miss_bytes=miss, qps=(s or {}).get("qps"))
    if not tr:
        out["valid"] = False
        out["checks"]["trace"] = False
        return out
    t0 = tr[0]["t0_epoch"]
    lo, hi = t0 + WARM, t0 + WARM + DUR
    meas = [r for r in tr if r["is_warmup"] == 0 and r["error"] == 0]
    for st in STAGES:
        xs = [r[f"{st}_ms"] for r in meas if r[f"{st}_ms"] is not None]
        out[f"{st}_mean"] = float(np.mean(xs)) if xs else None
        out[f"{st}_p95"] = float(np.percentile(xs, 95)) if xs else None
    tot = [(r["t_done"] - r["t_enq"]) * 1e3 for r in meas]
    out["total_mean"] = float(np.mean(tot)) if tot else None
    out["total_p95"] = float(np.percentile(tot, 95)) if tot else None
    fx = [r["fetch_ms"] for r in meas if r["fetch_ms"]]
    out["fetch_cv"] = float(np.std(fx, ddof=1) / np.mean(fx)) if len(fx) > 1 else None
    # 단계별 동시 진행 수 (측정 구간 시간 평균). 단계 시작 = t_acq + 앞 단계 시간 합(근사)
    occ = defaultdict(float)
    ivs = []
    for r in tr:
        if r["error"]:
            continue
        a = r["t_acq"]
        segs = {}
        for st in STAGES:
            d = (r[f"{st}_ms"] or 0.0) / 1e3
            segs[st] = (a, a + d)
            a += d
        ivs.append((r, segs["fetch"]))
        for st, (x, y) in segs.items():
            occ[st] += max(0.0, min(y, hi) - max(x, lo))
    for st in STAGES:
        out[f"inflight_{st}"] = occ[st] / DUR
    # 동시 Fetch 간 중복 (측정 구간 안에서 Fetch 구간이 겹치는 질의 쌍)
    res = []
    try:
        with open(os.path.join(run["dir"], "results.jsonl"), encoding="utf-8") as f:
            res = [json.loads(l).get("fetched_ids") or [] for l in f]
    except OSError:
        pass
    if len(res) == len(tr):
        lay = LAYOUT[run["cond"]]
        blk = [set(((np.asarray(ids, dtype=np.int64) if lay == "doc" else perm[np.asarray(ids, dtype=np.int64)]) // R_BLOCK).tolist())
               if ids else set() for ids in res]
        chs = [set(ids) for ids in res]
        # ivs의 순서 = trace 순서 = results.jsonl 순서. 시작 시각으로 정렬해 쓸어 가며 겹치는 Fetch 쌍만 비교(O(F·동시 수))
        F = sorted(((int(r["seq"]), x, y) for r, (x, y) in ivs if y > lo and x < hi), key=lambda z: z[1])
        hc, hb, active = set(), set(), []
        for i, x, y in F:
            active = [(j, y2) for j, y2 in active if y2 > x]
            for j, _ in active:
                if chs[i] & chs[j]:
                    hc.update((i, j))
                if blk[i] & blk[j]:
                    hb.update((i, j))
            active.append((i, y))
        out["conc_fetch_chunk_share"] = len(hc) / len(F) if F else None
        out["conc_fetch_block_share"] = len(hb) / len(F) if F else None
    out["milvus_cores"] = window_rate(rec_rows, run["label"], lo, hi, "milvus_cpu_s")
    out["client_cores"] = window_rate(rec_rows, run["label"], lo, hi, "client_cpu_s")
    nt = window_rate(rec_rows, run["label"], lo, hi, "nr_throttled")
    out["throttled_per_s"] = nt
    out["disk_read_bytes"] = disk_delta(rec_rows, run["label"], lo, hi)
    return out


LAYOUT = {"A1": "rand", "A2": "doc", "A3": "doc"}


def fmt(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


def main():
    global SESS
    ap = argparse.ArgumentParser()
    ap.add_argument("--sess", required=True)
    ap.add_argument("--out")
    ap.add_argument("--warm", type=float, default=5.0, help="모의 점검용. 실측정은 기본값(5초)")
    ap.add_argument("--dur", type=float, default=20.0, help="모의 점검용. 실측정은 기본값(20초)")
    a = ap.parse_args()
    global WARM, DUR
    WARM, DUR = a.warm, a.dur
    SESS = a.sess
    perm = np.random.default_rng(1).permutation(N_CHUNK).astype(np.int64)
    rec_rows = read_rec(SESS)
    stats = [run_stats(r, rec_rows, perm) for r in load_runs(SESS)]
    p(f"# C3 A 실험 결과 (analyze_c3a2.py) — {SESS}\n")
    p(f"- 실행 {len(stats)}개, 유효 {sum(s['valid'] for s in stats)}개")
    for s in stats:
        if not s["valid"]:
            p(f"  - 무효: {s['label']} {s['checks']} miss={s['miss_bytes']}")
    cell = defaultdict(list)
    for s in stats:
        if s["valid"]:
            cell[(s["cond"], s["n"])].append(s)
    usable = {k: v for k, v in cell.items() if len(v) == 3}

    p("\n## 칸별 평균 (3회, 측정 구간)\n")
    p("| 조건 | 동시 | QPS 평균 [범위] | total 평균/p95 | search 평균 | c2 평균 | fetch 평균/p95 | 진행 중 search/fetch | Milvus 코어 | 클라 코어 | fetch CV | 동시 Fetch chunk/블록 공유 | 유효 |")
    p("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    M = {}
    for c in CONDS:
        for n in NS:
            v = cell.get((c, n), [])
            def mean(k):
                xs = [s[k] for s in v if s.get(k) is not None]
                return float(np.mean(xs)) if xs else None
            q = [s["qps"] for s in v if s["qps"] is not None]
            M[(c, n)] = {k: mean(k) for k in ("qps", "total_mean", "total_p95", "search_mean", "c2_select_mean", "fetch_mean",
                                               "fetch_p95", "inflight_search", "inflight_fetch", "milvus_cores", "client_cores",
                                               "fetch_cv", "conc_fetch_chunk_share", "conc_fetch_block_share")}
            mm = M[(c, n)]
            p(f"| {c} | {n} | {fmt(mm['qps'])} [{fmt(min(q)) if q else '-'}~{fmt(max(q)) if q else '-'}] | {fmt(mm['total_mean'])}/{fmt(mm['total_p95'])} | "
              f"{fmt(mm['search_mean'])} | {fmt(mm['c2_select_mean'],2)} | {fmt(mm['fetch_mean'])}/{fmt(mm['fetch_p95'])} | "
              f"{fmt(mm['inflight_search'])}/{fmt(mm['inflight_fetch'])} | {fmt(mm['milvus_cores'],2)}{' (6.5코어 이상)' if (mm['milvus_cores'] or 0) >= 6.5 else ''} | "
              f"{fmt(mm['client_cores'],2)} | {fmt(mm['fetch_cv'],2)} | {fmt(mm['conc_fetch_chunk_share'],3)}/{fmt(mm['conc_fetch_block_share'],3)} | {len(v)} |")

    p("\n## 판정\n")
    J2 = {}
    shares = {}
    for c in CONDS:
        # 규칙: 유효 실행이 3개가 아닌 칸은 판정에 쓰지 않는다(그 칸만 뺀다). J1은 남은 칸으로, J2·J3는 동시 1과 16 칸이 필요.
        miss = [n for n in NS if (c, n) not in usable]
        if miss:
            p(f"- {c}: 판정에서 뺀 칸(유효 3회 아님): 동시 {miss}")
        ns_ok = [n for n in NS if (c, n) in usable]
        if ns_ok:
            qm = {n: M[(c, n)]["qps"] for n in ns_ok}
            nstar = max(ns_ok, key=lambda n: qm[n])
            best = qm[nstar]
            bests = [s["qps"] for s in usable[(c, nstar)]]
            kink = None
            for n in ns_ok:
                if n > nstar and qm[n] < 0.9 * best and max(s["qps"] for s in usable[(c, n)]) < min(bests):
                    kink = n
                    break
            p(f"- {c} J1: 최고 평균 QPS {best:.1f} (동시 {nstar}) → " + (f"꺾임 있음 (동시 {kink}, 평균 {qm[kink]:.1f})" if kink else "꺾임 확인 안 됨"))
        if (c, 1) not in usable or (c, 16) not in usable:
            p(f"- {c} J2: 동시 1 또는 16 칸이 없어 판정 불가")
            continue
        d = {st: M[(c, 16)][f"{st}_mean"] - M[(c, 1)][f"{st}_mean"] for st in STAGES}
        tot = sum(d.values())
        sh = {st: (d[st] / tot if tot > 0 else None) for st in STAGES}
        lab = next((st for st in STAGES if sh[st] is not None and sh[st] >= 0.6), None)
        J2[c] = lab or "고르게 증가"
        p(f"- {c} J2: 동시 1→16 평균 증가 search {d['search']:+.2f} / c2 {d['c2_select']:+.2f} / fetch {d['fetch']:+.2f} ms "
          f"(fetch 몫 {fmt(sh['fetch'] * 100 if sh['fetch'] is not None else None)}%) → 주요 증가 단계 = {J2[c]}")
        per = {}
        for s1 in usable[(c, 1)]:
            s16 = next((s for s in usable[(c, 16)] if s["rep"] == s1["rep"]), None)
            if s16:
                dd = {st: s16[f"{st}_mean"] - s1[f"{st}_mean"] for st in STAGES}
                t = sum(dd.values())
                per[s1["rep"]] = dd["fetch"] / t if t > 0 else None
        shares[c] = per
    for x, y, what in (("A1", "A2", "배치(무작위 vs 문서 순)"), ("A2", "A3", "C2(none vs BC1)")):
        if x not in J2 or y not in J2:
            p(f"- J3 {x} vs {y} ({what}): 판정 불가")
            continue
        reps = sorted(set(shares[x]) & set(shares[y]))
        diffs = [shares[y][r] - shares[x][r] for r in reps if shares[x][r] is not None and shares[y][r] is not None]
        same = len(diffs) == 3 and (all(dv > 0 for dv in diffs) or all(dv < 0 for dv in diffs))
        md = float(np.mean(diffs)) if diffs else None
        diff_lab = J2[x] != J2[y] or (md is not None and abs(md) >= 0.20 and same)
        p(f"- J3 {x} vs {y} ({what}): J2 {J2[x]} / {J2[y]}, fetch 몫 차이(회차별) {[round(dv * 100, 1) for dv in diffs]}%p → "
          + ("단계별 경합 모양이 다름" if diff_lab else "차이 확인 안 됨"))

    p("\n## 참고 (판정 외)\n")
    dsum = defaultdict(lambda: defaultdict(int))
    for s in stats:
        if s["valid"]:
            for k, v in (s.get("disk_read_bytes") or {}).items():
                dsum[(s["cond"], s["n"])][k] += v
    nz = {cn: d for cn, d in dsum.items() if any(v > 0 for v in d.values())}
    p(f"- 측정 구간 장치별 물리 읽기(유효 실행 합): 0보다 큰 칸 {len(nz)}개")
    for (c, n), d in sorted(nz.items()):
        p(f"  - {c} 동시 {n}: " + ", ".join(f"{k} {v / 1e6:.2f}MB" for k, v in sorted(d.items())))
    thr = [s for s in stats if (s.get("throttled_per_s") or 0) > 0]
    p(f"- 우리 컨테이너 throttling이 측정 구간에 늘어난 실행: {len(thr)}개" + (f" ({', '.join(s['label'] for s in thr)})" if thr else ""))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write("\n".join(L) + "\n")


SESS = ""
if __name__ == "__main__":
    main()
