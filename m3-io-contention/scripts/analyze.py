#!/usr/bin/env python3
"""M3 analysis: bottleneck diagnosis, controlled comparisons, figures.

usage: python3 analyze.py <results_dir>
needs pandas + matplotlib. Writes <results_dir>/diagnosis.md, diagnosis.csv, gains.csv and figs/*.png
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- thresholds (heuristics; the controlled comparisons are the primary evidence) ----
TH = dict(
    CPU_UTIL=0.85,      # server CPU >= 85% of the pinned CPUs
    SEARCH_INFL=1.5,    # search took >= 1.5x its single-thread time
    GAP_SHARE=0.30,     # >= 30% of latency between read calls (goroutines waiting to run)
    IOOP_OVER_DEV=3.0,  # app-observed read >= 3x device await (reads wait for a CPU to resume)
    SCHED_P99_US=1000,  # Go runnable->running p99 >= 1 ms
    RUNQ=1.0,           # on average >= 1 server thread runnable but waiting for a CPU
    AWAIT_X=2.0,        # device await >= 2x its QD1 value ...
    AQU=4.0,            # ... with >= 4 requests queued at the device
    DEV_FRAC=0.80,      # >= 80% of baseline max (only used if the baseline itself hit the device limit)
    QUEUE_SHARE=0.30,   # >= 30% of latency waiting for a worker
    IO_SHARE=0.40,      # >= 40% of latency in the I/O stage
    GAIN=1.30,          # counterpart is >= 1.3x faster
    FREE_X=1.2,         # latency within 1.2x of the lowest load -> no bottleneck yet
)
ROBUST_GAIN = 1.30

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 150, "lines.linewidth": 2, "lines.markersize": 6,
})
BASE_TAGS = ["A_base", "B_base", "C_base", "D_base", "S_base"]
SENS_FAMS = ["base", "k5", "k20", "c16k", "cache100", "cache500", "hit50", "hit90", "nosearch", "light", "heavy"]
COST_FAMS = ["nosearch", "light", "base", "heavy"]  # search CPU cost from none to heavy
# two-sided 95% t quantiles by degrees of freedom
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}


# ------------------------------------------------------------------ loading
def load(d):
    s = pd.read_csv(os.path.join(d, "summary.csv"))
    s["warn"] = s["warn"].fillna("")
    s["frac"] = np.where(s["load"] == "open", (s["rate_target"] / s["capacity"]).round(2), np.nan)
    rd = lambda f: pd.read_csv(os.path.join(d, f)) if os.path.exists(os.path.join(d, f)) else None
    plan = open(os.path.join(d, "plan.txt")).read() if os.path.exists(os.path.join(d, "plan.txt")) else ""
    b, be = rd("baseline.csv"), rd("baseline_end.csv")
    for x in (b, be):
        if x is not None and "engine" not in x:
            x.insert(0, "engine", "psync")
    return s, b, be, plan, rd("llm_probe.csv")


def agg(s, keys):
    num = [c for c in s.select_dtypes("number").columns if c not in keys + ["rep"]]
    g = s.groupby(keys, sort=True)
    m = g[num].mean()
    sd = g[num].std().fillna(0)
    m["warn"] = g["warn"].apply(lambda x: "|".join(sorted({w for v in x for w in v.split("|") if w})))
    m["reps"] = g.size()
    for c in ["mode", "variant", "family", "load"]:
        m[c] = g[c].first()
    m["qps_cv"] = (sd["qps"] / m["qps"]).replace([np.inf], np.nan)
    return m.reset_index(), sd.reset_index()


# ------------------------------------------------------------------ SSD baseline
def baseline_info(b, be, cpu_limit):
    """Per block size: QD1 latency/await (psync), the SSD's capability (aio if measured, else psync)
    and whether that maximum was a device limit or a CPU limit."""
    out = {}
    if b is None:
        return out
    for bs, xb in b.groupby("bs_kb"):
        ps = xb[xb.engine == "psync"].sort_values("qd")
        ai = xb[xb.engine == "aio"].sort_values("qd")
        ref = ps if not ps.empty else ai
        q1 = ref.iloc[0]
        aw1 = q1.dev_r_await_ms if q1.dev_r_await_ms > 0 else q1.lat_mean_us / 1000
        info = dict(await1=aw1, lat1_us=q1.lat_mean_us)
        for name, x in (("psync", ps), ("aio", ai)):
            if x.empty:
                continue
            top = x.loc[x.iops.idxmax()]
            aw_top = top.dev_r_await_ms if top.dev_r_await_ms > 0 else top.lat_mean_us / 1000
            dev_lim = aw_top >= TH["AWAIT_X"] * aw1 and top.dev_aqu >= TH["AQU"]
            cpu_lim = name == "psync" and top.cpu_pct >= TH["CPU_UTIL"] * cpu_limit
            status = ("장치 한계 도달" if dev_lim and not cpu_lim else
                      "CPU 한계 (SSD 최대치는 이보다 높음)" if cpu_lim else "포화 신호 없음")
            info[name] = dict(iops_max=top.iops, mbps_max=top.mbps, qd_max=int(top.qd), cpu_pct=top.cpu_pct,
                              dev_limited=dev_lim, status=status)
        cap = info.get("aio") or info.get("psync")
        info.update(iops_max=cap["iops_max"], mbps_max=cap["mbps_max"], dev_limited=cap["dev_limited"],
                    cap_engine="aio" if "aio" in info else "psync")
        drift = np.nan
        if be is not None:
            e = be[(be.bs_kb == bs) & (be.engine == "psync")].merge(ps, on="qd", suffixes=("_end", ""))
            if not e.empty:
                drift = float((e.iops_end / e.iops - 1).abs().max())
        info["drift"] = drift
        out[int(bs)] = info
    return out


# ------------------------------------------------------------------ diagnosis
def label_point(r, bl, gain_pool, gain_par, n1_lat, th, gain_async=np.nan):
    sig = []
    lat = r.lat_mean_ms if r.lat_mean_ms > 0 else np.nan
    cpu_util = r.cpu_pct / r.cpu_limit_pct
    gap_share = max(r.io_gap_mean_ms, 0) / lat
    infl = r.search_inflation if not pd.isna(r.search_inflation) else 0
    waits = infl >= th["SEARCH_INFL"] or gap_share >= th["GAP_SHARE"] or \
        (not pd.isna(r.ioop_over_dev) and r.ioop_over_dev >= th["IOOP_OVER_DEV"])
    cpu = cpu_util >= th["CPU_UTIL"] or (waits and (r.sched_p99_us >= th["SCHED_P99_US"] or r.runq_threads >= th["RUNQ"]))
    if cpu:
        sig.append(f"CPU {r.cpu_pct:.0f}/{r.cpu_limit_pct:.0f}%, CPU 대기 스레드 {r.runq_threads:.1f}")
    ssd = False
    ref = bl.get(int(r.chunk_kb))
    if ref:
        if ref["await1"] > 0 and r.dev_r_await_ms >= th["AWAIT_X"] * ref["await1"] and r.dev_aqu >= th["AQU"]:
            ssd = True
        if ref["dev_limited"] and (r.our_iops >= th["DEV_FRAC"] * ref["iops_max"] or r.our_mbps >= th["DEV_FRAC"] * ref["mbps_max"]):
            ssd = True
        if ssd:
            sig.append(f"장치 await {r.dev_r_await_ms:.3f}ms (QD1 {ref['await1']:.3f}), 큐 {r.dev_aqu:.1f}")
    q_share = r.queue_mean_ms / lat
    io_share = r.io_mean_ms / lat
    if cpu:
        lab = "CPU 병목"
    elif ssd:
        lab = "저장장치 경합"
    elif q_share >= th["QUEUE_SHARE"] and not pd.isna(gain_async) and gain_async >= th["GAIN"]:
        lab = "동기식 대기 (비동기로 해소됨)"
    elif q_share >= th["QUEUE_SHARE"] and (pd.isna(gain_pool) or gain_pool >= th["GAIN"]):
        lab = "동기식 대기 (worker 부족)" if not pd.isna(gain_pool) else "worker 대기 (대조군 없음)"
    elif q_share >= th["QUEUE_SHARE"]:
        lab = "worker 대기 (늘려도 효과 없음)"
    elif r["mode"] == "sync" and io_share >= th["IO_SHARE"] and not pd.isna(gain_async) and gain_async >= th["GAIN"]:
        lab = "동기식 대기 (비동기로 해소됨)"
    elif r["mode"] == "sync" and io_share >= th["IO_SHARE"] and not pd.isna(gain_par) and gain_par >= th["GAIN"]:
        lab = "동기식 대기 (chunk 순차 읽기)"
    elif r.lat_mean_ms <= th["FREE_X"] * n1_lat:
        lab = "여유"
    else:
        lab = "판단 보류"
    if q_share >= th["QUEUE_SHARE"]:
        sig.append(f"worker 대기 {q_share:.0%}")
    if not pd.isna(gain_pool):
        sig.append(f"pool 제거 x{gain_pool:.2f}")
    if not pd.isna(gain_async):
        sig.append(f"비동기(S) x{gain_async:.2f}")
    if not pd.isna(gain_par):
        sig.append(f"병렬 읽기 x{gain_par:.2f}")
    if r.bg_r_mbps > 0.2 * max(r.dev_r_mbps, 1) or "background_io" in r.warn:
        sig.append("⚠ 배경 I/O")
    return lab, sig, cpu_util, q_share, io_share


def counterpart(m, r, variant):
    x = m[(m.variant == variant) & (m.family == r.family) & (m.n == r.n) & (m.load == "closed")]
    return x.qps.iloc[0] / r.qps if (not x.empty and r.qps > 0) else np.nan


def diagnose(m, bl, th=TH):
    rows = []
    closed = m[m.load == "closed"]
    for tag in sorted(closed.tag.unique()):
        t = closed[closed.tag == tag].sort_values("n")
        n1 = t.iloc[0]
        for _, r in t.iterrows():
            gp = counterpart(m, r, "B") if r.variant == "A" else counterpart(m, r, "D") if r.variant == "C" else np.nan
            gq = counterpart(m, r, "C") if r.variant == "A" else counterpart(m, r, "D") if r.variant == "B" else np.nan
            ga = counterpart(m, r, "S") if r.variant == "A" else np.nan
            lab, sig, cu, qs, ios = label_point(r, bl, gp, gq, n1.lat_mean_ms, th, ga)
            eff = r.qps / (n1.qps * r.n / n1.n) if n1.qps > 0 else np.nan
            rows.append(dict(tag=tag, family=r.family, variant=r.variant, n=int(r.n), workers=int(r.workers), qps=r.qps,
                             qps_cv=r.qps_cv, p95_ms=r.lat_p95_ms, eff=eff, queue=qs, search=r.search_mean_ms / r.lat_mean_ms,
                             io=ios, cpu=cu, runq=r.runq_threads, dev_aqu=r.dev_aqu, await_ms=r.dev_r_await_ms,
                             label=lab, signals="; ".join(sig), warn=r.warn, reps=int(r.reps)))
    return pd.DataFrame(rows)


def label_stability(m, bl, dg):
    alt = []
    for f in (0.8, 1.2):
        frac = ("CPU_UTIL", "DEV_FRAC", "QUEUE_SHARE", "IO_SHARE", "GAP_SHARE")
        th = {k: (v if k == "FREE_X" else min(v * f, 0.95) if k in frac else v * f) for k, v in TH.items()}
        alt.append(diagnose(m, bl, th).set_index(["tag", "n"]).label)
    base = dg.set_index(["tag", "n"]).label
    same = (base == alt[0]) & (base == alt[1])
    dg["stable"] = same.reindex(pd.MultiIndex.from_frame(dg[["tag", "n"]])).values
    return dg


# ------------------------------------------------------------------ paired gains (per repetition)
def paired_gains(s):
    """Per family and n: throughput ratio of two variants measured in the same repetition.
    Geometric mean, 95% t-interval on log ratios (repetitions are the independent units),
    and a verdict that needs both statistical (CI excludes 1) and practical (>= 1.3x) size."""
    c = s[s.load == "closed"]
    piv = c.pivot_table(index=["family", "n", "rep"], columns="variant", values="qps")
    p95 = c.pivot_table(index=["family", "n", "rep"], columns="variant", values="lat_p95_ms")
    rows = []
    pairs = [("B", "A"), ("C", "A"), ("D", "A"), ("S", "A"), ("D", "B"), ("D", "C"), ("S", "D")]
    for (fam, n), grp in piv.groupby(level=[0, 1]):
        for num, den in pairs:
            if num not in grp or den not in grp:
                continue
            rr = (grp[num] / grp[den]).dropna()
            rr = rr[rr > 0]
            if rr.empty:
                continue
            lg = np.log(rr.values)
            k = len(lg)
            gm = float(np.exp(lg.mean()))
            lo = hi = np.nan
            if k >= 2:
                se = lg.std(ddof=1) / np.sqrt(k)
                t = T95.get(k - 1, 2.0)
                lo, hi = float(np.exp(lg.mean() - t * se)), float(np.exp(lg.mean() + t * se))
            pg = p95.loc[(fam, n)]
            lr = (pg[den] / pg[num]).dropna() if (num in pg and den in pg) else pd.Series(dtype=float)
            if k < 2:
                verdict = "반복 부족"
            elif lo > 1 and gm >= ROBUST_GAIN:
                verdict = "확실히 큼"
            elif hi < 1 and gm <= 1 / ROBUST_GAIN:
                verdict = "확실히 느림"
            elif lo > 1:
                verdict = "유의하게 빠름 (1.3배 미만)"
            elif hi < 1:
                verdict = "유의하게 느림"
            else:
                verdict = "차이 불확실"
            rows.append(dict(family=fam, n=int(n), cmp=f"{num}/{den}", qps_gain=gm, ci_low=lo, ci_high=hi,
                             gain_min=float(rr.min()), gain_max=float(rr.max()), reps=k,
                             p95_reduction=float(np.exp(np.log(lr[lr > 0]).mean())) if len(lr) else np.nan,
                             verdict=verdict, robust=verdict == "확실히 큼"))
    return pd.DataFrame(rows)


def overhead(s):
    on = s[(s.tag == "D_base")][["n", "rep", "qps"]]
    off = s[(s.tag == "D_tsoff")][["n", "rep", "qps"]]
    j = off.merge(on, on=["n", "rep"], suffixes=("_off", "_on"))
    if j.empty:
        return None
    j["overhead"] = 1 - j.qps_on / j.qps_off
    return j.groupby("n").overhead.agg(["mean", "min", "max", "count"]).reset_index()


def sens_cmp(gains):
    """The 'remove synchronous waiting' comparison used for sensitivity: S/A (async server) if measured, else D/A."""
    return "S/A" if (not gains.empty and (gains.cmp == "S/A").any() and (gains[gains.cmp == "S/A"].family != "base").any()) else "D/A"


def knee(dg, tag):
    t = dg[dg.tag == tag].sort_values("n")
    hit = t[t.eff < 0.7]
    return int(hit.iloc[0].n) if not hit.empty else None


# ------------------------------------------------------------------ report
def write_md(d, s, m, dg, gains, bl, ovh, plan, llm):
    L = ["# M3 병목 진단 결과", ""]
    if plan:
        L += ["## 실행 정보", "", "```", plan.strip(), "```", ""]
    L += ["## 판정 규칙", "",
          "- **주 근거는 대조군 비교**예요. 하드웨어는 그대로 두고 기다리는 방식만 바꿨을 때(A→B worker 제한 제거, A→C 병렬 읽기, "
          "A→D 둘 다, A→S 비동기 서버) 처리량이 오르면, 원인은 자원이 아니라 대기 구조로 봐요.",
          f"- 같은 회차 안의 짝 비교 배수를 회차를 단위로 묶어 기하평균과 95% 신뢰구간(로그 비율의 t 구간)을 냈어요. "
          f"**신뢰구간 하한 > 1 이면서 배수 ≥ {ROBUST_GAIN}**일 때만 '확실히 큼'이에요. 회차가 3이면 구간이 넓어서 보수적인 판정이에요.",
          "- 지점별 라벨(보조 근거), 우선순위는 CPU > SSD > 동기식 대기예요.",
          "  - CPU: 서버 CPU ≥ 85%, 또는 CPU를 기다린 흔적(검색 1.5배 이상 느려짐, I/O 단계 중 read 밖 시간 ≥ 30%, read 시간 ≥ 장치 await의 3배)이 있고 "
          "Go 스케줄 지연 p99 ≥ 1ms이거나 CPU 대기 스레드가 평균 1개 이상일 때.",
          "  - 저장장치(SSD 또는 가상 디스크): 장치 await ≥ QD1의 2배이면서 장치 큐 ≥ 4일 때. baseline이 장치 한계에 도달한 경우에만 'baseline 최대치의 80%' 규칙도 함께 써요.",
          "  - 동기식 대기: 지연의 30% 이상이 worker 대기이고, worker 제한을 없앤 대조군이 1.3배 이상 빠를 때.",
          "- '안정' 열: 모든 임계값을 ±20% 바꿔도 라벨이 그대로인지 여부예요.", ""]
    if bl:
        L += ["## SSD baseline (O_DIRECT, 서버와 같은 CPU 묶음)", "",
              "psync는 서버와 같은 방식(스레드당 블로킹 읽기 1개)이고, aio는 스레드 하나가 큐 깊이만큼 비동기 읽기를 유지해요. "
              "CPU를 거의 안 쓰므로 aio 최대치가 **SSD 자체의 한계**에 가까워요.", "",
              "| 블록 | QD1 지연 | QD1 장치 await | 방식 | 최대 IOPS (QD) | 최대 MB/s | 그때 CPU | 판정 | 시작↔끝 변화 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for bs, v in bl.items():
            dr = "" if pd.isna(v["drift"]) else f"{v['drift']:.1%}"
            for eng in ("psync", "aio"):
                if eng in v:
                    e = v[eng]
                    L.append(f"| {bs}KB | {v['lat1_us']:.0f} µs | {v['await1']:.3f} ms | {eng} | {e['iops_max']:,.0f} ({e['qd_max']}) | "
                             f"{e['mbps_max']:,.0f} | {e['cpu_pct']:.0f}% | {e['status']} | {dr if eng == 'psync' else ''} |")
        L.append("")
    g = gains[gains.family == "base"] if not gains.empty else gains
    if not g.empty:
        L += ["## 핵심: 대조군 비교 (base, 회차별 짝 비교)", "",
              "| n | 비교 | 처리량 배수 (기하평균) | 95% 신뢰구간 | 회차별 최소~최대 | p95 감소 배수 | 회차 | 판정 |", "|---|---|---|---|---|---|---|---|"]
        for _, r in g.sort_values(["cmp", "n"]).iterrows():
            ci = "" if pd.isna(r.ci_low) else f"{r.ci_low:.2f}~{r.ci_high:.2f}"
            L.append(f"| {r.n} | {r.cmp} | {r.qps_gain:.2f} | {ci} | {r.gain_min:.2f}~{r.gain_max:.2f} | "
                     f"{'' if pd.isna(r.p95_reduction) else f'{r.p95_reduction:.2f}'} | {r.reps} | {r.verdict} |")
        L += ["", "읽는 법: B/A가 크면 worker 제한(동기식 처리)이, C/A가 크면 chunk 순차 읽기가 병목이에요. "
              "S/A는 같은 worker 수(=CPU 수)로 **비동기 I/O 스케줄링**만 바꿨을 때의 이득이고, S/D가 1 근처면 스레드를 요청 수만큼 늘린 것과 "
              "같은 이득을 비동기 방식이 CPU 수만큼의 worker로 얻었다는 뜻이에요.", ""]
    for tag in sorted(dg.tag.unique()):
        t = dg[dg.tag == tag].sort_values("n")
        r0 = m[(m.tag == tag)].iloc[0]
        k = knee(dg, tag)
        sa = "" if pd.isna(r0.search_alone_ms) else f", 검색 단독 {r0.search_alone_ms:.3f} ms"
        L += [f"## {tag}  (io={r0['mode']}, workers={'n' if (t.workers == t.n).all() and len(t) > 1 else int(r0.workers)}, "
              f"k={int(r0.k)}, chunk={int(r0.chunk_kb)}KB, search={'on' if r0.search else 'off'}, hit={r0.hit:.2f}{sa})", "",
              f"꺾이는 지점(확장 효율 < 70%): {'n=' + str(k) if k else '없음'}", "",
              "| n | QPS (CV) | p95 ms | 효율 | 대기/검색/I/O | CPU | CPU대기 | 장치 큐 | 진단 | 안정 | 근거 | 경고 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in t.iterrows():
            cv = "" if pd.isna(r.qps_cv) else f" ({r.qps_cv:.0%})"
            L.append(f"| {r.n} | {r.qps:,.0f}{cv} | {r.p95_ms:.2f} | {r.eff:.0%} | {r.queue:.0%}/{r.search:.0%}/{r.io:.0%} | "
                     f"{r.cpu:.0%} | {r.runq:.1f} | {r.dev_aqu:.1f} | **{r.label}** | {'✓' if r.stable else '△'} | {r.signals} | {r.warn} |")
        L.append("")
    op = m[m.load == "open"]
    if not op.empty:
        L += ["## Open-loop (포아송 도착, 예정 도착 시각부터 잰 지연)", "",
              "| 시나리오 | 부하(용량 대비) | 목표 req/s | 실제 처리 | p50 | p95 | p99 ms | 생성기 지연 ms | 미완료/버림 |", "|---|---|---|---|---|---|---|---|---|"]
        for _, r in op.sort_values(["tag", "frac"]).iterrows():
            L.append(f"| {r.tag} | {r.frac:.2f} | {r.rate_target:,.0f} | {r.qps:,.0f} | {r.lat_p50_ms:.2f} | {r.lat_p95_ms:.2f} | "
                     f"{r.lat_p99_ms:.2f} | {r.gen_lag_ms:.3f} | {r.incomplete:.0f}/{r.dropped:.0f} |")
        L += ["", "용량은 각 회차 시작 시 closed loop(n=64)로 잰 값이에요. 시나리오마다 자기 용량의 비율로 부하를 줬기 때문에, "
              "A·D·S를 비교할 때는 **같은 절대 부하(req/s)**에서 봐야 해요(fig6의 x축). 아래 표는 A의 부하 지점마다 D와 S의 지연을 보간한 값이에요.", ""]
        others = [t for t in ("D_open", "S_open") if t in set(op.tag)]
        if "A_open" in set(op.tag) and others:
            a_ = op[op.tag == "A_open"].sort_values("offered_qps")
            hdr = "| 부하 req/s | A p95 | " + " | ".join(f"{t[0]} p95 (보간)" for t in others) + " | A p99 | " + \
                  " | ".join(f"{t[0]} p99 (보간)" for t in others) + " |"
            L += [hdr, "|---" * (2 + 2 * len(others) + 1) + "|"]
            for _, r in a_.iterrows():
                x = r.offered_qps
                c95, c99 = [], []
                for t in others:
                    o = op[op.tag == t].sort_values("offered_qps")
                    ok = o.offered_qps.min() <= x <= o.offered_qps.max()
                    c95.append(f"{np.interp(x, o.offered_qps, o.lat_p95_ms):.2f}" if ok else "범위 밖")
                    c99.append(f"{np.interp(x, o.offered_qps, o.lat_p99_ms):.2f}" if ok else "범위 밖")
                L.append(f"| {x:,.0f} | {r.lat_p95_ms:.2f} | " + " | ".join(c95) + f" | {r.lat_p99_ms:.2f} | " + " | ".join(c99) + " |")
            L += ["", "'범위 밖'은 그 부하가 해당 시나리오의 측정 범위(용량의 30~100%) 밖이라는 뜻이에요.", ""]
    sc = sens_cmp(gains)
    sg = gains[(gains.cmp == sc) & (gains.family != "base")] if not gains.empty else gains
    if not sg.empty:
        fams = [f for f in SENS_FAMS if f in set(gains.family)]
        ns = sorted(gains[gains.family.isin(fams)].n.unique())
        L += [f"## 민감도: {sc} 처리량 배수 (대기 구조를 풀었을 때의 이득)", "",
              "| family | " + " | ".join(f"n={n}" for n in ns) + " |", "|---|" + "---|" * len(ns)]
        for f in fams:
            x = gains[(gains.family == f) & (gains.cmp == sc)].set_index("n")
            cells = [(f"{x.loc[n].qps_gain:.2f}" + (" ✅" if x.loc[n].robust else "")) if n in x.index else "" for n in ns]
            L.append(f"| {f} | " + " | ".join(cells) + " |")
        L += ["", "✅ = 신뢰구간 하한 > 1 이면서 1.3배 이상. k(검색 결과 수), 청크 크기, LRU 캐시(인기 주제 쏠림), "
              "검색 CPU 비용(없음 → light → base → heavy)을 바꿔도 결론이 유지되는 범위를 보여줘요.", ""]
        hr = m[(m.load == "closed") & (m.family.isin(["cache100", "cache500"]))]
        if not hr.empty:
            L += ["캐시 적중률(관측): " + ", ".join(f"{t} n={int(n)}: {h:.0%}" for t, n, h in zip(hr.tag, hr.n, hr.hit_obs)), ""]
        cost = m[(m.load == "closed") & (m.variant == "A") & (m.family.isin(COST_FAMS))]
        if cost.family.nunique() > 1:
            L += ["### 검색 CPU 비용에 따른 이득 (교차점)", "", "| family | 검색 단독 ms | " +
                  " | ".join(f"{sc} n={n}" for n in ns) + " |", "|---|---|" + "---|" * len(ns)]
            for f in [x for x in COST_FAMS if x in set(cost.family)]:
                sa = cost[cost.family == f].search_alone_ms.mean()
                x = gains[(gains.family == f) & (gains.cmp == sc)].set_index("n")
                cells = [f"{x.loc[n].qps_gain:.2f}" if n in x.index else "" for n in ns]
                L.append(f"| {f} | {'0 (없음)' if pd.isna(sa) else f'{sa:.3f}'} | " + " | ".join(cells) + " |")
            L += ["", "검색이 무거워질수록 CPU가 먼저 막혀 대기 구조를 풀어도 이득이 줄어요. 이득이 1.3배 아래로 떨어지는 검색 비용이 "
                  "'I/O 대기가 병목인 영역'의 경계예요. base는 calibrate.sh로 FAISS에 맞춘 값이에요(plan.txt의 source 참고).", ""]
    if ovh is not None:
        L += ["## 계측 오버헤드 (D_tsoff vs D_base)", "", "| n | 처리량 손실 평균 (최소~최대) | 회차 |", "|---|---|---|"]
        for _, r in ovh.iterrows():
            L.append(f"| {int(r.n)} | {r['mean']:.1%} ({r['min']:.1%}~{r['max']:.1%}) | {int(r['count'])} |")
        L += ["", "초당 계측은 별도 client CPU에서 돌아요. 손실이 반복 간 변동 안이면 계측이 결과를 바꾸지 않았다고 볼 수 있어요.", ""]
    if llm is not None and not llm.empty:
        r = llm.iloc[0]
        base_a = m[(m.tag == "A_base") & (m.load == "closed")]
        L += ["## 생성 단계와 비교 (llm_probe, 선택 측정)", "",
              f"- {r.model} on {r.gpu}: 프롬프트 {int(r.prompt_tokens)} 토큰(검색 결과 포함), {int(r.gen_tokens)} 토큰 생성",
              f"- prefill(첫 토큰까지) {r.prefill_ms_median:.1f} ms, 전체 {r.total_ms_median:.1f} ms, 토큰당 {r.ms_per_token:.2f} ms"]
        if not base_a.empty:
            for _, x in base_a.sort_values("n").iterrows():
                share = x.lat_p95_ms / (x.lat_p95_ms + r.total_ms_median)
                L.append(f"  - A_base n={int(x.n)}: 검색 p95 {x.lat_p95_ms:.2f} ms → 검색+생성 중 검색 비중 {share:.1%}")
        L += ["", "검색 비중이 작아도 검색 서버의 처리량(=비용)과 multi-hop/agent형 RAG의 반복 검색에서는 대기 구조가 그대로 누적돼요.", ""]
    L += ["## 데이터 품질", ""]
    warns = s.warn.str.split("|").explode()
    warns = warns[warns != ""].value_counts()
    L.append("- 경고: " + (", ".join(f"{k} {v}회" for k, v in warns.items()) if len(warns) else "없음"))
    hv = dg[dg.qps_cv > 0.1]
    L.append(f"- 반복 간 처리량 변동계수 > 10%인 지점: {len(hv)}개" + (" (" + ", ".join(f"{a} n={b}" for a, b in zip(hv.tag, hv.n)) + ")" if len(hv) else ""))
    if "gen_lag_ms" in s and (s.load == "open").any():
        L.append(f"- open-loop 생성기 지연 평균: {s[s.load == 'open'].gen_lag_ms.mean():.3f} ms (과부하 지점 포함)")
    L.append("")
    with open(os.path.join(d, "diagnosis.md"), "w") as f:
        f.write("\n".join(L))
    dg.to_csv(os.path.join(d, "diagnosis.csv"), index=False)
    gains.to_csv(os.path.join(d, "gains.csv"), index=False)


# ------------------------------------------------------------------ figures
def xlog(ax, ns, label="concurrent requests (n)"):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(int(v)) for v in ns])
    ax.minorticks_off()
    ax.set_xlabel(label)


def series(ax, m, sd, tags, col, ylabel, scale=1.0, end_label=True, colors=None):
    for i, tag in enumerate(tags):
        t = m[m.tag == tag].sort_values("n")
        if t.empty:
            continue
        e = sd[sd.tag == tag].set_index("n").reindex(t.n)
        c = (colors or COLORS)[i % len(COLORS)]
        ax.errorbar(t.n, t[col] * scale, yerr=e[col].values * scale, color=c, marker="o", capsize=0, elinewidth=1, label=tag)
        if end_label:
            ax.annotate(tag.split("_")[0], (t.n.iloc[-1], t[col].iloc[-1] * scale), xytext=(5, 0), textcoords="offset points",
                        color=INK, fontsize=8, va="center")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)


def hline(ax, y, text, x0):
    ax.axhline(y, color=MUTED, ls="--", lw=1)
    lo, hi = ax.get_ylim()
    ax.set_ylim(0, max(hi, y * 1.12))
    ax.annotate(text, (x0, y), xytext=(2, 4), textcoords="offset points", color=MUTED, fontsize=8)


def title(ax, t):
    ax.set_title(t, loc="left", color=INK)


def figs(d, s, m, sd, b, be, bl, gains):
    out = os.path.join(d, "figs")
    os.makedirs(out, exist_ok=True)
    closed, sdc = m[m.load == "closed"], sd[sd.load == "closed"] if "load" in sd else sd
    main = [t for t in BASE_TAGS if t in set(closed.tag)]
    lim = m.cpu_limit_pct.max()
    if main:
        ns = sorted(closed[closed.tag.isin(main)].n.unique())
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        series(ax[0], closed, sdc, main, "qps", "throughput (requests/s)")
        series(ax[1], closed, sdc, main, "lat_p95_ms", "p95 latency (ms)")
        for a in ax:
            xlog(a, ns)
        ax[0].legend(loc="upper left", fontsize=8)
        title(ax[0], "Throughput")
        title(ax[1], "p95 latency")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig1_qps_p95.png"))
        plt.close(fig)

        comps = [("queue_mean_ms", "wait for worker"), ("search_mean_ms", "vector search (CPU)"),
                 ("io_call_mean_ms", "inside read calls"), ("io_gap_mean_ms", "I/O stage, outside reads")]
        rows = (len(main) + 1) // 2
        fig, axs = plt.subplots(rows, 2, figsize=(11, 3.4 * rows), squeeze=False, sharey=True)  # same scale -> comparable
        for i, tag in enumerate(main):
            a = axs[i // 2][i % 2]
            t = closed[closed.tag == tag].sort_values("n")
            x = np.arange(len(t))
            bottom = np.zeros(len(t))
            for j, (c, name) in enumerate(comps):
                v = t[c].clip(lower=0).values
                a.bar(x, v, bottom=bottom, color=COLORS[j], width=0.7, edgecolor="white", linewidth=1.5, label=name)
                bottom += v
            a.set_xticks(x)
            a.set_xticklabels([str(int(v)) for v in t.n])
            title(a, tag)
            a.set_ylabel("mean latency (ms)")
            a.set_xlabel("concurrent requests (n)")
        for i in range(len(main), rows * 2):
            axs[i // 2][i % 2].axis("off")
        axs[0][0].legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig2_breakdown.png"))
        plt.close(fig)

        fig, axs = plt.subplots(2, 2, figsize=(11, 7))
        series(axs[0][0], closed, sdc, main, "cpu_pct", "server CPU (%)")
        hline(axs[0][0], lim, f"pinned CPUs {lim:.0f}%", ns[0])
        title(axs[0][0], "CPU use")
        series(axs[0][1], closed, sdc, main, "runq_threads", "threads waiting for a CPU (avg)")
        title(axs[0][1], "CPU contention")
        series(axs[1][0], closed, sdc, main, "dev_aqu", "device queue length (aqu-sz)")
        hline(axs[1][0], TH["AQU"], "SSD rule: queue ≥ 4", ns[0])
        title(axs[1][0], "SSD queue")
        series(axs[1][1], closed, sdc, main, "dev_r_await_ms", "device read await (ms)")
        ref = bl.get(int(closed[closed.tag.isin(main)].chunk_kb.iloc[0]))
        if ref:
            hline(axs[1][1], TH["AWAIT_X"] * ref["await1"], "SSD rule: 2x QD1 await", ns[0])
        title(axs[1][1], "SSD latency")
        for a in axs.flat:
            xlog(a, ns)
        axs[0][0].legend(fontsize=8, loc="center right")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig3_resources.png"))
        plt.close(fig)

    if b is not None:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for i, bs in enumerate(sorted(b.bs_kb.unique())):
            for eng, ls, mk in (("psync", "-", "o"), ("aio", "--", "s")):
                x = b[(b.bs_kb == bs) & (b.engine == eng)].sort_values("qd")
                if x.empty:
                    continue
                ax[0].plot(x.qd, x.iops, marker=mk, ls=ls, color=COLORS[i], label=f"{bs} KB {eng}")
                ax[1].plot(x.qd, x.lat_mean_us, marker=mk, ls=ls, color=COLORS[i], label=f"{bs} KB {eng}")
            if be is not None:
                y = be[(be.bs_kb == bs) & (be.engine == "psync")].sort_values("qd")
                if not y.empty:
                    ax[0].plot(y.qd, y.iops, ls=":", color=COLORS[i], alpha=0.7, label=f"{bs} KB psync (end)")
        qs = sorted(b.qd.unique())
        for a, yl in zip(ax, ["IOPS", "mean latency (µs)"]):
            xlog(a, qs, "queue depth (reads in flight)")
            a.set_ylabel(yl)
            a.set_ylim(bottom=0)
        ax[0].legend(fontsize=8)
        title(ax[0], "SSD baseline: IOPS")
        title(ax[1], "SSD baseline: latency")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig4_baseline.png"))
        plt.close(fig)

    stress = [t for t in ["E_cpu", "F_ssd"] if t in set(closed.tag)]
    if stress:
        nss = sorted(closed[closed.tag.isin(stress)].n.unique())
        fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
        series(ax[0], closed, sdc, stress, "qps", "throughput (requests/s)")
        series(ax[1], closed, sdc, stress, "cpu_pct", "server CPU (%)")
        hline(ax[1], lim, f"{lim:.0f}%", nss[0])
        series(ax[2], closed, sdc, stress, "dev_r_mbps", "device read MB/s")
        series(ax[3], closed, sdc, stress, "dev_r_await_ms", "device read await (ms)")
        for a, t in zip(ax, ["Throughput", "CPU", "SSD bandwidth", "SSD latency"]):
            xlog(a, nss)
            title(a, t)
        ax[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig5_validation.png"))
        plt.close(fig)

    op = m[m.load == "open"]
    if not op.empty:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        # x = absolute offered rate, so A and D are compared at the same number of requests per second
        # (each scenario's rates are fractions of its OWN capacity, so the points do not line up exactly)
        for i, tag in enumerate(sorted(op.tag.unique())):
            t = op[op.tag == tag].sort_values("offered_qps")
            for a, col in zip(ax, ["lat_p95_ms", "lat_p99_ms"]):
                a.plot(t.offered_qps, t[col], marker="o", color=COLORS[i], label=f"{tag} (capacity {t.capacity.mean():,.0f}/s)")
                a.axvline(t.capacity.mean(), color=COLORS[i], lw=1, ls=":")
        for a, ttl in zip(ax, ["p95 latency vs offered load", "p99 latency vs offered load"]):
            a.set_xlabel("offered load (requests/s)")
            a.set_ylabel("latency (ms)")
            a.set_yscale("log")
            title(a, ttl)
        ax[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig6_open_loop.png"))
        plt.close(fig)

    if not gains.empty:
        sc = sens_cmp(gains)
        da = gains[gains.cmp == sc]
        wl = [f for f in ["base", "k5", "k20", "c16k", "cache100", "cache500", "hit50", "hit90"] if f in set(da.family)]
        if len(wl) > 1:
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            nsx = sorted(da[da.family.isin(wl)].n.unique())
            for i, f in enumerate(wl):
                x = da[da.family == f].sort_values("n")
                lo = (x.qps_gain - x.ci_low).fillna(0).clip(lower=0)
                hi = (x.ci_high - x.qps_gain).fillna(0).clip(lower=0)
                ax[0].errorbar(x.n, x.qps_gain, yerr=[lo, hi], color=COLORS[i], marker="o", capsize=0, elinewidth=1, label=f)
                a_rows = closed[(closed.variant == "A") & (closed.family == f)].sort_values("n")
                ax[1].plot(a_rows.n, a_rows.io_mean_ms / a_rows.lat_mean_ms, marker="o", color=COLORS[i], label=f)
            ax[0].axhline(1, color=MUTED, lw=1)
            ax[0].axhline(ROBUST_GAIN, color=MUTED, lw=1, ls="--")
            ax[0].set_ylabel(f"throughput gain {sc.replace('/', ' / ')} (95% CI)")
            ax[1].set_ylabel("share of A's latency in the I/O stage")
            ax[1].set_ylim(0, 1)
            for a in ax:
                xlog(a, nsx)
            ax[0].legend(fontsize=8)
            title(ax[0], "Gain from removing synchronous waiting")
            title(ax[1], "How much of A's time is I/O")
            fig.tight_layout()
            fig.savefig(os.path.join(out, "fig7_sensitivity.png"))
            plt.close(fig)
        cf = [f for f in COST_FAMS if f in set(da.family)]
        if len(cf) > 1:
            cost = {f: closed[(closed.variant == "A") & (closed.family == f)].search_alone_ms.mean() for f in cf}
            cost = {f: (0.0 if pd.isna(v) else v) for f, v in cost.items()}
            fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
            cnt = da[da.family.isin(cf)].groupby("n").family.nunique()
            label_n = cnt.idxmax()
            for i, n in enumerate(sorted(da[da.family.isin(cf)].n.unique())):
                x = da[(da.family.isin(cf)) & (da.n == n)].copy()
                x["cost"] = x.family.map(cost)
                x = x.sort_values("cost")
                ax.plot(x.cost, x.qps_gain, marker="o", color=COLORS[i], label=f"n={n}")
                for _, r in x.iterrows():
                    if n == label_n:
                        ax.annotate(r.family, (r.cost, r.qps_gain), xytext=(4, 6), textcoords="offset points", fontsize=8, color=MUTED)
            ax.axhline(1, color=MUTED, lw=1)
            ax.axhline(ROBUST_GAIN, color=MUTED, lw=1, ls="--")
            ax.set_xlabel("vector search CPU time per query (ms, single thread)")
            ax.set_ylabel(f"throughput gain {sc.replace('/', ' / ')}")
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=8)
            title(ax, "Where I/O waiting matters: gain vs search cost")
            fig.tight_layout()
            fig.savefig(os.path.join(out, "fig8_search_cost.png"))
            plt.close(fig)
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    d = sys.argv[1]
    s, b, be, plan, llm = load(d)
    # open-loop rows: one aggregate per (tag, fraction of capacity)
    if (s.load == "open").any():
        mo, sdo = agg(s[s.load == "open"], ["tag", "frac"])
        mo["n"], sdo["n"] = 0, 0
        mc, sdc = agg(s[s.load == "closed"], ["tag", "n"])
        m = pd.concat([mc, mo], ignore_index=True)
        sd = pd.concat([sdc.assign(load="closed"), sdo.assign(load="open")], ignore_index=True)
    else:
        m, sd = agg(s, ["tag", "n"])
        sd = sd.assign(load="closed")
    bl = baseline_info(b, be, m.cpu_limit_pct.max())
    dg = label_stability(m, bl, diagnose(m, bl))
    gains = paired_gains(s)
    ovh = overhead(s)
    write_md(d, s, m, dg, gains, bl, ovh, plan, llm)
    out = figs(d, s, m, sd, b, be, bl, gains)
    pd.set_option("display.width", 220)
    print(dg[["tag", "n", "qps", "p95_ms", "queue", "io", "cpu", "runq", "dev_aqu", "label", "stable"]].to_string(
        index=False, float_format=lambda v: f"{v:.2f}"))
    if not gains.empty:
        print("\npaired gains (base):")
        print(gains[gains.family == "base"].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nwrote {d}/diagnosis.md, diagnosis.csv, gains.csv, {out}/*.png")


if __name__ == "__main__":
    main()
