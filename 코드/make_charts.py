import csv, statistics as st, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 사용법: python make_charts.py [w12실험 폴더] [출력 폴더]
# 기본값: 이 파일이 있는 코드 폴더의 위 폴더(w12실험)에서 CSV를 읽고, w12실험/그래프 에 PNG를 쓴다.
# 필요: matplotlib, 한글 글꼴(맑은 고딕 또는 Noto Sans CJK KR)
HERE = os.path.dirname(os.path.abspath(__file__))
U = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(HERE)
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(U, "그래프")
os.makedirs(OUT, exist_ok=True)

names = {f.name for f in font_manager.fontManager.ttflist}
for cand in ["Noto Sans CJK KR", "Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK JP", "Noto Sans CJK SC"]:
    if cand in names:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff"})
INK, INK2 = "#0b0b0b", "#52514e"
BLUE, ORANGE, GRAY, LGRAY = "#2a78d6", "#eb6834", "#6f6e69", "#b5b4ae"
FAM = {"cap": BLUE, "merge": ORANGE, "base": GRAY}
FAMNAME = {"base": "기준 (동기 m1, 무제한 m3)", "cap": "동시 수 상한 (m6, m7)", "merge": "읽기 병합 (m5, m8)"}

def rows(path): return list(csv.DictReader(open(path, encoding="utf-8-sig")))
def key(r): return (r["method"], int(r["n"]), r["cap"] if r["method"] in ("m7", "m8") else "")
def vals(rs, col): return [float(r[col]) for r in rs]
def title(fig, t, sub):
    fig.text(0.012, 0.975, t, fontsize=13.5, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.915, sub, fontsize=9.5, color=INK2, va="top")
def source(fig, s): fig.text(0.012, 0.012, "자료: " + s, fontsize=8, color=INK2, va="bottom")

# ---------- 1. 같은 시간대 비교: 처리량 vs p99 ----------
cfg = [("m1", 64, "", "m1", "base"), ("m3", 64, "", "m3 무제한", "base"), ("m5", 64, "", "m5", "merge"), ("m6", 64, "", "m6", "cap"),
       ("m7", 64, "8", "m7 상한 8", "cap"), ("m7", 64, "16", "m7 상한 16", "cap"), ("m8", 64, "16", "m8 상한 16", "merge"), ("m8", 64, "32", "m8 상한 32", "merge")]
off = {  # 라벨 위치(점 기준, 포인트 단위) - 겹치지 않게 손으로 정함
    "NQ": {"m1": (8, -3), "m3 무제한": (-8, -3), "m5": (0, -16), "m6": (8, 8), "m7 상한 8": (-8, -6), "m7 상한 16": (8, -3), "m8 상한 16": (8, 3), "m8 상한 32": (8, -3)},
    "FiQA": {"m1": (8, -3), "m3 무제한": (8, -6), "m5": (8, -6), "m6": (8, 2), "m7 상한 8": (-8, -3), "m7 상한 16": (8, 6), "m8 상한 16": (8, -3), "m8 상한 32": (-8, 2)},
}
data = {"NQ": rows(f"{U}/04_같은시간대비교/NQ/results_cond_nq.csv"), "FiQA": rows(f"{U}/04_같은시간대비교/FiQA/results_cond_fiqa.csv")}
fig, axes = plt.subplots(1, 2, figsize=(11, 5.8))
fig.subplots_adjust(left=0.07, right=0.98, top=0.81, bottom=0.22, wspace=0.22)
for ax, ds in zip(axes, ["NQ", "FiQA"]):
    rs_all = [r for r in data[ds] if r["cond"] == "base"]
    for m, n, cap, lab, fam in cfg:
        rs = [r for r in rs_all if key(r) == (m, n, cap)]
        q, p = vals(rs, "qps"), vals(rs, "p99_ms")
        x, y = st.mean(p), st.mean(q)
        ax.plot([x, x], [min(q), max(q)], color=FAM[fam], alpha=0.45, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([x], [y], s=60, color=FAM[fam], edgecolor="white", linewidth=1.5, zorder=3)
        dx, dy = off[ds][lab]
        ax.annotate(f"{lab} · {y:,.0f}", (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, color=INK,
                    ha="left" if dx > 0 else ("right" if dx < 0 else "center"), va="center")
    ax.set_title(ds + (" (100만 조각)" if ds == "NQ" else " (13만 조각)"), loc="left", fontsize=11, color=INK, pad=6)
    ax.set_xlabel("p99 지연 (ms) - 왼쪽일수록 좋음")
    ax.set_xlim(0, 450)
axes[0].set_ylabel("처리량 (QPS) - 위일수록 좋음")
title(fig, "요청 전체 상한(m7)을 맞게 두면(NQ k=8, FiQA k=16) 무제한(m3)보다 처리량 1.16~1.21배, p99는 더 짧다",
      "동시 요청 64, 같은 시간대에 방식을 번갈아 3회. 점은 평균, 세로선은 3회 최소~최대. 읽기 병합은 FiQA에서 처리량이 가장 높지만 p99가 약 2배다.")
handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=FAM[f], markersize=9, label=FAMNAME[f]) for f in ["base", "cap", "merge"]]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.04), fontsize=9.5)
source(fig, "04_같은시간대비교/NQ/results_cond_nq.csv (11/14 11:58~12:13), 04_같은시간대비교/FiQA/results_cond_fiqa.csv (12:19~12:33)")
fig.savefig(f"{OUT}/1_같은시간대비교_처리량_p99.png", dpi=200)
plt.close(fig)

# ---------- 2. 상한 k 변화 ----------
cap = {"NQ": rows(f"{U}/03_상한k변화/results_cap_nq.csv"), "FiQA": rows(f"{U}/03_상한k변화/results_cap_fiqa.csv")}
ks = [2, 4, 6, 8, 12, 16, 32]
fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
fig.subplots_adjust(left=0.07, right=0.98, top=0.81, bottom=0.22, wspace=0.18)
for ax, ds in zip(axes, ["NQ", "FiQA"]):
    for m, color, name in [("m7", BLUE, "요청 전체 상한 (m7)"), ("m8", ORANGE, "상한 + 읽기 병합 (m8)")]:
        xs, med, lo, hi, kk = [], [], [], [], []
        for k in ks:
            q = [float(r["qps"]) for r in cap[ds] if r["method"] == m and int(r["cap"]) == k]
            if not q: continue
            xs.append(ks.index(k)); kk.append(k); med.append(st.median(q)); lo.append(min(q)); hi.append(max(q))
        for x, a, b in zip(xs, lo, hi):
            ax.plot([x, x], [a, b], color=color, alpha=0.35, lw=2, solid_capstyle="round")
        ax.plot(xs, med, color=color, lw=2, marker="o", markersize=6, markeredgecolor="white", markeredgewidth=1.2, label=name)
        i = max(range(len(med)), key=lambda j: med[j])
        ax.annotate(f"최고 k={kk[i]} · {med[i]:,.0f}", (xs[i], med[i]), xytext=(0 if xs[i] < 6 else -7, 10), textcoords="offset points",
                    ha="center" if xs[i] < 6 else "right", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(ks)), [str(k) for k in ks])
    ax.set_ylim(0, None)
    ax.set_title(ds, loc="left", fontsize=11, color=INK, pad=6)
    ax.set_xlabel("동시 수 상한 k (동시 요청 64 중 Milvus에 동시에 보내는 수)")
axes[0].set_ylabel("처리량 (QPS)")
title(fig, "상한만 둔 m7의 최고 구간은 NQ k=6~12, FiQA k=12~16으로 데이터셋마다 다르다",
      "점은 3회 중앙값, 세로선은 3회 최소~최대. 긴 세로선(FiQA k=12, 16 / NQ k=16, 32)은 한 회차의 급락으로 원인 미확인, 데이터에서 빼지 않았다.")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.04), fontsize=9.5)
source(fig, "03_상한k변화/results_cap_fiqa.csv (11/14 11:17~11:31), results_cap_nq.csv (11:32~11:46)")
fig.savefig(f"{OUT}/2_상한k변화.png", dpi=200)
plt.close(fig)

# ---------- 3. 캐시를 비운 직후 방식 비교 ----------
rec = [r for r in rows(f"{U}/05_캐시비운직후/results_reclaim_nq.csv") if r["cond"] == "reclaim"]
cfg3 = [("m7", 64, "8", "m7 상한 8", "cap"), ("m8", 64, "16", "m8 상한 16", "merge"), ("m3", 8, "", "m3 (동시 요청 8)", "base"),
        ("m5", 64, "", "m5 읽기 병합", "merge"), ("m6", 64, "", "m6 읽기 호출 상한 8", "cap"), ("m3", 64, "", "m3 무제한", "base"), ("m1", 64, "", "m1 동기", "base")]
fig, ax = plt.subplots(figsize=(10, 5.2))
fig.subplots_adjust(left=0.20, right=0.97, top=0.80, bottom=0.13)
ys = list(range(len(cfg3)))[::-1]
for y, (m, n, cp, lab, fam) in zip(ys, cfg3):
    rs = [r for r in rec if key(r) == (m, n, cp)]
    q = vals(rs, "qps"); p99 = st.mean(vals(rs, "p99_ms"))
    ax.barh(y, st.mean(q), height=0.62, color=FAM[fam], alpha=0.9)
    ax.plot([min(q), max(q)], [y, y], color=INK, lw=1.2)
    note = f"{st.mean(q):,.0f}   (p99 {p99:,.0f}ms)" if n == 64 else f"{st.mean(q):,.0f}"
    ax.text(max(q) + 8, y, note, va="center", fontsize=9, color=INK)
ax.set_yticks(ys, [c[3] for c in cfg3])
ax.set_xlim(0, 720); ax.grid(axis="y", visible=False)
ax.set_xlabel("처리량 (QPS), 막대는 3회 평균, 검은 선은 최소~최대")
title(fig, "본문 캐시를 비운 직후에는 요청 전체 상한(m7)이 읽기 호출 상한(m6)의 1.46배, 읽기 병합(m5)의 1.38배",
      "NQ, 원래 설정. 방식마다 직전에 Milvus 캐시를 비우고 예열 없이 20초씩 3회(11/15 05:38~05:46, IOPS 제한 없음). 캐시가 따뜻할 때는 m7, m6, m5가 구분되지 않았다.")
source(fig, "05_캐시비운직후/results_reclaim_nq.csv (cond=reclaim)")
fig.savefig(f"{OUT}/3_캐시비운직후_방식비교.png", dpi=200)
plt.close(fig)

# ---------- 4. 미리 읽기 여지: R = 고정 비용 + R_net ----------
srs = [("NQ, 캐시 따뜻함 (11/14 09:4x)", f"{U}/02_NQ_n변화/results_nq_sr.csv", None),
       ("NQ, 캐시 따뜻함 (11/14 11:58)", f"{U}/04_같은시간대비교/NQ/results_cond_nq_sr.csv", None),
       ("NQ, 캐시 비운 직후 (11/15 05:38)", f"{U}/05_캐시비운직후/results_reclaim_nq_sr.csv", "reclaim"),
       ("NQ, 캐시 비운 직후 + IOPS 3,000", f"{U}/05_캐시비운직후/results_reclaim_nq_sr.csv", "reclaim_iops3000"),
       ("NQ, 캐시 비운 직후 + IOPS 1,000", f"{U}/05_캐시비운직후/results_reclaim_nq_sr.csv", "reclaim_iops1000"),
       ("FiQA, 캐시 따뜻함 (11/14 새벽)", f"{U}/01_FiQA_n변화/results_sr.csv", None),
       ("FiQA, 캐시 따뜻함 (11/14 12:19)", f"{U}/04_같은시간대비교/FiQA/results_cond_fiqa_sr.csv", None)]
fig, ax = plt.subplots(figsize=(10, 5.2))
fig.subplots_adjust(left=0.27, right=0.97, top=0.80, bottom=0.14)
ys = list(range(len(srs)))[::-1]
for y, (lab, path, cond) in zip(ys, srs):
    r = [x for x in rows(path) if cond is None or x["cond"] == cond][0]
    fl, rn = float(r["floor_p50"]), float(r["R_net_p50"])
    ax.barh(y, fl, height=0.62, color=LGRAY)
    ax.barh(y, rn, left=fl + 0.03, height=0.62, color=BLUE if "비운" in lab else "#7fa9e0")
    ax.text(fl + rn + 0.12, y, f"R_net {rn:.2f}ms  (R {float(r['R_fetch_p50']):.2f}, 이론 상한 {float(r['gain_max']):.2f}배)", va="center", fontsize=8.8, color=INK)
ax.set_yticks(ys, [s[0] for s in srs])
ax.set_xlim(0, 11.5); ax.grid(axis="y", visible=False)
ax.set_xlabel("본문 조회 시간 R의 중앙값 (ms) = 고정 호출 비용(회색) + 순수 읽기 시간 R_net(파랑)")
title(fig, "캐시를 비운 직후 순수 읽기 시간 R_net이 0.73ms에서 2.3~2.6ms로 약 3.3배 늘었다",
      "요청 300개를 하나씩 보내 잰 중앙값. R의 대부분은 읽는 양과 상관없는 호출 비용이다. 이론 상한 = 검색과 본문 읽기를 완전히 겹칠 때의 배수.")
source(fig, "*_sr.csv (02_NQ_n변화, 04_같은시간대비교, 05_캐시비운직후, 01_FiQA_n변화)")
fig.savefig(f"{OUT}/4_미리읽기여지.png", dpi=200)
plt.close(fig)

# ---------- 5. 동시 요청 수 n 변화 (참고) ----------
grid = {"NQ": rows(f"{U}/02_NQ_n변화/results_nq.csv"), "FiQA": rows(f"{U}/01_FiQA_n변화/results.csv")}
ns = [1, 4, 8, 16, 32, 64]
lines = [("m0", "m0 DB 기본 경로", LGRAY, "--"), ("m2", "m2 요청당 스레드", LGRAY, ":"), ("m4", "m4 쿼리 내 병합", LGRAY, "-."),
         ("m1", "m1 동기", GRAY, "-"), ("m3", "m3 무제한 비동기", INK, "-"), ("m6", "m6 읽기 호출 상한 8", BLUE, "-"), ("m5", "m5 읽기 병합", ORANGE, "-")]
fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.20, wspace=0.18)
for ax, ds in zip(axes, ["NQ", "FiQA"]):
    for m, lab, color, ls in lines:
        y = [st.mean([float(r["qps"]) for r in grid[ds] if r["method"] == m and int(r["n"]) == n]) for n in ns]
        ax.plot(range(len(ns)), y, color=color, lw=2 if color not in (LGRAY,) else 1.5, ls=ls, marker="o", markersize=4, label=lab)
    ax.set_xticks(range(len(ns)), [str(n) for n in ns]); ax.set_ylim(0, None)
    ax.set_title(ds + (" (11/14 10:01~10:55)" if ds == "NQ" else " (11/14 03:50~04:45)"), loc="left", fontsize=11, color=INK, pad=6)
    ax.set_xlabel("동시 요청 수 n")
axes[0].set_ylabel("처리량 (QPS), 3회 평균")
title(fig, "무제한 비동기(m3)는 동시 요청이 늘면 처리량이 떨어진다 (정점 NQ n=8, FiQA n=16)",
      "참고용: 이 측정은 시간대 영향을 크게 받아, 방식 사이의 배수는 같은 시간대 비교(그래프 1)를 쓴다.")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.035), fontsize=9)
source(fig, "02_NQ_n변화/results_nq.csv, 01_FiQA_n변화/results.csv")
fig.savefig(f"{OUT}/5_동시요청수_n변화_참고.png", dpi=200)
plt.close(fig)
print("done", sorted(os.listdir(OUT)))
