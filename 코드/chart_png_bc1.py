"""C2 p8 그래프를 PNG로 (Canva가 pptx의 산점도 차트를 못 읽어서). 숫자는 pf3_quality_nq.csv에서.
chart_png_bc1.py: chart_png.py에서 강조하는 규칙만 B -> BC1(규칙 B+C1)로 바꾼 것. B는 회색 점으로 간다."""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

QUAL, OUT = sys.argv[1], sys.argv[2]
W, H = float(sys.argv[3]) if len(sys.argv) > 3 else 8.7, float(sys.argv[4]) if len(sys.argv) > 4 else 3.42
Q = {r["rule"]: r for r in csv.DictReader(open(QUAL, encoding="utf-8")) if r["subset"] == "ans"}
ans = {k: 100 * float(v["ans"]) for k, v in Q.items()}
pks = {k: float(v["fetched_pks"]) for k, v in Q.items()}

reg = fm.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
bold = fm.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = reg.get_name()

BLUE, DARK, LIGHT, AXIS, MUTED, GRID, SPINE = "#0B6FB8", "#404040", "#C8C8C8", "#808080", "#6B6B6B", "#E3E3E3", "#BFBFBF"
fig, ax = plt.subplots(figsize=(W, H), dpi=300)
fig.patch.set_facecolor("white")
others = ["C1", "C3", "A3", "B", "A"]
ax.scatter([pks[k] for k in others], [ans[k] for k in others], s=80, color=LIGHT, zorder=3, linewidths=0)
ax.scatter([pks[k] for k in ("none", "all")], [ans[k] for k in ("none", "all")], s=160, color=DARK, zorder=4, linewidths=0)
ax.scatter([pks["BC1"]], [ans["BC1"]], s=240, color=BLUE, zorder=5, linewidths=0)

def lab(k, text, color, dx, dy, ha, va):
    ax.annotate(text, (pks[k], ans[k]), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                fontproperties=bold, fontsize=16, color=color, zorder=6)
lab("none", f"top-10만 {ans['none']:.1f}%", DARK, 10, 0, "left", "center")
lab("BC1", f"규칙 B+C1 {ans['BC1']:.1f}%", BLUE, -12, 6, "right", "bottom")
lab("all", f"이웃 전부 {ans['all']:.1f}%", DARK, -12, 2, "right", "bottom")

ax.set_xlim(8, 27); ax.set_ylim(76, 90)
ax.set_xticks([8, 12, 16, 20, 24]); ax.set_yticks(range(76, 91, 2))
ax.grid(axis="y", color=GRID, linewidth=0.75, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(SPINE); ax.spines[s].set_linewidth(0.75)
ax.tick_params(colors=AXIS, labelsize=13, length=0, pad=4)
ax.set_xlabel("요청당 읽은 Chunk 수", fontsize=13, color=MUTED, labelpad=4)
ax.set_ylabel("정답 포함률 (%)", fontsize=13, color=MUTED, labelpad=4)
fig.tight_layout(pad=0.3)
fig.savefig(OUT, dpi=300, facecolor="white")
print("saved", OUT)
