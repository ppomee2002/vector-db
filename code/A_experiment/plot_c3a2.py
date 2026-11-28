"""A 실험 그래프 2장 (판정과 무관한 시각화). 결과 폴더의 summary.json·trace를 읽어 analyze_c3a2.py와 같은 값으로 그린다.
사용: py plot_c3a2.py <SESS> <출력 폴더> [--warm 15]
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
NS = [1, 4, 6, 8, 12, 16]
COND = {"A1": ("A1 무작위 배치", "#2a78d6", "-", "o"), "A2": ("A2 문서 순 배치", "#eb6834", "--", "s"),
        "A3": ("A3 문서 순 + BC1", "#1baf7a", ":", "^")}

ap = argparse.ArgumentParser()
ap.add_argument("sess")
ap.add_argument("out")
ap.add_argument("--warm", type=float, default=15.0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
qps = defaultdict(list)
stage = defaultdict(lambda: defaultdict(list))
for line in open(os.path.join(a.sess, "runs.tsv"), encoding="utf-8"):
    lab, rep, cond, n, rc, out, when = line.rstrip("\n").split("\t")
    d = os.path.join(a.sess, "out", os.path.basename(out))
    s = json.load(open(os.path.join(d, "summary.json"), encoding="utf-8"))
    qps[(cond, int(n))].append(s["qps"])
    rows = list(csv.DictReader(open(os.path.join(a.sess, "trace", lab + ".csv"), encoding="utf-8")))
    meas = [r for r in rows if r["is_warmup"] == "0" and r["error"] == "0"]
    for k in ("search_ms", "fetch_ms"):
        stage[(cond, int(n))][k].append(np.mean([float(r[k]) for r in meas]))
    stage[(cond, int(n))]["fetch_p95"].append(np.percentile([float(r["fetch_ms"]) for r in meas], 95))

fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
for c, (name, col, ls, mk) in COND.items():
    m = [np.mean(qps[(c, n)]) for n in NS]
    lo = [min(qps[(c, n)]) for n in NS]
    hi = [max(qps[(c, n)]) for n in NS]
    ax.plot(range(len(NS)), m, ls, color=col, marker=mk, lw=2, label=name)
    ax.fill_between(range(len(NS)), lo, hi, color=col, alpha=0.10, lw=0)
ax.set_xticks(range(len(NS)), [str(n) for n in NS])
ax.set_xlabel("동시 요청 수 (C3 끔, m3)")
ax.set_ylabel("QPS (3회 평균, 띠 = 3회 범위)")
ax.set_ylim(0, None)
ax.grid(axis="y", color="#e1e0d9")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=9)
ax.set_title("동시 수별 처리량 (warm, 실제 질문 3,452개)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(a.out, "fig1_qps_by_concurrency.png"))

fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
for c, (name, col, ls, mk) in COND.items():
    ax.plot(range(len(NS)), [np.mean(stage[(c, n)]["search_ms"]) for n in NS], ls, color="#6250d6", marker=mk, lw=1.6,
            alpha=0.9, label=f"{c} 검색 평균")
    ax.plot(range(len(NS)), [np.mean(stage[(c, n)]["fetch_ms"]) for n in NS], ls, color="#eb6834", marker=mk, lw=1.6,
            alpha=0.9, label=f"{c} Fetch 평균")
ax.set_xticks(range(len(NS)), [str(n) for n in NS])
ax.set_xlabel("동시 요청 수 (C3 끔, m3)")
ax.set_ylabel("단계 시간 평균 (ms)")
ax.set_ylim(0, None)
ax.grid(axis="y", color="#e1e0d9")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8, ncol=3)
ax.set_title("동시 수별 검색·Fetch 시간 (세 조건 거의 겹침)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(a.out, "fig2_stage_time_by_concurrency.png"))
print("ok")
