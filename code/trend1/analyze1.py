"""C3 14주차 경향 분석 1 (규칙: C3_경향분석1_규칙.md). 기존 결과만 읽는다.
사용: py analyze1.py <c1data 폴더> <출력 md>
"""
import csv, json, math, os, sys
from collections import OrderedDict
import numpy as np

BASE = r"D:\00_KAU\2026_2\캡스톤디자인"
C1 = sys.argv[1]
OUT = sys.argv[2]
N = 1_000_000
BLK = 4 * 1024 * 1024
L = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); L.append(s)

# ---- 접근 집합
z = np.load(os.path.join(BASE, r"w13실험\c3a_real2_1123_0303\c3a_out_real2\fixed_nq.npz"))
top10 = {int(q): set(map(int, t)) for q, t in zip(z["qid"], z["top10"])}
off, flat = z["ctx_off"], z["ctx_flat"]
all1 = {int(q): set(map(int, flat[off[i]:off[i + 1]])) for i, q in enumerate(z["qid"])}
by_top = {frozenset(v): q for q, v in top10.items()}
order = [json.loads(l)["q_idx"] for l in open(os.path.join(BASE, r"w13프로토타입\c3p2\1123_1115\out\c_serial_20261123_201537\results.jsonl"), encoding="utf-8")]
serial = {}
for l in open(os.path.join(BASE, r"w13프로토타입\c3p2\1123_1115\out\c_serial_20261123_201537\results.jsonl"), encoding="utf-8"):
    o = json.loads(l); serial[o["q_idx"]] = o["stage_ms"]
perm = np.random.default_rng(1).permutation(N).astype(np.int64)
PK = {"doc": lambda ids: np.fromiter(ids, np.int64), "rand": lambda ids: perm[np.fromiter(ids, np.int64)]}

def blocks(ids, lay, R):
    return set((PK[lay](ids) // R).tolist())

# ---- 지현 cold
rows = list(csv.DictReader(open(os.path.join(C1, "w13-c1_results_milvus_m1p_cold_20261123_1202_per_query.csv"), encoding="utf-8")))
b0 = [json.loads(l) for l in open(os.path.join(C1, "b0_100_results.jsonl"), encoding="utf-8")]
ans2real = {}
for o in b0:
    k = frozenset(h["chunk_id"] for h in o["hits"])
    if k in by_top: ans2real[o["q_idx"]] = by_top[k]
p(f"지현 100개 질의 중 fixed_nq와 top10 집합이 같은 것: {len(ans2real)}/100")
def cold(lay, rs, mode="one", impl="query"):
    return {int(r["q_idx"]): r for r in rows if r["layout"] == lay and r["read_set"] == rs and r["mode"] == mode and r["impl"] == impl and r["ok"] == "1"}
SETS = {"top10": top10, "all1": all1}

# ---- R 보정 (doc top10)
cd = cold("doc", "top10"); qs = [q for q in cd if q in ans2real]
act = np.mean([int(cd[q]["miss_bytes"]) for q in qs])
best = None
for R in range(1000, 200001, 250):
    pred = np.mean([len(blocks(top10[ans2real[q]], "doc", R)) for q in qs]) * BLK
    if best is None or abs(pred - act) < best[1]: best = (R, abs(pred - act), pred)
R = best[0]
p(f"\n## 블록 모델\n- 보정: doc top10 실제 평균 miss {act/1e6:.1f}MB, R={R} 행 → 예측 {best[2]/1e6:.1f}MB (질의 {len(qs)}개)")
ok_model = True
for lay, rs in (("doc", "all1"), ("rand", "top10"), ("rand", "all1")):
    c = cold(lay, rs); q2 = [q for q in c if q in ans2real]
    a = np.mean([int(c[q]["miss_bytes"]) for q in q2])
    pr = np.mean([len(blocks(SETS[rs][ans2real[q]], lay, R)) for q in q2]) * BLK
    err = (pr - a) / a
    ok_model &= abs(err) <= 0.25
    sp = spearman([len(blocks(SETS[rs][ans2real[q]], lay, R)) for q in q2], [int(c[q]["miss_bytes"]) for q in q2]) if False else None
    p(f"- 검증 {lay} {rs}: 실제 {a/1e6:.1f}MB, 예측 {pr/1e6:.1f}MB, 오차 {err*100:+.0f}%")
p(f"- 블록 모델 사용: {'예' if ok_model else '아니오 (모델 불일치)'}  (전체 블록 수 {math.ceil(N/R)})")

def rankdata(x):
    x = np.asarray(x, float); o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(len(x))
    # 동점 평균 순위
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    s = np.bincount(inv, r); return (s / cnt)[inv]
def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b); return float(np.corrcoef(ra, rb)[0, 1])

# ---- A
p("\n## A. 질의별 Fetch 비용 편차")
def desc(v):
    v = np.asarray(v, float); return f"평균 {v.mean():.1f}, p50 {np.median(v):.1f}, p90 {np.percentile(v,90):.1f}, max {v.max():.1f}, CV {v.std(ddof=1)/v.mean():.2f}"
for lay in ("doc", "rand"):
    for rs in ("bc1", "top10", "all1"):
        c = cold(lay, rs); lat = [float(c[q]["lat_ms"]) for q in c]; mb = [int(c[q]["miss_bytes"]) / 1e6 for q in c]
        p(f"- cold {lay} {rs} (n={len(c)}): 지연ms {desc(lat)} | 읽은MB {desc(mb)}")
c = cold("doc", "bc1"); q2 = [q for q in c if q in ans2real]
lat = [float(c[q]["lat_ms"]) for q in q2]
cv = np.std([float(c[q]["lat_ms"]) for q in c], ddof=1) / np.mean([float(c[q]["lat_ms"]) for q in c])
sp_all1 = spearman([len(blocks(all1[ans2real[q]], "doc", R)) for q in q2], lat)
sp_top = spearman([len(blocks(top10[ans2real[q]], "doc", R)) for q in q2], lat)
sp_cells = spearman([int(c[q]["cell_loads"]) for q in q2], lat)
sp_mb = spearman([int(c[q]["miss_bytes"]) for q in q2], lat)
p(f"- 판정 대상 cold doc bc1: CV {cv:.2f}, Spearman(예측 블록 수[all1 기준], 지연) {sp_all1:.2f} [top10 기준 {sp_top:.2f}] (n={len(q2)})")
p(f"  - 참고(실행 뒤에야 아는 값): Spearman(실제 읽은 바이트, 지연) {sp_mb:.2f}, Spearman(cell_loads, 지연) {sp_cells:.2f}")
A1 = cv >= 0.5; A2 = ok_model and sp_all1 >= 0.5
verdict = "비용 편차 있음, 예측 가능 → 비용 기반 스케줄링 검토할 만함" if A1 and A2 else ("편차는 있으나 실행 전 예측 어려움" if A1 else "편차 작음")
p(f"- **A 판정: {verdict}** (A1 {'충족' if A1 else '불충족'}, A2 {'충족' if A2 else '불충족'})")
fw = [serial[q]["fetch"] for q in order]; sw = [serial[q]["search"] for q in order]; tw = [serial[q]["total"] for q in order]
cvw = np.std(fw, ddof=1) / np.mean(fw)
p(f"- warm c_serial fetch ms (n={len(fw)}): {desc(fw)} → A1 {'충족' if cvw >= 0.5 else '불충족'}")
p(f"- warm c_serial search ms: {desc(sw)}; fetch 비중(합 기준) {np.sum(fw)/np.sum(tw)*100:.0f}%")

# ---- B
p("\n## B. 동시 질의 간 중복 (도착 순서 연속 16개, 겹치지 않는 창 %d개)" % (len(order) // 16))
def dup(units):
    tot = sum(len(u) for u in units); uni = len(set().union(*units)); return 1 - uni / tot if tot else 0.0
for rs in ("top10", "all1"):
    S = SETS[rs]
    ch = np.mean([dup([S[q] for q in order[i:i + 16]]) for i in range(0, len(order) - 15, 16)])
    p(f"- {rs} chunk 단위 중복률 {ch*100:.2f}%")
    for lay in ("doc", "rand"):
        for RR in (R // 2, R, R * 2):
            bl = {q: blocks(S[q], lay, RR) for q in order}
            d = np.mean([dup([bl[q] for q in order[i:i + 16]]) for i in range(0, len(order) - 15, 16)])
            per = np.mean([len(bl[q]) for q in order])
            p(f"  - {lay} 블록 단위(R={RR}) 중복률 {d*100:.1f}% (질의당 블록 {per:.1f})")

def lru(seq, cap):
    c = OrderedDict(); miss = 0
    for bs in seq:
        for b in bs:
            if b in c: c.move_to_end(b)
            else:
                miss += 1; c[b] = 1
                if len(c) > cap: c.popitem(last=False)
    return miss
def grouped(qs, bl, buf=64):
    out = []
    for i in range(0, len(qs), buf):
        rest = qs[i:i + buf]; cur = set()
        while rest:
            j = max(range(len(rest)), key=lambda k: (len(bl[rest[k]] & cur), -k))
            q = rest.pop(j); out.append(q); cur = bl[q]
    return out
p("\n- B2 LRU 모의 (한 번 통과, 도착 순서 vs 64개씩 묶어 재정렬)")
B2 = {}
for rs in ("all1", "top10"):
    for lay in ("doc", "rand"):
        for RR in (R // 2, R, R * 2):
            bl = {q: blocks(SETS[rs][q], lay, RR) for q in order}
            g = grouped(order, bl); nb = math.ceil(N / RR)
            res = []
            for frac in (0.25, 0.5):
                cap = max(1, int(nb * frac)); m0 = lru([bl[q] for q in order], cap); m1 = lru([bl[q] for q in g], cap)
                res.append((frac, m0, m1, 1 - m1 / m0))
                if rs == "all1" and lay == "doc" and RR == R: B2[frac] = 1 - m1 / m0
            p(f"  - {rs} {lay} R={RR} (블록 {nb}): " + ", ".join(f"용량 {int(f*100)}% miss {a}→{b} ({r*100:+.1f}%)" for f, a, b, r in [(f, a, b, -r) for f, a, b, r in res]))
B2ok = ok_model and all(v >= 0.10 for v in B2.values())
p(f"- **B 판정 (doc all1, R={R}): {'공동 스케줄링 검토할 만함' if B2ok else '효과 작음' if ok_model else '판정 불가(블록 모델 불일치)'}** (감소율 " + ", ".join(f"{int(f*100)}% {v*100:.1f}%" for f, v in B2.items()) + ")")

open(OUT, "w", encoding="utf-8").write("# C3 경향 분석 1 결과 (analyze1.py)\n\n" + "\n".join(L) + "\n")
