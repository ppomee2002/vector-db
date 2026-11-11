"""시뮬레이션 vs 실측 비교: payload를 '삽입 순서대로 빈틈없이' 저장했다고 가정했을 때(이상적 연속 배치)
쿼리당 접근 페이지 수를 계산해서, 실제 Qdrant 측정값(results/q_*_cold_each.csv)과 비교.
-> 실제 Qdrant가 삽입 순서(배치)를 얼마나 유지하는지 확인. 읽기 전용, 가벼움."""
import json, os, sys
import csv
import numpy as np
from common import load_data, expand
D = os.environ.get("M4_DATA", os.path.expanduser("~/m4-exp/data"))
R = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/m4-exp/results")
chunks, topk, train, test, bounds = load_data(D)
size = np.array([len(json.dumps({k: c[k] for k in ("text", "title", "doc", "pos")}, ensure_ascii=False).encode()) for c in chunks])
print(f"payload size: mean={size.mean():.0f}B median={np.median(size):.0f}B  (4KB 페이지당 약 {4096/size.mean():.1f}개)")
rows = []
for L in ["interleaved", "stream8", "doc", "doc_1seg", "coaccess", "oracle"]:
    f = os.path.join(D, f"order_{L}.npy")
    if not os.path.exists(f): continue
    order = np.load(f); off = np.zeros(len(order), np.int64); off[order] = np.concatenate([[0], np.cumsum(size[order])[:-1]])
    for P in [4096, 65536]:
        for w in [0, 1, 2]:
            pg = []
            for q in test[:500]:
                ids = expand(topk[q], bounds, w); s = set()
                for i in ids: s.update(range(off[i] // P, (off[i] + size[i] - 1) // P + 1))
                pg.append(len(s))
            m = None
            cf = os.path.join(R, f"q_{L}_w{w}_cold_each.csv")
            if P == 4096 and os.path.exists(cf):
                rr = list(csv.DictReader(open(cf))); key = "pages_pdata" if "pages_pdata" in rr[0] else "pages_payload"
                m = np.mean([float(r[key]) for r in rr])
            rows.append(dict(layout=L, unit=f"{P//1024}KB", window=w, sim_pages=np.mean(pg), measured_qdrant=m))
print(f"{'layout':12s}{'unit':>6s}{'w':>3s}{'sim_pages':>11s}{'measured':>10s}")
for r in rows:
    m = "" if r["measured_qdrant"] is None else "%.2f" % r["measured_qdrant"]
    print(f"{r['layout']:12s}{r['unit']:>6s}{r['window']:3d}{r['sim_pages']:11.2f}{m:>10s}")
with open(os.path.join(R, "sim_vs_measured.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
