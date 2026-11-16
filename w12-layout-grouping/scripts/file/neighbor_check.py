import numpy as np, csv
E = "/workspace/jihyeon/emb"; L = "/workspace/jihyeon/layouts"; D = "/workspace/jihyeon/data/nq"; B = 4096; N = 2681468
te = np.load(f"{E}/q_test_top20_ids.npy"); doc = np.load(f"{L}/doc.npy"); ln = np.load(f"{L}/len.npy").astype(np.int64)
offs = {x: np.load(f"{L}/off_{x}.npy") for x in ("D", "P64")}
pos = {q: n for n, q in enumerate(open(f"{E}/q_test.ids").read().split("\n"))}; rel = {}
for r in csv.DictReader(open(f"{D}/qrels/test.tsv"), delimiter="\t"):
    rel.setdefault(pos[r["query-id"]], set()).add(int(r["corpus-id"][3:]))
def need(top, w):
    s = []
    for p in top:
        for q in range(p - w, p + w + 1):
            if 0 <= q < N and doc[q] == doc[p] and q not in s: s.append(q)
    return s
def reads(off, pids):
    bl = sorted({b for p in pids for b in range(off[p] // B, (off[p] + ln[p] - 1) // B + 1)})
    return 1 + sum(b > a + 1 for a, b in zip(bl, bl[1:]))
print(f"{'읽는 범위':<10}{'passage수':>9}{'정답포함%':>9}{'정답회수%':>9}{'KB':>7}{'read(D)':>9}{'read(P64)':>10}")
for name, k, w in [("top-5", 5, 0), ("top-5 ±1", 5, 1), ("top-5 ±2", 5, 2), ("top-10", 10, 0), ("top-10 ±1", 10, 1), ("top-15", 15, 0), ("top-20", 20, 0)]:
    a = []
    for i, gold in rel.items():
        p = need(te[i, :k], w); h = set(p) & gold
        a.append((len(p), bool(h), len(h) / len(gold), ln[p].sum() / 1024, reads(offs["D"], p), reads(offs["P64"], p)))
    m = np.mean(a, axis=0)
    print(f"{name:<12}{m[0]:>8.1f}{m[1]*100:>10.1f}{m[2]*100:>10.1f}{m[3]:>9.1f}{m[4]:>8.2f}{m[5]:>10.2f}")
