import numpy as np, time, pymetis, scipy.sparse as sp
E = "/workspace/jihyeon/emb"; L = "/workspace/jihyeon/layouts"; N = 2681468
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
tr = np.load(f"{E}/q_train_top20_ids.npy")[:, :10]
ln = np.load(f"{L}/len.npy").astype(np.int64); off = np.concatenate([[0], np.cumsum(ln)])
kl = np.load(f"{L}/label_K.npy"); src = np.memmap(f"{L}/D.bin", dtype=np.uint8, mode="r")

def write_layout(name, order):
    new = np.empty(N, np.int64); cur = 0
    with open(f"{L}/{name}.bin", "wb", buffering=1 << 24) as w:
        for p in order:
            s, e = off[p], off[p + 1]; w.write(src[s:e].tobytes()); new[p] = cur; cur += e - s
    np.save(f"{L}/off_{name}.npy", new); np.save(f"{L}/order_{name}.npy", order)

def chain(A, parts):  # 그룹 안에서 같이 검색된 문서끼리 이어 붙이는 순서
    M = A.shape[0]; ip, ix, dt = A.indptr, A.indices, A.data
    row = np.repeat(np.arange(M), np.diff(ip)); same = parts[row] == parts[ix]
    deg = np.bincount(row[same], weights=dt[same], minlength=M)
    ipl, ixl, dtl, sml = ip.tolist(), ix.tolist(), dt.tolist(), same.tolist()
    placed = [False] * M; seq = []
    for u in np.lexsort((-deg, parts)).tolist():
        while u >= 0 and not placed[u]:
            placed[u] = True; seq.append(u); best, bw = -1, 0
            for j in range(ipl[u], ipl[u + 1]):
                if sml[j] and dtl[j] > bw and not placed[ixl[j]]: bw = dtl[j]; best = ixl[j]
            u = best
    return np.array(seq)

def build(name, qidx, B):
    top = tr[qidx]; nodes, inv = np.unique(top, return_inverse=True); inv = inv.reshape(top.shape); M = len(nodes)
    iu = np.triu_indices(10, 1); r, c = inv[:, iu[0]].ravel(), inv[:, iu[1]].ravel()
    A = sp.coo_matrix((np.ones(len(r), np.int32), (r, c)), shape=(M, M)); A = (A + A.T).tocsr(); A.setdiag(0); A.eliminate_zeros(); A.sort_indices()
    _, parts = pymetis.part_graph(-(-M // B), xadj=A.indptr.tolist(), adjncy=A.indices.tolist(), eweights=A.data.astype(int).tolist())
    parts = np.array(parts); seq = chain(A, parts)
    hot = np.zeros(N, bool); hot[nodes] = True
    cold = np.flatnonzero(~hot); cold = cold[np.lexsort((cold, kl[cold]))]
    order = np.concatenate([nodes[seq], cold]); assert len(np.unique(order)) == N
    write_layout(name, order); sz = np.bincount(parts)
    log(f"{name}: 로그 쿼리 {len(qidx):,}개, 로그에 등장한 passage {M:,}개({M/N:.1%}), 그룹 {len(sz):,}개, 그룹 크기 평균 {sz.mean():.0f} / 최대 {sz.max()}")

full = np.arange(len(tr)); small = np.load(f"{L}/qset_log.npy")
for name, q, B in [("P16", full, 16), ("P64", full, 64), ("P256", full, 256), ("P64s", small, 64)]: build(name, q, B)
log("ALL DONE")
