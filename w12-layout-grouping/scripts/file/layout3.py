import numpy as np, time, torch, pymetis, scipy.sparse as sp
E = "/workspace/jihyeon/emb"; L = "/workspace/jihyeon/layouts"; N = 2681468
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
# 0) 측정 쿼리의 top-80 (GPU, 몇 초)
C = torch.from_numpy(np.load(f"{E}/corpus.npy")).cuda()
for name, emb in [("seen", "q_train"), ("unseen", "q_test")]:
    idx = np.load(f"{L}/qset_{name}.npy"); Q = torch.from_numpy(np.load(f"{E}/{emb}.npy")[idx]).cuda()
    I = torch.cat([(Q[s:s + 256] @ C.T).topk(80, dim=1).indices for s in range(0, len(Q), 256)]).cpu().numpy().astype(np.int32)
    old = np.load(f"{E}/{emb}_top20_ids.npy")[idx, :10]
    np.save(f"{E}/{name}_top80.npy", I); log(f"{name} top-80 저장, 기존 top-10과 일치율 {(I[:, :10] == old).mean():.1%}")
del C, Q; torch.cuda.empty_cache()

tr20 = np.load(f"{E}/q_train_top20_ids.npy"); doc = np.load(f"{L}/doc.npy"); kl = np.load(f"{L}/label_K.npy")
ln = np.load(f"{L}/len.npy").astype(np.int64); off = np.concatenate([[0], np.cumsum(ln)])
src = np.memmap(f"{L}/D.bin", dtype=np.uint8, mode="r")
def write_layout(name, order):
    assert len(np.unique(order)) == N; new = np.empty(N, np.int64); cur = 0
    with open(f"{L}/{name}.bin", "wb", buffering=1 << 24) as w:
        for p in order:
            s, e = off[p], off[p + 1]; w.write(src[s:e].tobytes()); new[p] = cur; cur += e - s
    np.save(f"{L}/off_{name}.npy", new); np.save(f"{L}/order_{name}.npy", order); log(f"{name}.bin written")
def chain(A, parts):
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

# 1) P64w: top-20 가중치(top-10 쌍 10, 그 밖 3), 합이 10 미만인 약한 연결은 제외
nodes, inv = np.unique(tr20, return_inverse=True); inv = inv.reshape(tr20.shape).astype(np.int32); M = len(nodes)
iu = np.triu_indices(20, 1); w = np.where(iu[1] < 10, 10, 3).astype(np.int32); A = None
for s in range(0, len(tr20), 20000):
    blk = inv[s:s + 20000]
    Cm = sp.coo_matrix((np.tile(w, len(blk)), (blk[:, iu[0]].ravel(), blk[:, iu[1]].ravel())), shape=(M, M)).tocsr()
    A = Cm if A is None else A + Cm
A = (A + A.T).tocsr(); A.data[A.data < 10] = 0; A.eliminate_zeros()
keep = np.diff(A.indptr) > 0; A = A[keep][:, keep].tocsr(); A.sort_indices(); nodes = nodes[keep]; M = len(nodes)
_, parts = pymetis.part_graph(-(-M // 64), xadj=A.indptr.tolist(), adjncy=A.indices.tolist(), eweights=A.data.astype(int).tolist())
seq = chain(A, np.array(parts)); hot = np.zeros(N, bool); hot[nodes] = True
cold = np.flatnonzero(~hot); cold = cold[np.lexsort((cold, kl[cold]))]
log(f"P64w: 로그에 등장한 passage {M:,}개({M/N:.1%}), 연결 {A.nnz//2:,}개"); del A
write_layout("P64w", np.concatenate([nodes[seq], cold]))

# 2) 하이브리드: P64의 로그 부분 순서를 그대로 쓰고, 나머지는 문서 순
M10 = len(np.unique(tr20[:, :10])); hot_seq = np.load(f"{L}/order_P64.npy")[:M10]
hot = np.zeros(N, bool); hot[hot_seq] = True
write_layout("HC", np.concatenate([hot_seq, np.flatnonzero(~hot)]))
emitted = np.zeros(N, bool); out = []
for p in hot_seq.tolist():
    for q in (p - 1, p, p + 1):
        if 0 <= q < N and doc[q] == doc[p] and not emitted[q]: emitted[q] = True; out.append(q)
log(f"HN: 이웃 포함 로그 영역 {len(out):,}개({len(out)/N:.1%})")
write_layout("HN", np.concatenate([np.array(out, dtype=np.int64), np.flatnonzero(~emitted)]))
log("ALL DONE")
