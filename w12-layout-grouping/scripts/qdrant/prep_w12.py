import numpy as np, json, os, time, torch, pymetis, scipy.sparse as sp
from sentence_transformers import SentenceTransformer
torch.set_num_threads(4); rng = np.random.default_rng(0)
M4 = os.path.expanduser("~/m4-exp/data"); W = "/workspace/jihyeon/w12q"; NQ = "/workspace/jihyeon/data/nq"
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
emb = np.load(f"{M4}/emb.npy"); topk = np.load(f"{M4}/topk.npy"); train = np.load(f"{M4}/train_q.npy"); test = np.load(f"{M4}/test_q.npy")
bounds = np.load(f"{M4}/doc_bounds.npy"); N = len(emb); log(f"chunks={N:,} train_q={len(train)} test_q={len(test)}")

# 1) NQ train 쿼리 13만 개를 지난주와 같은 모델로 임베딩 → 15만 chunk에서 top-10
qs = [json.loads(l)["text"] for l in open(f"{NQ}/train_queries.jsonl")]
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
Q = m.encode(qs, batch_size=512, normalize_embeddings=True, convert_to_tensor=True)
C = torch.from_numpy(emb).cuda()
ext = torch.cat([(Q[s:s + 1024] @ C.T).topk(10, dim=1).indices for s in range(0, len(Q), 1024)]).cpu().numpy()
log(f"NQ train 쿼리 {len(ext):,}개 top-10 완료")

# 2) k-means (로그에 없는 chunk 배치용, 그룹당 약 650개)
NC = 230; X = C.half()
def nearest(Cn): return torch.cat([(X[s:s + 65536] @ Cn.T).argmax(1) for s in range(0, N, 65536)])
Cn = X[torch.from_numpy(rng.choice(N, NC, replace=False)).cuda()].clone()
for it in range(20):
    a = nearest(Cn); S = torch.zeros(NC, emb.shape[1], device="cuda").index_add_(0, a, X.float())
    cnt = torch.bincount(a, minlength=NC); S[cnt == 0] = Cn[cnt == 0].float(); Cn = torch.nn.functional.normalize(S, dim=1).half()
kl = nearest(Cn).cpu().numpy(); log("k-means 완료")

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
def constore_hot(log_top, B=64):
    nodes, inv = np.unique(log_top, return_inverse=True); inv = inv.reshape(log_top.shape); M = len(nodes)
    iu = np.triu_indices(10, 1); r, c = inv[:, iu[0]].ravel(), inv[:, iu[1]].ravel()
    A = sp.coo_matrix((np.ones(len(r), np.int32), (r, c)), shape=(M, M)); A = (A + A.T).tocsr(); A.setdiag(0); A.eliminate_zeros(); A.sort_indices()
    _, parts = pymetis.part_graph(max(2, -(-M // B)), xadj=A.indptr.tolist(), adjncy=A.indices.tolist(), eweights=A.data.astype(int).tolist())
    return nodes[chain(A, np.array(parts))]
def plus_cold_kmeans(hot_seq):
    hot = np.zeros(N, bool); hot[hot_seq] = True; cold = np.flatnonzero(~hot)
    return np.concatenate([hot_seq, cold[np.lexsort((cold, kl[cold]))]])
def save(name, order):
    assert len(np.unique(order)) == N; np.save(f"{W}/orders/order_w12_{name}.npy", order.astype(np.int64))

# 3) 로그: 지난주 train 1,400개 + NQ train 쿼리 (P64: 토요일과 같은 비율 6,000개 / P64x: 전부)
unseen = test[:500]; seen = rng.choice(train, 500, replace=False)
log_s = np.concatenate([topk[train], ext[rng.choice(len(ext), 6000, replace=False)]]); log_x = np.concatenate([topk[train], ext])
save("rand", np.random.default_rng(1).permutation(N)); save("doc", np.arange(N))
for name, lg in [("P64", log_s), ("P64x", log_x)]:
    h = constore_hot(lg); save(name, plus_cold_kmeans(h))
    log(f"{name}: 로그 쿼리 {len(lg):,}개, 로그에 등장한 chunk {len(h):,}개({len(h)/N:.1%}), "
        f"top-10이 로그에 있는 비율: 본 쿼리 {np.isin(topk[seen], h).mean():.1%} / 처음 보는 쿼리 {np.isin(topk[unseen], h).mean():.1%}")
    if name == "P64": hot_s = h
bl = bounds.tolist(); emitted = np.zeros(N, bool); out = []
for p in hot_s.tolist():
    s, e = bl[p]
    for q in (p - 1, p, p + 1):
        if s <= q < e and not emitted[q]: emitted[q] = True; out.append(q)
save("HN", np.concatenate([np.array(out, dtype=np.int64), np.flatnonzero(~emitted)]))
log(f"HN: 이웃 포함 로그 영역 {len(out):,}개({len(out)/N:.1%})")

# 4) 측정용 데이터 폴더 (지난주 파일은 링크만, 쿼리 집합만 다름)
for d, tq in [("data_unseen", unseen), ("data_seen", seen)]:
    for f in ("chunks.jsonl", "emb.npy", "topk.npy", "train_q.npy", "doc_bounds.npy"):
        if not os.path.lexists(f"{W}/{d}/{f}"): os.symlink(f"{M4}/{f}", f"{W}/{d}/{f}")
    np.save(f"{W}/{d}/test_q.npy", tq)
log("PREP DONE")
