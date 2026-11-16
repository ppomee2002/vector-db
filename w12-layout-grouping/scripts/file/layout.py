import numpy as np, json, time, os, torch, pymetis, scipy.sparse as sp
torch.set_num_threads(4); rng = np.random.default_rng(0)
E = "/workspace/jihyeon/emb"; D = "/workspace/jihyeon/data/nq"; L = "/workspace/jihyeon/layouts"
os.makedirs(L, exist_ok=True); N = 2681468; NC = 4096
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

# 1) 쿼리 집합: 로그 3만 / 본 쿼리 1천(로그 안에서) / 처음 보는 쿼리 1천(test)
tr = np.load(f"{E}/q_train_top20_ids.npy"); te = np.load(f"{E}/q_test_top20_ids.npy")
q_log = rng.choice(len(tr), 30000, replace=False)
q_seen = rng.choice(q_log, 1000, replace=False); q_unseen = rng.choice(len(te), 1000, replace=False)
for k, v in [("log", q_log), ("seen", q_seen), ("unseen", q_unseen)]: np.save(f"{L}/qset_{k}.npy", v)
log("query sets saved")

# 2) D(문서 순) 파일 = 원본 순서 그대로
off = np.zeros(N + 1, np.int64)
with open(f"{D}/corpus.jsonl") as f, open(f"{L}/D.bin", "wb", buffering=1 << 24) as w:
    for n, line in enumerate(f):
        d = json.loads(line); b = (d["title"] + "\n" + d["text"]).encode()
        w.write(b); off[n + 1] = off[n] + len(b)
np.save(f"{L}/off_D.npy", off[:-1]); np.save(f"{L}/len.npy", np.diff(off).astype(np.int32))
log(f"D.bin {off[-1]/1e9:.2f} GB")

def write_layout(name, order):
    src = np.memmap(f"{L}/D.bin", dtype=np.uint8, mode="r")
    new = np.empty(N, np.int64); cur = 0
    with open(f"{L}/{name}.bin", "wb", buffering=1 << 24) as w:
        for p in order:
            s, e = off[p], off[p + 1]; w.write(src[s:e].tobytes()); new[p] = cur; cur += e - s
    np.save(f"{L}/off_{name}.npy", new); np.save(f"{L}/order_{name}.npy", order); log(f"{name}.bin written")

# 3) R(랜덤)
write_layout("R", rng.permutation(N))

# 4) K(임베딩 k-means, 4096개 클러스터, GPU)
X = torch.from_numpy(np.load(f"{E}/corpus.npy")).cuda()
def nearest(C):
    a = torch.empty(N, dtype=torch.long, device="cuda")
    for s in range(0, N, 262144): a[s:s+262144] = (X[s:s+262144] @ C.T).argmax(1)
    return a
C = X[torch.from_numpy(rng.choice(N, NC, replace=False)).cuda()].clone()
for it in range(20):
    a = nearest(C); S = torch.zeros(NC, 384, device="cuda")
    for s in range(0, N, 262144): S.index_add_(0, a[s:s+262144], X[s:s+262144].float())
    cnt = torch.bincount(a, minlength=NC); S[cnt == 0] = C[cnt == 0].float()
    C = torch.nn.functional.normalize(S, dim=1).half()
kl = nearest(C).cpu().numpy(); np.save(f"{L}/label_K.npy", kl); log("k-means done")
write_layout("K", np.lexsort((np.arange(N), kl)))

# 5) L(로그 기반 = ConStore 방식): 공동 검색 그래프 → METIS 4096분할, 로그에 없는 passage는 가장 가까운 그룹에
top = tr[q_log, :10]; nodes, inv = np.unique(top, return_inverse=True); inv = inv.reshape(top.shape); M = len(nodes)
iu = np.triu_indices(10, 1); r, c = inv[:, iu[0]].ravel(), inv[:, iu[1]].ravel()
A = sp.coo_matrix((np.ones(len(r), np.int64), (r, c)), shape=(M, M)); A = (A + A.T).tocsr(); A.setdiag(0); A.eliminate_zeros()
_, parts = pymetis.part_graph(NC, xadj=A.indptr.tolist(), adjncy=A.indices.tolist(), eweights=A.data.astype(int).tolist())
parts = np.array(parts); log(f"METIS done: 로그에 등장한 passage {M:,}개")
PC = torch.zeros(NC, 384, device="cuda").index_add_(0, torch.from_numpy(parts).cuda(), X[torch.from_numpy(nodes).cuda()].float())
ll = nearest(torch.nn.functional.normalize(PC, dim=1).half()).cpu().numpy(); ll[nodes] = parts
logged = np.zeros(N, bool); logged[nodes] = True
np.save(f"{L}/label_L.npy", ll); np.save(f"{L}/logged_L.npy", logged)
write_layout("L", np.lexsort((np.arange(N), ~logged, ll)))
log("ALL DONE")
