import numpy as np, torch, time, csv
torch.set_num_threads(4)
E = "/workspace/jihyeon/emb"; D = "/workspace/jihyeon/data/nq"
C = torch.from_numpy(np.load(f"{E}/corpus.npy")).cuda()
print("corpus on GPU:", tuple(C.shape), flush=True)

def topk(name, k=20, bs=512):
    Q = np.load(f"{E}/{name}.npy"); t = time.time()
    I = np.empty((len(Q), k), np.int32); S = np.empty((len(Q), k), np.float16)
    for s in range(0, len(Q), bs):
        v, i = (torch.from_numpy(Q[s:s+bs]).cuda() @ C.T).topk(k, dim=1)
        I[s:s+bs] = i.cpu().numpy(); S[s:s+bs] = v.cpu().numpy()
    np.save(f"{E}/{name}_top{k}_ids.npy", I); np.save(f"{E}/{name}_top{k}_scores.npy", S)
    print(f"{name}: {len(Q):,} queries  {time.time()-t:.0f}s", flush=True)
    return I

It = topk("q_test"); topk("q_train")

# 임베딩 검증: test 쿼리의 정답 passage가 top-10 안에 있는 비율
qid = open(f"{E}/q_test.ids").read().split("\n"); pos = {q: n for n, q in enumerate(qid)}
rel = {}
for r in csv.DictReader(open(f"{D}/qrels/test.tsv"), delimiter="\t"):
    rel.setdefault(r["query-id"], set()).add(int(r["corpus-id"][3:]))
hit = sum(bool(set(It[pos[q], :10]) & docs) for q, docs in rel.items() if q in pos)
print(f"Recall@10 (정답이 top-10에 하나라도 있음): {hit/len(rel):.1%}")
