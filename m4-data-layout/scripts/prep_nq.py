"""Step 1: NQ(BEIR)에서 위키 문서 단위로 일부를 골라 chunk(passage) 데이터 구성 + 임베딩 + 정확한 top-k
- 같은 위키 문서의 passage들은 앞뒤 chunk 관계 -> Context Expansion(±w) 측정 가능
- top-k는 numpy brute-force(정확)로 미리 계산 -> 배치가 달라도 검색 결과는 동일, Context 읽기만 비교
"""
import argparse, json, os, random, re, urllib.request, zipfile
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.environ.get("M4_DATA", os.path.expanduser("~/m4-exp/data")))
ap.add_argument("--queries", type=int, default=2000)
ap.add_argument("--target-chunks", type=int, default=150000)
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip"
zp = os.path.join(a.out, "nq.zip")
alt = os.path.expanduser("~/m4-exp/data/nq.zip")  # 이미 받은 파일 재사용
if not os.path.exists(zp) and os.path.exists(alt): zp = alt
if not os.path.exists(zp):
    print("downloading NQ (~0.5GB) ...", flush=True); urllib.request.urlretrieve(URL, zp + ".part"); os.rename(zp + ".part", zp)
z = zipfile.ZipFile(zp)

print("reading corpus ...", flush=True)
title_rank, by_title, did2title = {}, {}, {}
with z.open("nq/corpus.jsonl") as f:
    for line in f:
        x = json.loads(line); t = x.get("title", ""); d = x["_id"]
        if t not in by_title: by_title[t] = []; title_rank[t] = len(title_rank)
        by_title[t].append((int(re.sub(r"\D", "", d)), d, x["text"])); did2title[d] = t
queries = {}
with z.open("nq/queries.jsonl") as f:
    for line in f: x = json.loads(line); queries[x["_id"]] = x["text"]
qrels = {}
with z.open("nq/qrels/test.tsv") as f:
    next(f)
    for line in f:
        q, d, s = line.decode().strip().split("\t")
        if int(s) > 0 and d in did2title: qrels.setdefault(q, []).append(d)

rng = random.Random(a.seed)
qs = sorted(q for q in qrels if q in queries); rng.shuffle(qs); qs = qs[: a.queries]
chosen = {did2title[d] for q in qs for d in qrels[q]}
n = sum(len(by_title[t]) for t in chosen)
others = [t for t in by_title if t not in chosen]; rng.shuffle(others)
for t in others:  # 관련 없는 문서도 섞어 넣어 현실적인 corpus 크기로
    if n >= a.target_chunks: break
    chosen.add(t); n += len(by_title[t])
titles = sorted(chosen, key=lambda t: title_rank[t])  # corpus 원래 순서 = 문서 순서

chunks, dstart, dend = [], [], []
for di, t in enumerate(titles):
    ps = sorted(by_title[t]); s = len(chunks)
    for pos, (_, d, text) in enumerate(ps):
        chunks.append({"cid": len(chunks), "doc": di, "pos": pos, "title": t, "text": text})
    dstart += [s] * len(ps); dend += [len(chunks)] * len(ps)
print(f"docs={len(titles)} chunks={len(chunks)} queries={len(qs)}", flush=True)

import torch; torch.set_num_threads(a.threads)
from sentence_transformers import SentenceTransformer
m = SentenceTransformer(a.model, device="cpu")
D = m.encode([c["title"] + " " + c["text"] for c in chunks], batch_size=256, show_progress_bar=True, normalize_embeddings=True).astype(np.float32)
Q = m.encode([queries[q] for q in qs], batch_size=256, normalize_embeddings=True).astype(np.float32)
topk = np.zeros((len(Q), a.k), np.int64)
for i in range(0, len(Q), 256):
    s = Q[i:i+256] @ D.T
    idx = np.argpartition(-s, a.k, 1)[:, : a.k]
    topk[i:i+256] = np.take_along_axis(idx, np.argsort(-np.take_along_axis(s, idx, 1), 1), 1)

perm = np.random.default_rng(a.seed).permutation(len(qs)); ntr = int(len(qs) * 0.7)
np.save(os.path.join(a.out, "emb.npy"), D); np.save(os.path.join(a.out, "topk.npy"), topk)
np.save(os.path.join(a.out, "train_q.npy"), perm[:ntr]); np.save(os.path.join(a.out, "test_q.npy"), perm[ntr:])
np.save(os.path.join(a.out, "doc_bounds.npy"), np.stack([dstart, dend], 1))
with open(os.path.join(a.out, "chunks.jsonl"), "w", encoding="utf-8") as f:
    for c in chunks: f.write(json.dumps(c, ensure_ascii=False) + "\n")

# 접근 패턴 통계 (DB 없이)
cdoc = np.array([c["doc"] for c in chunks])
st = {"chunks": len(chunks), "docs": len(titles), "queries": len(qs), "k": a.k,
      "avg_distinct_docs_in_topk": float(np.mean([len(set(cdoc[t])) for t in topk])),
      "avg_chunks_per_doc": len(chunks) / len(titles)}
freq = np.bincount(topk[perm[:ntr]].ravel(), minlength=len(chunks))
st["test_topk_seen_in_train"] = float(np.isin(topk[perm[ntr:]], np.nonzero(freq)[0]).mean())
json.dump(st, open(os.path.join(a.out, "prep_stats.json"), "w"), indent=1); print(st)
