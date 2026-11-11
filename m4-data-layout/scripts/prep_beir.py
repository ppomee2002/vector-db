"""다른 데이터셋(BEIR) 준비: 문서를 ~100단어 passage(chunk)로 나눔(DPR 방식) -> 임베딩 -> 정확한 top-k.
출력 형식은 prep_nq.py와 같음 (load/measure 그대로 사용)."""
import argparse, json, os, random, urllib.request, zipfile
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="hotpotqa")
ap.add_argument("--out", default=os.environ.get("M4_DATA"))
ap.add_argument("--queries", type=int, default=2000)
ap.add_argument("--target-chunks", type=int, default=150000)
ap.add_argument("--chunk-words", type=int, default=100)
ap.add_argument("--k", type=int, default=10); ap.add_argument("--threads", type=int, default=32)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
zp = os.path.join(a.out, f"{a.dataset}.zip")
if not os.path.exists(zp):
    print(f"downloading {a.dataset} ...", flush=True)
    urllib.request.urlretrieve(f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{a.dataset}.zip", zp + ".part"); os.rename(zp + ".part", zp)
z = zipfile.ZipFile(zp); P = a.dataset
print("reading corpus ...", flush=True)
docs, rank = {}, {}
with z.open(f"{P}/corpus.jsonl") as f:
    for line in f:
        x = json.loads(line); docs[x["_id"]] = (x.get("title", ""), x["text"]); rank[x["_id"]] = len(rank)
queries = {}
with z.open(f"{P}/queries.jsonl") as f:
    for line in f: x = json.loads(line); queries[x["_id"]] = x["text"]
qrels = {}
with z.open(f"{P}/qrels/test.tsv") as f:
    next(f)
    for line in f:
        q, d, s = line.decode().strip().split("\t")
        if int(s) > 0 and d in docs: qrels.setdefault(q, []).append(d)
rng = random.Random(a.seed)
qs = sorted(q for q in qrels if q in queries); rng.shuffle(qs); qs = qs[: a.queries]
def split(text):
    w = text.split(); return [" ".join(w[i:i + a.chunk_words]) for i in range(0, max(len(w), 1), a.chunk_words)]
chosen = {d for q in qs for d in qrels[q]}; n = sum(len(split(docs[d][1])) for d in chosen)
others = [d for d in docs if d not in chosen]; rng.shuffle(others)
for d in others:
    if n >= a.target_chunks: break
    chosen.add(d); n += len(split(docs[d][1]))
chunks, dstart, dend = [], [], []
for di, d in enumerate(sorted(chosen, key=lambda d: rank[d])):  # corpus 원래 순서 = 문서 순서
    t, text = docs[d]; ps = split(text); s = len(chunks)
    for pos, p in enumerate(ps): chunks.append({"cid": len(chunks), "doc": di, "pos": pos, "title": t, "text": p})
    dstart += [s] * len(ps); dend += [len(chunks)] * len(ps)
ndoc = di + 1
print(f"docs={ndoc} chunks={len(chunks)} queries={len(qs)}", flush=True)
import torch; torch.set_num_threads(a.threads)
from sentence_transformers import SentenceTransformer
m = SentenceTransformer(a.model, device="cpu")
D = m.encode([c["title"] + " " + c["text"] for c in chunks], batch_size=256, show_progress_bar=True, normalize_embeddings=True).astype(np.float32)
Q = m.encode([queries[q] for q in qs], batch_size=256, normalize_embeddings=True).astype(np.float32)
topk = np.zeros((len(Q), a.k), np.int64)
for i in range(0, len(Q), 256):
    s = Q[i:i+256] @ D.T; idx = np.argpartition(-s, a.k, 1)[:, : a.k]
    topk[i:i+256] = np.take_along_axis(idx, np.argsort(-np.take_along_axis(s, idx, 1), 1), 1)
perm = np.random.default_rng(a.seed).permutation(len(qs)); ntr = int(len(qs) * 0.7)
np.save(os.path.join(a.out, "emb.npy"), D); np.save(os.path.join(a.out, "topk.npy"), topk)
np.save(os.path.join(a.out, "train_q.npy"), perm[:ntr]); np.save(os.path.join(a.out, "test_q.npy"), perm[ntr:])
np.save(os.path.join(a.out, "doc_bounds.npy"), np.stack([dstart, dend], 1))
with open(os.path.join(a.out, "chunks.jsonl"), "w", encoding="utf-8") as f:
    for c in chunks: f.write(json.dumps(c, ensure_ascii=False) + "\n")
cdoc = np.array([c["doc"] for c in chunks])
freq = np.bincount(topk[perm[:ntr]].ravel(), minlength=len(chunks))
st = {"dataset": a.dataset, "chunks": len(chunks), "docs": ndoc, "queries": len(qs), "k": a.k,
      "avg_distinct_docs_in_topk": float(np.mean([len(set(cdoc[t])) for t in topk])),
      "avg_chunks_per_doc": len(chunks) / ndoc,
      "test_topk_seen_in_train": float(np.isin(topk[perm[ntr:]], np.nonzero(freq)[0]).mean())}
json.dump(st, open(os.path.join(a.out, "prep_stats.json"), "w"), indent=1); print(st)
