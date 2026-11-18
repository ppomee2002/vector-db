#!/usr/bin/env python3
# prep_nq.py: NQ(위키백과) 본문을 64단어 조각으로 나누고 임베딩한다.
# 출력 파일 형식은 prep_fiqa.py와 같다 (chunk_vecs.npy, query_vecs.npy, chunks.jsonl, queries.jsonl, meta.json).
# 같은 위키 글(제목이 같은 문단)을 한 문서로 묶어서, 이웃 조각 확장이 의미가 있게 만든다.
import os, sys, json, time, hashlib, argparse
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("HF_HOME", "/workspace/junlim/hf")
import numpy as np


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def chunk_words(words, size, min_tail):
    pieces = [words[i:i + size] for i in range(0, len(words), size)]
    if len(pieces) >= 2 and len(pieces[-1]) < min_tail:
        tail = pieces.pop()
        pieces[-1] = pieces[-1] + tail
    return [" ".join(p) for p in pieces]


def read_passages(want_words):
    from datasets import load_dataset
    ds = load_dataset("mteb/nq", "corpus", split="corpus", streaming=True)
    rows, total, t0 = [], 0, time.time()
    for r in ds:
        txt = (r.get("text") or "").strip()
        if not txt:
            continue
        rows.append((r.get("title") or "", txt))
        total += len(txt.split())
        if len(rows) % 100000 == 0:
            log(f"passages read {len(rows)}, words {total}, {time.time() - t0:.0f}s")
        if total >= want_words:
            break
    return rows


def read_real_queries():
    from datasets import load_dataset
    ds = load_dataset("mteb/nq", "queries", split="queries")
    return [r["text"].strip() for r in ds if (r.get("text") or "").strip()]


def group_articles(rows):
    order, groups = [], {}
    for i, (title, txt) in enumerate(rows):
        key = title if title else f"__no_title_{i}"
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(txt)
    return [(k, " ".join(groups[k])) for k in order]


def build_chunks(articles, target, size, min_tail):
    texts, doc_idx, chunk_idx, d = [], [], [], 0
    for _, txt in articles:
        pcs = chunk_words(txt.split(), size, min_tail)
        if not pcs:
            continue
        for j, p in enumerate(pcs):
            texts.append(p)
            doc_idx.append(d)
            chunk_idx.append(j)
        d += 1
        if len(texts) >= target:
            break
    return texts[:target], np.array(doc_idx[:target], np.int64), np.array(chunk_idx[:target], np.int32)


def stats(texts, doc_idx):
    n = len(texts)
    n_docs = int(doc_idx.max()) + 1 if n else 0
    per_doc = np.bincount(doc_idx)
    has_nb = float((per_doc[doc_idx] >= 2).mean()) if n else 0.0
    wc = np.array([len(t.split()) for t in texts[:20000]])
    log(f"chunks={n} docs={n_docs} chunks/doc mean={n / max(n_docs, 1):.2f} max={int(per_doc.max())} "
        f"docs with >=2 chunks={float((per_doc >= 2).mean()) * 100:.0f}% chunks that have a neighbor={has_nb * 100:.0f}% "
        f"words/chunk mean={wc.mean():.1f}")
    return n_docs


def embed(model, texts, path, bs, block, sig):
    n = len(texts)
    dim = model.get_sentence_embedding_dimension()
    part, prog = path.replace(".npy", ".partial.npy"), path + ".progress"
    start = 0
    if os.path.exists(part) and os.path.exists(prog):
        try:
            p = json.load(open(prog))
            if p.get("sig") == sig and p.get("n") == n:
                start = int(p["done"])
        except Exception:
            start = 0
    try:
        mm = np.lib.format.open_memmap(part, mode="r+" if start > 0 else "w+", dtype=np.float32, shape=(n, dim))
    except Exception:
        start = 0
        mm = np.lib.format.open_memmap(part, mode="w+", dtype=np.float32, shape=(n, dim))
    if start:
        log(f"resume {os.path.basename(path)} from {start}/{n}")
    t0, done0 = time.time(), start
    for i in range(start, n, block):
        v = model.encode(texts[i:i + block], batch_size=bs, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        mm[i:i + len(v)] = v
        mm.flush()
        done = i + len(v)
        json.dump({"sig": sig, "n": n, "done": done}, open(prog, "w"))
        rate = (done - done0) / max(time.time() - t0, 1e-6)
        log(f"{os.path.basename(path)} {done}/{n} ({rate:.0f}/s, left {(n - done) / max(rate, 1e-6) / 60:.1f} min)")
    del mm
    os.replace(part, path)
    if os.path.exists(prog):
        os.remove(prog)
    return dim


def load_model(a):
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=a.device)
    m.max_seq_length = 128
    if not a.fp32:
        m.half()
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/workspace/junlim/data_nq")
    p.add_argument("--chunks", type=int, default=1000000)
    p.add_argument("--words", type=int, default=64)
    p.add_argument("--min-tail", dest="min_tail", type=int, default=32)
    p.add_argument("--pseudo-q", dest="pseudo_q", type=int, default=100000)
    p.add_argument("--oversample", type=float, default=1.3)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--block", type=int, default=20000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--stats-only", dest="stats_only", action="store_true")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    want = int(a.chunks * a.words * a.oversample)
    while True:
        log(f"reading NQ passages until about {want} words")
        rows = read_passages(want)
        articles = group_articles(rows)
        texts, doc_idx, chunk_idx = build_chunks(articles, a.chunks, a.words, a.min_tail)
        if len(texts) >= a.chunks:
            break
        log(f"only {len(texts)} chunks, reading more")
        want = int(want * 1.3)
    log(f"passages used={len(rows)} articles={len(articles)}")
    n_docs = stats(texts, doc_idx)
    if a.stats_only:
        return

    with open(f"{a.out}/chunks.jsonl", "w", encoding="utf-8") as f:
        for t, d, j in zip(texts, doc_idx.tolist(), chunk_idx.tolist()):
            f.write(json.dumps({"doc_idx": d, "chunk_idx": j, "text": t}, ensure_ascii=False) + "\n")
    np.save(f"{a.out}/doc_idx.npy", doc_idx)
    log("chunks.jsonl written")

    rng = np.random.default_rng(a.seed)
    try:
        real = read_real_queries()
    except Exception as e:
        log("real NQ queries not loaded:", repr(e))
        real = []
    k = min(a.pseudo_q, len(texts))
    pick = rng.choice(len(texts), size=k, replace=False)
    pseudo = []
    for i in pick:
        w = texts[int(i)].split()
        s = int(rng.integers(0, max(1, len(w) - 12)))
        pseudo.append(" ".join(w[s:s + 12]))
    queries = [("real", q) for q in real] + [("pseudo", q) for q in pseudo]
    with open(f"{a.out}/queries.jsonl", "w", encoding="utf-8") as f:
        for kind, q in queries:
            f.write(json.dumps({"kind": kind, "text": q}, ensure_ascii=False) + "\n")
    log(f"queries: real={len(real)} pseudo={len(pseudo)}")

    model = load_model(a)
    sig = hashlib.md5(f"{len(texts)}|{texts[0]}|{texts[len(texts) // 2]}|{texts[-1]}".encode()).hexdigest()
    dim = embed(model, texts, f"{a.out}/chunk_vecs.npy", a.bs, a.block, sig)
    qtexts = [q for _, q in queries]
    qsig = hashlib.md5(f"{len(qtexts)}|{qtexts[0]}|{qtexts[-1]}".encode()).hexdigest()
    embed(model, qtexts, f"{a.out}/query_vecs.npy", a.bs, a.block, qsig)
    json.dump({"source": "mteb/nq corpus (first passages in stream order, grouped by title)",
               "model": "sentence-transformers/all-MiniLM-L6-v2", "dim": dim, "fp16": not a.fp32,
               "n_chunks": len(texts), "n_docs": n_docs, "words": a.words, "min_tail": a.min_tail,
               "n_real_queries": len(real), "n_pseudo_queries": len(pseudo), "seed": a.seed},
              open(f"{a.out}/meta.json", "w"), ensure_ascii=False, indent=1)
    log("DONE")


if __name__ == "__main__":
    main()
