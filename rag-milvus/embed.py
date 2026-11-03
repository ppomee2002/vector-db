#!/usr/bin/env python3
"""Embed BEIR corpus/queries on CPU with all cores (default: all-MiniLM-L6-v2 int8 ONNX).

Output (per dataset, under --out/<ds>/):
  corpus.npy  float32 [N,384], L2-normalized, row i = corpus_ids.txt line i
  queries.npy float32 [Q,384], row i = query_ids.txt line i
  corpus_ids.txt, query_ids.txt, meta.json
"""
import argparse, json, os, sys, time
# One compute thread per worker: tokenizers (Rayon) and BLAS otherwise spawn
# a pool per process sized to all CPUs and oversubscribe the machine.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

MODELS = {  # name: (dir, onnx file, pooling, query prefix, dim)
    "bge-small": ("models/bge-small-en-v1.5", "onnx/model.onnx", "cls",
                  "Represent this sentence for searching relevant passages: ", 384),
    "minilm-int8": ("models/all-MiniLM-L6-v2", "onnx/model_qint8_avx512_vnni.onnx", "mean", "", 384),
}
ROOT = Path(__file__).resolve().parent
_sess = _tok = None


def init(threads, model, max_len):
    global _sess, _tok, _pool
    mdir, mfile, _pool, _, _ = MODELS[model]
    import onnxruntime as ort
    from tokenizers import Tokenizer
    import multiprocessing as mp
    idx = mp.current_process()._identity[0] - 1
    ncpu = os.cpu_count()
    cores = {(idx * threads + k) % ncpu for k in range(threads)}
    os.sched_setaffinity(0, cores)
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _sess = ort.InferenceSession(str(ROOT / mdir / mfile), so, providers=["CPUExecutionProvider"])
    _tok = Tokenizer.from_file(str(ROOT / mdir / "tokenizer.json"))
    _tok.enable_truncation(max_len)
    _tok.enable_padding()


def encode(job):
    rows, texts = job
    enc = _tok.encode_batch(texts)
    ids = np.array([e.ids for e in enc], dtype=np.int64)
    mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
    feeds = {"input_ids": ids, "attention_mask": mask}
    if any(i.name == "token_type_ids" for i in _sess.get_inputs()):
        feeds["token_type_ids"] = np.zeros_like(ids)
    h = _sess.run(None, feeds)[0]
    if _pool == "cls":
        out = h[:, 0, :].copy()
    else:  # mean pooling over real tokens (sentence-transformers MiniLM)
        m = mask[..., None].astype(np.float32)
        out = (h * m).sum(1) / np.maximum(m.sum(1), 1e-9)
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return rows, out.astype(np.float32)


def run(texts, path, pool, batch, label):
    n = len(texts)
    order = np.argsort([len(t) for t in texts], kind="stable")  # length bucketing → less padding
    jobs = [(order[i:i + batch], [texts[j] for j in order[i:i + batch]]) for i in range(0, n, batch)]
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, 384))
    t0, done = time.time(), 0
    for rows, vec in pool.imap_unordered(encode, jobs, chunksize=1):
        out[rows] = vec
        done += len(rows)
        if done == n or done // batch % 200 == 0:
            el = time.time() - t0
            print(f"[{label}] {done}/{n} {done/el:.0f}/s eta {(n-done)/max(done/el,1e-9)/60:.1f} min", flush=True)
    out.flush()
    return time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=Path(__file__).resolve().parent / "datasets/raw/hf")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "datasets/emb")
    p.add_argument("--datasets", default="scifact,fiqa,nq,hotpotqa")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--model", choices=sorted(MODELS), default="minilm-int8")
    p.add_argument("--limit", type=int, default=0, help="smoke test: first N corpus rows")
    a = p.parse_args()
    with Pool(a.workers, initializer=init, initargs=(a.threads, a.model, a.max_len)) as pool:
        for ds in a.datasets.split(","):
            d = a.out / ds
            if (d / "meta.json").exists():
                print(f"skip {ds} (done)"); continue
            d.mkdir(parents=True, exist_ok=True)
            c = pq.read_table(next((a.raw / ds).glob("corpus-*.parquet")), columns=["_id", "title", "text"]).to_pydict()
            q = pq.read_table(next((a.raw / ds).glob("queries-*.parquet")), columns=["_id", "text"]).to_pydict()
            if a.limit:
                c = {k: v[:a.limit] for k, v in c.items()}
            ctext = [(t + " " + x).strip() if t else x for t, x in zip(c["title"], c["text"])]
            qtext = [MODELS[a.model][3] + x for x in q["text"]]
            (d / "corpus_ids.txt").write_text("\n".join(map(str, c["_id"])) + "\n")
            (d / "query_ids.txt").write_text("\n".join(map(str, q["_id"])) + "\n")
            tc = run(ctext, d / "corpus.npy", pool, a.batch, f"{ds}/corpus")
            tq = run(qtext, d / "queries.npy", pool, a.batch, f"{ds}/queries")
            meta = dict(dataset=ds, model=a.model, onnx=MODELS[a.model][1], dim=384, pooling=MODELS[a.model][2], normalized=True,
                        max_len=a.max_len, query_prefix=MODELS[a.model][3], corpus=len(ctext), queries=len(qtext),
                        corpus_text="title + ' ' + text", seconds=dict(corpus=tc, queries=tq),
                        workers=a.workers, threads=a.threads, limit=a.limit)
            if not a.limit:
                (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
