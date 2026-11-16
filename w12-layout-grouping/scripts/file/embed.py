import json, time, numpy as np, torch
torch.set_num_threads(4)
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda").half()
D = "/workspace/jihyeon/data/nq"; O = "/workspace/jihyeon/emb"
QP = "Represent this sentence for searching relevant passages: "

def embed(path, out, text_fn, chunk=50000):
    n = sum(1 for _ in open(path))
    arr = np.lib.format.open_memmap(f"{O}/{out}.npy", mode="w+", dtype=np.float16, shape=(n, 384))
    ids, buf, i, t = [], [], 0, time.time()
    def flush():
        nonlocal i, buf
        arr[i:i+len(buf)] = m.encode(buf, batch_size=256, normalize_embeddings=True)
        i += len(buf); buf = []
        print(f"{out}: {i:,}/{n:,}  {time.time()-t:.0f}s", flush=True)
    with open(path) as f:
        for line in f:
            d = json.loads(line); ids.append(d["_id"]); buf.append(text_fn(d))
            if len(buf) == chunk: flush()
    if buf: flush()
    arr.flush(); open(f"{O}/{out}.ids", "w").write("\n".join(ids))

embed(f"{D}/queries.jsonl", "q_test", lambda d: QP + d["text"])
embed(f"{D}/train_queries.jsonl", "q_train", lambda d: QP + d["text"])
embed(f"{D}/corpus.jsonl", "corpus", lambda d: (d["title"] + " " + d["text"]).strip())
