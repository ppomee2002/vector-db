import json, os
import numpy as np

def load_data(d):
    chunks = [json.loads(l) for l in open(os.path.join(d, "chunks.jsonl"), encoding="utf-8")]
    return (chunks, np.load(os.path.join(d, "topk.npy")), np.load(os.path.join(d, "train_q.npy")),
            np.load(os.path.join(d, "test_q.npy")), np.load(os.path.join(d, "doc_bounds.npy")))

def expand(hits, bounds, w):
    """top-k + 같은 문서 안의 앞뒤 ±w chunk (Context Expansion)"""
    out = []
    for h in hits:
        s, e = bounds[h]
        for j in range(max(s, h - w), min(e, h + w + 1)):
            if j not in out: out.append(int(j))
    return out
