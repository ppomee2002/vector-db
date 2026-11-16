# embed_train_minilm.py: NQ train 쿼리(김지현 님 train_queries.jsonl, 읽기 전용)를 텍스트 기준 중복 제거 후
# 공통 데이터와 같은 설정(all-MiniLM-L6-v2, max_seq_length=128, fp16, normalize, float32 저장)으로 재임베딩한다.
import os, json, time, resource
import numpy as np

SRC = '/workspace/jihyeon/data/nq/train_queries.jsonl'
J = '/workspace/junlim/data_nq'
D = '/workspace/jaeeun/sel_fetch/data'
OUT_V, OUT_M = f'{D}/q_train_minilm.npy', f'{D}/q_train_minilm_meta.jsonl'
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:6.0f}s]', *a, flush=True)
def maxrss_gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
def norm(t): return ' '.join(t.lower().split())

# 1) _id, text만 읽고 중복 제거 (첫 등장 유지)
ids, texts, cnt, seen, n_raw, n_empty = [], [], [], {}, 0, 0
with open(SRC, encoding='utf-8') as f:
    for l in f:
        d = json.loads(l); n_raw += 1
        t = (d.get('text') or '').strip()
        if not t: n_empty += 1; continue
        k = norm(t)
        if k in seen: cnt[seen[k]] += 1; continue
        seen[k] = len(ids); ids.append(d['_id']); texts.append(t); cnt.append(1)
log(f'raw {n_raw} | empty {n_empty} | dedup {len(ids)} | removed dups {n_raw - n_empty - len(ids)}')

# 2) 모델 (prep_nq.py와 같은 설정)
import torch
from sentence_transformers import SentenceTransformer
torch.set_num_threads(4)
m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
m.max_seq_length = 128
m.half()
def enc(x): return m.encode(x, batch_size=256, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

# 3) 검증: queries.jsonl 앞 3개 vs query_vecs.npy 앞 3행
vq = [json.loads(l) for _, l in zip(range(3), open(f'{J}/queries.jsonl', encoding='utf-8'))]
ref = np.asarray(np.load(f'{J}/query_vecs.npy', mmap_mode='r')[:3], dtype=np.float32)
mine = enc([r['text'] for r in vq])
cos = (mine * ref).sum(1) / np.linalg.norm(mine, axis=1) / np.linalg.norm(ref, axis=1)
log('verify kinds', [r['kind'] for r in vq], '| cosine', [round(float(c), 6) for c in cos], '| min', round(float(cos.min()), 6))
json.dump({'verify_cosine': [float(c) for c in cos], 'verify_kinds': [r['kind'] for r in vq]},
          open(f'{D}/q_train_minilm_verify.json', 'w'), indent=1)
if cos.min() < 0.99:
    raise SystemExit(f'STOP: verify cosine {cos.min():.6f} < 0.99')

# 4) 임베딩 + 저장
mm = np.lib.format.open_memmap(OUT_V + '.partial.npy', mode='w+', dtype=np.float32, shape=(len(texts), 384))
B = 20000
for st in range(0, len(texts), B):
    mm[st:st + B] = enc(texts[st:st + B]); mm.flush()
    log(f'embedded {min(st + B, len(texts))}/{len(texts)}')
del mm
os.replace(OUT_V + '.partial.npy', OUT_V)
with open(OUT_M, 'w', encoding='utf-8') as f:
    for i, t, c in zip(ids, texts, cnt):
        f.write(json.dumps({'_id': i, 'text': t, 'n_dup': c}, ensure_ascii=False) + '\n')
chk = np.load(OUT_V, mmap_mode='r')
log('saved', OUT_V, chk.shape, chk.dtype, '| norms first5', np.round(np.linalg.norm(chk[:5], axis=1), 4))
log(f'max RSS {maxrss_gb():.2f} GB | GPU max alloc {torch.cuda.max_memory_allocated()/2**30:.2f} GB')
log('ALL DONE')
