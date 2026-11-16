# milvus_ann.py: 공통 데이터(임준 NQ 100만 chunk)를 내 DB에 적재 -> 실제 쿼리 3452개 ANN Top-100 -> CSV
import json, time, numpy as np, pandas as pd
from pymilvus import MilvusClient, DataType
J = '/workspace/junlim/data_nq'; D = '/workspace/jaeeun/sel_fetch/data'
DB, COL, K, EF, NQ = 'jaeeun', 'nq1m_sel', 100, 256, 3452
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:6.0f}s]', *a, flush=True)

# 0) chunk 메타 (doc_idx, chunk_idx, 본문 바이트 수)
doc, cix, nb = [], [], []
with open(f'{J}/chunks.jsonl', encoding='utf-8') as f:
    for l in f:
        r = json.loads(l); doc.append(r['doc_idx']); cix.append(r['chunk_idx']); nb.append(len(r['text'].encode('utf-8')))
doc = np.array(doc, np.int64); cix = np.array(cix, np.int32); nb = np.array(nb, np.int32)
np.savez(f'{D}/chunk_meta.npz', doc_idx=doc, chunk_idx=cix, nbytes=nb)
vecs = np.load(f'{J}/chunk_vecs.npy', mmap_mode='r'); n, dim = vecs.shape
assert n == len(doc), 'vector / chunk count mismatch'
log('chunks', n, 'dim', dim, 'docs', int(doc.max()) + 1, '| mean bytes', round(float(nb.mean()), 1))

# 1) 내 database만 사용
root = MilvusClient(uri='http://localhost:19530')
if DB not in root.list_databases():
    root.create_database(DB); log('created database', DB)
root.close()
c = MilvusClient(uri='http://localhost:19530', db_name=DB)
log('db_name =', DB, '| collection =', COL)

# 2) collection 생성 + 적재 (이미 다 들어 있으면 건너뜀)
if c.has_collection(COL):
    cnt = int(c.get_collection_stats(COL).get('row_count', 0))
    log('collection exists, rows =', cnt)
    if cnt < n:
        raise SystemExit('collection이 불완전합니다. 이 로그를 그대로 전달해 주세요.')
else:
    s = c.create_schema(auto_id=False)
    s.add_field('pk', DataType.INT64, is_primary=True)
    s.add_field('doc_idx', DataType.INT64)
    s.add_field('chunk_idx', DataType.INT32)
    s.add_field('vec', DataType.FLOAT_VECTOR, dim=dim)
    ip = c.prepare_index_params()
    ip.add_index('vec', index_type='HNSW', metric_type='IP', params={'M': 16, 'efConstruction': 200})
    c.create_collection(COL, schema=s, index_params=ip)
    B = 5000
    for st in range(0, n, B):
        e = min(st + B, n)
        v = np.asarray(vecs[st:e], dtype=np.float32)
        c.insert(COL, [{'pk': i, 'doc_idx': int(doc[i]), 'chunk_idx': int(cix[i]), 'vec': v[i - st].tolist()} for i in range(st, e)])
        if e % 100000 == 0 or e == n: log('inserted', e)
    c.flush(COL)
    log('flushed')

# 3) 인덱스 완료 대기 + load
for _ in range(240):
    info = c.describe_index(COL, index_name='vec')
    done, pend = int(info.get('indexed_rows', 0)), int(info.get('pending_index_rows', 0))
    log('index', info.get('state'), 'indexed', done, 'pending', pend)
    if done >= n and pend == 0: break
    time.sleep(15)
c.load_collection(COL)
log('load state', c.get_load_state(COL).get('state'))

# 4) ANN Top-100 검색
Q = np.asarray(np.load(f'{J}/query_vecs.npy', mmap_mode='r')[:NQ], dtype=np.float32)
out, ts = [], time.time()
for st in range(0, NQ, 50):
    res = c.search(COL, data=Q[st:st + 50].tolist(), limit=K, search_params={'metric_type': 'IP', 'params': {'ef': EF}})
    for qi, hits in enumerate(res):
        for r, h in enumerate(hits):
            out.append((st + qi, r, int(h['pk']), float(h['distance'])))
log(f'search done: {NQ} queries, {time.time()-ts:.1f}s')
df = pd.DataFrame(out, columns=['q_idx', 'rank', 'chunk_id', 'ann_score'])
df['doc_idx'] = doc[df.chunk_id.values]; df['chunk_idx'] = cix[df.chunk_id.values]; df['nbytes'] = nb[df.chunk_id.values]
df.to_csv(f'{D}/ann_top100_real.csv', index=False)
log('saved ann_top100_real.csv rows', len(df), '| per-query min/max', df.groupby('q_idx').size().min(), df.groupby('q_idx').size().max())

# 5) 검증 A: 정확 검색 대비 ANN Recall@100 (쿼리 200개)
samp = np.random.default_rng(0).choice(NQ, 200, replace=False)
best = np.full((200, K), -1e9, np.float32); bid = np.zeros((200, K), np.int64)
for st in range(0, n, 100000):
    s_ = Q[samp] @ np.asarray(vecs[st:st + 100000], dtype=np.float32).T
    cs = np.concatenate([best, s_], 1); ci = np.concatenate([bid, np.arange(st, st + s_.shape[1])[None].repeat(200, 0)], 1)
    top = np.argpartition(-cs, K, 1)[:, :K]
    best = np.take_along_axis(cs, top, 1); bid = np.take_along_axis(ci, top, 1)
g = df.groupby('q_idx').chunk_id.apply(set)
rec = np.mean([len(g[q] & set(bid[i])) / K for i, q in enumerate(samp)])
log(f'ANN recall@100 vs exact (200 q): {rec:.4f}')

# 6) 검증 B: 정답 chunk가 ANN Top-10 / Top-100 안에 있는 쿼리 비율
qr = pd.read_csv(f'{D}/qrels_chunk.tsv', sep='\t')
rel = qr.groupby('q_idx').chunk_id.apply(set)
for k in (10, 100):
    sub = df[df['rank'] < k].groupby('q_idx').chunk_id.apply(set)
    hit = np.mean([len(sub.get(q, set()) & rel.get(q, set())) > 0 for q in range(NQ)])
    log(f'queries with relevant chunk in ANN top-{k}: {hit:.3f}')

# 7) 메모리 반환
c.release_collection(COL)
log('released', COL, '| state', c.get_load_state(COL).get('state'))
c.close()
log('ALL DONE')
