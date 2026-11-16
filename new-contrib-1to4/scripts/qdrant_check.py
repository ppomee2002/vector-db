# qdrant_check.py: Qdrant 교차 검증 - 같은 벡터/같은 HNSW 설정으로 Top-100 검색 후 ③ 질문 ①·② 핵심 지표 재계산
import json, time, numpy as np, pandas as pd
from qdrant_client import QdrantClient, models
J = '/workspace/junlim/data_nq'; D = '/workspace/jaeeun/sel_fetch/data'; R = '/workspace/jaeeun/sel_fetch/results'
COL, K, EF, NQ = 'jaeeun_nq1m_sel', 100, 256, 3452
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:6.0f}s]', *a, flush=True)

# 1) 적재 (내 collection만)
qc = QdrantClient(url='http://localhost:6333', timeout=600)
vecs = np.load(f'{J}/chunk_vecs.npy', mmap_mode='r'); n, dim = vecs.shape
existing = [c.name for c in qc.get_collections().collections]
log('existing qdrant collections:', existing)
if COL in existing:
    cnt = qc.count(COL, exact=True).count; log('collection exists, points =', cnt)
    if cnt < n: raise SystemExit('불완전한 collection이 있습니다. 이 로그를 전달해 주세요.')
else:
    qc.create_collection(COL, vectors_config=models.VectorParams(size=dim, distance=models.Distance.DOT, on_disk=True),
                         hnsw_config=models.HnswConfigDiff(m=16, ef_construct=200, max_indexing_threads=4))
    B = 2000
    for st in range(0, n, B):
        e = min(st + B, n)
        qc.upsert(COL, points=models.Batch(ids=list(range(st, e)), vectors=np.asarray(vecs[st:e], dtype=np.float32).tolist()), wait=True)
        if e % 100000 == 0 or e == n: log('upserted', e)
for _ in range(360):
    info = qc.get_collection(COL)
    log('status', info.status, '| indexed', info.indexed_vectors_count, '| points', info.points_count)
    if info.status == models.CollectionStatus.GREEN: break
    time.sleep(15)

# 2) Top-100 검색
meta = np.load(f'{D}/chunk_meta.npz')
Q = np.asarray(np.load(f'{J}/query_vecs.npy', mmap_mode='r')[:NQ], dtype=np.float32)
sp = models.SearchParams(hnsw_ef=EF); out = []; ts = time.time()
for qi in range(NQ):
    for r, p in enumerate(qc.query_points(COL, query=Q[qi].tolist(), limit=K, search_params=sp).points):
        out.append((qi, r, int(p.id), float(p.score)))
qd = pd.DataFrame(out, columns=['q_idx', 'rank', 'chunk_id', 'ann_score'])
for c in ('doc_idx', 'chunk_idx', 'nbytes'): qd[c] = meta[c][qd.chunk_id.values]
qd.to_csv(f'{D}/ann_top100_real_qdrant.csv', index=False)
log(f'qdrant search done {time.time()-ts:.1f}s, rows {len(qd)}')

# 3) Milvus와 비교 + 정확 검색 대비 recall
mil = pd.read_csv(f'{D}/ann_top100_real.csv')
cmp = {}
for k in (10, 100):
    a = mil[mil['rank'] < k].groupby('q_idx').chunk_id.apply(set); b = qd[qd['rank'] < k].groupby('q_idx').chunk_id.apply(set)
    cmp[f'overlap@{k}'] = float(np.mean([len(a[q] & b[q]) / k for q in range(NQ)]))
samp = np.random.default_rng(0).choice(NQ, 200, replace=False)
best = np.full((200, K), -1e9, np.float32); bid = np.zeros((200, K), np.int64)
for st in range(0, n, 100000):
    s_ = Q[samp] @ np.asarray(vecs[st:st + 100000], dtype=np.float32).T
    cs = np.concatenate([best, s_], 1); ci = np.concatenate([bid, np.arange(st, st + s_.shape[1])[None].repeat(200, 0)], 1)
    top = np.argpartition(-cs, K, 1)[:, :K]; best = np.take_along_axis(cs, top, 1); bid = np.take_along_axis(ci, top, 1)
gq = qd.groupby('q_idx').chunk_id.apply(set)
cmp['qdrant_recall@100_vs_exact'] = float(np.mean([len(gq[q] & set(bid[i])) / K for i, q in enumerate(samp)]))
log('Milvus vs Qdrant:', {k: round(v, 4) for k, v in cmp.items()})

# 4) 리랭킹 점수: 겹치는 후보는 재사용, 새 후보만 계산
old = pd.read_csv(f'{D}/rerank_top100_real.csv')[['q_idx', 'chunk_id', 'ce_score']]
qd = qd.merge(old, on=['q_idx', 'chunk_id'], how='left')
miss = qd[qd.ce_score.isna()]; log('pairs needing new rerank:', len(miss))
if len(miss):
    from sentence_transformers import CrossEncoder
    need = set(miss.chunk_id); text = {}
    with open(f'{J}/chunks.jsonl', encoding='utf-8') as f:
        for cid, l in enumerate(f):
            if cid in need: text[cid] = json.loads(l)['text']
    qs = [json.loads(l)['text'] for l in open(f'{J}/queries.jsonl', encoding='utf-8')][:NQ]
    try: ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda', max_length=256)
    except TypeError: ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')
    qd.loc[miss.index, 'ce_score'] = ce.predict([(qs[q], text[c]) for q, c in zip(miss.q_idx, miss.chunk_id)], batch_size=128, show_progress_bar=False)
qd['ce_rank'] = qd.groupby('q_idx').ce_score.rank(ascending=False, method='first').astype(int) - 1
qd.to_csv(f'{D}/rerank_top100_real_qdrant.csv', index=False)
log('saved rerank_top100_real_qdrant.csv')
del qd, mil, old

# 5) 질문 ①·② 핵심 지표 재계산 (q2_predict.py 특징 코드를 Qdrant 후보에 그대로 적용)
cmp_saved = cmp
src = open('/workspace/jaeeun/sel_fetch/scripts/q2_predict.py', encoding='utf-8').read().replace('rerank_top100_real.csv', 'rerank_top100_real_qdrant.csv')
exec(src.split('gkf = GroupKFold')[0])
from sklearn.model_selection import GroupKFold
qr = pd.read_csv(f'{D}/qrels_chunk.tsv', sep='\t'); rel = qr.groupby('q_idx').chunk_id.apply(set).to_dict()
gold = set(zip(qr.q_idx, qr.chunk_id)); rr['gold'] = [(q, c) in gold for q, c in zip(rr.q_idx, rr.chunk_id)]
def ndcg10(ids_, r):
    dcg = sum(1 / np.log2(i + 2) for i, c in enumerate(ids_[:10]) if c in r)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(r), 10)))
    return dcg / idcg if idcg else 0.0
q1 = []
for q, s in rr.groupby('q_idx'):
    a10 = s.sort_values('rank').chunk_id.head(10).tolist(); c10 = s.sort_values('ce_rank').chunk_id.head(10).tolist()
    q1.append((len(set(a10) & set(c10)), int((s[s.ce_rank < 10]['rank'] >= 10).sum()), ndcg10(a10, rel.get(q, set())), ndcg10(c10, rel.get(q, set()))))
q1 = np.array(q1)
POS = ['f_chunk_pos', 'f_rel_pos', 'f_first_chunk', 'f_doc_len']
MODELS = {'M0': F0, 'M1-noPos': [c for c in F0 + FDOC + FEMB + ['f_nbytes'] if c not in POS]}
gkf = GroupKFold(n_splits=5)
deep = (rr['rank'] >= 10).values; y = rr.y.values; auc_rows = []
for name, cols in MODELS.items():
    p = np.zeros(len(rr))
    for tr, te in gkf.split(rr, rr.y, rr.q_idx):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, early_stopping=False, random_state=0)
        m.fit(rr.iloc[tr][cols], rr.y.iloc[tr]); p[te] = m.predict_proba(rr.iloc[te][cols])[:, 1]
    rr['p_' + name] = p
    auc_rows.append(dict(engine='qdrant', model=name, auc_deep=roc_auc_score(y[deep], p[deep]), ap_deep=average_precision_score(y[deep], p[deep])))
auc_rows.insert(0, dict(engine='qdrant', model='ANN rank only', auc_deep=roc_auc_score(y[deep], -rr['rank'][deep]), ap_deep=average_precision_score(y[deep], -rr['rank'][deep])))
TOT = rr.nbytes.sum(); NQ2 = rr.q_idx.nunique(); PROMO = int(((rr.y == 1) & deep).sum())
def evaluate(score, B):
    sel = rr.assign(_s=score).sort_values(['q_idx', '_s'], ascending=[True, False]).groupby('q_idx').head(B)
    top = sel.sort_values(['q_idx', 'ce_score'], ascending=[True, False]).groupby('q_idx').head(10)
    return dict(retained=sel.y.sum() / (10 * NQ2), promoted_recovered=((sel.y == 1) & (sel['rank'] >= 10)).sum() / PROMO,
                ndcg10=np.mean([ndcg10(x.chunk_id.tolist(), rel.get(q, set())) for q, x in top.groupby('q_idx')]),
                gold_in_fetched=sel.gold.groupby(sel.q_idx).any().mean(), bytes_frac=sel.nbytes.sum() / TOT)
bud = [dict(engine='qdrant', budget=100, policy='FULL', **evaluate(-rr['rank'].values, 100))]
for B in (10, 20, 30):
    for name, sc in {'TopK(ANN)': -rr['rank'].values, 'M0': rr['p_M0'].values, 'M1-noPos': rr['p_M1-noPos'].values}.items():
        bud.append(dict(engine='qdrant', budget=B, policy=name, **evaluate(sc, B)))
pd.DataFrame([cmp_saved]).to_csv(f'{R}/qdrant_vs_milvus.csv', index=False)
pd.DataFrame(auc_rows).to_csv(f'{R}/qdrant_q2_auc.csv', index=False); pd.DataFrame(bud).to_csv(f'{R}/qdrant_q2_budget.csv', index=False)

pd.set_option('display.width', 200)
print('\n================ Qdrant 교차 검증 결과 ================')
print('[엔진 비교]', {k: round(v, 4) for k, v in cmp_saved.items()}, '| 참고: Milvus recall@100 0.9894')
print(f'[질문 ①] Overlap@10 {q1[:,0].mean():.2f}/10 (Milvus 5.06) | final from ANN rank>=10 {q1[:,1].sum()/(10*len(q1)):.3f} (Milvus 0.494) | '
      f'nDCG@10 ANN {q1[:,2].mean():.4f} -> rerank {q1[:,3].mean():.4f} (Milvus 0.4037 -> 0.4539)')
print('[질문 ② 예측력] 참고 Milvus: rank-only 0.7272/0.1439, M0 0.7463/0.1586, M1-noPos 0.7770/0.2026')
print(pd.DataFrame(auc_rows).round(4).to_string(index=False))
print('[질문 ② 같은 예산] 참고 Milvus B=20: TopK 0.6811, M1-noPos 0.7040 (retained)')
print(pd.DataFrame(bud).round(4).to_string(index=False))
log('ALL DONE')
