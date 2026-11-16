# q1_rerank.py: ③ 질문 ① - ANN Top-100을 cross-encoder로 리랭킹하고 ANN 순위와의 차이를 측정
import json, time, os, numpy as np, pandas as pd
D = '/workspace/jaeeun/sel_fetch/data'; R = '/workspace/jaeeun/sel_fetch/results'; J = '/workspace/junlim/data_nq'
MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:6.0f}s]', *a, flush=True)

ann = pd.read_csv(f'{D}/ann_top100_real.csv')
NQ = ann.q_idx.nunique()
out_p = f'{D}/rerank_top100_real.csv'

# 1) 리랭킹 (결과가 이미 있으면 건너뜀)
if not os.path.exists(out_p):
    import torch
    from sentence_transformers import CrossEncoder
    torch.set_num_threads(4)
    need = set(ann.chunk_id.unique()); text = {}
    with open(f'{J}/chunks.jsonl', encoding='utf-8') as f:
        for cid, l in enumerate(f):
            if cid in need: text[cid] = json.loads(l)['text']
    qs = [json.loads(l)['text'] for l in open(f'{J}/queries.jsonl', encoding='utf-8')][:NQ]
    log('candidate texts loaded', len(text))
    try:
        ce = CrossEncoder(MODEL, device='cuda', max_length=256)
    except TypeError:
        ce = CrossEncoder(MODEL, device='cuda')
    rows, t1 = [], time.time()
    for i, (q, sub) in enumerate(ann.sort_values(['q_idx', 'rank']).groupby('q_idx')):
        s = ce.predict([(qs[q], text[c]) for c in sub.chunk_id], batch_size=100, show_progress_bar=False)
        sub = sub.assign(ce_score=np.asarray(s, dtype=np.float32))
        sub['ce_rank'] = (-sub.ce_score).rank(method='first').astype(int) - 1
        rows.append(sub)
        if (i + 1) % 500 == 0: log('reranked', i + 1, '/', NQ, f'({(i+1)/(time.time()-t1):.1f} q/s)')
    rr = pd.concat(rows); rr.to_csv(out_p, index=False); log('saved', out_p)
else:
    rr = pd.read_csv(out_p); log('loaded existing', out_p)

# 2) 질문 ① 지표
from scipy.stats import kendalltau
qr = pd.read_csv(f'{D}/qrels_chunk.tsv', sep='\t')
rel = qr.groupby('q_idx').chunk_id.apply(set).to_dict()
def ndcg10(ids, r):
    dcg = sum(1 / np.log2(i + 2) for i, c in enumerate(ids[:10]) if c in r)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(r), 10)))
    return dcg / idcg if idcg else 0.0

pq = []
for q, sub in rr.groupby('q_idx'):
    a10 = sub.sort_values('rank').chunk_id.head(10).tolist()
    c10 = sub.sort_values('ce_rank').chunk_id.head(10).tolist()
    fin = sub[sub.ce_rank < 10]; r = rel.get(q, set())
    pq.append(dict(q_idx=q, overlap10=len(set(a10) & set(c10)),
                   final_from_gt10=int((fin['rank'] >= 10).sum()), final_from_gt20=int((fin['rank'] >= 20).sum()),
                   final_from_gt50=int((fin['rank'] >= 50).sum()), max_ann_rank_in_final=int(fin['rank'].max()),
                   kendall_tau=float(kendalltau(sub['rank'], sub.ce_rank)[0]),
                   ndcg10_ann=ndcg10(a10, r), ndcg10_ce=ndcg10(c10, r),
                   hit10_ann=int(len(set(a10) & r) > 0), hit10_ce=int(len(set(c10) & r) > 0)))
pq = pd.DataFrame(pq); pq.to_csv(f'{R}/q1_per_query.csv', index=False)

fin_all = rr[rr.ce_rank < 10]
bins = pd.cut(fin_all['rank'], [-1, 9, 19, 29, 49, 99], labels=['0-9', '10-19', '20-29', '30-49', '50-99'])
hist = bins.value_counts(normalize=True).sort_index(); hist.to_csv(f'{R}/q1_final_ann_rank_hist.csv')

kc = []
tot_bytes = rr.nbytes.sum()
for k in (10, 15, 20, 30, 50, 100):
    keep = rr[rr['rank'] < k]
    top = keep.sort_values(['q_idx', 'ce_score'], ascending=[True, False]).groupby('q_idx').head(10)
    ret = top.merge(fin_all[['q_idx', 'chunk_id']], on=['q_idx', 'chunk_id']).groupby('q_idx').size().reindex(range(NQ), fill_value=0) / 10
    nd = np.mean([ndcg10(g.chunk_id.tolist(), rel.get(q, set())) for q, g in top.groupby('q_idx')])
    kc.append(dict(k_fetch=k, final_top10_retained=float(ret.mean()), ndcg10=float(nd), fetch_bytes_frac=float(keep.nbytes.sum() / tot_bytes)))
kc = pd.DataFrame(kc); kc.to_csv(f'{R}/q1_kcurve.csv', index=False)

d = pq.ndcg10_ce - pq.ndcg10_ann
print('\n================ 질문 ① 결과 ================')
print(f'queries: {len(pq)}')
print(f'Overlap@10 (ANN top10 vs rerank top10): mean {pq.overlap10.mean():.2f} / 10, median {pq.overlap10.median():.0f}')
print('Overlap@10 distribution:', pq.overlap10.value_counts().sort_index().to_dict())
print(f'final top-10 items from ANN rank >=10: {pq.final_from_gt10.sum()/(10*len(pq)):.3f}, >=20: {pq.final_from_gt20.sum()/(10*len(pq)):.3f}, >=50: {pq.final_from_gt50.sum()/(10*len(pq)):.3f}')
print(f'queries with >=1 final item from ANN rank >=20: {(pq.final_from_gt20 > 0).mean():.3f}')
print(f'Kendall tau (ANN rank vs rerank rank, top-100): mean {pq.kendall_tau.mean():.3f}')
print('final top-10 ANN-rank histogram:', {k: round(v, 3) for k, v in hist.items()})
print(f'nDCG@10  ANN {pq.ndcg10_ann.mean():.4f}  ->  rerank {pq.ndcg10_ce.mean():.4f}  (diff {d.mean():+.4f}; better {(d>0).mean():.3f}, worse {(d<0).mean():.3f})')
print(f'Hit@10   ANN {pq.hit10_ann.mean():.4f}  ->  rerank {pq.hit10_ce.mean():.4f}')
print('\nk-curve (ANN 상위 k개만 Fetch 후 리랭킹 = Top-K 축소 기준선):')
print(kc.to_string(index=False))
print(f'\nsaved: {R}/q1_per_query.csv, q1_final_ann_rank_hist.csv, q1_kcurve.csv')
log('ALL DONE')
