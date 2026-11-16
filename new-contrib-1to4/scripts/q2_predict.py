# q2_predict.py: ③ 질문 ② - 본문 Fetch 전에 쓸 수 있는 정보만으로 최종 Top-10 후보를 예측할 수 있는가
import time, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
D = '/workspace/jaeeun/sel_fetch/data'; R = '/workspace/jaeeun/sel_fetch/results'; J = '/workspace/junlim/data_nq'
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:6.0f}s]', *a, flush=True)

rr = pd.read_csv(f'{D}/rerank_top100_real.csv').sort_values(['q_idx', 'rank']).reset_index(drop=True)
meta = np.load(f'{D}/chunk_meta.npz'); docarr = meta['doc_idx']; N = len(docarr); doc_len = np.bincount(docarr)
rr['y'] = (rr.ce_rank < 10).astype(int)
g = rr.groupby('q_idx')

# --- M0: ANN 정보만 ---
s1 = g.ann_score.transform('max')
rr['f_rank'] = rr['rank']; rr['f_score'] = rr.ann_score; rr['f_gap_top1'] = s1 - rr.ann_score
rr['f_rel_score'] = rr.ann_score / s1
rr['f_z'] = (rr.ann_score - g.ann_score.transform('mean')) / (g.ann_score.transform('std') + 1e-6)
rr['f_q_s1'] = s1; rr['f_q_gap10'] = s1 - g.ann_score.transform(lambda x: x.iloc[9])

# --- DOC: 문서 구조 ---
rr['is_top10'] = (rr['rank'] < 10).astype(int)
gd = rr.groupby(['q_idx', 'doc_idx'])
rr['f_doc_hits'] = gd['rank'].transform('size')
rr['f_doc_hits_top10'] = gd.is_top10.transform('sum')
rr['f_doc_best_rank'] = gd['rank'].transform('min')
rr['f_doc_score_sum'] = gd.ann_score.transform('sum')
rr['f_doc_rank_within'] = gd['rank'].rank(method='first') - 1
rr['f_chunk_pos'] = rr.chunk_idx
rr['f_doc_len'] = doc_len[rr.doc_idx.values]
rr['f_rel_pos'] = rr.chunk_idx / np.maximum(rr.f_doc_len - 1, 1)
rr['f_first_chunk'] = (rr.chunk_idx == 0).astype(int)
key = rr.q_idx.values.astype(np.int64) * 10**8 + rr.chunk_id.values
rank_of = pd.Series(rr['rank'].values, index=key)
def nb_rank(off):
    nid = rr.chunk_id.values + off; ok = (nid >= 0) & (nid < N)
    same = np.zeros(len(rr), bool); same[ok] = docarr[nid[ok]] == rr.doc_idx.values[ok]
    r = rank_of.reindex(rr.q_idx.values.astype(np.int64) * 10**8 + nid).values.astype(float)
    return np.where(same, r, np.nan)
nbmin = np.fmin(nb_rank(-1), nb_rank(1))
rr['f_nb_in_cand'] = (~np.isnan(nbmin)).astype(int)
rr['f_nb_best_rank'] = np.where(np.isnan(nbmin), 200, nbmin)
log('doc features done')

# --- EMB: 후보 간 관계 (인덱스에 이미 있는 벡터만 사용) ---
ids = np.unique(rr.chunk_id.values)
V = np.asarray(np.load(f'{J}/chunk_vecs.npy', mmap_mode='r')[ids], dtype=np.float32)
pos = np.searchsorted(ids, rr.chunk_id.values)
cen10 = np.zeros(len(rr), np.float32); maxhi = np.zeros(len(rr), np.float32)
sim1 = np.zeros(len(rr), np.float32); meanall = np.zeros(len(rr), np.float32)
low = None
for q, idx in g.indices.items():
    X = V[pos[idx]]; S = X @ X.T; n = len(idx)
    if low is None or low.shape[0] != n: low = np.tri(n, k=-1, dtype=bool)
    np.fill_diagonal(S, np.nan)
    cen10[idx] = np.nanmean(S[:, :10], 1)
    mh = np.where(low, S, -np.inf).max(1); mh[0] = 0; maxhi[idx] = mh
    sim1[idx] = np.nan_to_num(S[:, 0], nan=1.0)
    meanall[idx] = np.nanmean(S, 1)
rr['f_cen10'] = cen10; rr['f_maxsim_higher'] = maxhi; rr['f_sim_top1'] = sim1; rr['f_meansim'] = meanall
rr['f_nbytes'] = rr.nbytes
log('embedding features done')

F0 = ['f_rank', 'f_score', 'f_gap_top1', 'f_rel_score', 'f_z', 'f_q_s1', 'f_q_gap10']
FDOC = ['f_doc_hits', 'f_doc_hits_top10', 'f_doc_best_rank', 'f_doc_score_sum', 'f_doc_rank_within', 'f_chunk_pos',
        'f_doc_len', 'f_rel_pos', 'f_first_chunk', 'f_nb_in_cand', 'f_nb_best_rank']
FEMB = ['f_cen10', 'f_maxsim_higher', 'f_sim_top1', 'f_meansim']
MODELS = {'M0': F0, 'M0+DOC': F0 + FDOC, 'M0+EMB': F0 + FEMB, 'M1': F0 + FDOC + FEMB + ['f_nbytes']}

gkf = GroupKFold(n_splits=5)
def oof(cols):
    p = np.zeros(len(rr))
    for tr, te in gkf.split(rr, rr.y, rr.q_idx):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, early_stopping=False, random_state=0)
        m.fit(rr.iloc[tr][cols], rr.y.iloc[tr]); p[te] = m.predict_proba(rr.iloc[te][cols])[:, 1]
    return p

deep = (rr['rank'] >= 10).values; y = rr.y.values
auc_rows = [dict(model='ANN rank only', auc_all=roc_auc_score(y, -rr['rank']), ap_all=average_precision_score(y, -rr['rank']),
                 auc_deep=roc_auc_score(y[deep], -rr['rank'][deep]), ap_deep=average_precision_score(y[deep], -rr['rank'][deep]))]
for name, cols in MODELS.items():
    p = oof(cols); rr['p_' + name] = p
    auc_rows.append(dict(model=name, auc_all=roc_auc_score(y, p), ap_all=average_precision_score(y, p),
                         auc_deep=roc_auc_score(y[deep], p[deep]), ap_deep=average_precision_score(y[deep], p[deep])))
    log('model done', name)
auc = pd.DataFrame(auc_rows); auc.to_csv(f'{R}/q2_auc.csv', index=False)

# --- 같은 예산 비교 ---
rel = pd.read_csv(f'{D}/qrels_chunk.tsv', sep='\t').groupby('q_idx').chunk_id.apply(set).to_dict()
def ndcg10(ids_, r):
    dcg = sum(1 / np.log2(i + 2) for i, c in enumerate(ids_[:10]) if c in r)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(r), 10)))
    return dcg / idcg if idcg else 0.0
TOT = rr.nbytes.sum(); NQ = rr.q_idx.nunique(); PROMO = int(((rr.y == 1) & deep).sum())
def evaluate(score, B):
    sel = rr.assign(_s=score).sort_values(['q_idx', '_s'], ascending=[True, False]).groupby('q_idx').head(B)
    top = sel.sort_values(['q_idx', 'ce_score'], ascending=[True, False]).groupby('q_idx').head(10)
    nd = np.mean([ndcg10(x.chunk_id.tolist(), rel.get(q, set())) for q, x in top.groupby('q_idx')])
    return dict(retained=sel.y.sum() / (10 * NQ), promoted_recovered=((sel.y == 1) & (sel['rank'] >= 10)).sum() / PROMO,
                ndcg10=nd, bytes_frac=sel.nbytes.sum() / TOT)
is10 = (rr['rank'] < 10).values * 10.0
POL = {'TopK(ANN)': -rr['rank'].values, 'M0': rr.p_M0.values, 'M1': rr.p_M1.values,
       'Hyb-M0 (ANN10+M0)': is10 + rr.p_M0.values, 'Hyb-M1 (ANN10+M1)': is10 + rr.p_M1.values}
bud = []
for B in (10, 15, 20, 30):
    for name, sc in POL.items():
        if B == 10 and name.startswith('Hyb'): continue
        bud.append(dict(budget=B, policy=name, **evaluate(sc, B)))
bud = pd.DataFrame(bud); bud.to_csv(f'{R}/q2_budget.csv', index=False)
rr[['q_idx', 'rank', 'chunk_id', 'doc_idx', 'nbytes', 'y'] + [c for c in rr.columns if c.startswith('p_')]].to_csv(f'{R}/q2_oof_pred.csv', index=False)

pd.set_option('display.width', 200)
print('\n================ 질문 ② 결과 ================')
print(f'candidates {len(rr)}, positives {int(y.sum())}, deep(rank>=10) positives {PROMO}')
print('\n[예측력] auc_all/ap_all = 전체 후보, auc_deep/ap_deep = ANN 10위 밖 후보만')
print(auc.round(4).to_string(index=False))
print('\n[같은 예산 비교] retained=최종 Top-10 보존율, promoted_recovered=10위 밖 승격 후보 회수율')
print(bud.round(4).to_string(index=False))
print(f'\nsaved: {R}/q2_auc.csv, q2_budget.csv, q2_oof_pred.csv')
log('ALL DONE')
