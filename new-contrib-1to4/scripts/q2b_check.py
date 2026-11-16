# q2b_check.py: 질문 ② 검증 - 문서 내 위치(데이터셋 구성 편향 가능) 특징 없이도 Top-K보다 나은가
src = open('/workspace/jaeeun/sel_fetch/scripts/q2_predict.py', encoding='utf-8').read()
exec(src.split('gkf = GroupKFold')[0])   # 데이터 로드 + 특징 계산 부분만 재사용
from sklearn.model_selection import GroupKFold

# 1) 편향 확인
qr = pd.read_csv(f'{D}/qrels_chunk.tsv', sep='\t')
gold = set(zip(qr.q_idx, qr.chunk_id))
rr['gold'] = [(q, c) in gold for q, c in zip(rr.q_idx, rr.chunk_id)]
bias = []
for name, m in [('all candidates', np.ones(len(rr), bool)), ('final top-10 (y=1)', (rr.y == 1).values),
                ('gold chunks', rr.gold.values), ('gold & rank>=10', (rr.gold & (rr['rank'] >= 10)).values)]:
    s = rr[m]
    bias.append(dict(group=name, n=len(s), first_chunk_frac=s.f_first_chunk.mean(), chunk_pos_median=s.f_chunk_pos.median(),
                     rel_pos_mean=s.f_rel_pos.mean(), doc_len_median=s.f_doc_len.median(), doc_idx_median=s.doc_idx.median()))
bias = pd.DataFrame(bias); bias.to_csv(f'{R}/q2b_bias.csv', index=False)

# 2) 제거 실험
POS = ['f_chunk_pos', 'f_rel_pos', 'f_first_chunk', 'f_doc_len']
ALL = F0 + FDOC + FEMB + ['f_nbytes']
MODELS = {'M1': ALL, 'M1-noPos': [c for c in ALL if c not in POS], 'M0+POS': F0 + POS}
gkf = GroupKFold(n_splits=5)
def oof(cols):
    p = np.zeros(len(rr))
    for tr, te in gkf.split(rr, rr.y, rr.q_idx):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, early_stopping=False, random_state=0)
        m.fit(rr.iloc[tr][cols], rr.y.iloc[tr]); p[te] = m.predict_proba(rr.iloc[te][cols])[:, 1]
    return p
deep = (rr['rank'] >= 10).values; y = rr.y.values
auc_rows = []
for name, cols in MODELS.items():
    p = oof(cols); rr['p_' + name] = p
    auc_rows.append(dict(model=name, auc_deep=roc_auc_score(y[deep], p[deep]), ap_deep=average_precision_score(y[deep], p[deep])))
    log('model done', name)
auc = pd.DataFrame(auc_rows); auc.to_csv(f'{R}/q2b_auc.csv', index=False)

# 3) 같은 예산 비교
rel = qr.groupby('q_idx').chunk_id.apply(set).to_dict()
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
                ndcg10=nd, gold_in_fetched=sel.gold.groupby(sel.q_idx).any().mean(), bytes_frac=sel.nbytes.sum() / TOT)
bud = [dict(budget=100, policy='FULL (all 100)', **evaluate(-rr['rank'].values, 100))]
for B in (10, 15, 20, 30):
    for name, sc in {'TopK(ANN)': -rr['rank'].values, 'M1': rr['p_M1'].values,
                     'M1-noPos': rr['p_M1-noPos'].values, 'M0+POS': rr['p_M0+POS'].values}.items():
        bud.append(dict(budget=B, policy=name, **evaluate(sc, B)))
bud = pd.DataFrame(bud); bud.to_csv(f'{R}/q2b_budget.csv', index=False)

pd.set_option('display.width', 200)
print('\n================ 질문 ② 검증 결과 ================')
print('[1] 위치 편향 확인 (gold가 all보다 first_chunk_frac가 훨씬 높으면 편향)')
print(bias.round(3).to_string(index=False))
print('\n[2] 예측력 (ANN 10위 밖 후보) - 참고: ANN rank only auc 0.7272 / ap 0.1439, M0 0.7463 / 0.1586')
print(auc.round(4).to_string(index=False))
print('\n[3] 같은 예산 비교 (gold_in_fetched = 정답 chunk를 Fetch한 쿼리 비율)')
print(bud.round(4).to_string(index=False))
print(f'\nsaved: {R}/q2b_bias.csv, q2b_auc.csv, q2b_budget.csv')
log('ALL DONE')
