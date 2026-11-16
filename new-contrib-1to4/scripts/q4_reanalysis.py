# q4_reanalysis.py: ④ 결과 재분석 (기존 파일만, 읽기 전용; Milvus/GPU/새 모델 없음)
#  분석 1: 최종 Context ±1 확장 (같은 문서 안 chunk_id±1), 확장 없음과 나란히
#  분석 2: 문서 순 10분위 d0~d1 제외 (예측 순위·정답 모두 d2~d9로 제한) + 10분위별 상위 5% 포함률
#  paired bootstrap: k-occ10 - 첫 chunk (포함률, 회복률)
import os, json, time
import numpy as np, pandas as pd

BASE = '/workspace/jaeeun/sel_fetch'
IN = f'{BASE}/results/q4_full'
OUT = f'{BASE}/results/q4_reanalysis'
os.makedirs(OUT, exist_ok=True)
PCTS = (0.01, 0.05, 0.10)
BASES = ('chunk', 'byte')
NBOOT = 1000
t0 = time.time()
def log(*x): print(f'[{time.time()-t0:6.1f}s]', *x, flush=True)

# ---------------------------------------------------------------- 입력
meta = np.load(f'{BASE}/data/chunk_meta.npz')
doc, cix, nb = meta['doc_idx'], meta['chunk_idx'], meta['nbytes'].astype(np.int64)
N = len(doc)
cs = pd.read_csv(f'{IN}/chunk_scores.csv')
assert len(cs) == N and (cs.chunk_id.values == np.arange(N)).all()
assert (cs.doc_idx.values == doc).all() and (cs.chunk_idx.values == cix).all() and (cs.nbytes.values == nb).all()
kocc10, kocc25, log_cnt = cs.kocc10.values.astype(float), cs.kocc25.values.astype(float), cs.log_cnt.values.astype(float)

# 쿼리별 접근 목록
rr = pd.read_csv(f'{IN}/rerank_sample.csv')
split = pd.read_csv(f'{IN}/train_split.csv')
eval_ids = set(split[split.split == 'eval']._id)
assert set(rr._id) <= eval_ids, 'rerank sample contains non-eval queries'
tr_fin = rr[rr.ce_rank < 10].sort_values(['q_row', 'ce_rank']).chunk_id.values.reshape(-1, 10)
real = pd.read_csv(f'{BASE}/data/rerank_top100_real.csv', usecols=['q_idx', 'rank', 'chunk_id', 'ce_rank'])
re_fin = real[real.ce_rank < 10].sort_values(['q_idx', 'ce_rank']).chunk_id.values.reshape(-1, 10)
E10 = np.load(f'{IN}/train_search_cache.npz')['E10']   # train 평가 ANN Top-10 (주 정답의 쿼리별 목록)

def counts(H): return np.bincount(H[H >= 0], minlength=N)
# 기존 결과와 일치 확인
assert (counts(tr_fin) == cs.cnt_train_rr_final10.values).all()
assert (counts(re_fin) == cs.cnt_real_final10.values).all()
assert (counts(E10) == cs.cnt_train_eval_ann10.values).all()
log(f'inputs ok | train sample {len(tr_fin)} | real {len(re_fin)} | train eval {len(E10)}')

def expand(H):
    out = np.full((len(H), 30), -1, np.int64)
    for i, row in enumerate(H):
        s = set()
        for c in row:
            if c < 0: continue
            s.add(c)
            if c - 1 >= 0 and doc[c - 1] == doc[c]: s.add(c - 1)
            if c + 1 < N and doc[c + 1] == doc[c]: s.add(c + 1)
        s = sorted(s); out[i, :len(s)] = s
    return out

TARGETS = {
    'train_eval_ann10': E10,
    'train_rr_final10': tr_fin, 'train_rr_final10_pm1': expand(tr_fin),
    'real_final10': re_fin, 'real_final10_pm1': expand(re_fin),
}

# ---------------------------------------------------------------- 예측자 (기존 스크립트와 같은 seed)
PRED = {'kocc10': kocc10, 'kocc25': kocc25, 'log': log_cnt, 'first_chunk': (cix == 0).astype(float)}
TIE = {'kocc10': 100, 'kocc25': 101, 'log': 102, 'first_chunk': 103}
for s in range(10):
    PRED[f'random_s{s}'] = np.random.default_rng(1000 + s).random(N); TIE[f'random_s{s}'] = 105 + s
ORDER = {n: np.lexsort((np.random.default_rng(TIE[n]).random(N), -sc)) for n, sc in PRED.items()}
REPORT_P = ['kocc10', 'kocc25', 'first_chunk', 'log']

def sets_in(allowed, pcts=PCTS):
    """allowed(bool N) 범위 안에서 예측 상위 p% (chunk 수 / 바이트) 집합"""
    na, ba = int(allowed.sum()), nb[allowed].sum(); out = {}
    for n, o in ORDER.items():
        oa = o[allowed[o]]; cb = np.cumsum(nb[oa])
        for p in pcts:
            for b in BASES:
                k = int(round(p * na)) if b == 'chunk' else int(np.searchsorted(cb, p * ba, side='right'))
                m = np.zeros(N, bool); m[oa[:k]] = True; out[(n, p, b)] = m
    return out

def per_query(H, allowed, SETS, keys):
    Hc = np.where(H < 0, 0, H); V = (H >= 0) & allowed[Hc]; nbh = nb[Hc] * V
    M = np.stack([(SETS[k][Hc] * (V if k[2] == 'chunk' else nbh)).sum(1) for k in keys], 1).astype(np.float64)
    return M, V.sum(1).astype(np.float64), nbh.sum(1).astype(np.float64)

def evaluate(scope, allowed, pcts, targets, boot=True, seed=7):
    SETS = sets_in(allowed, pcts)
    keys = [(n, p, b) for n in PRED for p in pcts for b in BASES]; ci = {k: i for i, k in enumerate(keys)}
    rows_cov, rows_rec, rows_pair = [], [], []
    for t in targets:
        H = TARGETS[t]
        M, tc, tb = per_query(H, allowed, SETS, keys)
        Q = len(H)
        def cov_of(W):
            den = np.stack([(W @ tc) if b == 'chunk' else (W @ tb) for _, _, b in keys], 1)
            return (W @ M) / den
        def derive(cov):
            res = {}
            for p in pcts:
                for b in BASES:
                    rnd = np.mean([cov[:, ci[(f'random_s{s}', p, b)]] for s in range(10)], 0)
                    lg = cov[:, ci[('log', p, b)]]
                    for n in REPORT_P:
                        cv = cov[:, ci[(n, p, b)]]
                        res[(n, p, b)] = (cv, (cv - rnd) / (lg - rnd))
                    res[('random_mean10', p, b)] = (rnd, np.full_like(rnd, np.nan))
            return res
        pt = derive(cov_of(np.ones((1, Q))))
        for (n, p, b), (cv, rc) in pt.items():
            rows_cov.append(dict(scope=scope, target=t, predictor=n, top_pct=p, budget_basis=b, n_queries=Q,
                                 coverage=float(cv[0]), recovery=float(rc[0])))
        if not boot: continue
        rng = np.random.default_rng(seed)
        bc = np.concatenate([cov_of(np.stack([np.bincount(rng.integers(0, Q, Q), minlength=Q) for _ in range(min(100, NBOOT - st))]).astype(np.float64))
                             for st in range(0, NBOOT, 100)], 0)
        bd = derive(bc)
        for p in pcts:
            for b in BASES:
                for n in REPORT_P:
                    cv, rc = bd[(n, p, b)]
                    rows_rec.append(dict(scope=scope, target=t, predictor=n, top_pct=p, budget_basis=b, n_queries=Q,
                                         recovery=float(pt[(n, p, b)][1][0]), rec_ci_lo=float(np.nanpercentile(rc, 2.5)),
                                         rec_ci_hi=float(np.nanpercentile(rc, 97.5)), n_nan=int(np.isnan(rc).sum())))
                (ck, rk), (cf, rf) = bd[('kocc10', p, b)], bd[('first_chunk', p, b)]
                dc, dr = ck - cf, rk - rf
                rows_pair.append(dict(scope=scope, target=t, top_pct=p, budget_basis=b, n_queries=Q, n_boot=NBOOT,
                                      cov_diff_point=float(pt[('kocc10', p, b)][0][0] - pt[('first_chunk', p, b)][0][0]),
                                      cov_diff_mean=float(dc.mean()), cov_diff_lo=float(np.percentile(dc, 2.5)), cov_diff_hi=float(np.percentile(dc, 97.5)),
                                      rec_diff_point=float(pt[('kocc10', p, b)][1][0] - pt[('first_chunk', p, b)][1][0]),
                                      rec_diff_mean=float(np.nanmean(dr)), rec_diff_lo=float(np.nanpercentile(dr, 2.5)),
                                      rec_diff_hi=float(np.nanpercentile(dr, 97.5)), n_nan=int(np.isnan(dr).sum())))
    return rows_cov, rows_rec, rows_pair

# ---------------------------------------------------------------- 1) 정답 분포
dist = []
dec = np.minimum(np.arange(N) * 10 // N, 9)
for scope, allowed in [('all', np.ones(N, bool)), ('excl_d0d1', dec >= 2)]:
    for t, H in TARGETS.items():
        Hc = np.where(H < 0, 0, H); V = (H >= 0) & allowed[Hc]
        cnt = np.bincount(Hc[V], minlength=N); acc = cnt > 0
        dist.append(dict(scope=scope, target=t, n_queries=len(H), mean_chunks_per_query=float(V.sum(1).mean()),
                         total_access=int(cnt.sum()), n_accessed=int(acc.sum()), frac_accessed_of_scope=float(acc.sum() / allowed.sum()),
                         frac_ge2_of_accessed=float((cnt >= 2).sum() / max(acc.sum(), 1)), frac_ge2_of_scope=float((cnt >= 2).sum() / allowed.sum()),
                         max=int(cnt.max())))
dist = pd.DataFrame(dist); dist.to_csv(f'{OUT}/target_distribution.csv', index=False)
log('distribution done')

# ---------------------------------------------------------------- 분석 1 (전체 범위), 분석 2 (d0~d1 제외)
A1T = ['train_rr_final10', 'train_rr_final10_pm1', 'real_final10', 'real_final10_pm1']
c1, r1, p1 = evaluate('all', np.ones(N, bool), PCTS, A1T)
log('analysis 1 done')
c2, r2, p2 = evaluate('excl_d0d1', dec >= 2, PCTS, ['train_eval_ann10'] + A1T)
log('analysis 2 done')
cov = pd.DataFrame(c1 + c2); cov.to_csv(f'{OUT}/coverage_recovery.csv', index=False)
rec = pd.DataFrame(r1 + r2); rec.to_csv(f'{OUT}/recovery_ci.csv', index=False)
pair = pd.DataFrame(p1 + p2); pair.to_csv(f'{OUT}/paired_bootstrap_kocc10_minus_first.csv', index=False)

# 10분위별 상위 5% 포함률 (분위 안으로 예측·정답 제한)
drows = []
for d in range(10):
    cd, _, _ = evaluate(f'decile_{d}', dec == d, (0.05,), ['train_eval_ann10'] + A1T, boot=False)
    drows += [r for r in cd if r['predictor'] in ('kocc10', 'first_chunk')]
decdf = pd.DataFrame(drows); decdf.to_csv(f'{OUT}/decile_top5_coverage.csv', index=False)
log('decile table done')

json.dump(dict(inputs=[f'{IN}/chunk_scores.csv', f'{IN}/rerank_sample.csv', f'{IN}/train_split.csv', f'{IN}/train_search_cache.npz (E10 only)',
                       f'{BASE}/data/rerank_top100_real.csv', f'{BASE}/data/chunk_meta.npz'],
               n_boot=NBOOT, boot_seed=7, tie_seeds=TIE, random_score_seeds='1000+s',
               expansion='per query: final top-10 + chunk_id±1 within same doc_idx, deduplicated',
               excl_d0d1='chunk_id decile (chunk_id*10//N) >= 2; ranking and targets restricted to this range; budgets are % of in-range chunks/bytes',
               decile_table='within each decile; top 5% of that decile',
               coverage='chunk basis: fraction of accesses; byte basis: fraction of accessed bytes',
               recovery='(pred - random_mean10) / (log - random_mean10), log = existing log-half ANN top-10 counts (unchanged)'),
          open(f'{OUT}/config.json', 'w'), indent=1)

pd.set_option('display.width', 250); pd.set_option('display.max_rows', 1000)
print('\n==== target distribution ====\n' + dist.round(4).to_string(index=False))
for scope in ('all', 'excl_d0d1'):
    s = cov[cov.scope == scope]
    for col in ('coverage', 'recovery'):
        print(f'\n==== {scope} : {col} ====')
        print(s.pivot_table(index=['target', 'predictor'], columns=['budget_basis', 'top_pct'], values=col).round(4).to_string())
print('\n==== recovery CI ====\n' + rec.round(4).to_string(index=False))
print('\n==== decile top5% coverage ====')
print(decdf.pivot_table(index=['target', 'predictor', 'budget_basis'], columns='scope', values='coverage').round(4).to_string())
print('\n==== paired bootstrap (kocc10 - first_chunk) ====\n' + pair.round(4).to_string(index=False))
log('ALL DONE')
