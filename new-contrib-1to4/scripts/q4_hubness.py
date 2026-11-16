# q4_hubness.py: ④ Index-guided Data Placement 가능성 검증
# 쿼리 로그 없이 벡터 인덱스 구조(k-occurrence, hubness)만으로 자주 접근될 chunk를 예측할 수 있는지 측정한다.
# 배치 변경/I/O 측정은 하지 않는다.
import os, sys, json, time, argparse, resource, hashlib, urllib.request
import numpy as np, pandas as pd

J = '/workspace/junlim/data_nq'
BASE = '/workspace/jaeeun/sel_fetch'
D = f'{BASE}/data'
DB, COL = 'jaeeun', 'nq1m_sel'
CE_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
PCTS = (0.01, 0.05, 0.10)
BASES = ('chunk', 'byte')

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--kocc-n', type=int, default=0, help='k-occurrence 쿼리 chunk 수 (0이면 --kocc-frac 사용)')
ap.add_argument('--kocc-frac', type=float, default=1.0, help='1.0=전체 100만, 0.2=20%% 표본')
ap.add_argument('--n-train', type=int, default=0, help='사용할 train 쿼리 수 (0=전체)')
ap.add_argument('--n-rerank', type=int, default=2000)
ap.add_argument('--n-recall', type=int, default=200)
ap.add_argument('--ef-kocc', type=int, default=64)
ap.add_argument('--ef-top10', type=int, default=64)
ap.add_argument('--ef-top100', type=int, default=256)
ap.add_argument('--batch', type=int, default=1000)
ap.add_argument('--ckpt-every', type=int, default=50, help='k-occurrence 체크포인트 간격 (배치 수)')
ap.add_argument('--n-boot', type=int, default=1000)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--mem-limit-gb', type=float, default=6.0)
ap.add_argument('--fresh', action='store_true', help='기존 캐시/체크포인트를 무시하고 처음부터 (파일은 덮어씀)')
a = ap.parse_args()
OUT = a.out
os.makedirs(OUT, exist_ok=True)
K1, K2 = 10, 25

t0 = time.time()
def log(*x): print(f'[{time.time()-t0:7.0f}s]', *x, flush=True)
def maxrss_gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
def check_mem(where):
    m = maxrss_gb()
    if m > a.mem_limit_gb:
        raise SystemExit(f'STOP: max RSS {m:.2f} GB > {a.mem_limit_gb} GB at {where}')
def milvus_rss_gb():
    try:
        txt = urllib.request.urlopen('http://localhost:9091/metrics', timeout=3).read().decode()
        for l in txt.splitlines():
            if l.startswith('process_resident_memory_bytes'):
                return float(l.split()[-1]) / 2**30
    except Exception:
        return None
    return None

TIME, RES = {}, {'milvus_rss_gb': {}}
class stage:
    def __init__(s, name): s.name = name
    def __enter__(s): s.t = time.time(); log(f'== {s.name} start'); return s
    def __exit__(s, *e):
        TIME[s.name] = round(time.time() - s.t, 1); log(f'== {s.name} done {TIME[s.name]}s | maxRSS {maxrss_gb():.2f} GB'); check_mem(s.name)

# ---------------------------------------------------------------- 0) 준비
with stage('prep'):
    meta = np.load(f'{D}/chunk_meta.npz')
    doc, cix, nb = meta['doc_idx'], meta['chunk_idx'], meta['nbytes'].astype(np.int64)
    N = len(doc)
    vecs = np.load(f'{J}/chunk_vecs.npy', mmap_mode='r')
    assert vecs.shape == (N, 384)
    qmeta = [json.loads(l) for l in open(f'{D}/q_train_minilm_meta.jsonl', encoding='utf-8')]
    QV = np.load(f'{D}/q_train_minilm.npy', mmap_mode='r')
    NQd = len(qmeta); assert QV.shape[0] == NQd
    # train 쿼리 분할: (스모크면 고정 seed로 n_train개 추출) -> 고정 seed 순열로 반반
    pool = np.arange(NQd)
    if a.n_train and a.n_train < NQd:
        pool = np.sort(np.random.default_rng(a.seed + 11).choice(NQd, a.n_train, replace=False))
    perm = np.random.default_rng(a.seed).permutation(pool)
    h = len(perm) // 2
    log_q, eval_q = np.sort(perm[:h]), np.sort(perm[h:])
    rr_q = np.sort(np.random.default_rng(a.seed + 2).choice(eval_q, min(a.n_rerank, len(eval_q)), replace=False))
    # k-occurrence 쿼리 chunk
    if a.kocc_n:
        kq = np.sort(np.random.default_rng(a.seed + 3).choice(N, a.kocc_n, replace=False)); kmode = f'sample_{a.kocc_n}'
    elif a.kocc_frac < 1.0:
        kq = np.sort(np.random.default_rng(a.seed + 3).choice(N, int(round(N * a.kocc_frac)), replace=False)); kmode = f'sample_frac_{a.kocc_frac}'
    else:
        kq = np.arange(N); kmode = 'full'
    rs = np.sort(np.random.default_rng(a.seed + 4).choice(kq, min(a.n_recall, len(kq)), replace=False))
    log(f'N chunks {N} | train dedup {NQd} | used {len(pool)} -> log {len(log_q)} / eval {len(eval_q)} | rerank sample {len(rr_q)}')
    log(f'k-occ mode {kmode}: {len(kq)} query chunks | recall sample {len(rs)}')

def search(c, X, limit, ef):
    res = c.search(COL, data=X.tolist(), limit=limit, search_params={'metric_type': 'IP', 'params': {'ef': ef}})
    out = np.full((len(X), limit), -1, np.int64)
    for i, hits in enumerate(res):
        ids = [int(hh['pk']) for hh in hits]
        out[i, :len(ids)] = ids
    return out

def search_many(c, idx, src, limit, ef):
    out = np.full((len(idx), limit), -1, np.int64)
    for st in range(0, len(idx), a.batch):
        out[st:st + a.batch] = search(c, np.asarray(src[idx[st:st + a.batch]], dtype=np.float32), limit, ef)
    return out

# ---------------------------------------------------------------- 1) Milvus 검색 (load -> ... -> release)
from pymilvus import MilvusClient
c = MilvusClient(uri='http://localhost:19530', db_name=DB)
rows = int(c.get_collection_stats(COL).get('row_count', 0))
log('collection rows', rows)
if rows != N:
    c.close(); raise SystemExit(f'STOP: {COL} rows {rows} != {N}')
RES['milvus_rss_gb']['before_load'] = milvus_rss_gb()
tr_cache = f'{OUT}/train_search_cache.npz'
ck = f'{OUT}/kocc_ckpt.npz'
sig = hashlib.md5(kq.tobytes() + f'|{a.ef_kocc}|{K2}'.encode()).hexdigest()
loaded = False
try:
    with stage('milvus_load'):
        c.load_collection(COL); loaded = True
        log('load state', c.get_load_state(COL).get('state'))
        RES['milvus_rss_gb']['after_load'] = milvus_rss_gb()

    # 1-1) train 쿼리 ANN Top-10 (로그/평가) + 리랭킹 표본 Top-100
    tsig = hashlib.md5(log_q.tobytes() + eval_q.tobytes() + rr_q.tobytes() + f'|{a.ef_top10}|{a.ef_top100}'.encode()).hexdigest()
    cached = None
    if os.path.exists(tr_cache) and not a.fresh:
        z = np.load(tr_cache)
        if str(z['sig']) == tsig: cached = z
    if cached is not None:
        L10, E10, R100 = cached['L10'], cached['E10'], cached['R100']; log('train search: loaded cache')
    else:
        with stage('train_search_top10'):
            _t = time.time()
            L10 = search_many(c, log_q, QV, K1, a.ef_top10)
            E10 = search_many(c, eval_q, QV, K1, a.ef_top10)
            RES['train_top10_qps'] = round((len(log_q) + len(eval_q)) / max(time.time() - _t, 1e-6), 1)
        with stage('rerank_sample_top100'):
            _t = time.time()
            R100 = search_many(c, rr_q, QV, 100, a.ef_top100)
            RES['top100_qps'] = round(len(rr_q) / max(time.time() - _t, 1e-6), 1)
        np.savez(tr_cache.replace('.npz', '.tmp.npz'), L10=L10, E10=E10, R100=R100, sig=tsig)
        os.replace(tr_cache.replace('.npz', '.tmp.npz'), tr_cache)
    RES['train_missing_hits'] = int((L10 < 0).sum() + (E10 < 0).sum() + (R100 < 0).sum())

    # 1-2) k-occurrence (체크포인트에서 이어서)
    kocc10 = np.zeros(N, np.int64); kocc25 = np.zeros(N, np.int64); done = 0; self_missing = 0; short_rows = 0
    if os.path.exists(ck) and not a.fresh:
        z = np.load(ck)
        if str(z['sig']) == sig:
            kocc10, kocc25 = z['kocc10'].copy(), z['kocc25'].copy()
            done, self_missing, short_rows = int(z['done']), int(z['self_missing']), int(z['short_rows'])
            log(f'k-occ resume from {done}/{len(kq)}')
    with stage('kocc_search'):
        ts, d0, nb_ = time.time(), done, 0
        for st in range(done, len(kq), a.batch):
            ids = kq[st:st + a.batch]
            X = np.asarray(vecs[ids[0]:ids[-1] + 1], dtype=np.float32) if kmode == 'full' else np.asarray(vecs[ids], dtype=np.float32)
            nn = search(c, X, K2 + 1, a.ef_kocc)
            is_self = nn == ids[:, None]
            valid = (~is_self) & (nn >= 0)          # 자기 자신은 같은 ID만 제외
            cum = np.cumsum(valid, 1)
            kocc10 += np.bincount(nn[valid & (cum <= K1)], minlength=N)
            kocc25 += np.bincount(nn[valid & (cum <= K2)], minlength=N)
            self_missing += int((~is_self.any(1)).sum())
            short_rows += int((valid.sum(1) < K2).sum())
            done = st + len(ids); nb_ += 1
            if nb_ % a.ckpt_every == 0 or done == len(kq):
                np.savez(ck.replace('.npz', '.tmp.npz'), kocc10=kocc10, kocc25=kocc25, done=done,
                         self_missing=self_missing, short_rows=short_rows, sig=sig)
                os.replace(ck.replace('.npz', '.tmp.npz'), ck)
                rate = (done - d0) / max(time.time() - ts, 1e-6)
                log(f'k-occ {done}/{len(kq)} ({rate:.0f} q/s, left {(len(kq)-done)/max(rate,1e-6)/60:.1f} min) | maxRSS {maxrss_gb():.2f} GB')
                check_mem('kocc')
        RES['kocc_qps'] = round((done - d0) / max(time.time() - ts, 1e-6), 1) if done > d0 else None
    RES['milvus_rss_gb']['after_kocc'] = milvus_rss_gb()

    # 1-3) recall 표본 ANN
    nn_rs = search_many(c, rs, vecs, K2 + 1, a.ef_kocc)
finally:
    if loaded:
        c.release_collection(COL)
    st_ = c.get_load_state(COL).get('state')
    RES['release_state'] = str(st_)
    log('released', COL, '| state', st_)
    c.close()
RES['milvus_rss_gb']['after_release'] = milvus_rss_gb()

# ---------------------------------------------------------------- 2) 정확 검색 recall (전수 내적, mmap 배치)
with stage('exact_recall'):
    Xs = np.asarray(vecs[rs], dtype=np.float32); KK = K2 + 2
    best = np.full((len(rs), KK), -1e9, np.float32); bid = np.full((len(rs), KK), -1, np.int64)
    for st in range(0, N, 100000):
        s_ = Xs @ np.asarray(vecs[st:st + 100000], dtype=np.float32).T
        cs = np.concatenate([best, s_], 1)
        ci = np.concatenate([bid, np.broadcast_to(np.arange(st, st + s_.shape[1]), s_.shape)], 1)
        top = np.argpartition(-cs, KK, 1)[:, :KK]
        best = np.take_along_axis(cs, top, 1); bid = np.take_along_axis(ci, top, 1)
    o = np.argsort(-best, 1); bid = np.take_along_axis(bid, o, 1)
    rec_rows = []
    for i, cid in enumerate(rs):
        ex = [x for x in bid[i] if x != cid]; an = [x for x in nn_rs[i] if x != cid and x >= 0]
        rec_rows.append(dict(chunk_id=int(cid), recall10=len(set(ex[:K1]) & set(an[:K1])) / K1,
                             recall25=len(set(ex[:K2]) & set(an[:K2])) / K2, self_in_ann=int(cid in nn_rs[i])))
    rec = pd.DataFrame(rec_rows); rec.to_csv(f'{OUT}/kocc_recall_sample.csv', index=False)
    log(f'k-occ ANN recall vs exact ({len(rs)}): @10 {rec.recall10.mean():.4f} | @25 {rec.recall25.mean():.4f}')

# ---------------------------------------------------------------- 3) 리랭킹 표본 (GPU)
with stage('rerank'):
    import torch
    from sentence_transformers import CrossEncoder
    torch.set_num_threads(4)
    need = set(int(x) for x in R100.ravel() if x >= 0); text = {}
    with open(f'{J}/chunks.jsonl', encoding='utf-8') as f:
        for cid, l in enumerate(f):
            if cid in need: text[cid] = json.loads(l)['text']
    log('candidate texts', len(text))
    try:
        ce = CrossEncoder(CE_MODEL, device='cuda', max_length=256)
    except TypeError:
        ce = CrossEncoder(CE_MODEL, device='cuda')
    F10 = np.full((len(rr_q), K1), -1, np.int64); rrows = []; tr = time.time()
    for i, q in enumerate(rr_q):
        cands = [int(x) for x in R100[i] if x >= 0]
        s = np.asarray(ce.predict([(qmeta[q]['text'], text[x]) for x in cands], batch_size=100, show_progress_bar=False), np.float32)
        o = np.argsort(-s, kind='stable'); F10[i, :min(K1, len(o))] = np.array(cands)[o[:K1]]
        ce_rank = np.empty(len(o), int); ce_rank[o] = np.arange(len(o))
        rrows += [(int(q), qmeta[q]['_id'], r, x, float(s[r]), int(ce_rank[r])) for r, x in enumerate(cands)]
        if (i + 1) % 500 == 0: log(f'reranked {i+1}/{len(rr_q)} ({(i+1)/(time.time()-tr):.1f} q/s)')
    RES['rerank_qps'] = round(len(rr_q) / max(time.time() - tr, 1e-6), 2)
    RES['gpu_max_alloc_gb'] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    pd.DataFrame(rrows, columns=['q_row', '_id', 'rank', 'chunk_id', 'ce_score', 'ce_rank']).to_csv(f'{OUT}/rerank_sample.csv', index=False)
    del text, ce; torch.cuda.empty_cache()

# ---------------------------------------------------------------- 4) 지표
with stage('metrics'):
    from scipy.stats import spearmanr, rankdata
    real = pd.read_csv(f'{D}/rerank_top100_real.csv', usecols=['q_idx', 'rank', 'chunk_id', 'ce_rank'])
    RA10 = real[real['rank'] < 10].sort_values(['q_idx', 'rank']).chunk_id.values.reshape(-1, K1)
    RF10 = real[real.ce_rank < 10].sort_values(['q_idx', 'ce_rank']).chunk_id.values.reshape(-1, K1)
    TARGETS = {'train_eval_ann10': E10, 'train_rr_ann10': R100[:, :K1], 'train_rr_final10': F10,
               'real_ann10': RA10, 'real_final10': RF10}
    def counts(H): return np.bincount(H[H >= 0], minlength=N)
    TC = {t: counts(H) for t, H in TARGETS.items()}
    log_cnt = counts(L10)
    doc_len = np.bincount(doc)[doc]
    PRED = {'kocc10': kocc10.astype(float), 'kocc25': kocc25.astype(float), 'log': log_cnt.astype(float),
            'first_chunk': (cix == 0).astype(float), 'doc_len': doc_len.astype(float)}
    RAND = {f'random_s{s}': np.random.default_rng(1000 + s).random(N) for s in range(10)}
    TIE_SEED = {n: 100 + i for i, n in enumerate(list(PRED) + list(RAND) + ['oracle'])}
    tot_bytes = nb.sum()
    def topsets(score, seed):
        tk = np.random.default_rng(seed).random(N)
        order = np.lexsort((tk, -score))          # 점수 내림차순, 동점은 무작위
        cb = np.cumsum(nb[order]); out = {}
        for p in PCTS:
            for b in BASES:
                k = int(round(p * N)) if b == 'chunk' else int(np.searchsorted(cb, p * tot_bytes, side='right'))
                m = np.zeros(N, bool); m[order[:k]] = True; out[(p, b)] = m
        return out
    SETS = {n: topsets(s, TIE_SEED[n]) for n, s in {**PRED, **RAND}.items()}

    # 4-1) 포함률 (점 추정) + 부트스트랩용 쿼리별 행렬
    cov_rows, boot_rows, rec_ci_rows = [], [], []
    BOOT_P = list(PRED) + list(RAND)
    cols = [(n, p, b) for n in BOOT_P for p in PCTS for b in BASES]
    brng = np.random.default_rng(a.seed + 7)
    for t, H in TARGETS.items():
        cnt = TC[t]; Hc = np.where(H < 0, 0, H); V = H >= 0; nbh = nb[Hc] * V
        osets = topsets(cnt.astype(float), TIE_SEED['oracle'])
        for n, ss in {**SETS, 'oracle': osets}.items():
            for (p, b), m in ss.items():
                cov_rows.append(dict(target=t, predictor=n, top_pct=p, budget_basis=b,
                                     n_selected=int(m.sum()), selected_bytes_frac=float(nb[m].sum() / tot_bytes),
                                     cov_access=float(cnt[m].sum() / cnt.sum()),
                                     cov_access_bytes=float((cnt * nb)[m].sum() / (cnt * nb).sum())))
        # 쿼리별 분자: chunk 기준 = 접근 수, byte 기준 = 접근 바이트
        M = np.stack([(SETS[n][(p, b)][Hc] * (V if b == 'chunk' else nbh)).sum(1) for n, p, b in cols], 1).astype(np.float64)
        tot_c, tot_b = V.sum(1).astype(np.float64), nbh.sum(1).astype(np.float64)
        Q = len(H); ci = {c_: i for i, c_ in enumerate(cols)}
        def recov(cov):  # cov: (B, len(cols)) -> dict
            out = {}
            for p in PCTS:
                for b in BASES:
                    rnd = np.mean([cov[:, ci[(f'random_s{s}', p, b)]] for s in range(10)], 0)
                    lg = cov[:, ci[('log', p, b)]]
                    for n in ['kocc10', 'kocc25', 'first_chunk', 'doc_len']:
                        out[(n, p, b)] = ((cov[:, ci[(n, p, b)]] - rnd) / (lg - rnd), cov[:, ci[(n, p, b)]], rnd, lg)
            return out
        def cov_of(W):
            num = W @ M
            den_c, den_b = W @ tot_c, W @ tot_b
            den = np.stack([den_c if b == 'chunk' else den_b for _, _, b in cols], 1)
            return num / den
        point = recov(cov_of(np.ones((1, Q))))
        boots = []
        for st in range(0, a.n_boot, 100):
            nbt = min(100, a.n_boot - st)
            W = np.stack([np.bincount(brng.integers(0, Q, Q), minlength=Q) for _ in range(nbt)]).astype(np.float64)
            boots.append(cov_of(W))
        bc = np.concatenate(boots, 0); bres = recov(bc)
        for key, (r, cv, rnd, lg) in bres.items():
            n, p, b = key; pr = point[key]
            rec_ci_rows.append(dict(target=t, predictor=n, top_pct=p, budget_basis=b, n_queries=Q,
                                    cov_pred=float(pr[1][0]), cov_random=float(pr[2][0]), cov_log=float(pr[3][0]),
                                    recovery=float(pr[0][0]), ci_lo=float(np.nanpercentile(r, 2.5)), ci_hi=float(np.nanpercentile(r, 97.5)),
                                    boot_mean=float(np.nanmean(r)), n_boot=a.n_boot, n_nan=int(np.isnan(r).sum())))
            for bi in range(len(r)):
                boot_rows.append((t, n, p, b, bi, float(cv[bi]), float(rnd[bi]), float(lg[bi]), float(r[bi])))
        check_mem(f'metrics {t}')
    cov = pd.DataFrame(cov_rows); cov.to_csv(f'{OUT}/coverage.csv', index=False)
    rci = pd.DataFrame(rec_ci_rows); rci.to_csv(f'{OUT}/recovery_ci.csv', index=False)
    pd.DataFrame(boot_rows, columns=['target', 'predictor', 'top_pct', 'budget_basis', 'boot', 'cov_pred', 'cov_random_mean', 'cov_log', 'recovery']
                 ).to_csv(f'{OUT}/recovery_bootstrap_raw.csv', index=False)
    del boot_rows

    # 4-2) Spearman, Hot 분류
    sp_rows, hot_rows = [], []
    SP_P = {**PRED, 'random_s0': RAND['random_s0']}
    for t, cnt in TC.items():
        acc = cnt > 0
        hot_order = np.lexsort((np.random.default_rng(TIE_SEED['oracle']).random(N), -cnt))
        for n, s in SP_P.items():
            r_all = spearmanr(s, cnt)[0]; r_acc = spearmanr(s[acc], cnt[acc])[0]
            sp_rows.append(dict(target=t, predictor=n, spearman_all=float(r_all), spearman_accessed=float(r_acc), n_accessed=int(acc.sum())))
            rk = rankdata(s)
            for p in (0.01, 0.05):
                k = int(round(p * N)); hot = np.zeros(N, bool); hot[hot_order[:k]] = True
                prec = float((SETS[n][(p, 'chunk')] & hot).sum() / k) if (p, 'chunk') in SETS[n] else float('nan')
                n1 = hot.sum(); n0 = N - n1
                auc = float((rk[hot].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
                hot_rows.append(dict(target=t, predictor=n, hot_pct=p, precision_at_hot=prec, auc=auc,
                                     hot_min_count=int(cnt[hot_order[k - 1]])))
    sp = pd.DataFrame(sp_rows); sp.to_csv(f'{OUT}/spearman.csv', index=False)
    hot = pd.DataFrame(hot_rows); hot.to_csv(f'{OUT}/hot_classification.csv', index=False)

    # 4-3) 분포 진단, 10분위 편향
    def gini(x):
        x = np.sort(x.astype(np.float64)); n = len(x)
        return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if x.sum() > 0 else float('nan')
    dist_rows = []
    for t, cnt in {**TC, 'pred_kocc10': kocc10, 'pred_kocc25': kocc25, 'pred_log': log_cnt}.items():
        acc = cnt > 0
        dist_rows.append(dict(series=t, total=int(cnt.sum()), n_accessed=int(acc.sum()), frac_accessed=float(acc.mean()),
                              frac_ge2_of_all=float((cnt >= 2).mean()), frac_ge2_of_accessed=float((cnt >= 2).sum() / max(acc.sum(), 1)),
                              gini_all=gini(cnt), gini_accessed=gini(cnt[acc]), max=int(cnt.max()),
                              top1pct_share=float(np.sort(cnt)[::-1][:N // 100].sum() / max(cnt.sum(), 1))))
    dist = pd.DataFrame(dist_rows); dist.to_csv(f'{OUT}/distribution.csv', index=False)
    dec = np.minimum(np.arange(N) * 10 // N, 9)
    dec_rows = []
    for t, cnt in {**TC, 'pred_kocc10': kocc10, 'pred_kocc25': kocc25, 'pred_log': log_cnt, 'nbytes': nb}.items():
        sh = np.bincount(dec, weights=cnt.astype(float), minlength=10) / cnt.sum()
        dec_rows.append(dict(series=t, **{f'd{i}': float(sh[i]) for i in range(10)}))
    decdf = pd.DataFrame(dec_rows); decdf.to_csv(f'{OUT}/decile_by_chunk_order.csv', index=False)

    # 4-4) chunk별 원시 점수
    pd.DataFrame(dict(chunk_id=np.arange(N), doc_idx=doc, chunk_idx=cix, nbytes=nb, doc_len=doc_len,
                      kocc10=kocc10, kocc25=kocc25, log_cnt=log_cnt, **{f'cnt_{t}': TC[t] for t in TC})
                 ).to_csv(f'{OUT}/chunk_scores.csv', index=False)
    pd.DataFrame(dict(_id=[qmeta[i]['_id'] for i in perm], split=['log'] * h + ['eval'] * (len(perm) - h))
                 ).to_csv(f'{OUT}/train_split.csv', index=False)

# ---------------------------------------------------------------- 5) 요약 출력 + config
RES['max_rss_gb'] = round(maxrss_gb(), 3)
RES['self_missing'] = self_missing; RES['self_missing_frac'] = self_missing / len(kq)
RES['short_rows'] = short_rows
cfg = dict(args=vars(a), kocc_mode=kmode, n_kocc_queries=int(len(kq)), k_main=K1, k_aux=K2,
           n_train_raw=132803, n_train_dedup=NQd, n_train_used=int(len(pool)), n_log=int(len(log_q)), n_eval=int(len(eval_q)),
           n_rerank=int(len(rr_q)), n_real=int(len(RA10)), n_recall_sample=int(len(rs)),
           recall10=float(rec.recall10.mean()), recall25=float(rec.recall25.mean()),
           timings_s=TIME, resources=RES,
           notes=['self exclusion: same pk only, searched k+1=26 neighbors',
                  'coverage: chunk budget -> fraction of accesses; byte budget -> fraction of accessed bytes (both columns stored)',
                  'ties broken by fixed-seed random key per predictor; random baseline = mean of 10 seeds',
                  'bootstrap: resample queries of each target with replacement; predictor rankings fixed',
                  'hot set = top p% chunks by target count, random tie-break',
                  'train_rr_ann10 = rank<10 of top-100 search (ef_top100); train_eval_ann10 = top-10 search (ef_top10)',
                  'doc_len = number of chunks in document',
                  'real_* targets use existing rerank_top100_real.csv'])
if kmode != 'full':
    cfg['notes'].append(f'k-occurrence estimated from {len(kq)} query chunks ({kmode}); counts not scaled')
json.dump(cfg, open(f'{OUT}/config.json', 'w'), indent=1, ensure_ascii=False, default=str)

pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
print('\n==== config ====\n' + json.dumps({k: cfg[k] for k in cfg if k not in ('notes', 'args')}, indent=1, default=str))
print('\n==== distribution ====\n' + dist.round(4).to_string(index=False))
cv = cov[cov.target.isin(TC)].copy()
cv['metric'] = np.where(cv.budget_basis == 'chunk', cv.cov_access, cv.cov_access_bytes)
cv = cv[~cv.predictor.str.startswith('random_s')]
rmean = cov[cov.predictor.str.startswith('random_s')].assign(metric=lambda d: np.where(d.budget_basis == 'chunk', d.cov_access, d.cov_access_bytes)
        ).groupby(['target', 'top_pct', 'budget_basis']).metric.mean().reset_index().assign(predictor='random_mean10')
cv = pd.concat([cv[['target', 'predictor', 'top_pct', 'budget_basis', 'metric']], rmean])
print('\n==== coverage (chunk budget: access frac / byte budget: access-byte frac) ====')
print(cv.pivot_table(index=['target', 'predictor'], columns=['budget_basis', 'top_pct'], values='metric').round(4).to_string())
print('\n==== recovery vs log (bootstrap 95% CI) ====\n' + rci.round(4).to_string(index=False))
print('\n==== spearman ====\n' + sp.round(4).to_string(index=False))
print('\n==== hot classification ====\n' + hot.round(4).to_string(index=False))
print('\n==== decile share by chunk order ====\n' + decdf.round(4).to_string(index=False))
log('ALL DONE')
