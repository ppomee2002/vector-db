# reuse_check.py: ② Semantic Approximate Reuse 가능성 검증 (공통 데이터/쿼리, Milvus, M2 비교 설정 포함)
# 사용: python reuse_check.py smoke (쿼리 50개, 동작 확인) | python reuse_check.py full (쿼리 3452개)
import sys, os, re, json, time, string, subprocess, collections, numpy as np, pandas as pd
MODE = sys.argv[1] if len(sys.argv) > 1 else 'smoke'
NQ_USE = 50 if MODE == 'smoke' else 3452
WARM, REQ = (100, 400) if MODE == 'smoke' else (2000, 10000)          # M2: 12,000 requests, 2,000 warm-up
M2_SEED = 20261103; M2_TOTAL_OVER_CTX = 1395449 / 352256               # M2 layout: (ANN+context pages)/context pages
J = '/workspace/junlim/data_nq'; D = '/workspace/jaeeun/sel_fetch/data'
R = f'/workspace/jaeeun/sel_fetch/results/reuse_{MODE}'; os.makedirs(R, exist_ok=True)
TAUS_GATE = [0.98, 0.95, 0.90, 0.85, 0.80]; TAUS_SIM = [0.95, 0.90, 0.85]
EMB_BYTES = 384 * 4
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:6.0f}s]', *a, flush=True)
def sh(cmd): return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
cfg = dict(mode=MODE, n_queries=NQ_USE, warm=WARM, req=REQ, m2_seed=M2_SEED, taus_gate=TAUS_GATE, taus_sim=TAUS_SIM, emb_bytes=EMB_BYTES,
           requests='per query occurrence: rerank final top-10; cache holds only chunks of previous query occurrences',
           substitute_rule='substitute must not be in the current query request set',
           m2_caps='ctx-only = 5% of context bytes; all-to-ctx = 5% x (ANN+ctx)/ctx of context bytes (M2 layout ratio 3.96)',
           unit_note='M2: Qdrant full NQ, 4KB pages; here: common 1M-chunk set, chunk bytes')

# 0) 자원 확인
log('disk', sh(['df', '-h', '/workspace']).splitlines()[-1])
log('gpu', sh(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader']).replace('\n', ' | '))

# 1) 요청 (쿼리마다 리랭킹 최종 Top-10)
rr = pd.read_csv(f'{D}/rerank_top100_real.csv'); rr = rr[rr.q_idx < NQ_USE]
req = rr[rr.ce_rank < 10].sort_values(['q_idx', 'ce_rank'])[['q_idx', 'chunk_id']].reset_index(drop=True)
ce10 = rr[rr.ce_rank == 9].set_index('q_idx').ce_score
meta = np.load(f'{D}/chunk_meta.npz'); docarr = meta['doc_idx']; nbytes = meta['nbytes'].astype(np.int64)
vecs = np.load(f'{J}/chunk_vecs.npy', mmap_mode='r'); N = vecs.shape[0]
uniq = np.unique(req.chunk_id.values); U = np.asarray(vecs[uniq], dtype=np.float32); dim = U.shape[1]
pos = np.searchsorted(uniq, req.chunk_id.values); reqc = req.chunk_id.values
gi = req.groupby('q_idx', sort=True).indices
qreq = {int(q): [int(x) for x in reqc[idx]] for q, idx in gi.items()}; qpos = {int(q): pos[idx] for q, idx in gi.items()}
corpus_bytes = int(nbytes.sum()); uniq_bytes = int(nbytes[uniq].sum())
log(f'queries {NQ_USE} | requests {len(req)} | unique {len(uniq)} | corpus text {corpus_bytes/1e6:.1f} MB | requested text {uniq_bytes/1e6:.2f} MB '
    f'| mean chunk {nbytes.mean():.0f} B vs embedding {EMB_BYTES} B')

# 2) Milvus: (A) corpus 전체 최근접 '다른' chunk, (보완3a) 벡터를 결과와 함께 받을 때의 비용. 오류가 나도 release
from pymilvus import MilvusClient
c = MilvusClient(uri='http://localhost:19530', db_name='jaeeun'); log('db_name = jaeeun')
nn_id = np.full(len(uniq), -1, np.int64); nn_sim = np.zeros(len(uniq), np.float32); lat = []; vec_len = None
Qv = np.asarray(np.load(f'{J}/query_vecs.npy', mmap_mode='r')[:3452], dtype=np.float32)
rng = np.random.default_rng(0)
try:
    c.load_collection('nq1m_sel'); log('load state', c.get_load_state('nq1m_sel').get('state'))
    for st in range(0, len(uniq), 200):
        res = c.search('nq1m_sel', data=U[st:st + 200].tolist(), limit=6, search_params={'metric_type': 'IP', 'params': {'ef': 128}})
        for i, hits in enumerate(res):
            for h in hits:
                if int(h['pk']) != int(uniq[st + i]):
                    nn_id[st + i] = int(h['pk']); nn_sim[st + i] = float(h['distance']); break
    log('corpus NN search done')
    def one(qi, with_vec):
        t = time.perf_counter()
        r = c.search('nq1m_sel', data=[Qv[qi].tolist()], limit=10, search_params={'metric_type': 'IP', 'params': {'ef': 256}},
                     output_fields=['vec'] if with_vec else [])
        return (time.perf_counter() - t) * 1000, (len(r[0][0]['vec']) if with_vec else None)
    vec_len = one(0, True)[1]; log('vector returned with search result, length =', vec_len, '(expected', dim, ')')
    qsel = rng.choice(3452, 60 if MODE == 'smoke' else 300, replace=False)
    for qi in qsel[:20]: one(int(qi), True); one(int(qi), False)
    for rep in range(3):
        order = [(int(qi), w) for qi in qsel for w in (0, 1)]
        for k in rng.permutation(len(order)):
            qi, w = order[k]; lat.append((rep, qi, w, one(qi, bool(w))[0]))
    log('latency done (random order, 20 warm-up pairs, 3 reps)')
finally:
    c.release_collection('nq1m_sel'); log('released, state', c.get_load_state('nq1m_sel').get('state')); c.close()
lat = pd.DataFrame(lat, columns=['rep', 'q_idx', 'with_vec', 'ms']); lat.to_csv(f'{R}/reuse_milvus_vec_latency.csv', index=False)

# 2b) Milvus 최근접 검증 (정확 검색 대비)
samp = rng.choice(len(uniq), min(200, len(uniq)), replace=False); sid = uniq[samp]
best = np.full(len(samp), -1e9, np.float32); bid = np.full(len(samp), -1, np.int64)
for st in range(0, N, 100000):
    s = U[samp] @ np.asarray(vecs[st:st + 100000], dtype=np.float32).T
    inb = (sid >= st) & (sid < st + s.shape[1]); s[np.nonzero(inb)[0], sid[inb] - st] = -np.inf
    j = s.argmax(1); m = s[np.arange(len(samp)), j]; up = m > best; best[up] = m[up]; bid[up] = j[up] + st
nn_check = dict(top1_agree=float(np.mean(bid == nn_id[samp])), mean_abs_sim_diff=float(np.mean(np.abs(best - nn_sim[samp]))))
nn = pd.DataFrame(dict(chunk_id=uniq, nn_id=nn_id, nn_sim=nn_sim))
nn['nn_same_doc'] = docarr[nn.chunk_id.values] == docarr[nn.nn_id.values]
nn['nn_adjacent'] = nn.nn_same_doc & (np.abs(nn.chunk_id - nn.nn_id) == 1)
nn.to_csv(f'{R}/reuse_corpus_nn.csv', index=False)
gate = pd.DataFrame([dict(tau=t, frac_with_neighbor=float((nn.nn_sim >= t).mean()),
                          same_doc_frac=float(nn[nn.nn_sim >= t].nn_same_doc.mean()) if (nn.nn_sim >= t).any() else np.nan,
                          adjacent_frac=float(nn[nn.nn_sim >= t].nn_adjacent.mean()) if (nn.nn_sim >= t).any() else np.nan) for t in TAUS_GATE])
gate.to_csv(f'{R}/reuse_gate.csv', index=False)

# 3) 캐시 시뮬레이션 (바이트 LRU, 본문만 vs 본문+임베딩, 흐름: distinct + M2 흐름)
TMIN = min(TAUS_SIM); nbr = [None] * len(uniq)
for st in range(0, len(uniq), 2000):
    S = U[st:st + 2000] @ U.T
    for r_ in range(S.shape[0]):
        row = S[r_]; row[st + r_] = -1; j = np.nonzero(row >= TMIN)[0]; o = np.argsort(-row[j]); nbr[st + r_] = (j[o], row[j][o])
def mk_trace(kind, alpha=None):
    rt = np.random.default_rng(M2_SEED); total = WARM + REQ
    if kind == 'distinct': return np.arange(NQ_USE), 0
    if kind == 'natural': return np.concatenate([rt.permutation(NQ_USE) for _ in range(-(-total // NQ_USE))])[:total], WARM
    ranks = np.arange(1, NQ_USE + 1, dtype=float) ** -alpha
    return rt.permutation(NQ_USE)[rt.choice(NQ_USE, size=total, p=ranks / ranks.sum())], WARM
TRACES = {'distinct': mk_trace('distinct'), 'natural(M2)': mk_trace('natural'), 'zipf1.2(M2)': mk_trace('zipf', 1.2), 'zipf0.9(M2)': mk_trace('zipf', 0.9)}
def simulate(trace, warm, cap, tau, with_emb, collect):
    cost = lambda cid: int(nbytes[cid]) + (EMB_BYTES if with_emb else 0)
    lru = collections.OrderedDict(); used = 0; ex = ap = nreq = 0; pairs = []
    for t_i, q in enumerate(trace):
        q = int(q); cnt = t_i >= warm; cur = set(qreq[q]); touch, ins = [], []
        for cid, p in zip(qreq[q], qpos[q]):
            nreq += cnt
            if cid in lru: ex += cnt; touch.append(cid); continue
            if tau is not None:
                hit = -1
                for j, sv in zip(*nbr[p]):
                    if sv < tau: break
                    cj = int(uniq[j])
                    if cj in lru and cj not in cur: hit = cj; hs = float(sv); break
                if hit >= 0:
                    ap += cnt; touch.append(hit)
                    if collect and cnt: pairs.append((t_i, q, cid, hit, hs))
                    continue
            ins.append(cid)
        for cid in touch:
            if cid in lru: lru.move_to_end(cid)
        for cid in ins:
            if cid in lru: continue
            cst = cost(cid)
            if cst > cap: continue
            while used + cst > cap: old, _ = lru.popitem(last=False); used -= cost(old)
            lru[cid] = True; used += cst
    return ex / max(nreq, 1), ap / max(nreq, 1), pairs
caps = {'M2 5% (ctx only)': int(0.05 * corpus_bytes), 'M2 5% (all to ctx)': int(0.05 * M2_TOTAL_OVER_CTX * corpus_bytes),
        '5%_reqset': int(0.05 * uniq_bytes), '25%_reqset': int(0.25 * uniq_bytes), '100%_reqset': uniq_bytes}
sim_rows, pair_rows = [], []
for tname, (trace, warm) in TRACES.items():
    for cname, cb in caps.items():
        for acc in ('text_only', 'text+emb'):
            for tau in [None] + TAUS_SIM:
                collect = (tname == 'distinct' and cname == '100%_reqset' and acc == 'text_only' and tau is not None)
                ex, ap, pairs = simulate(trace, warm, cb, tau, acc == 'text+emb', collect)
                sim_rows.append(dict(trace=tname, cache=cname, cap_MB=cb / 1e6, accounting=acc, tau='exact' if tau is None else f'{tau:.2f}',
                                     exact_hit=ex, approx_hit=ap, total_hit=ex + ap))
                if collect: pair_rows += [dict(tau=tau, occ=t_i, q_idx=q, orig=o, sub=s_, sim=sv) for t_i, q, o, s_, sv in pairs]
    log('cache simulation done:', tname)
simdf = pd.DataFrame(sim_rows)
base = simdf[simdf.tau == 'exact'].set_index(['trace', 'cache', 'accounting']).total_hit
simdf['gain_vs_exact'] = [r.total_hit - base[(r.trace, r.cache, r.accounting)] for r in simdf.itertuples()]
simdf.to_csv(f'{R}/reuse_cache_summary.csv', index=False)

# 3c) 캐시 안 유사 chunk 조회 계산 비용 (3회 반복 중앙값)
look = []
for k in sorted({min(1000, len(uniq)), min(10000, len(uniq)), len(uniq)}):
    ts = []
    for rep in range(3):
        t = time.perf_counter()
        for p in range(200): _ = U[:k] @ U[p % len(uniq)]
        ts.append((time.perf_counter() - t) / 200 * 1000)
    look.append(dict(cache_chunks=k, brute_force_lookup_ms=float(np.median(ts))))
look = pd.DataFrame(look); look.to_csv(f'{R}/reuse_lookup_cost.csv', index=False)

# 4) 사실관계 검증 도구 로드 (스모크에서 반드시 확인)
qs = [json.loads(l)['text'] for l in open(f'{J}/queries.jsonl', encoding='utf-8')][:3452]
def norm(s):
    s = s.lower(); s = ''.join(ch for ch in s if ch not in string.punctuation); s = re.sub(r'\b(a|an|the)\b', ' ', s); return ' '.join(s.split())
ans, ans_cov = {}, None
try:
    from datasets import load_dataset
    try: ds = load_dataset('google-research-datasets/nq_open', split='validation')
    except Exception: ds = load_dataset('nq_open', split='validation')
    qn = lambda s: ' '.join(s.lower().replace('?', ' ').split())
    amap = {qn(r['question']): r['answer'] for r in ds}
    for qi in range(NQ_USE):
        a = amap.get(qn(qs[qi]))
        if a: ans[qi] = [norm(x) for x in a if norm(x)]
    ans_cov = len(ans) / NQ_USE; log(f'NQ-open answers matched: {len(ans)}/{NQ_USE} ({ans_cov:.1%})')
except Exception as e:
    log('NQ-open answers unavailable:', repr(e)[:300])
from sentence_transformers import CrossEncoder
from transformers import AutoConfig
def load_ce(name):
    try: return CrossEncoder(name, device='cuda', max_length=320)
    except TypeError: return CrossEncoder(name, device='cuda')
ce = load_ce('cross-encoder/ms-marco-MiniLM-L-6-v2')
NLI_NAME, nli, lab = None, None, None
for name in ('cross-encoder/nli-deberta-v3-small', 'cross-encoder/nli-MiniLM2-L6-H768'):
    try:
        nli = load_ce(name); lab = {str(v).lower(): int(k) for k, v in AutoConfig.from_pretrained(name).id2label.items()}; NLI_NAME = name; break
    except Exception as e:
        log('NLI load failed', name, repr(e)[:200])
log('NLI model', NLI_NAME, '| labels', lab)
def nli_probs(pairs):
    a = np.asarray(nli.predict(pairs, batch_size=64, show_progress_bar=False), dtype=np.float64)
    if not (a.min() >= 0 and np.allclose(a.sum(1), 1, atol=1e-3)):
        a = np.exp(a - a.max(1, keepdims=True)); a /= a.sum(1, keepdims=True)
    return a
log('NLI sanity:', {k: round(float(v), 3) for k, v in zip(['entail_same', 'entail_diff'],
    [nli_probs([('Paris is the capital of France.', 'The capital of France is Paris.')])[0, lab['entailment']],
     nli_probs([('The film was released in 1999.', 'The film was released in 2005.')])[0, lab['entailment']]])})

# 5) 근사 재사용 쌍의 사실관계 검증 (distinct, 캐시 100%_reqset, 본문만)
pairs = pd.DataFrame(pair_rows); qual = pd.DataFrame(); samples = []
NUM = re.compile(r'\d+(?:[.,]\d+)*'); ENT = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*')
qr = pd.read_csv(f'{D}/qrels_chunk.tsv', sep='\t'); gold = set(zip(qr.q_idx, qr.chunk_id))
if len(pairs):
    ev = pairs.drop_duplicates(['q_idx', 'orig', 'sub'])[['q_idx', 'orig', 'sub', 'sim']].copy().reset_index(drop=True)
    need = set(ev.orig) | set(ev['sub']); text = {}
    with open(f'{J}/chunks.jsonl', encoding='utf-8') as f:
        for cid, l in enumerate(f):
            if cid in need: text[cid] = json.loads(l)['text']
    ev['ce_orig'] = ce.predict([(qs[q], text[o]) for q, o in zip(ev.q_idx, ev.orig)], batch_size=128, show_progress_bar=False)
    ev['ce_sub'] = ce.predict([(qs[q], text[s_]) for q, s_ in zip(ev.q_idx, ev['sub'])], batch_size=128, show_progress_bar=False)
    ev['equiv_ce'] = ev.ce_sub >= ce10.reindex(ev.q_idx).values
    def has_ans(q, t):
        a = ans.get(int(q))
        if not a: return np.nan
        nt = f' {norm(t)} '; return float(any(f' {x} ' in nt for x in a))
    ev['orig_has_ans'] = [has_ans(q, text[o]) for q, o in zip(ev.q_idx, ev.orig)]
    ev['sub_has_ans'] = [has_ans(q, text[s_]) for q, s_ in zip(ev.q_idx, ev['sub'])]
    ev['answer_lost'] = (ev.orig_has_ans == 1) & (ev.sub_has_ans == 0)
    fw = nli_probs([(text[s_], text[o]) for o, s_ in zip(ev.orig, ev['sub'])]); bw = nli_probs([(text[o], text[s_]) for o, s_ in zip(ev.orig, ev['sub'])])
    ev['nli_sub_entails_orig'] = fw[:, lab['entailment']]; ev['nli_orig_entails_sub'] = bw[:, lab['entailment']]
    ev['nli_contradiction'] = np.maximum(fw[:, lab['contradiction']], bw[:, lab['contradiction']])
    def nums(t): return set(NUM.findall(t))
    ev['num_mismatch'] = [bool(nums(text[o])) and not nums(text[o]) <= nums(text[s_]) for o, s_ in zip(ev.orig, ev['sub'])]
    def ent_recall(a, b):
        A = set(ENT.findall(a)); return np.nan if not A else len(A & set(ENT.findall(b))) / len(A)
    ev['entity_recall'] = [ent_recall(text[o], text[s_]) for o, s_ in zip(ev.orig, ev['sub'])]
    ev['jaccard'] = [len(set(text[o].lower().split()) & set(text[s_].lower().split())) / max(len(set(text[o].lower().split()) | set(text[s_].lower().split())), 1) for o, s_ in zip(ev.orig, ev['sub'])]
    ev['same_doc'] = docarr[ev.orig.values] == docarr[ev['sub'].values]
    ev['orig_gold'] = [(q, o) in gold for q, o in zip(ev.q_idx, ev.orig)]; ev['sub_gold'] = [(q, s_) in gold for q, s_ in zip(ev.q_idx, ev['sub'])]
    ev['factual_ok'] = (~ev.answer_lost) & (~ev.num_mismatch) & (ev.nli_sub_entails_orig >= 0.5)
    ev.to_csv(f'{R}/reuse_pairs_checked.csv', index=False)
    pe = pairs.merge(ev, on=['q_idx', 'orig', 'sub', 'sim'], how='left'); qrows = []
    for t in TAUS_SIM:
        s = pe[pe.tau == t]
        if not len(s): qrows.append(dict(tau=t, pairs=0)); continue
        app = s[s.orig_has_ans == 1]
        qrows.append(dict(tau=t, pairs=len(s), factual_ok=s.factual_ok.mean(), nli_entail=(s.nli_sub_entails_orig >= 0.5).mean(),
                          nli_bidir=((s.nli_sub_entails_orig >= 0.5) & (s.nli_orig_entails_sub >= 0.5)).mean(), contradiction=(s.nli_contradiction >= 0.5).mean(),
                          num_mismatch=s.num_mismatch.mean(), answer_lost=(app.answer_lost.mean() if len(app) else np.nan), answer_applicable=len(app),
                          gold_lost=int((s.orig_gold & ~s.sub_gold).sum()), entity_recall=s.entity_recall.mean(), equiv_ce=s.equiv_ce.mean(),
                          jaccard=s.jaccard.mean(), same_doc=s.same_doc.mean()))
    qual = pd.DataFrame(qrows); qual.to_csv(f'{R}/reuse_quality.csv', index=False)
    hi = ev[ev.sim >= 0.90]; pick = hi.sample(min(5, len(hi)), random_state=0) if len(hi) else ev.nlargest(5, 'sim')
    for r in pick.itertuples():
        samples.append(f"sim={r.sim:.3f} | factual_ok={r.factual_ok} nli={r.nli_sub_entails_orig:.2f} contra={r.nli_contradiction:.2f} num_mismatch={r.num_mismatch} "
                       f"ans {r.orig_has_ans}->{r.sub_has_ans} | ce {r.ce_orig:.2f}->{r.ce_sub:.2f}\n  Q : {qs[int(r.q_idx)]}\n  원래: {text[int(r.orig)][:200]}\n  대체: {text[int(r.sub)][:200]}")

# 6) 판정 (미리 정한 기준: distinct 흐름, M2 5% (ctx only), 본문만) + 저장
cfg.update(caps_bytes=caps, nn_check=nn_check, nli_model=NLI_NAME, nli_labels=lab, answer_coverage=ans_cov, vec_len=vec_len)
json.dump(cfg, open(f'{R}/reuse_config.json', 'w'), indent=1, default=str)
mv = simdf[(simdf.trace == 'distinct') & (simdf.cache == 'M2 5% (ctx only)')]
verdict = []
for t in TAUS_SIM:
    lbl = f'{t:.2f}'
    g_txt = float(mv[(mv.tau == lbl) & (mv.accounting == 'text_only')].gain_vs_exact.iloc[0])
    g_emb = float(mv[(mv.tau == lbl) & (mv.accounting == 'text+emb')].gain_vs_exact.iloc[0])
    qq = qual[qual.tau == t] if len(qual) else pd.DataFrame()
    q = qq.iloc[0] if len(qq) and qq.iloc[0].get('pairs', 0) else None
    if g_txt < 0.05: v = 'FAIL (gain < 5%p)'
    elif q is None: v = 'FAIL (no pairs)'
    elif q.factual_ok >= 0.9 and (np.isnan(q.answer_lost) or q.answer_lost < 0.05) and q.num_mismatch < 0.05: v = 'PASS'
    else: v = 'AMBIGUOUS (opportunity but factual check fails)'
    verdict.append(dict(tau=t, gain_text_only=g_txt, gain_text_plus_emb=g_emb, verdict=v))
if float(gate[gate.tau == 0.90].frac_with_neighbor.iloc[0]) < 0.05:
    for d_ in verdict: d_['verdict'] = 'FAIL (upper bound: <5% have corpus neighbor at 0.90)'
verdict = pd.DataFrame(verdict); verdict.to_csv(f'{R}/reuse_verdict.csv', index=False)

pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
lm = lat.groupby('with_vec').ms.agg(median='median', p95=lambda x: x.quantile(.95))
pv = simdf.pivot_table(index=['trace', 'cache', 'accounting'], columns='tau', values='total_hit', aggfunc='first', sort=False)
print(f'\n================ ② Semantic Reuse 결과 [{MODE}] ================')
print(f'쿼리 {NQ_USE}, 요청 {len(req)}개(쿼리당 최종 Top-10), 서로 다른 chunk {len(uniq)}개 | 요청 chunk 전체 {uniq_bytes/1e6:.2f} MB')
print('캐시 크기 MB:', {k: round(v / 1e6, 2) for k, v in caps.items()})
print('\n[A] 상한: corpus 전체에 유사도 >= tau인 다른 chunk가 있는 비율 | Milvus 최근접 검증', {k: round(v, 4) for k, v in nn_check.items()})
print(gate.round(4).to_string(index=False))
print('\n[B] 캐시 hit 비율 (열: exact = 정확 재사용만, 0.95/0.90/0.85 = 근사 재사용 포함). distinct = 판정용, (M2) = M2와 같은 흐름')
print(pv.round(4).to_string())
print('\n[보완3] 임베딩 조회 비용')
print(f'  (a) Milvus 검색 latency ms (with_vec 0=ID만, 1=벡터 포함; 랜덤 순서/예열/3회). vec length {vec_len}')
print(lm.round(3).to_string())
print(f'  (b) 캐시에 임베딩 저장 시 chunk당 {EMB_BYTES} B 추가 (본문 평균 {nbytes.mean():.0f} B의 {EMB_BYTES/nbytes.mean():.1f}배) -> [B]의 text+emb 행')
print('  (c) 캐시 안 유사 chunk 조회 계산 비용 (brute force, 3회 중앙값)'); print(look.round(4).to_string(index=False))
print('  ※ HNSW(메모리 상주)에서만 측정. DiskANN/IVF_PQ 등 원본 벡터가 디스크/압축이면 추가 I/O 발생 가능 (미측정)')
print('\n[보완1] 사실관계 검증 (distinct, 캐시 100%_reqset, 본문만). factual_ok = 정답 유지 & 숫자 일치 & NLI 함의>=0.5')
print(f'  NLI={NLI_NAME}, NQ-open 정답 매칭률={ans_cov}')
print(qual.round(4).to_string(index=False) if len(qual) else '  근사 재사용 쌍 없음')
print('\n[샘플 쌍]'); print('\n'.join(samples) if samples else '  (없음)')
print('\n[판정] distinct 흐름, M2 5% (ctx only), 본문만 기준'); print(verdict.round(4).to_string(index=False))
print(f'\nsaved: {R}/')
log('ALL DONE')
