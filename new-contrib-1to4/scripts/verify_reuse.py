# verify_reuse.py: ② 결과 신뢰성 검증 (기존 CSV 재사용 + 소규모 추가 검증). 새 알고리즘/대규모 실험 없음
import os, json, re, time, collections, numpy as np, pandas as pd
J = '/workspace/junlim/data_nq'; D = '/workspace/jaeeun/sel_fetch/data'
R = '/workspace/jaeeun/sel_fetch/results/reuse_full'; O = f'{R}/verify'; os.makedirs(O, exist_ok=True)
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:6.0f}s]', *a, flush=True)

rr = pd.read_csv(f'{D}/rerank_top100_real.csv')
req = rr[rr.ce_rank < 10].sort_values(['q_idx', 'ce_rank'])[['q_idx', 'chunk_id']].reset_index(drop=True)
nn = pd.read_csv(f'{R}/reuse_corpus_nn.csv'); ev = pd.read_csv(f'{R}/reuse_pairs_checked.csv')
sim_sum = pd.read_csv(f'{R}/reuse_cache_summary.csv')
docarr = np.load(f'{D}/chunk_meta.npz')['doc_idx']
vecs = np.load(f'{J}/chunk_vecs.npy', mmap_mode='r')
uniq = np.unique(req.chunk_id.values); U = np.asarray(vecs[uniq], dtype=np.float32)
qs = [json.loads(l)['text'] for l in open(f'{J}/queries.jsonl', encoding='utf-8')][:3452]

# ---------- V1: 구현 검증 ----------
v1 = {'exact_dup_rate_recomputed': 1 - len(uniq) / len(req)}
w = req.merge(nn, on='chunk_id')
for t in (0.95, 0.90, 0.85, 0.80):
    v1[f'ceiling_request_weighted@{t}'] = float((w.nn_sim >= t).mean())
inreq = np.isin(nn.nn_id.values, uniq)
for t in (0.90, 0.85):
    m = nn.nn_sim.values >= t
    v1[f'corpus_neighbor_is_in_requested_set@{t}'] = float(inreq[m].mean()) if m.any() else np.nan
first_q = {}
for q, cid in zip(req.q_idx.values, req.chunk_id.values): first_q.setdefault(int(cid), int(q))
fq = np.array([first_q[int(c)] for c in uniq]); pos = np.searchsorted(uniq, req.chunk_id.values)
qsets = req.groupby('q_idx').chunk_id.apply(lambda s: np.searchsorted(uniq, s.values)).to_dict()
cnt = {0.95: 0, 0.90: 0, 0.85: 0}
for st in range(0, len(req), 2000):
    S = U[pos[st:st + 2000]] @ U.T
    for r_ in range(S.shape[0]):
        i = st + r_; q = int(req.q_idx[i]); cid = int(req.chunk_id[i])
        if first_q[cid] < q: continue
        ok = fq < q; ok[qsets[q]] = False
        if not ok.any(): continue
        b = S[r_][ok].max()
        for t in cnt: cnt[t] += int(b >= t)
for t in cnt: v1[f'independent_upper_bound_approx_rate@{t}'] = cnt[t] / len(req)
g = sim_sum[(sim_sum.trace == 'distinct') & (sim_sum.cache == '100%_reqset') & (sim_sum.accounting == 'text_only')]
for t in ('0.95', '0.90', '0.85'):
    v1[f'simulation_gain@{t}'] = float(g[g.tau.astype(str).str.startswith(t[:4])].gain_vs_exact.iloc[0])
json.dump(v1, open(f'{O}/v1_impl.json', 'w'), indent=1)
log('V1 done')

# ---------- V2: 사실관계 판정 검증 ----------
need = set(ev.orig) | set(ev['sub']); text = {}
with open(f'{J}/chunks.jsonl', encoding='utf-8') as f:
    for cid, l in enumerate(f):
        if cid in need: text[cid] = json.loads(l)['text']
CIT = re.compile(r'\[\d+\]'); NUM = re.compile(r'\d+(?:[.,]\d+)*')
def nums(t): return set(NUM.findall(CIT.sub(' ', t)))
ev['num_mismatch_nocite'] = [bool(nums(text[o])) and not nums(text[o]) <= nums(text[s]) for o, s in zip(ev.orig, ev['sub'])]
ev['mismatch_only_from_citation'] = ev.num_mismatch & ~ev.num_mismatch_nocite
ev['nli_either'] = (ev.nli_sub_entails_orig >= 0.5) | (ev.nli_orig_entails_sub >= 0.5)
al = ev.answer_lost.astype(bool)
ev['ok_A_original'] = ev.factual_ok.astype(bool)
ev['ok_B_no_citation'] = (~al) & (~ev.num_mismatch_nocite) & (ev.nli_sub_entails_orig >= 0.5)
ev['ok_C_relaxed'] = (~al) & (~ev.num_mismatch_nocite) & ev.nli_either
ev['ok_D_nli_either_only'] = ev.nli_either
ev['sim_bin'] = pd.cut(ev.sim, [0.85, 0.90, 0.95, 1.01], right=False, labels=['0.85-0.90', '0.90-0.95', '>=0.95'])
v2 = ev.groupby('sim_bin', observed=True).agg(pairs=('sim', 'size'), num_mismatch=('num_mismatch', 'mean'), num_mismatch_nocite=('num_mismatch_nocite', 'mean'),
        citation_only=('mismatch_only_from_citation', 'mean'), ok_A_original=('ok_A_original', 'mean'), ok_B_no_citation=('ok_B_no_citation', 'mean'),
        ok_C_relaxed=('ok_C_relaxed', 'mean'), ok_D_nli_either=('ok_D_nli_either_only', 'mean'), contradiction=('nli_contradiction', lambda x: (x >= 0.5).mean()),
        same_doc=('same_doc', 'mean')).reset_index()
tot = ev[['num_mismatch', 'num_mismatch_nocite', 'ok_A_original', 'ok_B_no_citation', 'ok_C_relaxed', 'ok_D_nli_either_only']].mean().to_dict()
ev.to_csv(f'{O}/v2_pairs_rejudged.csv', index=False); v2.to_csv(f'{O}/v2_summary.csv', index=False)
show = pd.concat([ev[ev.sim >= 0.90].sort_values('sim', ascending=False), ev[(ev.sim < 0.90)].sample(min(8, int((ev.sim < 0.90).sum())), random_state=1)])
samples = []
for k, r in enumerate(show.itertuples(), 1):
    samples.append(f"[{k}] sim={r.sim:.3f} same_doc={r.same_doc} | NLI sub=>orig {r.nli_sub_entails_orig:.2f}, orig=>sub {r.nli_orig_entails_sub:.2f}, contra {r.nli_contradiction:.2f} "
                   f"| num_mismatch {r.num_mismatch}->{r.num_mismatch_nocite}(인용 제외) | ans {r.orig_has_ans}->{r.sub_has_ans} | ce {r.ce_orig:.2f}->{r.ce_sub:.2f}\n"
                   f"    Q   : {qs[int(r.q_idx)]}\n    원래: {text[int(r.orig)][:300]}\n    대체: {text[int(r.sub)][:300]}")
log('V2 done')

# ---------- V3: 임베딩과 무관한 글자 기반 근접 중복 ----------
K, STRIDE = 8, 8; us = set(int(x) for x in uniq); utext = {}; hs, hc = [], []
with open(f'{J}/chunks.jsonl', encoding='utf-8') as f:
    for cid, l in enumerate(f):
        t = json.loads(l)['text']; wd = t.lower().split()
        if cid in us: utext[cid] = wd
        for p in range(0, max(len(wd) - K + 1, 1), STRIDE):
            hs.append(hash(' '.join(wd[p:p + K]))); hc.append(cid)
hs = np.array(hs, np.int64); hc = np.array(hc, np.int64); o = np.argsort(hs, kind='stable'); hs = hs[o]; hc = hc[o]
log('corpus shingles', len(hs))
lex = []
for cid in uniq:
    cid = int(cid); wd = utext[cid]
    gr = np.array([hash(' '.join(wd[p:p + K])) for p in range(0, max(len(wd) - K + 1, 1))], np.int64)
    a = np.searchsorted(hs, gr, 'left'); b = np.searchsorted(hs, gr, 'right'); c = collections.Counter()
    for x, y in zip(a, b):
        if y > x: c.update(set(hc[x:y].tolist()))
    c.pop(cid, None)
    if c:
        oc, hits = c.most_common(1)[0]
        lex.append((cid, hits, oc, bool(docarr[oc] == docarr[cid]), float(np.dot(np.asarray(vecs[cid], np.float32), np.asarray(vecs[oc], np.float32)))))
    else:
        lex.append((cid, 0, -1, False, np.nan))
lex = pd.DataFrame(lex, columns=['chunk_id', 'max_shared_8gram_hits', 'partner', 'partner_same_doc', 'partner_emb_sim'])
lex.to_csv(f'{O}/v3_lexical_neardup.csv', index=False)
lw = req.merge(lex, on='chunk_id')
v3 = []
for h, desc in ((3, '>=24 words shared (~37% of chunk)'), (5, '>=40 words shared (~63%)')):
    m = lex.max_shared_8gram_hits >= h; mw = lw.max_shared_8gram_hits >= h
    v3.append(dict(threshold=desc, frac_unique_chunks=float(m.mean()), frac_requests=float(mw.mean()),
                   partner_same_doc=float(lex[m].partner_same_doc.mean()) if m.any() else np.nan,
                   partner_emb_sim_median=float(lex[m].partner_emb_sim.median()) if m.any() else np.nan,
                   frac_partner_emb_sim_ge_0_90=float((lex[m].partner_emb_sim >= 0.90).mean()) if m.any() else np.nan))
v3 = pd.DataFrame(v3); v3.to_csv(f'{O}/v3_summary.csv', index=False)
log('V3 done')

pd.set_option('display.width', 250)
print('\n================ ② 신뢰성 검증 결과 ================')
print('[V1 구현] (independent_upper_bound >= simulation_gain 이고 차이가 작으면 정상)')
for k, v in v1.items(): print(f'  {k}: {v:.4f}')
print('\n[V2 사실관계] ok_A=원래 기준, ok_B=인용번호 제외, ok_C=인용 제외+NLI 한쪽 방향, ok_D=NLI 한쪽 방향만')
print(v2.round(3).to_string(index=False))
print('  전체:', {k: round(v, 3) for k, v in tot.items()})
print('\n[V2 원문 대조 샘플] (유사도 0.90 이상 전부 + 0.85~0.90 무작위 8개)')
print('\n'.join(samples))
print('\n[V3 글자 기반 근접 중복] (임베딩과 무관; partner_emb_sim = 그 쌍의 MiniLM 유사도)')
print(v3.round(4).to_string(index=False))
print(f'\nsaved: {O}/ (v1_impl.json, v2_summary.csv, v2_pairs_rejudged.csv, v3_summary.csv, v3_lexical_neardup.csv)')
log('ALL DONE')
