# q1_overlap.py: ① Document-based Range Read 축소 확인 (텍스트 바이트 계산만; 디스크 I/O·latency·Milvus·GPU 없음)
# (가) 재분할 chunk 개별 연속 저장 + 겹치는 chunk 모두 읽기(연속 chunk는 한 번에)
# (나) 원문 1회 저장 + 필요 구간 합집합만 읽기
# 바이트 모델: UTF-8, 단어 사이 공백 1바이트. 원소 = 단어(바이트=len) + 단어 사이 간격(gap k: 단어 k와 k+1 사이, 1바이트).
#   구간 [a,b)는 단어 a..b-1 과 gap a..b-2 를 포함. chunk 텍스트 = " ".join(단어) 이므로 같은 규칙.
import os, sys, json, time, math, resource
import numpy as np, pandas as pd

J = '/workspace/junlim/data_nq/chunks.jsonl'
BASE = '/workspace/jaeeun/sel_fetch'
OUT = f'{BASE}/results/q1_overlap'
os.makedirs(OUT, exist_ok=True)
WIN, MIN_TAIL, PAGE = 64, 32, 4096
OVERLAPS = {'0pct': 0, '10pct': 6, '20pct': 13}
EXPS = (0, 1)
t0 = time.time()
def log(*x): print(f'[{time.time()-t0:6.0f}s]', *x, flush=True)
def maxrss_gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

# ---------------------------------------------------------------- 쿼리별 최종 Context
meta = np.load(f'{BASE}/data/chunk_meta.npz')
doc_of, cix_of, nb_of = meta['doc_idx'], meta['chunk_idx'], meta['nbytes']
real = pd.read_csv(f'{BASE}/data/rerank_top100_real.csv', usecols=['q_idx', 'chunk_id', 'ce_rank'])
real = real[real.ce_rank < 10]
trs = pd.read_csv(f'{BASE}/results/q4_full/rerank_sample.csv', usecols=['q_row', 'chunk_id', 'ce_rank'])
trs = trs[trs.ce_rank < 10]
QUERIES = {'real': real.groupby('q_idx').chunk_id.apply(list).to_dict(),
           'train_sample': trs.groupby('q_row').chunk_id.apply(list).to_dict()}
for t, d in QUERIES.items():
    assert all(len(v) == 10 for v in d.values()), t
# doc -> [(target, q, [chunk_idx...])]
NEED = {}
for t, d in QUERIES.items():
    for q, cids in d.items():
        per = {}
        for c in cids: per.setdefault(int(doc_of[c]), []).append(int(cix_of[c]))
        for dd, lst in per.items(): NEED.setdefault(dd, []).append((t, q, lst))
log(f'queries: real {len(QUERIES["real"])}, train_sample {len(QUERIES["train_sample"])} | docs touched {len(NEED)}')

# ---------------------------------------------------------------- 도우미
def iter_docs():
    cur, buf = None, []
    with open(J, encoding='utf-8') as f:
        for cid, l in enumerate(f):
            r = json.loads(l)
            if r['doc_idx'] != cur:
                if buf: yield cur, buf
                cur, buf = r['doc_idx'], []
            buf.append((cid, r['chunk_idx'], r['text']))
    if buf: yield cur, buf

def resplit(L, ov):
    s = WIN - ov; spans = []; st = 0
    while True:
        e = min(st + WIN, L); spans.append([st, e])
        if e == L: break
        st += s
    if len(spans) >= 2 and spans[-1][1] - spans[-1][0] < MIN_TAIL:
        spans.pop(); spans[-1][1] = L
    return np.array(spans, np.int64)

def merge(iv):
    iv = sorted(iv); out = []
    for a, b in iv:
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out

class Doc:
    def __init__(s, chunks):
        s.words, s.orig = [], []
        for _, _, text in chunks:
            w = text.split(); s.orig.append((len(s.words), len(s.words) + len(w))); s.words += w
        s.L = len(s.words)
        s.wl = np.array([len(w.encode('utf-8')) for w in s.words], np.int64)
        s.P = np.concatenate([[0], np.cumsum(s.wl)])
        s.orig = np.array(s.orig, np.int64)
        s.splits = {k: (s.orig if ov == 0 else resplit(s.L, ov)) for k, ov in OVERLAPS.items()}
    def sb(s, a, b): return int(s.P[b] - s.P[a] + (b - a - 1)) if b > a else 0

def mult(spans, lo, hi):
    """구간 목록의 단어/gap 다중도 (로컬 [lo,hi))"""
    n = hi - lo; dw = np.zeros(n + 1, np.int64); dg = np.zeros(max(n, 1), np.int64)
    for a, b in spans:
        dw[a - lo] += 1; dw[b - lo] -= 1
        if b - a >= 2: dg[a - lo] += 1; dg[b - 1 - lo] -= 1
    return np.cumsum(dw)[:n], np.cumsum(dg)[:max(n - 1, 0)]

def query_doc(D, cidx, exp, key):
    """한 쿼리·한 문서·한 조건의 (가)/(나) 계산. 반환: dict(지표), 검증용 플래그"""
    n = len(D.orig); need = set()
    for c in cidx:
        for d in range(-exp, exp + 1):
            if 0 <= c + d < n: need.add(c + d)
    raw = [tuple(D.orig[c]) for c in sorted(need)]
    N = merge([list(x) for x in raw])
    # (나)
    nb_b = [D.sb(a, b) for a, b in N]
    r_na = dict(bytes=sum(nb_b), reads=len(N), pages=sum(math.ceil(x / PAGE) for x in nb_b))
    # (가)
    S = D.splits[key]; st, en = S[:, 0], S[:, 1]
    idx = set()
    for x, y in N:
        j0 = int(np.searchsorted(en, x, side='right')); j1 = int(np.searchsorted(st, y, side='left'))
        idx.update(range(j0, j1))
    idx = sorted(idx)
    runs, cur = [], [idx[0]]
    for j in idx[1:]:
        if j == cur[-1] + 1: cur.append(j)
        else: runs.append(cur); cur = [j]
    runs.append(cur)
    cb = {j: D.sb(int(st[j]), int(en[j])) for j in idx}
    rb = [sum(cb[j] for j in r) for r in runs]
    r_ga = dict(bytes=sum(rb), reads=len(runs), pages=sum(math.ceil(x / PAGE) for x in rb))
    # 분해 (로컬 배열)
    lo = min(int(st[idx[0]]), N[0][0]); hi = max(int(en[idx[-1]]), N[-1][1])
    mw, mg = mult([(int(st[j]), int(en[j])) for j in idx], lo, hi)
    nw, ng = mult(N, lo, hi)
    rw, _ = mult(raw, lo, hi)                     # 필요 chunk 구간(병합 전)의 단어 집합
    wl = D.wl[lo:hi]
    dup = int(((np.maximum(mw - 1, 0)) * wl).sum() + np.maximum(mg - 1, 0).sum())
    excess = int((wl * ((mw > 0) & (nw == 0))).sum() + ((mg > 0) & (ng == 0)).sum())
    gapmiss = int(((ng > 0) & (mg == 0)).sum())
    v3_ga_missing = int(((nw > 0) & (mw == 0)).sum())
    v3_na_mismatch = int(((nw > 0) != (rw > 0)).sum()) + int((nw > 1).sum())
    ident = (r_ga['bytes'] - r_na['bytes']) == dup + excess - gapmiss
    # 연속 읽기 구간(N) 안에 있는 원래 chunk 경계 수 (경계마다 공백 1바이트)
    ost = D.orig[:, 0]
    n_bound = int(sum(((ost > x) & (ost < y)).sum() for x, y in N))
    return r_ga, r_na, dict(dup=dup, excess=excess, gapmiss=gapmiss), dict(
        v3_ga_missing=v3_ga_missing, v3_na_mismatch=v3_na_mismatch, ident_ok=bool(ident), n_bound=n_bound)

# ---------------------------------------------------------------- 1단계: 검증
V = {'V1_word_count': 0, 'V1_text_roundtrip': 0, 'V1_chunk_bytes_vs_meta': 0, 'V1_order': 0,
     'V2_cover_0pct': 0, 'V2_cover_10pct': 0, 'V2_cover_20pct': 0, 'V2_tail_rule': 0, 'V2_resplit0_matches_orig': 0,
     'V3_ga_missing_words': 0, 'V3_na_not_exact': 0, 'V4_0pct_dup_nonzero': 0, 'V4_0pct_excess_nonzero': 0,
     'V4_0pct_diff_not_boundary_spaces': 0, 'decomp_identity': 0}
EX = {k: [] for k in V}
def bad(k, info):
    V[k] += 1
    if len(EX[k]) < 5: EX[k].append(info)

log('== validation pass')
n_docs = n_qd = n_v4 = n_v4_nonzero = 0
for dd, chunks in iter_docs():
    n_docs += 1
    if [c[1] for c in chunks] != list(range(len(chunks))): bad('V1_order', dict(doc=dd))
    D = Doc(chunks)
    if D.L != sum(len(c[2].split()) for c in chunks): bad('V1_word_count', dict(doc=dd))
    for (cid, ci, text), (a, b) in zip(chunks, D.orig):
        if text != ' '.join(text.split()): bad('V1_text_roundtrip', dict(doc=dd, chunk_id=cid))
        if D.sb(a, b) != len(text.encode('utf-8')) or D.sb(a, b) != int(nb_of[cid]):
            bad('V1_chunk_bytes_vs_meta', dict(doc=dd, chunk_id=cid, model=D.sb(a, b), meta=int(nb_of[cid])))
    if not np.array_equal(resplit(D.L, 0), D.orig): bad('V2_resplit0_matches_orig', dict(doc=dd, L=D.L))
    for k, S in D.splits.items():
        ok = S[0, 0] == 0 and S[-1, 1] == D.L and (S[:, 0] >= 0).all() and (S[:, 1] <= D.L).all() \
             and (S[:, 1] > S[:, 0]).all() and (S[1:, 0] <= S[:-1, 1]).all() and (S[1:, 0] > S[:-1, 0]).all()
        if not ok: bad(f'V2_cover_{k}', dict(doc=dd, L=D.L, first=S[0].tolist(), last=S[-1].tolist()))
        if len(S) >= 2 and S[-1, 1] - S[-1, 0] < MIN_TAIL: bad('V2_tail_rule', dict(doc=dd, ov=k, last=S[-1].tolist()))
    for t, q, cidx in NEED.get(dd, []):
        n_qd += 1
        for k in OVERLAPS:
            for e in EXPS:
                ga, na, dc, vf = query_doc(D, cidx, e, k)
                info = dict(target=t, q=int(q), doc=dd, overlap=k, exp=e, chunks=cidx, ga=ga['bytes'], na=na['bytes'], **dc)
                if vf['v3_ga_missing']: bad('V3_ga_missing_words', info)
                if vf['v3_na_mismatch']: bad('V3_na_not_exact', info)
                if not vf['ident_ok']: bad('decomp_identity', info)
                if k == '0pct':
                    n_v4 += 1
                    if dc['dup'] != 0: bad('V4_0pct_dup_nonzero', info)
                    if dc['excess'] != 0: bad('V4_0pct_excess_nonzero', info)
                    if not (ga['bytes'] - na['bytes'] == -dc['gapmiss'] and dc['gapmiss'] == vf['n_bound']):
                        bad('V4_0pct_diff_not_boundary_spaces', dict(info, n_bound=vf['n_bound']))
                    n_v4_nonzero += int(ga['bytes'] != na['bytes'])
log(f'validation done: docs {n_docs}, query-doc pairs {n_qd} | V4 checked {n_v4}, of which ga!=na (boundary spaces only) {n_v4_nonzero} | maxRSS {maxrss_gb():.3f} GB')
vdf = pd.DataFrame([dict(check=k, violations=v, examples=json.dumps(EX[k], default=int)) for k, v in V.items()])
vdf.to_csv(f'{OUT}/validation.csv', index=False)
print('\n==== validation ====')
for k, v in V.items(): print(f'{k:28s} {v}')
if any(V.values()):
    print('\n==== violation examples ====')
    for k, v in V.items():
        if v:
            print(k)
            for e in EX[k]: print('   ', json.dumps(e, default=int))
    log('STOP: validation violations -> metrics not computed')
    sys.exit(2)

# ---------------------------------------------------------------- 2단계: 지표
log('== metrics pass')
store = {k: dict(ga_bytes=0, na_bytes=0, n_chunks=0, dup=0, gapmiss=0) for k in OVERLAPS}
rows = {}
for dd, chunks in iter_docs():
    D = Doc(chunks)
    nab = D.sb(0, D.L)
    for k, S in D.splits.items():
        mw, mg = mult([(int(a), int(b)) for a, b in S], 0, D.L)
        s_ = store[k]
        s_['ga_bytes'] += sum(D.sb(int(a), int(b)) for a, b in S); s_['na_bytes'] += nab; s_['n_chunks'] += len(S)
        s_['dup'] += int((np.maximum(mw - 1, 0) * D.wl).sum() + np.maximum(mg - 1, 0).sum())
        s_['gapmiss'] += int((mg == 0).sum())
    for t, q, cidx in NEED.get(dd, []):
        for k in OVERLAPS:
            for e in EXPS:
                ga, na, dc, _ = query_doc(D, cidx, e, k)
                r = rows.setdefault((t, q, k, e), dict(target=t, q=int(q), overlap=k, exp=e, n_docs=0,
                    ga_bytes=0, na_bytes=0, dup=0, excess=0, gapmiss=0, ga_reads=0, na_reads=0, ga_pages=0, na_pages=0))
                r['n_docs'] += 1
                r['ga_bytes'] += ga['bytes']; r['na_bytes'] += na['bytes']
                r['ga_reads'] += ga['reads']; r['na_reads'] += na['reads']
                r['ga_pages'] += ga['pages']; r['na_pages'] += na['pages']
                for x in ('dup', 'excess', 'gapmiss'): r[x] += dc[x]
log(f'metrics pass done | maxRSS {maxrss_gb():.3f} GB')

st = pd.DataFrame([dict(overlap=k, overlap_words=OVERLAPS[k], **v) for k, v in store.items()])
st['ga_minus_na'] = st.ga_bytes - st.na_bytes
st['ga_over_na'] = st.ga_bytes / st.na_bytes
st.to_csv(f'{OUT}/storage.csv', index=False)

pq = pd.DataFrame(list(rows.values())).sort_values(['target', 'overlap', 'exp', 'q'])
pq['diff'] = pq.ga_bytes - pq.na_bytes
pq.to_csv(f'{OUT}/per_query.csv', index=False)
agg = []
for (t, k, e), g in pq.groupby(['target', 'overlap', 'exp']):
    d = dict(target=t, overlap=k, exp=e, n_queries=len(g))
    for c in ['ga_bytes', 'na_bytes', 'diff', 'dup', 'excess', 'gapmiss', 'ga_reads', 'na_reads', 'ga_pages', 'na_pages']:
        d[f'{c}_mean'] = float(g[c].mean()); d[f'{c}_total'] = int(g[c].sum())
    d['bytes_increase_total'] = (g.ga_bytes.sum() - g.na_bytes.sum()) / g.na_bytes.sum()
    d['bytes_increase_meanratio'] = float(((g.ga_bytes - g.na_bytes) / g.na_bytes).mean())
    d['reads_increase_total'] = (g.ga_reads.sum() - g.na_reads.sum()) / g.na_reads.sum()
    d['pages_increase_total'] = (g.ga_pages.sum() - g.na_pages.sum()) / g.na_pages.sum()
    agg.append(d)
agg = pd.DataFrame(agg); agg.to_csv(f'{OUT}/summary.csv', index=False)
json.dump(dict(window=WIN, min_tail=MIN_TAIL, overlaps_words=OVERLAPS, page_bytes=PAGE, expansions=EXPS,
               byte_model='UTF-8; words joined by single spaces; elements = words + inter-word gaps (1 byte)',
               decomposition='ga - na = dup (elements read more than once, extra copies) + excess (elements read outside needed range) - gapmiss (gaps inside needed range not contained in any read chunk, e.g. chunk-boundary spaces)',
               pages='per contiguous read: ceil(bytes/4096), summed (ideal, alignment ignored)',
               increase='(ga - na) / na', max_rss_gb=maxrss_gb(), runtime_s=time.time() - t0),
          open(f'{OUT}/config.json', 'w'), indent=1)
pd.set_option('display.width', 250)
print('\n==== storage ====\n' + st.to_string(index=False))
print('\n==== summary ====\n' + agg.round(4).T.to_string())
log('ALL DONE')
