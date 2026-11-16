# build_qrels.py: 임준 data_nq(100만 chunk)에 맞춰 NQ test 정답을 chunk 단위로 재구성 (입력은 읽기만)
import json, time, collections
J = '/workspace/junlim/data_nq'; H = '/workspace/jihyeon/data/nq'; OUT = '/workspace/jaeeun/sel_fetch/data'
NDOC = json.load(open(f'{J}/meta.json'))['n_docs']
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:5.0f}s]', *a, flush=True)

# 1) 쿼리 대응: 임준 real 쿼리 i -> 지현 test id
jq = [json.loads(l)['text'].strip() for l in open(f'{J}/queries.jsonl', encoding='utf-8')][:3452]
hl = [json.loads(l) for l in open(f'{H}/queries.jsonl', encoding='utf-8')]
if len(hl) >= len(jq) and all(hl[i]['text'].strip() == jq[i] for i in range(len(jq))):
    qid_of = [hl[i]['_id'] for i in range(len(jq))]; log('query order: identical')
else:
    m = {}
    for r in hl: m.setdefault(r['text'].strip(), r['_id'])
    qid_of = [m.get(t) for t in jq]; log('query order: by text, matched', sum(x is not None for x in qid_of))

# 2) qrels 읽기
qrels = collections.defaultdict(list)
with open(f'{H}/qrels/test.tsv', encoding='utf-8') as f:
    next(f)
    for l in f:
        a = l.rstrip('\n').split('\t')
        if len(a) >= 3 and int(a[2]) > 0: qrels[a[0]].append(a[1])
need = {d for v in qrels.values() for d in v}
log('qrels queries', len(qrels), '| relevant passages', len(need))

# 3) corpus 스트리밍: 임준 코드와 같은 규칙(빈 본문 제외, 제목 첫 등장 순서)으로 doc_idx 재현
title_idx, info, row = {}, {}, 0
with open(f'{H}/corpus.jsonl', encoding='utf-8') as f:
    for l in f:
        r = json.loads(l); txt = (r.get('text') or '').strip()
        if not txt: continue
        key = r.get('title') or f'__no_title_{row}'
        if key not in title_idx: title_idx[key] = len(title_idx)
        if r['_id'] in need: info[r['_id']] = (title_idx[key], txt)
        row += 1
log('corpus rows', row, '| titles', len(title_idx), '| qrel passages found', len(info))

# 4) 필요한 문서의 chunk만 로드
want = {d for d, _ in info.values() if d < NDOC}
docs = collections.defaultdict(list)
with open(f'{J}/chunks.jsonl', encoding='utf-8') as f:
    for cid, l in enumerate(f):
        r = json.loads(l)
        if r['doc_idx'] in want: docs[r['doc_idx']].append((cid, r['text']))
log('docs needed', len(want), '| loaded', len(docs))

def locate(d, txt):
    ch = docs.get(d)
    if not ch: return []
    starts, s = [], 0
    for _, t in ch:
        starts.append(s); s += len(t.split())
    words = ' '.join(t for _, t in ch)
    w = txt.split()
    pos = (' ' + words + ' ').find(' ' + ' '.join(w[:8]) + ' ')
    if pos < 0: return []
    a = words[:pos].count(' ')
    b = a + len(w) - 1
    return [cid for (cid, _), st, nx in zip(ch, starts, starts[1:] + [s]) if st <= b and nx > a]

# 5) chunk 단위 정답 저장
stat = collections.Counter(); covered = 0
with open(f'{OUT}/qrels_chunk.tsv', 'w') as out:
    out.write('q_idx\tqid\tcorpus_id\tdoc_idx\tchunk_id\n')
    for qi, qid in enumerate(qid_of):
        hit = False
        for pid in qrels.get(qid, []):
            if pid not in info: stat['passage_missing'] += 1; continue
            d, txt = info[pid]
            if d >= NDOC: stat['doc_out_of_subset'] += 1; continue
            cs = locate(d, txt)
            if not cs: stat['text_not_found'] += 1; continue
            stat['matched'] += 1; hit = True
            for c in cs: out.write(f'{qi}\t{qid}\t{pid}\t{d}\t{c}\n')
        covered += hit
log('passage stats', dict(stat))
log(f'queries with >=1 relevant chunk: {covered}/{len(qid_of)} ({covered/len(qid_of):.1%})')
log('saved', f'{OUT}/qrels_chunk.tsv')
