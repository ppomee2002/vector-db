import json, os, itertools, re
J = '/workspace/junlim/data_nq'; H = '/workspace/jihyeon/data/nq'

def head(p, n=2):
    with open(p, encoding='utf-8') as f:
        for line in itertools.islice(f, n):
            print('   ', line.strip()[:300])

print('== junlim chunks.jsonl'); head(f'{J}/chunks.jsonl')
print('== junlim queries.jsonl'); head(f'{J}/queries.jsonl')
n = 0; keys = set(); kinds = {}
with open(f'{J}/queries.jsonl', encoding='utf-8') as f:
    for line in f:
        q = json.loads(line); n += 1; keys |= set(q)
        for k in ('kind', 'type', 'real', 'is_real', 'source', 'split'):
            if k in q:
                kinds[(k, str(q[k]))] = kinds.get((k, str(q[k])), 0) + 1
print('   n =', n, '| keys =', sorted(keys), '| kinds =', kinds)

print('== junlim prep_nq.py (qrel/real 관련 줄)')
with open('/workspace/junlim/prep_nq.py', encoding='utf-8') as f:
    for i, l in enumerate(f, 1):
        if re.search(r'qrel|relev|gold|answer|real|pseudo', l, re.I):
            print(f'   {i}: {l.rstrip()[:160]}')

print('== jihyeon qrels')
for fn in sorted(os.listdir(f'{H}/qrels')):
    p = f'{H}/qrels/{fn}'; print('  ', fn, os.path.getsize(p), 'bytes'); head(p, 3)
print('== jihyeon queries.jsonl'); head(f'{H}/queries.jsonl', 2)
print('== jihyeon corpus.jsonl'); head(f'{H}/corpus.jsonl', 1)

from pymilvus import MilvusClient
c = MilvusClient(uri='http://localhost:19530')
for db in c.list_databases():
    cc = MilvusClient(uri='http://localhost:19530', db_name=db)
    for col in cc.list_collections():
        rows = cc.get_collection_stats(col).get('row_count')
        st = cc.get_load_state(col).get('state')
        print(f'== milvus db={db} col={col} rows={rows} state={st}')
    cc.close()
c.close()
