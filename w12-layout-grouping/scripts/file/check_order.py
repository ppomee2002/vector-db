import json, sys, random, collections
random.seed(0)
path = sys.argv[1]
n = id_mismatch = 0
runs = []; cur_title = None; cur = []
title_runs = collections.Counter()
def close():
    if cur: runs.append(cur[:]); title_runs[cur_title] += 1
with open(path) as f:
    for line in f:
        d = json.loads(line)
        if d["_id"] != f"doc{n}": id_mismatch += 1
        if d["title"] != cur_title:
            close(); cur_title = d["title"]; cur = []
        cur.append(len(d["text"].split())); n += 1
close()
multi = [r for r in runs if len(r) >= 2]
ok = sum(all(w == 100 for w in r[:-1]) for r in multi)
sizes = collections.Counter(min(len(r), 10) for r in runs)
print(f"passage 수: {n:,}")
print(f"_id가 doc0,doc1.. 순서와 다른 것: {id_mismatch:,}")
print(f"제목 수: {len(title_runs):,} / 연속 구간 수: {len(runs):,}  (같으면 같은 제목 passage가 모두 붙어 있음)")
print(f"passage 2개 이상인 문서: {len(multi):,} ({len(multi)/len(runs):.1%})")
print(f"  그중 '마지막 빼고 전부 100단어'인 문서: {ok:,} ({ok/max(len(multi),1):.1%})  <- 높을수록 원래 순서·누락 없음")
print("문서당 passage 수 분포(10=10개 이상):", dict(sorted(sizes.items())))
