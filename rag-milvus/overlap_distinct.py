#!/usr/bin/env python3
"""Cross-query overlap split: repeats of the same query vs different queries sharing chunks/blocks."""
import collections, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_sim import Corpus, GRAN

R = Path(__file__).resolve().parent / "results"
for ds, run in (("fiqa", "trend-fiqa"), ("nq", "trend-nq")):
    C = Corpus(ds)
    for t in ("natural", "zipf"):
        rows = sorted((json.loads(l) for l in open(R / run / f"{ds}-C-b0.5-{t}-c32.jsonl")), key=lambda r: r["i"])
        for g in ("chunk", "4KB"):
            for n in (32, 128):
                tot = dup_all = dup_distinct = 0
                for s in range(0, len(rows) - n + 1, n):
                    win = rows[s:s + n]
                    cnt = collections.Counter(); seen_q = {}
                    for r in win:
                        cnt.update(C.blocks(r["pks"], GRAN[g]))
                        seen_q.setdefault(r["q"], r["pks"])
                    cd = collections.Counter()
                    for pks in seen_q.values():
                        cd.update(C.blocks(pks, GRAN[g]))
                    for b, k in cnt.items():
                        sz = C.bsize(b, GRAN[g]); tot += k * sz; dup_all += (k - 1) * sz
                    for b, k in cd.items():
                        dup_distinct += (k - 1) * C.bsize(b, GRAN[g])
                print(f"{ds} {t} {g} N={n}: overlap all {dup_all/tot:.1%}, from different queries {dup_distinct/tot:.1%}, from repeated identical queries {(dup_all-dup_distinct)/tot:.1%}")
