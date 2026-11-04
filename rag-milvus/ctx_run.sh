#!/bin/bash
# Same-budget context-layer comparison: 2 backends x 2 budgets x 2 traces x 2 expansions x 4 policies x 3 reps
cd "$(dirname "$0")"
for rep in 1 2 3; do
 for be in local s3; do for b in 0.05 0.2; do for t in natural zipf; do for w in 0 1; do
  for p in cell2m chunk block group; do
   f=results/ctx-bench/rep$rep/$be-$p-b$b-$t-w$w.json
   [ -s "$f" ] && continue
   timeout 900 .venv/bin/python ctx_bench.py --policy $p --backend $be --budget $b --trace $t --w $w --out results/ctx-bench/rep$rep || echo "FAILED $f"
  done
 done; done; done; done
done
echo ALLDONE
