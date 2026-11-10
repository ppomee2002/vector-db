# M2: ANN vs Context reuse pattern and split cache (Qdrant, NQ)

Report page: `report/m2-report.html` (open in a browser).

## Method
- `qdrant-m2trace.patch` (on top of the M1 patch; applied by `scripts/patch_m2.py`): with `QDRANT_M2TRACE=<file>`, logs per-segment HNSW visits (vectors / links) and Gridstore payload page reads.
- 3,452 NQ queries collected at concurrency 1 (`m2_collect.py`), parsed into 4KB page accesses (`m2_parse.py`), exact rescoring reads added (`m2_rescore.py`).
- Exp 1 (`m2_exp1.py`): reuse frequency, top-k overlap, Spearman, Gini, reuse distance for ANN (traverse / rescore) vs Context.
- Exp 2 (`m2_sim.py`): same cache budget (1/2/5/10/20% of on-disk pages): unified LRU / LFU, split 50:50, best split (post-hoc upper bound), OPT.
  Scenarios S1 (graph + vectors on disk), S2 (quantized in RAM, links + originals on disk), S3 (only originals on disk).
  Traces: natural (repeated permutations), zipf 1.2, zipf 0.9; 12,000 requests (2,000 warm-up).

## Results (`results/`)
`exp1.json`, `sim-S1.json`, `sim-S2.json`, `sim-S3.json`, `layout.json`.
Raw trace (214MB) and `trace.npz` (576MB) are too large for git; they stay on server 109 at `~/gran/stats/m2t-nq.txt` and `~/gran/m2/nq/trace.npz`.

## Key findings
- Exp 1 supports the claim: ANN traversal is hub-skewed (Gini 0.53) while Context is flat (0.02); Spearman 0.05. Rescore reads look more like Context (Spearman 0.38).
- Exp 2 does not: best split (upper bound) beats unified LRU by at most 4.4% (LFU: 5.0%); 50:50 split reads a median 41% more; the best ratio gives >=90% to ANN in 25 of 27 tight-budget cases. OPT is up to 56% better than unified LRU, and the gap is inside ANN.
