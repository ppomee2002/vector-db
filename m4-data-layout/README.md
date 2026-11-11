# M4: Poor data locality of RAG context (Qdrant, Milvus, NQ)

Chunks of the same document are read together (top-k hits, neighbor chunks), but both DBs scatter them on disk.
Storing them contiguously cut per-query reads at Context Expansion ±1 on NQ by **58% (Qdrant data pages)** and **82% (Milvus bytes)**.

## Method
- Qdrant 1.20.0 (own process, port 6343, vectors in RAM, `on_disk_payload=True`); Milvus 2.6.23 (team server, collection property `warmup.scalarField=disable`).
- Data: BEIR NQ 150k chunks (5,348 whole Wikipedia docs, 28.0 chunks/doc) and BEIR HotpotQA 150k chunks (1.07 chunks/doc); 2,000 queries, 70/30 train/test split.
- Exact top-10 precomputed (MiniLM), so every layout reads the same chunks; only the on-disk placement changes.
- Context Expansion: ±w = top-10 + up to w chunks before/after each hit within the same document.
- Cold read per query. Qdrant: `posix_fadvise(DONTNEED)` on our files only, page count by `mincore` before/after, data pages (`page_*.dat`) and tracker counted separately, Qdrant restarted per run, 500 test queries. Milvus: release/load per query, cell miss bytes from `:9091` metrics + latency, 200 test queries.
- Paired comparison per query with bootstrap 95% CI. No `drop_caches` or sudo on the shared server.

## Layouts
`interleaved` random insert · `doc` document-order insert (default: 8 segments, batch 256) · `doc_1seg` 1 segment · `doc_b1` batch 1 · `doc_1seg_b1`, `doc_1seg_b16` both · `stream8` 8 documents interleaved · `coaccess` placed by train-query co-access · `oracle` upper bound using test queries · `doc_rpk` (Milvus) document-order insert with random PK.

## Scripts (`scripts/`)
- `setup.sh` venv + Qdrant binary; `qdrant_ctl.sh` start/stop our Qdrant only.
- `prep_nq.py`, `prep_beir.py` chunking, embedding, exact top-10, split; `common.py` data loading + expansion.
- `load.py` Qdrant load per layout (`_1seg`, `_b<N>` suffixes); `measure.py` per-query pages / bytes / latency; `qps.py` concurrent QPS.
- `locate.py` find the segment / page each chunk is stored in (read-only); `sim_pages.py` ideal contiguous page count.
- `milvus_load.py`, `milvus_measure.py`, `milvus_common.py` Milvus; `milvus_check*.py` probes.
- `analyze.py`, `analyze2.py` summary tables and paired CIs; `make_figs.py` early figures.
- `run_all.sh`, `run_extra.sh`, `run_important.sh`, `run_final.sh`, `run_milvus.sh` one experiment each (table below).

## Results (`results/`)
| dir | script | experiment |
|---|---|---|
| results | run_all.sh | Exp 1: interleaved vs doc vs coaccess, w=0/1 |
| results2 | run_extra.sh | Exp 2-3: + stream8, doc_1seg, oracle, w=0/1/2; QPS (C=1/16, 3 reps) |
| results3 | run_important.sh | Exp 4: data pages vs tracker pages; ideal-layout simulation |
| results_hotpot | run_important.sh | Exp 5: HotpotQA |
| results4 | run_final.sh | Exp 6: segment count x batch size, `locate.txt` |
| results_milvus | run_milvus.sh | Exp 7: Milvus interleaved / doc / doc_rpk |

`summary.jsonl` = per-layout means, `summary*.txt` = tables with paired CIs, `sim*.txt/csv` = ideal vs measured, `qps.jsonl` = QPS runs.
Per-query CSVs (`q_*.csv`, `mq_*.csv`, ~110 files) are not included; they stay in `~/m4-exp/results*/` on the server.

## Key findings
- Qdrant (NQ, ±1, data pages/query): interleaved 17.2 → doc 14.2 → doc_1seg 14.0, doc_b1 12.3, doc_1seg_b16 8.35 → **doc_1seg_b1 5.9 (−58% vs doc)**.
  Cause: round-robin over 8 segments + reordering inside a batch; one document's 12 consecutive chunks sat in 6+ segments, 6-9 pages.
- Milvus (±1): interleaved 52.9 MB, doc_rpk 51.9 MB, **per-document contiguous PK 9.5 MB (−82%)**; median latency 192 → 59 ms (−69%). Milvus stores by PK order, not insert order; ±0 and ±1 both read 9.5 MB.
- HotpotQA: −1.7% (top-10 spans 9.7 docs vs 2.5 on NQ). Query-log placement (coaccess) adds little over document order on NQ (8.5% of test hits seen in train).
- No consistent QPS difference on the fast local disk (Qdrant). `doc_1seg_b1` is a one-off diagnostic, not a practical load path.
