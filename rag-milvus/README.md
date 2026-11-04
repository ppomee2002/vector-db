# RAG Context Read on Milvus and Qdrant

Checks whether real VectorDBs show RAG context-read inefficiencies: (1) duplicate
context reads, (2) gap between the engine's cache and a chunk-granular LRU/Belady cache
at equal capacity, (3) share of query cost spent on context reads, (4) read amplification
(bytes read from storage / text bytes returned).

Engines: Milvus v2.6.23 standalone (tiered storage, config C) and Qdrant v1.20.0
(`on_disk_payload`, page cache capped by the container memory limit). Datasets: BEIR
FiQA (57,638 chunks, 44 MB text) and NQ (2,681,468 chunks, 1.3 GB text). Containers run
with k3s containerd (`k3s ctr`), host network.

| step | script |
|---|---|
| download BEIR (HF parquet mirror) + QASPER | `datasets/raw/download.sh` (copy `download.sh` there) |
| CPU embedding, all cores (all-MiniLM-L6-v2 int8 ONNX, max_len 128) | `embed.py --datasets scifact,fiqa,nq` |
| Milvus: start config A (in-memory) / B (mmap) / C (tiered storage) | `milvus_ctl.sh start C` (`MILVUS_STORAGE=local` for local-disk storage) |
| Milvus: create + load collection (pk, doc_id, text, vec HNSW IP) | `load.py --dataset fiqa` |
| Milvus: closed-loop search load + server counter deltas | `bench.py` |
| Milvus: text-cache budget sweep with S3 GET capture | `run_trend.sh <dataset> <out>` |
| Qdrant: start / load / bench / budget sweep | `qdrant_ctl.sh`, `qdrant_load.py`, `qdrant_bench.py`, `qdrant_trend.sh` |
| metrics table (TREND.md, analysis.json) | `analyze.py <out>` |
| Milvus config B (mmap) budget sweep, page cache capped by cgroup `memory.max` | `milvus_b_trend.sh <dataset> <out>` |
| what-if replay: cache granularity, neighbour-chunk expansion, cross-query overlap | `context_sim.py`, `overlap_distinct.py` |
| ANN search access vs context access on the same queries (faiss HNSW replay) | `ann_vs_context.py --datasets fiqa,nq` |
| run order used | `chain.sh`, `chain2.sh`, `chain3.sh` (main sweeps); `chain4.sh`–`chain6.sh` (extra runs); `watchdog.sh` restarts a hung S3 |

Milvus config C cache budget is set through
`queryNode.segcore.tieredStorage.memory{High,Low}WatermarkRatio`; eviction needs both
`evictionEnabled` and `backgroundEvictionEnabled`. Qdrant budget = memory limit of
`M0 * 1.1 + budget * payload_disk_bytes`.

## Results (`results/`)

`results/<run>/TREND.md` holds the tables, `*.json` the per-case summaries. Per-query
`*.jsonl` and the S3 packet captures are not committed (26 MB).

Read amplification (storage bytes read / text bytes returned):

| text cache | Milvus FiQA | Milvus NQ | Qdrant FiQA | Qdrant NQ |
|---|---|---|---|---|
| 100% | 0 | 0 | 0 | 0 |
| 50% | 2.6–75x | 5–162x | 0–2.1x | 0–7x |
| 20% | 10–20x | 289–637x | 0.2–3.7x | 0–7x |

- **Milvus.** When the text cache does not fit, it reads up to 640–1,485x more than a
  chunk-granular Belady cache of the same size (NQ, 20%). With text included, p99 rises to
  0.5–3 s; without text it is about 130 ms. It caches and evicts text in cells of about
  2 MB, while a chunk is about 770 B.
- **Qdrant.** Amplification stays low. With text included, QPS and p99 are within 1–5% of
  the runs without text. Our explanation, not tested separately: the OS page cache works
  in 4 KB pages, which is close to chunk size.

### Extra runs

- **Same engine, cache type only (NQ, `trend-nq-mmap` vs `trend-nq`, `trend-nq-low`).** At a 5% budget, Milvus
  tiered (C) reads 695–1,322x the returned text, p99 is 1.0–1.2 s at C=32, and QPS with text is 139 vs 282
  without text. Milvus mmap (B) reads 17–99x the returned text, p99 is 120–150 ms, and QPS is 400 vs 408.
  The cache-drop between B cases is incomplete (up to 434 MB stayed resident), so B amplification varies
  between runs. `trend-nq-mmap-invalid` is a first run whose memory limit covered all of the text.
- **Qdrant NQ at 1–5% (`qdrant-nq-low`).** Amplification is 1.6–19x. QPS with and without text is about equal.
- **What-if replay (`context-sim`).** At the same budget, LRU at 4 KB / 64 KB / 2 MB blocks reads about 5x /
  40–130x / 600–7,000x more than at chunk granularity. Expanding ±1–2 neighbour chunks and merging within a
  query saves 20–29%. Different queries rarely need the same chunk at the same time (0–2.5%). The overlap seen
  under zipf comes from the same query repeating.
- **ANN vs context access (`ann-vs-context`, NQ).** A query reads 1,734 KB in HNSW but 5.1 KB of context.
  43% of the HNSW nodes a query reads were already read by an earlier query, against 1.3% for context chunks.
  LRU hit ratio with 10% of items cached is 22% vs 1.3%. With 2 MB blocks, the bytes touched are 635x what HNSW
  needs and 1,607x what context needs.

Caveats:

- Each case ran once.
- Milvus FiQA at 100/50/20% ran while the embedding job used the CPUs, so QPS is not
  comparable across budgets.
- Milvus NQ used local-disk storage because SeaweedFS stalled on multipart uploads at NQ
  scale (see `results/fixes.log`), so NQ has no S3 GET data.
- At 20% budget the Qdrant NQ working set (2,000 queries) still fits in the cache.
- The Milvus NQ 100% natural c32 row is the cold first load.
