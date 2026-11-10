# M1: read granularity trade-off (Qdrant, Milvus)

Context(payload) read size vs QPS / I/O on NQ (and FiQA), measured on server 109.
Report page: `report/gran-report.html` (open in a browser).

## Source patches
- `qdrant-read-granularity.patch`: Qdrant v1.20.0 Gridstore payload reads widened to `QDRANT_GRAN` bytes (io_uring, readahead off), counters to `QDRANT_GRAN_STATS`.
- `milvus-cell-size.patch`: Milvus tiered storage cell = N Parquet row groups (min 1 RG).

## Scripts (`scripts/`)
- `qgran_ctl.sh` start/stop patched Qdrant (k3s containerd). `QVOL` env selects the storage dir.
- `qgran_sweep.sh A|B|C <out>`: A = cache off (local NVMe), B = cache budget 5/20%, C = gp3-like cap (3000 IOPS, 125 MiB/s) with/without a 4KB neighbor.
- `delay_dev.sh`, `qgran_delay.sh`: dm-delay volume (1 ms per read I/O) sweep.
- `milvus_gran_sweep.sh`, `milvus_gran_ctl.sh`: Milvus cell-size sweep (text on SeaweedFS S3).
- `qgran_bench.py` load generator; `evictor.py` page-cache eviction; `neighbor.py` 4KB neighbor tenant; `summarize_q.py` / `summarize_m.py`.

## Results (`results/`)
| dir | experiment |
|---|---|
| res-A2 | Qdrant, cache off, local NVMe, natural, C=1/32/128, 2 reps |
| res-B2 | Qdrant, cache budget 5% / 20%, zipf, C=32 |
| res-C  | Qdrant, gp3-like IOPS/bandwidth cap, neighbor n0/n1000 |
| res-D  | Qdrant, dm-delay 1 ms per I/O (d0cap, d1cap, d1cap8, d1nocap) |
| res-M  | Milvus, cell 1/2/4/8 RG x budget 5/20% x natural/zipf |

`*.jsonl.gz` = per-request latencies, `*.gran.json` = app-level read counters, `*.s3.log.gz` = captured S3 Range requests.
Discarded runs (noisy / S3 capture failed: res-A, res-B, res-M-noS3) are not included.

## Figures (`figs/`)
PNG/SVG for Milvus QPS/amplification, Qdrant cache-budget QPS/I-O, gp3 QPS and neighbor latency, dm-delay QPS.

## Key findings
- Large reads (>=256KB) collapse QPS 3-25x under a capped disk and waste cache; 16-64KB is the sweet spot.
- Small reads (4KB) cost ~15-20% QPS vs the optimum only when IOPS is capped; adding 1 ms per-I/O latency does not change this (Qdrant issues the top-10 payload reads concurrently), so the small-read cost is I/O count x IOPS cap.
- Milvus cannot go below 1 RG (~1MB), so only the large-read half is visible there.
