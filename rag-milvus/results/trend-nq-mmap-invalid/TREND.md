# Trend: nq, Milvus v2.6.23 mmap (config B, page-cache budget via memory limit)

text 1326.2 MB, container memory after vector-only probe 7410.7 MB. Budget = text cache share (Qdrant: of payload disk size). Chunk LRU/Belady = same byte budget, chunk-granular, cold replay of the same results.

| budget | trace | C | QPS | p95 ms | p99 ms | returned MB | storage text read MB | amplification | chunk LRU MB | chunk Belady MB | Milvus/Belady | S3 GETs | dup GET % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50% | natural | 32 | 304 | 146.9 | 176.3 | 10.1 | 0.0 | 0.0x | 10.1 | 10.1 | 0.0x | 0 | 0 |
| 50% | natural | 128 | 290 | 483.3 | 557.7 | 10.1 | 0.0 | 0.0x | 10.1 | 10.1 | 0.0x | 0 | 0 |
| 20% | natural | 32 | 363 | 108.6 | 129.1 | 10.1 | 0.0 | 0.0x | 10.1 | 10.1 | 0.0x | 0 | 0 |
| 20% | natural | 128 | 336 | 401.8 | 438.7 | 10.1 | 0.0 | 0.0x | 10.1 | 10.1 | 0.0x | 0 | 0 |
| 5% | natural | 32 | 389 | 103.8 | 120.5 | 10.1 | 0.0 | 0.0x | 10.1 | 10.1 | 0.0x | 0 | 0 |
| 5% | natural | 128 | 352 | 390.4 | 439.3 | 10.1 | 0.0 | 0.0x | 10.1 | 10.1 | 0.0x | 0 | 0 |
| 50% | zipf | 32 | 400 | 100.1 | 123.4 | 10.0 | 0.0 | 0.0x | 2.2 | 2.2 | 0.0x | 0 | 0 |
| 50% | zipf | 128 | 345 | 394.3 | 439.7 | 10.0 | 0.0 | 0.0x | 2.2 | 2.2 | 0.0x | 0 | 0 |
| 20% | zipf | 32 | 437 | 91.3 | 106.6 | 10.0 | 0.0 | 0.0x | 2.2 | 2.2 | 0.0x | 0 | 0 |
| 20% | zipf | 128 | 388 | 345.9 | 382.4 | 10.0 | 0.0 | 0.0x | 2.2 | 2.2 | 0.0x | 0 | 0 |
| 5% | zipf | 32 | 404 | 100.3 | 123.9 | 10.0 | 0.0 | 0.0x | 2.2 | 2.2 | 0.0x | 0 | 0 |
| 5% | zipf | 128 | 278 | 428.9 | 479.0 | 10.0 | 0.0 | 0.0x | 2.2 | 2.2 | 0.0x | 0 | 0 |

## Context-read cost (zipf, C=32): text vs no text

| budget | QPS text | QPS no text | p99 text | p99 no text |
|---:|---:|---:|---:|---:|
| 50% | 400 | 422 | 123.4 | 113.1 |
| 20% | 437 | 449 | 106.6 | 112.5 |
| 5% | 404 | 329 | 123.9 | 133.8 |
