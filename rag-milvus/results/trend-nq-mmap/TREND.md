# Trend: nq, Milvus v2.6.23 mmap (config B, page-cache budget via memory limit)

text 1326.2 MB, container memory after vector-only probe 7438.3 MB. Budget = text cache share (Qdrant: of payload disk size). Chunk LRU/Belady = same byte budget, chunk-granular, cold replay of the same results.

| budget | trace | C | QPS | p95 ms | p99 ms | returned MB | storage text read MB | amplification | chunk LRU MB | chunk Belady MB | Milvus/Belady | S3 GETs | dup GET % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50% | natural | 32 | 246 | 197.9 | 245.6 | 10.1 | 893.4 | 88.0x | 10.1 | 10.1 | 88.8x | 0 | 0 |
| 50% | natural | 128 | 264 | 531.3 | 566.0 | 10.1 | 29.2 | 2.9x | 10.1 | 10.1 | 2.9x | 0 | 0 |
| 20% | natural | 32 | 283 | 149.9 | 176.9 | 10.1 | 1001.8 | 98.7x | 10.1 | 10.1 | 99.5x | 0 | 0 |
| 20% | natural | 128 | 272 | 539.9 | 582.5 | 10.1 | 53.3 | 5.3x | 10.1 | 10.1 | 5.3x | 0 | 0 |
| 5% | natural | 32 | 346 | 121.6 | 150.4 | 10.1 | 1006.8 | 99.2x | 10.1 | 10.1 | 100.0x | 0 | 0 |
| 5% | natural | 128 | 310 | 466.2 | 511.0 | 10.1 | 70.6 | 7.0x | 10.1 | 10.1 | 7.0x | 0 | 0 |
| 50% | zipf | 32 | 360 | 115.0 | 134.6 | 10.0 | 70.1 | 7.0x | 2.2 | 2.2 | 31.4x | 0 | 0 |
| 50% | zipf | 128 | 290 | 478.7 | 506.4 | 10.0 | 109.5 | 11.0x | 2.2 | 2.2 | 49.0x | 0 | 0 |
| 20% | zipf | 32 | 352 | 112.8 | 127.1 | 10.0 | 104.8 | 10.5x | 2.2 | 2.2 | 46.9x | 0 | 0 |
| 20% | zipf | 128 | 317 | 445.1 | 489.7 | 10.0 | 27.1 | 2.7x | 2.2 | 2.2 | 12.1x | 0 | 0 |
| 5% | zipf | 32 | 400 | 104.9 | 120.1 | 10.0 | 216.7 | 21.8x | 2.2 | 2.2 | 97.0x | 0 | 0 |
| 5% | zipf | 128 | 356 | 387.5 | 418.7 | 10.0 | 171.3 | 17.2x | 2.2 | 2.2 | 76.6x | 0 | 0 |

## Context-read cost (zipf, C=32): text vs no text

| budget | QPS text | QPS no text | p99 text | p99 no text |
|---:|---:|---:|---:|---:|
| 50% | 360 | 336 | 134.6 | 157.2 |
| 20% | 352 | 392 | 127.1 | 116.5 |
| 5% | 400 | 408 | 120.1 | 129.7 |
