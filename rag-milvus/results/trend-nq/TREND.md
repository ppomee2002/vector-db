# Trend: nq, Milvus v2.6.23 tiered storage (config C)

text 1326.2 MB, vector index 4505.7 MB. Budget = text cache share (Qdrant: of payload disk size). Chunk LRU/Belady = same byte budget, chunk-granular, cold replay of the same results.

| budget | trace | C | QPS | p95 ms | p99 ms | returned MB | storage text read MB | amplification | chunk LRU MB | chunk Belady MB | Milvus/Belady | S3 GETs | dup GET % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100% | natural | 32 | 252 | 199.8 | 302.5 | 10.1 | 210.1 | 20.7x | 10.1 | 10.1 | 20.9x | 0 | 0 |
| 100% | natural | 128 | 265 | 562.6 | 635.2 | 10.1 | 0.0 | 0.0x | 10.1 | 10.1 | 0.0x | 0 | 0 |
| 50% | natural | 32 | 255 | 307.4 | 483.3 | 10.1 | 1328.2 | 130.9x | 10.1 | 10.1 | 132.0x | 0 | 0 |
| 50% | natural | 128 | 260 | 955.7 | 1471.3 | 10.1 | 1639.2 | 161.5x | 10.1 | 10.1 | 162.9x | 0 | 0 |
| 20% | natural | 32 | 198 | 439.7 | 588.9 | 10.1 | 6435.8 | 634.2x | 10.1 | 10.1 | 639.4x | 0 | 0 |
| 20% | natural | 128 | 169 | 1493.7 | 3072.7 | 10.1 | 6467.8 | 637.4x | 10.1 | 10.1 | 642.6x | 0 | 0 |
| 100% | zipf | 32 | 336 | 142.8 | 189.6 | 10.0 | 0.0 | 0.0x | 2.2 | 2.2 | 0.0x | 0 | 0 |
| 100% | zipf | 128 | 338 | 411.5 | 457.7 | 10.0 | 0.0 | 0.0x | 2.2 | 2.2 | 0.0x | 0 | 0 |
| 50% | zipf | 32 | 371 | 135.2 | 296.3 | 10.0 | 176.6 | 17.7x | 2.2 | 2.2 | 79.0x | 0 | 0 |
| 50% | zipf | 128 | 340 | 380.7 | 423.4 | 10.0 | 51.5 | 5.2x | 2.2 | 2.2 | 23.0x | 0 | 0 |
| 20% | zipf | 32 | 280 | 338.4 | 493.3 | 10.0 | 2879.3 | 289.2x | 2.2 | 2.2 | 1288.0x | 0 | 0 |
| 20% | zipf | 128 | 271 | 1173.4 | 2109.4 | 10.0 | 3320.0 | 333.4x | 2.2 | 2.2 | 1485.1x | 0 | 0 |

## Context-read cost (zipf, C=32): text vs no text

| budget | QPS text | QPS no text | p99 text | p99 no text |
|---:|---:|---:|---:|---:|
| 100% | 336 | 390 | 189.6 | 117.6 |
| 50% | 371 | 371 | 296.3 | 137.5 |
| 20% | 280 | 383 | 493.3 | 132.1 |
