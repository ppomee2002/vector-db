# Trend: nq, Milvus v2.6.23 tiered storage (config C)

text 1326.2 MB, vector index 4505.7 MB. Budget = text cache share (Qdrant: of payload disk size). Chunk LRU/Belady = same byte budget, chunk-granular, cold replay of the same results.

| budget | trace | C | QPS | p95 ms | p99 ms | returned MB | storage text read MB | amplification | chunk LRU MB | chunk Belady MB | Milvus/Belady | S3 GETs | dup GET % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5% | natural | 32 | 87 | 874.9 | 1235.8 | 10.1 | 13413.7 | 1321.9x | 10.1 | 10.1 | 1332.7x | 0 | 0 |
| 5% | natural | 128 | 87 | 2474.4 | 3272.1 | 10.1 | 9501.3 | 936.3x | 10.1 | 10.1 | 944.0x | 0 | 0 |
| 5% | zipf | 32 | 139 | 711.2 | 981.7 | 10.0 | 6924.8 | 695.4x | 2.2 | 2.2 | 3097.7x | 0 | 0 |
| 5% | zipf | 128 | 130 | 2054.7 | 2939.7 | 10.0 | 7681.0 | 771.4x | 2.2 | 2.2 | 3435.9x | 0 | 0 |

## Context-read cost (zipf, C=32): text vs no text

| budget | QPS text | QPS no text | p99 text | p99 no text |
|---:|---:|---:|---:|---:|
| 5% | 139 | 282 | 981.7 | 151.3 |
