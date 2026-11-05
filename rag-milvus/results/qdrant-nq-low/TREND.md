# Trend: nq, Qdrant (on_disk_payload, page-cache budget via memory limit)

text 1326.2 MB, payload on disk 1542.9 MB. Budget = text cache share (Qdrant: of payload disk size). Chunk LRU/Belady = same byte budget, chunk-granular, cold replay of the same results.

| budget | trace | C | QPS | p95 ms | p99 ms | returned MB | storage text read MB | amplification | chunk LRU MB | chunk Belady MB | Milvus/Belady | S3 GETs | dup GET % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5% | natural | 32 | 1233 | 22.1 | 29.7 | 10.2 | 71.4 | 7.0x | 10.1 | 10.1 | 7.1x | 0 | 0 |
| 5% | natural | 128 | 803 | 139.3 | 183.0 | 10.2 | 140.1 | 13.8x | 10.1 | 10.1 | 13.9x | 0 | 0 |
| 2% | natural | 32 | 1230 | 21.9 | 29.3 | 10.2 | 77.5 | 7.6x | 10.1 | 10.1 | 7.7x | 0 | 0 |
| 2% | natural | 128 | 788 | 149.2 | 198.6 | 10.2 | 193.1 | 19.0x | 10.1 | 10.1 | 19.2x | 0 | 0 |
| 1% | natural | 32 | 1258 | 21.0 | 28.4 | 10.2 | 83.9 | 8.3x | 10.1 | 10.1 | 8.3x | 0 | 0 |
| 1% | natural | 128 | 828 | 129.6 | 182.2 | 10.2 | 177.5 | 17.5x | 10.1 | 10.1 | 17.6x | 0 | 0 |
| 5% | zipf | 32 | 1358 | 20.3 | 24.7 | 10.0 | 31.0 | 3.1x | 2.2 | 2.2 | 13.9x | 0 | 0 |
| 5% | zipf | 128 | 869 | 109.6 | 138.4 | 10.0 | 33.1 | 3.3x | 2.2 | 2.2 | 14.8x | 0 | 0 |
| 2% | zipf | 32 | 1330 | 20.8 | 30.0 | 10.0 | 29.0 | 2.9x | 2.2 | 2.2 | 13.0x | 0 | 0 |
| 2% | zipf | 128 | 875 | 109.9 | 137.7 | 10.0 | 16.4 | 1.6x | 2.2 | 2.2 | 7.3x | 0 | 0 |
| 1% | zipf | 32 | 1405 | 18.9 | 26.7 | 10.0 | 20.2 | 2.0x | 2.2 | 2.2 | 9.0x | 0 | 0 |
| 1% | zipf | 128 | 916 | 97.5 | 122.1 | 10.0 | 23.1 | 2.3x | 2.2 | 2.2 | 10.3x | 0 | 0 |

## Context-read cost (zipf, C=32): text vs no text

| budget | QPS text | QPS no text | p99 text | p99 no text |
|---:|---:|---:|---:|---:|
| 5% | 1358 | 1374 | 24.7 | 30.9 |
| 2% | 1330 | 1379 | 30.0 | 29.6 |
| 1% | 1405 | 1433 | 26.7 | 25.9 |
