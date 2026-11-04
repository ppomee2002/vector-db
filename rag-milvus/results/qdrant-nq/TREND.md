# Trend: nq, Qdrant (on_disk_payload, page-cache budget via memory limit)

text 1326.2 MB, payload on disk 1542.9 MB. Budget = text cache share (Qdrant: of payload disk size). Chunk LRU/Belady = same byte budget, chunk-granular, cold replay of the same results.

| budget | trace | C | QPS | p95 ms | p99 ms | returned MB | storage text read MB | amplification | chunk LRU MB | chunk Belady MB | Milvus/Belady | S3 GETs | dup GET % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100% | natural | 32 | 1158 | 24.7 | 30.4 | 10.2 | 70.2 | 6.9x | 10.1 | 10.1 | 7.0x | 0 | 0 |
| 100% | natural | 128 | 821 | 120.8 | 149.1 | 10.2 | 0.0 | 0.0x | 10.1 | 10.1 | 0.0x | 0 | 0 |
| 50% | natural | 32 | 1212 | 22.6 | 27.2 | 10.2 | 70.2 | 6.9x | 10.1 | 10.1 | 7.0x | 0 | 0 |
| 50% | natural | 128 | 825 | 125.7 | 160.3 | 10.2 | 0.0 | 0.0x | 10.1 | 10.1 | 0.0x | 0 | 0 |
| 20% | natural | 32 | 1241 | 22.0 | 28.7 | 10.2 | 70.2 | 6.9x | 10.1 | 10.1 | 7.0x | 0 | 0 |
| 20% | natural | 128 | 849 | 115.8 | 151.5 | 10.2 | 0.0 | 0.0x | 10.1 | 10.1 | 0.0x | 0 | 0 |
| 100% | zipf | 32 | 1347 | 20.6 | 28.3 | 10.0 | 1.0 | 0.1x | 2.2 | 2.2 | 0.4x | 0 | 0 |
| 100% | zipf | 128 | 890 | 107.2 | 132.6 | 10.0 | 0.0 | 0.0x | 2.2 | 2.2 | 0.0x | 0 | 0 |
| 50% | zipf | 32 | 1360 | 20.1 | 26.9 | 10.0 | 1.0 | 0.1x | 2.2 | 2.2 | 0.4x | 0 | 0 |
| 50% | zipf | 128 | 878 | 107.1 | 136.1 | 10.0 | 0.0 | 0.0x | 2.2 | 2.2 | 0.0x | 0 | 0 |
| 20% | zipf | 32 | 1409 | 18.8 | 27.4 | 10.0 | 1.0 | 0.1x | 2.2 | 2.2 | 0.4x | 0 | 0 |
| 20% | zipf | 128 | 913 | 98.1 | 125.3 | 10.0 | 0.0 | 0.0x | 2.2 | 2.2 | 0.0x | 0 | 0 |

## Context-read cost (zipf, C=32): text vs no text

| budget | QPS text | QPS no text | p99 text | p99 no text |
|---:|---:|---:|---:|---:|
| 100% | 1347 | 1376 | 28.3 | 27.8 |
| 50% | 1360 | 1377 | 26.9 | 31.6 |
| 20% | 1409 | 1413 | 27.4 | 27.5 |
