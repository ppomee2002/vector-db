# Same-budget context-layer comparison (NQ, 2,000 requests after 300 warmup, C=32, median of reps)

Each request sleeps for the Milvus search latency measured without text (mmap, 5%, zipf), then fetches the text
of its top-10 chunks (w=1: plus +-1 neighbour of the same article) through the layer. Cache budget = share of the
1.33 GB text. local = NVMe with O_DIRECT; s3 = SeaweedFS ranged GETs (~33 ms each).

cell2m = 2 MB cell LRU (Milvus tiered); chunk = chunk LRU, one read per missing chunk; block = 4 KB block LRU +
intra-request range merge; group = block + coalesce gaps < 64 KB + share in-flight reads.

## backend = local

| budget | trace | w | layer | QPS | p50 ms | p99 ms | fetch mean ms | fetch p99 ms | read amp | storage reads/query | reps |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5% | natural | 0 | cell2m | 336 | 91 | 166 | 20.0 | 76 | 1453.6x | 3.53 | 3 |
| 5% | natural | 0 | chunk | 424 | 73 | 131 | 2.1 | 9 | 8.9x | 9.84 | 3 |
| 5% | natural | 0 | block | 428 | 73 | 130 | 1.4 | 10 | 5.2x | 4.39 | 3 |
| 5% | natural | 0 | group | 427 | 73 | 130 | 1.6 | 11 | 6.8x | 3.67 | 3 |
| 5% | natural | 1 | cell2m | 331 | 93 | 171 | 21.4 | 79 | 629.6x | 3.54 | 3 |
| 5% | natural | 1 | chunk | 410 | 75 | 133 | 4.5 | 18 | 8.7x | 22.25 | 3 |
| 5% | natural | 1 | block | 427 | 73 | 130 | 1.5 | 11 | 2.6x | 4.27 | 3 |
| 5% | natural | 1 | group | 426 | 73 | 130 | 1.7 | 11 | 3.2x | 3.67 | 3 |
| 5% | zipf | 0 | cell2m | 392 | 78 | 143 | 7.5 | 41 | 1025.5x | 2.44 | 3 |
| 5% | zipf | 0 | chunk | 435 | 71 | 130 | 0.3 | 4 | 1.7x | 1.79 | 3 |
| 5% | zipf | 0 | block | 435 | 71 | 130 | 0.2 | 2 | 1.0x | 0.81 | 3 |
| 5% | zipf | 0 | group | 435 | 71 | 130 | 0.2 | 2 | 1.3x | 0.67 | 3 |
| 5% | zipf | 1 | cell2m | 390 | 79 | 142 | 7.8 | 42 | 451.4x | 2.44 | 3 |
| 5% | zipf | 1 | chunk | 433 | 71 | 130 | 0.4 | 6 | 1.7x | 4.06 | 3 |
| 5% | zipf | 1 | block | 435 | 71 | 130 | 0.2 | 2 | 0.5x | 0.78 | 3 |
| 5% | zipf | 1 | group | 435 | 71 | 130 | 0.2 | 2 | 0.6x | 0.67 | 3 |
| 20% | natural | 0 | cell2m | 379 | 80 | 146 | 10.0 | 45 | 1075.3x | 2.61 | 3 |
| 20% | natural | 0 | chunk | 423 | 73 | 130 | 2.2 | 10 | 8.9x | 9.84 | 3 |
| 20% | natural | 0 | block | 428 | 73 | 130 | 1.4 | 10 | 5.2x | 4.39 | 3 |
| 20% | natural | 0 | group | 427 | 73 | 130 | 1.6 | 11 | 6.8x | 3.67 | 3 |
| 20% | natural | 1 | cell2m | 376 | 81 | 147 | 10.7 | 48 | 464.5x | 2.61 | 3 |
| 20% | natural | 1 | chunk | 411 | 75 | 133 | 4.3 | 16 | 8.7x | 22.25 | 3 |
| 20% | natural | 1 | block | 427 | 73 | 130 | 1.5 | 12 | 2.6x | 4.27 | 3 |
| 20% | natural | 1 | group | 426 | 73 | 130 | 1.7 | 12 | 3.2x | 3.67 | 3 |
| 20% | zipf | 0 | cell2m | 421 | 73 | 134 | 2.3 | 18 | 454.2x | 1.08 | 3 |
| 20% | zipf | 0 | chunk | 435 | 71 | 130 | 0.3 | 4 | 1.7x | 1.79 | 3 |
| 20% | zipf | 0 | block | 435 | 71 | 130 | 0.1 | 2 | 1.0x | 0.81 | 3 |
| 20% | zipf | 0 | group | 435 | 71 | 130 | 0.1 | 2 | 1.3x | 0.67 | 3 |
| 20% | zipf | 1 | cell2m | 421 | 73 | 132 | 2.3 | 18 | 199.7x | 1.08 | 3 |
| 20% | zipf | 1 | chunk | 433 | 71 | 130 | 0.4 | 6 | 1.7x | 4.06 | 3 |
| 20% | zipf | 1 | block | 435 | 71 | 130 | 0.2 | 2 | 0.5x | 0.78 | 3 |
| 20% | zipf | 1 | group | 435 | 71 | 130 | 0.2 | 2 | 0.6x | 0.67 | 3 |

## backend = s3

| budget | trace | w | layer | QPS | p50 ms | p99 ms | fetch mean ms | fetch p99 ms | read amp | storage reads/query | reps |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5% | natural | 0 | cell2m | 165 | 186 | 363 | 113.7 | 279 | 1465.6x | 3.56 | 3 |
| 5% | natural | 0 | chunk | 207 | 150 | 229 | 75.6 | 136 | 1.0x | 9.84 | 3 |
| 5% | natural | 0 | block | 257 | 121 | 185 | 50.0 | 88 | 5.2x | 4.40 | 3 |
| 5% | natural | 0 | group | 262 | 119 | 186 | 47.6 | 87 | 6.8x | 3.67 | 3 |
| 5% | natural | 1 | cell2m | 166 | 188 | 359 | 113.0 | 278 | 632.6x | 3.56 | 3 |
| 5% | natural | 1 | chunk | 106 | 298 | 474 | 208.7 | 360 | 1.0x | 22.25 | 3 |
| 5% | natural | 1 | block | 260 | 119 | 189 | 48.9 | 85 | 2.6x | 4.27 | 3 |
| 5% | natural | 1 | group | 262 | 119 | 188 | 48.0 | 89 | 3.2x | 3.67 | 3 |
| 5% | zipf | 0 | cell2m | 215 | 151 | 293 | 70.4 | 210 | 1090.8x | 2.60 | 3 |
| 5% | zipf | 0 | chunk | 397 | 74 | 150 | 7.3 | 50 | 0.2x | 1.81 | 3 |
| 5% | zipf | 0 | block | 399 | 74 | 146 | 7.0 | 47 | 1.0x | 0.82 | 3 |
| 5% | zipf | 0 | group | 399 | 74 | 147 | 6.9 | 48 | 1.3x | 0.67 | 3 |
| 5% | zipf | 1 | cell2m | 214 | 150 | 297 | 70.8 | 213 | 482.0x | 2.61 | 3 |
| 5% | zipf | 1 | chunk | 378 | 76 | 179 | 10.1 | 89 | 0.2x | 4.10 | 3 |
| 5% | zipf | 1 | block | 399 | 74 | 148 | 7.0 | 48 | 0.5x | 0.79 | 3 |
| 5% | zipf | 1 | group | 398 | 74 | 148 | 7.1 | 49 | 0.6x | 0.67 | 3 |
| 20% | natural | 0 | cell2m | 197 | 159 | 297 | 83.6 | 215 | 1096.8x | 2.66 | 3 |
| 20% | natural | 0 | chunk | 205 | 153 | 236 | 78.3 | 146 | 1.0x | 9.84 | 3 |
| 20% | natural | 0 | block | 257 | 121 | 190 | 49.9 | 91 | 5.2x | 4.40 | 3 |
| 20% | natural | 0 | group | 262 | 119 | 185 | 48.1 | 88 | 6.8x | 3.67 | 3 |
| 20% | natural | 1 | cell2m | 196 | 161 | 293 | 85.2 | 211 | 471.8x | 2.65 | 3 |
| 20% | natural | 1 | chunk | 106 | 297 | 487 | 210.5 | 395 | 1.0x | 22.25 | 3 |
| 20% | natural | 1 | block | 258 | 121 | 185 | 49.5 | 86 | 2.6x | 4.27 | 3 |
| 20% | natural | 1 | group | 261 | 119 | 190 | 48.1 | 88 | 3.2x | 3.67 | 3 |
| 20% | zipf | 0 | cell2m | 342 | 85 | 170 | 19.0 | 82 | 465.3x | 1.11 | 3 |
| 20% | zipf | 0 | chunk | 396 | 74 | 150 | 7.4 | 49 | 0.2x | 1.81 | 3 |
| 20% | zipf | 0 | block | 399 | 74 | 146 | 6.8 | 47 | 1.0x | 0.82 | 3 |
| 20% | zipf | 0 | group | 399 | 74 | 147 | 6.9 | 48 | 1.3x | 0.67 | 3 |
| 20% | zipf | 1 | cell2m | 340 | 86 | 173 | 19.2 | 85 | 205.2x | 1.11 | 3 |
| 20% | zipf | 1 | chunk | 378 | 75 | 179 | 10.0 | 92 | 0.2x | 4.10 | 3 |
| 20% | zipf | 1 | block | 399 | 74 | 147 | 7.0 | 48 | 0.5x | 0.79 | 3 |
| 20% | zipf | 1 | group | 399 | 74 | 148 | 7.0 | 48 | 0.6x | 0.67 | 3 |

## Proposed (block / group) vs best simple layer (chunk), same condition

| backend | budget | trace | w | best of block/group | QPS gain | p99 reduction | fetch-mean reduction | storage reads/query reduction |
|---|---:|---|---:|---|---:|---:|---:|---:|
| local | 5% | natural | 0 | group | 1.01x | 1.00x | 1.3x | 2.7x |
| local | 5% | natural | 1 | block | 1.04x | 1.02x | 3.0x | 5.2x |
| local | 5% | zipf | 0 | group | 1.00x | 1.00x | 1.7x | 2.7x |
| local | 5% | zipf | 1 | group | 1.00x | 1.00x | 2.3x | 6.0x |
| local | 20% | natural | 0 | block | 1.01x | 1.00x | 1.5x | 2.2x |
| local | 20% | natural | 1 | block | 1.04x | 1.02x | 2.8x | 5.2x |
| local | 20% | zipf | 0 | group | 1.00x | 1.00x | 1.7x | 2.7x |
| local | 20% | zipf | 1 | block | 1.00x | 1.00x | 2.4x | 5.2x |
| s3 | 5% | natural | 0 | block | 1.24x | 1.24x | 1.5x | 2.2x |
| s3 | 5% | natural | 1 | group | 2.48x | 2.52x | 4.3x | 6.1x |
| s3 | 5% | zipf | 0 | block | 1.01x | 1.03x | 1.1x | 2.2x |
| s3 | 5% | zipf | 1 | block | 1.06x | 1.21x | 1.5x | 5.2x |
| s3 | 20% | natural | 0 | group | 1.28x | 1.28x | 1.6x | 2.7x |
| s3 | 20% | natural | 1 | block | 2.44x | 2.63x | 4.3x | 5.2x |
| s3 | 20% | zipf | 0 | block | 1.01x | 1.03x | 1.1x | 2.2x |
| s3 | 20% | zipf | 1 | block | 1.06x | 1.22x | 1.4x | 5.2x |

## Milvus measured (NQ, C=32) for reference

| case | QPS | p50 ms | p99 ms |
|---|---:|---:|---:|
| nq-B-b0.05-natural-c32 | 346 | 85 | 150 |
| nq-B-b0.05-zipf-c32 | 400 | 74 | 120 |
| nq-B-b0.05-zipf-c32-notext | 408 | 71 | 130 |
| nq-B-b0.2-natural-c32 | 283 | 106 | 177 |
| nq-B-b0.2-zipf-c32 | 352 | 85 | 127 |
| nq-B-b0.2-zipf-c32-notext | 392 | 76 | 117 |
| nq-C-b0.05-natural-c32 | 87 | 295 | 1236 |
| nq-C-b0.05-zipf-c32 | 139 | 96 | 982 |
| nq-C-b0.05-zipf-c32-notext | 282 | 88 | 151 |
| nq-C-b0.2-natural-c32 | 198 | 107 | 589 |
| nq-C-b0.2-zipf-c32 | 280 | 63 | 493 |
| nq-C-b0.2-zipf-c32-notext | 383 | 77 | 132 |
