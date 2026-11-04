# Context-read what-if (trace replay of measured Milvus results)

Simulation on the recorded top-10 results, not a measurement. Blocks = contiguous byte ranges in pk order.

## (1) Cache granularity: storage text read (MB) at the same budget, LRU, C=32

| dataset | budget | trace | returned | chunk | 4 KB | 64 KB | 2 MB | Milvus measured | 2 MB / chunk |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| fiqa | 50% | natural | 26.9 | 14.7 | 63.5 | 976.4 | 33313.3 | 104.9 | 2261x |
| fiqa | 50% | zipf | 26.9 | 4.9 | 20.7 | 424.7 | 30582.8 | 157.3 | 6302x |
| fiqa | 20% | natural | 9.0 | 7.1 | 37.8 | 531.1 | 15988.7 | 177.8 | 2267x |
| fiqa | 20% | zipf | 8.8 | 2.4 | 12.1 | 312.9 | 16091.4 | 142.9 | 6716x |
| fiqa | 10% | natural | 9.0 | 7.3 | 43.6 | 601.2 | 16747.9 | 149.4 | 2284x |
| fiqa | 10% | zipf | 8.8 | 2.4 | 15.0 | 420.3 | 16817.1 | 52.4 | 7019x |
| nq | 50% | natural | 10.2 | 10.1 | 54.1 | 376.8 | 6182.4 | 1328.2 | 614x |
| nq | 50% | zipf | 10.0 | 2.2 | 12.3 | 105.1 | 1738.5 | 176.6 | 778x |
| nq | 20% | natural | 10.2 | 10.1 | 54.1 | 390.6 | 10838.1 | 6435.8 | 1077x |
| nq | 20% | zipf | 10.0 | 2.2 | 12.3 | 105.1 | 4559.2 | 2879.3 | 2039x |

## (2) Neighbour-chunk expansion (NQ, +-w chunks of the same article, C=32)

| trace | w | top-10 only MB | requested, no merge MB | after intra-query merge MB | merge saving | LRU 4 KB @20% | LRU 2 MB @20% | LRU 4 KB @5% | LRU 2 MB @5% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| natural | 0 | 10.2 | 10.2 | 10.2 | 0% | 54.1 | 10838.1 | 54.1 | 14734.6 |
| natural | 1 | 10.2 | 29.2 | 23.5 | 20% | 61.9 | 10848.6 | 61.9 | 14749.3 |
| natural | 2 | 10.2 | 47.1 | 33.7 | 28% | 68.7 | 10854.9 | 68.7 | 14764.0 |
| zipf | 0 | 10.0 | 10.0 | 10.0 | 0% | 12.3 | 4559.2 | 12.3 | 9902.8 |
| zipf | 1 | 10.0 | 28.8 | 22.6 | 21% | 14.1 | 4559.2 | 14.1 | 9902.8 |
| zipf | 2 | 10.0 | 46.4 | 32.8 | 29% | 15.6 | 4561.3 | 15.6 | 9904.8 |

## (3) Cross-query overlap: share of block bytes in a window of N consecutive requests also needed by another request in the window

N ~ requests in flight (closed loop: C workers keep ~C in flight). This is the read volume a shared read could remove with no cache.

| dataset | trace | C | w | chunk N=8 | chunk N=32 | chunk N=128 | 4 KB N=32 | 4 KB N=128 | 2 MB N=32 | 2 MB N=128 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fiqa | natural | 32 | 0 | 0.1% | 0.9% | 3.3% | 2.5% | 9.3% | 91.6% | 97.9% |
| fiqa | natural | 128 | 0 | 0.1% | 0.9% | 3.3% | 2.5% | 9.3% | 91.6% | 97.9% |
| fiqa | zipf | 32 | 0 | 15.8% | 33.3% | 51.1% | 34.5% | 52.7% | 91.7% | 97.9% |
| fiqa | zipf | 128 | 0 | 15.8% | 33.3% | 51.1% | 34.5% | 52.7% | 91.7% | 97.9% |
| nq | natural | 32 | 0 | 0.0% | 0.0% | 0.0% | 0.1% | 0.4% | 16.3% | 40.5% |
| nq | natural | 32 | 1 | 0.0% | 0.0% | 0.1% | 0.1% | 0.5% | 16.3% | 40.5% |
| nq | natural | 32 | 2 | 0.0% | 0.0% | 0.1% | 0.1% | 0.5% | 16.3% | 40.5% |
| nq | natural | 128 | 0 | 0.0% | 0.0% | 0.0% | 0.1% | 0.4% | 16.3% | 40.5% |
| nq | natural | 128 | 1 | 0.0% | 0.0% | 0.1% | 0.1% | 0.5% | 16.3% | 40.5% |
| nq | natural | 128 | 2 | 0.0% | 0.0% | 0.1% | 0.1% | 0.5% | 16.3% | 40.5% |
| nq | zipf | 32 | 0 | 17.0% | 35.4% | 51.7% | 31.7% | 49.8% | 40.4% | 63.9% |
| nq | zipf | 32 | 1 | 14.9% | 32.1% | 49.3% | 30.1% | 48.2% | 40.4% | 63.9% |
| nq | zipf | 32 | 2 | 13.9% | 31.1% | 48.9% | 31.4% | 49.6% | 40.4% | 63.9% |
| nq | zipf | 128 | 0 | 17.0% | 35.4% | 51.7% | 31.7% | 49.8% | 40.4% | 63.9% |
| nq | zipf | 128 | 1 | 14.9% | 32.1% | 49.3% | 30.1% | 48.2% | 40.4% | 63.9% |
| nq | zipf | 128 | 2 | 13.9% | 31.1% | 48.9% | 31.4% | 49.6% | 40.4% | 63.9% |
