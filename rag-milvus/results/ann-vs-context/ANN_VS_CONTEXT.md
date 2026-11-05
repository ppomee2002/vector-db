# ANN search access vs RAG context access (same queries)

faiss HNSW M=16 efC=128 IP, search replayed in Python with ef=64, k=10. Distinct queries in the natural (permutation) order.
ANN stream = nodes whose vector is read (distance computed). Context stream = top-10 chunks whose text is fetched.

## fiqa (57,638 chunks, 2000 queries; replay agrees with faiss top-10: 100.0%)

| | ANN search (HNSW) | RAG context fetch |
|---|---:|---:|
| items read per query | 1086 vectors + 73 neighbour lists | 10 chunks |
| bytes read per query | 1678 KB | 9.0 KB |
| accesses on top 1% items | 8.5% | 14.3% |
| accesses on top 10% items | 33.6% | 65.1% |
| items already used by an earlier query | 97.4% | 36.2% |
| reads on upper HNSW levels | 14.9% | - |
| LRU hit ratio, cache = 0.1% of items | 1.7% | 0.2% |
| LRU hit ratio, cache = 1.0% of items | 2.6% | 3.4% |
| LRU hit ratio, cache = 5.0% of items | 11.5% | 15.5% |
| LRU hit ratio, cache = 10.0% of items | 20.5% | 25.7% |
| 4KB blocks touched per query (pk-order layout), touched/needed bytes | 1290, 3.2x | 12.2, 5.6x |
| 2MB blocks touched per query (pk-order layout), touched/needed bytes | 43, 55.4x | 8.1, 1900.6x |

## nq (2,681,468 chunks, 2000 queries; replay agrees with faiss top-10: 100.0%)

| | ANN search (HNSW) | RAG context fetch |
|---|---:|---:|
| items read per query | 1122 vectors + 80 neighbour lists | 10 chunks |
| bytes read per query | 1734 KB | 5.1 KB |
| accesses on top 1% items | 14.1% | 100.0% |
| accesses on top 10% items | 46.1% | 100.0% |
| items already used by an earlier query | 43.1% | 1.3% |
| reads on upper HNSW levels | 19.6% | - |
| LRU hit ratio, cache = 0.1% of items | 3.6% | 0.3% |
| LRU hit ratio, cache = 1.0% of items | 7.4% | 1.3% |
| LRU hit ratio, cache = 5.0% of items | 14.8% | 1.3% |
| LRU hit ratio, cache = 10.0% of items | 21.9% | 1.3% |
| 4KB blocks touched per query (pk-order layout), touched/needed bytes | 1159, 2.8x | 7.1, 5.7x |
| 2MB blocks touched per query (pk-order layout), touched/needed bytes | 512, 634.9x | 3.9, 1607.0x |

