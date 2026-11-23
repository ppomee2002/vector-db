# C1 Milvus 실험 결과 (m1p_cold_20261123_1202)

- M1P cold: pipeline results.jsonl chunk sets, release/load before every fetch, database jihyeon, 쿼리 100개
- 측정 전 Milvus 상태: {'proxy_req_per_s': 0.0, 'milvus_cpu_cores': 0.11, 'miss_MB_per_s': 0.0, 'quiet': True} / 측정 후: {'proxy_req_per_s': 0.0, 'milvus_cpu_cores': 0.12, 'miss_MB_per_s': 0.0, 'quiet': True}
- 벡터는 8차원 가짜 벡터(FLAT)다. 읽는 Chunk 집합은 파이프라인 결과 파일에서 가져오고(/workspace/jihyeon/vector-db/pipeline-core/results/b0_100_jihyeon_20261123_205049/results.jsonl), 본문 조회만 잰다.

| 읽는 범위 | 쿼리당 Chunk | 정답 chunk 포함률 |
|---|---|---|
| top-10 | 10.0 | - |
| top-10 + B+C1 이웃 | 19.32 | - |
| top-10 ±1 전부 | 24.79 | - |

## M1P 콜드 조회: 파이프라인 Chunk 집합 (질문마다 release/load)

| 읽는 범위 | 요청 방식 | 조회 | 배치 | 쿼리당 Chunk | 요청 수 | 읽은 MB (평균) | 지연 p50 ms | 지연 p99 ms | 본문 일치 |
|---|---|---|---|---|---|---|---|---|---|
| top-10 | 한 번에 | query(pk in) | rand | 10.0 | 1.0 | 39.67 | 103.1 | 137.5 | 100% |
| top-10 | 한 번에 | query(pk in) | doc | 10.0 | 1.0 | 19.93 | 76.5 | 108.2 | 100% |
| top-10 | Chunk마다 따로 | query(pk in) | rand | 10.0 | 10.0 | 39.67 | 468.3 | 530.3 | 100% |
| top-10 | Chunk마다 따로 | query(pk in) | doc | 10.0 | 10.0 | 19.93 | 257.7 | 475.0 | 100% |
| top-10 + B+C1 이웃 | 한 번에 | query(pk in) | rand | 19.3 | 1.0 | 73.39 | 153.3 | 243.7 | 100% |
| top-10 + B+C1 이웃 | 한 번에 | query(pk in) | doc | 19.3 | 1.0 | 19.93 | 77.5 | 113.4 | 100% |
| top-10 ±1 전부 | 한 번에 | query(pk in) | rand | 24.8 | 1.0 | 92.05 | 192.5 | 272.6 | 100% |
| top-10 ±1 전부 | 한 번에 | query(pk in) | doc | 24.8 | 1.0 | 19.93 | 77.4 | 115.5 | 100% |
| top-10 ±1 전부 | 한 번에 | get(ids) | rand | 24.8 | 1.0 | 92.05 | 189.0 | 259.2 | 100% |
| top-10 ±1 전부 | 한 번에 | get(ids) | doc | 24.8 | 1.0 | 19.93 | 76.0 | 106.9 | 100% |
| top-10 ±1 전부 | Chunk마다 따로 | query(pk in) | rand | 24.8 | 24.8 | 92.05 | 1062.6 | 1291.2 | 100% |
| top-10 ±1 전부 | Chunk마다 따로 | query(pk in) | doc | 24.8 | 24.8 | 19.93 | 334.2 | 530.0 | 100% |

### doc ÷ rand (같은 질문끼리 짝지어 비교)

| 읽는 범위 | 요청 방식 | 조회 | 읽은 MB 비율 [95% 구간] | 지연 비율 [95% 구간] |
|---|---|---|---|---|
| top-10 ±1 전부 | Chunk마다 따로 | query(pk in) | 0.22 [0.20~0.23] | 0.32 [0.30~0.33] |
| top-10 ±1 전부 | 한 번에 | get(ids) | 0.22 [0.20~0.23] | 0.42 [0.40~0.43] |
| top-10 ±1 전부 | 한 번에 | query(pk in) | 0.22 [0.20~0.23] | 0.43 [0.41~0.45] |
| top-10 + B+C1 이웃 | 한 번에 | query(pk in) | 0.27 [0.24~0.30] | 0.50 [0.48~0.54] |
| top-10 | Chunk마다 따로 | query(pk in) | 0.50 [0.46~0.54] | 0.56 [0.53~0.60] |
| top-10 | 한 번에 | query(pk in) | 0.50 [0.46~0.54] | 0.77 [0.74~0.80] |

### 같은 배치 안에서 비교

| 비교 | 읽는 범위 | 배치 | 읽은 MB 비율 | 지연 비율 [95% 구간] |
|---|---|---|---|---|
| Chunk마다 따로 ÷ 한 번에 | top-10 | doc | 1.00 | 3.29 [3.13~3.45] |
| Chunk마다 따로 ÷ 한 번에 | top-10 | rand | 1.00 | 4.48 [4.38~4.58] |
| Chunk마다 따로 ÷ 한 번에 | top-10 ±1 전부 | doc | 1.00 | 4.17 [4.00~4.33] |
| Chunk마다 따로 ÷ 한 번에 | top-10 ±1 전부 | rand | 1.00 | 5.57 [5.42~5.74] |
| get ÷ query | top-10 ±1 전부 | doc | 1.00 | 0.98 [0.96~1.00] |
| get ÷ query | top-10 ±1 전부 | rand | 1.00 | 1.01 [0.97~1.05] |

읽는 Chunk 집합: 파이프라인 results.jsonl의 질문별 context_ids(top-10)에 C2 규칙(None / B+C1 / All)을 적용. 읽은 MB는 Milvus 메트릭 internal_cache_cell_access_miss_bytes_total(서버 전체 값)의 조회 전후 차이이고, Milvus 캐시 기준 cold다(운영체제 캐시는 비우지 않음).
