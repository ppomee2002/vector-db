# PLAN_integ: 통합 실험 계획 (B1 품질, B2 성능) — 실험 전에 고정

- 작성: 2026-11-24 17:34 KST (실험 시작 전). **결과를 본 뒤 이 문서의 질문·조건·지표·판정 기준·예측은 바꾸지 않는다.** 실행 중 바뀐 점은 RESULTS_integ.md에 따로 적는다.
- 측정 예약 시간대 `run.reserved_window`: **`2026-11-24 17:34~2026-11-24 21:34`** (KST, 시작 시각부터 4시간). 모든 실행의 overlay에 같은 값을 넣고 실행 메타 `reserved_window`에 기록된다. 시간대 안에 끝나지 않은 조건은 실행하지 않고 보고한다.
- 사용 허락: `junlim/nq_seq`(junlim), `jihyeon/w13c1_doc`·`w13c1_rand`(jihyeon). 팀원과 실행 시간 조율 완료(총괄 확인).
- 공통 안전 규칙: 매 실행 전에 MilvusGuard 사용 중 검사(10초 표본, 대상 컬렉션 요청도 남의 것으로 봄). 사용 중이면 그 자리에서 멈추고 보고. 원래 NotLoad였고 이번 실행이 load한 컬렉션만 release. mmap·인덱스 설정은 바꾸지 않음. **clients=64 금지, 동시 요청 최대 16.** 팀원 어댑터(c1_jihyeon.py, c3_m7.py), C2 어댑터(c2_bc1.py), interfaces.py는 수정하지 않음(문제는 보고만).
- 어댑터: C1 `milvus_query_ref`(기본) / `layout_jihyeon`(C1 담당), C2 `none_ref` / `bc1_v1`, C3 `serial_ref` / `m7_junlim`(C3 담당).

## B1. 통합 파이프라인에서 C2 효과 (품질)

**질문:** 통합 파이프라인(ANN Top-10 → C2 → C1 Fetch)에서, 같은 최종 Context 토큰 예산 기준으로 C2(BC1)가 "그냥 더 많이 읽기"보다 나은가?

**설정 (`runs/b1_c2_compare.yaml`)**
- 질문: `nq_test_ans` 2,254개 전체(seed 42 순서), 개수 기반 `max_queries=2254`, `purpose: correctness`(품질 평가), `evaluate: true`, 원문 대조 표본 2,000개.
- 검색: `junlim/nq_seq`, **search_topk 40, context_topk 10, ef 64**(D9). C1 `milvus_query_ref`, C3 `serial_ref`(품질 비교라 동시성 없음).
- 상위 10개 일치(D9): 실행기 기본 검사(첫 10개 질문, 실행 메타 `top10_consistency`) + 대조군 스크립트에서 **2,254개 전체**를 search_topk 10으로 다시 검색해 search_topk 40의 상위 10개와 비교(순서까지).

**조건 (5개, 모두 같은 ANN 검색 결과, 같은 Top-10)**
| 조건 | 읽는 chunk | 만드는 곳 |
|---|---|---|
| none | Top-10 | 파이프라인 `c2.policy=none` (main.py) |
| BC1 | Top-10 ∪ BC1 이웃(k개) | 파이프라인 `bc1_v1` BC1 (main.py) |
| all | Top-10 ∪ 모든 ±1 이웃 | 파이프라인 `bc1_v1` all (main.py) |
| ANN Top-(10+k) | ANN 1~(10+k)위. k = 그 질문에서 BC1이 추가한 수 | 실험 계층 `experiments/b1_c2_controls.py` |
| R3 | Top-10 ∪ (같은 질문의 ±1 이웃 후보 = all의 이웃 집합에서 무작위 k개), seed 0~19 | 실험 계층 `experiments/b1_c2_controls.py` |

- 대조군은 C2 모듈이 아니라 `experiments/`에서 만든다(D8). 대조군 스크립트는 같은 ANN 결과(BC1 실행의 hits 40개, 세 실행의 hits가 같은지 확인)를 쓰고, 본문은 C1 `milvus_query_ref`로 Top-10 ∪ all 이웃 ∪ ANN 11~(10+k)위를 질문마다 한 번에 읽는다(MilvusGuard 규칙 동일). R3는 all 이웃 본문 안에서 고르므로 추가로 읽지 않는다.
- none/BC1/all은 대조군 스크립트에서도 같은 본문으로 다시 평가해 main.py의 질문별 판정과 일치하는지 확인한다(불일치 0이어야 함).

**지표**
- 정답 문자열 포함률(`modules/evaluate.py`, c2x·pf3와 같은 정의). R3는 질문마다 seed 20개 평균 hit(0~1)로 계산.
- 질문당 추가 chunk 수(Top-10 대비), 최종 Context 토큰 수(GPT-2 BPE, `sel_fetch_c2x/results/chunk_tokens_gpt2.npy`, c2x와 같음), 반환 본문 바이트.
- 질문 단위 paired bootstrap 95% CI (10,000회, seed 12345): BC1 − none, BC1 − ANN Top-(10+k), BC1 − R3, all − BC1, ANN Top-(10+k) − none.
- 부분집합: 전체 / **짧은 답 제외**(정답 별칭 중 하나라도 정규화 후 2단어 이하이거나 숫자면 짧은 답, c2x `is_short`와 같음).

**판정 기준**
- "BC1이 그냥 더 많이 읽기보다 낫다" = BC1 − ANN Top-(10+k)의 95% CI 하한 > 0. CI가 0을 포함하면 "차이 없음", 상한 < 0이면 "더 많이 읽기가 낫다".
- "BC1 선택 규칙에 추가 가치가 있다" = BC1 − R3의 95% CI 하한 > 0.
- 토큰 예산: 대조군은 chunk 수를 맞춘 것이라 토큰 수가 정확히 같지는 않다. BC1 대비 평균 토큰 차이가 ±5% 안이면 "같은 토큰 예산"으로 보고, 넘으면 결과에 명시한다. 토큰 수는 보고만 한다(D8).
- 짧은 답 제외에서도 부호가 같으면 "유지"로 적는다.

**예측 (실험 전)**
- BC1 − none: 약 +6~9%p (pf3 ans 기준 0.781 → 0.867, +8.6%p). all − BC1: +0~2%p.
- BC1 − R3: 약 +1%p (c2x Rerank Top-10 조건 +0.99%p [0.51, 1.48]과 같은 경향). CI 하한이 0 근처일 수 있음.
- BC1 − ANN Top-(10+k): **BC1이 약간 낫거나 비슷할 것(+0~4%p)**. 근거: c2x에서 BC1이 가져온 정답 chunk의 88%가 이미 Top-10에 있던 passage의 나머지였고, Rerank가 없는 ANN 11위 이후는 Rerank 확대보다 질이 낮을 것. 단 c2x에서 같은 토큰 예산의 Rerank 확대와는 비슷하거나 약간 못했으므로 차이가 없을 가능성도 있음.
- 토큰: ANN Top-(10+k)·R3와 BC1의 평균 토큰 차이는 ±5% 안.
- 상위 10개 일치: HNSW라 search_topk 40과 10의 상위 10개가 일부 질문에서 다를 수 있음(ef 64 동일, 일치율 95% 이상 예상).

## B2. 통합 성능 (warm)

**질문:** C1·C2·C3를 함께 켰을 때 성능이 어떻게 달라지고, 각 기여의 몫은 얼마인가?

**설정 (`runs/b2_integ.yaml` + 조건별 `--set`)**
- `purpose: measure`, `evaluate: false`, `experiment.cache_condition: warm`, 질문 `nq_test_real` 3,452개 seed 42 고정 순서 순환, 시간 기반(예열 5초 + 측정 20초), **clients 16**, m7 cap 8, 검색 search_topk 10 / context_topk 10 / ef 64.
- 조건(하나씩 빼 보기):
  | # | 이름 | C2 | C1 | C3 |
  |---|---|---|---|---|
  | ① | B0 | none_ref (none) | milvus_query_ref (nq_seq) | m7_junlim m3 |
  | ② | ALL | bc1_v1 BC1 | layout_jihyeon seq (nq_seq) | m7_junlim m7 cap 8 |
  | ③ | ALL−C2 | bc1_v1 none | layout_jihyeon seq | m7_junlim m7 |
  | ④ | ALL−C3 | bc1_v1 BC1 | layout_jihyeon seq | m7_junlim m3 |
  | ⑤ | ALL−C1(rand) | bc1_v1 BC1 | layout_jihyeon rand (jihyeon/w13c1_rand) | m7_junlim m7 |
  - ①의 C3도 `m7_junlim`의 m3 모드로 돌린다(serial_ref는 동시성이 없어 B0 m3 정의와 다름).
- 반복 3회, 조건 순서 교차: 1회차 ①②③④⑤, 2회차 ⑤④③②①, 3회차 ②④①⑤③. 실행마다 MilvusGuard가 load → 측정 → release(실행 사이에 load 상태가 초기화되므로 조건 간 조건이 같음). 실행이 하나라도 사용 중(종료 코드 3) 등으로 멈추면 그 자리에서 중단하고 보고.
- 드라이버: `experiments/b2_driver.py`(main.py를 순서대로 호출, 종료 코드 0이 아니면 중단, 시간대 끝을 넘길 실행은 시작하지 않음).

**지표 (summary.json / run_meta.json)**
- QPS(`qps`), `strict_qps`, total p50/p95/p99, queue_wait p50/p95, 단계별 시간 search / c2_select / fetch (p50, p95), 질문당 요청 수·읽은 chunk 수·반환 바이트, 오류율, 측정 구간 Milvus 저장소 읽기 메트릭 차이(`storage_read_delta`).
- **A/A 잡음:** 같은 조건 3회의 QPS·p50·p95 범위(최대 − 최소)와 변동계수.
- 기여 몫(QPS·p95 차이): C2 = ② − ③, C3 = ② − ④, C1(배치) = ② − ⑤, 전체 = ② − ①.

**판정 기준**
- 두 조건의 차이는 (a) 평균 차이의 절대값이 두 조건의 A/A 범위 중 큰 값보다 크고 (b) 3회차 모두 같은 부호일 때만 "차이 있음". 아니면 "잡음 범위".
- warm에서 C1 배치 차이(②−⑤)는 물리 I/O 근거로 쓰지 않는다(D11). 저장소 읽기 메트릭 차이가 1 MiB를 넘으면 기록만 한다.

**예측 (실험 전)**
- ⑤(C1 rand)와 ②의 차이는 작을 것(잡음 범위). warm에서는 디스크 읽기가 거의 없음. C1 효과는 지현 님 cold 결과(w13-c1-layout 브랜치)와 이후 cache_constrained 실험에서 본다.
- C2를 켜면(② vs ③) 읽는 chunk가 10 → 약 19개로 늘어 fetch 시간이 약간 늘고(p50 +수 ms 이하) QPS가 약간 줄 것.
- m7 vs m3(② vs ④): clients 16에서는 차이가 작을 수 있음(기존 C3 결과상 m7의 이점은 과부하일 때). m7은 queue_wait가 생겨 total p50이 약간 늘 수 있음.
- ①(B0) vs ②: 차이는 주로 C2(읽기량 증가)에서 오고, C1 seq 어댑터는 같은 nq_seq에서 같은 방식(query pk in)으로 읽으므로 ①과 같은 수준.
- 측정 구간 저장소 읽기 메트릭 차이는 모든 조건에서 0에 가깝거나 작을 것(warm).
