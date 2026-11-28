# C3 14주차 A 실험: 동시 수에 따른 단계별 지연·자원 (warm) — 규칙 (측정 전 고정)

작성: 2026-11-28, 측정 전. 결과를 본 뒤 기준·조건을 바꾸지 않는다. 기준값(0.9, 60%, 20%p, 1MiB)은 Claude가 정했고 강한 근거는 없다.
팀 문서(14주차 임준) 실험 ③ "C1·C2 적용 전후의 데이터 접근 패턴 및 최적 동시 실행 수 변화", ④ "동시 실행 수 증가에 따른 병목 분석"에 해당.
고정 상한을 조절하는 실험이 아니다. C3는 모든 조건에서 끈다(m3). 동시 수는 병목을 드러내는 변수로만 쓴다.

## 조건
| 이름 | 본문 컬렉션 (C1 배치) | C2 | C3 |
|---|---|---|---|
| A1 | jihyeon/w13c1_rand (무작위) | none (bc1_v1, policy none) | m3 (m3trace_junlim) |
| A2 | jihyeon/w13c1_doc (문서 순) | none (bc1_v1, policy none) | m3 |
| A3 | jihyeon/w13c1_doc (문서 순) | BC1 (bc1_v1) | m3 |
- 검색은 모두 junlim/nq_seq (top-10, ef 64). 본문은 한 번에 읽기(query pk in)로 같다 → A1 vs A2는 "C1 전체"가 아니라 "문서 순 배치의 효과"를 본다.
- 왜 nq_seq가 아니라 w13c1_doc인가: nq_seq는 본문 mmap, w13c1_*는 mmap 없이 캐시 계층으로 읽는다. 배치 외의 차이(읽기 경로)를 없애려고 A1·A2를 같은 경로(w13c1_*)로 맞췄다. 팀 통합 경로(nq_seq)와 다르다는 점은 결과에 적는다.
- 동시 수(clients): 1, 4, 6, 8, 12, 16. 64는 Milvus 종료 원인 확인 전이라 하지 않는다.
- 질문: nq_test_real 3,452개, 팀 고정 seed 순서 순환. 시간 기반: 예열 5초 + 측정 20초. evaluate false.
- 반복 3회. 회차마다 18칸(3조건 × 6동시 수)의 순서를 고정 seed(회차 번호)로 무작위로 섞는다. 실행마다 팀 main.py가 MilvusGuard로 load → 측정 → release.

## 기록
- 팀 main.py 결과(results.jsonl, summary.json, run_meta.json: 측정 구간 storage_read_delta, Milvus CPU 초 등).
- m3trace_junlim 어댑터가 질의마다 실제 시각(epoch)으로 큐 진입·시작·완료와 단계별 시간을 trace.csv에 남긴다(results.jsonl과 같은 순서). 단계 시작 시각은 시작 시각 + 앞 단계 시간의 합으로 근사한다.
- 기록기(rec_c3a2.py, 2초마다): Milvus RSS·누적 CPU초, main.py 프로세스 CPU·RSS, /proc/diskstats 장치별 읽은 섹터, 우리 컨테이너 cpu.stat(nr_throttled, throttled_usec), 다른 계정 python 목록.
- 실행 앞뒤 스위치 status(읽기만).

## 실행 유효성 (하나라도 어기면 그 실행은 "무효", 결과는 모두 보고)
- V1 main.py 종료 코드 0, 측정 구간 오류 0
- V2 측정 구간 storage_read_delta(cache_cell_access_miss_bytes) ≤ 1 MiB (warm 확인)
- V3 Milvus 재시작 없음(storage_read_delta.milvus_restarted false)
- 같은 칸(조건 × 동시 수)에 유효 실행이 3개가 아니면 그 칸은 판정에 쓰지 않는다.
- Milvus가 꺼지면(healthz 실패) 그 자리에서 전체 중단.

## 지표 (모두 측정 구간, 칸마다 3회 평균과 범위)
- QPS(팀 qps), total·search·c2_select·fetch의 평균과 p95.
- 단계별 동시 진행 수: trace에서 측정 구간 동안 검색 단계·Fetch 단계에 있던 질의 수의 시간 평균.
- Milvus CPU 코어 = 측정 구간 누적 CPU초 차이 / 시간(기록기), 클라이언트 CPU 코어도 같은 방식.
- 장치별 물리 읽기 바이트(diskstats 차이). 어느 장치가 Milvus 저장소인지는 측정 전에 확인하지 못했다 → 장치별로 모두 적는다.
- 질의별 Fetch 시간 변동계수.
- 동시 Fetch 간 중복: Fetch 구간이 시간상 겹치는 질의 쌍에서 chunk 중복, 예측 블록(경향 분석 1의 블록 모델, R = 11,750) 중복.

## 판정 (조건마다)
- J1 처리량 꺾임: 평균 QPS가 가장 높은 동시 수 N*보다 큰 동시 수 N 가운데, 평균 QPS < 0.9 × 최고 평균이고 3회 모두 N의 QPS가 N*의 3회 QPS보다 낮은 N이 있으면 "꺾임 있음(가장 작은 그런 N)", 없으면 "꺾임 확인 안 됨".
- J2 주요 지연 증가 단계: 동시 1 → 16에서 평균 단계 시간 증가량(search, c2_select, fetch)을 구해, 한 단계가 세 단계 증가량 합의 60% 이상이면 "주요 증가 단계 = 그 단계", 아니면 "고르게 증가". "병목"이라고 부르지 않는다(원인 확정이 아님).
- J3 조건 간 차이 (A1 vs A2 = 배치, A2 vs A3 = C2): 두 조건의 J2 표시가 다르거나, 동시 1 → 16 증가량 중 fetch 몫(%)의 차이가 20%p 이상이고 3회 모두 같은 방향이면 "단계별 경합 모양이 다름", 아니면 "차이 확인 안 됨".
- CPU 표시: Milvus CPU 평균이 6.5코어 이상인 칸은 "6.5코어 이상"으로만 표시. 실제 한도를 모르므로 "포화"라고 부르지 않는다.

## 말할 수 없는 것
- 물리 I/O 절감, cold·메모리 압박 조건의 경향(warm 실험).
- 팀 통합 경로(nq_seq 본문 mmap)의 값(본문 경로가 다름).
- 원인(경합 위치, CPU 한도 등)의 확정. 판정은 "어디서 늘었나"까지.
