# Milvus 기반 C1·C2·C3 1차 통합 파이프라인 공통 명세서

**상태:** 2026-11-23 팀 회의용 **공통 명세 (계약 v1 후보)**. 설계는 총괄이 확정했다. 회의에서는 컬렉션 사용 동의, Milvus load 시간표, 팀원별 기존 코드와의 충돌만 확인한 뒤 계약 v1로 고정한다(8절). 구현·성능 검증은 아직 하지 않았다. 서버·데이터 점검 근거는 `CHECK.md`(2026-11-22)에 있다.

## 1. 이번에 구현할 범위

- **포함:** C1(원문 Fetch·배치/병합/배치 구조), C2(이웃 Chunk 선별), C3(전체 Query 동시성 제어).
- **제외(2차):** 신규 Selective Fetch, Cross-Encoder Reranking, LLM 응답 생성, range Fetch(D7).
- **완료의 의미:** Milvus에서 실제 Search → C2 → Context Fetch가 연결되고, C3가 전체 Query 실행을 제어하며, 실행 결과를 저장함. 실제 C1의 *물리 I/O 개선*은 별도 비교 실험으로 검증.

## 2. 공통 실행 구조

```text
(query_set, q_idx) → 미리 계산한 Query Vector
    └── C3 Scheduler: 질문 실행 전체(검색~Fetch)의 동시 실행 수를 제어 (m3: 제한 없음 / m7: cap)
         ├── Milvus ANN Search (search_topk=10, ef=64) → hits (순위·점수 보존)
         ├── C2.select_neighbors(context_ids = hits 상위 10) → 추가 이웃 chunk_id
         ├── context_ids ∪ 추가 이웃 (중복 제거)
         ├── C1.fetch_text(읽을 chunk_id, 한 번에) → ({chunk_id: Text}, FetchStats)
         └── 최종 Context, 결과 및 단계별 Metrics 기록
```

- C2는 **Text를 직접 읽지 않음**. ID와 ChunkMeta(doc_idx, flags)만으로 *추가 이웃 ID만* 반환함.
- C1은 **같은 chunk_id**를 받아 해당 레이아웃의 Milvus PK로 변환해 Text를 읽음. PK 변환은 C1 내부에서만.
- Top-10 ∪ 이웃을 **한 번의 Fetch**로 읽음. 기존 C2 코드(bench_pf4 `plan()` → `afetch` 1회)와 같은 순서. 이웃을 ID만으로 고르기 때문에 가능하고, 두 번 읽기(split)보다 지연이 작음(jihyeon cold: 읽은 MB 같고 지연 +7~23%).
- C3는 C2만 감싸는 게 아니라 **Search~Fetch 전체 실행**을 대상으로 동시성 제어(기존 bench_v2 m7과 같은 구조. Fetch만 제한하는 m6과 다름).
- 실험 종류에 따라 모듈 내부 정책을 바꾸되, 공통 입출력 계약과 ID 정의는 유지.

## 3. 공통 데이터·Milvus 설정 (확정값, 경로는 config.yaml)

| 항목 | 1차 확정 | 상태 / 확인 사항 |
|---|---|---|
| 데이터 | NQ, 64-word Chunk, 100만 Chunk, 문서 27,133개. 원본 `/workspace/junlim/data_nq` 참조 | 복사하지 않고 참조, 실행 시작 시 길이·md5 자동 검사(D2). **junlim 수정·이동 금지 동의 대기** |
| Milvus | cap-vdb / Milvus 2.6.23, pymilvus 3.0.3 | 확인됨 |
| 컬렉션 | 검색·본문 모두 `junlim/nq_seq` (A) | **제안(소유자 동의 대기, junlim).** 대안 C: 검색 `jaeeun/nq1m_sel` + 본문 `jihyeon/w13c1_doc`(jihyeon 동의 필요). D(새로 적재)는 디스크 확인 전까지 보류(D1) |
| ANN | HNSW/IP (M=16, efC=200). B0·C3 비교: search_topk=10, context_topk=10, ef=64. C2 비교(대조군 포함): search_topk=40, context_topk=10, ef=64 | ef ≥ search_topk 규칙(D9). search_topk 40의 상위 10개가 search_topk 10 결과와 같은지 첫 실행에서 확인해 실행 메타에 기록. 2차 선별용 100/256은 주석으로만 |
| 이웃 | 같은 문서의 ±1, `none`/`BC1`/`all` | `BC1`은 C2 내부 규칙명(규칙 B + 1위 이웃). Layout 기여 C1과 구분 |
| 질문 | 품질 `nq_test_ans`(2,254), 성능 `nq_test_real`(3,452, 고정 seed 순환), 보조 `c2x_test`(2,000) | 질문 키 (query_set, q_idx)(D3) |
| C1 비교 컬렉션 | `jihyeon/w13c1_doc`, `w13c1_rand` (본문 조회 전용) | **제안(소유자 동의 대기, jihyeon)** |

**중요:** w12 C1 실험(다른 데이터·레이아웃)의 수치는 이번 결과와 직접 비교하지 않음. **w13 C1 실험(11/21)은 같은 NQ 100만 chunk**(`data_nq/chunks.jsonl`, md5 일치)로 만든 doc/rand 배치 컬렉션 `jihyeon/w13c1_doc`, `w13c1_rand`를 썼음. 이 컬렉션은 8차원 가짜 벡터(FLAT L2)라 **검색에는 못 쓰고 본문 조회 전용**이며, 조회 경로도 `get(ids)`로 B0의 `query(pk in)`과 다름. 따라서 w13 수치는 "같은 데이터·다른 조회 경로"로 참고함. NQ는 1차 개발용 **하나의 데이터셋**이며 최종 연구 결과에는 다른 문서 순서 정보를 가진 데이터셋을 추가 검증 예정.

## 4. Baseline 및 실험 설계

- **공통 실행 기준선 B0:** `junlim/nq_seq`에서 `ANN Top-10 → C2 none → query(filter="pk in [...]") 일괄 Fetch → C3 m3`. nq_seq는 **이미 문서 순 배치**이므로 B0는 "공통 실행 기준선"이지 **C1을 끈 기준선이 아님**. C1 효과는 아래 C1 비교로만 판단.
- **C2 비교:** `none` vs `BC1` vs `all` + 대조군 2개(D8). 모두 같은 ANN 검색 결과 사용. 대조군 ANN Top-(10+k)를 만들 수 있도록 C2 비교는 **search_topk 40, context_topk 10, ef 64**로 검색함(D9). search_topk 40 hits의 상위 10개가 search_topk 10 결과와 같은지 첫 실행에서 확인하고 실행 메타에 기록.
  - 대조군: **질문별로 추가 수를 맞춘 ANN Top-(10+k)**, **R3**(같은 ±1 이웃 후보 안에서 무작위, seed 20개). 토큰 수는 보고만 함. 대조군은 C2 모듈이 아니라 실험 계층에서 만듦.
  - 근거: c2x에서 BC1 − none = +5.55%p 중 대부분이 Context 증가 효과였고, BC1 − R3 = +0.99%p였음. 대조군 없이 none과만 비교하면 Context를 늘린 효과를 C2 규칙의 효과로 오인함.
  - 지표: 정답 문자열 포함률(`nq_test_ans` 주 평가, `c2x_test` 독립 검증), 추가 chunk 수, 읽은 바이트, Fetch 지연.
- **C1 비교:** C2가 만든 **읽을 chunk_id 집합을 고정**하고, 같은 ANN 결과(nq_seq 검색)로 `jihyeon/w13c1_doc` vs `w13c1_rand`에서 본문을 읽어 비교. 조회 방식은 batch(`query_pk_in`, `get`)와 individual. w13c1은 가짜 벡터라 검색에 쓰지 않음. **제안(소유자 동의 대기, jihyeon).** 동의를 못 받으면 C1 Milvus 배치 비교는 보류하고 파일 저장소(`config.yaml dataset.file_store`)의 seq/rand 비교만 진행. 요청을 묶었다는 사실만으로 실제 디스크 I/O가 병합되었다고 주장하지 않음. 참고로 warm에서는 doc과 rand 차이가 없었고(p50 5.25 vs 5.20ms), cold(쿼리마다 release/load)에서만 차이가 있었음(BC1 읽은 MB 비율 0.34, 지연 비율 0.56).
- **C3 비교:** 같은 파이프라인·질문 목록에서 m3 vs m7(cap=8). **clients 64가 주 조건**(기존 m7 비교 조건), 16은 보조. 예열 5초, 측정 20초, 3회, 조건 순서 교차. 통합 주 결과의 C2 정책은 **BC1**. 기존 C3 수치(C2=`all`)와 이어 보는 확인용으로 `all` 조건을 **1회만** 실행(m3 vs m7, clients 64). 질문은 `nq_test_real`을 고정 seed 순서로 순환하므로, pseudo 질문을 포함해 무작위로 복원 추출한 기존 C3와 수치가 똑같지 않을 수 있음. 처리량(QPS)뿐 아니라 p50/p95/p99, queue_wait, 오류율 측정.
- **통합:** C1·C2·C3가 모두 연결되는 소규모 시험 후 동일 조건의 결합 효과 검증.

**평가 지표**
- 품질: 정답 문자열 포함률, 읽은 chunk 수, 읽은 바이트.
- 성능: QPS, p50/p95/p99, 단계별 시간(특히 fetch), 요청 수, 오류율.

**측정 단계 구분**
1. 기능 테스트: 고정된 10개 Query로 ID·Text·오류·단계 출력 확인.
2. Warm 측정: 예열·실행 반복 및 조건 순서 교차; 단계 지연, QPS, p95/p99.
3. 실제 Disk I/O 실험: 메모리·캐시 조건과 부하를 통제한 뒤 읽기량·I/O 횟수 측정. 실제 디스크 읽기가 없는 측정은 C1의 물리 I/O 이득 증거로 쓰지 않음.

**I/O 기록 (D11)**
- 질문 단위: `FetchStats`(요청 수, PK 수, 반환 바이트). 모든 경로에서 필수.
- 구간 단위: Milvus 저장소 읽기 메트릭(`internal_cache_cell_access_miss_bytes_total`)의 측정 전후 차이. **서버 전체 값이라 질문 단위 값처럼 다루지 않음.**
- 블록 수와 물리 I/O 주장은 **cold 조건이나 파일 경로에서만** 함.

**공유 서버 주의:** 같은 서버라도 동시 작업/캐시 상태가 다르면 비교 불가. 측정 시간 공유, 동일 부하·Query 목록·반복, 실행 순서 교차, 오류/캐시 조건 기록. 실행마다 load 상태, Milvus reset 기록, swap 사용량, 측정 시각, 원본 md5 검사 결과, 다른 사용자 활동 여부를 실행 메타로 남김(config.yaml `experiment.run_meta`). 남의 컬렉션의 load/release·mmap 설정은 시간표와 동의 없이 바꾸지 않음.

## 5. 공통 인터페이스 (자세한 타입: interfaces.py)

| 기능 | 입력 | 출력 | 담당 |
|---|---|---|---|
| ChunkMeta | 원본 파일 경로(config) | doc_idx, chunk_idx, flags, nbytes, 레이아웃별 pk 매핑(+역매핑), 파일 offset/length | 공통 코어 담당 |
| Search | `(query_set, q_idx)`, search_topk, ef | 순위·점수가 있는 `hits` (공통 chunk_id) | 공통 코어 담당/기존 코드 재사용 |
| C2 | context_ids(순위 순), 정책. ChunkMeta는 생성자 주입 | **R에 없는** 추가 이웃 chunk_id, chunk_id 순 정렬 | C2 담당 |
| C1 | 공통 chunk_id 목록, fetch_mode, batch_impl | `({chunk_id: text}, FetchStats)` | C1 담당 |
| C3 | 질문 키 목록, `run_one` 함수, 실행 정책 | 질문별 실행 결과 (queue_wait·total·service 포함) | C3 담당 |
| Pipeline core | 위 모듈들, 설정값 | 질문별 결과·단계별 시간·FetchStats·오류 | 통합 총괄 |

**stage_ms 키 (interfaces.py 상수):** `embed`(미리 계산한 벡터면 기록 안 함), `search`, `c2_select`(chunk_id↔pk 매핑 포함), `fetch`, `queue_wait`(semaphore 획득 전후로 따로 측정), `total`(큐 진입부터, 대기 포함), `service`(= total − queue_wait).

> 모듈 사이에서 Milvus PK를 직접 전달하지 않음. 모든 단계는 `chunk_id`를 쓰고, 레이아웃별 PK 변환은 C1 내부에서만 함. 기준 ID·원본 데이터 매핑이 맞는지 실행 시작 시 길이·md5로 검사.

## 6. 진행 순서와 분담

1. 총괄이 팀 GitHub 저장소 `ppomee2002/vector-db`의 `w13-pipeline-design` 브랜치, `pipeline-core/` 폴더로 설계 파일(이 문서·`interfaces.py`·`config.yaml`·`REVIEW_FOR_TEAM.md`)을 공유.
2. 팀원별로 AI에게 `REVIEW_FOR_TEAM.md`의 확인용 프롬프트를 맡겨 **자기 기존 코드와의 충돌만** 확인. "충돌 있음" 항목만 D 번호나 interfaces 항목 이름으로 정리.
3. 회의에서 컬렉션 사용 동의, Milvus load/release 시간표, 충돌 항목을 정리한 뒤 **"계약 v1" 고정**. 이후 인터페이스 변경은 PR로만.
4. 각자 `feature/c1`, `feature/c2`, `feature/c3` 브랜치에서 구현(공통 뼈대는 `develop`). 내부 구현은 담당자 자유.
5. 10개 질문 Smoke Test(ID·Text·오류·stage_ms·FetchStats 확인).
6. PR → 순차 통합 → B0 실행 → C2/C1/C3 비교.

이번 주 **필수 목표**: 계약 v1 고정 + 실행 가능한 B0. **가능하면**: C1/C2/C3 모두 연결한 10-Query Smoke Test. 대규모 통합 성능 비교는 그 다음.

## 7. ID·데이터 주의사항

- **질문 키는 (query_set, q_idx).** `nq_test_real`/`nq_test_ans`는 `data_nq/queries.jsonl`의 번호(0~3451)이고, `c2x_test`는 NQ train에서 뽑은 2,000개로 q_idx가 0부터 다시 시작함.
- **pseudo 질문(q_idx 3452~)에는 정답이 없음.** 품질 평가에 쓰지 않음. 성능 측정도 1차는 real 3,452개만 씀.
- **rand 배치는 pk ≠ chunk_id.** `w13c1_rand`는 `pk_rand.npy`(chunk_id → pk)로 변환해야 함. 컬렉션의 `chunk_id` 필드로 결과를 역확인. seq/doc 배치와 검색 컬렉션은 pk == chunk_id.
- **마지막 문서(doc 27132)는 100만 개에서 잘려 끝부분이 없을 수 있음.** ±1 이웃 경계에서 주의.
- **Top-10 ∪ 이웃을 한 번에 Fetch할 때 ID가 정렬되어 순위가 사라짐**(기존 C2 `plan()`은 PK 정렬). 순위는 `QueryResult.hits`로 보존하고 Fetch 목록에서 복원하지 않음.
- 64단어용 정답·flags는 `pf3_out/`. `gold_w100/`(100단어용)과 섞지 않음.

## 8. 결정 사항과 근거

**회의에서 확인할 것 (동의 대기)**
- junlim: `nq_seq` 사용(D1), 원본 파일 참조와 수정·이동 금지(D2).
- jihyeon: `w13c1_doc`·`w13c1_rand` 사용(C1 비교, D1 대안 C).
- 전원: Milvus load/release 시간표(점검 당시 모든 컬렉션 NotLoad)(D12).
- 전원: 각자 기존 코드와의 충돌 항목(`REVIEW_FOR_TEAM.md`).

| # | 결정 | 근거 (CHECK.md) | 상태 |
|---|---|---|---|
| D1 | 통합 컬렉션 A = `junlim/nq_seq`(검색 + 본문). 동의를 못 받으면 C = 검색 `jaeeun/nq1m_sel` + 본문 `jihyeon/w13c1_doc`. D(새로 적재)는 보류 | 기존 C2·C3 실험이 모두 nq_seq, 추가 디스크 0. nq1m_sel과 인덱스 파라미터·본문 일치 확인됨. Milvus 저장소 디스크는 확인 못 함(1-2, 1-3) | 동의 대기 (junlim / C는 jihyeon) |
| B0 | nq_seq에서 ANN Top-10 → C2 none → query(pk in) 일괄 Fetch → C3 m3. 공통 실행 기준선이며 C1을 끈 기준선 아님. C1 효과는 w13c1_doc vs rand 본문 읽기 비교 | nq_seq는 이미 문서 순 배치. w13c1은 같은 데이터의 doc/rand 컬렉션(1-2, 1-6) | 확정 (C1 비교는 jihyeon 동의 대기) |
| D2 | 원본 데이터는 복사하지 않고 참조. 실행 시작 시 길이·md5 자동 검사 | 원본 전부 junlim 폴더. 길이·ID 대응 확인됨(1-1, 1-4) | 동의 대기 (junlim) |
| D3 | 품질 주 평가 `nq_test_ans`(2,254), 보조 `c2x_test`(2,000). 성능 `nq_test_real`(3,452) 고정 seed 순환. 키 (query_set, q_idx) | 정답 있는 질문 2,255개, q_idx 체계 둘, 기존 C3는 pseudo 포함 무작위(1-4, 1-5) | 확정 |
| D4 | C3 clients 64 주 조건, 16 보조. m7 cap 8. 예열 5초, 측정 20초, 3회, 순서 교차 | 기존 m7 비교는 n=64, n=16은 예열에만 사용(1-5) | 확정 |
| D5 | C3 비교 시 C2 정책 BC1(통합 주 결과). 기존 C3와 이어 보는 `all` 조건 1회(m3 vs m7, clients 64) | 기존 C3는 expand=1 = all(1-5) | 확정 |
| D6 | B0 batch = `query(filter="pk in [...]")`. `get(ids)`는 C1 비교 항목 | C2·C3는 query, C1 w13은 get(1-6) | 확정 |
| D7 | range Fetch 1차 제외 | 기존 m4(range filter)가 m3보다 느렸고 rand 배치에서는 의미 없음(1-6) | 확정 |
| D8 | C2 대조군: 질문별로 수를 맞춘 ANN Top-(10+k) + R3(seed 20). 토큰 수는 보고만. 실험 계층에서 생성 | c2x: BC1 − none +5.55%p 대부분이 Context 증가, BC1 − R3 +0.99%p(2-a) | 확정 |
| D9 | B0와 C3 비교는 search_topk 10. C2 비교(대조군 포함)는 search_topk 40, context_topk 10, ef 64(ef ≥ search_topk 만족). 두 topk 분리. search_topk 40 hits의 상위 10개가 search_topk 10과 같은지 첫 실행에서 확인해 실행 메타에 기록. 2차용 100/256은 주석 | Milvus HNSW는 ef ≥ limit 필요, 대조군 ANN Top-(10+k)는 10개보다 많은 검색 결과가 필요, 2차 선별은 Top-100(2-f) | 확정 |
| D10 | 미리 계산한 query_vecs 사용. embed 키는 두되 기록 안 함 | 기존 C2·C3 모두 미리 계산한 벡터 사용(2-e) | 확정 |
| D11 | 질문 단위 FetchStats(요청 수, PK 수, 반환 바이트) + 구간 단위 Milvus 메트릭 차이. 블록 수·물리 I/O 주장은 cold 또는 파일 경로에서만 | Milvus 저장소 읽기 메트릭은 서버 전체 값(2-c) | 확정 |
| 지표 | 품질: 정답 문자열 포함률, 읽은 chunk 수·바이트. 성능: QPS, p50/p95/p99, 단계별 시간(특히 fetch), 요청 수, 오류율 | c2x·C3 기존 지표와 같은 정의 | 확정 |
| D12 | 회의 확인: junlim(nq_seq, 원본 파일), jihyeon(w13c1_doc·rand) 동의, Milvus load/release 시간표 | 점검 당시 모든 컬렉션 NotLoad, 오늘 Milvus reset 5회(1-7) | 회의에서 확인 |
