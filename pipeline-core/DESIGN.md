# Milvus 기반 C1·C2·C3 1차 통합 파이프라인 공통 명세서

**상태:** 2026-11-23 팀 회의용 **공통 명세 (계약 v1 후보)**. 설계는 총괄이 확정했다. junlim 동의로 D1(nq_seq)·D2(원본 참조)·D12(load 시간대)는 확정(2026-11-23). 회의에서는 남은 동의(jihyeon w13c1, 관리자 cold 설정)와 팀원별 기존 코드와의 충돌만 확인한 뒤 계약 v1로 고정한다(8절). 성능 검증은 아직 하지 않았다. 서버·데이터 점검 근거는 `CHECK.md`(2026-11-22)에 있다.

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
| 데이터 | NQ, 64-word Chunk, 100만 Chunk, 문서 27,133개. 원본 `/workspace/junlim/data_nq` 참조 | 복사하지 않고 참조, 수정·이동 금지(D2) **확정(junlim 동의, 2026-11-23)**. 일반 실행은 크기·수정 시각만 확인(캐시), 전체 md5는 별도 명령 `verify_md5.py`로 하루 한 번 |
| Milvus | cap-vdb / Milvus 2.6.23, pymilvus 3.0.3 | 확인됨 |
| 컬렉션 | 검색·본문 모두 `junlim/nq_seq` (A) | **확정(junlim 동의, 2026-11-23)**(D1). nq_seq의 mmap·인덱스 설정은 바꾸지 않음. `jaeeun/nq1m_sel`은 기존 B0 호환 확인(`runs/b0_smoke.yaml`)에만 씀 |
| ANN | HNSW/IP (M=16, efC=200). B0·C3 비교: search_topk=10, context_topk=10, ef=64. C2 비교(대조군 포함): search_topk=40, context_topk=10, ef=64 | ef ≥ search_topk 규칙(D9). search_topk 40의 상위 10개가 search_topk 10 결과와 같은지 첫 실행에서 확인해 실행 메타에 기록. 2차 선별용 100/256은 주석으로만 |
| 이웃 | 같은 문서의 ±1, `none`/`BC1`/`all` | `BC1`은 C2 내부 규칙명(규칙 B + 1위 이웃). Layout 기여 C1과 구분 |
| 질문 | 품질 `nq_test_ans`(2,254), 성능 `nq_test_real`(3,452, 고정 seed 순환), 보조 `c2x_test`(2,000) | 질문 키 (query_set, q_idx)(D3) |
| C1 비교 컬렉션 | `jihyeon/w13c1_doc`, `w13c1_rand` (본문 조회 전용) | **제안(소유자 동의 대기, jihyeon)** |

**중요:** w12 C1 실험(다른 데이터·레이아웃)의 수치는 이번 결과와 직접 비교하지 않음. **w13 C1 실험(11/21)은 같은 NQ 100만 chunk**(`data_nq/chunks.jsonl`, md5 일치)로 만든 doc/rand 배치 컬렉션 `jihyeon/w13c1_doc`, `w13c1_rand`를 썼음. 이 컬렉션은 8차원 가짜 벡터(FLAT L2)라 **검색에는 못 쓰고 본문 조회 전용**이며, 조회 경로도 `get(ids)`로 B0의 `query(pk in)`과 다름. 따라서 w13 수치는 "같은 데이터·다른 조회 경로"로 참고함. NQ는 1차 개발용 **하나의 데이터셋**이며 최종 연구 결과에는 다른 문서 순서 정보를 가진 데이터셋을 추가 검증 예정.

## 4. Baseline 및 실험 설계

- **공통 실행 기준선 B0:** `junlim/nq_seq`에서 `ANN Top-10 → C2 none → query(filter="pk in [...]") 일괄 Fetch → C3 m3`. nq_seq는 **이미 문서 순 배치**이므로 B0는 "공통 실행 기준선"이지 **C1을 끈 기준선이 아님**. C1 효과는 아래 C1 비교로만 판단.
  - B0의 Fetch = 기본 Milvus Fetch 어댑터 `c1.impl=milvus_query_ref`(`modules/c1_milvus_query_reference.py`): AsyncMilvusClient `query(filter="pk in [...]")`, PK를 512개씩 나눠 요청(기존 C3 `bench_v2.py`와 같음), 최적화 없음. **C1 기여 구현은 C1 담당자 어댑터**로 따로 둔다. 정확성 확인 설정: `runs/correctness_milvus_query.yaml`.
- **C2 비교:** `none` vs `BC1` vs `all` + 대조군 2개(D8). 모두 같은 ANN 검색 결과 사용. 대조군 ANN Top-(10+k)를 만들 수 있도록 C2 비교는 **search_topk 40, context_topk 10, ef 64**로 검색함(D9). search_topk 40 hits의 상위 10개가 search_topk 10 결과와 같은지 첫 실행에서 확인하고 실행 메타에 기록.
  - 대조군: **질문별로 추가 수를 맞춘 ANN Top-(10+k)**, **R3**(같은 ±1 이웃 후보 안에서 무작위, seed 20개). 토큰 수는 보고만 함. 대조군은 C2 모듈이 아니라 실험 계층에서 만듦.
  - 근거: c2x에서 BC1 − none = +5.55%p 중 대부분이 Context 증가 효과였고, BC1 − R3 = +0.99%p였음. 대조군 없이 none과만 비교하면 Context를 늘린 효과를 C2 규칙의 효과로 오인함.
  - 지표: 정답 문자열 포함률(`nq_test_ans` 주 평가, `c2x_test` 독립 검증), 추가 chunk 수, 읽은 바이트, Fetch 지연.
- **C1 비교:** C2가 만든 **읽을 chunk_id 집합을 고정**하고, 같은 ANN 결과(nq_seq 검색)로 `jihyeon/w13c1_doc` vs `w13c1_rand`에서 본문을 읽어 비교. 조회 방식은 batch(`query_pk_in`, `get`)와 individual. w13c1은 가짜 벡터라 검색에 쓰지 않음. **제안(소유자 동의 대기, jihyeon).** 동의를 못 받으면 C1 Milvus 배치 비교는 보류하고 파일 저장소(`config.yaml dataset.file_store`)의 seq/rand 비교만 진행. 요청을 묶었다는 사실만으로 실제 디스크 I/O가 병합되었다고 주장하지 않음. 참고로 warm에서는 doc과 rand 차이가 없었고(p50 5.25 vs 5.20ms), cold(쿼리마다 release/load)에서만 차이가 있었음(BC1 읽은 MB 비율 0.34, 지연 비율 0.56).
- **C3 공식 비교 = 기본 Milvus 일괄 Fetch 어댑터(`milvus_query_ref`) + warm 환경.** C1 구현을 기다리지 않아도 C3 비교를 진행할 수 있는 구조다. **C3 비교에는 `file_ref`를 쓰지 않는다:** 파일 Fetch는 스레드 풀에서 돌아서 스레드 수가 C3 동시성 제어(m3 무제한 / m7 cap)와 별개인 **숨은 동시성 상한**이 된다(예전에는 `asyncio.to_thread` 기본 풀 = min(32, CPU+4)). 지금 `file_ref`는 기본 풀에 의존하지 않고 `c1.file_ref_threads`를 반드시 적게 했고(없으면 실행 거부, B0 정확성 확인은 8) 실행 메타 `c1_threads`에 기록하지만, 그래도 Milvus 경로와 다른 상한이므로 C3 비교 대상이 아니다.
- **C1 Layout 비교 = C1 담당자 어댑터 + cold 환경(관리자 설정 후).** 아래 C1 비교 항목과 9절 참고.
- **C3 비교:** 같은 파이프라인·질문 목록에서 m3 vs m7(cap=8). **clients 64가 주 조건**(기존 m7 비교 조건), 16은 보조. 예열 5초, 측정 20초, 3회, 조건 순서 교차. 통합 주 결과의 C2 정책은 **BC1**. 기존 C3 수치(C2=`all`)와 이어 보는 확인용으로 `all` 조건을 **1회만** 실행(m3 vs m7, clients 64). 질문은 `nq_test_real`을 고정 seed 순서로 순환하므로, pseudo 질문을 포함해 무작위로 복원 추출한 기존 C3와 수치가 똑같지 않을 수 있음. 처리량(QPS)뿐 아니라 p50/p95/p99, queue_wait, 오류율 측정.
- **통합:** C1·C2·C3가 모두 연결되는 소규모 시험 후 동일 조건의 결합 효과 검증.

**평가 지표**
- 품질: 정답 문자열 포함률, 읽은 chunk 수, 읽은 바이트.
- 성능: QPS, p50/p95/p99, 단계별 시간(특히 fetch), 요청 수, 오류율.

**QPS 정의 (summary.json, 시간 기반 실행)**
- `qps` (기존 C3와 같은 집계, 그대로 유지): `qps = N_meas / duration_s`. N_meas = `is_warmup=False`인 결과 수 = 시작 후 warmup_s 이후에 끝난 결과 수(측정 구간이 끝난 뒤 끝난 꼬리 결과 포함, 오류 포함). 기존 `bench_v2.py`의 `r[0] >= t_warm` 필터 / 측정 구간 길이와 같은 방식.
- `strict_qps` (추가): `strict_qps = N_strict / duration_s`. N_strict = `warmup_s ≤ t_done ≤ warmup_s + duration_s`인 결과 수(t_done = run_many 시작부터 그 질문의 run_one이 끝난 시각, 실행기가 기록). 측정 구간이 끝난 뒤에 끝난 꼬리 결과는 빠지므로 항상 `strict_qps ≤ qps`. 차이 = `n_tail_after_window / duration_s`.
- 호환: summary.json의 기존 키(`qps`, `qps_basis` 등)는 이름·의미가 그대로이고 `strict_qps`, `n_strict`, `n_tail_after_window`, `strict_qps_basis`만 추가됐다. 개수 기반 실행(max_queries)에서 `qps`는 기존대로 `N / wall_s`, `strict_qps`는 None.

**성능 측정과 정확성 검증 분리**
- 성능 측정 실행의 기본값은 `run.evaluate=false`(config.yaml). 평가(정답 문자열 포함률)와 본문 복사만 끄고, **원문 조회 자체는 실제로 수행**한다(C1 `fetch_text`가 실제 저장소에서 본문을 읽음, 가짜 데이터 없음).
- 정확성 검증은 별도 설정 `runs/correctness_file_ref.yaml`(검색 nq_seq + file_ref), `runs/correctness_milvus_query.yaml`(검색·본문 nq_seq + milvus_query_ref = nq_seq B0). `evaluate=true`, 질문 10개 개수 기반, `chunks.jsonl` 원문 대조 유지.

**측정 단계 구분**
1. 기능 테스트: 고정된 10개 Query로 ID·Text·오류·단계 출력 확인.
2. Warm 측정: 예열·실행 반복 및 조건 순서 교차; 단계 지연, QPS, p95/p99.
3. 실제 Disk I/O 실험: 메모리·캐시 조건과 부하를 통제한 뒤 읽기량·I/O 횟수 측정. 실제 디스크 읽기가 없는 측정은 C1의 물리 I/O 이득 증거로 쓰지 않음.

**I/O 기록 (D11)**
- 질문 단위: `FetchStats`(요청 수, PK 수, 반환 바이트). 모든 경로에서 필수.
- 구간 단위: Milvus 저장소 읽기 메트릭(`internal_cache_cell_access_miss_bytes_total`)의 측정 전후 차이. **서버 전체 값이라 질문 단위 값처럼 다루지 않음.**
- 블록 수와 물리 I/O 주장은 **cold 조건이나 파일 경로에서만** 함.

**Milvus 실행 안전 규칙 (MilvusGuard, `modules/run_meta.py`)**
- 실행 전 대상 컬렉션의 load 상태를 기록. 사용 중 검사에서는 **대상 컬렉션으로 들어온 요청도 남의 요청**으로 본다(실행 전이므로). 다른 컬렉션이 load되어 있거나, 데이터 요청이 있거나, 메트릭을 못 읽으면 중단.
- 대상 컬렉션이 **이미 Loaded면 기본 중단**. `run.allow_preloaded_targets: true`를 명시했을 때만 진행하고, 그때도 load·release하지 않는다.
- release 대상 = **원래 NotLoad였고 이번 실행이 직접 load한 컬렉션만**. 원래 Loaded였던 컬렉션은 절대 자동 release하지 않는다.
- release 직전 다시 표본(`run.release_check_s`)을 보고, 대상 컬렉션 요청·Milvus 재시작·메트릭 조회 실패가 있으면 release하지 않고 실행 메타 `release_report`/`release_skipped`에 이유와 **수동 release 명령**을 남긴다(종료 코드 6).
- 예약 시간대 `run.reserved_window`(KST): 측정 실행(`run.purpose=measure`)과 cold/cache_constrained 실행은 시간대 밖이면 거부(종료 코드 7). 정확성 확인(`purpose=correctness`, 개수 기반)은 적었을 때만 확인.
- 내 실행끼리 겹치지 않게 내 폴더의 잠금 파일(`locks/milvus_run.lock`). **팀 공통 잠금은 합의 전이라 만들지 않음(미결, 8절).**

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

1. 총괄이 팀 GitHub 저장소 `ppomee2002/vector-db`의 `pipeline-design` 브랜치, `pipeline-core/` 폴더로 설계 파일(이 문서·`interfaces.py`·`config.yaml`·`REVIEW_FOR_TEAM.md`)과 공통 실행기를 공유.
2. 팀원별로 AI에게 `REVIEW_FOR_TEAM.md`의 확인용 프롬프트를 맡겨 **자기 기존 코드와의 충돌만** 확인. "충돌 있음" 항목만 D 번호나 interfaces 항목 이름으로 정리.
3. 회의에서 컬렉션 사용 동의, Milvus load/release 시간표, 충돌 항목을 정리한 뒤 **"계약 v1" 고정**. 이후 인터페이스 변경은 PR로만.
4. 각자 **본인 작업 폴더**(예: `/workspace/<내 이름>/`)에 저장소를 clone하거나 `pipeline-core/`를 복사해서 `feature/c1`, `feature/c2`, `feature/c3` 브랜치에서 어댑터를 구현(공통 뼈대는 `pipeline-design`). 내부 구현은 담당자 자유. 총괄 폴더(`/workspace/jaeeun/pipeline-core`)에 직접 코드를 넣지 않는다.
5. 본인 폴더에서 계약 테스트(`experiments/contract_test.py`, **C1 동시 실행 검사 포함**) FAIL 0 → 10개 질문 Smoke Test(ID·Text·오류·stage_ms·FetchStats 확인).
6. PR(`pipeline-design` 대상) → 총괄이 순차 통합 → B0 실행 → C2/C1/C3 비교.

이번 주 **필수 목표**: 계약 v1 고정 + 실행 가능한 B0. **가능하면**: C1/C2/C3 모두 연결한 10-Query Smoke Test. 대규모 통합 성능 비교는 그 다음.

## 7. ID·데이터 주의사항

- **질문 키는 (query_set, q_idx).** `nq_test_real`/`nq_test_ans`는 `data_nq/queries.jsonl`의 번호(0~3451)이고, `c2x_test`는 NQ train에서 뽑은 2,000개로 q_idx가 0부터 다시 시작함.
- **pseudo 질문(q_idx 3452~)에는 정답이 없음.** 품질 평가에 쓰지 않음. 성능 측정도 1차는 real 3,452개만 씀.
- **rand 배치는 pk ≠ chunk_id.** `w13c1_rand`는 `pk_rand.npy`(chunk_id → pk)로 변환해야 함. 컬렉션의 `chunk_id` 필드로 결과를 역확인. seq/doc 배치와 검색 컬렉션은 pk == chunk_id.
- **마지막 문서(doc 27132)는 100만 개에서 잘려 끝부분이 없을 수 있음.** ±1 이웃 경계에서 주의.
- **Top-10 ∪ 이웃을 한 번에 Fetch할 때 ID가 정렬되어 순위가 사라짐**(기존 C2 `plan()`은 PK 정렬). 순위는 `QueryResult.hits`로 보존하고 Fetch 목록에서 복원하지 않음.
- 64단어용 정답·flags는 `pf3_out/`. `gold_w100/`(100단어용)과 섞지 않음.

## 8. 결정 사항과 근거

**확정 (junlim 동의, 2026-11-23):** `nq_seq` 사용(D1), 원본 파일 참조와 수정·이동 금지(D2), load 시간대(D12).

**회의에서 확인할 것 (동의 대기)**
- jihyeon: `w13c1_doc`·`w13c1_rand` 사용(C1 비교, D1 대안 C).
- 관리자: cold / cache_constrained 실험용 캐시 설정(9절). 다른 팀원에게 영향.
- 전원: 각자 기존 코드와의 충돌 항목(`REVIEW_FOR_TEAM.md`).

**미결 사항**
- **팀 공통 잠금:** 합의된 방식이 없어 만들지 않았다. 지금은 예약 시간대(`run.reserved_window`)와 내 폴더 잠금 파일(내 실행끼리만)뿐이라, 다른 팀원의 실행과는 MilvusGuard 사용 중 검사로만 겹침을 막는다.
- **고부하 실험 중 Milvus가 종료된 원인:** 관리자 확인 필요(2026-11-23 13:29 KST에도 Milvus 프로세스가 다시 시작된 기록이 메트릭에 있음). **원인이 확인되기 전에는 m3, clients=64 부하 실험을 실행하지 않는다.**

| # | 결정 | 근거 (CHECK.md) | 상태 |
|---|---|---|---|
| D1 | 통합 컬렉션 A = `junlim/nq_seq`(검색 + 본문). nq_seq의 mmap·인덱스 설정은 바꾸지 않음 | 기존 C2·C3 실험이 모두 nq_seq, 추가 디스크 0. nq1m_sel과 인덱스 파라미터·본문 일치 확인됨(1-2, 1-3) | 확정(junlim 동의, 2026-11-23) |
| B0 | nq_seq에서 ANN Top-10 → C2 none → query(pk in) 일괄 Fetch → C3 m3. 공통 실행 기준선이며 C1을 끈 기준선 아님. C1 효과는 w13c1_doc vs rand 본문 읽기 비교 | nq_seq는 이미 문서 순 배치. w13c1은 같은 데이터의 doc/rand 컬렉션(1-2, 1-6) | 확정 (C1 비교는 jihyeon 동의 대기) |
| D2 | 원본 데이터는 복사하지 않고 참조, 수정·이동 금지. 실행 시작 시 길이 검사 + md5(일반 실행은 크기·수정 시각, 전체 md5는 하루 한 번 별도 명령) | 원본 전부 junlim 폴더. 길이·ID 대응 확인됨(1-1, 1-4) | 확정(junlim 동의, 2026-11-23) |
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
| D12 | Milvus load 시간대. 실행마다 `run.reserved_window`(KST)에 적고, 측정 실행은 그 밖에서 거부. load/release는 MilvusGuard 규칙(4절)으로만 | 점검 당시 모든 컬렉션 NotLoad, 오늘 Milvus reset 5회(1-7) | 확정(junlim 동의, 2026-11-23). jihyeon(w13c1_doc·rand) 동의는 별도로 대기 |

## 9. 실험 환경 구분

`config.yaml`의 `experiment.cache_condition`으로 고른다(`warm` / `cold` / `cache_constrained`, 기본 `warm`). **이 값은 기록과 선택만 한다.** 실제 캐시 제어(캐시 비우기, 캐시 예산 제한)는 관리자 설정과 실행 스크립트가 담당한다. 관리자가 적용한 설정 내용은 `experiment.admin_cache_setting`에 적는다.

| 조건 | 정의 | 용도 |
|---|---|---|
| **warm** | 공통 파이프라인이 정상 동작하는 상태(캐시를 따로 비우거나 제한하지 않음). 처리량 측정 | **C3 비교의 기본 조건**(기본 Milvus Fetch `milvus_query_ref`), B0, C2 비교 |
| **cold** | 질문마다(또는 조건마다) 캐시를 비운 초기 상태 | C1 배치 효과(doc vs rand) 확인. C1 담당자 어댑터 |
| **cache_constrained** | 캐시 예산을 제한한 상태로 여러 질문을 계속 처리해서 실제 저장소 읽기가 계속 일어나는 조건 | C1·C2·C3 결합 효과 검증 |

- **cold와 cache_constrained는 서로 다른 조건이다.** cold는 "비운 직후의 초기 상태"를 질문(또는 조건)마다 다시 만드는 것이고, cache_constrained는 비우지 않고 계속 돌리되 캐시 예산이 작아서 읽기가 계속 저장소까지 가는 정상 상태다. 한쪽 결과로 다른 쪽을 대신하지 않는다.
- **유효성 규칙:** cold와 cache_constrained 실행은 **측정 구간의 저장소 읽기 메트릭**(`internal_cache_cell_access_miss_bytes_total` 차이)이 0보다 확실히 커야(`experiment.io_valid_min_bytes`, 기본 1 MiB 초과) 유효하다. 아니면 실행 메타 `cache_condition_evidence.verdict`에 **"실제 I/O 없음"**으로 표시하고 C1 효과의 근거로 쓰지 않는다. 측정 중 Milvus가 재시작되면 무효.
- 실행 메타 `cache_condition_evidence`: 적용한 조건, 관리자 설정 내용, 측정 전후 저장소 읽기 메트릭 차이(시간 기반이면 측정 구간 [warmup_s, warmup_s+duration_s]만, 개수 기반이면 run_many 전체), 유효 판정.
- **cold와 cache_constrained는 관리자 설정이 필요하고 다른 팀원에게 영향을 주므로 예약 시간대(`run.reserved_window`)에만, 다른 사람이 Milvus를 쓰지 않을 때만(MilvusGuard 사용 중 검사 통과) 실행한다.** 실행기는 이 두 조건에서 시간대가 없거나 밖이면 거부하고, 원본 md5를 다시 계산해야 하는 상황이면(파일 변경) 계산하지 않고 중단한다(측정 직전 대용량 읽기 금지). 전체 md5 검사(`verify_md5.py`)도 cold 직전에는 하지 않는다.
