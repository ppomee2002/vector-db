# CHANGES: 설계 초안 → 공통 명세 (계약 v1 후보)

- 날짜: 2026-11-22. 원본 백업: `/workspace/jaeeun/pipeline-core_orig_1122/`
- 근거: `CHECK.md`(서버·데이터 점검)와 총괄 결정 사항(D1~D12). 이번 작업에서는 서버·팀원 폴더를 추가로 조회하지 않음.
- 바꾼 파일: `DESIGN.md`, `interfaces.py`, `config.yaml`. 새로 만든 파일: `CHANGES.md`, `REVIEW_FOR_TEAM.md`. `modules/`, `experiments/`, `main.py`는 그대로.

## DESIGN.md
| 위치 | 변경 | 이유 |
|---|---|---|
| 제목·상태 | "제안서/설계 초안" → "공통 명세서 (계약 v1 후보)". 회의 범위를 동의·시간표·충돌 확인으로 한정 | 총괄이 설계를 확정함 |
| 1절 | 제외 항목에 range Fetch 추가 | D7 |
| 2절 | 입력을 (query_set, q_idx), hits(순위·점수), Top-10 ∪ 이웃 한 번에 Fetch, FetchStats 반환으로 갱신. 기존 C2 코드와 같은 순서라는 근거, m7 구조 명시 | CHECK 2-(d), 1-5 |
| 3절 표 | 제안값 → 확정값. 컬렉션 A + 대안 C, ANN topk 분리, 질문 집합, C1 비교 컬렉션 추가. 동의 대기 항목 표시 | D1, D2, D3, D9 |
| 3절 "중요" | "기존 C1은 약 15만 chunk / FLAT·COSINE" → w13 C1은 같은 NQ 100만 chunk, doc/rand 컬렉션, 가짜 벡터, 본문 조회 전용(get 경로) | CHECK (g)의 첫 번째 행 |
| 4절 | B0 정의(공통 실행 기준선, C1을 끈 기준선 아님), C2 대조군(D8)과 c2x 근거, C1 비교(w13c1 doc vs rand, 동의 대기와 대안), C3 조건(D4·D5), 평가 지표, I/O 기록(D11), 실행 메타 항목 | 결정 사항 |
| 5절 | ChunkMeta 행 추가, C2·C1·C3 입출력을 interfaces.py와 맞춤, stage_ms 키 정의 | interfaces 변경 반영 |
| 6절 | 설계 공유 → AI 충돌 확인 → 회의 후 계약 v1 고정(이후 PR로만 변경) → feature 구현 → Smoke Test → 통합 | 진행 방식 변경 |
| 7절 (신규) | ID·데이터 주의사항: (query_set, q_idx), pseudo 질문에 정답 없음, rand pk 변환, 마지막 문서 잘림, PK 정렬로 순위가 사라지는 문제, 100단어 파일과 섞지 말 것 | CHECK 1-4, 2-(d) |
| 8절 (신규) | 회의 확인 목록 + D1~D12·B0·지표 결정표(결정 / 근거 / 상태) | 결정 사항 기록 |

## interfaces.py
| 항목 | 변경 | 이유 |
|---|---|---|
| 모듈 docstring | 공통 규칙 4개(줄 번호 = chunk_id, PK 변환은 C1 내부에서만, 순위 보존, 질문 키). 검색 컬렉션은 pk == chunk_id라 Search는 항등 확인만 한다고 명시 | 계약 고정 |
| 타입 별칭 | `BatchImpl`, `Layout`, `QuerySet`, `QueryKey` 추가. `FetchMode.range`는 1차 제외(D7) 주석 | D3, D6, D7 |
| stage_ms 상수 | `STAGE_*`, `STAGE_KEYS` 추가(embed, search, c2_select, fetch, queue_wait, total, service) | 2-(e) |
| `Settings` | `search_collection`/`fetch_collection`, `layout`, `query_set`, `search_topk`/`context_topk` 분리, `batch_impl`, clients 기본 64. ef ≥ search_topk 규칙 docstring | D1, D4, D6, D9 |
| `ChunkMeta`, `ChunkMetaLoader` (신규) | doc_idx, chunk_idx, flags, nbytes, 레이아웃별 pk_of_chunk/chunk_of_pk, Optional offset/length. 길이·md5 검사 계약 | 2-(b), D2 |
| `SearchHit` | rank 0부터, score는 항상 채움 | 2-(f) |
| `FetchStats` (신규) | n_requests, n_pks, bytes_returned 필수, blocks, device_bytes Optional(Milvus에서는 질문 단위로 얻을 수 없음) | 2-(c), D11 |
| `QueryResult` | query_set, hits, context_ids, fetch_stats 추가. `top10_ids` 삭제(hits로 대체) | D3, 2-(d), 2-(f) |
| `SearchAdapter` | 입력을 `q_idx` → `QueryKey` | D3 |
| `C2NeighborAdapter` | ChunkMeta 생성자 주입, 반환 규칙(R에 없는 이웃, chunk_id 정렬), 대조군은 실험 계층 | 2-(b), D8 |
| `C1FetchAdapter` | `fetch_text(..., batch_impl)` → `(texts, FetchStats)` | 2-(c), D6 |
| `C3SchedulerAdapter` | 질문 키 목록을 받음. queue_wait/total/service는 스케줄러가 기록 | 2-(e) |
| 문법 | `ast.parse` 통과 | – |

## config.yaml
| 항목 | 변경 | 이유 |
|---|---|---|
| dataset | `root: TO_BE_CONFIRMED` → 실제 경로. 파일별 경로, 소유자, 확인된 md5(chunks, gold, flags), 파일 저장소(seq/doc/rand, len, 4KB 블록) | CHECK 1-1, D2 |
| milvus | `database`/`collection` 하나 → `search`/`fetch` 분리, 기본값 A(nq_seq), 동의 대기와 대안 C 주석. `c1_compare`(w13c1_doc/rand, rand pk 매핑) 추가. `load_schedule: TO_BE_CONFIRMED` | D1, D12 |
| ann | `topk` → `search_topk`/`context_topk`, ef 규칙, 2차값 주석 | D9 |
| c2 | `integration_policy: BC1`, 대조군 주석 | D5, D8 |
| c1 | `batch_impl: query_pk_in`, compare 항목, range 제외 | D6, D7 |
| c3 | clients 16 → 64(주), `clients_aux: 16`, 예열·측정·반복·순서, `c2_policy: BC1`, `legacy_check`(all, 1회) | D4, D5 |
| query_sets (신규) | nq_test_real / nq_test_ans / c2x_test 경로·개수·용도 | D3 |
| metrics (신규) | 품질·성능 지표 이름 | 평가 지표 결정 |
| experiment | 반복 3회는 유지. `run_meta` 항목 추가(측정 시각, load 상태, reset 기록, swap, md5 검사, 다른 사용자 활동, 구간 메트릭 차이) | 공유 서버 기록, D11 |
| 문법 | `yaml.safe_load` 통과 | – |

## 추가 수정 (2026-11-22, 팀 검토 전 보완 5건)
| # | 파일 | 변경 | 이유 |
|---|---|---|---|
| 1 | DESIGN.md 3·4·8절(D9), config.yaml `ann.c2_compare`·`experiment.run_meta`, interfaces.py `Settings` docstring | B0와 C3 비교는 search_topk 10. C2 비교(대조군 포함)는 search_topk 40, context_topk 10, ef 64. search_topk 40 상위 10개와 search_topk 10 결과의 일치 여부를 첫 실행에서 확인하고 실행 메타(`top10_consistency`)에 기록 | D8 대조군 ANN Top-(10+k)는 10개보다 많은 검색 결과가 필요해서 D9와 충돌했음 |
| 2 | interfaces.py `C3SchedulerAdapter.run_many`, `QueryResult`, REVIEW_FOR_TEAM.md C3 표·프롬프트 | `warmup_s`, `duration_s` 인자 추가. query_keys는 고정 seed 순서 목록을 순환하고, 시간이 끝나면 멈춤. `QueryResult.is_warmup`(기본 False) 추가, 집계에서 예열 구간 제외 | 기존 C3 측정(예열 5초 + 측정 20초, 시간 기반)을 그대로 구현할 수 있게 |
| 3 | DESIGN.md 8절 회의 확인 목록, REVIEW_FOR_TEAM.md C1 표 | "B0의 C1 비교" → "C1 비교" | C1 비교는 B0에 포함되지 않음 |
| 4 | REVIEW_FOR_TEAM.md 설계 파일 위치·확인용 프롬프트 4개·단톡방 메시지, DESIGN.md 6절 1단계 | 공유 위치를 서버 폴더 → GitHub `ppomee2002/vector-db`의 `w13-pipeline-design` 브랜치, `pipeline-core/` 폴더로. 프롬프트는 "브랜치에서 파일을 받아 읽고", 비교 대상은 서버 경로와 팀 저장소 브랜치를 함께 적음. 단톡방 메시지에 `<링크>` 자리 | 팀원이 자기 컴퓨터의 AI로도 확인할 수 있게 |
| 5 | CHANGES.md | 이 절 추가 | 변경 기록 |
| 검사 | interfaces.py, config.yaml | `python -m py_compile`, `yaml.safe_load` 통과 | – |

## Smoke Test 후 계약 보완 (2026-11-23, 계약 v1 고정 전, 총괄 결정)
- 근거: SMOKE.md 5절(B0 10-Query Smoke Test에서 계약과 안 맞아 고치지 못한 항목). 변경 전 파일은 세션 scratchpad에 백업.
- 검사: `python -m py_compile`(interfaces.py, main.py, modules/*.py, experiments/contract_test.py), `yaml.safe_load`(config.yaml, runs/b0_smoke.yaml), 계약 테스트 PASS 93 / FAIL 0 / SKIP 4 / WARN 4, B0 Smoke Test 2회(오류 0, 결과 검사 위반 0).

| # | 파일 | 변경 | 이유 |
|---|---|---|---|
| 1 | interfaces.py `C3SchedulerAdapter.run_many` | `max_queries: Optional[int] = None` 추가. 정수면 query_keys를 고정 순서로(필요하면 순환) 정확히 max_queries개 실행하고 멈춤, warmup_s/duration_s 무시, 모든 결과 is_warmup=False. None이면 기존 시간 기반 | 품질 평가·스모크용 "목록 1회 실행" 모드가 계약에 없었음 |
| 1 | modules/c3_serial_reference.py, main.py, experiments/contract_test.py | reference C3를 개수 기반 계약대로 수정. 실행기는 `run.max_queries`를 항상 전달(시그니처 검사 제거). 계약 테스트에 개수 기반 시나리오 2개(목록보다 큰 수로 순환, duration보다 오래 걸려도 멈추지 않음)와 `max_queries_count` 검사 | 〃 |
| 2 | interfaces.py 모듈 docstring(공통 규칙 5), `SearchAdapter` docstring | 모든 어댑터(Search, C1, C2, C3) 생성자 `__init__(meta: ChunkMeta, cfg: dict)`. 추가 인자는 기본값 있는 키워드만 | C1/C3/Search 생성자가 정해져 있지 않았음 |
| 2 | modules/search_milvus.py | `MilvusSearchAdapter(meta, cfg, queries=None)`로 변경, doc_idx 대조 여부는 cfg `run.verify_search_doc_idx`에서 읽음 | 〃 |
| 3 | config.yaml | `c1.impl: file_ref`, `c2.impl: none_ref`, `c3.impl: serial_ref`(reference 어댑터) 추가. `run` 블록 추가(name, query_set, n_queries, max_queries, warmup_s, duration_s, evaluate, text_check_n, verify_search_doc_idx, milvus_allowed_load, preflight_sample_s, metrics_url, out_dir) | 구현 선택·실행 설정이 overlay에만 있었음 |
| 3 | runs/b0_smoke.yaml, main.py, modules/registry.py | overlay는 덮어쓸 값만 남김(검색 컬렉션 nq1m_sel, run 일부). 실행기의 하드코딩 기본값 제거, run 키나 impl이 없으면 중단 | 기본값의 출처를 config.yaml 하나로 |
| 4 | interfaces.py `C1FetchAdapter.fetch_source()`, modules/c1_file_reference.py, main.py, contract_test.py | C1이 실제 본문 출처를 `"file:<경로>"` 또는 `"milvus:<db>/<collection>"`으로 밝힘. 실행 메타 `fetch_source`에 c1_impl, source, settings_fetch_collection, matches_settings 기록(다르면 note와 함께 둘 다). 계약 테스트가 존재·형식 검사 | 파일 저장소에서 읽어도 Settings에는 junlim/nq_seq가 남았음 |
| 5 | modules/chunk_meta.py | config md5가 null인 파일은 계산해서 실행 메타에 기록(`computed`), 값이 있으면 검사(불일치 시 중단) | ChunkMeta 계약대로 |
| 5 | config.yaml `dataset.files` | md5 7개 채움(doc_idx `3f79e043…`, pk_seq `db6977c5…`, chunk_vecs `97bcdc9a…`, queries `928acd35…`, query_vecs `b03bfe8a…`, qrels_chunk `f8d2753f…`, chunk_meta `450e85b9…`). 출처: `results/b0_smoke_20261123_112611/run_meta.json`(2026-11-23 11:26 KST 계산). 다음 실행(11:27)에서 10개 모두 일치 | 원본 고정 검사 |
| 6 | interfaces.py `FetchStats` | `bytes_returned` = 반환된 본문 텍스트의 UTF-8 바이트 합(응답·메타·블록 크기 제외). n_requests, n_pks 뜻도 명시 | 경로마다 정의가 다를 수 있었음 |
| 6 | modules/c1_file_reference.py, contract_test.py | reference C1은 반환 텍스트를 UTF-8로 인코딩해 셈. 계약 테스트의 bytes 검사를 WARN → FAIL(`bytes_returned_utf8`) | 〃 |
| 7 | main.py | `milvus_reset_log: "not collected"` 유지 | 결정 |
| – | SMOKE.md | 1~3절 갱신, 5절 표에 해결 여부 표시(1~3, 5~7번 해결, `milvus.search` 값은 D1 미결, reset 기록은 유지) | 기록 |
| – | 그대로 둔 것 | DESIGN.md, `modules/C1.py`·`C2.py`·`C3.py`(담당자용 빈 파일), 팀원 코드 | 지시 |

## C3 검토 반영 (2026-11-23)
- 근거: C3 담당자 검토 의견, C1 실험 조건(cold), junlim 동의(D1 nq_seq, D2 원본 참조·수정·이동 금지, D12 load 시간대).
- 그대로 둔 것: `interfaces.py`(공통 계약), `modules/C1.py`·`C2.py`·`C3.py`, 기존 결과(`results/`)와 데이터셋, 팀원 코드. 변경 전 파일은 세션 scratchpad `before_c3_review/`에 백업.
- 검사: 아래 표 끝과 SMOKE.md 6·7절.

| # | 파일 | 변경 | 이유 |
|---|---|---|---|
| 1 | modules/run_meta.py `MilvusGuard` | 실행 전 대상 컬렉션 load 상태 기록(`initial_state`). 사용 중 검사에서 **대상 컬렉션 요청도 남의 요청**으로 셈(이전에는 대상 컬렉션 요청을 제외). 대상이 이미 Loaded면 기본 중단, `run.allow_preloaded_targets: true`일 때만 진행하고 load·release 안 함. Loading·상태 확인 불가·메트릭 조회 실패도 중단. load 직전에 상태를 다시 확인(그 사이 남이 load했으면 중단). release 대상 = 원래 NotLoad이고 이번 실행이 load한 것만(`release_safe`). release 직전 다시 표본(`run.release_check_s`)을 보고 대상 컬렉션 요청·Milvus 재시작(`process_start_time_seconds` 변화)·메트릭 실패가 있으면 release하지 않고 `release_report`에 이유와 수동 release 명령 기록. 이전 `release_all()`(무조건 release) 제거 | C3 검토: 실행 전 대상 요청을 내 것으로 오인, 원래 Loaded였던 컬렉션까지 release할 수 있었음 |
| 1 | modules/run_meta.py | `check_reserved_window`(run.reserved_window, KST: `HH:MM-HH:MM` 매일·자정 넘김 가능, 또는 `YYYY-MM-DD HH:MM~YYYY-MM-DD HH:MM`). purpose=measure와 cold/cache_constrained는 필수, 밖이면 거부. `RunLock`(내 폴더 `locks/milvus_run.lock`, flock, 내 실행끼리만). 팀 공통 잠금은 만들지 않음(미결) | 합의된 팀 잠금이 없어서 시간대와 내 잠금만 |
| 1 | main.py | 실행 순서: 실행 설정 검사(purpose, cache_condition, correctness는 max_queries 필수) → 예약 시간대 → 잠금 → md5(quick) → 어댑터(설정 누락이면 거부) → Guard preflight → `load_needed` → 실행 → finally `release_safe`. 실행 메타에 `reserved_window`, `run_lock`, `release_report`, `release_skipped`(수동 명령), `milvus_ops`. 종료 코드 6(자동 release 보류), 7(시간대·잠금·설정 거부) 추가. RUN_KEYS에 purpose, allow_preloaded_targets, reserved_window, release_check_s | 〃 |
| 1 | experiments/test_milvus_guard.py (신규) | 가짜 Milvus 클라이언트·가짜 메트릭으로 26개 검사: 기존 Loaded 미release, 대상 요청 시 중단, 이미 Loaded 기본 중단/허용 시 진행·미release, NotLoad → load → release, release 직전 요청·메트릭 실패·재시작 시 보류, 시간대 밖 거부(자정 넘김·날짜 형식·형식 오류), 다른 컬렉션 load, Loading, load 직전 상태 변화, 허용 목록 밖 조회 금지, 잠금 중복 | 요구 단위 테스트 |
| 1 | experiments/contract_test.py `--allow-load` | main.py와 같은 Guard 흐름(시간대, 잠금, preflight, load_needed, release_safe). Milvus C1을 `--allow-load` 없이 고르면 C1·공통 검사를 SKIP(이전에는 Milvus를 호출할 수 있었음) | 〃 |
| 2 | modules/c1_milvus_query_reference.py (신규) | `c1.impl=milvus_query_ref`. "B0 기본 Fetch(최적화 없음). C1 기여 구현은 C1 담당자 어댑터". 실제 AsyncMilvusClient `query(filter="pk in [...]", output_fields=[pk, text])`, PK 512개씩(bench_v2 CHUNK_LIMIT과 같음), batch/individual, range·get은 오류. 생성자 (meta, cfg), chunk_id↔PK 변환은 어댑터 안(ChunkMeta.pk_of_chunk/chunk_of_pk[layout]), 본문은 Milvus text 필드 값, bytes_returned = 반환 본문 UTF-8 합, text·pk 필드 없으면 명확한 ValueError, `milvus_collections()` 선언, load/release·컬렉션 설정 변경 없음 | C3 공식 비교를 C1 구현 없이 진행 |
| 2 | experiments/test_c1_milvus_query.py (신규) | 가짜 AsyncMilvusClient로 어댑터 로직 14개 검사 | 실제 Milvus를 못 쓰는 상황에서도 로직 확인 |
| 2 | modules/c1_file_reference.py | `asyncio.to_thread`(기본 풀) 대신 전용 `ThreadPoolExecutor(c1.file_ref_threads)`. 값이 없으면 생성자 ValueError → 실행 거부. describe에 threads, 실행 메타 `c1_threads` | 기본 풀 = 숨은 동시성 상한 |
| 2 | config.yaml `c1` | `file_ref_threads: null`(암묵적 기본값 없음), impl 주석에 milvus_query_ref 등록. `milvus.fetch.text_field: text`, `pk_field: pk` | 〃 |
| 2 | experiments/contract_test.py | C1 동시 실행 검사 `test_c1_concurrent`: 겹치지 않는 chunk_id 묶음 16개로 fetch_text 동시 호출, 각 결과의 키·원문·bytes_returned·n_pks가 자기 요청과 맞는지, 섞이면 FAIL 한 줄. 사용법을 본인 폴더 clone 기준·`--run runs/correctness_*.yaml`로 변경 | 요구 |
| 3 | modules/chunk_meta.py | md5_mode: full(별도 명령, 전 파일) / quick(일반 실행: 크기·수정 시각이 캐시와 같고 캐시 md5 == config면 통과, 바뀐 파일만 다시 계산) / off. cold/cache_constrained에서는 다시 계산하지 않고 중단. `quick_exclude` 파일(chunk_vecs, qrels_chunk) 제외. 캐시 `cache/md5_cache.json`(경로, 크기, mtime_ns, md5, 줄 수, 검사 시각). 실행 메타 `md5_check.mode`, 파일별 status, `full_verify`(마지막 전체 검사 시각, 24시간 초과 경고) | 매 실행 수 GB 읽기 제거 |
| 3 | verify_md5.py (신규) | 전체 md5 별도 명령(하루 한 번). 내 실행 잠금을 잡고 실행, cold/cache_constrained 조건이면 거부 | 측정 실행과 분리 |
| 3 | config.yaml `dataset.integrity` | cache, full_max_age_h 24, quick_exclude | 〃 |
| 4 | main.py `summarize`, `execute` | `qps`는 그대로(예열 뒤 끝난 결과 / duration_s). `strict_qps` = 측정 구간 [warmup_s, warmup_s+duration_s] 안에서 끝난 결과 / duration_s, `n_strict`, `n_tail_after_window`, `strict_qps_basis` 추가(키 추가만, 기존 키 유지). 완료 시각은 실행기가 run_one을 감싸 기록 | C3 기존 결과와 비교 유지 + 엄격 지표 |
| 5 | config.yaml `run` | `purpose: measure`, `evaluate: false`(성능 측정 기본). 원문 조회는 그대로 실제 수행(평가·본문 복사만 끔) | 성능 측정과 정확성 검증 분리 |
| 5 | runs/correctness_file_ref.yaml, runs/correctness_milvus_query.yaml (신규), runs/b0_smoke.yaml | 정확성 확인 설정: purpose correctness, evaluate true, 10개 개수 기반, 원문 대조. file_ref 스레드 8. b0_smoke는 기존 B0(nq1m_sel + file_ref) 호환 확인용으로 유지하고 purpose·evaluate·스레드만 추가 | 〃 |
| 6 | config.yaml `experiment` | `cache_condition` 선택지 warm / cold / cache_constrained(기본 warm) 주석, 기록·선택만 하고 캐시 제어는 관리자 설정·실행 스크립트 담당. `admin_cache_setting`, `io_valid_min_bytes`(1 MiB). run_meta 목록에 새 항목 | C1 cold 조건 |
| 6 | main.py | 실행 메타 `storage_read_delta`(시간 기반이면 측정 구간 시작·끝에서 메트릭을 따로 찍음, data_type/location별), `cache_condition_evidence`(적용 조건, 관리자 설정, 저장소 읽기 차이, cold/cache_constrained 유효 판정 또는 "실제 I/O 없음"). summary.json에 `cache_condition` 요약 | 〃 |
| 6 | DESIGN.md 9절 (신규) "실험 환경 구분" | warm / cold / cache_constrained 정의·용도, 둘은 다른 조건, 유효성 규칙, 예약 시간대·사용 중 아닐 때만 | 〃 |
| 7 | DESIGN.md | 상태 문단, 3절(D1·D2 확정), 4절(B0 기본 Fetch, C3 공식 비교 = milvus_query_ref + warm·C1 대기 불필요, file_ref 미사용 이유, C1 Layout 비교 = C1 어댑터 + cold, QPS 정의, 성능·정확성 분리, Milvus 실행 안전 규칙), 6절(브랜치 pipeline-design, 본인 폴더 clone → 계약 테스트 → PR), 8절(D1·D2·D12 확정, 회의 확인 목록에서 제외, 미결: 팀 공통 잠금·Milvus 종료 원인·m3 clients=64 금지) | 요구 |
| 7 | REVIEW_FOR_TEAM.md | 브랜치 이름 `pipeline-design`, "자기 어댑터를 붙이는 방법"을 본인 폴더 clone·PR 방식으로 다시 씀(총괄 폴더에 코드 넣는 지침 제거, 프롬프트에 "/workspace/jaeeun/pipeline-core에는 아무것도 쓰지 말 것"), 계약 테스트 명령에 `--run runs/correctness_*.yaml`·동시 실행 검사, C1·C3·junlim 표 갱신(C3 공식 비교, cold, strict_qps, D1·D2·D12 확정), 회의에 가져올 것·미결 항목(milvus.search 해결, 팀 잠금, Milvus 종료 원인, cold 관리자 설정), 단톡방 메시지의 브랜치 이름·동의 문장(`<링크>` 유지) | 〃 |
| 7 | SMOKE.md | 2절(실행 명령, 정확성/성능 설정, 스레드, md5, 종료 코드 6·7), 3절(본인 폴더 clone·PR), 4절(nq_seq 확정 후 overlay), 5절 4번 해결(nq_seq 확정), 6절 검증 결과, 7절 nq_seq B0(실행하지 않은 이유와 실행 순서) | 〃 |
| 7 | experiments/compare_top10.py (신규) | 두 results.jsonl의 같은 질문 상위 10개 일치율 | nq_seq B0 vs nq1m_sel B0 비교용 |
| 검사 | – | py_compile 전체 통과, YAML 4개 파싱, 계약 테스트 PASS 94 / FAIL 0 / SKIP 4 / WARN 4, Guard 단위 26/26, Milvus 어댑터 가짜 클라이언트 14/14, 전체 md5 10/10. 실제 Milvus 실행(nq_seq 계약 테스트, nq_seq B0, nq1m_sel B0)은 `junlim/nq_seq`가 이미 Loaded라 Guard가 중단 → 실행 안 함 | SMOKE.md 6·7절 |
| 추가 (15:43) | runs/correctness_milvus_query.yaml, runs/correctness_file_ref.yaml, main.py, experiments/contract_test.py, config.yaml 주석 | junlim 확인에 따라 정확성 확인 설정에만 `run.allow_preloaded_targets: true`. main.py는 purpose≠correctness에서 이 값이 켜져 있으면 거부(7). 계약 테스트는 `--allow-load` 없이도 이 설정 + 대상 Loaded일 때 load·release 없이 Milvus C1 검사(사용 중 검사는 그대로). 결과: 계약 테스트 PASS 21 / FAIL 0, nq_seq B0 오류 0·원문 100/100·포함률 6/10·nq1m_sel 대비 상위 10개 겹침 0.99 (SMOKE.md 7절) | 이미 Loaded인 nq_seq로 정확성 확인 |


## PR #3 지적 반영: 검색용 seq 매핑 로드 (2026-11-24)
- 근거: C1 담당 지현 님이 PR #3에서 찾은 공통 실행기 버그. main.py가 ChunkMeta를 `layouts=(settings.layout,)`(c1.layout 하나)로만 로드하는데, 검색 어댑터는 항상 `chunk_of_pk["seq"]`를 써서 c1.layout=doc/rand면 검색 단계에서 `KeyError: 'seq'`(pipeline이 질문별 error로 기록 → 결과 검사 위반, 종료 코드 5). 계약 테스트 공통 검사는 가짜 검색이라 못 잡았음.
- 그대로 둔 것: `interfaces.py`, `modules/C1.py`·`C2.py`·`C3.py`, 검색 어댑터 로직. Milvus 사용 안 함. 변경 전 파일은 세션 scratchpad `before_pr3/`에 백업.

| 파일 | 변경 |
|---|---|
| main.py | `loader.load(layouts=tuple(dict.fromkeys(("seq", settings.layout))))`: 검색용 seq + C1용 배치를 둘 다 로드 |
| experiments/contract_test.py | 실제 ChunkMeta 로드도 같은 방식(seq + c1.layout) |
| modules/search_milvus.py | `"seq"` 고정 이유 주석: 검색 컬렉션 nq_seq·nq1m_sel은 pk == chunk_id인 문서 순(seq) 배치, c1.layout은 C1 본문 컬렉션 배치일 뿐. 로직 변경 없음 |
| experiments/test_search_layout.py (신규) | 가짜 AsyncMilvusClient·가짜 Guard/메트릭으로 `main.main()` 전체 경로(실제 FileChunkMetaLoader → 실제 MilvusSearchAdapter → C2 none_ref → C1 milvus_query_ref → serial C3 → 결과 검사)를 c1.layout = seq/doc/rand로 실행. 종료 코드 0·오류 0·검색 pk 항등·C1 요청 pk == `pk_of_chunk[layout]` 확인. `--old-main <수정 전 main.py>`로 doc/rand에서 버그 재현 확인 |

- 검사: py_compile 전체 통과. 새 테스트 PASS 18 / FAIL 0(현재 코드 seq·doc·rand 통과, 수정 전 main.py는 seq 통과·doc/rand에서 질문 3개 모두 `search: KeyError: 'seq'`, 종료 코드 5). 계약 테스트(`--run runs/correctness_file_ref.yaml`) PASS 85 / FAIL 0 / SKIP 4 / WARN 4(수정 전과 같음, 94는 `--results` 포함 수치). MilvusGuard 26/26, C1 Milvus 가짜 클라이언트 14/14.
