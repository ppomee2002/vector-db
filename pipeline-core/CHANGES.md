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
