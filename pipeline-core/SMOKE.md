# SMOKE: 공통 실행기 + B0 10-Query Smoke Test (2026-11-23)

- 실행: 2026-11-23 09:26:07~09:26:30 KST, 결과 `results/b0_smoke_20261123_092607/`
- 구성: 검색 `jaeeun/nq1m_sel`(HNSW/IP, search_topk 10, context_topk 10, ef 64) + C2 `none_ref` + C1 `file_ref`(파일 저장소 seq) + C3 `serial_ref`
- 질문: `nq_test_ans` 2,254개를 seed 42로 섞은 순서의 앞 10개 (첫 3개: q_idx 1175, 691, 1276)
- **계약 보완 후 재실행 (2026-11-23 11:26~11:27 KST):** 5절 항목을 고친 뒤(CHANGES.md "Smoke Test 후 계약 보완") 같은 구성으로 두 번 다시 돌림. `results/b0_smoke_20261123_112611/`(md5 7개 첫 계산), `results/b0_smoke_20261123_112701/`(config에 고정한 md5 10개 전부 검사, **최종 기준 결과**). 두 번 모두 오류 0, 포함률 7/10, 원문 대조 100/100, 결과 검사 위반 0, release 후 서버 load 0. C3는 계약 인자 `max_queries=10`(개수 기반)으로 실행.
- 실패한 첫 시도 `results/b0_smoke_20261123_092514/`(09:25)는 남겨 둠. 10개 모두 `search: KeyError 'id'`. 원인은 pymilvus 3.0.3이 검색 결과의 PK를 `id`가 아니라 PK 필드 이름 `pk`로 주는 것. Search 어댑터에서 고친 뒤 다시 실행함. 계약과는 관계없음.

## 1. 확인 결과

| 확인 항목 | 결과 | 근거 |
|---|---|---|
| 오류 없음 | **통과** 10/10, error 0 | `summary.json` `n_error: 0` |
| hits 순위 보존 | **통과** rank 0~9 순서대로, 점수 내림차순, context_ids = hits 상위 10 | `checks.hits_rank` 위반 0, 계약 테스트 `결과파일 hits_rank` |
| fetched_ids = context_ids | **통과** (policy none, 이웃 0) | `checks.policy_none`, `checks.ids` 위반 0 |
| 본문 원문 일치 (샘플) | **통과** 100/100 (10질문 × 10 chunk 전부를 `chunks.jsonl` 원문과 대조) | `eval.json` `text_check` |
| 검색 결과 ID 대응 | **통과** pk == chunk_id, Milvus `doc_idx` == ChunkMeta `doc_idx` (100 hits) | `verify_search_doc_idx: true` |
| reference 어댑터 계약 테스트 | **전부 통과**: PASS 67 / FAIL 0 / SKIP 4 / WARN 2 → 보완 후 **PASS 93 / FAIL 0 / SKIP 4 / WARN 4** (C3 max_queries 시나리오 2개, C1 bytes_returned·fetch_source 검사, 결과 파일 검사 포함) | SKIP = C2 reference의 BC1·all 미구현(fake/real 각 2개). WARN = serial C3의 동시 실행이 1 (reference라 정상) |
| 계약 테스트가 위반을 잡는지 | **확인** 일부러 틀리게 만든 C1/C2/C3(scratchpad 임시 등록)에서 FAIL 50건, 이유가 한 줄씩 출력됨. 보완 후 새 검사도 확인: max_queries 무시 C3, 메타 크기를 bytes_returned에 넣은 C1, fetch_source 없는 C1 → FAIL 5건 | numpy 정수 반환, R 포함, 문서 밖 이웃, 본문 한 칸 밀림, 누락 키, FetchStats None, range 무시, service 불일치, STAGE_KEYS 밖 키, 순환 순서 위반, 늦게 멈춤, 결과 누락 |
| 정답 문자열 포함률 계산 | **동작** 7/10 = 0.70 (참고용 값) | `eval.json` `answer_hit` |
| stage_ms 키 | STAGE_KEYS 안에서만: search, c2_select, fetch, queue_wait(0), total, service. embed는 기록 안 함(D10) | `checks.stage_keys` 위반 0 |
| ChunkMeta | 길이 1,000,000, 문서 27,133, pk_seq 항등, npz doc_idx == junlim doc_idx, diff(offset) == nbytes | `run_meta.json` `md5_check` |
| md5 | 처음: chunks ✔ (줄 수 1,000,000), gold ✔, flags ✔, 나머지 7개 `not_checked` → 보완 후: 7개를 11:26 실행에서 계산해 config.yaml에 고정, 11:27 실행에서 **10개 모두 일치** | `run_meta.json` `md5_check.md5` |
| Fetch 출처 | `fetch_source`: c1_impl `file_ref`, source `file:/workspace/jaeeun/sel_fetch/data/chunk_store_seq.bin`, settings_fetch_collection `junlim/nq_seq`, matches_settings false (둘 다 기록) | `run_meta.json` `fetch_source` |
| nq_test_ans q_idx 고정 | 2,254개, `data/nq_test_ans_qidx.txt` md5 `906ceace…` | `run_meta.json` `queries.info` |
| Milvus 사용 전 검사 | 서버에 load된 컬렉션 0, 10초 동안 데이터 요청 0, 다른 사용자 프로세스 없음 → load 4.1초 | `run_meta.json` `other_user_activity.before` |
| release | try/finally로 release, 끝난 뒤 `NotLoad`. 서버 load 수 0으로 돌아옴 | `load_state.after_release`, 메트릭 `collection_num 0` |
| 실행 메타 | 측정 시각, load 상태(전/load 후/release 후), md5, swap(7,434/8,191 MiB 사용, 전후 같음), 다른 사용자 활동(전후), Milvus 구간 메트릭 차이, top10_consistency(n/a) | `run_meta.json` |

참고 수치(질문 10개, warm 아님, 성능 해석 금지): total p50 7.6 ms / p95 72.6 ms(첫 질문 콜드). search p50 6.9 ms, fetch p50 0.67 ms, chunk 10개·3,974 B·pread 10회/질문.

## 2. 실행 방법

```bash
cd /workspace/jaeeun/pipeline-core && nohup /workspace/jaeeun/venv/bin/python main.py --run runs/b0_smoke.yaml > logs/b0_smoke.log 2>&1 &              # 기존 B0 호환 (nq1m_sel + file_ref)
cd /workspace/jaeeun/pipeline-core && nohup /workspace/jaeeun/venv/bin/python main.py --run runs/correctness_milvus_query.yaml > logs/nq_seq_b0.log 2>&1 &   # nq_seq B0 (D1 확정)
```
- 결과는 `results/<run.name>_<시각>/`에 `results.jsonl`(QueryResult, 본문 없음), `eval.json`(포함률·원문 대조), `summary.json`(지연 백분위·검사), `run_meta.json`으로 저장.
- 값 바꾸기: 기본값은 config.yaml(`c1/c2/c3.impl`, `run` 블록). 실행별 값은 `--set c2.policy=BC1 --set run.n_queries=50`처럼 쓰거나 `runs/`에 overlay 파일을 새로 만들어 덮어쓴다.
- 품질 평가(목록 1회)는 `run.max_queries`를 질문 수로, 성능 측정(시간 기반)은 `run.max_queries: null`로 둔다.
- 정확성 확인 설정(`run.purpose: correctness`, `evaluate: true`, 개수 기반)은 `runs/b0_smoke.yaml`, `runs/correctness_file_ref.yaml`, `runs/correctness_milvus_query.yaml`. 성능 측정(`purpose: measure`)은 config.yaml 기본값이 `evaluate: false`이고 예약 시간대 `run.reserved_window`가 필수.
- `file_ref`는 스레드 수 `c1.file_ref_threads`를 반드시 적는다(없으면 실행 거부, B0 정확성 확인 = 8, 실행 메타 `c1_threads`).
- md5: 일반 실행은 크기·수정 시각만 캐시(`cache/md5_cache.json`)와 비교하고 바뀐 파일만 다시 계산한다(chunk_vecs·qrels_chunk 제외). 전체 md5는 별도 명령 `nohup /workspace/jaeeun/venv/bin/python verify_md5.py > logs/verify_md5.log 2>&1 &`로 하루 한 번(cold 직전 금지).
- 종료 코드: 0 정상 / 2 load 허용 목록 밖 / 3 Milvus 사용 중(또는 대상이 이미 Loaded, 확인 불가)이라 멈춤 / 1 예외 / 4 release 실패 / 5 결과 검사 위반 / 6 자동 release 보류(수동 명령은 로그와 `run_meta.json` `release_skipped`) / 7 예약 시간대 밖·잠금·설정 누락으로 거부.
- 등록된 어댑터 보기: `/workspace/jaeeun/venv/bin/python main.py --list`

## 3. 담당자가 자기 어댑터를 붙이는 방법

**본인 작업 폴더에서 한다.** 팀 저장소 `ppomee2002/vector-db`의 `pipeline-design` 브랜치를 본인 폴더(예: `/workspace/<내 이름>/vector-db`)에 clone하거나 `pipeline-core/`를 복사해서 작업하고, 계약 테스트(동시 실행 검사 포함)를 통과시킨 뒤 `pipeline-design` 대상으로 PR을 올린다. `/workspace/jaeeun/pipeline-core`에는 코드를 넣지 않는다. 자세한 순서는 `REVIEW_FOR_TEAM.md` "자기 어댑터를 붙이는 방법"과 같다.

1. **파일 위치:** 본인 clone의 `modules/` 아래에 새 파일(예: `modules/c2_bc1.py`). `modules/C1.py`, `C2.py`, `C3.py`는 원래 있던 빈 파일이다.
2. **등록:**
   ```python
   from modules.registry import register

   @register("c2", "bc1_v1")              # 종류(c1/c2/c3), 이름
   class MyC2:
       def __init__(self, meta, cfg): ...  # 생성자 규약 (interfaces.py 공통 규칙 5): (ChunkMeta, 합친 config dict). 모든 어댑터 공통
       def select_neighbors(self, ranked_chunk_ids, *, policy): ...
   ```
   - C1은 `async fetch_text(chunk_ids, *, mode, batch_impl) -> (dict[int,str], FetchStats)`와 `fetch_source() -> "file:<경로>" | "milvus:<db>/<collection>"`. `FetchStats.bytes_returned` = 반환 본문 텍스트의 UTF-8 바이트 합(계약 테스트가 FAIL로 검사). 동시 호출에서 결과·통계가 섞이면 안 됨.
   - C3은 `async run_many(query_keys, run_one, *, mode, clients, cap, warmup_s, duration_s, max_queries=None)`. `max_queries`가 정수면 정확히 그 수만큼 실행(warmup/duration 무시, 모두 is_warmup=False), None이면 시간 기반.
   - 선택 훅: `async aclose()`(종료 시 정리), `describe()`(실행 메타에 기록), `milvus_collections() -> ["db/coll"]`(Milvus를 쓰는 C1: 실행기가 MilvusGuard 규칙으로 검사 후 필요할 때만 load·release).
   - 예시 코드: reference `modules/c1_file_reference.py`, `c1_milvus_query_reference.py`, `c2_none_reference.py`, `c3_serial_reference.py`.
3. **고르기:** overlay에 `c2: {impl: bc1_v1}`를 넣거나 `--set c2.impl=bc1_v1`.
4. **계약 테스트:**
   ```bash
   cd /workspace/<내 이름>/vector-db/pipeline-core
   python experiments/contract_test.py --run runs/correctness_file_ref.yaml --c2 bc1_v1 --only c2,common --require-policies BC1,all
   python experiments/contract_test.py --run runs/correctness_file_ref.yaml --c1 <이름> --only c1,common                     # 파일 C1
   python experiments/contract_test.py --run runs/correctness_milvus_query.yaml --c1 <이름> --only c1,common --allow-load     # Milvus C1
   python experiments/contract_test.py --run runs/correctness_file_ref.yaml --c3 <이름> --only c3,common
   python experiments/contract_test.py --run runs/correctness_file_ref.yaml --results results/<run>/results.jsonl
   ```
   FAIL이 하나라도 있으면 종료 코드 1이고, 끝에 FAIL 목록이 한 줄씩 다시 나온다. C1 검사에는 서로 다른 chunk_id 묶음 16개를 동시에 요청하는 동시 실행 검사가 들어 있다. C2·C3·공통 검사는 Milvus 없이 돌아간다.
5. 통과하면 본인 clone에서 `main.py --run runs/correctness_milvus_query.yaml --set c2.impl=bc1_v1 --set c2.policy=BC1`로 10개 질문을 돌려 보고(MilvusGuard 규칙, D12 시간대), PR을 올린다.

## 4. nq_seq로 옮기기 (D1 확정, 2026-11-23)

D1이 확정되어 config.yaml 기본값(`milvus.search`·`milvus.fetch` = `junlim/nq_seq`)을 그대로 쓴다. 별도 overlay를 만들었다.
- `runs/correctness_milvus_query.yaml`: 검색·본문 모두 nq_seq, C1 = 기본 Milvus Fetch `milvus_query_ref`(query pk in, 512개씩). = **nq_seq B0**. `run.milvus_allowed_load: [junlim/nq_seq]`.
- `runs/correctness_file_ref.yaml`: 검색 nq_seq + C1 `file_ref`(스레드 8).
- `runs/b0_smoke.yaml`: 기존 B0 경로(검색 nq1m_sel + file_ref) 호환 확인용으로 남김.
- nq_seq의 PK 필드는 `pk`, 본문 필드는 `text`(VARCHAR 4096, mmap on)로 확인함(2026-11-23 describe). mmap·인덱스 설정은 바꾸지 않는다.

## 5. 계약과 안 맞아서 고치지 않은 부분 → Smoke Test 후 계약 보완 (2026-11-23, 총괄 결정)

| # | 항목 | 문제 | 상태 | 결정·조치 (CHANGES.md "Smoke Test 후 계약 보완") |
|---|---|---|---|---|
| 1 | `C3SchedulerAdapter.run_many` | 시간 기반 실행만 있어서 "10개 질문을 한 번씩" 실행할 방법이 없음 | **해결** | 계약에 `max_queries: Optional[int] = None` 추가. 정수면 정확히 그 수만큼(한 바퀴) 실행하고 멈춤, warmup/duration 무시, 모두 is_warmup=False. None이면 기존 시간 기반. reference C3·실행기(항상 전달)·계약 테스트(개수 기반 시나리오 2개) 반영 |
| 2 | 어댑터 생성자 | C1/C3/Search 생성자가 정해져 있지 않음 | **해결** | interfaces.py 공통 규칙 5: 모든 어댑터(Search, C1, C2, C3) `__init__(meta, cfg)`, 추가 인자는 기본값 있는 키워드만. Search 어댑터를 `(meta, cfg, queries=None)`로 변경 |
| 3 | config.yaml에 구현 선택·실행 항목 없음 | `c1/c2/c3.impl`, `run.*` 키가 없음 | **해결** | config.yaml에 `c1/c2/c3.impl`(기본 `file_ref`/`none_ref`/`serial_ref`)과 `run` 블록(13개 키) 추가. 실행기는 run 키가 빠지면 중단. `runs/b0_smoke.yaml`은 덮어쓸 값만 남김 |
| 4 | `milvus.search` 값 | config는 `junlim/nq_seq`(동의 대기), 오늘 쓴 것은 `jaeeun/nq1m_sel` | **해결 (nq_seq 확정)** | junlim 동의(2026-11-23)로 D1 확정. config.yaml 기본값 nq_seq 그대로, nq_seq B0 설정 `runs/correctness_milvus_query.yaml`. `runs/b0_smoke.yaml`(nq1m_sel)은 기존 B0 호환 확인용 |
| 5 | 실제 Fetch 출처 vs `Settings.fetch_collection` | 본문은 파일 저장소인데 Settings에는 `junlim/nq_seq`가 기록됨 | **해결** | C1 계약에 `fetch_source()` 추가. 실행 메타 `fetch_source`에 c1_impl, source, settings_fetch_collection, matches_settings를 기록(다르면 note와 함께 둘 다 남김) |
| 6 | md5 `null` 파일 7개 | 검사하지 않음 | **해결** | 로더가 null이면 계산해 실행 메타에 기록(`computed`). 11:26 실행 값으로 config.yaml에 고정했고, 11:27 실행에서 10개 모두 일치. 원본은 읽기만 함 |
| 7 | `FetchStats.bytes_returned` 정의 | 본문 바이트인지 응답 크기인지 미정 | **해결** | 반환 본문 텍스트의 UTF-8 바이트 합으로 정의(interfaces.py). reference C1은 텍스트를 인코딩해서 셈. 계약 테스트는 WARN → FAIL |
| 8 | `milvus_reset_log` | 관리자 스위치 조회에 ssh 필요 | 유지 (결정) | `not collected` 그대로 기록 |

그 밖의 사항
- Milvus는 `jaeeun/nq1m_sel`만 load·search·release했다. 다른 컬렉션 API는 호출하지 않았고, 서버 상태는 메트릭(로드된 컬렉션 수, 요청 수 차이)으로만 확인했다.
- 팀원 코드는 옮기거나 감싸지 않았다. 평가 정규화는 총괄 소유의 c2x 평가기와 같은 정의로 새로 작성했다.

## 6. C3 검토 반영 후 검증 (2026-11-23 14:30~14:40 KST)

| 검증 | 결과 | 근거 |
|---|---|---|
| py_compile (main.py, verify_md5.py, interfaces.py, modules/*.py, experiments/*.py) | 통과 | – |
| YAML 파싱 (config.yaml, runs/b0_smoke.yaml, runs/correctness_file_ref.yaml, runs/correctness_milvus_query.yaml) | 통과 | `yaml.safe_load` |
| 계약 테스트, reference(file_ref 스레드 8 + none_ref + serial_ref) + 저장된 B0 결과 | **PASS 94 / FAIL 0 / SKIP 4 / WARN 4**(이전 93 + 동시 실행 검사 1). 동시 실행 16개: chunk 181개, 결과·원문·통계 모두 일치 | `logs/contract_file_ref_results.log`. SKIP = C2 reference BC1·all 미구현, WARN = serial C3 동시 실행 1 |
| 동시 실행 검사가 섞임을 잡는지 | 확인. 공유 상태로 결과가 섞이는 C1 → FAIL("요청 #0: 다른 요청(#14)의 chunk_id가 섞임"), 통계만 섞이는 C1 → FAIL("bytes_returned != 자기 결과 UTF-8 합") | scratchpad 임시 등록 |
| file_ref 스레드 수 누락 | config.yaml만으로 실행하면 생성 거부(계약 테스트 FAIL, main.py 종료 코드 7) | – |
| Milvus 기본 Fetch 어댑터 (가짜 AsyncMilvusClient) | **PASS 14 / FAIL 0**: 512개 분할(1100 → 512+512+76), rand PK 변환, UTF-8 바이트, 요청 안 한 PK 무시, individual, range·get 오류, text 필드 없는 컬렉션 오류, 동시 16개 | `experiments/test_c1_milvus_query.py` |
| Milvus 기본 Fetch 어댑터 계약 테스트 (실제 Milvus) | **실행 안 함(MilvusGuard 중단)**: `junlim/nq_seq`가 이미 Loaded | `logs/contract_milvus_query.log` |
| MilvusGuard 단위 테스트 (가짜 클라이언트) | **PASS 26 / FAIL 0** | `experiments/test_milvus_guard.py` |
| 전체 md5 (`verify_md5.py`) | 10개 모두 일치, 캐시 생성(4.2초) | `logs/verify_md5.log`, `cache/md5_cache.json` |
| 기존 B0 경로 호환 (`runs/b0_smoke.yaml`, 실제 서버) | 설정 해석·md5 quick(8개 quick_ok, 2개 제외, 0.1초)·ChunkMeta·어댑터 생성(스레드 8)·예약 시간대(정확성 확인이라 생략)까지 통과. **Milvus 사용 중 검사에서 중단(종료 코드 3)**: "다른 컬렉션 1개가 load되어 있음" (= Loaded 상태인 `junlim/nq_seq`) | `results/b0_smoke_20261123_143453/run_meta.json` |
| 기존 B0 경로 호환 (Milvus만 가짜로 바꾼 오프라인 실행) | B0 설정 종료 코드 0, 결과 검사 위반 0, 원문 대조 100/100. 시간 기반 측정에서 qps·strict_qps·측정 구간 메트릭 기록 확인. 시간대 없는 측정 → 7, 스레드 누락 → 7, release 직전 대상 요청 → release 보류·수동 명령·종료 코드 6 | scratchpad `offline_results/` (검색·Guard 클라이언트·메트릭만 가짜, C1 file_ref·C2·C3·평가는 실제) |

## 7. nq_seq B0

- 설정: `runs/correctness_milvus_query.yaml`. 검색·본문 `junlim/nq_seq`, C1 `milvus_query_ref`, C2 `none_ref`, C3 `serial_ref`, `nq_test_ans` 10개, `max_queries=10`, `evaluate=true`.
- **상태: 14:39 KST에는 실행하지 않음(아래 이유). 15:43 KST junlim 확인 후 정확성 확인 설정에만 `run.allow_preloaded_targets: true`를 켜고 실행함 → 아래 "실행 결과 (15:44 KST)".**
- 이유: MilvusGuard 안전 조건을 충족하지 않는다. `junlim/nq_seq`가 실행 전부터 **Loaded**였다(서버 load 수 1, 표본 5초 동안 데이터 요청 0). Milvus 프로세스가 13:29 KST에 다시 시작됐고, 그때 이전 load 상태가 복원된 것으로 보인다. 누가 load했는지는 다른 사람의 API를 조회하지 않으므로 모른다. 기본 규칙(이미 Loaded면 중단)에 따라 중단했다. `run.allow_preloaded_targets: true`(진행하되 load·release 안 함)는 총괄 판단 사항이라 켜지 않았다. 같은 이유로 기존 B0(nq1m_sel) 실제 실행도 "다른 컬렉션 load됨"으로 중단됐다.
- 자동 release를 하지 않은 컬렉션: 없음(아무것도 load하지 않았음). `junlim/nq_seq`는 원래 Loaded라 건드리지 않았다.
- 안전 조건이 갖춰지면(nq_seq NotLoad, 다른 컬렉션 load 0, 요청 0) 실행할 순서:
  ```bash
  cd /workspace/jaeeun/pipeline-core
  nohup /workspace/jaeeun/venv/bin/python experiments/contract_test.py --run runs/correctness_milvus_query.yaml --only c1,common --allow-load > logs/contract_milvus_query.log 2>&1
  nohup /workspace/jaeeun/venv/bin/python main.py --run runs/correctness_milvus_query.yaml > logs/nq_seq_b0.log 2>&1
  nohup /workspace/jaeeun/venv/bin/python main.py --run runs/b0_smoke.yaml > logs/b0_smoke_6.log 2>&1
  /workspace/jaeeun/venv/bin/python experiments/compare_top10.py results/correctness_milvus_query_<시각>/results.jsonl results/b0_smoke_<시각>/results.jsonl
  ```
  확인할 것: 오류 수, hits 순위 보존(checks.hits_rank), 원문 대조(eval.json text_check), 정답 문자열 포함률, nq1m_sel B0(기준 `results/b0_smoke_20261123_112701/`)와 같은 질문에서 상위 10개 일치율.

### 실행 결과 (2026-11-23 15:43~15:44 KST, junlim 확인 후)

- 변경: `runs/correctness_milvus_query.yaml`, `runs/correctness_file_ref.yaml`에만 `run.allow_preloaded_targets: true`(정확성 확인에만, 이미 Loaded인 nq_seq를 그대로 쓰고 load·release 안 함). config.yaml 기본값은 false. main.py는 `purpose`가 correctness가 아닌데 이 값이 켜져 있으면 거부(종료 코드 7, 확인함). 계약 테스트는 `--allow-load` 없이도, 이 설정이 있고 대상이 이미 Loaded일 때만 load·release 없이 Milvus C1을 검사한다(NotLoad면 FAIL로 멈추고 `--allow-load` 안내).
- 실행 직전 확인(15:43:42, 10초 표본): 데이터 요청 0, 서버 load 수 1(= nq_seq), 재시작 없음, 다른 사용자 프로세스 없음. 계약 테스트와 main.py도 시작 직전에 각각 10초 표본을 다시 봤고, 둘 다 요청 0.
- nq_seq는 실행 전·후 모두 Loaded. load·release 호출 없음(`milvus_ops`: skip_load, release_report: not_released "실행 전부터 Loaded"). mmap·인덱스 설정은 건드리지 않음. nq1m_sel은 다시 돌리지 않음.

| 항목 | 결과 | 근거 |
|---|---|---|
| 계약 테스트 (C1 `milvus_query_ref`, 실제 nq_seq, `--only c1,common`, `--allow-load` 없음) | **PASS 21 / FAIL 0 / SKIP 0 / WARN 0**. 원문 대조, 키, FetchStats, bytes_returned(UTF-8), fetch_source `milvus:junlim/nq_seq`, range 오류, 동시 실행 16개(chunk 181개, 결과·원문·통계 일치, 47 ms), 공통 run_one(가짜 검색 + 실제 Fetch) 결과 94개 검사 위반 0. fetch 중앙값 14.3 ms(참고) | `logs/contract_milvus_query_preloaded.log` |
| nq_seq B0 오류 수 | **0 / 10** (종료 코드 0, 결과 검사 위반 0) | `results/correctness_milvus_query_20261123_154411/` |
| hits 순위 보존 | 통과 (rank 0~9, 점수 내림차순, context_ids = hits 상위 10, doc_idx 대조 100 hits 일치) | `summary.json` checks |
| 원문 대조 | **100/100 일치** (Milvus text 필드 vs `chunks.jsonl`) | `eval.json` text_check |
| 정답 문자열 포함률 | **6/10 = 0.60** (nq1m_sel B0는 7/10) | `eval.json` |
| Fetch 출처 | `milvus:junlim/nq_seq`, Settings.fetch_collection과 일치(matches_settings true). 질문당 query 1회, 평균 3,976 B | `run_meta.json` fetch_source, summary |
| 상위 10개 일치율 (nq1m_sel B0 `results/b0_smoke_20261123_112701/`와 같은 질문 10개) | 집합 겹침 평균 **0.99**(최소 0.9), 순서까지 같은 질문 **9/10**, 1위 일치 10/10 | `logs/compare_top10_nqseq_vs_nq1msel.json` |
| 다른 질문 | q958만 10위가 다름: nq_seq 798971(점수 0.5473) vs nq1m_sel 44957. 두 컬렉션의 HNSW 그래프가 달라서 생긴 근사 검색 차이로 봄(같은 M=16/efC=200, ef 64). 이 질문의 정답 문자열이 44957에 있어 포함률이 1개 차이(7 → 6) | 결과 파일 비교 |

참고 수치(질문 10개, 성능 해석 금지): total p50 21.7 ms / p95 57.3 ms, search p50 15.4 ms, fetch p50 5.95 ms.
