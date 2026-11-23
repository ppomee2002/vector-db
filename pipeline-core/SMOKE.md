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
cd /workspace/jaeeun/pipeline-core && nohup /workspace/jaeeun/venv/bin/python main.py --run runs/b0_smoke.yaml > logs/b0_smoke.log 2>&1 &
```
- 결과는 `results/<run.name>_<시각>/`에 `results.jsonl`(QueryResult, 본문 없음), `eval.json`(포함률·원문 대조), `summary.json`(지연 백분위·검사), `run_meta.json`으로 저장.
- 값 바꾸기: 기본값은 config.yaml(`c1/c2/c3.impl`, `run` 블록). 실행별 값은 `--set c2.policy=BC1 --set run.n_queries=50`처럼 쓰거나 `runs/`에 overlay 파일을 새로 만들어 덮어쓴다.
- 품질 평가(목록 1회)는 `run.max_queries`를 질문 수로, 성능 측정(시간 기반)은 `run.max_queries: null`로 둔다.
- 종료 코드: 0 정상 / 2 load 허용 목록 밖 / 3 Milvus 사용 중이라 멈춤 / 1 예외 / 4 release 실패 / 5 결과 검사 위반.
- 등록된 어댑터 보기: `/workspace/jaeeun/venv/bin/python main.py --list`

## 3. 담당자가 자기 어댑터를 붙이는 방법

1. **파일 위치:** `modules/` 아래에 새 파일을 만든다(예: `modules/c2_bc1.py`). `modules/C1.py`, `C2.py`, `C3.py`는 원래 있던 빈 파일이라 그대로 두었다. 담당자가 이 파일에 써도 된다.
2. **등록:**
   ```python
   from modules.registry import register

   @register("c2", "bc1_v1")              # 종류(c1/c2/c3), 이름
   class MyC2:
       def __init__(self, meta, cfg): ...  # 생성자 규약 (interfaces.py 공통 규칙 5): (ChunkMeta, 합친 config dict). 모든 어댑터 공통
       def select_neighbors(self, ranked_chunk_ids, *, policy): ...
   ```
   - C1은 `async fetch_text(chunk_ids, *, mode, batch_impl) -> (dict[int,str], FetchStats)`와 `fetch_source() -> "file:<경로>" | "milvus:<db>/<collection>"`. `FetchStats.bytes_returned` = 반환 본문 텍스트의 UTF-8 바이트 합(계약 테스트가 FAIL로 검사).
   - C3은 `async run_many(query_keys, run_one, *, mode, clients, cap, warmup_s, duration_s, max_queries=None)`. `max_queries`가 정수면 정확히 그 수만큼 실행(warmup/duration 무시, 모두 is_warmup=False), None이면 시간 기반.
   - 선택 훅: `async aclose()`(종료 시 정리), `describe()`(실행 메타에 기록), `milvus_collections() -> ["db/coll"]`(Milvus를 쓰는 C1: 실행기가 사용 중 검사 후 load·release).
   - 예시 코드: reference 3개 `modules/c1_file_reference.py`, `c2_none_reference.py`, `c3_serial_reference.py`.
3. **고르기:** overlay에 `c2: {impl: bc1_v1}`를 넣거나 `--set c2.impl=bc1_v1`.
4. **계약 테스트:**
   ```bash
   cd /workspace/jaeeun/pipeline-core
   /workspace/jaeeun/venv/bin/python experiments/contract_test.py --c2 bc1_v1 --only c2,common --require-policies BC1,all
   /workspace/jaeeun/venv/bin/python experiments/contract_test.py --c1 <이름> --only c1,common        # Milvus C1이면 --allow-load (run.milvus_allowed_load 안에서만)
   /workspace/jaeeun/venv/bin/python experiments/contract_test.py --c3 <이름> --only c3,common
   /workspace/jaeeun/venv/bin/python experiments/contract_test.py --results results/<run>/results.jsonl   # 저장된 결과 검사
   ```
   FAIL이 하나라도 있으면 종료 코드 1이고, 끝에 FAIL 목록이 한 줄씩 다시 나온다. C2·C3·공통 검사는 Milvus 없이 돌아간다(가짜 검색과 가짜 run_one 사용).
5. 통과하면 `main.py --run runs/b0_smoke.yaml --set c2.impl=bc1_v1 --set c2.policy=BC1`로 10개 질문을 돌려 본다.

## 4. 회의 후 nq_seq로 옮길 때 바꿀 항목

config.yaml에는 이미 `milvus.search = junlim/nq_seq`로 적혀 있다. 오늘은 `runs/b0_smoke.yaml`이 이 값을 덮어쓰고 있다. 바꿀 것은 아래 셋이다.
- `runs/b0_smoke.yaml`의 `milvus.search` 블록: 지우거나 `database: junlim`, `collection: nq_seq`로 변경
- `run.milvus_allowed_load`: `[junlim/nq_seq]` (load·release는 시간표(D12)와 junlim 동의 뒤에만)
- B0 정의대로 본문까지 nq_seq에서 읽으려면: `c1.impl`을 C1 담당의 Milvus `query_pk_in` 구현으로 변경. 그 구현이 `milvus_collections()`로 `junlim/nq_seq`를 선언하면 실행기가 같이 load한다. `file_ref`(파일)를 그대로 두어도 검색만 nq_seq로 바뀌어 실행은 된다.
- nq_seq의 PK 필드 이름이 `pk`가 아니면 `milvus.search.pk_field`를 추가한다(CHECK.md 기준 `pk`라 필요 없을 것).

## 5. 계약과 안 맞아서 고치지 않은 부분 → Smoke Test 후 계약 보완 (2026-11-23, 총괄 결정)

| # | 항목 | 문제 | 상태 | 결정·조치 (CHANGES.md "Smoke Test 후 계약 보완") |
|---|---|---|---|---|
| 1 | `C3SchedulerAdapter.run_many` | 시간 기반 실행만 있어서 "10개 질문을 한 번씩" 실행할 방법이 없음 | **해결** | 계약에 `max_queries: Optional[int] = None` 추가. 정수면 정확히 그 수만큼(한 바퀴) 실행하고 멈춤, warmup/duration 무시, 모두 is_warmup=False. None이면 기존 시간 기반. reference C3·실행기(항상 전달)·계약 테스트(개수 기반 시나리오 2개) 반영 |
| 2 | 어댑터 생성자 | C1/C3/Search 생성자가 정해져 있지 않음 | **해결** | interfaces.py 공통 규칙 5: 모든 어댑터(Search, C1, C2, C3) `__init__(meta, cfg)`, 추가 인자는 기본값 있는 키워드만. Search 어댑터를 `(meta, cfg, queries=None)`로 변경 |
| 3 | config.yaml에 구현 선택·실행 항목 없음 | `c1/c2/c3.impl`, `run.*` 키가 없음 | **해결** | config.yaml에 `c1/c2/c3.impl`(기본 `file_ref`/`none_ref`/`serial_ref`)과 `run` 블록(13개 키) 추가. 실행기는 run 키가 빠지면 중단. `runs/b0_smoke.yaml`은 덮어쓸 값만 남김 |
| 4 | `milvus.search` 값 | config는 `junlim/nq_seq`(동의 대기), 오늘 쓴 것은 `jaeeun/nq1m_sel` | 미결 (이번 범위 밖) | overlay로 덮어쓰는 방식 유지. 회의 결과(D1)로 확정 |
| 5 | 실제 Fetch 출처 vs `Settings.fetch_collection` | 본문은 파일 저장소인데 Settings에는 `junlim/nq_seq`가 기록됨 | **해결** | C1 계약에 `fetch_source()` 추가. 실행 메타 `fetch_source`에 c1_impl, source, settings_fetch_collection, matches_settings를 기록(다르면 note와 함께 둘 다 남김) |
| 6 | md5 `null` 파일 7개 | 검사하지 않음 | **해결** | 로더가 null이면 계산해 실행 메타에 기록(`computed`). 11:26 실행 값으로 config.yaml에 고정했고, 11:27 실행에서 10개 모두 일치. 원본은 읽기만 함 |
| 7 | `FetchStats.bytes_returned` 정의 | 본문 바이트인지 응답 크기인지 미정 | **해결** | 반환 본문 텍스트의 UTF-8 바이트 합으로 정의(interfaces.py). reference C1은 텍스트를 인코딩해서 셈. 계약 테스트는 WARN → FAIL |
| 8 | `milvus_reset_log` | 관리자 스위치 조회에 ssh 필요 | 유지 (결정) | `not collected` 그대로 기록 |

그 밖의 사항
- Milvus는 `jaeeun/nq1m_sel`만 load·search·release했다. 다른 컬렉션 API는 호출하지 않았고, 서버 상태는 메트릭(로드된 컬렉션 수, 요청 수 차이)으로만 확인했다.
- 팀원 코드는 옮기거나 감싸지 않았다. 평가 정규화는 총괄 소유의 c2x 평가기와 같은 정의로 새로 작성했다.
