# REVIEW_FOR_TEAM: 기존 코드와의 충돌 확인 안내

## 원칙
- **내부 구현은 담당자 자유다.** 고정하는 것은 계약뿐이다: 입력, 출력, ID 규칙, 기록 항목(`interfaces.py`, `config.yaml`).
- 확인할 것은 "내 기존 코드를 계약에 맞출 수 있는가"이다. 설계 자체의 재논의는 이번 범위가 아니다(결정 근거: `DESIGN.md` 8절, `CHECK.md`).
- 공통 ID 규칙 4개: ① 원본 `chunks.jsonl` 줄 번호 = 공통 chunk_id ② 레이아웃별 PK 변환은 C1 내부에서만 ③ 검색 순위는 `QueryResult.hits`로 보존 ④ 질문 키는 (query_set, q_idx).
- 공통 생성자 규약(모든 어댑터: Search, C1, C2, C3): `__init__(self, meta: ChunkMeta, cfg: dict)`. meta = 실행기가 로드한 ChunkMeta, cfg = config.yaml + 실행 overlay를 합친 dict. 추가 인자는 기본값 있는 키워드만 허용(실행기와 계약 테스트는 `(meta, cfg)`만 넘김). `interfaces.py` 공통 규칙 5.
- 보고 분류: **① 그대로 맞음 / ② 감싸면 됨 / ③ 실제 충돌(이유 + 대안)**. 회의에는 **③만** D 번호나 interfaces 항목 이름(예: `C1FetchAdapter.fetch_text`, `FetchStats`, D4)으로 가져온다.

## 담당자별 영향

팀 저장소 `/workspace/jaeeun/team_ref` 브랜치 기준. C2·C3 브랜치 코드는 `/workspace/junlim`에도 같은 이름으로 있다. 한 사람이 여러 역할을 맡았다면 해당 표를 합쳐서 보면 된다.

### C1 (jihyeon)
| 구분 | 내용 |
|---|---|
| 그대로 쓰는 것 | w13 C1 컬렉션 `jihyeon/w13c1_doc`, `w13c1_rand`. 매핑 `c1_w13/milvus_meta/pk_doc.npy`(항등), `pk_rand.npy`. 파일 저장소 `c1_w13/store/{doc,rand}.bin`, `*_off.npy`, `len.npy`. `c1_milvus.py`의 `fetch()`(get), `parts_for("one"/"each")`. 비교 축(doc vs rand, 한 번에 vs 따로). w12 결과는 브랜치 `w12-layout-grouping` |
| 감싸거나 바꿀 것 | `C1FetchAdapter.fetch_text(chunk_ids, mode, batch_impl)`가 `({chunk_id: text}, FetchStats)`를 반환. `FetchStats.n_requests / n_pks / bytes_returned`는 필수, blocks와 device_bytes는 파일 경로에서만 채움. `batch_impl`에 `query_pk_in`(B0 기본, D6) 추가. chunk_id → pk 변환은 `ChunkMeta.pk_of_chunk[layout]`으로 C1 내부에서. 검색은 하지 않고 nq_seq 검색 결과(chunk_id)를 입력으로 받음. 결과 dict의 키는 pk가 아니라 chunk_id. **생성자 `__init__(meta, cfg)`**(컬렉션·저장소 경로는 cfg에서 읽음). **`fetch_source()`**로 실제 본문 출처를 `"file:<데이터 파일 경로>"` 또는 `"milvus:<db>/<collection>"`으로 밝힘(실행기가 실행 메타에 기록하고, `Settings.fetch_collection`과 다르면 둘 다 남김). **`FetchStats.bytes_returned` = 반환된 본문 텍스트의 UTF-8 바이트 합**(`sum(len(t.encode("utf-8")) for t in texts.values())`. 응답 메시지·메타 필드·블록 크기는 넣지 않음. 계약 테스트가 FAIL로 검사). Milvus를 쓰면 `milvus_collections()`로 컬렉션을 선언(실행기가 MilvusGuard 규칙으로 사용 중 검사 후 필요할 때만 load·release). **동시 호출 안전:** `fetch_text`가 동시에 여러 번 불려도 결과·FetchStats가 섞이면 안 됨(계약 테스트 동시 실행 검사, 16개). 스레드 풀을 쓰면 스레드 수를 설정에 명시(암묵적 기본 풀 금지, `file_ref`의 `c1.file_ref_threads` 참고) |
| 기존 수치와 이으려면 | 같은 100만 chunk(md5 `aca9a920…`), 같은 읽기 집합(top10 / BC1 / all1), warm과 cold(쿼리마다 release/load) 구분 유지. **C1 Layout 비교 = C1 담당자 어댑터 + cold 환경(관리자 설정 후)**(`DESIGN.md` 9절, cold와 cache_constrained는 다른 조건). 측정 구간 저장소 읽기 메트릭이 0보다 확실히 커야 유효, 아니면 "실제 I/O 없음"으로 C1 근거에서 제외. 물리 I/O 주장은 유효한 cold나 파일 경로에서만(D11). cold는 예약 시간대(`run.reserved_window`)에, 다른 사람이 Milvus를 쓰지 않을 때만 |
| 확인·동의할 항목 | **w13c1_doc·rand 사용 동의**(C1 비교), D6, D7, D11. cold 실험용 관리자 설정 일정 |

### C2 담당
| 구분 | 내용 |
|---|---|
| 그대로 쓰는 것 | 브랜치 `w12-c2-prefetching` `코드/pf3_rules.py`의 `select(rule, R, doc, flags)`, `nbrs()`, 규칙 none/BC1/all(+A 등 내부 규칙). `코드/bench_pf4.py`의 검색 → `plan()` → `afetch` 1회 순서. 정답·flags `pf3_out/nq_gold.jsonl`, `nq_flags.npy`. 품질 평가 방식(`analyze_pf3.py`, 정답 문자열 포함률) |
| 감싸거나 바꿀 것 | `C2NeighborAdapter.select_neighbors(ranked_chunk_ids, policy)`로 `pf3_rules.select`를 감쌈. ChunkMeta(doc_idx, flags)는 **생성자 `__init__(meta, cfg)`로 주입**(공통 규약). 반환은 `list[int]`(numpy 정수가 아닌 Python int). 반환은 R에 없는 이웃, chunk_id 정렬(지금 `select`와 같음). 입력과 출력은 chunk_id. PK 변환(`plan()` 안의 `pk_of_chunk`)은 C2에서 빼고 C1으로 넘김. 순위는 `hits`로 보존(`plan()`의 PK 정렬에 의존하지 않음). 대조군(ANN Top-(10+k), R3)은 C2 모듈이 아니라 실험 계층(D8) |
| 기존 수치와 이으려면 | nq_seq, ANN Top-10, ef 64, 64단어용 gold·flags, 같은 평가기(c2x에서 `pf3_quality_nq.csv`를 소수 넷째 자리까지 재현한 방식) |
| 확인·동의할 항목 | D3(평가 질문 `nq_test_ans` 2,254), D5, D8, D9 |

### C3 담당
| 구분 | 내용 |
|---|---|
| 그대로 쓰는 것 | 브랜치 `w12-c3-scheduling` `코드/bench_v2.py`의 PRESET m3(async, 제한 없음), m7(`rsem = Semaphore(cap)`이 요청 전체를 감쌈). `MilvusBackend.asearch_ids`, `afetch`(`query(filter="pk in [...]")`). `run_cond.sh`의 조건(n 64, cap 8, 예열 5초, 측정 20초, 3회, 교차) |
| 감싸거나 바꿀 것 | `C3SchedulerAdapter.run_many(query_keys, run_one, mode, clients, cap, warmup_s, duration_s, max_queries=None)`, **생성자 `__init__(meta, cfg)`**. **`max_queries`가 정수면 개수 기반 실행**(품질 평가용): `query_keys`를 고정 순서로(필요하면 순환) 정확히 그 수만큼 실행하고 멈춤. `max_queries = len(query_keys)`면 한 바퀴. warmup_s·duration_s는 쓰지 않고 모든 결과 `is_warmup=False`. m3/m7 동시성 제어와 queue_wait·total·service 기록은 시간 기반과 같음. **`max_queries=None`이면** 기존과 같은 **시간 기반 실행**(예열 `warmup_s` 5초 + 측정 `duration_s` 20초): `query_keys`는 고정 seed 순서 목록이고 클라이언트들이 순환하며 꺼내 쓰며, warmup_s + duration_s가 지나면 멈춤. 예열 구간에 끝난 결과는 `QueryResult.is_warmup=True`로 표시하고 집계(QPS, 지연 백분위, 오류율)에서 뺌(기존 `r[0] >= t_warm` 필터와 같은 역할). 요청 본문(검색·이웃·Fetch)은 `run_one`으로 받고 스케줄러 안에 두지 않음(`Mapper.context`의 expand는 C2 정책 `all`로 대체). **queue_wait를 semaphore 획득 전후로 따로 측정**(지금은 대기 전에 `t`를 찍어 지연에 섞임). 결과에 `queue_wait`, `total`(큐 진입부터), `service`를 기록. 질문은 `rng.randrange(103452)` 대신 `nq_test_real` 3,452개를 고정 seed 순서로 순환(D3). **C3 공식 비교의 Fetch는 기본 Milvus 일괄 Fetch 어댑터 `c1.impl=milvus_query_ref`**(query pk in, 512개씩, bench_v2의 `afetch`와 같은 방식) + **warm 환경**. C1 구현을 기다리지 않아도 됨. `file_ref`는 C3 비교에 쓰지 않음(파일 Fetch 스레드 수가 C3 동시성과 별개인 숨은 상한이 됨, `DESIGN.md` 4절) |
| 기존 수치와 이으려면 | `summary.json`의 `qps`는 기존 집계와 같음(예열 뒤 끝난 결과 수 / duration_s, 측정 구간 뒤 꼬리 포함). 추가 `strict_qps`는 측정 구간 안에서 끝난 결과만 셈(정의: `DESIGN.md` 4절). clients 64, cap 8, topk 10, ef 64, 같은 시간대 교차. **고부하 실험 중 Milvus 종료 원인을 관리자가 확인하기 전에는 m3, clients=64 부하 실험을 하지 않음(미결)**. 확인용 `all` 조건 1회(m3 vs m7, clients 64, D5). 질문 선택이 달라졌으므로(pseudo 포함 무작위 → real 고정 순환) 수치가 똑같지 않을 수 있음을 전제로 비교 |
| 확인·동의할 항목 | D3, D4, D5. 측정 예약 시간대(`run.reserved_window`, D12 확정) |

### 데이터·컬렉션 소유자 (junlim)
| 구분 | 내용 |
|---|---|
| 그대로 쓰는 것 | `/workspace/junlim/data_nq/`(chunks.jsonl, doc_idx.npy, pk_seq.npy, chunk_vecs.npy, queries.jsonl, query_vecs.npy), `/workspace/junlim/pf3_out/`(nq_gold.jsonl, nq_flags.npy), 컬렉션 `junlim/nq_seq`(HNSW IP 16/200, text mmap on, vec·index mmap off). 생성 코드 `prep_nq.py`, `load_big.py`, `gold_map.py`. 상태 점검 `check_busy.sh` |
| 감싸거나 바꿀 것 | 없음. 파이프라인은 위 파일과 컬렉션을 **읽기만** 하고 복사하지 않음(D2). 실행 시작 시 길이·md5 자동 검사 |
| 기존 수치와 이으려면 | 측정 기간에 위 파일 수정·이동 금지. `nq_seq`의 mmap·인덱스 설정 변경(`set_mmap.py` 등) 금지. load/release는 시간대 안에서 MilvusGuard 규칙으로만(실행 전 Loaded였던 컬렉션은 파이프라인이 release하지 않음) |
| 확인·동의할 항목 | 없음. **D1(nq_seq 사용), D2(원본 참조, 수정·이동 금지), D12(load 시간대)는 확정(junlim 동의, 2026-11-23)** |

## 자기 어댑터를 붙이는 방법 (SMOKE.md 3절과 같음)

**각자 본인 작업 폴더에서 한다.** 총괄 폴더(`/workspace/jaeeun/pipeline-core`)에는 코드를 넣지 않는다. 팀 저장소 `ppomee2002/vector-db`의 `pipeline-design` 브랜치를 본인 폴더(예: `/workspace/<내 이름>/vector-db`)에 clone하거나 `pipeline-core/` 폴더를 복사해서 작업하고, 계약 테스트를 통과시킨 뒤 PR을 올린다. Python은 pymilvus 3.0.3·numpy·pyyaml이 있는 환경(서버에서는 각자 venv).

0. **가져오기:** `git clone -b pipeline-design <저장소 URL> /workspace/<내 이름>/vector-db && cd /workspace/<내 이름>/vector-db/pipeline-core && git checkout -b feature/c2`(C1은 `feature/c1`, C3는 `feature/c3`). 원본 데이터 경로는 config.yaml에 절대 경로로 있어서 그대로 읽힌다(읽기만, D2).
1. **파일 추가:** 본인 clone의 `modules/` 아래에 자기 파일을 만든다(예: `modules/c2_bc1.py`). `modules/C1.py`, `C2.py`, `C3.py`는 원래 있던 빈 파일이라 여기에 써도 된다. 기존 코드는 옮기지 말고 이 파일에서 감싼다.
   ```python
   from modules.registry import register

   @register("c2", "bc1_v1")              # 종류(c1/c2/c3), 이름
   class MyC2:
       def __init__(self, meta, cfg): ...  # 생성자 규약 (meta, cfg)
       def select_neighbors(self, ranked_chunk_ids, *, policy): ...
   ```
   - C1: `async fetch_text(chunk_ids, *, mode, batch_impl) -> (dict[int, str], FetchStats)`, `fetch_source() -> str`. 선택: `milvus_collections()`, `aclose()`, `describe()`. 동시에 여러 번 불려도 결과가 섞이지 않아야 함.
   - C3: `async run_many(query_keys, run_one, *, mode, clients, cap, warmup_s, duration_s, max_queries=None)`.
   - 예시 코드: reference 4개 `modules/c1_file_reference.py`, `c1_milvus_query_reference.py`(B0 기본 Milvus Fetch), `c2_none_reference.py`, `c3_serial_reference.py`.
2. **이름 등록:** 본인 clone의 `config.yaml` `c1.impl` / `c2.impl` / `c3.impl`에 등록한 이름을 적거나, overlay(`runs/*.yaml`에 `c2: {impl: bc1_v1}`)·`--set c2.impl=bc1_v1`로 덮어쓴다. 등록 확인: `python main.py --list`
3. **계약 테스트 (C1 동시 실행 검사 포함):** FAIL 0이 될 때까지 고친다(FAIL마다 무엇이 왜 틀렸는지 한 줄씩 나옴). `file_ref`는 스레드 수(`c1.file_ref_threads`)를 반드시 적어야 하므로 `--run runs/correctness_file_ref.yaml`을 준다.
   ```bash
   cd /workspace/<내 이름>/vector-db/pipeline-core
   python experiments/contract_test.py --run runs/correctness_file_ref.yaml --c2 bc1_v1 --only c2,common --require-policies BC1,all
   python experiments/contract_test.py --run runs/correctness_file_ref.yaml --c1 <이름> --only c1,common   # 파일 C1
   python experiments/contract_test.py --run runs/correctness_milvus_query.yaml --c1 <이름> --only c1,common --allow-load   # Milvus C1
   python experiments/contract_test.py --run runs/correctness_file_ref.yaml --c3 <이름> --only c3,common
   python experiments/contract_test.py --run runs/correctness_file_ref.yaml --results results/<run>/results.jsonl   # 저장된 결과 검사
   ```
   C2·C3·공통 검사는 Milvus 없이 돌아간다(가짜 검색과 가짜 run_one). C1 검사는 본문을 `chunks.jsonl` 원문과 대조하고, **서로 다른 chunk_id 묶음 16개로 `fetch_text`를 동시에 불러** 각 결과가 자기 요청의 chunk_id·원문·FetchStats와 정확히 맞는지 본다(섞이면 FAIL). Milvus C1은 `--allow-load`가 있어야 Milvus를 쓰고, 그때도 MilvusGuard 규칙(아래 4)을 따른다.
4. **Smoke Test (질문 10개, `max_queries=10`, 정확성 확인):**
   ```bash
   nohup python main.py --run runs/correctness_milvus_query.yaml --set c2.impl=bc1_v1 --set c2.policy=BC1 > logs/smoke_bc1.log 2>&1 &
   ```
   - 결과는 본인 clone의 `results/<run.name>_<시각>/`(results.jsonl, eval.json, summary.json, run_meta.json). 종료 코드 0이고 `summary.json`의 checks 위반이 0이면 통과.
   - MilvusGuard: load 전에 Milvus 사용 여부(다른 컬렉션 load, 대상 컬렉션 포함 데이터 요청)를 확인하고 사용 중이면 멈춘다(종료 코드 3). 대상(`junlim/nq_seq`)이 **이미 Loaded면 기본 중단**(누가 쓰는 중일 수 있음). 원래 NotLoad였던 것만 load하고, 끝나면 release 직전 다시 확인한 뒤 내가 load한 것만 release한다. 확인할 수 없으면 release하지 않고 수동 명령을 남긴다(종료 코드 6). 측정 실행(`run.purpose: measure`)은 예약 시간대 `run.reserved_window`(KST) 밖이면 거부(종료 코드 7).
   - 잠금 파일은 각자 clone 안에만 있어 **다른 사람 실행과는 공유되지 않는다**(팀 공통 잠금은 미결). Milvus를 쓰는 실행은 D12 시간대 안에서 팀에 알리고 돌린다.
5. **PR:** 계약 테스트 결과(PASS/FAIL/SKIP 수)와 Smoke Test 결과 폴더 이름을 적어 `pipeline-design` 대상으로 PR을 올린다. 총괄이 순차 통합한다.

## 확인용 프롬프트 (각자 AI에게 그대로 붙여넣기)

설계 파일 위치: 팀 GitHub 저장소 `ppomee2002/vector-db`의 **`pipeline-design` 브랜치, `pipeline-core/` 폴더** (`DESIGN.md`, `interfaces.py`, `config.yaml`, `REVIEW_FOR_TEAM.md`). 서버에 접속하지 않고 자기 컴퓨터의 AI로도 확인할 수 있다. 비교할 기존 코드는 서버 경로와 팀 저장소 브랜치 중 손에 있는 쪽을 쓰면 된다. 어댑터를 붙여 계약 테스트를 돌리는 것은 **본인 작업 폴더의 clone**에서 한다(총괄 폴더에 코드를 넣지 않음).

**C1 (jihyeon)**
```text
팀 GitHub 저장소 ppomee2002/vector-db 의 pipeline-design 브랜치에서 pipeline-core/ 폴더의
DESIGN.md, interfaces.py, config.yaml, REVIEW_FOR_TEAM.md를 받아 읽고,
내 기존 코드(서버 경로: /workspace/jihyeon/c1_w13 의 C1 코드와 저장소 / 팀 저장소 브랜치: w12-layout-grouping)와 비교해서
① 그대로 맞는 것 ② 감싸면 되는 것 ③ 실제로 충돌하는 것(이유와 대안 포함)으로 나눠 보고하라.
특히 C1FetchAdapter.fetch_text의 (texts, FetchStats) 반환, batch_impl(query_pk_in / get), chunk_id↔pk 변환 위치,
w13c1_doc/rand 사용 방식, cold 측정 조건(D11), 생성자 (meta, cfg), fetch_source()("file:<경로>" 또는 "milvus:<db>/<collection>"),
FetchStats.bytes_returned 정의(반환 본문 텍스트의 UTF-8 바이트 합), 동시 호출에서 결과가 섞이지 않는지,
C1 Layout 비교는 내 어댑터 + cold 환경(관리자 설정 후, DESIGN.md 9절)이라는 점을 확인하라.
내 작업 폴더(/workspace/<내 이름>/)에 위 저장소·브랜치를 clone(또는 pipeline-core/ 복사)하고, 위 "자기 어댑터를 붙이는 방법"대로
그 clone의 modules/에 내 어댑터를 추가·등록한 뒤 experiments/contract_test.py를 내 어댑터로 돌려서 FAIL 0을 확인하라
(C1은 동시 실행 검사 포함). FAIL이 있으면 줄마다 원인과 고칠 방법을 정리하라. /workspace/jaeeun/pipeline-core에는 아무것도 쓰지 말 것.
통과하면 feature 브랜치로 커밋해 pipeline-design 대상 PR을 올릴 준비를 하라(PR 본문에 계약 테스트 PASS/FAIL/SKIP 수).
(명령: experiments/contract_test.py --run runs/correctness_file_ref.yaml --c1 <내 이름> --only c1,common.
 Milvus를 쓰는 C1이면 --run runs/correctness_milvus_query.yaml --allow-load, D12 시간대 안에서만.)
③은 D 번호나 interfaces 항목 이름으로 표시하라. 기존 코드는 수정하지 말 것(어댑터 파일 추가만 허용). 계약 테스트의 --allow-load 말고는 Milvus에 load/release 등 상태를 바꾸는 호출을 하지 말 것.
```

**C2 담당**
```text
팀 GitHub 저장소 ppomee2002/vector-db 의 pipeline-design 브랜치에서 pipeline-core/ 폴더의
DESIGN.md, interfaces.py, config.yaml, REVIEW_FOR_TEAM.md를 받아 읽고,
내 기존 코드(팀 저장소 브랜치: w12-c2-prefetching 의 코드/pf3_rules.py, bench_pf4.py, analyze_pf3.py, gold_map.py /
서버 경로: /workspace/junlim 의 같은 이름 파일 또는 내 작업 폴더 <경로>)와 비교해서
① 그대로 맞는 것 ② 감싸면 되는 것 ③ 실제로 충돌하는 것(이유와 대안 포함)으로 나눠 보고하라.
특히 C2NeighborAdapter.select_neighbors의 입력·반환 규칙, ChunkMeta 생성자 주입, PK 변환을 C1으로 넘기는 것,
순위 보존(hits), 대조군을 실험 계층에서 만드는 것(D8), 평가 질문 집합(D3), 생성자 (meta, cfg)를 확인하라.
내 작업 폴더(/workspace/<내 이름>/)에 위 저장소·브랜치를 clone(또는 pipeline-core/ 복사)하고, 위 "자기 어댑터를 붙이는 방법"대로
그 clone의 modules/에 내 어댑터를 추가·등록한 뒤 experiments/contract_test.py를 내 어댑터로 돌려서 FAIL 0을 확인하라
(C1은 동시 실행 검사 포함). FAIL이 있으면 줄마다 원인과 고칠 방법을 정리하라. /workspace/jaeeun/pipeline-core에는 아무것도 쓰지 말 것.
통과하면 feature 브랜치로 커밋해 pipeline-design 대상 PR을 올릴 준비를 하라(PR 본문에 계약 테스트 PASS/FAIL/SKIP 수).
(명령: experiments/contract_test.py --run runs/correctness_file_ref.yaml --c2 <내 이름> --only c2,common --require-policies BC1,all)
③은 D 번호나 interfaces 항목 이름으로 표시하라. 기존 코드는 수정하지 말 것(어댑터 파일 추가만 허용).
```

**C3 담당**
```text
팀 GitHub 저장소 ppomee2002/vector-db 의 pipeline-design 브랜치에서 pipeline-core/ 폴더의
DESIGN.md, interfaces.py, config.yaml, REVIEW_FOR_TEAM.md를 받아 읽고,
내 기존 코드(팀 저장소 브랜치: w12-c3-scheduling 의 코드/bench_v2.py, run_cond.sh, run_reclaim.sh /
서버 경로: /workspace/junlim 의 같은 이름 파일 또는 내 작업 폴더 <경로>)와 비교해서
① 그대로 맞는 것 ② 감싸면 되는 것 ③ 실제로 충돌하는 것(이유와 대안 포함)으로 나눠 보고하라.
특히 C3SchedulerAdapter.run_many의 계약(max_queries=None이면 warmup_s·duration_s 시간 기반 실행과 QueryResult.is_warmup으로 예열 제외,
max_queries가 정수면 그 수만큼 한 바퀴 실행하고 멈춤·모두 is_warmup=False), 생성자 (meta, cfg),
m3/m7 정의, queue_wait를 semaphore 전후로 따로 재는 것, total/service 정의,
고정 seed 질문 순환(D3), clients 64 / cap 8 / 예열·측정·반복 조건(D4), all 확인 조건(D5),
C3 공식 비교는 기본 Milvus 일괄 Fetch(milvus_query_ref) + warm이고 file_ref는 쓰지 않는다는 점, qps / strict_qps 정의를 확인하라.
부하 실험(m3, clients=64)은 Milvus 종료 원인이 확인될 때까지 실행하지 말 것.
내 작업 폴더(/workspace/<내 이름>/)에 위 저장소·브랜치를 clone(또는 pipeline-core/ 복사)하고, 위 "자기 어댑터를 붙이는 방법"대로
그 clone의 modules/에 내 어댑터를 추가·등록한 뒤 experiments/contract_test.py를 내 어댑터로 돌려서 FAIL 0을 확인하라
(C1은 동시 실행 검사 포함). FAIL이 있으면 줄마다 원인과 고칠 방법을 정리하라. /workspace/jaeeun/pipeline-core에는 아무것도 쓰지 말 것.
통과하면 feature 브랜치로 커밋해 pipeline-design 대상 PR을 올릴 준비를 하라(PR 본문에 계약 테스트 PASS/FAIL/SKIP 수).
(명령: experiments/contract_test.py --run runs/correctness_file_ref.yaml --c3 <내 이름> --only c3,common)
③은 D 번호나 interfaces 항목 이름으로 표시하라. 기존 코드는 수정하지 말 것(어댑터 파일 추가만 허용).
```

**데이터·컬렉션 소유자 (junlim)**
```text
팀 GitHub 저장소 ppomee2002/vector-db 의 pipeline-design 브랜치에서 pipeline-core/ 폴더의
DESIGN.md, interfaces.py, config.yaml, REVIEW_FOR_TEAM.md를 받아 읽고,
내 데이터와 코드(서버 경로: /workspace/junlim/data_nq, /workspace/junlim/pf3_out, prep_nq.py, load_big.py, gold_map.py, set_mmap.py,
Milvus 컬렉션 junlim/nq_seq / 팀 저장소 브랜치: w12-c2-prefetching·w12-c3-scheduling 의 코드/prep_nq.py, load_big.py, set_mmap.py)와 비교해서
① 그대로 맞는 것 ② 감싸면 되는 것 ③ 실제로 충돌하는 것(이유와 대안 포함)으로 나눠 보고하라.
특히 config.yaml의 경로·md5·컬렉션 설정이 실제와 맞는지 확인하라(D1·D2·D12는 확정됨, 2026-11-23 동의).
D12 load 시간대에 걸리는 내 일정이나 실험이 있으면 알려 달라.
③은 D 번호나 config 항목 이름으로 표시하라. 파일과 코드는 수정하지 말 것. Milvus에는 상태를 바꾸는 호출을 하지 말 것.
```

## 회의에 가져올 것
- **③ 실제 충돌 항목만**, 한 줄씩: `[D 번호 또는 interfaces/config 항목] 충돌 내용 / 이유 / 제안 대안`.
- 동의 여부: jihyeon(w13c1_doc·rand), 관리자(cold 실험용 캐시 설정). junlim(D1, D2, D12)은 확정(2026-11-23).
- Milvus 실행이 필요한 본인 일정(날짜·시간대·컬렉션) → `run.reserved_window`에 적을 시간대.
- 계약 테스트 결과: 자기 어댑터의 PASS / FAIL 수(FAIL 0이 목표).

## 미결 항목 (회의에서 결정)
| 항목 | 현재 상태 | 결정할 것 |
|---|---|---|
| `milvus.search` (D1) | **해결: nq_seq 확정(junlim 동의, 2026-11-23).** config.yaml 기본값 `junlim/nq_seq`. 기존 B0 호환 확인용 `runs/b0_smoke.yaml`만 `jaeeun/nq1m_sel` | – |
| 팀 공통 잠금 | 합의된 방식 없음. 지금은 예약 시간대(`run.reserved_window`)와 각자 폴더 안 잠금 파일(자기 실행끼리만) + MilvusGuard 사용 중 검사 | 팀 전체가 Milvus 실행 전에 잡을 공통 잠금의 위치·방식 |
| 고부하 실험 중 Milvus 종료 원인 | 관리자 확인 필요(11-23 13:29 KST에도 Milvus 재시작 기록) | 원인 확인 전에는 **m3, clients=64 부하 실험을 실행하지 않음** |
| cold 실험 관리자 설정 | cold / cache_constrained는 관리자 설정 필요, 다른 팀원에게 영향 | 설정 방법·시간대(`DESIGN.md` 9절) |

## 팀 단톡방 안내 메시지 (초안)

```text
이번 주에 Milvus 기반 C1·C2·C3 통합 프로토타입을 만들려고 공통 구조와 인터페이스를 정리했습니다.
NQ 100만 chunk 기준으로 ANN 검색 → C2 이웃 선택 → C1 원문 Fetch 순서이고, C3가 검색부터 Fetch까지 전체 동시 실행을 제어합니다.
기존 데이터와 코드를 최대한 재사용하는 방향이라, 각자 AI로 자기 코드와 충돌하는 부분만 확인해서 회의에 가져와 주세요.
파일: 팀 GitHub 저장소 ppomee2002/vector-db 의 pipeline-design 브랜치, pipeline-core/ 폴더 (DESIGN.md, interfaces.py, config.yaml)
브랜치: <링크>
확인 방법과 붙여넣을 프롬프트: 같은 폴더의 REVIEW_FOR_TEAM.md (자기 컴퓨터의 AI로도 확인 가능)
nq_seq 사용·원본 참조·load 시간대는 junlim 님 동의로 확정됐고, w13c1_doc·rand 사용 동의와 cold 실험 관리자 설정은 회의에서 같이 확인하겠습니다.
```
