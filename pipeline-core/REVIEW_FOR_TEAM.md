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
| 감싸거나 바꿀 것 | `C1FetchAdapter.fetch_text(chunk_ids, mode, batch_impl)`가 `({chunk_id: text}, FetchStats)`를 반환. `FetchStats.n_requests / n_pks / bytes_returned`는 필수, blocks와 device_bytes는 파일 경로에서만 채움. `batch_impl`에 `query_pk_in`(B0 기본, D6) 추가. chunk_id → pk 변환은 `ChunkMeta.pk_of_chunk[layout]`으로 C1 내부에서. 검색은 하지 않고 nq_seq 검색 결과(chunk_id)를 입력으로 받음. 결과 dict의 키는 pk가 아니라 chunk_id. **생성자 `__init__(meta, cfg)`**(컬렉션·저장소 경로는 cfg에서 읽음). **`fetch_source()`**로 실제 본문 출처를 `"file:<데이터 파일 경로>"` 또는 `"milvus:<db>/<collection>"`으로 밝힘(실행기가 실행 메타에 기록하고, `Settings.fetch_collection`과 다르면 둘 다 남김). **`FetchStats.bytes_returned` = 반환된 본문 텍스트의 UTF-8 바이트 합**(`sum(len(t.encode("utf-8")) for t in texts.values())`. 응답 메시지·메타 필드·블록 크기는 넣지 않음. 계약 테스트가 FAIL로 검사). Milvus를 쓰면 `milvus_collections()`로 컬렉션을 선언(실행기가 사용 중 검사 후 load·release) |
| 기존 수치와 이으려면 | 같은 100만 chunk(md5 `aca9a920…`), 같은 읽기 집합(top10 / BC1 / all1), warm과 cold(쿼리마다 release/load) 구분 유지. 물리 I/O 주장은 cold나 파일 경로에서만(D11). cold는 load/release 시간표 안에서만 |
| 확인·동의할 항목 | **w13c1_doc·rand 사용 동의**(C1 비교, D1 대안 C의 본문 컬렉션), D6, D7, D11, D12 |

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
| 감싸거나 바꿀 것 | `C3SchedulerAdapter.run_many(query_keys, run_one, mode, clients, cap, warmup_s, duration_s, max_queries=None)`, **생성자 `__init__(meta, cfg)`**. **`max_queries`가 정수면 개수 기반 실행**(품질 평가용): `query_keys`를 고정 순서로(필요하면 순환) 정확히 그 수만큼 실행하고 멈춤. `max_queries = len(query_keys)`면 한 바퀴. warmup_s·duration_s는 쓰지 않고 모든 결과 `is_warmup=False`. m3/m7 동시성 제어와 queue_wait·total·service 기록은 시간 기반과 같음. **`max_queries=None`이면** 기존과 같은 **시간 기반 실행**(예열 `warmup_s` 5초 + 측정 `duration_s` 20초): `query_keys`는 고정 seed 순서 목록이고 클라이언트들이 순환하며 꺼내 쓰며, warmup_s + duration_s가 지나면 멈춤. 예열 구간에 끝난 결과는 `QueryResult.is_warmup=True`로 표시하고 집계(QPS, 지연 백분위, 오류율)에서 뺌(기존 `r[0] >= t_warm` 필터와 같은 역할). 요청 본문(검색·이웃·Fetch)은 `run_one`으로 받고 스케줄러 안에 두지 않음(`Mapper.context`의 expand는 C2 정책 `all`로 대체). **queue_wait를 semaphore 획득 전후로 따로 측정**(지금은 대기 전에 `t`를 찍어 지연에 섞임). 결과에 `queue_wait`, `total`(큐 진입부터), `service`를 기록. 질문은 `rng.randrange(103452)` 대신 `nq_test_real` 3,452개를 고정 seed 순서로 순환(D3) |
| 기존 수치와 이으려면 | clients 64, cap 8, topk 10, ef 64, 같은 시간대 교차. 확인용 `all` 조건 1회(m3 vs m7, clients 64, D5). 질문 선택이 달라졌으므로(pseudo 포함 무작위 → real 고정 순환) 수치가 똑같지 않을 수 있음을 전제로 비교 |
| 확인·동의할 항목 | D3, D4, D5, D12 |

### 데이터·컬렉션 소유자 (junlim)
| 구분 | 내용 |
|---|---|
| 그대로 쓰는 것 | `/workspace/junlim/data_nq/`(chunks.jsonl, doc_idx.npy, pk_seq.npy, chunk_vecs.npy, queries.jsonl, query_vecs.npy), `/workspace/junlim/pf3_out/`(nq_gold.jsonl, nq_flags.npy), 컬렉션 `junlim/nq_seq`(HNSW IP 16/200, text mmap on, vec·index mmap off). 생성 코드 `prep_nq.py`, `load_big.py`, `gold_map.py`. 상태 점검 `check_busy.sh` |
| 감싸거나 바꿀 것 | 없음. 파이프라인은 위 파일과 컬렉션을 **읽기만** 하고 복사하지 않음(D2). 실행 시작 시 길이·md5 자동 검사 |
| 기존 수치와 이으려면 | 측정 기간에 위 파일 수정·이동 금지. `nq_seq`의 mmap·인덱스 설정 변경(`set_mmap.py` 등) 금지. load/release는 시간표대로 |
| 확인·동의할 항목 | **D1(nq_seq 사용), D2(원본 참조, 수정·이동 금지), D12(load/release 시간표)** |

## 자기 어댑터를 붙이는 방법 (SMOKE.md 3절과 같음)

서버 `/workspace/jaeeun/pipeline-core`(팀 저장소에서는 `pipeline-core/`) 기준. Python은 `/workspace/jaeeun/venv/bin/python`.

1. **파일 추가:** `modules/` 아래에 자기 파일을 만든다(예: `modules/c2_bc1.py`). `modules/C1.py`, `C2.py`, `C3.py`는 원래 있던 빈 파일이라 여기에 써도 된다. 기존 코드는 옮기지 말고 이 파일에서 감싼다.
   ```python
   from modules.registry import register

   @register("c2", "bc1_v1")              # 종류(c1/c2/c3), 이름
   class MyC2:
       def __init__(self, meta, cfg): ...  # 생성자 규약 (meta, cfg)
       def select_neighbors(self, ranked_chunk_ids, *, policy): ...
   ```
   - C1: `async fetch_text(chunk_ids, *, mode, batch_impl) -> (dict[int, str], FetchStats)`, `fetch_source() -> str`. 선택: `milvus_collections()`, `aclose()`, `describe()`.
   - C3: `async run_many(query_keys, run_one, *, mode, clients, cap, warmup_s, duration_s, max_queries=None)`.
   - 예시 코드: reference 3개 `modules/c1_file_reference.py`, `c2_none_reference.py`, `c3_serial_reference.py`.
2. **이름 등록:** `config.yaml`의 `c1.impl` / `c2.impl` / `c3.impl`에 등록한 이름을 적는다(기본값은 reference: `file_ref` / `none_ref` / `serial_ref`). 한 번만 바꿔 돌릴 때는 overlay(`runs/*.yaml`에 `c2: {impl: bc1_v1}`)나 `--set c2.impl=bc1_v1`로 덮어써도 된다. 등록 확인: `/workspace/jaeeun/venv/bin/python main.py --list`
3. **계약 테스트:** FAIL 0이 될 때까지 고친다(FAIL마다 무엇이 왜 틀렸는지 한 줄씩 나옴).
   ```bash
   cd /workspace/jaeeun/pipeline-core
   /workspace/jaeeun/venv/bin/python experiments/contract_test.py --c2 bc1_v1 --only c2,common --require-policies BC1,all
   /workspace/jaeeun/venv/bin/python experiments/contract_test.py --c1 <이름> --only c1,common        # Milvus C1이면 --allow-load (run.milvus_allowed_load 안에서만)
   /workspace/jaeeun/venv/bin/python experiments/contract_test.py --c3 <이름> --only c3,common
   /workspace/jaeeun/venv/bin/python experiments/contract_test.py --results results/<run>/results.jsonl   # 저장된 결과 검사
   ```
   C2·C3·공통 검사는 Milvus 없이 돌아간다(가짜 검색과 가짜 run_one 사용). C1 검사는 본문을 `chunks.jsonl` 원문과 대조하므로 서버에서 돌린다.
4. **B0 Smoke Test (질문 10개, `max_queries=10`):**
   ```bash
   cd /workspace/jaeeun/pipeline-core && nohup /workspace/jaeeun/venv/bin/python main.py --run runs/b0_smoke.yaml --set c2.impl=bc1_v1 --set c2.policy=BC1 > logs/b0_smoke_bc1.log 2>&1 &
   ```
   - 결과는 `results/<run.name>_<시각>/`(results.jsonl, eval.json, summary.json, run_meta.json). 종료 코드 0이고 `summary.json`의 checks 위반이 0이면 통과.
   - 실행기는 load 전에 Milvus 사용 여부(다른 컬렉션 load, 데이터 요청)를 확인하고, 사용 중이면 멈춘다(종료 코드 3). load는 `run.milvus_allowed_load` 목록에 있는 컬렉션만, 끝나면 반드시 release. 지금 overlay의 검색 컬렉션은 총괄 소유 `jaeeun/nq1m_sel`이므로, 돌리기 전에 총괄에게 시간을 알려 주거나 load 시간표(D12) 안에서 돌린다.

## 확인용 프롬프트 (각자 AI에게 그대로 붙여넣기)

설계 파일 위치: 팀 GitHub 저장소 `ppomee2002/vector-db`의 **`w13-pipeline-design` 브랜치, `pipeline-core/` 폴더** (`DESIGN.md`, `interfaces.py`, `config.yaml`, `REVIEW_FOR_TEAM.md`). 서버에 접속하지 않고 자기 컴퓨터의 AI로도 확인할 수 있다. 비교할 기존 코드는 서버 경로와 팀 저장소 브랜치 중 손에 있는 쪽을 쓰면 된다.

**C1 (jihyeon)**
```text
팀 GitHub 저장소 ppomee2002/vector-db 의 w13-pipeline-design 브랜치에서 pipeline-core/ 폴더의
DESIGN.md, interfaces.py, config.yaml, REVIEW_FOR_TEAM.md를 받아 읽고,
내 기존 코드(서버 경로: /workspace/jihyeon/c1_w13 의 C1 코드와 저장소 / 팀 저장소 브랜치: w12-layout-grouping)와 비교해서
① 그대로 맞는 것 ② 감싸면 되는 것 ③ 실제로 충돌하는 것(이유와 대안 포함)으로 나눠 보고하라.
특히 C1FetchAdapter.fetch_text의 (texts, FetchStats) 반환, batch_impl(query_pk_in / get), chunk_id↔pk 변환 위치,
w13c1_doc/rand 사용 방식, cold 측정 조건(D11), 생성자 (meta, cfg), fetch_source()("file:<경로>" 또는 "milvus:<db>/<collection>"),
FetchStats.bytes_returned 정의(반환 본문 텍스트의 UTF-8 바이트 합)를 확인하라.
서버의 pipeline-core에서 위 "자기 어댑터를 붙이는 방법"대로 내 어댑터를 modules/에 추가하고 등록한 뒤,
experiments/contract_test.py를 내 어댑터로 돌려서 FAIL 0을 확인하라. FAIL이 있으면 줄마다 원인과 고칠 방법을 정리하라.
(명령: experiments/contract_test.py --c1 <내 이름> --only c1,common. Milvus를 쓰는 C1이라 --allow-load가 필요하면 load 시간표(D12) 안에서만.)
③은 D 번호나 interfaces 항목 이름으로 표시하라. 기존 코드는 수정하지 말 것(어댑터 파일 추가만 허용). 계약 테스트의 --allow-load 말고는 Milvus에 load/release 등 상태를 바꾸는 호출을 하지 말 것.
```

**C2 담당**
```text
팀 GitHub 저장소 ppomee2002/vector-db 의 w13-pipeline-design 브랜치에서 pipeline-core/ 폴더의
DESIGN.md, interfaces.py, config.yaml, REVIEW_FOR_TEAM.md를 받아 읽고,
내 기존 코드(팀 저장소 브랜치: w12-c2-prefetching 의 코드/pf3_rules.py, bench_pf4.py, analyze_pf3.py, gold_map.py /
서버 경로: /workspace/junlim 의 같은 이름 파일 또는 내 작업 폴더 <경로>)와 비교해서
① 그대로 맞는 것 ② 감싸면 되는 것 ③ 실제로 충돌하는 것(이유와 대안 포함)으로 나눠 보고하라.
특히 C2NeighborAdapter.select_neighbors의 입력·반환 규칙, ChunkMeta 생성자 주입, PK 변환을 C1으로 넘기는 것,
순위 보존(hits), 대조군을 실험 계층에서 만드는 것(D8), 평가 질문 집합(D3), 생성자 (meta, cfg)를 확인하라.
서버의 pipeline-core에서 위 "자기 어댑터를 붙이는 방법"대로 내 어댑터를 modules/에 추가하고 등록한 뒤,
experiments/contract_test.py를 내 어댑터로 돌려서 FAIL 0을 확인하라. FAIL이 있으면 줄마다 원인과 고칠 방법을 정리하라.
(명령: experiments/contract_test.py --c2 <내 이름> --only c2,common --require-policies BC1,all)
③은 D 번호나 interfaces 항목 이름으로 표시하라. 기존 코드는 수정하지 말 것(어댑터 파일 추가만 허용).
```

**C3 담당**
```text
팀 GitHub 저장소 ppomee2002/vector-db 의 w13-pipeline-design 브랜치에서 pipeline-core/ 폴더의
DESIGN.md, interfaces.py, config.yaml, REVIEW_FOR_TEAM.md를 받아 읽고,
내 기존 코드(팀 저장소 브랜치: w12-c3-scheduling 의 코드/bench_v2.py, run_cond.sh, run_reclaim.sh /
서버 경로: /workspace/junlim 의 같은 이름 파일 또는 내 작업 폴더 <경로>)와 비교해서
① 그대로 맞는 것 ② 감싸면 되는 것 ③ 실제로 충돌하는 것(이유와 대안 포함)으로 나눠 보고하라.
특히 C3SchedulerAdapter.run_many의 계약(max_queries=None이면 warmup_s·duration_s 시간 기반 실행과 QueryResult.is_warmup으로 예열 제외,
max_queries가 정수면 그 수만큼 한 바퀴 실행하고 멈춤·모두 is_warmup=False), 생성자 (meta, cfg),
m3/m7 정의, queue_wait를 semaphore 전후로 따로 재는 것, total/service 정의,
고정 seed 질문 순환(D3), clients 64 / cap 8 / 예열·측정·반복 조건(D4), all 확인 조건(D5)을 확인하라.
서버의 pipeline-core에서 위 "자기 어댑터를 붙이는 방법"대로 내 어댑터를 modules/에 추가하고 등록한 뒤,
experiments/contract_test.py를 내 어댑터로 돌려서 FAIL 0을 확인하라. FAIL이 있으면 줄마다 원인과 고칠 방법을 정리하라.
(명령: experiments/contract_test.py --c3 <내 이름> --only c3,common)
③은 D 번호나 interfaces 항목 이름으로 표시하라. 기존 코드는 수정하지 말 것(어댑터 파일 추가만 허용).
```

**데이터·컬렉션 소유자 (junlim)**
```text
팀 GitHub 저장소 ppomee2002/vector-db 의 w13-pipeline-design 브랜치에서 pipeline-core/ 폴더의
DESIGN.md, interfaces.py, config.yaml, REVIEW_FOR_TEAM.md를 받아 읽고,
내 데이터와 코드(서버 경로: /workspace/junlim/data_nq, /workspace/junlim/pf3_out, prep_nq.py, load_big.py, gold_map.py, set_mmap.py,
Milvus 컬렉션 junlim/nq_seq / 팀 저장소 브랜치: w12-c2-prefetching·w12-c3-scheduling 의 코드/prep_nq.py, load_big.py, set_mmap.py)와 비교해서
① 그대로 맞는 것 ② 감싸면 되는 것 ③ 실제로 충돌하는 것(이유와 대안 포함)으로 나눠 보고하라.
특히 config.yaml의 경로·md5·컬렉션 설정이 실제와 맞는지, 원본 참조와 수정·이동 금지(D2), nq_seq 사용(D1),
load/release 시간표(D12)에 걸리는 내 일정이나 실험이 있는지 확인하라.
③은 D 번호나 config 항목 이름으로 표시하라. 파일과 코드는 수정하지 말 것. Milvus에는 상태를 바꾸는 호출을 하지 말 것.
```

## 회의에 가져올 것
- **③ 실제 충돌 항목만**, 한 줄씩: `[D 번호 또는 interfaces/config 항목] 충돌 내용 / 이유 / 제안 대안`.
- 동의 여부: junlim(D1, D2), jihyeon(w13c1_doc·rand).
- load/release가 필요한 본인 일정(날짜·시간대·컬렉션).
- 계약 테스트 결과: 자기 어댑터의 PASS / FAIL 수(FAIL 0이 목표).

## 미결 항목 (회의에서 결정)
| 항목 | 현재 상태 | 결정할 것 |
|---|---|---|
| `milvus.search` (D1) | config.yaml 기본값은 `junlim/nq_seq`(junlim 동의 대기). B0 Smoke Test는 `runs/b0_smoke.yaml` overlay에서 총괄 소유 `jaeeun/nq1m_sel`로 덮어써서 실행 중 | 통합 검색 컬렉션 확정(A: `junlim/nq_seq` / 대안 C: `jaeeun/nq1m_sel` + 본문 `jihyeon/w13c1_doc`). A로 정하면 overlay의 `milvus` 블록을 지우고 `run.milvus_allowed_load`를 바꿈(SMOKE.md 4절) |

## 팀 단톡방 안내 메시지 (초안)

```text
이번 주에 Milvus 기반 C1·C2·C3 통합 프로토타입을 만들려고 공통 구조와 인터페이스를 정리했습니다.
NQ 100만 chunk 기준으로 ANN 검색 → C2 이웃 선택 → C1 원문 Fetch 순서이고, C3가 검색부터 Fetch까지 전체 동시 실행을 제어합니다.
기존 데이터와 코드를 최대한 재사용하는 방향이라, 각자 AI로 자기 코드와 충돌하는 부분만 확인해서 회의에 가져와 주세요.
파일: 팀 GitHub 저장소 ppomee2002/vector-db 의 w13-pipeline-design 브랜치, pipeline-core/ 폴더 (DESIGN.md, interfaces.py, config.yaml)
브랜치: <링크>
확인 방법과 붙여넣을 프롬프트: 같은 폴더의 REVIEW_FOR_TEAM.md (자기 컴퓨터의 AI로도 확인 가능)
컬렉션 사용 동의(nq_seq, w13c1_doc·rand)와 Milvus load 시간표도 회의에서 같이 확인하겠습니다.
```
