# CHECK: 통합 파이프라인 설계 초안 점검 (2026-11-22 조회)

> 원래 위치는 `/workspace/jaeeun/pipeline-core/CHECK.md`였다. 그런데 `pipeline-core` 폴더 권한이 `dr-x------`(쓰기 불가)라서 만들 수 없었다. 권한 변경도 수정에 해당하므로 하지 않고 임시 위치에 저장했다. 옮기려면: `chmod u+w /workspace/jaeeun/pipeline-core && cp <이 파일> /workspace/jaeeun/pipeline-core/CHECK.md`

- 대상: `DESIGN.md`, `interfaces.py`, `config.yaml` (11/23 회의용 초안)
- 방법: **조회만 했다.** Milvus는 `list_databases / list_collections / describe_collection / get_collection_stats / list_indexes / describe_index / get_load_state / list_partitions`와 메트릭(`localhost:9091/metrics`) GET만 썼다. load·release·query·insert·설정 변경은 하지 않았다. 팀원 폴더와 `team_ref`는 읽기만 했다(`git show origin/<브랜치>:<파일>`).
- 조회 스크립트와 원본 출력(같은 임시 폴더): `milvus_ro.py`, `milvus_ro.out`
- 못 한 것: junlim의 `check_busy.sh`는 관리자 스위치 상태를 보려고 원격 서버에 ssh로 접속하는데, 이 실행이 권한 단계에서 거부됐다. 그래서 Milvus 컨테이너 메모리와 디스크 여유 공간은 직접 확인하지 못했고, junlim이 남긴 최근 기록(`/workspace/junlim/post_check.txt`, 11/22 15:54)을 인용했다.

---

## 1. 미확정 항목

### 1-1. dataset.root: 원본 파일 (모두 NQ 64단어 100만 chunk 기준)

| 항목 | 실제 경로 | 소유자 | 크기·형태 | 근거 |
|---|---|---|---|---|
| **원본 chunks** | `/workspace/junlim/data_nq/chunks.jsonl` | junlim (vdb03) | 445,687,794 B, 100만 줄, `{doc_idx, chunk_idx, text}` | `ls -la`, md5 `aca9a920…` (jihyeon w13c1 meta와 같음) |
| doc_idx | `/workspace/junlim/data_nq/doc_idx.npy` | junlim | int64 (1,000,000), 단조 증가, 문서 27,133개 | 로드해서 확인 |
| pk_seq | `/workspace/junlim/data_nq/pk_seq.npy` | junlim | int64, **항등(pk == 줄 번호)** | `(pk==arange).all() → True` |
| chunk 임베딩 | `/workspace/junlim/data_nq/chunk_vecs.npy` | junlim | **float32** (1,000,000 × 384), MiniLM-L6-v2. `meta.json`에는 `fp16: true`라고 되어 있음(fp16로 반올림한 뒤 f32로 저장한 것, norm ≈ 1.0004) | shape/dtype, 샘플 3개 재임베딩 cos 1.0000~1.0004 |
| 질문 텍스트 | `/workspace/junlim/data_nq/queries.jsonl` | junlim | 103,452줄 = **real 3,452 (q_idx 0~3451)** + pseudo 100,000 (3452~) | `Counter(kind)` |
| 질문 임베딩 | `/workspace/junlim/data_nq/query_vecs.npy` | junlim | float32 (103,452 × 384) | shape |
| 정답 문자열 | `/workspace/junlim/pf3_out/nq_gold.jsonl` (pf4_out, pf4b_out에 같은 파일, md5 `95ae90dc…`) | junlim | 3,452줄(`qi` 0~3451). `ans`가 있는 것 2,255개(c2x 평가 가능 2,254개), `ans=null` 1,197개 | `Counter(ans is not None)` |
| 문장 경계 flags | `/workspace/junlim/pf3_out/nq_flags.npy` (pf4/pf4b와 같음, md5 `5d007e77…`) | junlim | uint8 (1,000,000). bit0 = 문장 중간에서 시작, bit1 = 문장 중간에서 끝남 | `pf3_rules.py` 주석 |
| qrels (chunk 단위) | `/workspace/jaeeun/sel_fetch/data/qrels_chunk.tsv` | jaeeun | 9,822행 `q_idx, qid, corpus_id, doc_idx, chunk_id` | head |
| qrels (원본 BEIR) | `/workspace/jihyeon/data/nq/qrels/` (+ `corpus.jsonl` 1.5GB) | jihyeon | mteb/nq 원본 | `ls` |
| 메타 묶음 (내 것) | `/workspace/jaeeun/sel_fetch/data/chunk_meta.npz` (`doc_idx, chunk_idx, nbytes`) | jaeeun | ChunkMeta 후보 | `zipfile.namelist()` |
| 파일 저장소 (C1 문서 순) | `/workspace/jaeeun/sel_fetch/data/chunk_store_seq.bin` + `chunk_store_off.npy` (내 것), `/workspace/jihyeon/c1_w13/store/{doc,rand}.bin` + `*_off.npy`, `len.npy` (jihyeon) | 각자 | 397MB, 4KB 블록 기준 | `ls`, jihyeon `store/meta.json` |
| c2x test 질문 | `/workspace/jaeeun/sel_fetch_c2x/results/test_queries.jsonl` (2,000개, NQ train에서 뽑음) + 임베딩 `/workspace/jaeeun/sel_fetch/data/q_train_minilm.npy` | jaeeun | **q_idx가 0부터 다시 시작하는 다른 번호 체계** | head |

**주의할 파일:** `/workspace/junlim/gold_w100/nq_{gold,flags}`는 100단어 chunking용이다(flags 640,336개). 64단어 파이프라인에 섞으면 안 된다. `/workspace/junlim/data/`는 FiQA(133,919 chunk)다.

**제안:** `dataset.root: /workspace/junlim/data_nq`, `gold: /workspace/junlim/pf3_out/nq_gold.jsonl`, `flags: /workspace/junlim/pf3_out/nq_flags.npy`. 원본이 전부 junlim 폴더에 있으니 **junlim의 동의와 "수정·이동 금지" 약속이 필요하다.** 대안은 md5를 고정해서 내 폴더로 복사하는 것이다(약 2.2GB, `/workspace` 여유 25GB).

### 1-2. Milvus 서버·DB·컬렉션

- 서버 **2.6.23** (`get_server_version`), 클라이언트 pymilvus 3.0.3 (`/workspace/jaeeun/venv`). C3 README도 같은 버전이다.
- DB: `default`(비어 있음), `junlim`, `jaeeun`, `jihyeon`.
- 저장 크기는 메트릭 `milvus_datacoord_stored_binlog_size`와 `stored_index_files_size`에서 읽었다.

| DB/컬렉션 | 행 수 | 필드 (PK = `pk` INT64, auto_id 아님) | 본문 | mmap | 인덱스 | 저장 (binlog + index) | load |
|---|---|---|---|---|---|---|---|
| **junlim/nq_seq** | 1,000,000 | pk, doc_idx(INT64), chunk_idx(INT32), vec(FLOAT_VECTOR 384), text(VARCHAR 4096) | **있음** | text **on**, vec **off**(명시), index mmap **off**(명시) | HNSW IP M=16 efC=200, Finished | 1.98GB + 1.68GB = **3.66GB** | NotLoad |
| **jaeeun/nq1m_sel** | 1,000,000 | pk, doc_idx, chunk_idx, vec(384) | **없음** (본문은 파일) | 설정 없음(기본값) | HNSW IP M=16 efC=200, Finished | 1.58GB + 1.68GB = **3.26GB** | NotLoad |
| junlim/nq100_seq | 640,336 | nq_seq와 같음 | 있음 | text on | HNSW IP 16/200 | 1.41 + 1.08GB | NotLoad |
| junlim/fiqa_seq | 133,919 | nq_seq와 같음 | 있음 | text on | HNSW IP 16/200 | 0.26 + 0.23GB | NotLoad |
| **jihyeon/w13c1_doc** | 1,000,000 | pk, **chunk_id**, doc_idx, chunk_idx, text(VARCHAR 65535), vec(**8차원 가짜**) | 있음 | 설정 없음. 컬렉션 속성 `warmup.scalarField=disable` | FLAT L2 (가짜) | 0.48GB | NotLoad |
| **jihyeon/w13c1_rand** | 1,000,000 | w13c1_doc과 같음 | 있음 | 같음 | FLAT L2 (가짜) | 0.48GB | NotLoad |

- nq_seq와 nq1m_sel은 **같은 인덱스 파라미터**(HNSW/IP/16/200)이고 c2x에서 본문·doc_idx·pk_seq 전수 일치를 이미 확인했다.
- **새로 확인한 것:** jihyeon의 `w13c1_doc/rand`는 **같은 `data_nq/chunks.jsonl`**(md5 같음)로 만든 100만 chunk 컬렉션이다. `chunk_id` = 줄 번호. doc 배치는 pk = chunk_id(항등), rand 배치는 `default_rng(1).permutation`. 확인: `milvus_meta/pk_doc.npy` 항등, `pk_rand.npy` 순열, `store/doc_idx.npy` == junlim `doc_idx.npy`. 벡터가 가짜이므로 **검색은 못 하고 본문 조회 전용**이다. **C1 물리 배치 비교용 "같은 데이터의 비교 컬렉션"이 이미 있다는 뜻이다**(DESIGN 4절의 전제 조건).

### 1-3. 통합 컬렉션 후보 비교

| | A. junlim/nq_seq (검색 + 본문) | B. jaeeun/nq1m_sel (검색) + 본문 파일 | C. nq1m_sel (검색) + jihyeon/w13c1_doc·rand (본문 조회) | D. 내 DB에 본문 포함 컬렉션을 새로 적재 |
|---|---|---|---|---|
| 기존 실험과의 연속성 | **C2·C3 실험이 모두 이 컬렉션**(bench_v2/pf4의 `junlim`, `nq_seq`) | c2x·v2·io 실험(파일 Fetch, O_DIRECT) | C1 w13 실험이 이 컬렉션 | 없음 (새로 검증해야 함) |
| Fetch 경로 | Milvus query 한 번 (검색과 같은 컬렉션) | 파일 pread. Milvus Fetch가 아님 | Milvus `get`. 검색 컬렉션과 다름 | Milvus |
| C1 배치 비교 | 불가 (seq 하나뿐) | 파일로는 가능 | **doc vs rand 바로 가능** | 다시 적재해야 함 |
| 소유권·위험 | junlim 소유. load/release/mmap 변경이 C3 측정에 영향. **동의 필수** | 내 소유, 위험 없음 | 검색은 내 것, 본문은 jihyeon 동의 필요 | 내 소유 |
| 추가 디스크 | 0 | 0 | 0 | Milvus 저장 약 **3.7GB** (nq_seq 기준: binlog 1.98 + index 1.68). 본문만 추가하면 nq1m_sel 대비 +0.40GB. 적재 중 임시 공간은 별도 |
| load 메모리 (추정) | HNSW 1.68GB + vec(mmap off) + 본문 0.40GB (mmap) | HNSW만 | HNSW + 본문 컬렉션 2개(각 약 0.4GB) | A와 비슷 |

- **남은 공간:** 이 셸의 `/workspace`(=/, `/dev/loop1`)는 98G 중 73G 사용, **여유 25G**(`df -h`). **Milvus 데이터는 이 셸에서 보이지 않는다**(프로세스 목록에 milvus/etcd/minio가 없다. junlim 기록상 원격 컨테이너 `mv-milvus`). 따라서 Milvus 저장소의 여유 공간은 **확인하지 못했다.** D를 고르면 관리자나 junlim에게 확인을 받아야 한다.
- **의견:** 1차 기능 Baseline과 C2·C3는 **A(nq_seq)** 가 가장 싸다(기존 C2·C3 수치와 이어진다). 동의를 못 받으면 **C**(검색은 nq1m_sel, 본문은 w13c1_doc)로 가고, C1 배치 비교도 C를 쓰면 적재가 필요 없다. D는 디스크 확인 전에는 보류한다.

### 1-4. canonical chunk_id = pk_seq 타당성, q_idx 대응

| 점검 | 결과 | 근거 |
|---|---|---|
| pk_seq == chunks.jsonl 줄 번호 | 항등, 문제 없음 | `pk_seq.npy` 비교 |
| nq_seq / nq1m_sel의 pk == 줄 번호 | c2x 전수 대조로 확인됨 (본문·doc_idx 일치) | REPORT_c2x 1절 |
| w13c1_doc pk == chunk_id, w13c1_rand pk ≠ chunk_id | 맞다. rand에서는 C1 adapter가 `pk_rand[chunk_id]`로 바꿔야 한다. 컬렉션에 `chunk_id` 필드가 있어 결과를 역으로 확인할 수 있다 | `pk_doc/pk_rand.npy` |
| C2·C3 코드의 ID 체계 | bench_v2 `Mapper`: `pk_of_chunk = pk_{layout}.npy`, `chunk_of_pk` 역매핑. pf4 `plan()`도 chunk 번호로 규칙을 적용한 뒤 pk로 바꾼다 → **설계의 "chunk_id로 통일, adapter에서 PK 변환"과 같은 구조** | bench_v2.py `Mapper`, bench_pf4.py L86–90 |
| chunk_vecs[k] ↔ chunks[k] | 줄 0, 500000, 999999 재임베딩 cos 1.00005 / 1.00003 / 1.00043 | MiniLM CPU 재임베딩 |
| q_idx → 질문 텍스트 → 정답 | `nq_gold.qi`(3,452개)의 `q`가 `queries.jsonl[qi].text`와 **모두 일치(불일치 0)** | 전수 비교 |
| q_idx → 질문 임베딩 | q 0, 1, 1000, 3451(real), 3452, 50000(pseudo): cos 0.9998~1.0004. 각 질문의 최근접 real 질문이 자기 자신 | 재임베딩 6개 |

- **결론: pk_seq(=줄 번호)를 canonical chunk_id로 써도 문제없다.** 남는 함정은 세 가지다.
  1. **q_idx 이름 공간이 둘이다.** `data_nq/queries.jsonl`(NQ test, real 0~3451)과 c2x test(NQ train에서 뽑은 2,000개, 다시 0부터). → `query_set` + `q_idx`를 함께 키로 쓰자.
  2. **pseudo 질문(3452~)에는 정답이 없다.** C3 bench_v2는 103,452개 전체에서 무작위로 뽑는다(아래 1-5).
  3. 마지막 문서(doc 27132)는 100만 개에서 잘려 끝부분이 없을 수 있다(`prep_nq.py build_chunks`의 `texts[:target]`). 영향은 미미하지만 ±1 이웃 경계에서 알고 있어야 한다.

### 1-5. C3 m3 / m7 정의와 config 조건 (w12-c3-scheduling `코드/bench_v2.py`)

| | 정의 (코드) |
|---|---|
| 요청 1건 | `asearch_ids(qv, topk=10)` → `Mapper.context(ids, expand=1)` = **Top-10 ∪ 같은 문서 ±1 전부(= C2 `all`)** → `afetch` (`query(filter="pk in [...]")`, 512개씩) |
| **m3** | `mode=async, merge=none`. AsyncMilvusClient, 클라이언트 n개가 각각 요청을 연달아 보낸다. 동시 수 제한 없음 |
| **m7** | m3 + `rsem = asyncio.Semaphore(cap)`로 **요청 전체(검색+읽기)** 를 감쌈. 지연 측정 시작(`t`)이 semaphore 대기 **전**이라 기록된 지연에 대기 시간이 포함된다 |
| (참고) m4 / m6 / m8 | m4 = 쿼리 안 PK 연속 구간 병합(range filter), m6 = Fetch만 상한, m8 = m7 + 쿼리 간 병합(XBatcher) |
| 질문 선택 | `qi = rng.randrange(len(M.qv))` → **103,452개(대부분 pseudo)에서 복원 추출**. 고정 목록이 아니다 |

**기존 실험 조건 (`run_cond.sh`, `run_reclaim.sh`, `run_cap.sh`):** `CFGS=("m1 64" "m3 64" "m3 8" "m5 64" "m6 64" "m7 64 8" "m8 64 16")`, dur 20s, warm 5s, 3회. **n=16은 예열(warmup)에만 썼다.**

| config 값 | 기존 조건 | 판정 |
|---|---|---|
| clients=16 | m7 비교는 **n=64**, m3는 n=64와 n=8 | **불일치.** 기존 m7 우위 결과(같은 시간대 NQ: m7 n64 cap8 p99 162~180ms vs m3 n64 209~237ms)를 재현하려면 clients=64 |
| cap=8 | m7 64/8 | 일치 |
| topk=10, ef=64 | 같음 (`MilvusBackend(ef=64)`) | 일치 |
| C2 정책 | 기존 C3는 expand=1 = `all` | B0(`none`)와 다르다. C3 비교 시 C2 정책을 명시해야 함 |

- 참고: 같은 시간대 NQ에서 **m3 n=8**이 622~641 QPS, p50 약 10ms로 m7(64/8, 632~653 QPS, p50 약 95ms)과 처리량이 같았다. m7의 이점은 "클라이언트가 많아 과부하일 때 대기열을 앞단에서 제한하는 것"이다. 따라서 **clients=16에서는 m3와 m7 차이가 작을 가능성이 높다**(n=16 m7은 측정한 적 없음).

### 1-6. C1 fetch_mode ↔ Milvus 호출 (기존 코드)

| fetch_mode | 기존 구현 | Milvus 호출 | 기존 결과 |
|---|---|---|---|
| individual | jihyeon `c1_milvus.py parts_for("each")` | chunk마다 `client.get(ids=[pk], output_fields=["chunk_id","text"])` | warm: BC1 18.2요청 p50 약 85ms vs 한 번에 5.2ms |
| batch | jihyeon `"one"` → `client.get(ids=[...])` 한 번 / C2·C3 `afetch` → `query(filter="pk in [...]")` | **두 가지 경로가 섞여 있음**(get vs query) | warm p50 5.0~5.4ms |
| (split) | jihyeon `"split"`: Top-10 먼저, 이웃은 두 번째 요청 | get 2회 | cold: 읽은 MB 같음, 지연 ×1.07(doc) / ×1.23(rand) |
| range | C3 `m4`: `f_range()` → `"(pk >= a and pk <= b) or pk == c ..."` filter로 `query` | query 1회 (filter만 다름) | NQ n=16: m4 250~269 QPS vs m3 371~413 (**range가 더 느림**) |

- **range는 PK 범위 query로 구현할 수 있고 이미 구현되어 있다(m4).** 다만 (1) 기존 측정에서 `pk in`보다 느렸다. (2) 범위가 의미 있으려면 PK가 물리 순서와 같아야 한다(seq/doc 배치). rand 배치에서는 연속 구간이 거의 없어 `pk ==` OR 나열로 퇴화한다. (3) Milvus가 PK 범위 filter를 디스크 연속 읽기로 바꾼다는 보장이 없다. DESIGN의 "range는 구현·검증된 경우에만"이 맞다.
- C1 물리 효과 근거(jihyeon m1_cold, 쿼리마다 release/load, 100개): **doc/rand 읽은 MB 비율 BC1 0.34, 지연 비율 0.56**, top-10 0.59/0.79. warm에서는 doc과 rand 차이가 없다(p50 5.25 vs 5.20ms).

### 1-7. 서버 공유 상태 (2026-11-22 16:10~16:40 KST)

| 항목 | 상태 | 근거 |
|---|---|---|
| load된 컬렉션 | **없음** (6개 모두 NotLoad) | `get_load_state`, 메트릭 `milvus_querynode_collection_num 0` |
| Milvus 활동 | CPU 0.16코어, 요청 0.0건/초 (10초 표본) | metrics 두 번 표본 |
| Milvus 프로세스 RSS | 0.79~0.85GiB | `process_resident_memory_bytes` |
| Milvus 컨테이너 메모리 | 3.54GiB (anon 0.89, file 2.50), memory.high/max 제한 없음. 오늘 reset 기록 12:42, 13:12, 13:25, 15:35, 15:53 (junlim) | `/workspace/junlim/post_check.txt` (15:54, **직접 조회 아님**) |
| 다른 계정 프로세스 | jihyeon(vdb02) tmux·bash만, CPU 0 | `ps` |
| GPU | A2 × 2, 둘 다 0MiB, 0% | `nvidia-smi` |
| 호스트 메모리 | 188Gi 중 used 17Gi, buff/cache 103Gi, **swap 7.3/8.0Gi 사용** | `free -h` |
| 디스크 (/workspace) | 98G, 사용 73G, 여유 25G (75%) | `df -h` |
| Milvus 저장소 디스크 | **확인 못 함** (원격 컨테이너) | – |

---

## 2. 설계 빈틈 점검

| | 제안 | 판정 | 근거·의견 |
|---|---|---|---|
| (a) | C2 비교에 "같은 토큰 수 ANN Top-(10+k)"와 "±1 이웃 무작위(R3)" 추가 | **동의. 필수로 넣자** | c2x: BC1 − none +5.55%p 중 대부분이 Context 증가 효과, BC1 − R3 = +0.99%p. 이번 파이프라인에는 Rerank가 없으므로 c2x의 F1(Rerank 확대)에 해당하는 것은 **ANN Top-(10+k)** 가 맞다. jihyeon 기존 수치(chunk 정답 포함률)도 BC1 18.2개 0.842 vs ANN 30개 0.834라서 비교가 실제로 갈릴 수 있다. **정할 것:** k를 질문별 BC1 추가 수로 맞출지(권장: 질문별 매칭, 토큰은 평균 86/chunk로 거의 비례하므로 토큰 수는 보고만) 평균으로 맞출지. R3는 seed 20개. 주 지표는 `nq_gold` 정답 문자열 포함률(평가 가능 2,254개) |
| (b) | ChunkMeta 모듈 | **동의** | pf3_rules `select(rule, R, doc, flags)`가 이미 doc 배열과 flags를 따로 받는다. bench_v2 `Mapper`도 doc_idx와 pk 매핑을 들고 있다. ChunkMeta = {doc_idx, chunk_idx, flags, nbytes, layout별 pk_of_chunk, (파일 C1) offset/len}. 출처: `data_nq/doc_idx.npy`, `pf3_out/nq_flags.npy`, `sel_fetch/data/chunk_meta.npz`(nbytes), `pk_seq.npy`, jihyeon `milvus_meta/pk_{doc,rand}.npy`, `store/{doc,rand}_off.npy, len.npy`. **로드할 때 md5와 길이(1,000,000)를 검사**하자 |
| (c) | fetch_text가 FetchStats(요청 수, 바이트, 블록 수) 반환 | **조건부 동의** | 요청 수, PK 수, 반환 글자·바이트는 Milvus 경로에서도 질문 단위로 셀 수 있다(bench_v2 `count()`가 이미 그렇게 함). **블록 수·실제 디스크 바이트는 Milvus 경로에서 질문 단위로 얻을 수 없다.** jihyeon이 쓴 `internal_cache_cell_access_miss_bytes_total`은 서버 전체 값이라 동시 요청에서 섞인다. 블록 수는 파일 저장소 C1(offset/len으로 계산)에서만 정확하다. → FetchStats의 블록·디바이스 필드는 Optional로 두고, Milvus 경로에서는 "구간 전체 메트릭 차이"로 따로 기록 |
| (d) | Top-10 ∪ 이웃을 한 번에 Fetch | **동의, 기존 C2 코드가 실제로 이 순서였다** | bench_pf4 `one()`: `ids = asearch_ids()` → `pks = C.plan(ids, rule)` (= ids ∪ 규칙 이웃, **PK 정렬**) → `afetch(pks)` **query 1회**. BC1·A·all 모두 ID와 메타만으로 고른다(텍스트 불필요). 두 번 읽는 경로는 정답을 아는 참고 정책(`lazy_need`, `text_need`)뿐이다. jihyeon cold: split은 읽은 MB 같고 지연 +7~23%. 주의: `plan()`은 PK 순으로 정렬해서 **순위 정보가 사라진다.** QueryResult에는 hit 순위를 따로 보존해야 한다 |
| (e) | stage_ms = embed, search, c2_select, fetch, queue_wait, total | **대체로 동의, 정의를 보완** | (1) `embed`: 지금 질문 벡터는 `query_vecs.npy`로 미리 계산되어 있다(C2·C3 모두 그대로 사용). 실시간 임베딩을 넣을지 정해야 하고, 안 넣으면 키를 두되 기록하지 않는 것으로 명시. (2) `queue_wait`: bench_v2 m7은 대기 시간을 지연에 섞어 기록한다 → semaphore 획득 전후로 시각을 따로 찍어야 한다. (3) `total` = 큐 진입부터(대기 포함). 서비스 시간 = total − queue_wait도 같이. (4) `map`(chunk_id↔pk)은 미미하지만 c2_select에 포함한다고 명시 |
| (f) | SearchHit에 점수, topk는 설정값 | **동의 + 주의 1개** | `SearchHit.score`와 `Settings.ann_topk`는 이미 있다. 그러나 `QueryResult.top10_ids`라는 이름이 topk 고정을 전제한다 → `hits: list[SearchHit]`로. **2차 선별(A1 등)은 Top-100을 쓴다(v2·c2x: ef=256).** Milvus HNSW는 ef ≥ limit이어야 하므로 **topk=100이면 ef=64를 그대로 못 쓴다.** search_topk(후보)와 context_topk(10)를 분리하고 ef를 topk에 맞추는 규칙을 config에 둬야 한다 |
| (g) | 그 밖의 불일치 | 아래 표 | |

### (g) interfaces.py / DESIGN.md와 실제 코드·데이터가 다른 곳

| 위치 | 문서 내용 | 실제 | 제안 |
|---|---|---|---|
| DESIGN 3절 "중요" | "기존 C1 Milvus 실험은 약 15만 Chunk / FLAT·COSINE / 다른 레이아웃" | **w13 C1 실험(11/21)은 같은 NQ 100만 chunk**(`data_nq/chunks.jsonl`, md5 일치), doc/rand 컬렉션, 가짜 8차원 FLAT L2, 본문 조회만 | 문장 갱신. "w12 수치는 비교 불가, w13 수치는 같은 데이터·다른 경로(get, 가짜 벡터)"로 |
| DESIGN 4절 C1 비교 | "물리 Layout 비교는 비교 컬렉션이 확보되었을 때" | `jihyeon/w13c1_doc`, `w13c1_rand`가 이미 있다 | jihyeon 동의를 받으면 바로 사용 가능하다고 적기 |
| DESIGN 3절 결과 데이터 | "기존 NQ 평가 Query 중 고정된 목록" | C3 bench는 103,452개(pseudo 포함)에서 무작위 복원 추출. 정답이 있는 질문은 2,255개 | 품질 평가는 real 중 `ans` 있는 2,254개, 부하 측정은 고정 seed 순서 목록(real 3,452 순환 또는 pseudo 포함 여부 결정) |
| config `c3.clients: 16` | – | 기존 m7 비교는 n=64 | 1-5 참고. 결정 사항 |
| config `c2.policy` 기본 none, C3 비교 | – | 기존 C3 수치는 C2=`all` | C3 비교 조건에 C2 정책 고정값을 명시 |
| interfaces `FetchMode` | individual/batch/range | batch에 `get(ids)`(C1)와 `query(pk in)`(C2·C3) 두 구현이 있다 | B0의 batch를 하나로 고정. 기존 C2·C3와 이으려면 `query(pk in)` |
| interfaces `C1FetchAdapter.fetch_text` | `Mapping[int,str]`만 반환 | – | `(texts, FetchStats)` 반환 ((c)) |
| interfaces `C2NeighborAdapter.select_neighbors` | ID만 받음 | 규칙에 doc_idx, flags 필요 | 생성자에서 ChunkMeta 주입 ((b)). 반환은 "R에 없는 이웃, chunk_id 정렬"(pf3_rules와 같음)로 명시 |
| interfaces `QueryResult` | `q_idx`만 | q_idx 체계가 둘(NQ test / c2x train) | `query_set` 필드 추가 |
| interfaces `Settings` | 컬렉션·DB·layout 없음 | 검색 컬렉션과 본문 컬렉션이 다를 수 있다(후보 C) | `search_collection`, `fetch_collection`, `layout` 분리 |
| config `ann.metric: IP` | – | chunk_vecs는 fp16 반올림 후 f32라 norm ≈ 1.0004 | IP ≈ cosine. 문제는 없지만 기록해 두기 |
| config `milvus.collection` 하나 | – | 후보 C면 2개 | search/fetch 분리 |
| DESIGN 2절 | "C3가 Search~Fetch 전체 제어" | bench_v2 m7이 정확히 이 구조(`rsem`이 `one(qi)` 전체를 감쌈) | 일치. m6(Fetch만 상한)과 혼동하지 않도록 명시 |
| 공유 서버 주의 | – | 오늘만 Milvus reset 5회(junlim 기록), swap 7.3/8GiB 사용 | 측정 전 load 상태·reset 기록·swap을 run 메타에 남기기 |

---

## 3. 회의에서 팀이 결정할 항목

| # | 결정 사항 | 선택지 | 근거 / 권장 |
|---|---|---|---|
| D1 | 통합 컬렉션 | A nq_seq / B nq1m_sel + 파일 / C nq1m_sel + w13c1_doc·rand / D 새로 적재 | **A 권장**(C2·C3 연속성, 추가 디스크 0). junlim이 동의하지 않으면 **C**. D는 Milvus 디스크 확인 후. 어느 쪽이든 **load·release를 누가 언제 하는지**(공유 시간표)를 정해야 한다 |
| D2 | 원본 데이터 위치 | junlim 폴더를 그대로 참조 / md5를 고정해서 공용 위치에 복사 | 참조하면 공간 0이지만 수정 위험. 복사는 약 2.2GB(여유 25GB). 권장: 참조 + md5 검사를 실행 시작 시 자동으로 |
| D3 | 질문 집합 | NQ test real 3,452 / 그중 정답 있는 2,254 / c2x test 2,000(train) / pseudo 포함 부하 | 품질은 2,254(dev) + c2x test 2,000(독립). 부하는 고정 seed 목록. `query_set` 키 도입 |
| D4 | C3 clients | 16 / 64 / 둘 다 | 기존 결과 재현은 64. 16에서는 m3와 m7 차이가 작을 것으로 예상(m3 n=8이 이미 m7과 같은 처리량). **권장: 64를 주 조건, 16은 보조** |
| D5 | C3 비교 시 C2 정책 | all(기존 C3와 같음) / BC1(통합 목표) / none(B0) | 기존 수치와 비교하려면 all. 통합 결과는 BC1. 둘 다 돌리면 조건이 2배 |
| D6 | B0의 batch 구현 | `query(filter="pk in")` / `get(ids)` | C2·C3 연속성은 query. C1 w13은 get. 하나로 고정하고 다른 하나는 C1 비교 항목으로 |
| D7 | range 모드 | 1차 제외 / m4 코드 재사용 | 기존 측정에서 m4가 m3보다 느렸고 rand 배치에서는 의미가 없다. **1차 제외 권장** |
| D8 | C2 대조군 | ANN Top-(10+k) 질문별 매칭 / 평균 매칭, R3 seed 수 | c2x 결론상 필수. 질문별 매칭 + R3 seed 20 권장 |
| D9 | 2차 선별 대비 search topk와 ef | topk 10 / ef 64 고정 · 후보 topk 100 / ef 256 분리 | Milvus HNSW는 ef ≥ limit이 필요. 1차는 10/64, 인터페이스는 후보 topk를 분리해 두기 |
| D10 | embed 단계 | 미리 계산한 벡터(현행) / 실시간 MiniLM(GPU) | 기존 C2·C3는 미리 계산. 실시간이면 GPU 2장 중 사용 규칙 필요 |
| D11 | I/O 기록 방식 | 질문 단위 FetchStats(요청·PK·바이트) + 구간 단위 Milvus 메트릭 차이 / 파일 저장소 블록 수 | Milvus에서는 질문 단위 블록 수를 얻을 수 없다. 물리 I/O 주장은 cold 조건(jihyeon m1처럼 release/load)이나 파일 경로에서만 |
| D12 | 컬렉션 소유자 동의 | junlim(nq_seq, 원본 파일), jihyeon(w13c1_*) | 회의에서 바로 확인. mmap·load 설정 변경 금지 범위도 같이 |
