# C1 실험 (13주차): 문서 순 저장소 + 묶어 읽기

김지현 2024134018, 캡스톤 VectorDB팀 C1 담당

## 무엇을 하나

NQ 100만 Chunk(64단어)의 본문을 Milvus 밖 파일 저장소에 두 가지 배치로 저장하고, 읽는 방식에 따라 쿼리당 디스크 읽기가 얼마나 달라지는지 잰다.

- 배치
  - `doc`: 문서 순. `chunks.jsonl` 순서 그대로이고, 이 순서가 문서 번호, 문서 안 순서와 같은지 만들 때 확인한다.
  - `rand`: 무작위(seed 1). 기준선이다.
- 읽기
  - 개별: chunk마다 따로 pread한다.
  - 묶어 읽기: 붙어 있는 4KB 블록을 pread 한 번으로 읽는다. `gap`을 주면 그만큼 떨어진 블록도 이어서 읽는다.
- 모든 읽기는 O_DIRECT라서 페이지 캐시를 거치지 않는다. 캐시를 따로 비울 필요가 없다.

| 실험 | 조건 | 개수 |
|---|---|---|
| E1 | 배치 {rand, doc} × 읽기 {개별, 묶어 읽기} × 범위 {top-10, top-10 + B+C1 이웃, top-10 ±1 전부} | 12 |
| E2 | doc + 묶어 읽기에서 gap {4, 16, 64KB} × 범위 {B+C1 이웃, ±1 전부} | 6 |
| E3 | 배치 × 읽기 × Rerank 전 후보 {ANN 30, ANN 100} | 8 |

조건 26개를 3회씩, 총 78번 실행한다. 회차마다 조건 순서를 섞는다. 쿼리는 NQ test 3,452개다.

## 입력 (읽기만 함, 수정·삭제하지 않음)

| 파일 | 주인 | 용도 |
|---|---|---|
| `/workspace/junlim/data_nq/chunks.jsonl` | 임준 | 본문, 문서 번호. chunk_id = 줄 번호 = Milvus pk (nq_seq, nq1m_sel 둘 다) |
| `/workspace/jaeeun/sel_fetch/data/ann_top100_real.csv` | 이재은 | ANN top-100 (E3) |
| `/workspace/jaeeun/sel_fetch/data/rerank_top100_real.csv` | 이재은 | Rerank 후 최종 top-10 (E1, E2) |
| `/workspace/jaeeun/sel_fetch/data/qrels_chunk.tsv` | 이재은 | 정답 chunk (포함률 계산, 없으면 건너뜀) |

- 입력 파일의 크기, 수정 시각, md5를 `env.json`과 `store/meta.json`에 기록한다. 나중에 원본이 바뀌었는지 확인할 수 있다.
- 이웃 규칙은 임준 `pf3_rules.py`(md5 2754643f)를 `nbr_rules.py`로 복사해 쓴다.

## 안전장치

- 모든 쓰기는 `/workspace/jihyeon` 아래에서만 한다(`common.safe_path`). 밖으로 쓰려고 하면 실행 전에 멈춘다.
- 입력 파일은 읽기 모드로만 연다(`common.open_read`).
- 서버 규칙을 지킨다: 단일 스레드, 메모리 약 1~2GB, 저장소 약 0.8GB(배치 2개), 작업은 tmux에서.
- 측정 전에 같은 장치를 다른 사람이 많이 읽는지 5초 동안 보고, 바쁘면 최대 30분 기다린다(아무것도 바꾸지 않음).
- 측정 중에는 앱이 센 읽기와 `/proc/diskstats`를 비교해서 다른 사람 영향이 있었는지 기록한다.

## 실행 방법

### 1. 코드 올리기 (PC에서, Tailscale 연결 후)

```bash
scp -P 2222 -r c1_w13 vdb0X@100.82.53.54:/workspace/jihyeon/c1_w13_code
```

### 2. 서버에서 실행

```bash
ssh -p 2222 vdb0X@100.82.53.54
cd /workspace/jihyeon/c1_w13_code
python3 -c "import numpy, pandas; print('ok')"      # 없으면 내 venv를 만들어 설치: python3 -m venv /workspace/jihyeon/venv
bash run_all.sh build      # 저장소 만들기, 몇 분
bash run_all.sh check      # 검증, PASS가 나와야 함
bash run_all.sh quick      # 쿼리 300개 점검, 1~2분
tmux new -d -s c1 'bash run_all.sh full'           # 본 실험. 시작 전 단톡방에 측정 시간 공유
tmux attach -t c1          # 진행 보기 (나올 때 Ctrl+b, d)
```

venv를 쓰면 `PY=/workspace/jihyeon/venv/bin/python bash run_all.sh ...`처럼 실행한다.

### 3. 결과를 PC로 받기

- Windows: `powershell -ExecutionPolicy Bypass -File get_results.ps1 -Account vdb0X`
- macOS/Linux: `bash get_results.sh vdb0X`

기본 저장 위치는 `문서\capstone\c1_w13` (Windows), `~/capstone/c1_w13` (macOS/Linux)다.

## 결과 파일 (`/workspace/jihyeon/c1_w13/`)

| 경로 | 내용 |
|---|---|
| `store/` | 저장소 (`doc.bin`, `rand.bin`, offset 표, `meta.json`) |
| `results/check_store.json` | 검증 결과 (전수 비교, read() 비교, diskstats 비율) |
| `results/run_YYYYMMDD_HHMM/summary.md` | 발표용 표 (E1, E2, E3, 읽는 범위별 정답 포함률) |
| `results/run_.../summary.csv` | 조건별 요약 (회차 중앙값, 범위) |
| `results/run_.../per_run.csv` | 실행 78개 각각의 앱·디스크 합계 |
| `results/run_.../per_query.csv.gz` | 쿼리별 원자료 (약 27만 줄) |
| `results/run_.../env.json` | 실행 환경, 입력 md5, 측정 전 장치 상태 |
| `results/run_.../code/` | 실행에 쓴 코드 사본과 MD5SUMS |
| `results/run_YYYYMMDD_HHMM.tgz` | 위 폴더 묶음 (PC로 받는 파일) |
| `logs/` | build, check, quick, full 로그 |

## 파이프라인에서 쓰기

```python
from c1_store import Store
st = Store("/workspace/jihyeon/c1_w13/store", layout="doc")
texts, stats = st.read(chunk_ids)                 # chunk_id = Milvus pk
# stats = {"n_chunks", "n_reads", "bytes_read", "useful_bytes", "latency_us"}
texts, stats = st.read(chunk_ids, merge=False)    # 기준선 (chunk마다 따로 읽기)
```

## Milvus 실험 (c1_milvus.py, run_milvus.sh)

본문을 Milvus 안에 두고 C1을 잰다. 파이프라인 검색과 준이 C2·C3가 Milvus라서 같은 환경이다.

- database `jihyeon`에 `w13c1_doc`(pk = 줄 번호 = 문서 순), `w13c1_rand`(pk 무작위, seed 1)를 만든다. 각 100만 행이다.
- `w13c1_`로 시작하는 내 컬렉션만 만들고, release하고, 지운다. 다른 database와 컬렉션은 조회하지도 않는다.
- 벡터는 8차원 가짜 벡터(FLAT)다. 검색 결과는 재은이 파일을 쓰고, 본문 조회만 잰다. 팀 설정(HNSW)과 다른 점이다.
- `warmup.scalarField=disable`로 설정해서, load 직후 첫 조회는 저장소에서 읽는다(W12와 같은 방식).
- M1 콜드: 쿼리 100개 × 14조건. 조회마다 release/load하고 Milvus 메트릭(읽은 바이트, cell 로드)과 지연을 잰다.
- M2 웜: 한 번 load하고 예열한 뒤 쿼리 500개 × 20조건 × 3회. 한 번에 요청과 Chunk마다 요청의 지연을 비교한다.

```bash
export PY=/workspace/jihyeon/venv/bin/python
bash run_milvus.sh check      # 접속, 메트릭 확인 (아무것도 안 만듦)
bash run_milvus.sh probe      # 2만 행 테스트 후 삭제, PROBE PASS가 나와야 함
bash run_milvus.sh load       # 단톡방 공지 후. 100만 행 컬렉션 2개
bash run_milvus.sh quick      # M1 쿼리 5개 + M2 쿼리 50개 점검
tmux new -d -s c1m 'cd /workspace/jihyeon/c1_w13_code && PY=/workspace/jihyeon/venv/bin/python bash run_milvus.sh m1 && PY=/workspace/jihyeon/venv/bin/python bash run_milvus.sh m2'
```

결과는 `/workspace/jihyeon/c1_w13/milvus/m1_cold_*`, `m2_warm_*`에 생기고, 같은 이름의 `.tgz`를 PC로 받는다. 실험이 다 끝나면 `bash run_milvus.sh drop`으로 내 컬렉션을 지운다.

## 한계

- 단일 스레드, 쿼리 하나씩 읽는다. 동시 요청은 C3와 붙인 뒤에 본다.
- 공유 장치(loop)라서 지연은 다른 사람 영향을 받을 수 있다. 그래서 읽기 횟수와 바이트를 주 지표로 쓴다.
- 정답 포함률은 정답 문단(qrels) 기준이다. 준이 C2 발표의 정답 문자열 기준 숫자와는 다르다.
