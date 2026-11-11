# m3bench: RAG Context Retrieval의 I/O 대기·경합 측정 도구

캡스톤 디자인 VectorDB 팀, 역할 **M3 (I/O Waiting & Contention)** 실험 코드와 결과입니다.

> **주장:** RAG의 Context Retrieval 과정에서 I/O 대기와 동시 요청 간 자원 경합이 성능 병목을 유발한다.

RAG의 문맥 검색(벡터 검색 → 문서 chunk 읽기)을 실제 NVMe SSD 서버에서 재현했습니다. 동시 요청을 1개에서 64개까지 늘리며 처리량, p95 응답 시간, CPU, SSD 지표를 측정했습니다. 이를 통해 **CPU 부족 / SSD 경합 / 동기식 대기** 중 무엇이 병목인지 구분합니다.

## 핵심 결과

| | 동기 (기준) | 스레드 증설 | 비동기 I/O |
|---|---|---|---|
| 최대 처리량 (req/s) | 3,350 | 5,160 (1.54배) | **5,390 (1.61배)** |
| p95 응답 시간 (n=64) | 19.6 ms | 21.3 ms | **13.7 ms** |
| CPU 사용 | 64% | 99% | 100% |
| CPU 차례를 기다리는 스레드 | 0.3개 | 69개 | 0.8개 |

- **기다리는 구조가 병목**
  - 동기 서버는 CPU 64%, SSD 성능의 10%만 쓰는데도 3,350 req/s에서 멈춥니다.
  - 응답 시간의 89%가 worker가 비기를 기다린 시간입니다.
  - 간단한 계산 값(worker 7개 ÷ 요청당 2.08 ms = 3,365 req/s)과도 일치합니다.
- **스레드를 늘리면 CPU 경합 발생**
  - 처리량은 늘지만 CPU 7개를 두고 스레드가 줄을 서서 p95 응답 시간은 오히려 늘어납니다.
  - 이때 앱이 잰 read 1회는 1.42 ms인데 SSD 처리 시간은 0.078 ms입니다. 나머지는 CPU를 기다린 시간입니다.
- **비동기 I/O로 해소:** 같은 CPU 7개로 처리량 1.61배(95% 신뢰구간 1.59–1.63), p95 응답 시간은 가장 짧습니다.
- **조건을 바꿔도 같은 결과**
  - k, chunk 크기, 캐시, 검색 비용을 바꾼 8가지 조건 모두에서 비동기가 더 빨랐습니다(1.06~3.39배).
  - 읽기 대기 비중이 클수록 이득도 컸습니다.
- **SSD 경합은 특정 조건에서만 나타남:** chunk를 64 KB로 키우거나 검색 없이 읽기만 할 때만 SSD가 한계에 닿습니다.

| 조건 (동기 서버의 읽기 대기 비중) | 비동기 ÷ 동기 |
|---|---|
| 캐시 큼 (9%) | 1.06배 |
| 검색 무거움 (12%) | 1.12배 |
| 캐시 작음 (22%) | 1.23배 |
| k=5 (26%) | 1.30배 |
| **기본 조건 (41%)** | **1.61배** |
| chunk 16 KB (45%) | 1.71배 |
| k=20 (58%) | 2.21배 |
| 검색 가벼움 (79%) | 3.39배 |

그래프 파일은 다음과 같습니다.

- [`results/core_1110/figs/fig1_qps_p95.png`](results/core_1110/figs/fig1_qps_p95.png): 처리량과 p95
- [`fig3_resources.png`](results/core_1110/figs/fig3_resources.png): CPU와 SSD 자원
- [`fig5_validation.png`](results/core_1110/figs/fig5_validation.png): 병목 유도 대조군
- [`fig6_open_loop.png`](results/core_1110/figs/fig6_open_loop.png): 같은 부하에서의 지연
- [`results/sens_1110/figs/fig7_sensitivity.png`](results/sens_1110/figs/fig7_sensitivity.png): 민감도

## 실험 설계

### 요청 하나가 하는 일

```
요청 생성기 → [작업 큐] → worker: ① 벡터 검색 (IVF, CPU) → ② 문서 chunk k=10개 읽기 (SSD, 4 KB, O_DIRECT)
```

### 비교한 서버 구조 (하드웨어는 같고, 기다리는 방식만 다름)

| 코드 이름 | 발표 이름 | worker | chunk 읽기 |
|---|---|---|---|
| A | 동기 (기준) | 7 (CPU 수) | 하나씩, 끝날 때까지 대기 |
| B | | 요청마다 1개 | 하나씩, 대기 |
| C | | 7 | 10개 동시, 대기 |
| D | 스레드 증설 | 요청마다 1개 | 10개 동시, 대기 |
| S | 비동기 I/O | 7 | Linux native AIO로 제출만 하고 다음 요청 처리. 완료는 별도 스레드가 받음 |

### 병목을 일부러 만든 대조군

| 이름 | 하는 일 |
|---|---|
| E_cpu | 검색을 2.6배 무겁게 해서 CPU 병목을 만듦 |
| F_ssd | 검색 없이 64 KB chunk를 읽어 SSD 경합을 만듦 |

- open loop(포아송 도착)로 같은 절대 부하에서의 p95/p99도 비교했습니다.
- 초당 계측을 끈 버전과 비교해 계측 오버헤드도 쟀습니다(−0.4% 이하).

### 판정 방법

- **주 근거는 대조군 비교입니다.** 같은 회차 안에서 짝지어 처리량 배수를 구하고, 회차 단위로 기하평균과 95% t 신뢰구간을 냅니다. 신뢰구간 하한이 1보다 크면서 1.3배 이상일 때만 "확실히 큼"으로 판정합니다.
- **지점별 병목 라벨은 보조 근거입니다.** 우선순위는 CPU > SSD > 동기식 대기이고, 임계값을 ±20% 바꿔도 라벨이 유지되는지 함께 표시합니다.
- **공정성 장치**
  - 회차마다 시나리오 순서를 무작위로 섞었습니다.
  - 실험 시작과 끝에 SSD 기준 성능을 재서 비교했습니다.
  - 요청 생성기와 계측은 별도 CPU에서 돌렸습니다.
  - 지점마다 다른 사용자의 I/O·CPU 사용을 기록했습니다.

### 검색 비용 보정

검색 CPU 비용은 실제 임베딩(ann-benchmarks NYTimes-256)에서 FAISS HNSW가 recall@10 0.9를 낼 때의 시간에 맞췄습니다. 500k×384 규모로 환산한 값이 1.17 ms이고, 실험 설정은 1.24 ms(`nlist 4096`, `nprobe 32`)입니다.

## 실험 환경

| 항목 | 값 |
|---|---|
| 서버 | Intel Xeon Platinum 8470 ×2, RAM 188 GB, Ubuntu (kernel 6.8), 컨테이너 |
| SSD | Samsung NVMe (MZQL21T9) |
| SSD 성능 (O_DIRECT) | 4 KB 최대 약 34만 IOPS, 64 KB 최대 6.1 GB/s, QD1 지연 약 0.07 ms |
| CPU | 서버 7개 + 요청 생성기·계측 1개를 고정 (같은 NUMA 노드) |
| 반복 | 3회 |

## 폴더 구성

| 경로 | 설명 |
|---|---|
| `*.go` | Go 소스 (외부 라이브러리 없음, 정적 바이너리) |
| `scripts/run_all.sh` | 실험 드라이버 (`quick` / `core` / `sens` / `full`) |
| `scripts/calibrate.sh`, `calibrate_faiss.py`, `calib_choose.py` | 검색 비용 보정 (FAISS) |
| `scripts/analyze.py` | 진단표, 신뢰구간 비교표, 그래프 생성 (pandas, matplotlib) |
| `scripts/cleanup.sh` | 실험이 만든 파일만 지워 서버를 실험 전 상태로 되돌림 |
| `scripts/llm_probe.*` | (선택) GPU로 LLM 생성 시간 측정. 실행 검증은 하지 않음 |
| `scripts/package.sh` | 서버에 올릴 `dist/m3bench.zip` 생성 |
| `results/core_1110/` | 본 실험 결과 (`summary.csv`, `timeseries.csv`, `diagnosis.md`, `figs/`) |
| `results/sens_1110/` | 민감도 실험 결과 |

## 실행 방법

**요구 사항**

- 빌드: Go 1.24 이상
- 서버: Linux x86-64, O_DIRECT를 지원하는 SSD
- 분석: Python 3, pandas, matplotlib
- sudo는 필요 없습니다.

```bash
# 1. 빌드와 패키징 (내 PC 또는 빌드 머신)
./scripts/package.sh                       # → dist/m3bench.zip

# 2. 서버에서 압축 풀기
python3 -m zipfile -e m3bench.zip . && cd m3bench && chmod +x m3bench *.sh *.py

# 3. 데이터 만들기와 자가 점검
./m3bench prep -file ~/m3data/data.bin -size 16     # 16 GB 데이터 파일
./m3bench selftest -file ~/m3data/data.bin          # FAIL이 있으면 run_all.sh가 시작하지 않음

# 4. (권장) 검색 비용 보정, 10~15분, 설치물은 ~/m3venv에만
./calibrate.sh

# 5. 실험
./run_all.sh quick                                  # 약 10분
M3_DEV=sda nohup ./run_all.sh core > core.log 2>&1 &   # 약 2시간 (M3_DEV: 측정할 장치 이름)
M3_DEV=sda nohup ./run_all.sh sens > sens.log 2>&1 &   # 약 2시간

# 6. 분석 (PC에서 해도 됨)
python3 analyze.py results_core_MMDD_HHMM

# 7. 정리
./cleanup.sh          # 데이터, 설치물, 임시 파일 삭제 (코드와 결과는 남김)
./cleanup.sh --all    # 결과를 옮긴 뒤 코드까지 삭제
```

## 공용 서버에서 쓰기

- 모든 파일은 홈 폴더 안에만 만듭니다: `~/m3bench`, `~/m3data`, `~/m3venv`.
- 다운로드 캐시도 `~/m3venv` 안에만 쌓이게 했습니다.
- sudo, 시스템 설정 변경, 다른 사용자 파일 접근은 하지 않습니다.
- 실행 중에는 CPU 8개, 메모리 1~2 GB, SSD 읽기 부하를 씁니다. 다른 사람과 겹치지 않는 시간에 돌리세요.
- 끝나면 `cleanup.sh`로 실험 전 상태로 되돌립니다.

## 한계

- 벡터는 합성 데이터입니다. 검색 CPU 비용만 실제 임베딩으로 보정했고, 검색 정확도는 측정 대상이 아닙니다.
- 서버 1대, SSD 1개 조건입니다. 절대 수치보다 비교 배수와 병목 패턴이 결론입니다.
- 비동기 서버는 Linux native AIO로 만든 프로토타입입니다(io_uring 아님).
- 네트워크 계층과 LLM 생성 단계는 포함하지 않았습니다.
- 동시 요청 1개 지점은 회차 간 변동이 커서(약 15%) 해석에서 제외했습니다.
- Milvus·Qdrant 같은 실제 VDB에서의 검증은 다음 단계입니다. Milvus의 디스크 인덱스 엔진인 DiskANN을 후보로 보고 있습니다.
