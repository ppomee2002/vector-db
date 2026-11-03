# Concurrent Context Read 실험

새로 구현한 Linux C++ FAISS retrieval 벤치마크다. Python은 데이터 준비와 프로세스 실행만 담당한다. 측정 구간의 ANN, 페이지 읽기, Chunk 복원, 동시 요청 병합은 모두 C++이다.

검증할 주장은 세 가지로 분리한다: 동시 요청의 읽기 중복, 그 중복의 병목 기여, 병합의 QPS/Tail 개선. 이 저장소의 합성 실험은 구현과 측정 경로를 검증한다. 실제 VectorDB 제품에 관한 일반적 주장을 입증하는 실험은 아직 아니다.

## 동일 실행 경로

|mode|완료 데이터 캐시|in-flight 병합|
|---|---|---|
|none|없음|없음|
|shared|없음|있음|
|lru|LRU|없음|
|lru_shared|LRU|있음|

모두 같은 HNSW 인덱스, top-k, efSearch, Query 순서, 4KiB pread, Chunk memcpy/내용 체크섬을 사용한다. 검색은 Query당 한 스레드이며 OpenMP/BLAS 내부 스레드는 1이다. 기본 top-k=10, efSearch=32. 인덱스 생성은 측정 전에 완료한다.

Reader는 64개 shard를 사용한다. 캐시는 shard별 LRU이며 전체 페이지 용량을 균등 분할한다. 기본 4096 pages=16MiB payload이며 map/list 메모리는 별도다. 엄밀한 전역 LRU와는 다르다. 페이지 키는 프로세스가 여는 단일 불변 Context 파일의 페이지 번호다. Shared Read는 완료 직후 pending 항목을 삭제한다. TTL이나 완료 결과 재사용은 캐시 모드에서만 한다.

모든 구성에서 pending registry를 관측하므로 관측 비용도 baseline에 포함된다. `overlap_requests`는 registry에 다른 읽기가 남아 있을 때 도착한 요청이며 커널 I/O 실행 구간만을 뜻하지 않는다. `joined_reads`는 실제 병합 수다. 각 실행에서 `logical_pages = cache_hits + joined_reads + pread_calls`를 검증하고 구성/동시성 간 Query별 결과 체크섬을 비교한다.

## 빌드

FAISS 소스는 수정하지 않고 외부 source tree로 빌드한다. 이 서버의 소스 revision은 `ef28c6beb081d265cb6bdace8a3e85606bfd8a56`이다. 빌드 도구 CMake 3.31.6은 이 폴더의 tools에 설치했다. 다른 서버에서는 CMake >=3.24, C++20 컴파일러, OpenMP, BLAS/LAPACK 및 FAISS 소스가 필요하다.

```bash
tools/cmake/data/bin/cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DFAISS_SOURCE=/home/dev_path/ceph/faiss
tools/cmake/data/bin/cmake --build build --target context_bench -j 8
```

FAISS CPU 인덱스의 변경 없는 동시 검색 및 singleton Query 실행은 [FAISS threading 문서](https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls)를 따른다.

## 입력 형식과 실제 데이터 연결

- `base.fvecs`: Chunk당 임베딩 한 행. 행 번호가 ANN ID다.
- `queries.fvecs`: Query 임베딩. base와 같은 임베딩 모델/차원/정규화 규칙이 필요하다. 현재 인덱스는 L2다.
- `context.bin`: 실제 Chunk 내용이 들어간 파일. 파일 끝을 4096 bytes로 padding한다. SSD 실험 파일은 전체를 실제로 써서 sparse/unwritten extent를 없앤다.
- `chunks.tsv`: ANN ID 순으로 `offset length` 두 정수. 길이가 가변적이거나 페이지를 걸치는 Chunk도 지원한다. 길이는 1 이상이다.
- `<workload>.trace`: 각 줄에 queries.fvecs의 0-based 행 번호. 구성 간 같은 파일을 사용한다.
- `index.faiss`: 아래 명령으로 한 번만 생성한 HNSW 인덱스.

```bash
OPENBLAS_NUM_THREADS=1 build/context_bench build --base data/real/base.fvecs --index data/real/index.faiss --threads 8
```

Unique trace는 Query ID를 반복하지 않는다. 이것만으로 Context 중복이 없다고 가정하지 않고 실제 페이지 요청을 측정한다. Zipf trace는 고정 Query bank에 대한 반복이다. 합성 fixture 기본값은 alpha=1.2, bank=4096이다.

QASPER는 실제 corpus에서 동일 임베딩 모델로 Chunk/질문을 임베딩하고 실제 질문의 paper_id를 유지해야 한다. JSONL `{"query_id":0,"paper_id":"..."}` 메타데이터로 아래 trace를 만들 수 있다. 이 스크립트는 동일 논문의 서로 다른 질문을 연속 배치하며 각 Query ID를 한 번만 사용한다. 관측된 서비스 trace가 아니라 **논문별 locality를 설정한 실험 시나리오**다. ANN은 전체 corpus를 검색하며 같은 논문의 결과를 강제로 반환하지 않는다. 실제 반환 Chunk의 paper_id 중복률 분석은 별도로 필요하다.

```bash
python3 scripts/same_paper_trace.py --metadata data/real/query_papers.jsonl --output data/real/same_paper.trace
```

## 실행 순서

합성 fixture는 131072 vectors, dimension 64, 1GiB 파일, Chunk 6144 bytes, 슬롯 8192 bytes다. 실제 텍스트/문서 구조를 모사하지 않는다. 파일은 새 경로에만 생성하고 기존 파일을 덮어쓰지 않는다.

```bash
OPENBLAS_NUM_THREADS=1 python3 scripts/prepare_synthetic.py --out data/synthetic
OPENBLAS_NUM_THREADS=1 build/context_bench build --base data/synthetic/base.fvecs --index data/synthetic/index.faiss --threads 8
python3 scripts/fio_probe.py --file data/synthetic/context.bin --out results/fio-full --seconds 30 --repeats 5
python3 scripts/run_matrix.py --data data/synthetic --out results/direct-full --repeats 5 --device nvme0n1
python3 scripts/run_matrix.py --data data/synthetic --out results/buffered-full --io buffered --buffered-state warm --repeats 5 --device nvme0n1
python3 scripts/summarize.py --matrix results/direct-full --fio results/fio-full --output results/direct-full/report.md
```

`--device`는 해당 파일의 실제 backing device를 확인한 경우에만 지정한다. `/root`는 Docker overlay 경로이며 이 서버에서는 nvme0n1p2가 root backing 장치로 추정된다. 독립 마운트에서의 확인 측정이 필요하다. 디스크 카운터는 호스트 전체 값이다. `fio`와 retrieval 및 다른 무거운 작업은 동시에 실행하지 않는다. fio는 기존 파일을 `--readonly --allow_file_create=0`으로 읽고 raw device에 쓰지 않는다.

fio는 두 엔진을 구분한다. `psync, numjobs=C, iodepth=1`은 Query worker의 동기 pread에 가까운 비교다. `libaio, numjobs=1, iodepth=C`는 비동기 I/O 제출의 처리 능력을 본다. 동기 엔진의 iodepth만 올려서는 outstanding I/O가 늘어나지 않는다. 옵션 의미는 [fio 공식 문서](https://fio.readthedocs.io/en/latest/fio_doc.html)를 참고한다.

runner는 동시성 1/4/16/32/64/128과 4개 구성을 반복마다 무작위 순서로 실행한다. 모든 구성에 같은 고정 trace를 끝까지 적용하며 시작 1024 Query는 순차 warmup으로 제외한다. 실행 중 구성별로 다른 길이의 trace를 소비하지 않는다. warmup 후에도 LRU 상태는 유지한다. main 측정은 동시 worker가 다음 Query를 가져오는 closed-loop다. 처리 완료 순서는 스케줄링에 따라 달라진다. 파일 해시, 실행 순서, cgroup 제한, binary 해시와 Query별 latency를 보존한다.

buffered/warm은 각 실행 전에 전체 Context 파일을 읽는다. RAM에 안 들어가는 파일에는 완전한 warm 상태를 보장하지 못한다. buffered/drop-file은 대상 파일에만 fadvise(DONTNEED)를 요청하는 best-effort 조건이며 전역 drop_caches를 사용하지 않는다. 그 뒤 동일 prefix warmup이 실행된다. O_DIRECT는 Linux Page Cache 우회이며 SSD 내부 캐시 우회가 아니다.

## 지표와 판단

- `qps`, P50/P95/P99: ANN부터 Chunk 복원까지. 서비스 queue 대기는 포함하지 않는다. production Tail 주장은 도착시간이 있는 open-loop trace로 추가 확인한다.
- `ann_mean_us`, `context_mean_us`, `pread_mean_us`: 병목 구분용 wall time이다. context에는 대기, 복원, 잠금이 포함된다.
- `logical_pages`, `pread_calls`, `cache_hits`, `joined_reads`: 요청당 계산과 병합/캐시 기여를 구분한다. pread가 곧 SSD의 물리 I/O라는 가정은 하지 않는다.
- `process_read_bytes`: Linux가 이 프로세스에 귀속한 storage read bytes. device counters와 함께 확인한다.
- `device_read_iops`, `device_read_bytes`, `device_read_await_ms`: 선택한 장치의 호스트 전체 카운터. 타 작업의 오염 가능성이 있다.
- `cpu_cores`: process CPU seconds / wall seconds. 1.0은 CPU 한 코어 분량이다. 호스트 전체 CPU %가 아니다.

QPS가 증가하면 I/O/query가 줄어도 IOPS/s는 유지되거나 증가할 수 있다. 따라서 동일 trace의 총 읽기 및 읽기/query 감소를 우선 확인하고 IOPS/s도 함께 제시한다. QPS 정체 + 높은 동시성만으로 SSD 병목을 확정하지 않는다. 동일 파일/접근 분포의 fio 곡선, storage read bytes, CPU/ANN 소요, device await, 동시성 증가에 따른 변화를 함께 확인한다.

실제 논문 주장에는 충분한 측정 길이(예: 각 실행 30초 이상), 5회 이상의 독립 반복, seed와 trace의 별도 반복, paired 효과 크기 및 신뢰구간, cache budget 변화가 필요하다. 특히 Unique trace를 단순히 순환시켜 시간을 늘리면 Unique 조건이 깨진다. Query corpus/고유 trace 길이를 늘려야 한다. 이 pilot의 짧은 실행은 통계적 증명으로 사용하지 않는다.

LRU 대비 LRU+Shared 추가 이득이 없으면 그 사실을 그대로 보고한다. SSD 병목이 관측되지 않으면 CPU/소프트웨어 병목 또는 충분한 SSD 여유를 결론에 반영한다. 인위적 지연이나 IOPS 제한은 기본 실험에 넣지 않는다. 네트워크 스토리지/제한 환경은 명시적인 별도 적용 대상 실험으로만 추가한다.
