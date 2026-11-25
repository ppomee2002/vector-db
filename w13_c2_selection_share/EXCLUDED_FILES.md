# 공유본에서 뺀 파일 목록

원본 폴더(`/workspace/jaeeun/sel_fetch_*`)에는 그대로 남아 있다. 공유본에는 보고서(REPORT\*.md), 계획(PLAN\*.md), 요약 CSV·JSON, 그림(png), 스크립트(py, sh)만 넣었다.

## 사유별 요약

| 사유 | 파일 수 | 용량 |
|---|---|---|
| 지난주 실험 tgz 복제본 (이번 주 결과 아님) | 113 | 264.8MB |
| 5MB 초과 | 6 | 50.0MB |
| 허용 형식 아님 (.pkl) | 8 | 13.8MB |
| *_scores.csv | 2 | 11.8MB |
| npy/npz | 2 | 5.9MB |
| 허용 형식 아님 (.gz) | 4 | 2.8MB |
| 원시 측정값 CSV (요약 아님) | 5 | 1.7MB |
| 허용 형식 아님 (.joblib) | 1 | 1.1MB |
| 허용 형식 아님 (.jsonl) | 1 | 291.8KB |
| 로그 | 17 | 88.5KB |
| 스모크 테스트 (본 측정 아님) | 3 | 28.4KB |
| __pycache__ | 2 | 26.0KB |
| 허용 형식 아님 (.txt) | 5 | 14.2KB |
| 허용 형식 아님 (.md) | 3 | 8.7KB |
| **합계** | **172** | **352.2MB** |

## 01_v2_nq (원본 `sel_fetch_v2/`)

- `sel_fetch/` 전체: 113개, 264.8MB. 지난주 `sel_fetch_backup_final.tgz`를 풀어 둔 복제본이다 (tgz sha256은 원본의 `tgz_sha256_before.txt`). 지난주 분석 보고서는 그 안의 `results/FINAL_ANALYSIS.md`.

| 파일 | 크기 | 사유 |
|---|---|---|
| `PROGRESS.md` | 7.5KB | 허용 형식 아님 (.md) |
| `tgz_sha256_before.txt` | 93B | 허용 형식 아님 (.txt) |
| `results_v2/s1/c_microbench.log` | 1.3KB | 로그 |
| `results_v2/s1/c_microbench_raw.csv` | 87.1KB | 원시 측정값 CSV (요약 아님) |
| `results_v2/s1/s1_analysis.log` | 15.8KB | 로그 |
| `results_v2/s1/s1b.log` | 1.7KB | 로그 |
| `results_v2/s4/m1nopos_dev.joblib` | 1.1MB | 허용 형식 아님 (.joblib) |
| `results_v2/s4/s4.log` | 17.5KB | 로그 |
| `results_v2/s4/test_per_query.csv.gz` | 503.6KB | 허용 형식 아님 (.gz) |
| `results_v2/s4/test_topB_curve_per_query.csv.gz` | 1.6MB | 허용 형식 아님 (.gz) |
| `results_v2/s42/check_busy_before.txt` | 590B | 허용 형식 아님 (.txt) |
| `results_v2/s42/e2e.log` | 1.2KB | 로그 |
| `results_v2/s42/e2e_raw.csv` | 991.7KB | 원시 측정값 CSV (요약 아님) |
| `results_v2/s42/smoke/config.json` | 970B | 스모크 테스트 (본 측정 아님) |
| `results_v2/s42/smoke/e2e_raw.csv` | 26.9KB | 스모크 테스트 (본 측정 아님) |
| `results_v2/s42/smoke/monitor.csv` | 562B | 스모크 테스트 (본 측정 아님) |
| `scripts_v2/__pycache__/common.cpython-312.pyc` | 19.0KB | __pycache__ |

## 02_v3_fiqa (원본 `sel_fetch_v3/`)

| 파일 | 크기 | 사유 |
|---|---|---|
| `PROGRESS.md` | 645B | 허용 형식 아님 (.md) |
| `results_v3/fiqa_chunk_meta.npz` | 2.0MB | npy/npz |
| `results_v3/prep.log` | 591B | 로그 |
| `results_v3/rerank_top100_fiqa.csv` | 5.4MB | 5MB 초과 |
| `results_v3/s3/ann_only_vs_rerank.txt` | 279B | 허용 형식 아님 (.txt) |
| `results_v3/s3/eval.log` | 15.1KB | 로그 |
| `results_v3/s3/test_per_query.csv.gz` | 431.1KB | 허용 형식 아님 (.gz) |

## 03_c2x_selection_c2 (원본 `sel_fetch_c2x/`)

| 파일 | 크기 | 사유 |
|---|---|---|
| `PROGRESS.md` | 610B | 허용 형식 아님 (.md) |
| `results/chunk_tokens_gpt2.npy` | 3.8MB | npy/npz |
| `results/exp12_dev.log` | 190B | 로그 |
| `results/exp12_test.log` | 157B | 로그 |
| `results/exp3.log` | 501B | 로그 |
| `results/prep.log` | 1.8KB | 로그 |
| `results/test_queries.jsonl` | 291.8KB | 허용 형식 아님 (.jsonl) |
| `results/test_rerank_top100.csv` | 8.3MB | 5MB 초과 |
| `results/dev/exp1_ann10.pkl` | 726.2KB | 허용 형식 아님 (.pkl) |
| `results/dev/exp1_c2_hits_ef64.pkl` | 726.2KB | 허용 형식 아님 (.pkl) |
| `results/dev/exp1_curve_R3.pkl` | 987.0KB | 허용 형식 아님 (.pkl) |
| `results/dev/exp1_rerank10_B20.pkl` | 5.8MB | 5MB 초과 |
| `results/dev/exp2.pkl` | 12.8MB | 5MB 초과 |
| `results/exp3/exp3_raw.csv` | 558.7KB | 원시 측정값 CSV (요약 아님) |
| `results/test/analyze.txt` | 13.2KB | 허용 형식 아님 (.txt) |
| `results/test/exp1_ann10.pkl` | 394.2KB | 허용 형식 아님 (.pkl) |
| `results/test/exp1_curve_R3.pkl` | 875.9KB | 허용 형식 아님 (.pkl) |
| `results/test/exp1_rerank10_B20.pkl` | 2.7MB | 허용 형식 아님 (.pkl) |
| `results/test/exp2.pkl` | 10.5MB | 5MB 초과 |

## 04_io_analysis (원본 `sel_fetch_io/`)

| 파일 | 크기 | 사유 |
|---|---|---|
| `results/io_analysis.log` | 11.4KB | 로그 |
| `results/io_per_query.pkl` | 3.4MB | 허용 형식 아님 (.pkl) |

## 05_a1io_measure (원본 `sel_fetch_a1io/`)

| 파일 | 크기 | 사유 |
|---|---|---|
| `results/read_plans_test.pkl` | 4.0MB | 허용 형식 아님 (.pkl) |
| `results/stage0_per_query.pkl` | 7.1MB | 5MB 초과 |
| `results/stage1.log` | 330B | 로그 |

## 06_v4_strong_reranker (원본 `sel_fetch_v4/`)

| 파일 | 크기 | 사유 |
|---|---|---|
| `logs/s1_select.log` | 1.3KB | 로그 |
| `logs/s1_unit.log` | 2.2KB | 로그 |
| `logs/s2.log` | 1.1KB | 로그 |
| `logs/s3.log` | 16.4KB | 로그 |
| `results/plan_sha256_before_results.txt` | 101B | 허용 형식 아님 (.txt) |
| `results/s1_select/raw.csv` | 3.3KB | 원시 측정값 CSV (요약 아님) |
| `results/s1_unit/raw.csv` | 70.7KB | 원시 측정값 CSV (요약 아님) |
| `results/s2/dev_scores.csv` | 6.3MB | *_scores.csv |
| `results/s2/test_scores.csv` | 5.5MB | *_scores.csv |
| `results/s3/per_query.csv.gz` | 307.7KB | 허용 형식 아님 (.gz) |
| `scripts/__pycache__/v4common.cpython-312.pyc` | 7.0KB | __pycache__ |
