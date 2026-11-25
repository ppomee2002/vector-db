# PLAN_v4: 더 강한 Reranker에서 학습 없는 후보 선별(A1, B1)의 가치 (결과를 보기 전에 고정)

- 작성: 2026-11-23, 어떤 v4 결과(속도·품질)도 보기 전.
- 원칙: 판정 기준(§6), 지표 정의(§4), 비교 대상과 하이퍼파라미터(§3)는 결과를 본 뒤 바꾸지 않는다. 바꿔야 하면 맨 아래 "변경 기록"에 남기고 원래 내용은 지우지 않는다. 결과가 부정적이어도 그대로 보고한다. 데이터셋과 Reranker를 더 늘리지 않는다. 새 선별 방법은 만들지 않는다.
- 재사용(읽기 전용): v2 dev Top-100 `sel_fetch_v2/sel_fetch/data/rerank_top100_real.csv`(HNSW ef=256, MiniLM 점수 포함), c2x test Top-100 `sel_fetch_c2x/results/test_rerank_top100.csv`(정확 검색, MiniLM 점수 포함), 질문 `junlim/data_nq/queries.jsonl`, `sel_fetch_c2x/results/test_queries.jsonl`, 정답 `junlim/pf3_out/nq_gold.jsonl`, qrels `sel_fetch_v2/sel_fetch/data/qrels_chunk.tsv`, 본문 `sel_fetch/data/chunk_store_seq.bin`, 임베딩 `junlim/data_nq/chunk_vecs.npy`, v2 코드 `sel_fetch_v2/scripts_v2/common.py`·`s4_offline.py`(선별 함수, 평가 공식을 그대로 가져옴). Milvus는 쓰지 않는다. GPU는 CUDA_VISIBLE_DEVICES=0.

## 1. Reranker 선택 (속도로만 결정, 품질은 보지 않음)
- 후보: `BAAI/bge-reranker-v2-m3`(XLM-R large급, 568M), `BAAI/bge-reranker-base`(XLM-R base, 278M). 둘 다 MS MARCO MiniLM-L-6(22M)보다 크고, 공개 벤치마크에서 더 강한 것으로 알려진 범용 cross-encoder다(문헌 수치는 기억 의존, 미확인).
- 측정: 같은 조건(아래 §5 단가 측정과 같음)에서 질문 30개(+warmup 5개)로 후보 100개 Rerank 시간만 잰다. **정답 품질은 계산하지 않는다.**
- 규칙: bge-reranker-v2-m3의 전체 Rerank 예상 시간(dev+test 425,400쌍)이 **90분 이하**이고 GPU 메모리(15GB)에 들어가면 v2-m3, 아니면 bge-reranker-base. 하나만 쓴다.
- 실행 조건(두 Reranker 공통): max_length 256(MiniLM과 같음), 입력 (질문, chunk 본문), fp16, 질문당 후보 100개를 한 배치. MiniLM 점수는 기존 파일 것(fp32)을 재사용한다.

## 2. 질문 집합
- **dev:** NQ test 중 정답 문자열이 있는 2,254개(v2 Top-100 재사용). 정답 문자열 + qrels(passage 중복 제거 nDCG@10).
- **test:** c2x test 2,000개(NQ train, c2x Top-100 재사용). 정답 문자열만(qrels 없음). Top-100에 정답이 있는 질문만 걸렀으므로 절대값이 dev보다 높다(비교에는 영향 없음).
- M1-noPos는 dev로만 학습한다. dev 지표는 dev 안 5-fold OOF(질문 단위 GroupKFold)로, test에는 dev 전체로 학습한 모델을 적용한다.

## 3. 비교 대상 (같은 Rerank 후보 수 |S| = B, B ∈ {10, 15, 20, 30})
v2 §4.3 공식과 v2에서 고른 값을 그대로 쓴다(`results_v2/s4/chosen_hp.json`). 다시 튜닝하지 않는다.
- **Top-B:** ANN 상위 B개.
- **A1:** 문서 점수 = max(ANN)×log(1+후보 수) (agg=maxlog), α=0.3. Top-(B−x)는 ANN 순서, 나머지 x=round(0.3B)개는 (문서 점수, ANN 점수) 사전순.
- **B1:** β=0.3, γ=0.05. 1라운드 ANN Top-10 Rerank → 남은 후보를 f = 0.7·ann + 0.3·max cos(CE 상위 3) + 0.05·[CE 상위 3과 같은 문서]로 정렬해 10개씩(마지막 라운드는 남은 수) 추가. CE 점수는 **해당 Reranker(새 Reranker 조건에서는 새 Reranker)** 점수를 쓴다. Rerank 호출 횟수 = 1 + ⌈(B−10)/10⌉ (B=10:1, 15·20:2, 30:3). B=10에서는 정의상 Top-B와 같다.
- **M1-noPos (참고):** v2 피처(`common.M1NOPOS_COLS`), HGB(v2와 같은 설정), 라벨 = 새 Reranker의 Top-100 전체 Rerank 상위 10개. dev에서만 학습.
- **Random-deep (대조):** Top-(B−x) + 순위 [B−x, 100)에서 무작위 x=⌈0.2B⌉개, seed 0.
- 같은 비교를 MiniLM 조건(기존 MiniLM 점수, M1-noPos는 MiniLM 라벨로 dev 학습)에서도 계산해 나란히 둔다(포화도·Δ 비교용).

## 4. 지표
- **정답 문자열 포함률 (주):** 최종 Context = S를 해당 Reranker로 정렬한 상위 10개 chunk. 10개 chunk 중 하나라도 정규화된 정답 문자열을 포함하면 1 (v2 `s4_offline.py`의 `ans`와 같은 chunk 단위 정의, 정규화 = NFD·결합문자 제거·소문자·\w+ 토큰).
- **passage 중복 제거 nDCG@10 (dev만):** v2 정의. 정답 passage마다 처음 나온 chunk만 이득 1, IDCG = min(정답 passage 수, 10).
- **Top-B 밖 승격 후보 회수율:** |S ∩ 기준Top10 ∩ {ANN 순위 ≥ B}| / |기준Top10 ∩ {ANN 순위 ≥ B}|, 기준Top10 = 해당 Reranker의 Top-100 전체 Rerank 상위 10 (진단용. 품질 판정에는 쓰지 않는다).
- **통계:** 질문 단위 paired bootstrap 2,000회, seed 0, 95% CI. 비교 기준은 같은 B의 Top-B. 다중 비교는 보정하지 않는다(보고에 명시).
- **비용:** |S|, 선별 계산 시간(CPU, 오프라인 함수, ms/질문), Rerank 호출 횟수, 지연 환산 = Σ_호출 (a + b·n_호출) (a, b는 §5에서 Reranker별로 측정한 선형 적합값).
- **headroom 질문 (B별):** 정답 문자열을 포함한 chunk가 Top-100 안에 있지만 ANN Top-B 안에는 없는 질문. 이 부분집합에서 Top-B의 포함률은 정의상 0이므로 방법의 포함률 자체가 Δ다. dev는 qrels 기준 headroom(정답 chunk가 Top-100 안, Top-B 밖)도 nDCG로 보고.
- A1의 I/O 감소(문서 순 배치에서 요청 −21%, REPORT_a1io)는 후보 집합과 배치로만 정해지고 Reranker와 무관하므로 다시 재지 않는다.

## 5. 후보당 Rerank 단가 (MiniLM과 같은 조건)
- GPU0(A2), fp16, max_length 256, 배치 = 후보 수 n (한 번에), n ∈ {10, 20, 30, 50, 100}, dev에서 seed 1로 고른 질문 100개 + warmup 10개, torch.cuda.synchronize로 경계. MiniLM과 새 Reranker를 같은 질문·같은 순서로 교대 측정. fp32도 참고로 잰다.
- 보고: n별 p50 ms, 선형 적합 a + b·n, 후보 1개당 단가 b, 그리고 B=20에서의 Rerank 지연.
- 측정 전후 nvidia-smi 기록, 다른 GPU 작업이 있으면 기록.

## 6. 판정 기준 (고정)
- **전제 확인:** 새 Reranker Top-100 전체 Rerank의 정답 문자열 포함률이 (a) ANN 순서 Top-10보다, (b) MiniLM 전체 Rerank보다 높은가. "높다" = 차이의 95% CI 하한 > 0, dev와 test 모두에서. dev 중복 제거 nDCG@10도 같이 보고(참고). **(b)를 만족하지 못하면** 그 사실을 보고하고 2단계(포화도·선별 비교)는 참고로만 해석한다.
- **포화:** B=20에서 Top-B 품질 / 전체 Top-100 Rerank 품질(정답 기준)이 **98% 이상이면 "여전히 포화"**. 계산하는 지표: dev 정답 문자열 포함률, dev 중복 제거 nDCG@10, test 정답 문자열 포함률. 세 값이 모두 ≥ 98%이면 "여전히 포화", 하나라도 < 98%이면 "덜 포화"(어느 지표인지 명시).
- **채택:** A1 또는 B1이 같은 |S|(어느 B ∈ {10, 15, 20, 30})에서 Top-B보다
  - dev 정답 문자열 포함률 **+1%p 이상** 또는 dev 중복 제거 nDCG@10 **+0.01 이상**이고,
  - 그 차이의 95% CI가 0을 포함하지 않으며,
  - **test에서도 유지**될 때만 "강한 Reranker 조건에서 학습 없는 선별이 유효". test 유지 = 같은 방법·같은 B에서 test 정답 문자열 포함률 차이의 95% CI 하한 > 0이고, dev에서 포함률 기준(+1%p)으로 통과했다면 test에서도 차이 ≥ +1%p. (test에는 qrels가 없어 nDCG로 통과한 경우는 포함률의 방향·유의성만 확인한다.)
  - 그 밖에는 **"Top-B로 충분"**.
- M1-noPos와 Random-deep은 판정에 쓰지 않는다(참고·대조).

## 7. 산출물
- `scripts/`, `results/`(config.json에 버전·seed·명령), `REPORT_v4.md`. 기존 파일·팀원 폴더는 읽기만 한다.

## 변경 기록
| 시각 (KST) | 변경 | 이유 |
|---|---|---|
