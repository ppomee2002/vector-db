# RAG Context Read on Milvus (in progress)

Goal: check whether a real VectorDB (Milvus v2.6.23 standalone) shows the RAG
context-read problems measured in the synthetic harness (`../context-read-bench`):
(1) duplicate context reads, (2) gap between Milvus' cache and Belady at equal
capacity, (3) share of query latency spent on context reads, (4) read amplification.

Status: data download + embedding stage. Milvus deployment and benchmark scripts
are not written yet.

- `download.sh`: BEIR scifact/fiqa/nq/hotpotqa (Hugging Face parquet mirror) + QASPER.
- `embed.py`: CPU embedding with all cores (default all-MiniLM-L6-v2 int8 ONNX, max_len 128).
  Models go under `models/`, outputs under `datasets/emb/<dataset>/`.
