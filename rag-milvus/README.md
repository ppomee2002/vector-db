# RAG Context Read on Milvus

Checks whether a real VectorDB (Milvus v2.6.23 standalone) shows RAG context-read
inefficiencies: (1) duplicate context reads, (2) gap between Milvus' cache and
Belady at equal capacity, (3) share of query latency spent on context reads,
(4) read amplification.

Work in progress.

- `download.sh`: BEIR scifact/fiqa/nq/hotpotqa (Hugging Face parquet mirror) + QASPER.
- `embed.py`: CPU embedding using all cores (default all-MiniLM-L6-v2 int8 ONNX,
  max_len 128). Models under `models/`, output under `datasets/emb/<dataset>/`.
