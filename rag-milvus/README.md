# RAG Context Read on Milvus

Checks whether a real VectorDB (Milvus v2.6.23 standalone) shows RAG context-read
inefficiencies: (1) duplicate context reads, (2) gap between Milvus' cache and a
chunk-granular LRU/Belady cache at equal capacity, (3) share of query cost spent on
context reads, (4) read amplification.

Runs without docker: containers are started with k3s containerd (`k3s ctr`), host
network, etcd + SeaweedFS (S3) + Milvus.

| step | script |
|---|---|
| download BEIR (HF parquet mirror) + QASPER | `datasets/raw/download.sh` (copy `download.sh` there) |
| CPU embedding, all cores (all-MiniLM-L6-v2 int8 ONNX, max_len 128) | `embed.py --datasets scifact,fiqa,nq` |
| start Milvus with cache config A (in-memory) / B (mmap) / C (tiered storage) | `milvus_ctl.sh start C` |
| create + load collection (pk, doc_id, text, vec HNSW IP) | `load.py --dataset fiqa` |
| closed-loop search load + server counter deltas | `bench.py` |
| text-cache budget sweep with S3 GET capture | `run_trend.sh <dataset> <out>` |
| metrics table (TREND.md) | `analyze.py <out>` |

Config C cache budget is set through
`queryNode.segcore.tieredStorage.memory{High,Low}WatermarkRatio`; eviction needs both
`evictionEnabled` and `backgroundEvictionEnabled`.
