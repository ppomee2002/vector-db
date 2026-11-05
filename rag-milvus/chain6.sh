#!/usr/bin/env bash
cd $(dirname $0); export MILVUS_STORAGE=local
echo "[$(date -u +%H:%M:%S)] Milvus B rerun"
BUDGETS="0.5 0.2 0.05" ./milvus_b_trend.sh nq results/trend-nq-mmap && .venv/bin/python analyze.py results/trend-nq-mmap >/dev/null && echo "[$(date -u +%H:%M:%S)] TREND.md written"
./milvus_ctl.sh stop; echo CHAIN_DONE
