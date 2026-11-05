#!/usr/bin/env bash
# Extra runs: Milvus mmap (B) vs tiered (C) on NQ, Milvus C at 5%, Qdrant NQ at 1-5%.
set -uo pipefail
cd "$(dirname "$0")"
export MILVUS_STORAGE=local
PY=.venv/bin/python
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
log "Milvus B (mmap): NQ trend"
BUDGETS="0.5 0.2 0.05" ./milvus_b_trend.sh nq results/trend-nq-mmap
$PY analyze.py results/trend-nq-mmap > /dev/null 2>&1 && log "nq milvus B TREND.md written"
log "Milvus C: NQ 5%"
BUDGETS="0.05" CONCS="32 128" TRACES="natural zipf" REQ=2000 WARM=300 ./run_trend.sh nq results/trend-nq-low
$PY analyze.py results/trend-nq-low > /dev/null 2>&1 && log "nq milvus C low TREND.md written"
./milvus_ctl.sh stop
log "Qdrant: NQ 1-5%"
BUDGETS="0.05 0.02 0.01" CONCS="32 128" TRACES="natural zipf" REQ=2000 WARM=300 ./qdrant_trend.sh nq results/qdrant-nq-low
$PY analyze.py results/qdrant-nq-low > /dev/null 2>&1 && log "nq qdrant low TREND.md written"
./qdrant_ctl.sh stop
log CHAIN_DONE
