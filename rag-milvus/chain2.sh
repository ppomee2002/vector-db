#!/usr/bin/env bash
# Resume after the SeaweedFS volume-limit stall: Milvus NQ load+trend, then Qdrant NQ trend.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
log "Milvus: load NQ"
./milvus_ctl.sh start C && $PY load.py --dataset nq --drop --threads 16 2>&1 | tail -n2
log "Milvus: NQ trend"
BUDGETS="1.0 0.5 0.2" CONCS="32 128" TRACES="natural zipf" REQ=2000 WARM=300 ./run_trend.sh nq results/trend-nq
$PY analyze.py results/trend-nq > /dev/null 2>&1 && log "nq milvus TREND.md written"
./milvus_ctl.sh stop
log "Qdrant: NQ trend"
BUDGETS="1.0 0.5 0.2" CONCS="32 128" TRACES="natural zipf" REQ=2000 WARM=300 ./qdrant_trend.sh nq results/qdrant-nq
$PY analyze.py results/qdrant-nq > /dev/null 2>&1 && log "nq qdrant TREND.md written"
./qdrant_ctl.sh stop
log CHAIN_DONE
