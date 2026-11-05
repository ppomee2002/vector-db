#!/usr/bin/env bash
# Run the remaining trend steps back to back (one engine measured at a time).
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "wait for Milvus FiQA low-budget runs"
while pgrep -f "[r]un_trend.sh fiqa" >/dev/null; do sleep 15; done
$PY analyze.py results/trend-fiqa > /dev/null 2>&1 && log "fiqa milvus TREND.md written"

log "wait for Qdrant NQ load"
while pgrep -f "[q]drant_load.py --dataset nq" >/dev/null; do sleep 15; done
tail -n1 qdrant_load_nq.log
log "Qdrant: FiQA trend"
BUDGETS="1.0 0.5 0.2 0.1" CONCS="1 32 128" TRACES="natural zipf" REQ=2000 WARM=300 ./qdrant_trend.sh fiqa results/qdrant-fiqa
$PY analyze.py results/qdrant-fiqa > /dev/null 2>&1 && log "fiqa qdrant TREND.md written"
./qdrant_ctl.sh stop

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
