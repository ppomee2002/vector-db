#!/usr/bin/env bash
# Rerun the Milvus parts of chain4 (etcd was stopped) after chain4 finishes.
set -uo pipefail
cd "$(dirname "$0")"
export MILVUS_STORAGE=local
PY=.venv/bin/python
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
while pgrep -f "[c]hain4.sh" >/dev/null; do sleep 15; done
rm -rf results/trend-nq-mmap results/trend-nq-low
log "Milvus B (mmap): NQ trend"
BUDGETS="0.5 0.2 0.05" ./milvus_b_trend.sh nq results/trend-nq-mmap
$PY analyze.py results/trend-nq-mmap > /dev/null 2>&1 && log "nq milvus B TREND.md written"
log "Milvus C: NQ 5%"
BUDGETS="0.05" CONCS="32 128" TRACES="natural zipf" REQ=2000 WARM=300 ./run_trend.sh nq results/trend-nq-low
$PY analyze.py results/trend-nq-low > /dev/null 2>&1 && log "nq milvus C low TREND.md written"
./milvus_ctl.sh stop
log CHAIN_DONE
