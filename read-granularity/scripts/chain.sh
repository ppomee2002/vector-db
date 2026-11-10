#!/usr/bin/env bash
# Run order on 109 once the Milvus core build and data load finish:
#   1. Qdrant A (cache off) and B (cache) again on a quiet machine, longer runs
#   2. Milvus cell-size sweep on FiQA then NQ
set -uo pipefail
log() { echo "[$(date -u +%m-%dT%H:%M:%S)] $*"; }
cd ~/gran

log "waiting for milvus core build"
until sudo -n grep -aq "BUILD_EXIT=" mbuild.log; do sleep 30; done
sudo -n grep -a "BUILD_EXIT=" mbuild.log | tail -1
LIB=~/gran/milvus/internal/core/output/lib/libmilvus_core.so
[ -f "$LIB" ] || { log "no $LIB, build failed"; exit 1; }
log "waiting for milvus data load"
while pgrep -f "load.py --dataset" >/dev/null; do sleep 30; done
tail -2 mload-fiqa.log mload-nq.log

log "qdrant: stop milvus for a quiet machine"
~/rag-milvus/milvus_gran_ctl.sh stop
log "qdrant A"
REQ=10000 WARM=500 ./qgran_sweep.sh A ~/gran/res-A2 > sweepA2.log 2>&1
log "qdrant B"
REQ=10000 WARM=2000 SLACK=1.02 ./qgran_sweep.sh B ~/gran/res-B2 > sweepB2.log 2>&1

log "milvus sweep"
for DS in fiqa nq; do
  ~/gran/milvus_gran_sweep.sh "$DS" ~/gran/res-M > "msweep-$DS.log" 2>&1 || log "milvus sweep $DS failed"
done
log CHAIN_DONE
