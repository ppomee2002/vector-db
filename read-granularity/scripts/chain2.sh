#!/usr/bin/env bash
# Second run order on 109 (core build done, FiQA already in Milvus):
#   1. Qdrant A (cache off) and B (cache) on a quiet machine (Milvus stopped), longer runs
#   2. Milvus: start, load NQ, check no storage write errors, cell-size sweep on FiQA then NQ
set -uo pipefail
log() { echo "[$(date -u +%m-%dT%H:%M:%S)] $*"; }
cd ~/gran

log "qdrant A"
REQ=10000 WARM=500 ./qgran_sweep.sh A ~/gran/res-A2 > sweepA2.log 2>&1
log "qdrant B"
REQ=10000 WARM=2000 SLACK=1.02 ./qgran_sweep.sh B ~/gran/res-B2 > sweepB2.log 2>&1

log "milvus: load nq"
(cd ~/rag-milvus && ./milvus_gran_ctl.sh start C && .venv/bin/python -u load.py --dataset nq --drop) > mload-nq2.log 2>&1
tail -n 2 mload-nq2.log
ERRS=$(sudo -n grep -ac "Failed to create packed writer" milvus-C.log)
log "packed writer errors in milvus log so far: $ERRS"

log "milvus sweep"
for DS in fiqa nq; do
  ~/gran/milvus_gran_sweep.sh "$DS" ~/gran/res-M > "msweep-$DS.log" 2>&1 || log "milvus sweep $DS failed"
done
log CHAIN_DONE
