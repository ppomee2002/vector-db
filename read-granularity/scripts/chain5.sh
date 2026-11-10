#!/usr/bin/env bash
set -uo pipefail
log() { echo "[$(date -u +%m-%dT%H:%M:%S)] $*"; }
cd ~/gran
mv res-M res-M-noS3 2>/dev/null
log "milvus sweep nq"
~/gran/milvus_gran_sweep.sh nq ~/gran/res-M > msweep-nq.log 2>&1 || log "milvus sweep nq failed"
[ -f neigh.dat ] || { log "make neighbor file"; dd if=/dev/urandom of=neigh.dat bs=4M count=1024 status=none; sync; }
log "qdrant C"
DSS=nq REQ=1500 WARM=200 ./qgran_sweep.sh C ~/gran/res-C > sweepC.log 2>&1 || log "qdrant C failed"
log CHAIN_DONE
