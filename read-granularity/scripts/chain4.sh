#!/usr/bin/env bash
# After the Milvus sweep: Qdrant on a gp3-like shared disk (3000 IOPS, 125 MiB/s) with/without a 4K co-tenant.
set -uo pipefail
log() { echo "[$(date -u +%m-%dT%H:%M:%S)] $*"; }
cd ~/gran
log "waiting for milvus sweep"
until grep -q CHAIN_DONE chain3.log; do sleep 60; done
[ -f neigh.dat ] || { log "make neighbor file"; dd if=/dev/urandom of=neigh.dat bs=4M count=1024 status=none; sync; }
log "qdrant C"
DSS=nq REQ=1500 WARM=200 ./qgran_sweep.sh C ~/gran/res-C > sweepC.log 2>&1 || log "qdrant C failed"
log CHAIN_DONE
