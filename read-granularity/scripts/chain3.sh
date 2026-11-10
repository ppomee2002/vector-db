#!/usr/bin/env bash
set -uo pipefail
log() { echo "[$(date -u +%m-%dT%H:%M:%S)] $*"; }
cd ~/gran
rm -rf ~/gran/res-M
for DS in fiqa nq; do
  log "milvus sweep $DS"
  ~/gran/milvus_gran_sweep.sh "$DS" ~/gran/res-M > "msweep-$DS.log" 2>&1 || log "milvus sweep $DS failed"
done
log CHAIN_DONE
