#!/usr/bin/env bash
# Restart SeaweedFS (data kept) if its S3 port stops answering for 3 checks in a row.
cd "$(dirname "$0")"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
bad=0
while ! grep -q CHAIN_DONE results/chain2.log 2>/dev/null; do
  if sudo -n k3s ctr task ls | grep -q "^rm-s3 .*RUNNING"; then
    if timeout 10 curl -s -o /dev/null http://127.0.0.1:8333/; then bad=0; else bad=$((bad+1)); log "s3 no answer ($bad)"; fi
    if [ $bad -ge 3 ]; then
      log "restarting rm-s3"
      sudo -n k3s ctr task kill -s SIGKILL rm-s3; sleep 3; sudo -n k3s ctr task delete -f rm-s3
      ./milvus_ctl.sh deps; bad=0
    fi
  fi
  sleep 60
done
log "chain done, watchdog exit"
