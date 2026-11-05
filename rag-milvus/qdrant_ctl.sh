#!/usr/bin/env bash
# Qdrant (single node) via k3s containerd, host network (6333 REST / 6334 gRPC).
#   qdrant_ctl.sh start [mem_bytes]   # (re)start; mem_bytes caps anon + page cache of the container
#   qdrant_ctl.sh stop
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
V=$ROOT/volumes/qdrant
C="sudo -n k3s ctr"
IMG=docker.io/qdrant/qdrant:latest

stop() {
  if $C task ls -q 2>/dev/null | grep -qx rm-qdrant; then
    $C task kill -s SIGTERM rm-qdrant || true
    for _ in $(seq 30); do [ "$($C task ls | awk '$1=="rm-qdrant"{print $3}')" = STOPPED ] && break; sleep 1; done
    $C task kill -s SIGKILL rm-qdrant 2>/dev/null || true
    $C task delete -f rm-qdrant 2>/dev/null || true
  fi
  $C container delete rm-qdrant 2>/dev/null || true
}

start() {
  local mem=${1:-0}
  mkdir -p "$V"
  stop
  local lim=(); [ "$mem" != 0 ] && lim=(--memory-limit "$mem")
  $C run -d --net-host "${lim[@]}" --log-uri "file://$ROOT/volumes/qdrant.log" \
    --mount "type=bind,src=$V,dst=/qdrant/storage,options=rbind:rw" \
    --env QDRANT__TELEMETRY_DISABLED=true "$IMG" rm-qdrant ./entrypoint.sh
  for _ in $(seq 120); do curl -sf http://127.0.0.1:6333/readyz >/dev/null 2>&1 && { echo "qdrant up (mem=$mem) $(curl -s http://127.0.0.1:6333 | tr -d '\n' | cut -c1-80)"; return; }; sleep 1; done
  echo "qdrant did not become ready" >&2; exit 1
}

case ${1:-} in
  start) start "${2:-0}" ;;
  stop) stop ;;
  *) sed -n 2,4p "$0"; exit 2 ;;
esac
