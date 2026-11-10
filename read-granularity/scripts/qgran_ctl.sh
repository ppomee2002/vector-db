#!/usr/bin/env bash
# Patched Qdrant v1.20.0 (Gridstore read granularity) via k3s containerd, host network.
#   qgran_ctl.sh start <gran_bytes> [mem_bytes]   # gran 0 = stock read path
#   qgran_ctl.sh stop
# Env: FADV=1 (default) turns off kernel readahead on payload files (posix_fadvise RANDOM),
#      URING=auto (default) puts the on-disk payload on io_uring; URING=disabled keeps mmap.
set -euo pipefail
G=$HOME/gran
V=${QVOL:-$G/qvol}
C="sudo -n k3s ctr"
IMG=docker.io/library/ubuntu:24.04
NAME=rm-qgran

stop() {
  if $C task ls -q 2>/dev/null | grep -qx $NAME; then
    $C task kill -s SIGTERM $NAME || true
    for _ in $(seq 30); do [ "$($C task ls | awk -v n=$NAME '$1==n{print $3}')" = STOPPED ] && break; sleep 1; done
    $C task kill -s SIGKILL $NAME 2>/dev/null || true
    $C task delete -f $NAME 2>/dev/null || true
  fi
  $C container delete $NAME 2>/dev/null || true
}

start() {
  local gran=${1:?gran} mem=${2:-0}
  stop
  $C images ls -q | grep -qx "$IMG" || $C images pull "$IMG" >/dev/null
  local lim=(); [ "$mem" != 0 ] && lim=(--memory-limit "$mem")
  $C run -d --net-host "${lim[@]}" --log-uri "file://$G/qgran.log" --cwd /qdrant \
    --mount "type=bind,src=$V,dst=/qdrant/storage,options=rbind:rw" \
    --mount "type=bind,src=$G/qdrant/config,dst=/qdrant/config,options=rbind:ro" \
    --mount "type=bind,src=$G/qdrant/target/perf/qdrant,dst=/qdrant/qdrant,options=rbind:ro" \
    --mount "type=bind,src=$G/stats,dst=/stats,options=rbind:rw" \
    --mount "type=bind,src=/etc/ssl/certs,dst=/etc/ssl/certs,options=rbind:ro" \
    --env QDRANT__TELEMETRY_DISABLED=true \
    --env QDRANT__STORAGE__PERFORMANCE__IO_URING="${URING:-auto}" \
    --env QDRANT_GRAN="$gran" \
    --env QDRANT_GRAN_STATS=/stats/gran.json \
    --env QDRANT_PAYLOAD_FADV_RANDOM="${FADV:-1}" \
    --env QDRANT_M2TRACE="${M2TRACE:-}" \
    --env RUST_LOG=info \
    "$IMG" $NAME /qdrant/qdrant
  for _ in $(seq 180); do curl -sf http://127.0.0.1:6333/readyz >/dev/null 2>&1 && { echo "qgran up gran=$gran mem=$mem"; return; }; sleep 1; done
  echo "qgran did not become ready" >&2; tail -20 "$G/qgran.log" >&2; exit 1
}

mkdir -p "$G/stats"
case ${1:-} in
  start) shift; start "$@" ;;
  stop) stop ;;
  *) echo "usage: $0 start <gran> [mem] | stop" >&2; exit 2 ;;
esac
