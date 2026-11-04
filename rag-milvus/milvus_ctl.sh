#!/usr/bin/env bash
# Start/stop the Milvus v2.6.23 standalone stack with k3s containerd (no docker).
#   milvus_ctl.sh deps                      # etcd + SeaweedFS S3 (idempotent)
#   milvus_ctl.sh start <A|B|C> [mem_bytes]  # (re)start Milvus with a cache config
#   milvus_ctl.sh stop
# Configs (env → milvus.yaml key, formatKey lowercases and strips _ . /):
#   A  default: scalar (text) fields fully loaded in memory
#   B  mmap:    queryNode.mmap.scalarField=true  → text served via OS page cache
#   C  tiered:  scalar warmup disabled + eviction → text cells fetched from S3 on demand
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
V=$ROOT/volumes
C="sudo -n k3s ctr"
IMG=docker.io/milvusdb/milvus:v2.6.23

kill_task() {
  if $C task ls -q 2>/dev/null | grep -qx "$1"; then
    $C task kill -s SIGTERM "$1" || true
    for _ in $(seq 30); do $C task ls -q | grep -qx "$1" || break; sleep 1
      [ "$($C task ls | awk -v n="$1" '$1==n{print $3}')" = STOPPED ] && break; done
    $C task kill -s SIGKILL "$1" 2>/dev/null || true
    $C task delete -f "$1" 2>/dev/null || true
  fi
  $C container delete "$1" 2>/dev/null || true
}

deps() {
  mkdir -p "$V/etcd" "$V/seaweed" "$V/milvus"
  $C task ls | awk '$1=="rm-etcd"&&$3=="RUNNING"{f=1} END{exit !f}' || { $C task delete -f rm-etcd 2>/dev/null || true; $C container delete rm-etcd 2>/dev/null || true
    $C run -d --net-host --mount "type=bind,src=$V/etcd,dst=/etcd,options=rbind:rw" \
      --env ETCD_AUTO_COMPACTION_MODE=revision --env ETCD_AUTO_COMPACTION_RETENTION=1000 \
      --env ETCD_QUOTA_BACKEND_BYTES=4294967296 quay.io/coreos/etcd:v3.5.5 rm-etcd \
      etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls=http://127.0.0.1:2379 \
      -listen-peer-urls=http://127.0.0.1:2380 --data-dir /etcd; }
  $C task ls | awk '$1=="rm-s3"&&$3=="RUNNING"{f=1} END{exit !f}' || { $C task delete -f rm-s3 2>/dev/null || true; $C container delete rm-s3 2>/dev/null || true
    $C run -d --net-host --rlimit-nofile 1048576 --mount "type=bind,src=$V/seaweed,dst=/data,options=rbind:rw" \
      docker.io/chrislusf/seaweedfs:latest rm-s3 weed server -dir=/data -ip=127.0.0.1 \
      -ip.bind=127.0.0.1 -s3 -s3.port=8333 -master.volumeSizeLimitMB=1024 -volume.max=0; }
}

start() {
  local cfg=$1 mem=${2:-0}
  deps
  kill_task rm-milvus
  local env=(--env ETCD_ENDPOINTS=127.0.0.1:2379 --env MINIO_ADDRESS=127.0.0.1:8333
             --env DATACOORD_ENABLECOMPACTION=false --env QUERYCOORD_AUTOBALANCE=false)
  case $cfg in
    A) ;;
    B) env+=(--env QUERYNODE_MMAP_SCALARFIELD=true --env QUERYNODE_MMAP_SCALARINDEX=true) ;;
    C) env+=(--env QUERYNODE_SEGCORE_TIEREDSTORAGE_WARMUP_SCALARFIELD=disable
             --env QUERYNODE_SEGCORE_TIEREDSTORAGE_EVICTIONENABLED=true
             --env QUERYNODE_SEGCORE_TIEREDSTORAGE_BACKGROUNDEVICTIONENABLED=true) ;;
    *) echo "unknown config $cfg" >&2; exit 2 ;;
  esac
  # MILVUS_STORAGE=local: keep segment files on local disk instead of the S3 endpoint
  [ "${MILVUS_STORAGE:-remote}" = local ] && env+=(--env COMMON_STORAGETYPE=local --env LOCALSTORAGE_PATH=/var/lib/milvus/data)
  # Extra overrides, e.g. EXTRA_ENV="QUERYNODE_SEGCORE_TIEREDSTORAGE_MEMORYHIGHWATERMARKRATIO=0.05 ..."
  for kv in ${EXTRA_ENV:-}; do env+=(--env "$kv"); done
  local lim=(); [ "$mem" != 0 ] && lim=(--memory-limit "$mem")
  $C run -d --net-host "${lim[@]}" --log-uri "file://$V/milvus-$cfg.log" \
    --mount "type=bind,src=$V/milvus,dst=/var/lib/milvus,options=rbind:rw" \
    "${env[@]}" "$IMG" rm-milvus milvus run standalone
  for _ in $(seq 120); do curl -sf http://127.0.0.1:9091/healthz 2>/dev/null | grep -q OK && { echo "milvus $cfg up (mem=$mem)"; return; }; sleep 2; done
  echo "milvus did not become healthy" >&2; exit 1
}

case ${1:-} in
  deps) deps ;;
  start) start "${2:?config}" "${3:-0}" ;;
  stop) kill_task rm-milvus ;;
  *) sed -n 2,10p "$0"; exit 2 ;;
esac
