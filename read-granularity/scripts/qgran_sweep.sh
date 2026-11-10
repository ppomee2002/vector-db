#!/usr/bin/env bash
# Read-granularity sweep on patched Qdrant (Gridstore payload on io_uring, readahead off).
#   ./qgran_sweep.sh A <out>   # cache off: payload page cache evicted every 20 ms, natural trace, C x G, REPS reps
#   ./qgran_sweep.sh B <out>   # cache on: page-cache budget = BUDGETS share of payload bytes, zipf, C=32, 1 rep
# Per case: <case>.json from qdrant_bench (QPS, p50/p99, cgroup storage bytes/ios) and
# <case>.gran.json with the app-level read counters (before/after) of the patched Gridstore.
set -euo pipefail
cd ~/rag-milvus
PY=.venv/bin/python
CTL=~/gran/qgran_ctl.sh
EXP=${1:?A|B}; OUT=${2:?out}; mkdir -p "$OUT"
DSS=${DSS:-"nq fiqa"}
GRANS=${GRANS:-"0 4096 16384 65536 262144 1048576 2097152"}
REQ=${REQ:-2000}; WARM=${WARM:-300}

drop_payload_cache() {
  sudo -n python3 - "$HOME/gran/qvol/collections" <<'EOF'
import os, sys
for root, dirs, files in os.walk(sys.argv[1]):
    if "payload" not in root:
        continue
    for f in files:
        try:
            fd = os.open(os.path.join(root, f), os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
        except OSError:
            pass
EOF
}
cg_current() { cat "/sys/fs/cgroup$(sudo -n cat /proc/"$(sudo -n k3s ctr task ls | awk '$1=="rm-qgran"{print $2}')"/cgroup | cut -d: -f3)/memory.current"; }
gstat() { sleep 1.2; cat ~/gran/stats/gran.json 2>/dev/null || echo '{}'; }

# memory baseline (vector-only searches, no payload resident) and payload size per dataset
declare -A M0 PAY
$CTL start 0 >/dev/null
for DS in $DSS; do
  PAY[$DS]=$(sudo -n find ~/gran/qvol/collections/$DS -type d -name '*payload_storage*' -prune -exec du -sb {} + | awk '{s+=$1} END{print s}')
done
drop_payload_cache
for DS in $DSS; do $PY qgran_bench.py --dataset "$DS" --out "$OUT/probe" --case "probe-$DS" --trace zipf --concurrency 8 --requests 500 --text 0 >/dev/null; done
M=$(cg_current); for DS in $DSS; do M0[$DS]=$M; done
for DS in $DSS; do echo "{\"dataset\":\"$DS\",\"payload_disk_bytes\":${PAY[$DS]},\"m0_bytes\":${M0[$DS]}}"; done | tee "$OUT/sizes.jsonl"

one() {  # one <case> <ds> <gran> <mem> <trace> <conc> [URING]
  local case=$1 ds=$2 g=$3 mem=$4 tr=$5 cc=$6
  [ -f "$OUT/$case.json" ] && { echo "skip $case"; return; }
  sudo -n rm -f ~/gran/stats/gran.json
  URING=${7:-auto} $CTL start "$g" "$mem" >/dev/null
  drop_payload_cache
  local before ev=""; before='{"reads":0,"value_bytes":0,"io_bytes":0}'
  if [ "${NOCACHE:-0}" = 1 ]; then sudo -n PAGES_ONLY=${PAGES_ONLY:-0} python3 ~/gran/evictor.py ~/gran/qvol/collections/$ds 20 & ev=$!; fi
  # optional: emulate an IOPS/bandwidth-capped cloud disk shared by Qdrant and a co-located tenant
  local cg="" nb=""
  cg=/sys/fs/cgroup$(sudo -n cat /proc/"$(sudo -n k3s ctr task ls | awk '$1=="rm-qgran"{print $2}')"/cgroup | cut -d: -f3)
  if [ -n "${IOMAX:-}" ]; then echo "$(lsblk -no MAJ:MIN "$(findmnt -no SOURCE --target ~/gran/qvol | sed 's/p[0-9]*$//')" | head -1 | tr -d ' ') $IOMAX" | sudo -n tee "$cg/io.max" >/dev/null; fi
  if [ "${NEIGH:-0}" != 0 ]; then sudo -n python3 ~/gran/neighbor.py ~/gran/neigh.dat "$NEIGH" 8 "$OUT/$case.neigh.json" "$cg" & nb=$!; fi
  $PY qgran_bench.py --dataset "$ds" --out "$OUT" --case "$case" --trace "$tr" --concurrency "$cc" \
      --requests "$REQ" --warmup "$WARM" --text 1
  [ -n "$ev" ] && { sudo -n pkill -f "[g]ran/evictor.py" || true; wait "$ev" 2>/dev/null || true; }
  if [ -n "$nb" ]; then
    sudo -n pkill -TERM -f "[g]ran/neighbor.py" || true
    for _ in $(seq 20); do pgrep -f "[g]ran/neighbor.py" >/dev/null || break; sleep 0.5; done
    sudo -n pkill -KILL -f "[g]ran/neighbor.py" || true; wait "$nb" 2>/dev/null || true
  fi
  echo "{\"before\":$before,\"after\":$(gstat)}" > "$OUT/$case.gran.json"
}

if [ "$EXP" = A ]; then
  CONCS=${CONCS:-"1 32 128"}; REPS=${REPS:-2}; export NOCACHE=1
  for r in $(seq 1 "$REPS"); do
    for DS in $DSS; do
      LIM=$($PY -c "print(int(${M0[$DS]}*1.1))")
      for CC in $CONCS; do
        # stock Qdrant read path: Gridstore on mmap (MADV_RANDOM), no patch effect
        one "$DS-stockmmap-c$CC-r$r" "$DS" 0 "$LIM" natural "$CC" disabled
        for G in $(shuf -e $GRANS); do one "$DS-g$G-c$CC-r$r" "$DS" "$G" "$LIM" natural "$CC"; done
      done
    done
  done
elif [ "$EXP" = C ]; then
  # small-read side: shared capped disk (gp3-like 3000 IOPS, 125 MiB/s) + a co-located 4K tenant
  export NOCACHE=1 PAGES_ONLY=1 IOMAX=${IOMAX:-"riops=3000 rbps=131072000"}; CC=${CC:-32}
  for DS in $DSS; do
    LIM=$($PY -c "print(int(${M0[$DS]}*1.1))")
    for NB in ${NEIGHS:-0 1000}; do
      for G in $(shuf -e $GRANS); do NEIGH=$NB one "$DS-g$G-n$NB-c$CC" "$DS" "$G" "$LIM" natural "$CC"; done
    done
  done
else
  BUDGETS=${BUDGETS:-"0.05 0.2"}; CC=${CC:-32}
  for DS in $DSS; do
    for B in $BUDGETS; do
      LIM=$($PY -c "print(int(${M0[$DS]}*${SLACK:-1.1} + $B*${PAY[$DS]}))")
      one "$DS-stockmmap-b$B-c$CC" "$DS" 0 "$LIM" zipf "$CC" disabled
      for G in $(shuf -e $GRANS); do one "$DS-g$G-b$B-c$CC" "$DS" "$G" "$LIM" zipf "$CC"; done
    done
  done
fi
$CTL stop
echo done
