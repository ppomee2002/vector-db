#!/usr/bin/env bash
# Small-read side under per-I/O latency: Qdrant storage on dm-delay volume (~/gran/qvol-d), payload cache off.
#   qgran_delay.sh <out>    env: CONDS="name:delay_ms:iomax(0|gp3):conc:reps ..."
set -euo pipefail
cd ~/rag-milvus
PY=.venv/bin/python; CTL=~/gran/qgran_ctl.sh; DD=~/gran/delay_dev.sh
OUT=${1:?out}; mkdir -p "$OUT"
export QVOL=$HOME/gran/qvol-d
GRANS=${GRANS:-"0 16384 65536 262144 1048576"}
REQ=${REQ:-1500}; WARM=${WARM:-200}
LIM=$(python3 -c "print(int(5181444096*1.1))")   # same memory limit as res-C (M0 * 1.1)
GP3="riops=3000 rbps=131072000"
CONDS=${CONDS:-"d0cap:0:gp3:32:1 d1cap:1:gp3:32:2 d1cap8:1:gp3:8:2 d1nocap:1:0:32:2"}
MAJMIN=$(lsblk -no MAJ:MIN /dev/mapper/gran-delay | tr -d ' ')
cur_delay=-1
one() {  # one <case> <gran> <iomax> <conc>
  local case=$1 g=$2 iom=$3 cc=$4
  [ -f "$OUT/$case.json" ] && { echo "skip $case"; return; }
  sudo -n rm -f ~/gran/stats/gran.json
  $CTL start "$g" "$LIM" >/dev/null
  local cg; cg=/sys/fs/cgroup$(sudo -n cat /proc/"$(sudo -n k3s ctr task ls | awk '$1=="rm-qgran"{print $2}')"/cgroup | cut -d: -f3)
  [ "$iom" = gp3 ] && echo "$MAJMIN $GP3" | sudo -n tee "$cg/io.max" >/dev/null
  sudo -n PAGES_ONLY=1 python3 ~/gran/evictor.py "$QVOL/collections/nq" 20 & local ev=$!
  $PY qgran_bench.py --dataset nq --out "$OUT" --case "$case" --trace natural --concurrency "$cc" \
      --requests "$REQ" --warmup "$WARM" --text 1
  sudo -n pkill -f "[g]ran/evictor.py" || true; wait "$ev" 2>/dev/null || true
  sleep 1.2; cat ~/gran/stats/gran.json > "$OUT/$case.gran.json" 2>/dev/null || echo '{}' > "$OUT/$case.gran.json"
}
for c in $CONDS; do
  IFS=: read -r name ms iom cc reps <<<"$c"
  if [ "$ms" != "$cur_delay" ]; then $CTL stop; $DD set "$ms"; cur_delay=$ms; fi
  for r in $(seq 1 "$reps"); do
    for G in $(shuf -e $GRANS); do one "nq-$name-g$G-r$r" "$G" "$iom" "$cc"; done
  done
done
$CTL stop
echo done
