#!/usr/bin/env bash
# Trend run for Qdrant: payload (text) on disk, page-cache budget set by the container memory limit.
#   ./qdrant_trend.sh <dataset> <out_dir>
# Budget B: memory limit = M0 * 1.1 + B * payload_disk_bytes, where M0 is the container's
# memory.current after vector-only searches (no payload pages resident).
set -euo pipefail
cd "$(dirname "$0")"
DS=${1:?dataset}; OUT=${2:?out}; mkdir -p "$OUT"
PY=.venv/bin/python
BUDGETS=${BUDGETS:-"1.0 0.5 0.2 0.1"}
CONCS=${CONCS:-"1 32 128"}
TRACES=${TRACES:-"natural zipf"}
REQ=${REQ:-0}; WARM=${WARM:-1000}
COLL=volumes/qdrant/collections/$DS

drop_payload_cache() {  # evict only this collection's payload files from the page cache
  sudo -n python3 - "$COLL" <<'EOF'
import os, sys
n = 0
for root, dirs, files in os.walk(sys.argv[1]):
    if "payload" not in root:
        continue
    for f in files:
        try:
            fd = os.open(os.path.join(root, f), os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd); n += 1
        except OSError:
            pass
print(f"dropped page cache of {n} payload files", file=sys.stderr)
EOF
}

cg_current() { cat "/sys/fs/cgroup$(sudo -n cat /proc/"$(sudo -n k3s ctr task ls | awk '$1=="rm-qdrant"{print $2}')"/cgroup | cut -d: -f3)/memory.current"; }

./qdrant_ctl.sh start
PAYLOAD=$(sudo -n find "$COLL" -type d -name '*payload*' -prune -exec du -sb {} + | awk '{s+=$1} END{print s}')
drop_payload_cache
$PY qdrant_bench.py --dataset "$DS" --out "$OUT/probe" --case probe --trace zipf --concurrency 8 --requests 500 --text 0 >/dev/null
M0=$(cg_current)
TEXT=$($PY -c "import json;print(json.load(open('datasets/emb/$DS/qdrant_load.json'))['text_bytes'])")
echo "{\"payload_disk_bytes\":$PAYLOAD,\"m0_bytes\":$M0,\"text_bytes\":$TEXT}" | tee "$OUT/sizes.json"

for B in $BUDGETS; do
  LIM=$($PY -c "print(int($M0*1.1 + $B*$PAYLOAD))")
  ./qdrant_ctl.sh start "$LIM"
  for T in $TRACES; do
    for CC in $CONCS; do
      drop_payload_cache
      $PY qdrant_bench.py --dataset "$DS" --out "$OUT" --case "$DS-Q-b$B-$T-c$CC" --trace "$T" --concurrency "$CC" \
        --requests "$REQ" --warmup "$WARM" --text 1
    done
  done
  drop_payload_cache
  $PY qdrant_bench.py --dataset "$DS" --out "$OUT" --case "$DS-Q-b$B-zipf-c32-notext" --trace zipf --concurrency 32 \
    --requests "$REQ" --warmup "$WARM" --text 0
done
echo TREND_DONE
