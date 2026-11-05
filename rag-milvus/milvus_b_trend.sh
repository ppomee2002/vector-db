#!/usr/bin/env bash
# Trend run for Milvus config B (mmap scalar fields): text served through the OS page cache,
# page-cache budget set by the container cgroup memory.max (same method as qdrant_trend.sh).
#   ./milvus_b_trend.sh <dataset> <out_dir>      (uses MILVUS_STORAGE from the environment)
# Budget B: memory.max = anon * 1.05 + B * mapped_text + 256 MiB, where anon / mapped_text are the cgroup's
# anonymous and mmap'd-file bytes after load. Pages mapped by Milvus cannot be dropped with fadvise, so the
# cache is emptied between cases with cgroup memory.reclaim instead.
set -euo pipefail
cd "$(dirname "$0")"
DS=${1:?dataset}; OUT=${2:?out}; mkdir -p "$OUT"
PY=.venv/bin/python
BUDGETS=${BUDGETS:-"0.5 0.2 0.05"}
CONCS=${CONCS:-"32 128"}
TRACES=${TRACES:-"natural zipf"}
REQ=${REQ:-2000}; WARM=${WARM:-300}

drop_cache() {  # evict Milvus data files (binlogs, mmap files) from the page cache
  sudo -n python3 - volumes/milvus <<'PYEOF'
import os, sys
n = 0
for root, dirs, files in os.walk(sys.argv[1]):
    for f in files:
        try:
            fd = os.open(os.path.join(root, f), os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd); n += 1
        except OSError:
            pass
print(f"dropped page cache of {n} files", file=sys.stderr)
PYEOF
}
cg() { echo "/sys/fs/cgroup$(sudo -n cat /proc/"$(sudo -n k3s ctr task ls | awk '$1=="rm-milvus"{print $2}')"/cgroup | cut -d: -f3)"; }

./milvus_ctl.sh start B
$PY - "$DS" <<'PYEOF'
import sys
from pymilvus import MilvusClient
c = MilvusClient("http://127.0.0.1:19530", timeout=1800)
c.release_collection(sys.argv[1]); c.load_collection(sys.argv[1])
print("loaded with config B")
PYEOF
CG=$(cg)
stat() { awk -v k="$1" '$1==k{print $2}' "$CG/memory.stat"; }
drop_cache() {  # reclaim mmap'd text pages of the container (best effort)
  local f; f=$(stat file); echo "$f" | sudo -n tee "$CG/memory.reclaim" >/dev/null 2>&1 || true
  echo "reclaimed: file now $(stat file) (mapped $(stat file_mapped))" >&2
}
$PY bench.py --dataset "$DS" --out "$OUT/probe" --case probe --trace zipf --concurrency 8 --requests 500 --text 0 >/dev/null
ANON=$(stat anon); MAPPED=$(stat file_mapped); M0=$(cat "$CG/memory.current")
TEXT=$($PY -c "import json;print(json.load(open('datasets/emb/$DS/milvus_load.json'))['text_bytes'])")
echo "{\"mmap\":true,\"m0_bytes\":$M0,\"anon_bytes\":$ANON,\"mapped_bytes\":$MAPPED,\"text_bytes\":$TEXT,\"cgroup\":\"$CG\"}" | tee "$OUT/sizes.json"

for B in $BUDGETS; do
  LIM=$($PY -c "print(int($ANON*1.05 + $B*$MAPPED + (256<<20)))")
  echo "$LIM" | sudo -n tee "$CG/memory.max" >/dev/null; echo "budget $B memory.max=$LIM"
  for T in $TRACES; do
    for CC in $CONCS; do
      drop_cache
      $PY bench.py --dataset "$DS" --out "$OUT" --case "$DS-B-b$B-$T-c$CC" --trace "$T" --concurrency "$CC" \
        --requests "$REQ" --warmup "$WARM" --text 1 || { echo "case failed; memory.events: $(tr '\n' ' ' < "$CG/memory.events")"; exit 1; }
    done
  done
  drop_cache
  $PY bench.py --dataset "$DS" --out "$OUT" --case "$DS-B-b$B-zipf-c32-notext" --trace zipf --concurrency 32 \
    --requests "$REQ" --warmup "$WARM" --text 0
done
echo max | sudo -n tee "$CG/memory.max" >/dev/null
echo TREND_DONE
