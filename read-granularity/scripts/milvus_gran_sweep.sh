#!/usr/bin/env bash
# Milvus tiered-storage cell-size sweep: patched libmilvus_core.so reads MILVUS_ROW_GROUPS_PER_CELL.
#   ./milvus_gran_sweep.sh <dataset> <out>
# Per case: bench.py summary (<case>.json), S3 request log (<case>.s3.log: GET lines + Range headers
# captured on lo:8336), and the cell sizes Milvus logged at load (<case>.cells.txt).
set -euo pipefail
cd ~/rag-milvus
PY=.venv/bin/python
DS=${1:?dataset}; OUT=${2:?out}; mkdir -p "$OUT"
RGPCS=${RGPCS:-"1 2 4 8 16"}
BUDGETS=${BUDGETS:-"0.05 0.2"}
TRACES=${TRACES:-"natural zipf"}
CONCS=${CONCS:-"32"}
REQ=${REQ:-2000}; WARM=${WARM:-1000}
LIBCORE=${LIBCORE:-$HOME/gran/milvus/internal/core/output/lib/libmilvus_core.so}
LOG=$HOME/gran/milvus-C.log

metric() {
  curl -s http://127.0.0.1:9091/metrics | awk -v m="$1" -v t="$2" '
    index($0, m"{")==1 && index($0, "data_type=\""t"\"") && index($0, "location=\"memory\"") {s+=$NF} END{printf "%.0f", s}'
}
capacity() { curl -s http://127.0.0.1:9091/metrics | awk 'index($0,"internal_cache_capacity_bytes{location=\"memory\"}")==1{printf "%.0f",$NF}'; }
load() {  # load, then wait until a search actually succeeds (load_collection can return before the channel is serviceable)
  $PY - "$DS" <<'PYEOF'
import sys, time
from pymilvus import MilvusClient
ds = sys.argv[1]
c = MilvusClient('http://127.0.0.1:19530', timeout=1800)
c.load_collection(ds)
dim = c.describe_collection(ds)['fields']
dim = next(f['params']['dim'] for f in dim if 'dim' in f.get('params', {}))
t0 = time.time()
while True:
    try:
        c.search(ds, data=[[0.01] * int(dim)], limit=1, output_fields=['text']); break
    except Exception as e:
        if time.time() - t0 > 600: raise
        time.sleep(3)
print(f"searchable after {time.time()-t0:.0f}s", file=sys.stderr)
PYEOF
}

# sizes with the stock cell size (budget computation only)
LIBCORE=$LIBCORE RGPC=4 ./milvus_gran_ctl.sh start C >/dev/null
load
VIDX=$(metric internal_cache_loaded_bytes vector_index); OTHER=$(metric internal_cache_loaded_bytes other)
CAP=$(capacity)
TEXT=$($PY -c "import json;print(json.load(open('datasets/emb/$DS/milvus_load.json'))['text_bytes'])")
echo "{\"vector_index_bytes\":$VIDX,\"other_bytes\":$OTHER,\"text_bytes\":$TEXT,\"memory_capacity\":$CAP}" | tee "$OUT/sizes-$DS.json"

run_case() {  # run_case <case> <trace> <conc>
  local CASE=$1 T=$2 CC=$3
  [ -f "$OUT/$CASE.json" ] && { echo "skip $CASE"; return; }
  sudo -n timeout 3600 tcpdump -i lo -s 1024 -A -l 'tcp dst port 8336' 2>/dev/null \
    | grep --line-buffered -aoE '(GET|HEAD) /[^ ]+ HTTP|[Rr]ange: bytes=[0-9-]+' > "$OUT/$CASE.s3.log" &
  local TP=$!
  sleep 1
  $PY bench.py --dataset "$DS" --out "$OUT" --case "$CASE" --trace "$T" --concurrency "$CC" \
    --requests "$REQ" --warmup "$WARM" --text 1
  sleep 1; sudo -n pkill -f "[t]cpdump -i lo -s 1024 -A -l tcp dst port 8336" || true; wait $TP 2>/dev/null || true
}

for B in $BUDGETS; do
  HIGH=$($PY -c "print(min(0.95, ($VIDX+$OTHER+ $B*$TEXT*1.15)/$CAP))")
  LOW=$($PY -c "print($HIGH*0.95)")
  WM="QUERYNODE_SEGCORE_TIEREDSTORAGE_MEMORYHIGHWATERMARKRATIO=$HIGH QUERYNODE_SEGCORE_TIEREDSTORAGE_MEMORYLOWWATERMARKRATIO=$LOW"
  # stock image (no patched core) as the reference for rgpc=4
  for R in stock $(shuf -e $RGPCS); do
    if [ "$R" = stock ]; then
      EXTRA_ENV="$WM" LIBCORE= ./milvus_gran_ctl.sh start C >/dev/null
    else
      EXTRA_ENV="$WM" LIBCORE=$LIBCORE RGPC=$R ./milvus_gran_ctl.sh start C >/dev/null
    fi
    load
    sudo -n grep -a "row groups per cell" "$LOG" | tail -200 > "$OUT/$DS-r$R-b$B.cells.txt" || true
    for T in $TRACES; do
      for CC in $CONCS; do
        run_case "$DS-r$R-b$B-$T-c$CC" "$T" "$CC"
      done
    done
  done
done
./milvus_gran_ctl.sh stop
echo SWEEP_DONE
