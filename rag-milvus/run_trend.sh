#!/usr/bin/env bash
# Trend run: Milvus tiered storage (config C) under shrinking text-cache budgets.
#   ./run_trend.sh <dataset> <out_dir>
# For each budget: restart Milvus with memory watermarks = vector_index + budget*text,
# load, then run natural/zipf traces at several concurrencies while capturing S3 GETs.
set -euo pipefail
cd "$(dirname "$0")"
DS=${1:?dataset}; OUT=${2:?out}; mkdir -p "$OUT"
PY=.venv/bin/python
BUDGETS=${BUDGETS:-"1.0 0.5 0.2 0.1"}
CONCS=${CONCS:-"1 32 128"}
TRACES=${TRACES:-"natural zipf"}
REQ=${REQ:-0}          # 0 = one pass over all queries
WARM=${WARM:-1000}

metric() {  # sum of a labeled metric for memory location and data_type
  curl -s http://127.0.0.1:9091/metrics | awk -v m="$1" -v t="$2" '
    index($0, m"{")==1 && index($0, "data_type=\""t"\"") && index($0, "location=\"memory\"") {s+=$NF} END{printf "%.0f", s}'
}
capacity() { curl -s http://127.0.0.1:9091/metrics | awk 'index($0,"internal_cache_capacity_bytes{location=\"memory\"}")==1{printf "%.0f",$NF}'; }

# 1) probe sizes with no pressure: vector index bytes (sync warmup) + total text bytes
./milvus_ctl.sh start C
$PY -c "from pymilvus import MilvusClient; MilvusClient('http://127.0.0.1:19530',timeout=900).load_collection('$DS')"
VIDX=$(metric internal_cache_loaded_bytes vector_index); OTHER=$(metric internal_cache_loaded_bytes other)
CAP=$(capacity)
TEXT=$($PY -c "import json;print(json.load(open('datasets/emb/$DS/milvus_load.json'))['text_bytes'])")
echo "{\"vector_index_bytes\":$VIDX,\"other_bytes\":$OTHER,\"text_bytes\":$TEXT,\"memory_capacity\":$CAP}" | tee "$OUT/sizes.json"

for B in $BUDGETS; do
  HIGH=$($PY -c "print(min(0.95, ($VIDX+$OTHER+ $B*$TEXT*1.15)/$CAP))")   # 1.15: pk/doc_id + encoding overhead
  LOW=$($PY -c "print($HIGH*0.95)")
  EXTRA_ENV="QUERYNODE_SEGCORE_TIEREDSTORAGE_MEMORYHIGHWATERMARKRATIO=$HIGH QUERYNODE_SEGCORE_TIEREDSTORAGE_MEMORYLOWWATERMARKRATIO=$LOW" \
    ./milvus_ctl.sh start C
  $PY -c "from pymilvus import MilvusClient; MilvusClient('http://127.0.0.1:19530',timeout=900).load_collection('$DS')"
  for T in $TRACES; do
    for CC in $CONCS; do
      CASE="$DS-C-b$B-$T-c$CC"
      # S3 GET capture (plain HTTP on loopback): request line + Range header only
      sudo -n timeout 3600 tcpdump -i lo -s 1024 -A -l 'tcp dst port 8333' 2>/dev/null \
        | grep --line-buffered -aoE '(GET|HEAD) /[^ ]+ HTTP|[Rr]ange: bytes=[0-9-]+' > "$OUT/$CASE.s3.log" &
      TP=$!
      sleep 1
      $PY bench.py --dataset "$DS" --out "$OUT" --case "$CASE" --trace "$T" --concurrency "$CC" \
        --requests "$REQ" --warmup "$WARM" --text 1
      sleep 1; sudo -n pkill -f "[t]cpdump -i lo -s 1024" || true; wait $TP 2>/dev/null || true
    done
  done
  # (3) context-read cost: same trace without returning text
  $PY bench.py --dataset "$DS" --out "$OUT" --case "$DS-C-b$B-zipf-c32-notext" --trace zipf --concurrency 32 \
    --requests "$REQ" --warmup "$WARM" --text 0
done
echo TREND_DONE
