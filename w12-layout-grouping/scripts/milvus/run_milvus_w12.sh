#!/bin/bash
set -o pipefail
W=/workspace/jihyeon/w12q; R=$W/results/milvus; mkdir -p $R; rm -f $R/summary.jsonl
source ~/m4-exp/venv/bin/activate; cd $W
LAY="doc rand P64 P64x HN"
for L in $LAY; do echo "$(date +%T) === 적재 $L"; python milvus_w12.py load --layout $L 2>&1 | tee -a $R/load.log || echo "FAILED load $L"; done
for QS in unseen seen; do for WN in 0 1; do for L in $LAY; do
  echo "$(date +%T) === 측정 $QS ±$WN $L"
  python milvus_w12.py measure --layout $L --qs $QS --window $WN --nq 100 2>&1 | tee -a $R/measure.log || echo "FAILED measure $QS $WN $L"
done; done; done
python milvus_w12.py drop 2>&1 | tee -a $R/load.log
python analyze_milvus_w12.py
echo "$(date +%T) ALL DONE"
