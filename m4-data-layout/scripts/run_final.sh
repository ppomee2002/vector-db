#!/bin/bash
# 마지막 실험: Qdrant가 연속 chunk를 흩어 저장하는 원인 확인 (약 30~40분) -> ~/m4-exp/results4
#  doc          : 기존 (세그먼트 8개, 256개씩 적재)
#  doc_1seg     : 기존 (세그먼트 1개, 256개씩)
#  doc_b1       : 세그먼트 8개, 1개씩 순서대로 적재
#  doc_1seg_b1  : 세그먼트 1개, 1개씩 순서대로 적재  <- 순서가 지켜질 가능성이 가장 큰 조건
#  doc_1seg_b16 : 세그먼트 1개, 16개씩 적재
set -eo pipefail
D=~/m4-exp; S=$D/scripts; cd $S
source $D/venv/bin/activate
trap 'bash $S/qdrant_ctl.sh stop' EXIT
R=$D/results4; mkdir -p $R; rm -f $R/summary.jsonl $R/locate.txt
NEW="doc_b1 doc_1seg_b1 doc_1seg_b16"
bash qdrant_ctl.sh start
for L in $NEW; do nice -n 10 python load.py --layout $L 2>&1 | tee -a $R/load.log; done
bash qdrant_ctl.sh stop
for W in 0 1; do for L in doc doc_1seg $NEW; do
  bash qdrant_ctl.sh start
  ionice -c2 -n7 python measure.py --layout $L --window $W --mode cold_each --out $R 2>&1 | tee -a $R/measure.log
  bash qdrant_ctl.sh stop
done; done
python analyze2.py $R
for L in doc doc_1seg $NEW; do python locate.py $L 2>&1 | tee -a $R/locate.txt; done
echo "ALL DONE"
