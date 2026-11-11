#!/bin/bash
# 중요 실험 2개를 차례로 실행.
#  1) tracker 분리 측정 (기존 작은 규모 데이터/컬렉션 재사용, 약 20분) -> ~/m4-exp/results3
#  2) 다른 데이터셋 HotpotQA (약 1~1.5시간)                             -> ~/m4-exp/results_hotpot
set -eo pipefail
D=~/m4-exp; S=$D/scripts; cd $S
source $D/venv/bin/activate
trap 'bash $S/qdrant_ctl.sh stop' EXIT
LAYOUTS="interleaved stream8 doc doc_1seg coaccess oracle"

echo "########## 1) tracker 분리 측정 ##########"
R=$D/results3; mkdir -p $R; rm -f $R/summary.jsonl
for W in 0 1 2; do for L in $LAYOUTS; do
  bash qdrant_ctl.sh start
  ionice -c2 -n7 python measure.py --layout $L --window $W --mode cold_each --out $R 2>&1 | tee -a $R/measure.log
  bash qdrant_ctl.sh stop
done; done
python analyze2.py $R
python sim_pages.py $R | tee $R/sim.txt
echo "PART1 DONE"

echo "########## 2) HotpotQA ##########"
export M4_DATA=$D/data_hotpot M4_STORAGE=$D/qdrant_storage_hotpot
R=$D/results_hotpot; mkdir -p $R $M4_DATA; rm -f $R/summary.jsonl
[ -f $M4_DATA/emb.npy ] || nice -n 10 python prep_beir.py --dataset hotpotqa 2>&1 | tee $R/prep.log
cat $M4_DATA/prep_stats.json; echo
L2="interleaved stream8 doc coaccess oracle"
bash qdrant_ctl.sh start
for L in $L2; do nice -n 10 python load.py --layout $L 2>&1 | tee -a $R/load.log; done
bash qdrant_ctl.sh stop
for W in 0 1; do for L in $L2; do
  bash qdrant_ctl.sh start
  ionice -c2 -n7 python measure.py --layout $L --window $W --mode cold_each --out $R 2>&1 | tee -a $R/measure.log
  bash qdrant_ctl.sh stop
done; done
python analyze2.py $R
python sim_pages.py $R | tee $R/sim.txt
echo "ALL DONE"
