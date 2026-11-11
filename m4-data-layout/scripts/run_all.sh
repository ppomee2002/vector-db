#!/bin/bash
# 전체 실행. 모든 작업은 ~/m4-exp 안에서만. tmux 안에서 실행 권장.
set -eo pipefail
D=~/m4-exp; S=$D/scripts; cd $S
source $D/venv/bin/activate
mkdir -p $D/results
trap 'bash $S/qdrant_ctl.sh stop' EXIT
[ -f $D/data/emb.npy ] || nice -n 10 python prep_nq.py --out $D/data 2>&1 | tee $D/results/prep.log
bash qdrant_ctl.sh start
for L in doc interleaved coaccess; do nice -n 10 python load.py --layout $L 2>&1 | tee -a $D/results/load.log; done
bash qdrant_ctl.sh stop
rm -f $D/results/summary.jsonl
for W in 0 1; do for M in cold_each warm; do for L in doc interleaved coaccess; do
  bash qdrant_ctl.sh start   # 매 측정마다 재시작 -> Qdrant 내부 캐시도 초기화
  ionice -c2 -n7 python measure.py --layout $L --window $W --mode $M 2>&1 | tee -a $D/results/measure.log
  bash qdrant_ctl.sh stop
done; done; done
python analyze.py $D/results
echo "ALL DONE"
