#!/bin/bash
# 추가 실험. 임베딩은 재사용(다시 안 함). 결과는 ~/m4-exp/results2 에 따로 저장.
set -eo pipefail
D=~/m4-exp; S=$D/scripts; R=$D/results2; cd $S
source $D/venv/bin/activate; mkdir -p $R
trap 'bash $S/qdrant_ctl.sh stop' EXIT
echo "== prep stats =="; cat $D/data/prep_stats.json; echo
bash qdrant_ctl.sh start
for L in oracle stream8 doc_1seg; do nice -n 10 python load.py --layout $L 2>&1 | tee -a $R/load.log; done
bash qdrant_ctl.sh stop
rm -f $R/summary.jsonl $R/qps.jsonl
LAYOUTS="interleaved stream8 doc doc_1seg coaccess oracle"
# (1) 쿼리당 I/O: Expansion 범위 ±0, ±1, ±2
for W in 0 1 2; do for L in $LAYOUTS; do
  bash qdrant_ctl.sh start
  ionice -c2 -n7 python measure.py --layout $L --window $W --mode cold_each --out $R 2>&1 | tee -a $R/measure.log
  bash qdrant_ctl.sh stop
done; done
# (2) QPS: 매번 Qdrant 재시작 + evict 해서 진짜 cold, 동시요청 1/16, 3회 반복
for REP in 0 1 2; do for W in 0 1; do for C in 1 16; do for L in $LAYOUTS; do
  bash qdrant_ctl.sh start >/dev/null
  ionice -c2 -n7 python qps.py --layout $L --window $W --conc $C --rep $REP --out $R 2>&1 | tee -a $R/qps.log
  bash qdrant_ctl.sh stop >/dev/null
done; done; done; done
python analyze2.py $R
echo "ALL DONE"
