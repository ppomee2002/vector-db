#!/bin/bash
# 12주차: 이번 주 배치를 실제 Qdrant에 넣고 지난주와 같은 방식으로 측정
set -o pipefail
D=~/m4-exp; S=$D/scripts; W=/workspace/jihyeon/w12q
export M4_STORAGE=$W/qdrant_storage   # 지난주 컬렉션과 분리된 저장 위치
source $D/venv/bin/activate
trap 'bash $S/qdrant_ctl.sh stop' EXIT
LAY="w12_doc w12_rand w12_P64 w12_P64x w12_HN"
measure() {  # $1=쿼리집합 $2=이웃범위 $3=배치 $4=결과폴더
  bash $S/qdrant_ctl.sh start || { echo "FAILED start $1 $2 $3"; return; }
  M4_DATA=$W/data_$1 ionice -c2 -n7 python $S/measure.py --layout $3 --window $2 --mode cold_each --out $4 2>&1 | tee -a $4/measure.log || echo "FAILED measure $1 $2 $3"
  bash $S/qdrant_ctl.sh stop
}
bash $S/qdrant_ctl.sh start || exit 1
first=1
for L in $LAY; do
  echo "$(date +%T) === 적재 $L"
  nice -n 10 python $W/load_w12.py --layout $L 2>&1 | tee -a $W/results/load.log || echo "FAILED load $L"
  if [ $first = 1 ]; then  # 첫 배치로 측정 코드가 도는지 바로 확인 (지난주 doc_1seg_b1의 약 5.9 페이지와 비교)
    first=0; bash $S/qdrant_ctl.sh stop; mkdir -p $W/results/smoke; rm -f $W/results/smoke/summary.jsonl
    echo "$(date +%T) === 확인용 측정 (unseen ±1 $L)"; measure unseen 1 $L $W/results/smoke
    bash $S/qdrant_ctl.sh start || exit 1
  fi
done
bash $S/qdrant_ctl.sh stop
for QS in unseen seen; do
  R=$W/results/$QS; mkdir -p $R; rm -f $R/summary.jsonl
  for WN in 0 1; do for L in $LAY; do echo "$(date +%T) === 측정 $QS ±$WN $L"; measure $QS $WN $L $R; done; done
done
echo "$(date +%T) ALL DONE"
