#!/bin/bash
# Milvus M4 실험 (m4m_ 컬렉션만 사용, 끝나면 삭제). 결과: ~/m4-exp/results_milvus
set -eo pipefail
D=~/m4-exp; S=$D/scripts; cd $S; source $D/venv/bin/activate
R=$D/results_milvus; mkdir -p $R; rm -f $R/summary.jsonl
LAYOUTS="interleaved doc doc_rpk"
for L in $LAYOUTS; do python milvus_load.py --layout $L 2>&1 | tee -a $R/load.log; done
for W in 0 1; do for L in $LAYOUTS; do
  python milvus_measure.py --layout $L --window $W --mode cold_each --out $R 2>&1 | tee -a $R/measure.log
done; done
python -c "
from pymilvus import MilvusClient; c=MilvusClient('http://localhost:19530')
[c.drop_collection(n) for n in c.list_collections() if n.startswith('m4m_')]; print('m4m_ 컬렉션 정리 완료')"
echo "ALL DONE"
