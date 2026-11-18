#!/bin/bash
# usage: bash run_cap.sh fiqa|nq
# 요청 전체(검색+읽기)의 동시 수 상한(--cap)을 바꿔 가며 처리량을 잰다. 동시 요청 수 n=64 고정.
cd /workspace/junlim || exit 1
source venv/bin/activate
if [ "$1" = "nq" ]; then
  export BENCH_DATA=/workspace/junlim/data_nq BENCH_COLL=nq
  export ACTIVE=nq_seq OTHER=fiqa_seq
elif [ "$1" = "fiqa" ]; then
  export BENCH_DATA=/workspace/junlim/data BENCH_COLL=fiqa
  export ACTIVE=fiqa_seq OTHER=nq_seq
else
  echo "usage: bash run_cap.sh fiqa|nq"; exit 1
fi
OUT=results_cap_$1.csv
echo "$(date '+%H:%M:%S') start $1 (kst clock: add 9h to UTC)"
python - <<'PYEOF'
import os
from pymilvus import MilvusClient as M
c = M(uri="http://localhost:19530", db_name="junlim")
c.release_collection(os.environ["OTHER"])
c.load_collection(os.environ["ACTIVE"])
print("loaded", os.environ["ACTIVE"], c.get_load_state(os.environ["ACTIVE"]), flush=True)
PYEOF
sleep 20
python bench_v2.py --method m3 --n 16 --dur 10 --warm 3 --cond warmup --out warmup_$1.csv > /dev/null 2>&1
for rep in 1 2 3; do
  for k in 2 4 6 8 12 16 32; do
    python bench_v2.py --method m7 --n 64 --cap $k --rep $rep --cond cap --dur 20 --warm 5 --out $OUT
  done
  for k in 4 8 16 32; do
    python bench_v2.py --method m8 --n 64 --cap $k --rep $rep --cond cap --dur 20 --warm 5 --out $OUT
  done
done
echo "$(date '+%H:%M:%S') done $1"
