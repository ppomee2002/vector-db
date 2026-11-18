#!/bin/bash
# usage: nohup bash run_reclaim.sh > reclaim.log 2>&1 &
# 원래 설정(벡터 인덱스는 메모리, 본문만 mmap)의 nq_seq에서 "본문 캐시를 비운 직후"를 잰다. 재시작은 하지 않는다.
# IOPS 조건(없음, 3000, 1000)마다:
#   1) Milvus 캐시 비우기(reclaim) → --sr (예열 없이 S, R, R_net)
#   2) 방식마다 [reclaim → 예열 없이 20초 측정] × 3회, 방식 순서는 회차마다 같음
# 측정 하나라도 실패하거나 Milvus 상태(healthz)가 이상하면 거기서 전체를 멈춘다. 끝나면 항상 reset, nq_seq release.
# 결과: results_reclaim_nq.csv, results_reclaim_nq_sr.csv (cond = reclaim, reclaim_iops3000, reclaim_iops1000)
ROOT=${ROOT:-/workspace/junlim}
BACKEND=${BACKEND:-milvus}      # 도구 점검용 mock
DUR=${DUR:-20}; REPS=${REPS:-3}
LEVELS=${LEVELS:-"off 3000 1000"}
SW_CMD=${SW_CMD:-"timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=10 capvdb@203.253.145.110"}
cd "$ROOT" || exit 1
[ -f venv/bin/activate ] && source venv/bin/activate
export BENCH_DATA=$ROOT/data_nq BENCH_COLL=nq
OUT=results_reclaim_nq.csv
CFGS=("m1 64" "m3 64" "m3 8" "m5 64" "m6 64" "m7 64 8" "m8 64 16")

ts() { date '+%H:%M:%S'; }
sw() { echo "$(ts) switch: $*"; $SW_CMD "$*"; }
status() { $SW_CMD status 2>&1; }
health() { [ "$BACKEND" != "milvus" ] && return 0; curl -s -m 3 localhost:9091/healthz | grep -q OK; }
stop() { echo "$(ts) STOP: $1"; exit 1; }
bench() {
  health || stop "Milvus healthz 실패"
  python bench_v2.py --backend "$BACKEND" "$@" || stop "실패한 측정: $*"
}
cleanup() {
  echo "$(ts) cleanup: reset"; $SW_CMD reset
  if [ "$BACKEND" = "milvus" ]; then
    python -c "from pymilvus import MilvusClient as M; c=M(uri='http://localhost:19530',db_name='junlim'); c.release_collection('nq_seq'); print('released nq_seq', c.get_load_state('nq_seq'))" || echo "release 실패"
  fi
  status | head -4
}
trap cleanup EXIT
trap 'exit 130' INT TERM

NRUN=$(( ${#CFGS[@]} * REPS * $(echo $LEVELS | wc -w) ))
echo "$(ts) start run_reclaim levels=[$LEVELS] runs=$NRUN (약 $(( NRUN * (DUR + 8) / 60 + 3 ))분) backend=$BACKEND"
health || stop "시작 전 Milvus healthz 실패"
sw reset

# 원래 설정인지 확인하고 nq_seq를 올린다 (인덱스 mmap이 켜져 있으면 멈춘다)
if [ "$BACKEND" = "milvus" ]; then
python - <<'PYEOF' || stop "nq_seq 준비 실패"
import time
from pymilvus import MilvusClient as M
c = M(uri="http://localhost:19530", db_name="junlim")
idx = c.describe_index("nq_seq", "vec")
if str(idx.get("mmap.enabled", "False")).lower() == "true":
    raise SystemExit("index mmap is ON: run `python set_mmap.py off` first")
c.release_collection("fiqa_seq")
if "Loaded" not in str(c.get_load_state("nq_seq")):
    t = time.time(); c.load_collection("nq_seq")
    print(f"loaded nq_seq in {time.time() - t:.1f}s", flush=True)
print("nq_seq", c.get_load_state("nq_seq"), "index mmap:", idx.get("mmap.enabled", "False"), flush=True)
PYEOF
sleep 20
fi
status | head -12

for lv in $LEVELS; do
  if [ "$lv" = "off" ]; then
    COND=reclaim; sw iops off
  else
    COND=reclaim_iops$lv; sw iops "$lv"
    status | sed -n '/io.max/,/floor/p' | grep -q "$lv" || stop "IOPS $lv 이 걸리지 않음"
  fi
  echo "$(ts) ===== $COND"
  sw reclaim || stop "reclaim 실패"
  bench --sr --warm-q 0 --cond "$COND" --out "$OUT"
  for rep in $(seq 1 $REPS); do
    echo "$(ts) $COND rep $rep"
    for cfg in "${CFGS[@]}"; do
      set -- $cfg
      $SW_CMD reclaim > /dev/null 2>&1 || stop "reclaim 실패"
      bench --method $1 --n $2 --cap ${3:-8} --rep $rep --cond "$COND" --dur $DUR --warm 0 --out "$OUT"
    done
  done
  status | head -12
done
echo "$(ts) done run_reclaim"
