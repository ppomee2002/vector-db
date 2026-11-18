#!/bin/bash
# usage: bash run_cond.sh nq|fiqa COND [std|full|cold]
#   COND : 조건 이름(결과 CSV의 cond 칸). 예) base, mem60, mem35, mem35_iops3000, mem35_iops1000, cold
#   std  : 방식 7개를 같은 시간대에 번갈아 3회 측정 (약 10분)
#   full : std + 상한 m7 k=12,16,32, m8 k=32 (이상 구간 재측정 포함, 약 15분)
#   cold : 재시작/캐시 비운 직후용. 예열 없이 --sr → m7 k=8 연속 6구간 → std 1회 (약 6분)
# 결과: results_cond_<ds>.csv, results_cond_<ds>_sr.csv (조건이 바뀌어도 같은 파일에 cond 칸으로 구분해 쌓임)
ROOT=${ROOT:-/workspace/junlim}
BACKEND=${BACKEND:-milvus}      # 도구 점검용 mock
DUR=${DUR:-20}; WARM=${WARM:-5}; REPS=${REPS:-3}
cd "$ROOT" || exit 1
[ -f venv/bin/activate ] && source venv/bin/activate

DS=$1; COND=$2; MODE=${3:-std}
if [ "$DS" = "nq" ]; then
  export BENCH_DATA=$ROOT/data_nq BENCH_COLL=nq ACTIVE=nq_seq OTHER=fiqa_seq
elif [ "$DS" = "fiqa" ]; then
  export BENCH_DATA=$ROOT/data BENCH_COLL=fiqa ACTIVE=fiqa_seq OTHER=nq_seq
else
  echo "usage: bash run_cond.sh nq|fiqa COND [std|full|cold]"; exit 1
fi
if [ -z "$COND" ] || [[ "$COND" =~ [^A-Za-z0-9_] ]]; then
  echo "COND는 영문, 숫자, _ 만 (예: base, mem35_iops1000)"; exit 1
fi
case "$MODE" in std|full|cold) ;; *) echo "MODE는 std|full|cold"; exit 1;; esac
OUT=results_cond_$DS.csv

# 방식 목록: "method n cap" (cap은 m7, m8만 의미 있음)
CFGS=("m1 64" "m3 64" "m3 8" "m5 64" "m6 64" "m7 64 8" "m8 64 16")
EXTRA=("m7 64 12" "m7 64 16" "m7 64 32" "m8 64 32")
[ "$MODE" = "full" ] && CFGS+=("${EXTRA[@]}")
[ "$MODE" = "cold" ] && REPS=1

ABORT=0
health() { [ "$BACKEND" != "milvus" ] && return 0; curl -s -m 3 localhost:9091/healthz | grep -q OK; }
# 측정 하나가 실패하거나 Milvus 상태가 이상하면 이 조건의 남은 측정을 모두 건너뛰고 끝에 실패로 끝낸다
bench() {
  [ "$ABORT" = 1 ] && return 1
  if ! health; then echo "$(date '+%H:%M:%S') ABORT: Milvus healthz 실패" >&2; ABORT=1; return 1; fi
  python bench_v2.py --backend "$BACKEND" "$@" || { echo "$(date '+%H:%M:%S') ABORT: 실패한 측정: $*" >&2; ABORT=1; return 1; }
}

snap() {
  echo "----- [$(date '+%F %T')] $1"
  # 관리자 스위치: Milvus의 anon/file 메모리와 현재 걸린 제한(mem, iops)을 기록
  timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=5 capvdb@203.253.145.110 status 2>&1 | head -30 || echo "status: 못 읽음"
  curl -s -m 3 localhost:9091/metrics | grep -E "^process_resident_memory_bytes" || echo "metrics: 못 읽음"
  free -m | head -2
  grep -w loop1 /proc/diskstats || echo "loop1: 없음"
}

NRUN=$(( ${#CFGS[@]} * REPS ))
[ "$MODE" = "cold" ] && NRUN=$(( NRUN + 6 ))
echo "start ds=$DS cond=$COND mode=$MODE runs=$NRUN (약 $(( NRUN * 27 / 60 + 2 ))분) backend=$BACKEND"
snap "before"

# 다른 컬렉션은 내리고, 이 컬렉션이 안 올라가 있으면 올린다(걸린 시간 기록)
export JUST_LOADED_FILE=$(mktemp)
if [ "$BACKEND" = "milvus" ]; then
python - <<'PYEOF' || { echo "LOAD FAILED"; snap "load failed"; exit 1; }
import os, time
from pymilvus import MilvusClient as M
c = M(uri="http://localhost:19530", db_name="junlim")
a, o = os.environ["ACTIVE"], os.environ["OTHER"]
c.release_collection(o)
st = c.get_load_state(a)
if "Loaded" in str(st):
    print("already loaded", a, st, flush=True)
else:
    t = time.time(); c.load_collection(a)
    print(f"loaded {a} in {time.time() - t:.1f}s", c.get_load_state(a), flush=True)
    open(os.environ["JUST_LOADED_FILE"], "w").write("1")
PYEOF
fi
if [ -s "$JUST_LOADED_FILE" ] && [ "$MODE" != "cold" ]; then sleep 20; fi
rm -f "$JUST_LOADED_FILE"

if [ "$MODE" = "cold" ]; then
  echo "$(date '+%H:%M:%S') cold: sr (예열 없음)"
  bench --sr --warm-q 0 --cond "$COND" --out "$OUT"
  for i in 1 2 3 4 5 6; do
    bench --method m7 --n 64 --cap 8 --rep $i --cond "${COND}_series" --dur $DUR --warm 0 --out "$OUT"
  done
  snap "after series"
else
  echo "$(date '+%H:%M:%S') warm-up (분석 제외)"
  bench --method m3 --n 16 --dur 30 --warm 5 --cond "${COND}_warmup" --out warmup_cond_$DS.csv > /dev/null
  bench --sr --cond "$COND" --out "$OUT"
fi

for rep in $(seq 1 $REPS); do
  echo "$(date '+%H:%M:%S') rep $rep"
  for cfg in "${CFGS[@]}"; do
    set -- $cfg
    bench --method $1 --n $2 --cap ${3:-8} --rep $rep --cond "$COND" --dur $DUR --warm $WARM --out "$OUT"
  done
done
snap "after"
if [ "$ABORT" = 1 ]; then echo "$(date '+%H:%M:%S') ABORTED ds=$DS cond=$COND"; exit 1; fi
echo "$(date '+%H:%M:%S') done ds=$DS cond=$COND"
