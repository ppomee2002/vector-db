#!/bin/bash
# usage: cd /workspace/junlim && bash run_pf2.sh > pf2.log 2>&1      (보통은 wait_run_pf.sh가 대신 부른다)
# C2 미리 읽기 검증 v2 (bench_pf2.py). 원래 설정(벡터 인덱스 메모리, 본문만 mmap)의 nq_seq, 평소 상태(캐시 따뜻함).
#   1) 순서대로 보내기: 이웃이 필요한 비율 p = 0, 5, 10, 20, 50, 100% × 정책 3개, p마다 300질의 × 3회   (약 3분)
#   2) 동시 요청 64개, 요청 전체 동시 수 상한 8: p = 0, 10, 50, 100% × 정책 3개, 구간 25초 × 3회          (약 17분)
#      회차마다 정책 순서를 돌린다.
# 시작 전에 check_busy.sh로 다른 팀원이 쓰는 중인지 보고, 쓰는 중이면 아무것도 바꾸지 않고 끝낸다.
# 측정이 하나라도 실패하거나 Milvus 상태(healthz)가 이상하면 멈춘다. 끝나면 항상 reset, nq_seq release.
# 결과: results_pf2_seq_nq.csv, results_pf2_load_nq.csv
ROOT=${ROOT:-/workspace/junlim}
BACKEND=${BACKEND:-milvus}      # 도구 점검용 mock
REPS=${REPS:-3}
LOAD_PS=${LOAD_PS:-"0 0.1 0.5 1"}
DUR=${DUR:-20}; WARM=${WARM:-5}; N=${N:-64}; CAP=${CAP:-8}
SW_CMD=${SW_CMD:-"timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=10 capvdb@203.253.145.110"}
cd "$ROOT" || exit 1
[ -f venv/bin/activate ] && source venv/bin/activate
export BENCH_DATA=$ROOT/data_nq BENCH_COLL=nq
OUT_SEQ=results_pf2_seq_nq.csv; OUT_LOAD=results_pf2_load_nq.csv

ts() { date '+%H:%M:%S'; }
sw() { echo "$(ts) switch: $*"; $SW_CMD "$*"; }
status() { $SW_CMD status 2>&1; }
health() { [ "$BACKEND" != "milvus" ] && return 0; curl -s -m 3 localhost:9091/healthz | grep -q OK; }
stop() { echo "$(ts) STOP: $1"; exit 1; }
bench() {
  health || stop "Milvus healthz 실패"
  python bench_pf2.py --backend "$BACKEND" "$@" || stop "실패한 측정: $*"
}
cleanup() {
  echo "$(ts) cleanup: reset"; $SW_CMD reset
  if [ "$BACKEND" = "milvus" ]; then
    python -c "from pymilvus import MilvusClient as M; c=M(uri='http://localhost:19530',db_name='junlim'); c.release_collection('nq_seq'); print('released nq_seq', c.get_load_state('nq_seq'))" || echo "release 실패"
  fi
  status | head -4
}

# 다른 팀원이 쓰는 중이면 아무것도 바꾸지 않고 끝낸다 (reset, release를 하기 전에 확인)
if [ "$BACKEND" = "milvus" ] && [ "${SKIP_BUSY_CHECK:-0}" != "1" ]; then
  bash check_busy.sh || { echo "$(ts) STOP: 다른 사람이 쓰는 중이거나 확인하지 못함. 아무것도 바꾸지 않고 끝냄 (위 결과 참고)"; exit 1; }
fi

trap cleanup EXIT
trap 'exit 130' INT TERM

echo "$(ts) start run_pf2 reps=$REPS backend=$BACKEND load_ps=[$LOAD_PS] n=$N cap=$CAP dur=$DUR"
python bench_pf2.py --backend mock --selftest > /dev/null || stop "bench_pf2.py selftest 실패"
health || stop "시작 전 Milvus healthz 실패"
sw reset

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

echo "$(ts) ===== seq (요청을 하나씩)"
for rep in $(seq 1 $REPS); do
  bench --mode seq --cond pf2_seq --rep $rep --out "$OUT_SEQ"
done

echo "$(ts) ===== load (동시 요청 $N, 상한 $CAP)"
POLS=(lazy_id prefetch lazy_text)
for rep in $(seq 1 $REPS); do
  for p in $LOAD_PS; do
    for i in 0 1 2; do
      pol=${POLS[$(( (i + rep - 1) % 3 ))]}
      bench --mode load --cond pf2_load --rep $rep --policy $pol --p $p --n $N --cap $CAP --dur $DUR --warm $WARM --out "$OUT_LOAD"
    done
  done
done
status | head -12
echo "$(ts) done run_pf2"
