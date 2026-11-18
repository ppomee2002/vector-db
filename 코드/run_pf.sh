#!/bin/bash
# usage: cd /workspace/junlim && nohup bash run_pf.sh > pf.log 2>&1 &
# C2 미리 읽기 비용·이득 측정 (bench_pf.py). 원래 설정(벡터 인덱스 메모리, 본문만 mmap)의 nq_seq에서 잰다.
#   1) 평소 상태(캐시 따뜻함) × 3회          cond = pf_warm
#   2) [본문 캐시 비우기 → 바로 측정] × 3회   cond = pf_reclaim  (재시작 없음)
# 측정이 하나라도 실패하거나 Milvus 상태(healthz)가 이상하면 멈춘다. 끝나면 항상 reset, nq_seq release.
# 시작 전에 check_busy.sh로 다른 팀원이 쓰는 중인지 보고, 쓰는 중이면 아무것도 바꾸지 않고 끝낸다.
# 결과: results_pf_nq.csv  (약 5분)
ROOT=${ROOT:-/workspace/junlim}
BACKEND=${BACKEND:-milvus}      # 도구 점검용 mock
REPS=${REPS:-3}
SW_CMD=${SW_CMD:-"timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=10 capvdb@203.253.145.110"}
cd "$ROOT" || exit 1
[ -f venv/bin/activate ] && source venv/bin/activate
export BENCH_DATA=$ROOT/data_nq BENCH_COLL=nq
OUT=results_pf_nq.csv

ts() { date '+%H:%M:%S'; }
sw() { echo "$(ts) switch: $*"; $SW_CMD "$*"; }
status() { $SW_CMD status 2>&1; }
health() { [ "$BACKEND" != "milvus" ] && return 0; curl -s -m 3 localhost:9091/healthz | grep -q OK; }
stop() { echo "$(ts) STOP: $1"; exit 1; }
bench() {
  health || stop "Milvus healthz 실패"
  python bench_pf.py --backend "$BACKEND" "$@" || stop "실패한 측정: $*"
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

echo "$(ts) start run_pf reps=$REPS backend=$BACKEND"
python bench_pf.py --backend mock --selftest > /dev/null || stop "bench_pf.py selftest 실패"
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

echo "$(ts) ===== pf_warm"
for rep in $(seq 1 $REPS); do
  bench --cond pf_warm --rep $rep --out "$OUT"
done

echo "$(ts) ===== pf_reclaim"
for rep in $(seq 1 $REPS); do
  sw reclaim || stop "reclaim 실패"
  bench --cond pf_reclaim --rep $rep --warm-q 0 --out "$OUT"
done
status | head -12
echo "$(ts) done run_pf"
