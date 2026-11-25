#!/bin/bash
# C3 프로토타입 실행 (11/23 추가). c3_pipeline.py를 부른다. run_c3a.sh·bench_c3a.py·bench_v2.py는 고치지 않음.
# usage (서버, /workspace/junlim):
#   bash run_c3p.sh selftest   # Milvus를 건드리지 않는 도구 점검 (결정적 mock, 약 1분)
#   bash run_c3p.sh verify     # 정확성 검증 (Milvus, 약 2분): 같은 계획 입력 → 무제한/상한 k 결과 비교, 전체 경로 결과 비교
# 시작 전 check_busy.sh로 다른 사람이 쓰는지 보고, 쓰는 중이면 아무것도 바꾸지 않고 끝낸다. 끝나면 reset, nq_seq release, 결과 묶기.
# 성능 비교(무제한 vs k=8 번갈아 재기)는 측정 틀(3번 작업)에서 따로 한다. 이 스크립트에는 넣지 않았다.
ROOT=${ROOT:-/workspace/junlim}
PHASE=$1
OUT=${OUT:-$ROOT/c3p_out}
CAP=${CAP:-8}; SETTING=${SETTING:-C}; QSET=${QSET:-real}; VQ=${VQ:-1000}; N=${N:-64}
SW_CMD=${SW_CMD:-"timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=10 <switch-user>@<switch-host>"}
cd "$ROOT" || exit 1
[ -f venv/bin/activate ] && source venv/bin/activate
NQENV="BENCH_DATA=$ROOT/data_nq BENCH_COLL=nq"
ts() { date '+%H:%M:%S'; }
echo "$(ts) run_c3p $PHASE out=$OUT cap=$CAP setting=$SETTING qset=$QSET vq=$VQ n=$N"
md5sum c3_pipeline.py bench_c3a.py bench_v2.py run_c3p.sh | cut -c1-8,33-

case "$PHASE" in
  selftest)
    env $NQENV python c3_pipeline.py --backend mock --mode selftest --outdir /tmp/c3p_selftest --cap "$CAP" --setting "$SETTING" --qset "$QSET"
    exit $? ;;
  verify) ;;
  *) echo "usage: bash run_c3p.sh selftest|verify"; exit 1 ;;
esac

[ -s "$OUT/verify_nq.json" ] && { echo "$(ts) STOP: $OUT/verify_nq.json이 이미 있음. OUT=다른폴더 로 실행"; exit 1; }
if [ "${SKIP_BUSY_CHECK:-0}" != "1" ]; then
  bash check_busy.sh > /dev/null 2>&1 || { echo "$(ts) STOP: check_busy.sh가 '조용함'이 아님. 아무것도 바꾸지 않고 끝냄 (bash check_busy.sh로 내용 확인)"; exit 1; }
fi
curl -s -m 3 localhost:9091/healthz | grep -q OK || { echo "$(ts) STOP: Milvus healthz 실패"; exit 1; }
mkdir -p "$OUT"
LOG=$OUT/c3p_verify.log
exec > >(trap '' INT TERM HUP; exec tee -a "$LOG") 2>&1

coll() {  # coll load|release 이름 (run_c3a.sh와 같음)
  python - "$1" "$2" <<'PYEOF'
import sys, time
from pymilvus import MilvusClient as M
act, name = sys.argv[1], sys.argv[2]
c = M(uri="http://localhost:19530", db_name="junlim")
if act == "release":
    c.release_collection(name); print("released", name, c.get_load_state(name), flush=True)
else:
    idx = c.describe_index(name, "vec")
    if str(idx.get("mmap.enabled", "False")).lower() == "true":
        raise SystemExit(f"{name}: index mmap is ON: run `python set_mmap.py off` first")
    if "Loaded" not in str(c.get_load_state(name)):
        t = time.time(); c.load_collection(name); print(f"loaded {name} in {time.time() - t:.1f}s", flush=True)
    print(name, c.get_load_state(name), "index mmap:", idx.get("mmap.enabled", "False"), flush=True)
PYEOF
}
cleanup() {
  trap '' INT TERM HUP PIPE
  echo "$(ts) cleanup: reset"; $SW_CMD reset
  coll release nq_seq || echo "nq_seq release 실패"
  tgz="$ROOT/c3p_verify_$(date +%m%d_%H%M).tgz"
  tar czf "$tgz" --exclude="*.tgz" -C "$(dirname "$OUT")" "$(basename "$OUT")" && echo "$(ts) 결과 묶음: $tgz ($(du -h "$tgz" | cut -f1))"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

$SW_CMD reset
coll load nq_seq || { echo "$(ts) STOP: nq_seq 준비 실패"; exit 1; }
sleep 20
echo "$(ts) ===== verify"
env $NQENV python c3_pipeline.py --mode verify --outdir "$OUT" --cap "$CAP" --setting "$SETTING" --qset "$QSET" --vq "$VQ" --n "$N"
rc=$?
echo "$(ts) verify 종료 코드 $rc (0 = 같은 계획 입력 검증 PASS, 2 = FAIL)"
echo "$(ts) done run_c3p verify"
