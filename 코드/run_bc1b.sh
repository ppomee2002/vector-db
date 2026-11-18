#!/bin/bash
# usage: cd /workspace/junlim && tmux new -d -s bc1bwait 'RUN=run_bc1b.sh LOG=bc1b.log INTERVAL=180 QUIET_N=2 bash wait_run_pf.sh > wait_bc1b.log 2>&1'
#        (wait_run_pf.sh가 다른 팀원이 조용해질 때까지 기다렸다가 이 스크립트를 부른다)
# C2: 규칙 B+C1(정책 rBC1)의 처리량을 다시 잰다 (run_bc1.sh의 재측정).
#   11/17 01:47 run_bc1.sh에서 top-10만(none)이 회차마다 516~688로 흔들렸고, rBC1은 늘 rB 바로 뒤 순서였다.
#   그래서 이번에는 (a) 순서 영향을 지우려고 정책 순서를 회차마다 바꿔 각 정책이 모든 자리와 모든 앞 정책을 한 번씩 갖게 하고(4×4 균형 순서),
#   (b) 시간에 따른 흔들림과 상관없는 비교로, 같은 질문에서 정책을 번갈아 재는 요청 하나씩 측정(seq)을 먼저 한다.
#   조건은 11/16 pf3 load와 같다: nq_seq(평소 상태), 동시 요청 64, 요청 상한 8, 예열 5초 + 측정 20초, 3회, 회차마다 정책 순서 회전.
#   11/16 결과(pf3_out)는 건드리지 않는다. 필요한 입력 3개(조각 경계, 필요 여부, 정답 질문 목록)만 새 폴더($OUT)로 복사해서 쓴다.
#   1) 캐시 데우기: 11/16과 같은 방식(실제 질문 3,452개의 결과와 이웃 전부를 한 번 읽음, 약 35초)
#   2) seq: 같은 질문으로 정책 9개를 번갈아, 회차마다 400질문 × 3회 (약 2분, 11/16 pf3 seq와 같은 방식)
#   3) load: 정책 4개 × 4회, 균형 순서 (약 7분)
# 시작 전에 check_busy.sh로 다른 팀원이 쓰는 중인지 보고, 쓰는 중이면 아무것도 바꾸지 않고 끝낸다.
# 측정이 실패하거나 Milvus 상태(healthz)가 이상하면 멈춘다. 끝나면 항상 reset, nq_seq release, 결과 묶기.
ROOT=${ROOT:-/workspace/junlim}
BACKEND=${BACKEND:-milvus}      # 도구 점검용 mock
REPS=${REPS:-3}; LREPS=${LREPS:-4}; SEQ_REQS=${SEQ_REQS:-400}
DUR=${DUR:-20}; WARM=${WARM:-5}; N=${N:-64}; CAP=${CAP:-8}
SRC=${SRC:-$ROOT/pf3_out}
OUT=${OUT:-$ROOT/pf4b_out}
SW_CMD=${SW_CMD:-"timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=10 capvdb@203.253.145.110"}
cd "$ROOT" || exit 1
[ -f venv/bin/activate ] && source venv/bin/activate
NQENV="BENCH_DATA=$ROOT/data_nq BENCH_COLL=nq"

ts() { date '+%H:%M:%S'; }
sw() { echo "$(ts) switch: $*"; $SW_CMD "$*"; }
status() { $SW_CMD status 2>&1; }
health() { [ "$BACKEND" != "milvus" ] && return 0; curl -s -m 3 localhost:9091/healthz | grep -q OK; }
stop() { echo "$(ts) STOP: $1"; exit 1; }
b4() { health || stop "Milvus healthz 실패"; env $NQENV python bench_pf4.py --backend "$BACKEND" --gold "$OUT" "$@" || stop "실패한 측정: bench_pf4 $*"; }
coll() {  # coll load|release 이름
  [ "$BACKEND" != "milvus" ] && return 0
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
  echo "$(ts) cleanup: reset"; $SW_CMD reset
  coll release nq_seq || echo "nq_seq release 실패"
  status | head -4
  cp -f "$ROOT/bc1b.log" "$OUT/" 2>/dev/null
  tgz="$ROOT/bc1b_results_$(date +%m%d_%H%M).tgz"
  tar czf "$tgz" -C "$ROOT" "$(basename "$OUT")" && echo "$(ts) 결과 묶음: $tgz ($(du -h "$tgz" | cut -f1))"
}

# 다른 팀원이 쓰는 중이면 아무것도 바꾸지 않고 끝낸다 (reset, release를 하기 전에 확인)
if [ "$BACKEND" = "milvus" ] && [ "${SKIP_BUSY_CHECK:-0}" != "1" ]; then
  bash check_busy.sh || { echo "$(ts) STOP: 다른 사람이 쓰는 중이거나 확인하지 못함. 아무것도 바꾸지 않고 끝냄 (위 결과 참고)"; exit 1; }
fi

# 입력 확인과 복사 (아직 Milvus는 건드리지 않음)
[ -d "$ROOT/data_nq" ] || stop "$ROOT/data_nq 없음"
mkdir -p "$OUT"
for f in nq_flags.npy need_nq.json nq_gold.jsonl; do
  [ -s "$SRC/$f" ] || stop "$SRC/$f 없음 (11/16 pf3_out이 그대로 있어야 함)"
  cp -f "$SRC/$f" "$OUT/"
done
{ [ -e "$OUT/results_pf3_load_nq.csv" ] || [ -e "$OUT/results_pf3_seq_nq.csv" ]; } && stop "$OUT/results_pf3_load_nq.csv가 이미 있음. 이전 결과를 옮긴 뒤 다시 실행 (OUT=다른폴더 로 바꿔도 됨)"

trap cleanup EXIT
trap 'exit 130' INT TERM

echo "$(ts) start run_bc1b seq_reps=$REPS load_reps=$LREPS backend=$BACKEND src=$SRC out=$OUT n=$N cap=$CAP dur=$DUR warm=$WARM"
md5sum bench_pf4.py bench_pf3.py pf3_rules.py bench_v2.py run_bc1b.sh | cut -c1-8,33-
python pf3_rules.py || stop "pf3_rules selftest 실패"
env $NQENV python bench_pf4.py --backend mock --selftest --gold "$OUT" > "$OUT/selftest_pf4.txt" 2>&1 || stop "bench_pf4 selftest 실패 ($OUT/selftest_pf4.txt)"
tail -n 2 "$OUT/selftest_pf4.txt"
health || stop "시작 전 Milvus healthz 실패"
sw reset

echo "$(ts) ===== 1) nq_seq 준비와 캐시 데우기 (11/16과 같은 방식)"
coll load nq_seq || stop "nq_seq 준비 실패"
[ "$BACKEND" = "milvus" ] && sleep 20
status | head -12
b4 --mode hits --ds nq

echo "$(ts) ===== 2) seq (요청을 하나씩, 같은 질문으로 정책 9개를 번갈아)"
for rep in $(seq 1 $REPS); do
  b4 --mode seq --ds nq --cond bc1_seq --rep $rep --reqs "$SEQ_REQS"
done

echo "$(ts) ===== 3) load (동시 요청 $N, 상한 $CAP, 정책 none all rB rBC1, 4×4 균형 순서)"
# 회차별 순서: 각 정책이 1~4번째 자리에 한 번씩, 다른 세 정책 바로 뒤에 한 번씩 온다
ORD=("none all rB rBC1" "all rBC1 none rB" "rB none rBC1 all" "rBC1 rB all none")
for rep in $(seq 1 $LREPS); do
  for pol in ${ORD[$(( (rep - 1) % 4 ))]}; do
    b4 --mode load --ds nq --cond bc1b_load --rep $rep --policy $pol --n $N --cap $CAP --dur $DUR --warm $WARM
  done
done
status | head -12
echo "$(ts) done run_bc1b"
