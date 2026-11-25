#!/bin/bash
# 실제 질문(QSET=real) 탐색 재측정 (11/23 추가). run_c3a.sh(4cd64fca)의 explore 부분을 복사하고 기준 설정 측정만 더했다. run_c3a.sh는 고치지 않음.
# 바뀐 점 (run_c3a.sh explore 대비)
#   - 기준 설정(A, k=8)을 1회차 앞, 각 회차 뒤에 한 번씩 잰다 (phase=anchor, rep 0~3, 4번, 약 2분 추가). 측정 중 서버 상태가 바뀌었는지 보는 용도.
#   - 기본값: OUT=/workspace/junlim/c3a_out_real2, QSET=real, 로그 c3a_real2.log (이전 탐색 로그에 이어 쓰지 않음)
#   - confirm 단계 없음
# 판정: python judge_c3a_real2.py --outdir /workspace/junlim/c3a_out_real2 --waitlog c3a_real2_wait.log --runlog c3a_real2.log (규칙은 그 파일 맨 위, 측정 전 고정)
# usage (서버, /workspace/junlim):
#   tmux new -d -s c3r2wait 'RUN=run_c3a_real2.sh LOG=c3a_real2_run.log bash wait_run_pf.sh > c3a_real2_wait.log 2>&1'
#   BACKEND=mock OUT=/tmp/c3a_mock_real2 DUR=1 WARM=0.3 SEQ_REQS=10 EREPS=1 bash run_c3a_real2.sh   # 도구 점검 (Milvus를 건드리지 않음)
ROOT=${ROOT:-/workspace/junlim}
BACKEND=${BACKEND:-milvus}
PHASE=explore
EREPS=${EREPS:-3}; SEQ_REPS=${SEQ_REPS:-3}; SEQ_REQS=${SEQ_REQS:-400}
DUR=${DUR:-20}; WARM=${WARM:-5}; N=${N:-64}
QSET=${QSET:-real}; NSUB=${NSUB:-10000}
OUT=${OUT:-$ROOT/c3a_out_real2}
ANCHOR_SET=A; ANCHOR_K=8
SW_CMD=${SW_CMD:-"timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=10 <switch-user>@<switch-host>"}
cd "$ROOT" || exit 1
[ -f venv/bin/activate ] && source venv/bin/activate
NQENV="BENCH_DATA=$ROOT/data_nq BENCH_COLL=nq"
mkdir -p "$OUT"
DEST=$ROOT; [ "$BACKEND" != "milvus" ] && DEST=$(dirname "$OUT")   # mock: 기록과 묶음을 OUT 밖(같은 상위 폴더)에 둔다 (OUT 안에 두면 tar가 자기 파일을 읽다 경고)
LOG=$DEST/c3a_real2.log
exec > >(trap '' INT TERM HUP; exec tee -a "$LOG") 2>&1

ts() { date '+%H:%M:%S'; }
mock() { [ "$BACKEND" != "milvus" ]; }
sw() { mock && return 0; echo "$(ts) switch: $*"; $SW_CMD "$*"; }
status() { mock && return 0; $SW_CMD status 2>&1; }
health() { mock && return 0; curl -s -m 3 localhost:9091/healthz | grep -q OK; }
stop() { echo "$(ts) STOP: $1"; exit 1; }
bc() { health || stop "Milvus healthz 실패"; env $NQENV python bench_c3a.py --backend "$BACKEND" --outdir "$OUT" "$@" || stop "실패한 측정: bench_c3a $*"; }
memsnap() { mock && return 0; status | grep -E "memory.current|^ *anon|^ *file |oom" | tr -s ' ' | tr '\n' ' '; }
load1() { bc --mode load --n "$N" --dur "$DUR" --warm "$WARM" "$@"; echo "$(ts) mem: $(memsnap)"; }
anchor() { echo "$(ts) ===== 기준 설정 $ANCHOR_SET k=$ANCHOR_K (anchor rep $1)"; load1 --phase anchor --rep "$1" --slot 0 --setting "$ANCHOR_SET" --cap "$ANCHOR_K"; }
coll() {  # coll load|release 이름 (run_c3a.sh와 같음)
  mock && return 0
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
  echo "$(ts) cleanup: reset"; sw reset
  coll release nq_seq || echo "nq_seq release 실패"
  status | head -4
  cp -f "$LOG" "$OUT/"
  tgz="$DEST/c3a_real2_$(date +%m%d_%H%M).tgz"
  tar czf "$tgz" --exclude="*.tgz" -C "$(dirname "$OUT")" "$(basename "$OUT")" && echo "$(ts) 결과 묶음: $tgz ($(du -h "$tgz" | cut -f1))"
}

if ! mock && [ "${SKIP_BUSY_CHECK:-0}" != "1" ]; then
  bash check_busy.sh || { echo "$(ts) STOP: 다른 사람이 쓰는 중이거나 확인하지 못함. 아무것도 바꾸지 않고 끝냄"; exit 1; }
fi
[ -d "$ROOT/data_nq" ] || stop "$ROOT/data_nq 없음"
grep -qs ',explore,' "$OUT/results_c3a_load_nq.csv" && stop "$OUT에 탐색 결과가 이미 있음. 옮기거나 OUT=다른폴더 로 실행"

trap cleanup EXIT
trap 'exit 130' INT TERM HUP

echo "$(ts) start run_c3a_real2 backend=$BACKEND out=$OUT n=$N dur=$DUR warm=$WARM ereps=$EREPS qset=$QSET nsub=$NSUB anchor=$ANCHOR_SET/k$ANCHOR_K"
md5sum bench_c3a.py analyze_c3a.py bench_v2.py run_c3a_real2.sh judge_c3a_real2.py 2>/dev/null | cut -c1-8,33-
env $NQENV python bench_c3a.py --backend mock --selftest --outdir "$OUT/selftest" > "$OUT/selftest.txt" 2>&1 || stop "bench_c3a selftest 실패 ($OUT/selftest.txt)"
tail -n 1 "$OUT/selftest.txt"
health || stop "시작 전 Milvus healthz 실패"
sw reset

echo "$(ts) ===== nq_seq 준비"
coll load nq_seq || stop "nq_seq 준비 실패"
mock || sleep 20
status | head -12

if [ ! -s "$OUT/fixed_nq.npz" ]; then
  echo "$(ts) ===== 질문별 PK 목록 만들기 (설정 A로 검색, 측정 아님) + 캐시 데우기"
  bc --mode prep --qset "$QSET" --nsub "$NSUB"
else
  echo "$(ts) ===== 저장한 PK 목록으로 캐시 데우기"
  bc --mode warmup
fi

echo "$(ts) ===== 요청 하나씩: 같은 질문으로 설정 A/B/C를 번갈아 ($SEQ_REPS회 × $SEQ_REQS질문)"
for rep in $(seq 1 $SEQ_REPS); do bc --mode seq --rep $rep --reqs "$SEQ_REQS"; done

echo "$(ts) ===== 탐색: 설정 3개 × k 4개 × $EREPS회 (회차마다 설정 순서와 k 순서를 돌림, run_c3a.sh와 같음) + 기준 설정 ${EREPS}+1번"
SETS=(A B C); KS=(6 8 12 16)
anchor 0
for rep in $(seq 1 $EREPS); do
  slot=0
  for i in 0 1 2; do
    s=${SETS[$(( (i + rep - 1) % 3 ))]}
    for j in 0 1 2 3; do
      k=${KS[$(( (j + rep - 1) % 4 ))]}
      slot=$((slot + 1))
      load1 --phase explore --rep $rep --slot $slot --setting $s --cap $k
    done
  done
  anchor $rep
done
python analyze_c3a.py seq --outdir "$OUT"
python analyze_c3a.py explore --outdir "$OUT"
status | head -12
echo "$(ts) done run_c3a_real2"
