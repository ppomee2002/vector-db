#!/bin/bash
# usage (서버, /workspace/junlim):
#   bash run_c3a.sh explore    # 탐색 세션 (약 20분): 준비 → 요청 하나씩 → 설정 A/B/C × k 6/8/12/16 × 3회 → 비교용 k 고정
#   bash run_c3a.sh confirm    # 확인 세션 (약 13분): 기준 설정 → 비교용 k vs k=12 × 4회 → 기준 설정 → 판정
#   두 세션은 다른 시간대에 따로 돌리는 것을 권장 (확인이 탐색과 독립이 되도록). tmux 안에서 실행할 것.
#   BACKEND=mock 이면 서버 없이 도구만 점검한다.
#
# C3 13주차: 검색 설정이 바뀌면 m7의 알맞은 상한 k가 달라지는가 (설계·판정 규칙은 README.md, analyze_c3a.py 참고)
#   A top-10/ef64 (지난주), B top-10/ef100, C top-100/ef100. 본문은 세 설정 모두 질문별로 미리 저장한 같은 PK 목록을 읽는다.
#   NQ(nq_seq, 64단어), 평소 상태(캐시 따뜻함), 동시 요청 64, 예열 5초 + 측정 20초.
# 시작 전에 check_busy.sh로 다른 팀원이 쓰는 중인지 보고, 쓰는 중이면 아무것도 바꾸지 않고 끝낸다.
# 측정이 실패하거나 Milvus 상태(healthz)가 이상하면 멈춘다. 끝나면 항상 reset, nq_seq release, 결과 묶기.
ROOT=${ROOT:-/workspace/junlim}
BACKEND=${BACKEND:-milvus}
PHASE=$1
EREPS=${EREPS:-3}; CREPS=${CREPS:-4}; SEQ_REPS=${SEQ_REPS:-3}; SEQ_REQS=${SEQ_REQS:-400}
DUR=${DUR:-20}; WARM=${WARM:-5}; N=${N:-64}
QSET=${QSET:-all}; NSUB=${NSUB:-10000}   # prep에서 쓸 질문: query_vecs 전체(실제+가짜)에서 고정 시드로 NSUB개. 지난주 m7도 전체에서 뽑았음
OUT=${OUT:-$ROOT/c3a_out}
SW_CMD=${SW_CMD:-"timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=10 <switch-user>@<switch-host>"}
cd "$ROOT" || exit 1
[ -f venv/bin/activate ] && source venv/bin/activate
NQENV="BENCH_DATA=$ROOT/data_nq BENCH_COLL=nq"
case "$PHASE" in explore|confirm) ;; *) echo "usage: bash run_c3a.sh explore|confirm"; exit 1;; esac
mkdir -p "$OUT"
DEST=$ROOT; [ "$BACKEND" != "milvus" ] && DEST=$OUT   # mock 점검의 기록과 묶음은 OUT 안에만 남긴다
LOG=$DEST/c3a_$PHASE.log
# tee는 Ctrl+C·종료 신호를 무시한다: 같이 죽으면 정리 단계 출력이 끊긴 파이프로 가서 정리가 중간에 죽는다 (11/22 확인)
exec > >(trap '' INT TERM HUP; exec tee -a "$LOG") 2>&1

ts() { date '+%H:%M:%S'; }
mock() { [ "$BACKEND" != "milvus" ]; }
sw() { mock && return 0; echo "$(ts) switch: $*"; $SW_CMD "$*"; }
status() { mock && return 0; $SW_CMD status 2>&1; }
health() { mock && return 0; curl -s -m 3 localhost:9091/healthz | grep -q OK; }
stop() { echo "$(ts) STOP: $1"; exit 1; }
bc() { health || stop "Milvus healthz 실패"; env $NQENV python bench_c3a.py --backend "$BACKEND" --outdir "$OUT" "$@" || stop "실패한 측정: bench_c3a $*"; }
# 구간마다 Milvus 메모리와 OOM 기록을 남긴다 (11/22 00:04 Milvus가 측정 중 꺼진 뒤 원인 확인용으로 추가)
memsnap() { mock && return 0; status | grep -E "memory.current|^ *anon|^ *file |oom" | tr -s ' ' | tr '\n' ' '; }
load1() { bc --mode load --n "$N" --dur "$DUR" --warm "$WARM" "$@"; echo "$(ts) mem: $(memsnap)"; }
coll() {  # coll load|release 이름 (run_bc1b.sh와 같음)
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
  trap '' INT TERM HUP PIPE   # 정리 중에는 다시 끊기지 않게
  echo "$(ts) cleanup: reset"; sw reset
  coll release nq_seq || echo "nq_seq release 실패"
  status | head -4
  [ "$DEST" != "$OUT" ] && cp -f "$LOG" "$OUT/"
  tgz="$DEST/c3a_${PHASE}_$(date +%m%d_%H%M).tgz"
  tar czf "$tgz" --exclude="*.tgz" -C "$(dirname "$OUT")" "$(basename "$OUT")" && echo "$(ts) 결과 묶음: $tgz ($(du -h "$tgz" | cut -f1))"
}

if ! mock && [ "${SKIP_BUSY_CHECK:-0}" != "1" ]; then
  bash check_busy.sh || { echo "$(ts) STOP: 다른 사람이 쓰는 중이거나 확인하지 못함. 아무것도 바꾸지 않고 끝냄"; exit 1; }
fi
[ -d "$ROOT/data_nq" ] || stop "$ROOT/data_nq 없음"
mkdir -p "$OUT"
if [ "$PHASE" = "explore" ]; then
  grep -qs ',explore,' "$OUT/results_c3a_load_nq.csv" && stop "$OUT에 탐색 결과가 이미 있음. 옮기거나 OUT=다른폴더 로 실행"
else
  [ -s "$OUT/ref_k_nq.json" ] || stop "$OUT/ref_k_nq.json 없음 (탐색 세션을 먼저)"
  grep -qs ',confirm,' "$OUT/results_c3a_load_nq.csv" && stop "확인 결과가 이미 있음. 옮기거나 OUT=다른폴더 로 실행"
fi

trap cleanup EXIT
trap 'exit 130' INT TERM HUP

echo "$(ts) start run_c3a $PHASE backend=$BACKEND out=$OUT n=$N dur=$DUR warm=$WARM ereps=$EREPS creps=$CREPS qset=$QSET nsub=$NSUB"
md5sum bench_c3a.py analyze_c3a.py bench_v2.py run_c3a.sh | cut -c1-8,33-
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

if [ "$PHASE" = "explore" ]; then
  echo "$(ts) ===== 요청 하나씩: 같은 질문으로 설정 A/B/C를 번갈아 ($SEQ_REPS회 × $SEQ_REQS질문)"
  for rep in $(seq 1 $SEQ_REPS); do bc --mode seq --rep $rep --reqs "$SEQ_REQS"; done

  echo "$(ts) ===== 탐색: 설정 3개 × k 4개 × $EREPS회 (회차마다 설정 순서와 k 순서를 돌림)"
  SETS=(A B C); KS=(6 8 12 16)
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
  done
  python analyze_c3a.py seq --outdir "$OUT"
  python analyze_c3a.py explore --outdir "$OUT"
else
  REF=$(python -c "import json; d=json.load(open('$OUT/ref_k_nq.json')); print(' '.join(f'{s}:{k}' for s, k in sorted(d.items()) if int(k) != 12))")
  echo "$(ts) 비교용 k (탐색 후 고정): $(cat "$OUT/ref_k_nq.json")  -> 확인할 설정: ${REF:-없음}"
  if [ -z "$REF" ]; then
    echo "$(ts) 모든 설정에서 탐색 최고가 k=12. 확인 측정 없이 끝냄"; exit 0
  fi
  read -ra PAIRS <<< "$REF"
  echo "$(ts) ===== 기준 설정 (처음): A, k=8"
  load1 --phase anchor --rep 0 --slot 0 --setting A --cap 8
  echo "$(ts) ===== 확인: 비교용 k vs k=12 × $CREPS회 (설정 순서는 회차마다 돌리고, 두 k의 순서는 회차마다 뒤집음)"
  np=${#PAIRS[@]}
  for rep in $(seq 1 $CREPS); do
    slot=0
    for i in $(seq 0 $((np - 1))); do
      p=${PAIRS[$(( (i + rep - 1) % np ))]}; s=${p%%:*}; k=${p##*:}
      if [ $((rep % 2)) = 1 ]; then ord="$k 12"; else ord="12 $k"; fi
      for kk in $ord; do
        slot=$((slot + 1))
        load1 --phase confirm --rep $rep --slot $slot --setting $s --cap $kk
      done
    done
  done
  echo "$(ts) ===== 기준 설정 (끝): A, k=8"
  load1 --phase anchor --rep 99 --slot 0 --setting A --cap 8
  python analyze_c3a.py confirm --outdir "$OUT" --reps "$CREPS"
fi
status | head -12
echo "$(ts) done run_c3a $PHASE"
