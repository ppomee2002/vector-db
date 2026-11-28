#!/bin/bash
# C3 14주차 A 실험 (규칙: C3_A실험_규칙.md, 2026-11-28 측정 전 고정)
# 3조건(A1 rand+none, A2 doc+none, A3 doc+BC1) × clients 1/4/6/8/12/16 × 3회, C3는 모두 m3(m3trace_junlim).
# 회차마다 18칸 순서를 seed(회차 번호)로 섞는다. 팀 main.py가 실행마다 MilvusGuard로 load → 측정 → release.
# 팀 코드·기존 파일은 고치지 않는다(PC = 팀 e36f40d 사본 + modules/c3_m3trace.py).
#
# 필수: WINDOW="2026-11-28 21:00~2026-11-28 22:30" (KST, main.py run.reserved_window 형식)
# 선택: ROOT=/workspace/junlim PC=$ROOT/c3a2/pipeline-core PY=$ROOT/venv/bin/python SESS=$ROOT/c3a2_runs/<시각>
#       REPS=3  SKIP_BUSY_CHECK=0  DRY=1(명령만 출력, Milvus·스위치 안 건드림)
# 실행(서버): tmux new -d -s c3a2 'WINDOW="..." bash /workspace/junlim/c3a2/run_c3a2.sh'  → tail -F <SESS>/run.log
# 멈추기: tmux send-keys -t c3a2 C-c (지금 실행이 끝난 뒤 정리)
ROOT=${ROOT:-/workspace/junlim}
PC=${PC:-$ROOT/c3a2/pipeline-core}
PY=${PY:-$ROOT/venv/bin/python}
REPS=${REPS:-3}
SESS=${SESS:-$ROOT/c3a2_runs/$(date +%m%d_%H%M)}
DRY=${DRY:-0}
SW_CMD=${SW_CMD:-"timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=10 <switch-user>@<switch-host>"}
HERE=$(cd "$(dirname "$0")" && pwd)

[ -n "$WINDOW" ] || { echo "WINDOW(예약 시간대) 없음"; exit 2; }
[ -f "$PC/main.py" ] || { echo "$PC/main.py 없음"; exit 2; }
[ -f "$PC/modules/c3_m3trace.py" ] || { echo "$PC/modules/c3_m3trace.py 없음"; exit 2; }
[ -e "$SESS" ] && { echo "$SESS 가 이미 있음"; exit 2; }
mkdir -p "$SESS/out" "$SESS/trace"
LOG=$SESS/run.log
exec > >(trap '' INT TERM HUP; exec tee -a "$LOG") 2>&1

ts() { date '+%H:%M:%S'; }
dry() { [ "$DRY" = "1" ]; }
status_to() {
  dry && { echo "(dry) status" > "$1"; return 0; }
  $SW_CMD status > "$1" 2>&1
  grep -q "memory.high" "$1" && return 0
  sleep 5; $SW_CMD status > "$1" 2>&1
}
health() { dry && return 0; curl -s -m 3 localhost:9091/healthz | grep -q OK; }
stop() { echo "$(ts) STOP: $1"; exit 1; }
REC_PID=""
cleanup() {
  trap '' INT TERM HUP PIPE
  [ -n "$REC_PID" ] && kill "$REC_PID" 2>/dev/null
  status_to "$SESS/status_end.txt"
  tgz="$ROOT/c3a2_$(date +%m%d_%H%M).tgz"
  dry || { tar czf "$tgz" -C "$(dirname "$SESS")" "$(basename "$SESS")" && echo "$(ts) 결과 묶음: $tgz ($(du -h "$tgz" | cut -f1))"; }
}

declare -A LAYOUT=([A1]=rand [A2]=doc [A3]=doc)
declare -A POLICY=([A1]=none [A2]=none [A3]=BC1)
CLIENTS_LIST="1 4 6 8 12 16"

run1() {  # run1 <rep> <cond> <clients>
  local rep=$1 cond=$2 n=$3 label="r$1_$2_c$3"
  health || stop "실행 전 Milvus healthz 실패 ($label)"
  echo "$label" > "$SESS/label"
  local args=(--set "run.name=$label" --set "run.out_dir=$SESS/out" --set "run.reserved_window=\"$WINDOW\""
              --set run.purpose=measure --set run.query_set=nq_test_real --set run.n_queries=null --set run.max_queries=null
              --set run.evaluate=false --set experiment.cache_condition=warm
              --set "run.milvus_allowed_load=[junlim/nq_seq, jihyeon/w13c1_doc, jihyeon/w13c1_rand]"
              --set c1.impl=layout_jihyeon --set c1.backend=milvus --set "c1.layout=${LAYOUT[$cond]}"
              --set c1.fetch_mode=batch --set c1.batch_impl=query_pk_in
              --set c2.impl=bc1_v1 --set "c2.policy=${POLICY[$cond]}"
              --set c3.impl=m3trace_junlim --set c3.mode=m3 --set "c3.clients=$n")
  echo "$(ts) ===== $label"
  local rc=0
  if dry; then echo "(dry) main.py ${args[*]}"; echo "out=$SESS/out/${label}_dry" > "$SESS/$label.log"
  else (cd "$PC" && C3_TRACE="$SESS/trace/$label.csv" "$PY" -B main.py "${args[@]}") > "$SESS/$label.log" 2>&1; rc=$?; fi
  local out; out=$(grep -m1 -o 'out=[^ ]*' "$SESS/$label.log" | cut -d= -f2)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$rep" "$cond" "$n" "$rc" "$out" "$(date '+%F %T')" >> "$SESS/runs.tsv"
  echo "$(ts) $label 종료 코드 $rc"
  [ "$rc" = "0" ] || stop "$label 종료 코드 $rc. 남은 실행 안 함 ($SESS/$label.log 확인. 6이면 수동 release 명령)"
  health || stop "실행 뒤 Milvus healthz 실패 ($label)"
}

if ! dry && [ "${SKIP_BUSY_CHECK:-0}" != "1" ]; then
  (cd "$ROOT" && bash check_busy.sh) || { echo "$(ts) STOP: 다른 사람이 쓰는 중이거나 확인 못 함. 아무것도 바꾸지 않음"; exit 1; }
fi
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

echo "$(ts) start run_c3a2 sess=$SESS window=$WINDOW reps=$REPS dry=$DRY"
md5sum "$HERE/run_c3a2.sh" "$HERE/rec_c3a2.py" "$HERE/c3_m3trace.py" "$PC/main.py" "$PC/modules/c3_m3trace.py" \
       "$PC/modules/c1_jihyeon.py" "$PC/modules/c2_bc1.py" 2>/dev/null | cut -c1-8,33-
health || stop "시작 전 Milvus healthz 실패"
status_to "$SESS/status_start.txt"; cat "$SESS/status_start.txt"
# 스위치 reset·제한 변경은 하지 않는다(읽기만). 제한이 걸려 있으면 사람이 보고 판단.
if ! dry; then "$PY" "$HERE/rec_c3a2.py" --out "$SESS/rec.csv" --label-file "$SESS/label" --interval 2 & REC_PID=$!; fi

for rep in $(seq 1 "$REPS"); do
  ORDER=$("${PY_ORDER:-$PY}" -c "
import random
cells=[(c,n) for c in ('A1','A2','A3') for n in (1,4,6,8,12,16)]
random.Random($rep).shuffle(cells)
print(' '.join(f'{c}:{n}' for c,n in cells))")
  echo "$(ts) 회차 $rep 순서: $ORDER"
  for cell in $ORDER; do
    run1 "$rep" "${cell%%:*}" "${cell##*:}"
  done
  status_to "$SESS/status_after_rep$rep.txt"
done
echo "$(ts) 전체 끝"
