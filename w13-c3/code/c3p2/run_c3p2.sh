#!/bin/bash
# C3 단독 효과 측정 (설계 v3 고정본: w13프로토타입\C3_파이프라인측정_설계_v3.md, 2026-11-23 고정)
# 팀 main.py(8c00e5d) + c3.impl=m7_junlim로 1단계 정확성 3번 → 판정 → 2단계 성능 6번(m3 m7 | m7 m3 | m3 m7).
# 팀 코드와 기존 파일은 고치지 않는다. main.py는 실행마다 MilvusGuard로 nq_seq를 load → 실행 → release 확인 후 release한다.
#
# 필수 환경 변수
#   WINDOW   예약 시간대(KST), main.py run.reserved_window 형식: "2026-11-24 14:00~2026-11-24 14:30"
# 선택 환경 변수 (기본값)
#   C2_IMPL=none_ref C2_POLICY=none   (BC1 어댑터가 오면 C2_IMPL=<이름> C2_POLICY=BC1, D5 주 결과)
#   CLIENTS=16 CAP=8  ROOT=/workspace/junlim  PC=$ROOT/c3_contract2/pipeline-core  PY=$ROOT/venv/bin/python
#   SESS=$ROOT/c3p2_runs/<시각>   SKIP_BUSY_CHECK=0   DRY=1 이면 명령만 출력(Milvus·스위치 안 건드림)
# 실행 (서버, /workspace/junlim)
#   tmux new -d -s c3p2 'WINDOW="2026-11-24 14:00~2026-11-24 14:30" bash run_c3p2.sh'  → tail -F 로 SESS/run.log
#   예약 시간대가 정해져 있어야 해서 wait_run_pf.sh(언제 시작할지 모름)는 쓰지 않는다. 시작할 때 check_busy만 본다.
#   그래서 V1은 main.py run_meta의 other_user_activity(실행 앞뒤 스냅숏)로만 본다(WAITLOG를 주면 그것도 봄).
# 판정: $PY judge_c3p2.py --sess <SESS>  (규칙은 그 파일 맨 위, 측정 전 고정)
ROOT=${ROOT:-/workspace/junlim}
PC=${PC:-$ROOT/c3_contract2/pipeline-core}
PY=${PY:-$ROOT/venv/bin/python}
C2_IMPL=${C2_IMPL:-none_ref}; C2_POLICY=${C2_POLICY:-none}
CLIENTS=${CLIENTS:-16}; CAP=${CAP:-8}
SESS=${SESS:-$ROOT/c3p2_runs/$(date +%m%d_%H%M)}
DRY=${DRY:-0}
SW_CMD=${SW_CMD:-"timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=10 <switch-user>@<switch-host>"}
HERE=$(cd "$(dirname "$0")" && pwd)

[ -n "$WINDOW" ] || { echo "WINDOW(예약 시간대) 없음. 예: WINDOW=\"2026-11-24 14:00~2026-11-24 14:30\""; exit 2; }
[ -f "$PC/main.py" ] || { echo "$PC/main.py 없음"; exit 2; }
[ -e "$SESS" ] && { echo "$SESS 가 이미 있음. SESS=다른폴더 로 실행"; exit 2; }
mkdir -p "$SESS/out"
LOG=$SESS/run.log
exec > >(trap '' INT TERM HUP; exec tee -a "$LOG") 2>&1

ts() { date '+%H:%M:%S'; }
dry() { [ "$DRY" = "1" ]; }
sw() { dry && { echo "(dry) switch $*"; return 0; }; echo "$(ts) switch: $*"; $SW_CMD "$*"; }
# status는 읽기만 하는 명령. 못 읽으면(memory.high 줄 없음) 5초 뒤 한 번 더 읽는다(설계 v3.1 실행 절차). 두 번 다 실패하면 그대로 둠 → judge가 "확인 못 함"
status_to() {
  dry && { echo "(dry) status" > "$1"; return 0; }
  $SW_CMD status > "$1" 2>&1
  grep -q "memory.high" "$1" && return 0
  echo "$(ts) status 읽기 실패($1) → 5초 뒤 한 번 더"; sleep 5
  $SW_CMD status > "$1" 2>&1
}
health() { dry && return 0; curl -s -m 3 localhost:9091/healthz | grep -q OK; }
stop() { echo "$(ts) STOP: $1"; exit 1; }
REC_PID=""
cleanup() {
  trap '' INT TERM HUP PIPE
  [ -n "$REC_PID" ] && kill "$REC_PID" 2>/dev/null
  echo "$(ts) cleanup: 스위치 status 기록"; status_to "$SESS/status_end.txt"
  echo "$(ts) (nq_seq release는 main.py MilvusGuard 몫. 종료 코드 6이면 run.log의 수동 명령 확인)"
  tgz="$ROOT/c3p2_$(date +%m%d_%H%M).tgz"
  tar czf "$tgz" -C "$(dirname "$SESS")" "$(basename "$SESS")" && echo "$(ts) 결과 묶음: $tgz ($(du -h "$tgz" | cut -f1))"
}

# run1 <label> <set...> : main.py 한 번. 앞뒤 스위치 status, 종료 코드·결과 폴더를 runs.tsv에 기록
run1() {
  local label=$1; shift
  health || stop "실행 전 Milvus healthz 실패 ($label). 그날 측정 끝 (설계 v3)"
  echo "$label" > "$SESS/label"
  status_to "$SESS/$label.status_before.txt"
  local args=(--set "run.name=$label" --set "run.out_dir=$SESS/out" --set "run.reserved_window=\"$WINDOW\""
              --set "run.milvus_allowed_load=[junlim/nq_seq]" --set "run.n_queries=null"
              --set "c1.impl=milvus_query_ref" --set "c2.impl=$C2_IMPL" --set "c2.policy=$C2_POLICY"
              --set "c3.clients=$CLIENTS" --set "c3.cap=$CAP" --set "experiment.cache_condition=warm" "$@")
  echo "$(ts) ===== $label: python main.py ${args[*]}"
  local rc=0
  if dry; then echo "(dry) out=$SESS/out/${label}_dry" > "$SESS/$label.log"
  else (cd "$PC" && "$PY" -B main.py "${args[@]}") > "$SESS/$label.log" 2>&1; rc=$?; fi
  status_to "$SESS/$label.status_after.txt"
  local out; out=$(grep -m1 -o 'out=[^ ]*' "$SESS/$label.log" | cut -d= -f2)
  printf '%s\t%s\t%s\t%s\n' "$label" "$rc" "$out" "$(date '+%F %T')" >> "$SESS/runs.tsv"
  tail -n 3 "$SESS/$label.log"
  echo "$(ts) $label 종료 코드 $rc"
  [ "$rc" = "6" ] && stop "$label: 자동 release 보류(종료 코드 6). $SESS/$label.log의 수동 release 명령 확인"
  health || stop "실행 뒤 Milvus healthz 실패 ($label). 그날 측정 끝 (설계 v3)"
  return "$rc"
}

correctness() {  # correctness <접두어> <query_set> <개수>
  local p=$1 qs=$2 n=$3
  local c=(--set run.purpose=correctness --set "run.query_set=$qs" --set "run.max_queries=$n"
           --set run.evaluate=true --set run.text_check_n=100000)
  run1 "${p}_serial" "${c[@]}" --set c3.impl=serial_ref --set c3.mode=m3
  run1 "${p}_m3" "${c[@]}" --set c3.impl=m7_junlim --set c3.mode=m3
  run1 "${p}_m7" "${c[@]}" --set c3.impl=m7_junlim --set c3.mode=m7
  return 0
}

if ! dry && [ "${SKIP_BUSY_CHECK:-0}" != "1" ]; then
  (cd "$ROOT" && bash check_busy.sh) || { echo "$(ts) STOP: 다른 사람이 쓰는 중이거나 확인하지 못함. 아무것도 바꾸지 않고 끝냄"; exit 1; }
fi
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

echo "$(ts) start run_c3p2 sess=$SESS window=$WINDOW c2=$C2_IMPL/$C2_POLICY clients=$CLIENTS cap=$CAP dry=$DRY"
md5sum "$HERE/run_c3p2.sh" "$HERE/rec_c3p2.py" "$HERE/judge_c3p2.py" "$PC/main.py" "$PC/modules/"c3_m7*.py 2>/dev/null | cut -c1-8,33-
health || stop "시작 전 Milvus healthz 실패"
# 스위치 reset은 하지 않는다: 스위치는 팀 공용 Milvus의 제한(mem, iops)을 바꾸므로 남이 건 제한까지 풀 수 있다.
# 설계 v3 사전 조건 5(제한 없음)는 측정 전에 사람이 status를 보고 확인한다. 여기서는 기록만 한다.
status_to "$SESS/status_start.txt"
cat "$SESS/status_start.txt"

# 측정 중 기록기 (2초마다 Milvus RSS·CPU, main.py 프로세스 RSS·CPU). 판정에 쓰지 않음
if ! dry; then "$PY" "$HERE/rec_c3p2.py" --out "$SESS/rec.csv" --label-file "$SESS/label" --interval 2 & REC_PID=$!; fi

# 1단계 정확성
correctness c nq_test_real 3452
"$PY" "$HERE/judge_c3p2.py" --sess "$SESS" --stage correctness --prefix c; jr=$?
if [ "$jr" = "10" ]; then
  echo "$(ts) evaluate 관련 실패 → nq_test_ans 2,254개로 1단계 다시 (설계 v3)"
  correctness ca nq_test_ans 2254
  "$PY" "$HERE/judge_c3p2.py" --sess "$SESS" --stage correctness --prefix ca; jr=$?
fi
[ "$jr" = "0" ] || stop "1단계 정확성 PASS 아님(판정 코드 $jr). 성능 측정 안 함"

# 2단계 성능: 쌍 (1,2) (3,4) (5,6), 순서 m3 m7 | m7 m3 | m3 m7
P=(--set run.purpose=measure --set run.query_set=nq_test_real --set run.max_queries=null --set run.evaluate=false
   --set c3.impl=m7_junlim)
i=0
for mode in m3 m7 m7 m3 m3 m7; do
  i=$((i + 1))
  run1 "p${i}_${mode}" "${P[@]}" --set "c3.mode=$mode" || stop "p${i}_${mode} 종료 코드가 0이 아님. 나머지 실행 안 함 (V3 위반)"
done
echo "$(ts) 2단계 끝"
"$PY" "$HERE/judge_c3p2.py" --sess "$SESS" --stage all --runlog "$LOG" ${WAITLOG:+--waitlog "$WAITLOG"}
