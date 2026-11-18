#!/bin/bash
# usage: cd /workspace/junlim && tmux new -d -s pfwait 'bash wait_run_pf.sh > wait.log 2>&1'
#        다른 측정 스크립트: RUN=run_pf2.sh LOG=pf2.log (기본 run_pf.sh, pf.log)
# 다른 팀원이 Milvus를 다 쓸 때까지 기다렸다가 $RUN을 돌린다.
#   - INTERVAL초마다 확인해서 연속 QUIET_N번 조용하면 시작한다 (기본 10분 간격 3번 = 처음 조용한 걸 본 뒤 20분 동안 계속 조용할 때)
#     조용함 = check_busy.sh가 "조용함" + 다른 계정의 python 프로세스 없음 + 메모리 제한이 걸려 있지 않음
#   - MAX_HOURS시간 안에 조용해지지 않으면 아무것도 하지 않고 끝낸다
#   - 측정 중에는 20초마다 다른 계정의 새 프로세스를 확인해서, 생기면 wait.log에 OVERLAP으로 남긴다
#   - 끝나면 check_busy.sh를 한 번 더 돌려 post_check.txt에 남긴다
# 서버 시각은 UTC다 (한국 시각 = +9시간).
ROOT=${ROOT:-/workspace/junlim}
INTERVAL=${INTERVAL:-600}; QUIET_N=${QUIET_N:-3}; MAX_HOURS=${MAX_HOURS:-6}
RUN=${RUN:-run_pf.sh}; LOG=${LOG:-pf.log}
SW_CMD=${SW_CMD:-"timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 capvdb@203.253.145.110"}
cd "$ROOT" || exit 1
ME=$(id -un)
ts() { date '+%m-%d %H:%M:%S'; }
# 다른 계정의 프로세스 중 셸, 접속, tmux를 뺀 것 (= 실제로 무언가 돌리는 것)
others() {
  if [ -n "$FAKE_OTHERS" ]; then cat "$FAKE_OTHERS"; return; fi
  ps -eo user:12,pid,etime,args --no-headers | awk -v me="$ME" '$1 != me && $1 != "root"' \
    | grep -v -E ' (-bash|bash|sh|sshd: .*|tmux .*)$'
}

echo "$(ts) wait_run_pf 시작 ($RUN): ${INTERVAL}초마다 확인, 연속 ${QUIET_N}번 조용하면 측정, 최대 ${MAX_HOURS}시간"
deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
quiet=0
while :; do
  why=""
  bash check_busy.sh > last_check.txt 2>&1 || why="check_busy: $(tail -n 1 last_check.txt)"
  py=$(others | grep -i python)
  [ -n "$py" ] && why="$why / 다른 계정 python: $(echo "$py" | head -2 | cut -c1-120 | tr '\n' ' ')"
  $SW_CMD status 2>/dev/null | grep -q "memory.high max" || why="$why / 메모리 제한이 걸려 있거나 스위치 확인 실패"
  if [ -z "$why" ]; then
    quiet=$((quiet + 1)); echo "$(ts) 조용함 ($quiet/$QUIET_N)"
  else
    quiet=0; echo "$(ts) 기다림:$why"
  fi
  [ "$quiet" -ge "$QUIET_N" ] && break
  [ "$(date +%s)" -ge "$deadline" ] && { echo "$(ts) GIVEUP: ${MAX_HOURS}시간 동안 조용해지지 않아 측정하지 않고 끝냄"; exit 1; }
  sleep "$INTERVAL"
done

others > others_before.txt
echo "$(ts) 측정 시작 ($RUN, 로그 $LOG). 시작 때 다른 계정 프로세스:"; sed 's/^/  /' others_before.txt
bash "$RUN" > "$LOG" 2>&1 < /dev/null &
pid=$!
overlap=0
while kill -0 "$pid" 2>/dev/null; do
  sleep 20
  new=$(others | awk -v f=others_before.txt 'BEGIN {while ((getline l < f) > 0) {split(l, x, " "); seen[x[2]] = 1}} !($2 in seen)')
  if [ -n "$new" ]; then
    overlap=1; echo "$(ts) OVERLAP: 측정 중에 다른 계정 프로세스가 새로 생김"; echo "$new" | cut -c1-150 | sed 's/^/  /'
    others > others_before.txt
  fi
done
wait "$pid"; rc=$?
echo "$(ts) $RUN 끝 (종료 코드 $rc). $LOG 마지막 줄: $(tail -n 1 "$LOG")"
bash check_busy.sh > post_check.txt 2>&1; echo "$(ts) 끝난 뒤 확인: $(tail -n 1 post_check.txt)"
if [ "$overlap" = 1 ]; then
  echo "$(ts) 주의: 측정 중에 다른 계정이 무언가를 돌렸음 -> 결과가 오염됐을 수 있음"
else
  echo "$(ts) 측정 중에 다른 계정의 새 프로세스는 없었음"
fi
