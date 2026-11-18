#!/bin/bash
# usage: nohup bash run_all.sh > all.log 2>&1 &
# NQ(nq_seq, 인덱스·벡터·본문 mmap)에서 조건을 차례로 측정한다. 조건마다 run_cond.sh nq <이름> std (약 11분).
#   1 base_mmap          제한 없음
#   2 mem<HI>            메모리 소프트 제한 HI
#   3 mem<LO>            메모리 소프트 제한 LO
#   4 mem<LO>_iops3000   + loop1 읽기 IOPS 3000 (Milvus만)
#   5 mem<LO>_iops1000   + IOPS 1000
#   6 cold (DO_COLD=1일 때만) 제한 해제 → Milvus 재시작 → 로드 → Milvus 캐시 비우기 → run_cond.sh cold
# 끝나거나 중간에 멈추면 reset으로 모든 제한을 푼다. 조건 하나라도 실패하면 거기서 전체를 멈춘다. 스위치가 거부되면 그 조건은 건너뛴다(잘못된 이름표로 기록하지 않음).
ROOT=${ROOT:-/workspace/junlim}
MEM_HI=${MEM_HI:-2850M}     # 약 2.78GiB = anon 1.55 + mmap 데이터 약 2.05GiB의 60%
MEM_LO=${MEM_LO:-2330M}     # 약 2.28GiB = anon 1.55 + 약 35% (하한 2.05GiB 위)
DO_COLD=${DO_COLD:-0}      # 1이면 마지막에 Milvus 재시작(같이 쓰는 Milvus에 영향, 앞 조건이 모두 성공했을 때만 도달)
SW_CMD=${SW_CMD:-"timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=10 capvdb@203.253.145.110"}
cd "$ROOT" || exit 1
[ -f venv/bin/activate ] && source venv/bin/activate

ts() { date '+%H:%M:%S'; }
sw() { echo "$(ts) switch: $*"; $SW_CMD "$*"; }
status() { $SW_CMD status 2>&1; }
health() { [ "${BACKEND:-milvus}" != "milvus" ] && return 0; curl -s -m 3 localhost:9091/healthz | grep -q OK; }
# 조건 하나라도 실패하면 전체를 멈춘다(끝에서 reset). 다음 조건(특히 재시작)으로 넘어가지 않는다.
run() {
  echo "$(ts) ===== $1 ($2)"
  if ! health; then echo "$(ts) STOP: Milvus healthz 실패, 전체 중단"; exit 1; fi
  bash run_cond.sh nq "$1" "$2" || { echo "$(ts) STOP: $1 실패, 전체 중단"; exit 1; }
}
mem_set() { status | grep -Eq "memory\.high [0-9]"; }
iops_set() { status | sed -n '/io.max/,/floor/p' | grep -q "$1"; }
cleanup() { echo "$(ts) cleanup: reset"; $SW_CMD reset; }
trap cleanup EXIT
trap 'exit 130' INT TERM

echo "$(ts) start run_all MEM_HI=$MEM_HI MEM_LO=$MEM_LO DO_COLD=$DO_COLD"
sw reset

run base_mmap std

sw mem "$MEM_HI"
if mem_set; then run "mem${MEM_HI}" std; else echo "$(ts) SKIP mem $MEM_HI (제한이 걸리지 않음)"; fi

sw mem "$MEM_LO"
if mem_set; then
  run "mem${MEM_LO}" std
  sw iops 3000
  if iops_set 3000; then run "mem${MEM_LO}_iops3000" std; else echo "$(ts) SKIP iops 3000"; fi
  sw iops 1000
  if iops_set 1000; then run "mem${MEM_LO}_iops1000" std; else echo "$(ts) SKIP iops 1000"; fi
else
  echo "$(ts) SKIP mem $MEM_LO 와 IOPS 조건 (메모리 제한이 걸리지 않음)"
fi

if [ "$DO_COLD" = "1" ]; then
  sw reset
  sw restart
  if [ "${BACKEND:-milvus}" = "milvus" ]; then
    echo "$(ts) Milvus가 다시 뜰 때까지 기다림 (최대 10분)"
    up=0
    for i in $(seq 1 60); do
      if python - <<'PYEOF' >/dev/null 2>&1
from pymilvus import MilvusClient as M
c = M(uri="http://localhost:19530", db_name="junlim", timeout=5)
c.get_load_state("nq_seq")
PYEOF
      then up=1; break; fi
      sleep 10
    done
    if [ "$up" = "1" ]; then
      python - <<'PYEOF'
import time
from pymilvus import MilvusClient as M
c = M(uri="http://localhost:19530", db_name="junlim")
st = c.get_load_state("nq_seq")
if "Loaded" not in str(st):
    t = time.time(); c.load_collection("nq_seq")
    print(f"loaded nq_seq after restart in {time.time() - t:.1f}s", flush=True)
else:
    print("nq_seq already loaded after restart", flush=True)
PYEOF
      sw reclaim
      run cold cold
    else
      echo "$(ts) SKIP cold (Milvus가 10분 안에 뜨지 않음)"
    fi
  else
    sw reclaim
    run cold cold
  fi
fi
echo "$(ts) done run_all"
