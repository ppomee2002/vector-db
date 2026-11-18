#!/bin/bash
# usage: cd /workspace/junlim && bash check_busy.sh      (약 15초. 아무것도 바꾸지 않고 보기만 한다)
# 지금 다른 팀원이 Milvus나 컨테이너를 쓰고 있는지 확인한다.
#   종료 코드 0: 조용함   1: 누가 쓰는 중으로 보임   2: 확인하지 못함
ROOT=${ROOT:-/workspace/junlim}
SW_CMD=${SW_CMD:-"timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 capvdb@203.253.145.110"}
SEC=${SEC:-10}              # Milvus 활동을 지켜보는 시간(초)
CPU_BUSY=${CPU_BUSY:-1.0}   # Milvus가 이만큼(코어) 넘게 쓰면 누가 쓰는 중으로 본다
RPS_BUSY=${RPS_BUSY:-5}     # 초당 요청이 이만큼 넘으면 누가 쓰는 중으로 본다
ME=$(id -un)
busy=0; unknown=0

echo "== 지금 접속한 계정"
who | awk '{print "  " $1, $3, $4}' | sort -u

echo "== 다른 계정의 프로세스 (CPU 순 상위 10개, %CPU는 시작 후 평균)"
others=$(ps -eo user:12,pid,etime,pcpu,args --sort=-pcpu --no-headers | awk -v me="$ME" '$1 != me && $1 != "root"')
if [ -n "$others" ]; then
  echo "$others" | head -10 | cut -c1-150 | sed 's/^/  /'
  heavy=$(echo "$others" | awk '$4 + 0 >= 10' | wc -l)
  [ "$heavy" -gt 0 ] && { echo "  -> CPU를 10% 넘게 쓰는 다른 계정 프로세스 ${heavy}개"; busy=1; }
else
  echo "  (없음)"
fi

echo "== Milvus 활동 (${SEC}초 동안)"
metrics() { curl -s -m 5 localhost:9091/metrics; }
m1=$(metrics); sleep "$SEC"; m2=$(metrics)
if [ -z "$m1" ] || [ -z "$m2" ]; then
  echo "  Milvus 메트릭(localhost:9091)을 읽지 못함"; unknown=1
else
  sum() { echo "$1" | awk -v k="$2" '$1 == k || index($1, k "{") == 1 {s += $NF} END {printf "%.3f", s}'; }
  cnt() { echo "$1" | grep -cE "^$2[{ ]"; }
  if [ "$(cnt "$m1" process_cpu_seconds_total)" -gt 0 ]; then
    cores=$(awk -v a="$(sum "$m1" process_cpu_seconds_total)" -v b="$(sum "$m2" process_cpu_seconds_total)" -v s="$SEC" 'BEGIN {printf "%.2f", (b - a) / s}')
    echo "  Milvus CPU ${cores}코어 (기준 ${CPU_BUSY})"
    awk -v c="$cores" -v t="$CPU_BUSY" 'BEGIN {exit !(c >= t)}' && busy=1
  else
    echo "  CPU 메트릭 없음"; unknown=1
  fi
  if [ "$(cnt "$m1" milvus_proxy_req_count)" -gt 0 ]; then
    rps=$(awk -v a="$(sum "$m1" milvus_proxy_req_count)" -v b="$(sum "$m2" milvus_proxy_req_count)" -v s="$SEC" 'BEGIN {printf "%.1f", (b - a) / s}')
    echo "  요청 ${rps}건/초 (기준 ${RPS_BUSY})"
    awk -v r="$rps" -v t="$RPS_BUSY" 'BEGIN {exit !(r >= t)}' && busy=1
  else
    echo "  요청 수 메트릭 없음 (CPU로만 판단)"
  fi
fi

echo "== 올라가 있는 컬렉션 (참고: 올라가 있다고 지금 쓰는 중이라는 뜻은 아님)"
[ -f "$ROOT/venv/bin/activate" ] && source "$ROOT/venv/bin/activate"
python - <<'PYEOF' 2>&1 | sed 's/^/  /'
from pymilvus import MilvusClient as M
uri = "http://localhost:19530"
found = False
for db in M(uri=uri).list_databases():
    c = M(uri=uri, db_name=db)
    for col in c.list_collections():
        st = str(c.get_load_state(col).get("state", ""))
        if "Loaded" in st or "Loading" in st:
            print(f"{db}.{col}: {st}"); found = True
if not found:
    print("(없음)")
PYEOF

echo "== 관리자 스위치: 지금 걸린 제한과 최근 기록 (참고)"
$SW_CMD status 2>&1 | head -8 | sed 's/^/  /'
$SW_CMD log 2>&1 | tail -5 | sed 's/^/  /'

echo "=="
if [ "$busy" = 1 ]; then
  echo "결과: 누가 쓰는 중으로 보임. 지금은 측정하지 마세요."; exit 1
elif [ "$unknown" = 1 ]; then
  echo "결과: 일부를 확인하지 못함. 위 내용을 보고 판단하세요."; exit 2
fi
echo "결과: 조용함. 측정해도 됩니다."; exit 0
