#!/bin/bash
# 우리 전용 Qdrant 시작/중지. 127.0.0.1:6343 에서만 열림. 우리가 띄운 프로세스만 종료함.
D=~/m4-exp; PORT=6343; PIDF=$D/qdrant.pid
case "$1" in
start)
  if curl -sf http://127.0.0.1:$PORT/readyz >/dev/null; then echo "port $PORT already in use -> abort"; exit 1; fi
  cd $D/qdrant
  QDRANT__STORAGE__STORAGE_PATH=${M4_STORAGE:-$D/qdrant_storage} \
  QDRANT__STORAGE__SNAPSHOTS_PATH=$D/qdrant_snapshots \
  QDRANT__SERVICE__HOST=127.0.0.1 QDRANT__SERVICE__HTTP_PORT=$PORT QDRANT__SERVICE__GRPC_PORT=6344 \
  QDRANT__TELEMETRY_DISABLED=true \
  QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=8 \
  QDRANT__STORAGE__PERFORMANCE__MAX_OPTIMIZATION_THREADS=4 \
  nohup nice -n 10 ./qdrant >> $D/results/qdrant.log 2>&1 &
  echo $! > $PIDF
  for i in $(seq 120); do curl -sf http://127.0.0.1:$PORT/readyz >/dev/null && { echo "qdrant ready (pid $(cat $PIDF))"; exit 0; }; sleep 1; done
  echo "qdrant not ready; see results/qdrant.log"; exit 1;;
stop)
  [ -f $PIDF ] || { echo "no pid file"; exit 0; }
  P=$(cat $PIDF)
  # 안전장치: 우리 qdrant 바이너리가 맞을 때만 종료
  if [ "$(readlink /proc/$P/exe 2>/dev/null)" = "$D/qdrant/qdrant" ]; then
    kill $P; for i in $(seq 60); do kill -0 $P 2>/dev/null || break; sleep 1; done; echo "qdrant stopped"
  else echo "pid $P is not our qdrant -> not killing"; fi
  rm -f $PIDF;;
*) echo "usage: $0 start|stop";;
esac
