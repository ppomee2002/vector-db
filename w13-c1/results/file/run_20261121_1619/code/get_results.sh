#!/bin/bash
# get_results.sh: 서버의 C1 실험 결과를 내 PC로 받기 (macOS / Linux)  김지현 2024134018
# 사용: Tailscale 연결 후  bash get_results.sh vdb0X [받을 폴더]
# 서버에서는 아무것도 바꾸지 않고 복사만 한다.
set -e
ACC=${1:?"cap-vdb 계정을 적어주세요 (예: bash get_results.sh vdb01)"}
LOCAL=${2:-$HOME/capstone/c1_w13}
SERVER=100.82.53.54; PORT=2222; REMOTE=/workspace/jihyeon/c1_w13
mkdir -p "$LOCAL/results" "$LOCAL/logs"
scp -P $PORT "$ACC@$SERVER:$REMOTE/results/*.tgz" "$ACC@$SERVER:$REMOTE/results/check_store.json" "$LOCAL/results/"
scp -P $PORT "$ACC@$SERVER:$REMOTE/logs/*" "$LOCAL/logs/"
for t in "$LOCAL"/results/*.tgz; do d="${t%.tgz}"; [ -d "$d" ] || tar xzf "$t" -C "$LOCAL/results"; done
echo "받은 곳: $LOCAL"; ls -d "$LOCAL"/results/*/ 2>/dev/null | sed 's|$|summary.md|; s|^|  |'
