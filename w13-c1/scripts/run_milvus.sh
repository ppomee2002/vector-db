#!/bin/bash
# run_milvus.sh: C1 Milvus 실험 실행  김지현 2024134018
# 사용 (서버, 코드 폴더 안에서. PY는 내 venv):
#   bash run_milvus.sh check     접속·메트릭 확인 (아무것도 만들지 않음)
#   bash run_milvus.sh probe     2만 행 테스트 컬렉션으로 점검 후 삭제, PASS가 나와야 함
#   bash run_milvus.sh load      내 DB(jihyeon)에 w13c1_doc, w13c1_rand 만들기 (각 100만 행, 단톡방 공지 후)
#   bash run_milvus.sh quick     M1 쿼리 5개 + M2 쿼리 50개로 점검
#   bash run_milvus.sh m1        M1 콜드 조회 본 실험 (쿼리 100개, 1시간 안팎)
#   bash run_milvus.sh m2        M2 웜 조회 본 실험 (쿼리 500개 x 3회)
#   bash run_milvus.sh release   내 컬렉션 메모리에서 내리기
#   bash run_milvus.sh drop      내 컬렉션(w13c1_*) 삭제
# 오래 걸리는 m1, m2는 tmux에서:
#   tmux new -d -s c1m 'cd /workspace/jihyeon/c1_w13_code && PY=/workspace/jihyeon/venv/bin/python bash run_milvus.sh m1'
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
ROOT=$(realpath -m "${C1_OUT_ROOT:-/workspace/jihyeon}")
WORK=${C1_WORK:-$ROOT/c1_w13}
export C1_OUT_ROOT="$ROOT" C1_WORK="$WORK" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
case "$(realpath -m "$WORK")" in "$ROOT"|"$ROOT"/*) ;; *) echo "WORK가 내 폴더($ROOT) 밖입니다: $WORK"; exit 1;; esac
mkdir -p "$WORK/logs" "$WORK/milvus"
STAMP=$(date +%Y%m%d_%H%M)

finish() {   # 결과 폴더에 로그·코드 사본을 넣고 요약표와 묶음 파일 만들기
  out="$1"; log="$2"
  cp "$log" "$out/run.log"
  mkdir -p "$out/code" && cp ./*.py ./*.sh "$out/code/" && (cd "$out/code" && md5sum ./* > MD5SUMS)
  $PY summarize_milvus.py "$out" > /dev/null
  (cd "$(dirname "$out")" && tar czf "$(basename "$out").tgz" "$(basename "$out")")
  echo "=================================================="
  echo "끝. 결과: $out/summary.md"
  echo "PC로 받을 파일: $out.tgz"
}
run() {   # run <명령> [인자...]
  log="$WORK/logs/milvus_$1_$STAMP.log"
  $PY c1_milvus.py "$@" 2>&1 | tee "$log"
  out=$(tail -1 "$log")
  if [ -d "$out" ]; then finish "$out" "$log"; fi
}

case "${1:-}" in
  check|probe|load|release) run "$1" ;;
  quick) run m1 --quick; run m2 --quick ;;
  m1) run m1 ;;
  m2) run m2 ;;
  drop) $PY c1_milvus.py drop --yes 2>&1 | tee "$WORK/logs/milvus_drop_$STAMP.log" ;;
  *) sed -n '2,15p' "$0"; exit 1 ;;
esac
