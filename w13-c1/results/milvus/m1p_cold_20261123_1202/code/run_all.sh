#!/bin/bash
# run_all.sh: C1 실험 전체 실행  김지현 2024134018
# 사용 (서버, 코드 폴더 안에서):
#   bash run_all.sh build    저장소 만들기 (처음 한 번)
#   bash run_all.sh check    저장소 검증
#   bash run_all.sh quick    쿼리 300개, 1회 (점검용, 1~2분)
#   bash run_all.sh full     본 실험 (조건 26개 x 3회) + 요약 + 묶음 파일
#   bash run_all.sh all      build -> check -> quick -> full 차례로
# 오래 걸리는 건 tmux에서:  tmux new -d -s c1 'bash run_all.sh all'
#
# 쓰는 곳: /workspace/jihyeon/c1_w13 아래만. 다른 사람 폴더는 읽기만 한다.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
ROOT=$(realpath -m "${C1_OUT_ROOT:-/workspace/jihyeon}")
WORK=${C1_WORK:-$ROOT/c1_w13}
export C1_OUT_ROOT="$ROOT" C1_WORK="$WORK"
case "$(realpath -m "$WORK")" in "$ROOT"|"$ROOT"/*) ;; *) echo "WORK가 내 폴더($ROOT) 밖입니다: $WORK"; exit 1;; esac
mkdir -p "$WORK/logs" "$WORK/results"
STAMP=$(date +%Y%m%d_%H%M)
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1

build() {
  if [ -f "$WORK/store/meta.json" ]; then echo "저장소가 이미 있습니다: $WORK/store (다시 만들려면 이 폴더를 직접 지우세요)"; return; fi
  $PY -c "import sys; sys.path.insert(0,'.'); import c1_store; c1_store.build('$WORK/store')" 2>&1 | tee "$WORK/logs/build_$STAMP.log"
}
check() { $PY check_store.py --store "$WORK/store" 2>&1 | tee "$WORK/logs/check_$STAMP.log"; }
quick() {
  out=$($PY run_c1.py --store "$WORK/store" --quick 2>&1 | tee "$WORK/logs/quick_$STAMP.log" | tail -1)
  $PY summarize.py "$out" > /dev/null && echo "quick 결과: $out/summary.md"
}
full() {
  out="$WORK/results/run_$STAMP"
  $PY run_c1.py --store "$WORK/store" --out "$out" 2>&1 | tee "$WORK/logs/full_$STAMP.log"
  cp "$WORK/logs/full_$STAMP.log" "$out/run.log"
  mkdir -p "$out/code" && cp ./*.py ./*.sh "$out/code/" && (cd "$out/code" && md5sum ./* > MD5SUMS)
  $PY summarize.py "$out" > /dev/null
  (cd "$WORK/results" && tar czf "run_$STAMP.tgz" "run_$STAMP")
  echo "=================================================="
  echo "끝. 결과: $out/summary.md"
  echo "PC로 받을 파일: $WORK/results/run_$STAMP.tgz"
}

case "${1:-}" in
  build) build ;;
  check) check ;;
  quick) quick ;;
  full) full ;;
  all) build; check; quick; full ;;
  *) sed -n '2,12p' "$0"; exit 1 ;;
esac
