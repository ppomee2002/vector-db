#!/bin/bash
# usage: cd /workspace/junlim && tmux new -d -s w100 'bash run_w100.sh > w100.log 2>&1'
# C2 추가 검증: 64단어 때와 같은 NQ 글을 100단어 Chunk로 다시 잘라, 이웃 선택 규칙의 품질(정답 포함률, 읽은 Chunk)을 다시 잰다.
#   100단어: DPR이 위키백과를 자른 크기. 끝 조각 기준(min_tail)은 64단어 때와 같은 비율(크기의 절반)인 50단어.
#   같은 글: 64단어 데이터(data_nq)의 문서가 그대로 들어가도록 Chunk 수(N)를 계산하고, 읽는 문단 양(want)을 64단어 때와 똑같이 맞춘다.
#   규칙은 바꾸지 않는다(규칙 B+C1을 미리 정해 두고 확인). 처리량은 재지 않는다.
#   1) 목표 Chunk 수 계산 (data_nq만 읽음, 1분 안쪽)
#   2) prep_nq100.py: 자르기 + 임베딩 (GPU 1장, 비어 있는 GPU를 골라 씀, 30~60분 예상) -> data_nq100
#   3) w100_check.py: 문서 글과 질문이 64단어 때와 같은지 확인 (다르면 멈춤)
#   4) gold_map.py: 정답 문단 -> 100단어 Chunk 번호, 전수 대조 (Milvus 안 씀, 2~10분)
#   5) Milvus: 다른 팀원이 조용할 때까지 기다렸다가 nq_seq를 내리고 nq100_seq 적재 + 색인 (약 64만 행, 5~10분) -> 실제 질문 3,452개 검색
#   6) analyze_pf3.py: 규칙별 품질
#   끝나면 nq100_seq를 release하고 결과를 묶는다. 컬렉션은 지우지 않는다(처리량을 다시 잴 수도 있어서).
#   64단어 데이터(data_nq), 11/16 결과(pf3_out), 컬렉션 nq_seq의 데이터는 바꾸지 않는다(nq_seq는 메모리 때문에 release만 함).
set -o pipefail
ROOT=${ROOT:-/workspace/junlim}
SRC=${SRC:-$ROOT/data_nq}
D=${D:-$ROOT/data_nq100}
G=${G:-$ROOT/gold_w100}
W=${W:-100}; MT=${MT:-50}
WANT0=${WANT0:-83200000}          # 64단어 때 읽은 단어 수 (pf3_out/gold_ok_nq.json want_words)
MAX_SEQ=${MAX_SEQ:-256}
GPU=${GPU:-auto}
BACKEND=${BACKEND:-milvus}        # 도구 점검용 mock
WAIT_MAX=${WAIT_MAX:-21600}; WAIT_INT=${WAIT_INT:-180}
PREP_CMD=${PREP_CMD:-"python prep_nq100.py"}
GOLD_CMD=${GOLD_CMD:-"python gold_map.py"}
cd "$ROOT" || exit 1
[ -f venv/bin/activate ] && source venv/bin/activate

ts() { date '+%H:%M:%S'; }
stop() { echo "$(ts) STOP: $1"; exit 1; }
health() { [ "$BACKEND" != "milvus" ] && return 0; curl -s -m 3 localhost:9091/healthz | grep -q OK; }
coll() {  # coll release|check 이름
  [ "$BACKEND" != "milvus" ] && return 0
  python - "$1" "$2" <<'PYEOF'
import sys
from pymilvus import MilvusClient as M
act, name = sys.argv[1], sys.argv[2]
c = M(uri="http://localhost:19530", db_name="junlim")
if not c.has_collection(name):
    print("no collection", name, flush=True); raise SystemExit(0 if act == "release" else 1)
if act == "release":
    c.release_collection(name); print("released", name, c.get_load_state(name), flush=True)
else:
    idx = c.describe_index(name, "vec")
    print(name, c.get_load_state(name), "rows", c.get_collection_stats(name), "index", idx.get("index_type"), idx.get("params", ""),
          "index mmap:", idx.get("mmap.enabled", "False"), flush=True)
    if "Loaded" not in str(c.get_load_state(name)):
        raise SystemExit(1)
PYEOF
}
cleanup() {
  coll release nq100_seq || echo "nq100_seq release 실패"
  cp -f "$ROOT/w100.log" "$G/" 2>/dev/null
  cp -f "$D/meta.json" "$G/data_meta.json" 2>/dev/null
  tgz="$ROOT/w100_results_$(date +%m%d_%H%M).tgz"
  tar czf "$tgz" -C "$ROOT" --exclude="cand_texts_*" "$(basename "$G")" && echo "$(ts) 결과 묶음: $tgz ($(du -h "$tgz" | cut -f1))"
}

# 입력 확인 (아직 아무것도 바꾸지 않음)
[ -s "$SRC/chunks.jsonl" ] && [ -s "$SRC/meta.json" ] && [ -s "$SRC/queries.jsonl" ] || stop "$SRC의 64단어 데이터가 없음"
[ -e "$D/meta.json" ] && stop "$D/meta.json이 이미 있음 (이전 결과). 옮기거나 D=다른폴더 로 실행"
[ -e "$G/pf3_quality_nq.csv" ] && stop "$G에 이전 결과가 있음. 옮기거나 G=다른폴더 로 실행"
avail=$(df -BG --output=avail /workspace 2>/dev/null | tail -1 | tr -dc 0-9)
[ -n "$avail" ] && [ "$avail" -lt 5 ] && stop "/workspace 남은 공간 ${avail}G (5G 이상 필요)"
mkdir -p "$D" "$G"
trap cleanup EXIT
trap 'exit 130' INT TERM

echo "$(ts) start run_w100 W=$W min_tail=$MT max_seq=$MAX_SEQ src=$SRC data=$D gold=$G backend=$BACKEND (남은 공간 ${avail:-?}G)"
md5sum prep_nq.py prep_nq100.py w100_target.py w100_check.py gold_map.py load_big.py bench_pf4.py bench_v2.py pf3_rules.py analyze_pf3.py run_w100.sh | cut -c1-8,33-
python pf3_rules.py || stop "pf3_rules selftest 실패"
python analyze_pf3.py --selftest 2>/dev/null || stop "analyze_pf3 selftest 실패"

echo "$(ts) ===== 1) 목표 Chunk 수"
read -r N OV < <(python w100_target.py "$SRC" "$W" "$MT" "$WANT0") || stop "목표 계산 실패"
[ -n "$N" ] && [ -n "$OV" ] || stop "목표 계산 결과 없음"
echo "N=$N OV=$OV"

echo "$(ts) ===== 2) 자르기 + 임베딩"
if [ "$GPU" = auto ]; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv || stop "nvidia-smi 실패"
  GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', *' '$2 < 1000 && $3 < 10 {print $1; exit}')
  [ -n "$GPU" ] || stop "비어 있는 GPU가 없음 (위 nvidia-smi 참고). 나중에 다시 실행하거나 GPU=번호 로 지정"
fi
echo "GPU $GPU 사용"
CUDA_VISIBLE_DEVICES=$GPU MAX_SEQ=$MAX_SEQ $PREP_CMD --out "$D" --chunks "$N" --words "$W" --min-tail "$MT" --oversample "$OV" --pseudo-q 0 --device cuda:0 || stop "prep 실패"

echo "$(ts) ===== 3) 같은 글, 같은 질문인지 확인"
python w100_check.py "$SRC" "$D" || stop "64단어 데이터와 글이나 질문이 다름"

echo "$(ts) ===== 4) gold_map (Milvus 안 씀)"
timeout 3600 $GOLD_CMD --nq-data "$D" --out "$G" --oversample "$OV" --skip-fiqa > "$G/gold.log" 2>&1 || stop "gold_map 실패 ($G/gold.log)"
cat "$G/gold_report.txt" 2>/dev/null | head -14
python -c "import json,sys; d=json.load(open('$G/gold_ok_nq.json')); print('gold ok:', d.get('ok'), 'chunk_mismatch', d.get('chunk_mismatch'), 'eval', d.get('eval_queries'), 'answers', d.get('eval_and_answers')); sys.exit(0 if d.get('ok') else 1)" \
  || stop "정답 위치 검증 실패 ($G/gold_ok_nq.json)"

echo "$(ts) ===== 5) Milvus: 적재 + 검색"
if [ "$BACKEND" = "milvus" ]; then
  waited=0
  until bash check_busy.sh > "$G/busy_last.txt" 2>&1; do
    [ "$waited" -ge "$WAIT_MAX" ] && { cat "$G/busy_last.txt"; stop "다른 팀원이 계속 쓰는 중이라 ${WAIT_MAX}초 기다리고 멈춤 (data_nq100, gold_w100은 남아 있음)"; }
    echo "$(ts) 기다림: 다른 사람이 쓰는 중 ($(grep -E '^결과' "$G/busy_last.txt" | head -1))"
    sleep "$WAIT_INT"; waited=$((waited + WAIT_INT))
  done
  echo "$(ts) 조용함. Milvus 사용 시작"
  health || stop "Milvus healthz 실패"
  coll release nq_seq || stop "nq_seq release 실패"
  python load_big.py seq --data "$D" --coll nq100 || stop "nq100_seq 적재 실패"
  coll check nq100_seq || stop "nq100_seq가 올라가 있지 않음"
else
  python -c "import numpy as np; np.save('$D/pk_seq.npy', np.arange(int(__import__('json').load(open('$D/meta.json'))['n_chunks']), dtype=np.int64))"
fi
health || stop "Milvus healthz 실패"
env BENCH_DATA="$D" BENCH_COLL=nq100 python bench_pf4.py --backend "$BACKEND" --gold "$G" --mode hits --ds nq || stop "검색 실패"

echo "$(ts) ===== 6) 규칙별 품질"
python analyze_pf3.py --ds nq --data "$D" --gold "$G" || stop "분석 실패"
echo "$(ts) done run_w100"
