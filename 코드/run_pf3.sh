#!/bin/bash
# usage: cd /workspace/junlim && tmux new -d -s pf3wait 'RUN=run_pf3.sh LOG=pf3.log bash wait_run_pf.sh > wait3.log 2>&1'
#        (wait_run_pf.sh가 다른 팀원이 조용해질 때까지 기다렸다가 이 스크립트를 부른다)
# C2 v3: 실제 질문으로 이웃 선택 규칙 검증. 결과는 전부 $OUT(기본 /workspace/junlim/pf3_out)에 쌓이고, 끝나면 묶음 파일 하나로 만든다.
#   1) gold_map.py   정답 근거(qrels, nq_open 정답 문자열)를 조각 번호로 옮기고 전수 검증 (Milvus 안 씀, 약 10분 예상)
#   2) FiQA          fiqa_seq만 잠깐 올려 실제 질문 검색 -> 내리고 품질 분석
#   3) NQ            nq_seq 올림 -> 실제 질문 3,452개 검색 -> 품질 분석, 필요 여부 파일(need_nq.json) 만들기
#   4) 시간 측정 (nq_seq, 평소 상태)
#      pf3 seq   정책 8개를 같은 질문으로 번갈아, 회차마다 400질문 × 3회                       (약 2분)
#      pf3 load  동시 요청 64, 요청 상한 8, 정책 8개 × 구간 25초 × 3회                         (약 10분)
#      pf2 seq   무작위 필요 비율 p = 0~100% × 정책 3개, p마다 300질문 × 3회 (측정한 손익분기)  (약 4분)
#      pf2 load  p = 0, 10, 50, 100% × 정책 3개 × 25초 × 3회                                  (약 17분)
# 시작 전에 check_busy.sh로 다른 팀원이 쓰는 중인지 보고, 쓰는 중이면 아무것도 바꾸지 않고 끝낸다.
# 측정이 실패하거나 Milvus 상태(healthz)가 이상하면 멈춘다. 끝나면 항상 reset, nq_seq·fiqa_seq release, 결과 묶기.
ROOT=${ROOT:-/workspace/junlim}
BACKEND=${BACKEND:-milvus}      # 도구 점검용 mock
REPS=${REPS:-3}
SEQ_REQS=${SEQ_REQS:-400}; PF2_REPS=${PF2_REPS:-300}
LOAD_PS=${LOAD_PS:-"0 0.1 0.5 1"}
DUR=${DUR:-20}; WARM=${WARM:-5}; N=${N:-64}; CAP=${CAP:-8}
OUT=${OUT:-$ROOT/pf3_out}
SKIP_GOLD=${SKIP_GOLD:-0}; SKIP_FIQA=${SKIP_FIQA:-0}; SKIP_PF2=${SKIP_PF2:-0}
GOLD_TIMEOUT=${GOLD_TIMEOUT:-2400}
SW_CMD=${SW_CMD:-"timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=10 capvdb@203.253.145.110"}
cd "$ROOT" || exit 1
[ -f venv/bin/activate ] && source venv/bin/activate
mkdir -p "$OUT"
NQENV="BENCH_DATA=$ROOT/data_nq BENCH_COLL=nq"
FQENV="BENCH_DATA=$ROOT/data BENCH_COLL=fiqa"

ts() { date '+%H:%M:%S'; }
sw() { echo "$(ts) switch: $*"; $SW_CMD "$*"; }
status() { $SW_CMD status 2>&1; }
health() { [ "$BACKEND" != "milvus" ] && return 0; curl -s -m 3 localhost:9091/healthz | grep -q OK; }
stop() { echo "$(ts) STOP: $1"; exit 1; }
b3() { health || stop "Milvus healthz 실패"; env $NQENV python bench_pf3.py --backend "$BACKEND" --gold "$OUT" "$@" || stop "실패한 측정: bench_pf3 $*"; }
b2() { health || stop "Milvus healthz 실패"; env $NQENV python bench_pf2.py --backend "$BACKEND" "$@" || stop "실패한 측정: bench_pf2 $*"; }
coll() {  # coll load|release 이름
  [ "$BACKEND" != "milvus" ] && return 0
  python - "$1" "$2" <<'PYEOF'
import sys, time
from pymilvus import MilvusClient as M
act, name = sys.argv[1], sys.argv[2]
c = M(uri="http://localhost:19530", db_name="junlim")
if act == "release":
    c.release_collection(name); print("released", name, c.get_load_state(name), flush=True)
else:
    idx = c.describe_index(name, "vec")
    if str(idx.get("mmap.enabled", "False")).lower() == "true":
        raise SystemExit(f"{name}: index mmap is ON: run `python set_mmap.py off` first")
    if "Loaded" not in str(c.get_load_state(name)):
        t = time.time(); c.load_collection(name); print(f"loaded {name} in {time.time() - t:.1f}s", flush=True)
    print(name, c.get_load_state(name), "index mmap:", idx.get("mmap.enabled", "False"), flush=True)
PYEOF
}
cleanup() {
  echo "$(ts) cleanup: reset"; $SW_CMD reset
  coll release nq_seq || echo "nq_seq release 실패"
  coll release fiqa_seq || echo "fiqa_seq release 실패"
  status | head -4
  cp -f "$ROOT/pf3.log" "$OUT/" 2>/dev/null
  tgz="$ROOT/pf3_results_$(date +%m%d_%H%M).tgz"
  tar czf "$tgz" -C "$ROOT" "$(basename "$OUT")" && echo "$(ts) 결과 묶음: $tgz ($(du -h "$tgz" | cut -f1))"
}

# 다른 팀원이 쓰는 중이면 아무것도 바꾸지 않고 끝낸다 (reset, release를 하기 전에 확인)
if [ "$BACKEND" = "milvus" ] && [ "${SKIP_BUSY_CHECK:-0}" != "1" ]; then
  bash check_busy.sh || { echo "$(ts) STOP: 다른 사람이 쓰는 중이거나 확인하지 못함. 아무것도 바꾸지 않고 끝냄 (위 결과 참고)"; exit 1; }
fi

trap cleanup EXIT
trap 'exit 130' INT TERM

echo "$(ts) start run_pf3 reps=$REPS backend=$BACKEND out=$OUT n=$N cap=$CAP dur=$DUR seq_reqs=$SEQ_REQS"
md5sum gold_map.py bench_pf3.py analyze_pf3.py pf3_rules.py bench_pf2.py bench_v2.py prep_nq.py run_pf3.sh | cut -c1-8,33-
python pf3_rules.py || stop "pf3_rules selftest 실패"
python analyze_pf3.py --selftest 2>/dev/null || stop "analyze_pf3 selftest 실패"
env $NQENV python bench_pf2.py --backend mock --selftest > /dev/null || stop "bench_pf2 selftest 실패"
health || stop "시작 전 Milvus healthz 실패"
sw reset

echo "$(ts) ===== 1) gold_map (Milvus 안 씀)"
if [ "$SKIP_GOLD" != "1" ]; then
  timeout "$GOLD_TIMEOUT" python gold_map.py --nq-data "$ROOT/data_nq" --fiqa-data "$ROOT/data" --out "$OUT" > "$OUT/gold.log" 2>&1 || echo "$(ts) gold_map 실패 또는 시간 초과 (정답 기준 일부가 빠질 수 있음, 계속)"
  grep -v "passages read" "$OUT/gold.log" | tail -n 30
fi
[ -f "$OUT/nq_flags.npy" ] || stop "nq_flags.npy가 없음 (gold.log 확인)"

if [ "$SKIP_FIQA" != "1" ] && [ -f "$OUT/fiqa_flags.npy" ]; then
  echo "$(ts) ===== 2) FiQA 실제 질문 검색 (품질용)"
  coll release nq_seq
  if coll load fiqa_seq; then
    sleep 5
    health && env $FQENV python bench_pf3.py --backend "$BACKEND" --mode hits --ds fiqa --gold "$OUT" || echo "$(ts) FiQA 검색 실패 (계속)"
  else
    echo "$(ts) fiqa_seq 올리기 실패 (계속)"
  fi
  coll release fiqa_seq
  if [ -f "$OUT/hits_fiqa.jsonl" ]; then
    python analyze_pf3.py --ds fiqa --data "$ROOT/data" --gold "$OUT" || echo "$(ts) FiQA 분석 실패 (계속)"
  fi
fi

echo "$(ts) ===== 3) NQ 준비와 실제 질문 검색 (품질용)"
coll load nq_seq || stop "nq_seq 준비 실패"
[ "$BACKEND" = "milvus" ] && sleep 20
status | head -12
b3 --mode hits --ds nq
python analyze_pf3.py --ds nq --data "$ROOT/data_nq" --gold "$OUT" || echo "$(ts) NQ 분석 실패: 시간 측정은 필요 여부 없이(정책 6개) 계속"
env $NQENV python bench_pf3.py --backend mock --selftest --gold "$OUT" > /dev/null || stop "bench_pf3 selftest 실패"

echo "$(ts) ===== 4a) pf3 seq (정책 8개, 요청을 하나씩)"
for rep in $(seq 1 $REPS); do
  b3 --mode seq --ds nq --cond pf3_seq --rep $rep --reqs "$SEQ_REQS"
done

echo "$(ts) ===== 4b) pf3 load (동시 요청 $N, 상한 $CAP)"
P3=(none all rA rB rC1 rC3 lazy_need text_need)
for rep in $(seq 1 $REPS); do
  for i in 0 1 2 3 4 5 6 7; do
    pol=${P3[$(( (i + rep - 1) % 8 ))]}
    b3 --mode load --ds nq --cond pf3_load --rep $rep --policy $pol --n $N --cap $CAP --dur $DUR --warm $WARM
  done
done

if [ "$SKIP_PF2" != "1" ]; then
  echo "$(ts) ===== 4c) pf2 seq (무작위 필요 비율)"
  for rep in $(seq 1 $REPS); do
    b2 --mode seq --cond pf2_seq --rep $rep --reps "$PF2_REPS" --out "$OUT/results_pf2_seq_nq.csv"
  done
  echo "$(ts) ===== 4d) pf2 load (동시 요청 $N, 상한 $CAP)"
  POLS=(lazy_id prefetch lazy_text)
  for rep in $(seq 1 $REPS); do
    for p in $LOAD_PS; do
      for i in 0 1 2; do
        pol=${POLS[$(( (i + rep - 1) % 3 ))]}
        b2 --mode load --cond pf2_load --rep $rep --policy $pol --p $p --n $N --cap $CAP --dur $DUR --warm $WARM --out "$OUT/results_pf2_load_nq.csv"
      done
    done
  done
fi
status | head -12
echo "$(ts) done run_pf3"
