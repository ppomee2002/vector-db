#!/usr/bin/env bash
# M3 experiment driver.   usage:  ./run_all.sh quick | core | sens | full
#   quick : ~10 min smoke test of every scenario (1 repetition, short points)
#   core  : ~2 h   main comparison A-D + async server S, validation E/F, open-loop, sampling overhead (3 reps)
#   sens  : ~2 h   sensitivity: k, chunk size, LRU cache, search cost (none/light/main/heavy)          (3 reps)
#   full  : core + sens (~4 h)
# Run long modes with nohup (see README). Nothing here needs sudo or installs anything.
# Env overrides: M3_DATA, M3_CPUS (7 server CPUs + 1 client CPU = 800%), M3_DEV, M3_REP, M3_OUT,
#               M3_EXTRA (extra flags, e.g. -direct=false for testing), M3_FORCE=1 (run even if selftest fails)
set -u -o pipefail
MODE=${1:-quick}
cd "$(dirname "$0")"
BIN=./m3bench
DATA=${M3_DATA:-$HOME/m3data/data.bin}
CPUS=${M3_CPUS:-7}
DEV=${M3_DEV:-auto}
OUT=${M3_OUT:-results_${MODE}_$(date +%m%d_%H%M)}

case $MODE in
  quick) REP=1; WU=3s;  DUR=5s;  PAUSE=0.5s; NS="1,8,32"; NS_SSD="8,64"; NS_OVH="32"; NS_SENS="32"
         RATES="0.5,0.9"; QD="1,8,64"; BDUR=2s; QUIET_MAX=30s; CORE=1; SENS=1 ;;
  core)  REP=3; WU=8s;  DUR=25s; PAUSE=2s; NS="1,4,8,16,32,64"; NS_SSD="1,4,16,32,64,128"; NS_OVH="8,64"; NS_SENS=""
         RATES="0.3,0.5,0.7,0.8,0.9,0.95,1.0"; QD="1,2,4,8,16,32,64,128,256"; BDUR=8s; QUIET_MAX=10m; CORE=1; SENS=0 ;;
  sens)  REP=3; WU=8s;  DUR=25s; PAUSE=2s; NS=""; NS_SSD=""; NS_OVH=""; NS_SENS="1,8,32,64"
         RATES=""; QD="1,2,4,8,16,32,64,128,256"; BDUR=8s; QUIET_MAX=10m; CORE=0; SENS=1 ;;
  full)  REP=3; WU=8s;  DUR=25s; PAUSE=2s; NS="1,4,8,16,32,64"; NS_SSD="1,4,16,32,64,128"; NS_OVH="8,64"; NS_SENS="1,8,32,64"
         RATES="0.3,0.5,0.7,0.8,0.9,0.95,1.0"; QD="1,2,4,8,16,32,64,128,256"; BDUR=8s; QUIET_MAX=10m; CORE=1; SENS=1 ;;
  *) echo "usage: $0 quick|core|sens|full"; exit 2 ;;
esac
REP=${M3_REP:-$REP}

log() { echo "[$(date +%H:%M:%S)] $*"; }
[ -f "$DATA" ] || { echo "data file $DATA not found. run:  $BIN prep -file $DATA -size 16"; exit 1; }
mkdir -p "$OUT"

# ---- search cost: calibrated against FAISS on this server if calib.env exists ----
NLIST=2048; NPROBE=4; NLIST_LIGHT=1024; NPROBE_LIGHT=1; NLIST_HEAVY=2048; NPROBE_HEAVY=64
CALIB_SOURCE="default (NOT calibrated: run ./calibrate.sh first)"
if [ -f calib.env ]; then . ./calib.env; cp calib.env calib_*.csv calib_faiss.log "$OUT"/ 2>/dev/null; fi
CALIB="main nlist=$NLIST nprobe=$NPROBE, light $NLIST_LIGHT/$NPROBE_LIGHT, heavy $NLIST_HEAVY/$NPROBE_HEAVY, source: $CALIB_SOURCE"

# ---- CPUs: pick once (least-busy, near the SSD), server = first $CPUS, client = one extra ----
SET=$($BIN cpus -cpus "$((CPUS + 1))" -dev "$DEV") || exit 1
read -r SERVER CLIENT < <(python3 -c "
s='$SET'; c=[]
for p in s.split(','):
    a,_,b=p.partition('-'); c+=range(int(a),int(b or a)+1)
srv=c[:$CPUS]; cl=c[$CPUS] if len(c)>$CPUS else 'none'
print(','.join(map(str,srv)), cl)")
PIN=(-cpus "$CPUS" -cpuset "$SERVER" -client-cpu "$CLIENT")

# ---- let the SSD settle after `prep` wrote the data file ----
AGE=$(( $(date +%s) - $(stat -c %Y "$DATA") ))
if [ "$AGE" -lt 300 ] && [ "$MODE" != quick ]; then
  log "data file written ${AGE}s ago; waiting $((300 - AGE))s for the SSD to settle"; sleep $((300 - AGE))
fi

# ---- check every assumption first (O_DIRECT, reads reach the SSD, pinning, AIO, metrics) ----
if ! $BIN selftest -file "$DATA" -dev "$DEV" -cpus "$CPUS" ${M3_EXTRA:-} | tee "$OUT/selftest.txt"; then
  [ "${M3_FORCE:-0}" = 1 ] || { log "selftest failed (see $OUT/selftest.txt). Fix it, or M3_FORCE=1 to run anyway."; exit 1; }
fi
AIO_OK=1; grep -q "PASS\] native AIO" "$OUT/selftest.txt" || AIO_OK=0

COMMON=(-file "$DATA" -dev "$DEV" "${PIN[@]}" -nlist "$NLIST" -nprobe "$NPROBE" -warmup "$WU" -dur "$DUR" -pause "$PAUSE" -out "$OUT" ${M3_EXTRA:-})

# ---- scenarios:  TAG|extra flags.  Tag = <variant>_<family>; analyze.py compares variants within a family ----
SCEN=()
if [ "$CORE" = 1 ]; then
  SCEN+=("A_base|-workers $CPUS -io sync -n $NS")         # typical synchronous server: pool = #CPUs, chunks one by one
  SCEN+=("B_base|-workers n -io sync -n $NS")             # remove the pool limit
  SCEN+=("C_base|-workers $CPUS -io parallel -n $NS")     # read the k chunks at once
  SCEN+=("D_base|-workers n -io parallel -n $NS")         # remove both (thread per request)
  [ "$AIO_OK" = 1 ] && SCEN+=("S_base|-workers $CPUS -io async -n $NS")   # event-driven: pool = #CPUs, native AIO, workers never block
  SCEN+=("E_cpu|-workers n -io parallel -n $NS -nlist 2048 -nprobe 64")       # provoke a CPU bottleneck
  SCEN+=("F_ssd|-workers n -io parallel -n $NS_SSD -search=false -chunk 64k") # provoke SSD contention
  SCEN+=("A_open|-load open -workers $CPUS -io sync -rates $RATES")           # open loop: latency vs offered load
  SCEN+=("D_open|-load open -workers n -io parallel -rates $RATES")
  [ "$AIO_OK" = 1 ] && SCEN+=("S_open|-load open -workers $CPUS -io async -rates $RATES")
  SCEN+=("D_tsoff|-workers n -io parallel -n $NS_OVH -ts=false")              # sampling overhead check (vs D_base)
fi
if [ "$SENS" = 1 ]; then
  # working set = 500k docs x 4 KB = 2 GB; cache100 / cache500 = 5% / 25% of it, popular topics (zipf 1.0)
  FAMS=("k5|-k 5" "k20|-k 20" "c16k|-chunk 16k" "cache100|-cache-mb 100 -zipf 1.0" "cache500|-cache-mb 500 -zipf 1.0" "nosearch|-search=false")
  [ "$NLIST_HEAVY/$NPROBE_HEAVY" != "$NLIST/$NPROBE" ] && FAMS+=("heavy|-nlist $NLIST_HEAVY -nprobe $NPROBE_HEAVY")
  # search-cost levels around main, only if calibration produced a different setting
  [ "$NLIST_LIGHT/$NPROBE_LIGHT" != "$NLIST/$NPROBE" ] && FAMS+=("light|-nlist $NLIST_LIGHT -nprobe $NPROBE_LIGHT")
  for fam in "${FAMS[@]}"; do
    name=${fam%%|*}; extra=${fam#*|}
    # sync server vs the async server (same #workers): the cleanest test of "remove synchronous waiting"
    SCEN+=("A_${name}|-workers $CPUS -io sync -n $NS_SENS $extra")
    if [ "$AIO_OK" = 1 ]; then SCEN+=("S_${name}|-workers $CPUS -io async -n $NS_SENS $extra")
    else SCEN+=("D_${name}|-workers n -io parallel -n $NS_SENS $extra"); fi
  done
fi

# rough duration estimate
PTS=$(python3 -c "
import re
sc='''$(printf '%s\n' "${SCEN[@]}")'''
n=0
for l in sc.splitlines():
    m=re.search(r'-n ([0-9,]+)',l) or re.search(r'-rates ([0-9.,]+)',l)
    n+=len(m.group(1).split(',')) if m else 0
    n+=1 if '-load open' in l else 0
print(n)")
SEC=$(python3 -c "
import re
d=lambda s: float(s[:-1])*(60 if s.endswith('m') else 1)
print(int($PTS*$REP*(d('$WU')+d('$DUR')+d('$PAUSE')+2)))")
log "mode=$MODE reps=$REP scenarios=${#SCEN[@]} points/rep=$PTS  ~$((SEC/60)) min (+ baselines)"
log "server CPUs=$SERVER client CPU=$CLIENT  search: $CALIB  out=$OUT"
{ echo "mode=$MODE reps=$REP server_cpus=$SERVER client_cpu=$CLIENT"; echo "search: $CALIB"; printf '%s\n' "${SCEN[@]}"; } > "$OUT/plan.txt"

$BIN env -dev "$DEV" -file "$DATA" > "$OUT/env_before.txt" 2>&1; cat "$OUT/env_before.txt"

log "SSD baseline (start): psync = same CPU budget as the server, aio = what the SSD itself can do"
$BIN quiet -dev "$DEV" -max "$QUIET_MAX" | tail -1
$BIN baseline -file "$DATA" -dev "$DEV" -cpus "$CPUS" -cpuset "$SERVER" -bs 4k,64k -qd "$QD" -dur "$BDUR" -out "$OUT/baseline.csv" ${M3_EXTRA:-} || log "baseline FAILED"
[ "$AIO_OK" = 1 ] && { $BIN baseline -engine aio -file "$DATA" -dev "$DEV" -cpus "$CPUS" -cpuset "$SERVER" -bs 4k,64k -qd "$QD" -dur "$BDUR" -out "$OUT/baseline.csv" ${M3_EXTRA:-} || log "aio baseline FAILED"; }

for rep in $(seq 1 "$REP"); do
  log "===== repetition $rep / $REP ====="
  $BIN quiet -dev "$DEV" -max "$QUIET_MAX" | tail -1
  # scenario order is shuffled per repetition so slow drift (other users, SSD state) does not line up with a scenario
  mapfile -t ORDER < <(printf '%s\n' "${SCEN[@]}" | python3 -c "import sys,random; l=sys.stdin.read().splitlines(); random.Random($rep).shuffle(l); print('\n'.join(l))")
  for s in "${ORDER[@]}"; do
    tag=${s%%|*}; extra=${s#*|}
    log "[r$rep] $tag ($extra)"
    # shellcheck disable=SC2086
    $BIN run "${COMMON[@]}" -tag "$tag" -rep 1 -rep-start "$rep" -seed "$rep" $extra || log "$tag FAILED"
  done
done

log "SSD baseline (end; compare with start to detect drift)"
$BIN baseline -file "$DATA" -dev "$DEV" -cpus "$CPUS" -cpuset "$SERVER" -bs 4k -qd "$QD" -dur "$BDUR" -out "$OUT/baseline_end.csv" ${M3_EXTRA:-} || log "baseline FAILED"
[ "$AIO_OK" = 1 ] && { $BIN baseline -engine aio -file "$DATA" -dev "$DEV" -cpus "$CPUS" -cpuset "$SERVER" -bs 4k -qd "$QD" -dur "$BDUR" -out "$OUT/baseline_end.csv" ${M3_EXTRA:-} || true; }
$BIN env -dev "$DEV" -file "$DATA" > "$OUT/env_after.txt" 2>&1
date > "$OUT/DONE"
log "finished -> $OUT   (copy this folder to your PC; see README)"
