#!/usr/bin/env bash
# Calibrate the CPU cost of m3bench's vector search against FAISS on THIS server.
#   ./calibrate.sh     -> calib.env (used automatically by run_all.sh), ~10-15 min
# Three search-cost levels (single-thread FAISS HNSW time on this server), see calib_choose.py:
#   light : synthetic data (easy) at recall@10 >= 0.95        -> lower bound
#   main  : real embeddings (NYTimes-256, ~315 MB) at recall@10 >= 0.90, scaled to 500k x 384
#   heavy : real embeddings at recall@10 >= 0.95
# If the real data cannot be downloaded, main falls back to the synthetic target (flagged).
# FAISS/numpy/h5py go into ~/m3venv (your user only; remove with: rm -rf ~/m3venv ~/m3data/ann). No sudo.
set -u -o pipefail
cd "$(dirname "$0")"
BIN=./m3bench
CPUS=${M3_CPUS:-7}
DEV=${M3_DEV:-auto}
RECALL=${M3_RECALL:-0.95}
log() { echo "[$(date +%H:%M:%S)] $*"; }
# everything we download or cache stays inside ~/m3venv (removed by cleanup.sh); nothing goes to ~/.cache or /tmp
export XDG_CACHE_HOME=~/m3venv/cache HF_HOME=~/m3venv/hf CUDA_CACHE_PATH=~/m3venv/nv PIP_NO_CACHE_DIR=1

SET=$($BIN cpus -cpus "$((CPUS + 1))" -dev "$DEV") || exit 1
SERVER=$(python3 -c "
c=[]
for p in '$SET'.split(','):
    a,_,b=p.partition('-'); c+=range(int(a),int(b or a)+1)
print(','.join(map(str,c[:$CPUS])))")
ONE=${SERVER%%,*}
log "server CPUs $SERVER; single-thread timing on CPU $ONE"

PY=python3
if ! $PY -c "import faiss, numpy, h5py" 2>/dev/null; then
  if [ -x ~/m3venv/bin/python ] && ~/m3venv/bin/python -c "import faiss, numpy, h5py" 2>/dev/null; then
    PY=~/m3venv/bin/python
  else
    log "installing faiss-cpu, numpy, h5py into ~/m3venv (user-only)"
    [ -x ~/m3venv/bin/python ] || python3 -m venv ~/m3venv 2>/dev/null || python3 -m venv --without-pip ~/m3venv || { log "cannot create venv"; exit 1; }
    if [ ! -x ~/m3venv/bin/pip ]; then
      ~/m3venv/bin/python -c "import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', '$HOME/m3venv/get-pip.py')" \
        || { log "cannot download get-pip.py (no network?) -> calibration skipped, run_all.sh uses defaults"; exit 1; }
      ~/m3venv/bin/python ~/m3venv/get-pip.py -q || exit 1
    fi
    ~/m3venv/bin/pip install -q --no-cache-dir faiss-cpu numpy h5py || { log "pip install failed -> calibration skipped, run_all.sh uses defaults"; exit 1; }
    PY=~/m3venv/bin/python
  fi
fi

log "1) Go search cost grid (single thread)"
rm -f calib_go.csv
$BIN searchbench -cpuset "$ONE" -nlists 256,512,1024,2048,4096 -nprobes 1,2,4,8,16,32,64 -out calib_go.csv || exit 1
log "2) FAISS on the synthetic data (light target)"
$PY calibrate_faiss.py --data synth --cpus "$SERVER" --recall "$RECALL" --out calib_faiss_synth.csv ${M3_FAISS_ARGS:-} | tee calib_faiss.log || exit 1
log "3) FAISS on real embeddings (main target)"
rm -f calib_faiss_real.csv
$PY calibrate_faiss.py --data nytimes --cpus "$SERVER" --recall "$RECALL" --out calib_faiss_real.csv | tee -a calib_faiss.log \
  || log "real-data calibration failed (download blocked?) -> main target falls back to synthetic"

log "4) choosing Go (nlist, nprobe) for each target  (re-run alone: python3 calib_choose.py)"
python3 calib_choose.py --light-recall "$RECALL" --heavy-recall "$RECALL" --main-recall "${M3_MAIN_RECALL:-0.90}" || exit 1
log "done -> calib.env (run_all.sh picks it up)"
