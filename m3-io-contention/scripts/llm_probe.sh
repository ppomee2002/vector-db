#!/usr/bin/env bash
# Optional: measure LLM generation time on one of the server's GPUs (NVIDIA A2), to put
# retrieval latency in context. Installs torch + transformers into ~/m3venv (~3 GB download,
# your user only) and downloads a small model (~1 GB, Qwen2.5-0.5B-Instruct) to ~/.cache.
# Uses the GPU that has the least memory in use and refuses to run if someone else is using it.
#   ./llm_probe.sh [results_dir]      (writes llm_probe.csv into results_dir, default: .)
set -u -o pipefail
cd "$(dirname "$0")"
OUT=${1:-.}
log() { echo "[$(date +%H:%M:%S)] $*"; }
# everything we download or cache stays inside ~/m3venv (removed by cleanup.sh); nothing goes to ~/.cache or /tmp
export XDG_CACHE_HOME=~/m3venv/cache HF_HOME=~/m3venv/hf CUDA_CACHE_PATH=~/m3venv/nv PIP_NO_CACHE_DIR=1
command -v nvidia-smi >/dev/null || { log "no nvidia-smi: no GPU in this container"; exit 1; }
read -r GPU USED < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | tr -d ' ' | tr ',' ' ')
if [ "${USED:-0}" -gt 500 ] && [ "${M3_FORCE:-0}" != 1 ]; then
  log "GPU $GPU already has ${USED} MiB in use (another team?). Not running. M3_FORCE=1 to override."; exit 1
fi
[ -x ~/m3venv/bin/python ] || python3 -m venv ~/m3venv 2>/dev/null || python3 -m venv --without-pip ~/m3venv || exit 1
if [ ! -x ~/m3venv/bin/pip ]; then
  ~/m3venv/bin/python -c "import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', '$HOME/m3venv/get-pip.py')" || exit 1
  ~/m3venv/bin/python ~/m3venv/get-pip.py -q || exit 1
fi
if ! ~/m3venv/bin/python -c "import torch, transformers" 2>/dev/null; then
  log "installing torch (CUDA 12.4 build) + transformers into ~/m3venv"
  ~/m3venv/bin/pip install -q --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 || exit 1
  ~/m3venv/bin/pip install -q --no-cache-dir transformers accelerate || exit 1
fi
log "running on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU ~/m3venv/bin/python llm_probe.py --out "$OUT/llm_probe.csv" "${@:2}"
