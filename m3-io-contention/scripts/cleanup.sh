#!/usr/bin/env bash
# Put the server back the way it was before the M3 experiment.
# Removes ONLY what the experiment created; nothing else in the home folder or on the server is touched.
#   ./cleanup.sh        -> removes data, installs and temp files; keeps ~/m3bench (code + results)
#   ./cleanup.sh --all  -> also removes ~/m3bench and ~/m3bench.zip (copy results_* to your PC first!)
# It lists everything and asks before deleting.
set -u
ALL=0; [ "${1:-}" = "--all" ] && ALL=1

if pgrep -u "$(id -u)" -f "m3bench (run|baseline|prep|searchbench)|run_all.sh|calibrate|llm_probe" >/dev/null; then
  echo "An experiment is still running:"; pgrep -u "$(id -u)" -af "m3bench|run_all.sh|calibrate|llm_probe"
  echo "Stop it first (pkill -f run_all.sh; pkill m3bench) or wait for it to finish."; exit 1
fi

T=(~/m3data ~/m3venv ~/env_check.txt ~/env_check2.txt ~/ds_before.txt ~/ds_after.txt ~/m3d1 ~/m3d2 ~/m3bt.csv /tmp/get-pip-m3.py)
[ "$ALL" = 1 ] && T+=(~/m3bench ~/m3bench.zip)

echo "Will delete (only items created by the M3 experiment):"
FOUND=()
for p in "${T[@]}"; do
  if [ -e "$p" ]; then FOUND+=("$p"); printf '  %-40s %s\n' "$p" "$(du -sh "$p" 2>/dev/null | cut -f1)"; fi
done
[ ${#FOUND[@]} -eq 0 ] && { echo "  (nothing left - already clean)"; exit 0; }
if [ "$ALL" = 1 ] && ls -d ~/m3bench/results_* >/dev/null 2>&1; then
  echo "WARNING: ~/m3bench contains results folders. Copy them to your PC first:"; ls -d ~/m3bench/results_*
fi
read -r -p "Delete these? [y/N] " ans
[ "$ans" = y ] || [ "$ans" = Y ] || { echo "cancelled"; exit 0; }
for p in "${FOUND[@]}"; do rm -rf -- "$p" && echo "removed $p"; done
echo "free space now:"; df -h ~ | tail -1
[ "$ALL" = 1 ] || echo "kept ~/m3bench (code + results). When results are on your PC: ./cleanup.sh --all"
