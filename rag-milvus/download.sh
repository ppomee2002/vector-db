#!/usr/bin/env bash
# BEIR (HF mirror, parquet) + QASPER. Raw text only; embedding is a separate step.
set -euo pipefail
cd "$(dirname "$0")"
HF=https://huggingface.co/datasets/BeIR
get() { [ -s "$2" ] || { curl -fsSL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"; }; }
ls_parquet() {  # $1 = repo/path
  curl -fs "https://huggingface.co/api/datasets/BeIR/$1" | python3 -c '
import json,sys
for x in json.load(sys.stdin):
    if x["path"].endswith((".parquet", ".tsv")): print(x["path"])'
}
for ds in scifact fiqa nq hotpotqa; do
  mkdir -p "hf/$ds"
  for part in corpus queries; do
    for f in $(ls_parquet "$ds/tree/main/$part"); do
      get "$HF/$ds/resolve/main/$f" "hf/$ds/$(basename "$f")"
    done
  done
  for f in $(ls_parquet "$ds-qrels/tree/main"); do
    get "$HF/$ds-qrels/resolve/main/$f" "hf/$ds/qrels-$f"
  done
  echo "done $ds $(du -sh "hf/$ds" | cut -f1)"
done
get https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz qasper.tgz
mkdir -p qasper && tar -xzf qasper.tgz -C qasper && echo "done qasper"
find hf qasper.tgz -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
echo ALL_DONE
