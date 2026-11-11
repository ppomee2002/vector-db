#!/usr/bin/env bash
# Build the Linux binary and pack what the server needs into dist/m3bench.zip.
# On the server everything sits in one folder: m3bench (binary), the scripts, src/ (Go source), README.md.
#   ./scripts/package.sh   (needs Go 1.24+; the binary is static, so the server needs no Go)
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf dist
mkdir -p dist/m3bench/src
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags "-s -w" -o dist/m3bench/m3bench .
for f in scripts/*.sh scripts/*.py; do
  [ "$(basename "$f")" = package.sh ] || cp "$f" dist/m3bench/
done
cp ./*.go go.mod dist/m3bench/src/
cp README.md dist/m3bench/
(cd dist && python3 -m zipfile -c m3bench.zip m3bench)
echo "built dist/m3bench.zip"
