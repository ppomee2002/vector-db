#!/bin/bash
# 환경 준비: ~/m4-exp 안에만 설치 (시스템/다른 사용자 영향 없음)
set -eo pipefail
D=~/m4-exp; mkdir -p $D/results $D/data $D/qdrant
cd $D
if [ ! -x venv/bin/python ]; then
  if ! python3 -m venv venv 2>/dev/null; then
    echo "[info] python3-venv 없음 -> uv(사용자 홈 설치)로 venv 생성"
    rm -rf venv
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 UV_INSTALL_DIR=$D/uv sh
    $D/uv/uv venv venv --python 3.12 --seed
  fi
fi
source venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -q numpy sentence-transformers qdrant-client matplotlib
cd qdrant
if [ ! -x ./qdrant ]; then
  curl -sL -o q.tgz https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz
  tar xzf q.tgz && rm -f q.tgz
fi
./qdrant --version
echo "SETUP OK"
