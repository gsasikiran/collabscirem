#!/bin/bash
# Fetch MINT at the pinned commit, apply our patch (vLLM endpoint, seeds, token cap,
# no-tool mode) and create its own environment (MINT needs openai==0.28).
#
#   bash experiments/05_mint/setup.sh

set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p third_party
if [ ! -d third_party/mint-bench ]; then
    git clone https://github.com/xingyaoww/mint-bench third_party/mint-bench
fi
cd third_party/mint-bench
git checkout 3f7f12c
git apply --check ../../experiments/05_mint/mint-bench.patch 2>/dev/null \
    && git apply ../../experiments/05_mint/mint-bench.patch \
    || echo "patch already applied"
python -m venv .venv-mint
.venv-mint/bin/pip install -r requirements.txt
