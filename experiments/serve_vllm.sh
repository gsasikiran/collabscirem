#!/bin/bash
# Serve a model with vLLM for the multi-turn evaluations (MINT, SciQuality).
#
#   bash experiments/serve_vllm.sh <model-id-or-merged-dir> <served-name>
#   bash experiments/serve_vllm.sh outputs/merged/qwen-2.5-7B-collabscirem qwen-2.5-7B-collabscirem
#
# Writes outputs/vllm_endpoints/<served-name>.url and .key, which experiments/05_mint/run_mint_eval.py
# and experiments/03_sciquality/generate.py read. SciQuality serves models as "sciq-<key>"
# (keys in experiments/03_sciquality/common.py).

set -euo pipefail
MODEL=$1
NAME=$2
PORT=${PORT:-8000}
cd "$(dirname "$0")/.."

ENDPOINTS=outputs/vllm_endpoints
mkdir -p "$ENDPOINTS"
API_KEY=$(python -c "import secrets; print(secrets.token_hex(24))")
( umask 077; echo "$API_KEY" > "$ENDPOINTS/$NAME.key" )
URL="http://$(hostname):$PORT/v1"
echo "$URL" > "$ENDPOINTS/$NAME.url"
cleanup() {
    if [ "$(cat "$ENDPOINTS/$NAME.url" 2>/dev/null)" = "$URL" ]; then
        rm -f "$ENDPOINTS/$NAME.url" "$ENDPOINTS/$NAME.key"
    fi
}
trap cleanup EXIT TERM INT

vllm serve "$MODEL" \
    --served-model-name "$NAME" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --api-key "$API_KEY" \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --seed 42
