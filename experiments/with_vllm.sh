#!/bin/bash
# Serve a model with vLLM, run one command against it, then stop the server.
#
#   bash experiments/with_vllm.sh <model-id-or-merged-dir> <served-name> <command...>
#   bash experiments/with_vllm.sh outputs/merged/llama-3.1-8B-collabscirem sciq-llama8b-collabscirem \
#       python experiments/03_sciquality/generate.py --model llama8b-collabscirem --split eval
#
# The server log goes to outputs/vllm_logs/<served-name>.log. PORT (default 8000) is
# passed through to experiments/serve_vllm.sh.

set -euo pipefail
MODEL=$1
NAME=$2
shift 2
cd "$(dirname "$0")/.."
mkdir -p outputs/vllm_logs
LOG=outputs/vllm_logs/$NAME.log
URL_FILE=outputs/vllm_endpoints/$NAME.url
KEY_FILE=outputs/vllm_endpoints/$NAME.key
rm -f "$URL_FILE"

setsid bash experiments/serve_vllm.sh "$MODEL" "$NAME" > "$LOG" 2>&1 &
PID=$!
trap 'kill -- -$PID 2>/dev/null || true; wait $PID 2>/dev/null || true' EXIT

echo "Waiting for vLLM ($NAME, log: $LOG) ..."
for _ in $(seq 1 180); do
    if [ -f "$URL_FILE" ] && curl -sf -H "Authorization: Bearer $(cat "$KEY_FILE")" \
            "$(cat "$URL_FILE")/models" > /dev/null; then
        break
    fi
    kill -0 "$PID" 2>/dev/null || { echo "vLLM exited early, see $LOG" >&2; exit 1; }
    sleep 10
done
"$@"
