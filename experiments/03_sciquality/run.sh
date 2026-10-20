#!/bin/bash
# SciQuality (Section 4, Figure 2, Appendix E): four-turn conversations on the 300 held-out
# papers with a gpt-4o-mini researcher simulator, then pairwise DeepSeek-V4-Pro judging in
# both response orders, then win rates with 95% bootstrap CIs.
#
#   bash experiments/03_sciquality/run.sh                  # all 16 models, all comparisons
#   KEYS="llama8b-base llama8b-collabscirem" bash experiments/03_sciquality/run.sh
#
# Needs one GPU (vLLM), OPENAI_API_KEY (simulator) and OPENROUTER_API_KEY (judge).
# The fixed researcher openings ship in results/sciquality/openings.jsonl. Outputs go to
# outputs/sciquality (SCIQ_OUT); finished conversations are skipped, so the script resumes.

set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; [ -f .env ] && source .env; set +a
SQ=experiments/03_sciquality
export SCIQ_OUT=${SCIQ_OUT:-outputs/sciquality}   # keep fresh runs apart from results/

KEYS=${KEYS:-$(python -c "import sys; sys.path.insert(0, '$SQ'); from common import MODELS; print(' '.join(MODELS))")}
for key in $KEYS; do
    model=$(python -c "import sys; sys.path.insert(0, '$SQ'); from common import MODELS; print(MODELS['$key'])")
    bash experiments/with_vllm.sh "$model" "sciq-$key" \
        python $SQ/generate.py --model "$key" --split eval
done

python $SQ/judge.py --judge deepseek --split eval
python $SQ/make_sciq_table.py
