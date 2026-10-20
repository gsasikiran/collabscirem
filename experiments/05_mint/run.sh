#!/bin/bash
# MINT, SR@5 on the TheoremQA / MATH / MMLU subsets, 4096-token cap per turn, seeds 1-3
# (Section 5, Tables 1a, 2a and 6a). Needs one GPU (vLLM) and OPENAI_API_KEY (gpt-4o-mini
# feedback provider).
#
#   bash experiments/05_mint/setup.sh                       # once
#   bash experiments/05_mint/run.sh                         # all models
#   MODELS=main bash experiments/05_mint/run.sh             # main | ablation | all

set -euo pipefail
cd "$(dirname "$0")/../.."
source experiments/models.sh
select_models
set -a; [ -f .env ] && source .env; set +a

for entry in "${SELECTED[@]}"; do
    name=${entry%%=*}; model=${entry#*=}
    bash experiments/with_vllm.sh "$model" "$name" bash -c "
        for seed in 1 2 3; do
            python experiments/05_mint/run_mint_eval.py --served-name $name --max-tokens 4096 --seed \$seed
        done"
done

python experiments/05_mint/make_mint_table.py --variant mt4096
python experiments/05_mint/make_mint_table.py --variant mt4096 --ablation
