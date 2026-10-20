#!/bin/bash
# GPQA-Diamond, MMLU-Validation and TheoremQA, pass@1, seeds 42-44
# (Section 5, Tables 1b, 2b and 6b). One GPU; no API access needed.
#
#   bash experiments/06_single_turn/run.sh                 # all models
#   MODELS=main bash experiments/06_single_turn/run.sh     # main | ablation | all

set -euo pipefail
cd "$(dirname "$0")/../.."
source experiments/models.sh
select_models

for entry in "${SELECTED[@]}"; do
    name=${entry%%=*}; model=${entry#*=}
    for bench in gpqa mmlu theoremqa; do
        for seed in 42 43 44; do
            out=outputs/single_turn/$name/${bench}_seed$seed.json
            [ -f "$out" ] && continue
            python experiments/06_single_turn/run_single_turn_eval.py \
                --model "$model" --benchmark $bench --seed $seed --out "$out"
        done
    done
done

python experiments/06_single_turn/make_single_turn_table.py
