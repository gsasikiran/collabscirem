#!/bin/bash
# MT-Bench-STEM: the 20 two-turn STEM and math questions, seeds 42-44, scored 1-10 by
# DeepSeek-V4-Pro on overall, accuracy, format, usefulness and relevance
# (Section 5, Tables 1a, 2a and 6a). Generation needs one GPU; scoring needs
# OPENROUTER_API_KEY.
#
#   bash experiments/04_mt_bench_stem/run.sh
#   MODELS=main bash experiments/04_mt_bench_stem/run.sh    # main | ablation | all

set -euo pipefail
cd "$(dirname "$0")/../.."
source experiments/models.sh
select_models
set -a; [ -f .env ] && source .env; set +a
OUT=outputs/mt_bench_stem

for entry in "${SELECTED[@]}"; do
    model=${entry#*=}
    python experiments/04_mt_bench_stem/run_mt_bench_eval.py --model "$model" \
        --seeds 42 43 44 --output-dir $OUT --summary-file $OUT/token_summary_$(basename "$model").json
done

python experiments/04_mt_bench_stem/score_mt_bench_with_judge.py --dialogues $OUT/*_dialogues.json \
    --judge openrouter/deepseek/deepseek-v4-pro-0813 --suffix _deepseek --workers 8
python experiments/04_mt_bench_stem/make_mtbench_table.py --dir $OUT
