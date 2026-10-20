#!/bin/bash
# Synthetic research conversations + multiturn-aware reward annotation (Section 4, Figure 4).
#
#   bash experiments/01_data_generation/run.sh
#
# Needs OPENAI_API_KEY in .env. User simulator and assistant: gpt-4o-mini; critic for all
# nine metrics: gpt-5-mini. Seeds from the 700 training papers in data/ossas_papers.
# The released output of this step is data/collabscirem_train (forward-sampled sessions removed),
# so you only need to run it to regenerate the corpus.

set -euo pipefail
cd "$(dirname "$0")/../.."

python experiments/01_data_generation/build_dataset.py \
    --dataset_name scientific-papers-summarization \
    --metric_names accuracy completeness clarity interactivity \
                   justification logical_flow novelty format token_amount \
    --metric_weights 1.3 0.9 0.4 0.3 0.8 0.6 0.3 0.2 -0.4 \
    --user_generation_kwargs '{"model": "gpt-4o-mini"}' \
    --assistant_generation_kwargs '{"model": "gpt-4o-mini", "temperature": 0.8}' \
    --reward_generation_kwargs '{"model": "gpt-5-mini"}' \
    --output_dir outputs/multiturn_data \
    --train_size 700 --num_candidate_responses 3 "$@"
