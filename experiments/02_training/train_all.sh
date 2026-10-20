#!/bin/bash
# Train the eight main models: CollabSciReM and CollabLLMSci for each of the four base
# models (Table 1, Table 6, Figure 2). Each call runs SFT -> offline IPO -> merge.
#
#   bash experiments/02_training/train_all.sh            # all eight, one after another
#   ONLY=llama-3.1-8B bash experiments/02_training/train_all.sh
#
# On a cluster, submit the individual train.sh calls as separate jobs instead.

set -euo pipefail
cd "$(dirname "$0")/../.."

declare -A BASES=(
    [llama-3.2-3B]=meta-llama/Llama-3.2-3B-Instruct
    [llama-3.1-8B]=meta-llama/Llama-3.1-8B-Instruct
    [qwen-2.5-3B]=Qwen/Qwen2.5-3B-Instruct
    [qwen-2.5-7B]=Qwen/Qwen2.5-7B-Instruct
)

for fam in llama-3.2-3B llama-3.1-8B qwen-2.5-3B qwen-2.5-7B; do
    [ -n "${ONLY:-}" ] && [ "$ONLY" != "$fam" ] && continue
    bash experiments/02_training/train.sh collabscirem "${BASES[$fam]}" "$fam-collabscirem"
    bash experiments/02_training/train.sh collabllm    "${BASES[$fam]}" "$fam-collabllmsci"
done
