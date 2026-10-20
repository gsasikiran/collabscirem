#!/bin/bash
# Train the Llama-3.1-8B reward ablations and the sensitivity run (Section 6, Table 2).
# The full CollabSciReM model comes from experiments/02_training/train_all.sh.
#
#   bash experiments/07_ablations/train_ablations.sh

set -euo pipefail
cd "$(dirname "$0")/../.."
BASE=meta-llama/Llama-3.1-8B-Instruct

bash experiments/02_training/train.sh gen_struct        $BASE llama-3.1-8B-ablation-gen_struct   # without R_sci
bash experiments/02_training/train.sh gen_sci           $BASE llama-3.1-8B-ablation-gen_sci      # without R_struct
bash experiments/02_training/train.sh struct_sci        $BASE llama-3.1-8B-ablation-struct_sci   # without R_gen
bash experiments/02_training/train.sh collabllm_sciw    $BASE llama-3.1-8B-collabllm-sciw        # CollabLLM metrics, CollabSciReM weights
bash experiments/02_training/train.sh collabscirem_x0.7 $BASE llama-3.1-8B-sensitivity           # all weights x 0.7
