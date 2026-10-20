#!/bin/bash
# SFT -> offline IPO -> merge for one model (paper Section 4, Appendix B).
#
#   bash experiments/02_training/train.sh <reward-preset> <base-model> <run-name>
#   bash experiments/02_training/train.sh collabscirem meta-llama/Llama-3.1-8B-Instruct llama-3.1-8B-collabscirem
#
# Reward presets are defined in collabscirem/training/reward_presets.py. The merged model of the
# LAST IPO checkpoint is written to outputs/merged/<run-name>; that is the model evaluated.
# NPROC sets the number of GPUs/processes (default 1, as used for the reported models).

set -euo pipefail
PRESET=$1
BASE=$2
RUN=$3
DATA=${DATA:-data/collabscirem_train}
NPROC=${NPROC:-1}
PORT=${MASTER_PORT:-29500}
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

last_ckpt() { ls -d "$1"/checkpoint-* | sort -t- -k2 -n | tail -1; }

torchrun --nproc_per_node="$NPROC" --master_port="$PORT" experiments/02_training/sft.py \
    --reward "$PRESET" --dataset_repo "$DATA" --output_dir "outputs/sft/$RUN" \
    --model_name "$BASE" --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 --num_train_epochs 7 --learning_rate 1e-5 \
    --eval_steps 1 --logging_steps 1 --use_lora --use_4bit

torchrun --nproc_per_node="$NPROC" --master_port="$PORT" experiments/02_training/offline_dpo.py \
    --reward "$PRESET" --dataset_repo "$DATA" --model_name "$(last_ckpt "outputs/sft/$RUN")" \
    --output_dir "outputs/offline_dpo/$RUN" --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 --gradient_accumulation_steps 4 --save_total_limit 10 \
    --num_train_epochs 8 --learning_rate 5e-6 --eval_steps 10 --logging_steps 1 --use_lora

python experiments/02_training/merge_lora.py --adapter "$(last_ckpt "outputs/offline_dpo/$RUN")" \
    --output "outputs/merged/$RUN"
