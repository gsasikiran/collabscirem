"""
Merge a PEFT/LoRA adapter into its base model so it can be served by vLLM.

Usage (on a GPU or CPU node, offline HF cache):
    python experiments/02_training/merge_lora.py \
        --adapter outputs/offline_dpo/qwen-2.5-7B-collabscirem/checkpoint-1816 \
        --output  outputs/merged/qwen-2.5-7B-collabscirem

The base model is read from the adapter's adapter_config.json unless --base is given.
"""

import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model")
    parser.add_argument("--adapter", required=True, help="Adapter directory containing adapter_config.json")
    parser.add_argument("--output", required=True, help="Directory for the merged model")
    parser.add_argument("--base", default=None, help="Override base model ID/path")
    args = parser.parse_args()

    with open(os.path.join(args.adapter, "adapter_config.json")) as f:
        base = args.base or json.load(f)["base_model_name_or_path"]

    print(f"Loading base model {base}")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    print(f"Applying adapter {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    # The adapter directory carries the tokenizer/chat template used during training.
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.output)
    print(f"Merged model saved to {args.output}")


if __name__ == "__main__":
    main()
