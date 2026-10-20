"""
Standalone MT-Bench-STEM runner.

Decoupled from run_benchmark_eval.py so MT-Bench-STEM can be scheduled and
iterated on independently of the other benchmarks (GPQA/MMLU/TheoremQA).

Each model is evaluated with several seeds (default 3) at temperature 0.7.
All seed runs for a model are written to a single dialogue JSON file so the
offline judge can compute mean/std of the score and token counts across seeds.

Offline workflow
----------------
1. With internet access (already done; the file ships in data/):
       python experiments/04_mt_bench_stem/download_mt_bench.py
          → writes data/mt_bench_stem.json

2. GPU (no internet needed):
       python experiments/04_mt_bench_stem/run_mt_bench_eval.py
          → reads  data/mt_bench_stem.json
          → writes outputs/mt_bench_stem/<model>_dialogues.json
            {model_id, temperature, seeds, runs: [{seed, avg_token_count, dialogues}]}

3. With internet access (judge API; the paper uses DeepSeek-V4-Pro via OpenRouter):
       python experiments/04_mt_bench_stem/score_mt_bench_with_judge.py \\
           --dialogues outputs/mt_bench_stem/<model>_dialogues.json \\
           --judge openrouter/deepseek/deepseek-v4-pro-0813 --suffix _deepseek
          → writes outputs/mt_bench_stem/<model>_scores_deepseek.json
            (per-seed scores + across-seed mean/std of score and token count)
"""

import argparse
import json
import os
import re
import sys

import numpy as np

from collabscirem.benchmarks.mt_bench_stem import (
    _load_model_and_tokenizer,
    run_mt_bench_stem_evaluation,
)

MODELS = [   # paper models; fine-tuned ones are merged last IPO checkpoints (README, "Training")
    "meta-llama/Llama-3.2-3B-Instruct",
    "outputs/merged/llama-3.2-3B-collabllmsci",
    "outputs/merged/llama-3.2-3B-collabscirem",
    "meta-llama/Llama-3.1-8B-Instruct",
    "outputs/merged/llama-3.1-8B-collabllmsci",
    "outputs/merged/llama-3.1-8B-collabscirem",
    "Qwen/Qwen2.5-3B-Instruct",
    "outputs/merged/qwen-2.5-3B-collabllmsci",
    "outputs/merged/qwen-2.5-3B-collabscirem",
    "Qwen/Qwen2.5-7B-Instruct",
    "outputs/merged/qwen-2.5-7B-collabllmsci",
    "outputs/merged/qwen-2.5-7B-collabscirem",
    "outputs/merged/llama-3.1-8B-ablation-gen_struct",
    "outputs/merged/llama-3.1-8B-ablation-gen_sci",
    "outputs/merged/llama-3.1-8B-ablation-struct_sci",
    "outputs/merged/llama-3.1-8B-collabllm-sciw",
    "outputs/merged/llama-3.1-8B-sensitivity",
]

# Output file names, matching results/mt_bench_stem/ and make_mtbench_table.py
BASE_NAMES = {
    "meta-llama/Llama-3.2-3B-Instruct": "llama-3.2-3B-instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "llama-3.1-8B-instruct",
    "Qwen/Qwen2.5-3B-Instruct": "qwen-2.5-3B-instruct",
    "Qwen/Qwen2.5-7B-Instruct": "qwen-2.5-7B-instruct",
}


def short_name(model_id: str) -> str:
    return BASE_NAMES.get(model_id, os.path.basename(model_id.rstrip("/")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MT-Bench-STEM evaluation")
    parser.add_argument("--model-index", type=int, default=None, help="Index of the model to evaluate (overrides SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Seeds for each run (one inference pass per seed)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--categories", type=str, nargs="+", default=["stem", "math"], help="MT-Bench categories to include")
    parser.add_argument("--dataset-path", type=str, default=os.environ.get("MT_BENCH_DATASET_PATH", "data/mt_bench_stem.json"), help="Pre-downloaded dataset JSON (see experiments/04_mt_bench_stem/download_mt_bench.py)")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--output-dir", type=str, default="outputs/mt_bench_stem", help="Directory for per-model dialogue JSON files")
    parser.add_argument("--summary-file", type=str, default="mt_bench_results_confidence_intervals.json", help="Aggregated token-count summary across all evaluated models")
    parser.add_argument("--system-prompt-file", type=str, default=None, help="Replace the default system prompt, e.g. collabscirem/prompts/proact_system_prompt.txt for the proactive-prompting baseline")
    parser.add_argument("--run-tag", type=str, default="", help="Suffix added to output file names, e.g. 'proactive'")
    parser.add_argument("--model", type=str, default=None, help="Evaluate this model ID/path instead of the MODELS list")
    args = parser.parse_args()

    categories = set(args.categories)
    system_prompt = None
    if args.system_prompt_file:
        with open(args.system_prompt_file) as f:
            system_prompt = f.read().strip()
    tag_suffix = f"_{args.run_tag}" if args.run_tag else ""

    chosen_index = None
    if args.model_index is not None:
        chosen_index = args.model_index
    else:
        slurm_tid = os.environ.get("SLURM_ARRAY_TASK_ID")
        if slurm_tid is not None:
            try:
                chosen_index = int(slurm_tid)
            except Exception:
                chosen_index = None

    if args.model:
        target_models = [args.model]
    elif chosen_index is None:
        target_models = MODELS
    else:
        if chosen_index < 0 or chosen_index >= len(MODELS):
            print(f"SLURM/--model-index {chosen_index} out of range for models list (len={len(MODELS)}). Exiting.")
            sys.exit(2)
        target_models = [MODELS[chosen_index]]

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    for model_id in target_models:
        print(f"\n=== Evaluating Model: {model_id} ===")

        # Load the model once and reuse it across all seeds.
        try:
            model, tokenizer = _load_model_and_tokenizer(model_id)
        except Exception as e:
            print(f"[MT-Bench] Skipping '{model_id}': failed to load ({e}).")
            continue

        runs = []
        token_counts = []
        for seed in args.seeds:
            print(f"\n--- Seed {seed} ---")
            dialogues, avg_tokens = run_mt_bench_stem_evaluation(
                model_id=model_id,
                categories=categories,
                dataset_path=args.dataset_path,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=seed,
                model=model,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
            )
            runs.append({
                "seed": seed,
                "avg_token_count": avg_tokens,
                "dialogues": dialogues,
            })
            token_counts.append(avg_tokens)

        # Free GPU memory before loading the next model.
        del model, tokenizer
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        # ---- Write one dialogue file holding all seeds for this model ----
        sanitized = re.sub(r"[^\w\-.]", "_", short_name(model_id))
        dialogue_path = os.path.join(args.output_dir, f"{sanitized}{tag_suffix}_dialogues.json")
        with open(dialogue_path, "w") as f:
            json.dump({
                "model_id": model_id,
                "run_tag": args.run_tag,
                "system_prompt_file": args.system_prompt_file,
                "temperature": args.temperature,
                "seeds": args.seeds,
                "runs": runs,
            }, f, indent=2)
        print(f"\nDialogues (all seeds) → {dialogue_path}")

        results[model_id] = {
            "seeds": args.seeds,
            "temperature": args.temperature,
            "mean_token_count": float(np.mean(token_counts)) if token_counts else None,
            "std_token_count": float(np.std(token_counts)) if len(token_counts) > 1 else 0.0,
            "runs_token_count": token_counts,
            "dialogue_file": dialogue_path,
        }

        tok = results[model_id]
        tok_str = f"{tok['mean_token_count']:.1f} ± {tok['std_token_count']:.1f}" if tok["mean_token_count"] is not None else "N/A"
        print(f"\n=== Token summary for {model_id} ===")
        print(f"  MT-Bench-STEM tokens/turn: {tok_str}")
        print(f"  Score with: python experiments/04_mt_bench_stem/score_mt_bench_with_judge.py --dialogues {dialogue_path} --judge openrouter/deepseek/deepseek-v4-pro-0813 --suffix _deepseek")

    with open(args.summary_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nToken summary saved to {args.summary_file}")


if __name__ == "__main__":
    main()
