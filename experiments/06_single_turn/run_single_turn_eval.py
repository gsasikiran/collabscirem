"""
One single-turn benchmark run (one model, one benchmark, one seed) on the GPU in
CUDA_VISIBLE_DEVICES, written to a JSON file. Same benchmark code and settings as
run_benchmark_eval.py, split per seed so each run fits on one GPU within the 24 h
job limit.

    python experiments/06_single_turn/run_single_turn_eval.py --model outputs/merged/<run> \
        --benchmark mmlu --seed 42 --out outputs/single_turn/<run>/mmlu_seed42.json

Exits non-zero when the benchmark returns no questions (the benchmark functions
return zeros instead of raising on a load error).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from collabscirem.benchmarks.gpqa_diamond import run_gpqa_evaluation
from collabscirem.benchmarks.mmlu import run_mmlu_evaluation
from collabscirem.benchmarks.theoremqa import run_theorem_qa_evaluation

RUNNERS = {
    "gpqa": lambda m, s: run_gpqa_evaluation(model_id=m, dataset_id="fingertap/GPQA-Diamond", num_samples=198, seed=s),
    "mmlu": lambda m, s: run_mmlu_evaluation(model_id=m, dataset_id="cais/mmlu", data_split="validation", seed=s),
    "theoremqa": lambda m, s: run_theorem_qa_evaluation(model_id=m, seed=s),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--benchmark", required=True, choices=sorted(RUNNERS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    accuracy, total, correct, avg_tokens = RUNNERS[args.benchmark](args.model, args.seed)
    if not total:
        sys.exit(f"{args.benchmark}: no questions evaluated for {args.model}")
    result = {"model": args.model, "benchmark": args.benchmark, "seed": args.seed,
              "accuracy": accuracy, "total": total, "correct": correct, "average_token_count": avg_tokens}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out + ".tmp", "w") as f:
        json.dump(result, f, indent=2)
    os.replace(args.out + ".tmp", args.out)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
