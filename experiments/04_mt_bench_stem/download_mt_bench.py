"""
Download MT-Bench STEM questions to a local JSON file.

The output (data/mt_bench_stem.json) is already included in this supplementary;
this script documents how it was produced.

Usage:
    python experiments/04_mt_bench_stem/download_mt_bench.py
    python experiments/04_mt_bench_stem/download_mt_bench.py --output data/mt_bench_stem.json
    python experiments/04_mt_bench_stem/download_mt_bench.py --categories stem math
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download MT-Bench STEM questions locally.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/mt_bench_stem.json",
        help="Path to save the JSON file (default: data/mt_bench_stem.json)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["stem", "math"],
        help="MT-Bench categories to include (default: stem math)",
    )
    args = parser.parse_args()

    categories = {c.lower() for c in args.categories}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    questions = []

    # --- Read from HF hub cache (populated by `hf download hf://datasets/philschmid/mt-bench`) ---
    import glob
    cache_patterns = [
        "$HF_HOME/hub/datasets--philschmid--mt-bench/snapshots/*/question.jsonl",
        str(Path.home() / ".cache/huggingface/hub/datasets--philschmid--mt-bench/snapshots/*/question.jsonl"),
    ]
    jsonl_files = []
    for pat in cache_patterns:
        jsonl_files.extend(glob.glob(pat))

    if jsonl_files:
        jsonl_path = jsonl_files[0]
        print(f"Reading from cached JSONL: {jsonl_path}")
        with open(jsonl_path) as f:
            for line in f:
                row = json.loads(line)
                if row.get("category", "").lower() in categories:
                    questions.append(
                        {
                            "question_id": row["question_id"],
                            "category": row["category"],
                            "turns": list(row["turns"]),
                        }
                    )
        print(f"  Loaded {len(questions)} questions from cached JSONL.")
    else:
        # --- Fallback: load via HuggingFace datasets (requires internet) ---
        try:
            from datasets import load_dataset
            print("Cache not found. Trying philschmid/mt-bench via HuggingFace ...")
            ds = load_dataset("philschmid/mt-bench", split="train")
            for row in ds:
                if row.get("category", "").lower() in categories:
                    questions.append(
                        {
                            "question_id": row["question_id"],
                            "category": row["category"],
                            "turns": list(row["turns"]),
                        }
                    )
            print(f"  Loaded {len(questions)} questions from philschmid/mt-bench.")
        except Exception as e:
            print(f"  philschmid/mt-bench failed: {e}")

    if not questions:
        print("ERROR: Could not load MT-Bench from any HuggingFace source. Check your HF token and internet connection.")
        raise SystemExit(1)

    with open(output_path, "w") as f:
        json.dump(questions, f, indent=2)

    print(f"\nSaved {len(questions)} questions → {output_path}")
    print("Pass this path to the benchmark:")
    print(f'  run_mt_bench_stem_evaluation(..., dataset_path="{output_path}")')


if __name__ == "__main__":
    main()
