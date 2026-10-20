"""
MT-Bench-STEM tables (Tables 1, 2 and 6): DeepSeek judge, 20 STEM/math questions x
2 turns, seeds 42-44. Reads <dir>/<model>_scores_deepseek.json (default: the released
scores in results/mt_bench_stem; pass outputs/mt_bench_stem for your own run).

  main / ablation           mean ± sample std over seeds for every dimension, tokens/turn
  per seed main / ablation  seed 42 / 43 / 44 values for every dimension

    python experiments/04_mt_bench_stem/make_mtbench_table.py [--dir outputs/mt_bench_stem]   # writes results/tables/mtbench_tables.tsv
"""

import argparse
import json
import os
import statistics as st

D = "results/mt_bench_stem"
OUT = "results/tables/mtbench_tables.tsv"
DIMS = ["overall", "accuracy", "format", "usefulness", "relevance"]

MAIN = [
    ("Llama-3.2-3B-Instruct", "llama-3.2-3B-instruct"),
    ("  CollabLLMSci", "llama-3.2-3B-collabllmsci"),
    ("  CollabSciReM", "llama-3.2-3B-collabscirem"),
    ("Llama-3.1-8B-Instruct", "llama-3.1-8B-instruct"),
    ("  CollabLLMSci", "llama-3.1-8B-collabllmsci"),
    ("  CollabSciReM", "llama-3.1-8B-collabscirem"),
    ("Qwen-2.5-3B-Instruct", "qwen-2.5-3B-instruct"),
    ("  CollabLLMSci", "qwen-2.5-3B-collabllmsci"),
    ("  CollabSciReM", "qwen-2.5-3B-collabscirem"),
    ("Qwen-2.5-7B-Instruct", "qwen-2.5-7B-instruct"),
    ("  CollabLLMSci", "qwen-2.5-7B-collabllmsci"),
    ("  CollabSciReM", "qwen-2.5-7B-collabscirem"),
]
ABLATION = [
    ("R_gen + R_struct", "llama-3.1-8B-ablation-gen_struct"),
    ("R_gen + R_sci", "llama-3.1-8B-ablation-gen_sci"),
    ("R_struct + R_sci", "llama-3.1-8B-ablation-struct_sci"),
    ("CollabLLMSci, CollabSciReM weights", "llama-3.1-8B-collabllm-sciw"),
    ("CollabSciReM, sensitivity", "llama-3.1-8B-sensitivity"),
    ("CollabSciReM", "llama-3.1-8B-collabscirem"),
]


def load(key):
    path = f"{D}/{key}_scores_deepseek.json"
    return json.load(open(path)) if os.path.exists(path) else None


def mean_std(vals):
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def summary_cells(d):
    if d is None:
        return ["—"] * (len(DIMS) + 1)
    cells = []
    for m in DIMS:
        mu, sd = mean_std([p[m] for p in d["per_seed_scores"] if p.get(m) is not None])
        cells.append(f"{mu:.2f} ± {sd:.2f}")
    mu, sd = mean_std([t for t in d["per_seed_token_count"] if t is not None])
    return cells + [f"{mu:.0f} ± {sd:.0f}"]


def seed_cells(d):
    if d is None:
        return ["—"] * (3 * len(DIMS))
    return [f"{p[m]:.2f}" if p.get(m) is not None else "—" for m in DIMS for p in d["per_seed_scores"]]


def block(title, groups, cols, rows):
    return [f"## {title}", "\t" + "\t".join(groups), "Model\t" + "\t".join(cols)] + \
           [label + "\t" + "\t".join(cells) for label, cells in rows] + [""]


def main():
    global D
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=D)
    D = ap.parse_args().dir
    names = [m.capitalize() for m in DIMS]
    note = "DeepSeek judge, 20 STEM/math questions x 2 turns, seeds 42-44, score 1-10"
    lines = []
    for part, rows in (("main", MAIN), ("ablation / sensitivity, Llama-3.1-8B", ABLATION)):
        lines += block(f"MT-Bench-STEM — {part} (mean ± sample std over seeds; {note})",
                       names + ["Tokens/turn"], names + ["Tokens/turn"],
                       [(label, summary_cells(load(k))) for label, k in rows])
    seed_groups = [g for n in names for g in (n, "", "")]
    seed_cols = ["seed 42", "seed 43", "seed 44"] * len(DIMS)
    for part, rows in (("main", MAIN), ("ablation / sensitivity, Llama-3.1-8B", ABLATION)):
        lines += block(f"MT-Bench-STEM per seed — {part} ({note})", seed_groups, seed_cols,
                       [(label, seed_cells(load(k))) for label, k in rows])
    text = "\n".join(lines)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
