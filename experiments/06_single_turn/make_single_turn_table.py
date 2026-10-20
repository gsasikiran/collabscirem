"""
Single-turn table (Tables 1b, 2b and 6a): accuracy, tokens per response (TC) and
accuracy per token (APT = Acc / TC x 1000), mean ± sample std over seeds 42-44.

Reads outputs/single_turn/<model>/<benchmark>_seed<N>.json written by
experiments/06_single_turn/run_single_turn_eval.py, one directory per model (the basename of --model).

    python experiments/06_single_turn/make_single_turn_table.py     # writes results/tables/single_turn_table.tsv
"""

import glob
import json
import os
import statistics as st

ROOT = "outputs/single_turn"
OUT = "results/tables/single_turn_table.tsv"
BENCHES = [("gpqa", "GPQA-D"), ("mmlu", "MMLU-Valid"), ("theoremqa", "TheoremQA")]


def pm(vals, fmt):
    if not vals:
        return "—"
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{fmt.format(st.mean(vals))} ± {fmt.format(sd)}"


def cells(model, bench):
    runs = [json.load(open(f)) for f in sorted(glob.glob(f"{ROOT}/{model}/{bench}_seed*.json"))]
    acc = [r["accuracy"] for r in runs]  # already in %
    tc = [r["average_token_count"] for r in runs]
    apt = [a / t * 1000 for a, t in zip(acc, tc)]
    return [pm(acc, "{:.2f}"), pm(tc, "{:.2f}"), pm(apt, "{:.2f}")]


def main():
    models = sorted(os.path.basename(d) for d in glob.glob(f"{ROOT}/*") if os.path.isdir(d))
    header = ["Model"] + [f"{name} {m}" for _, name in BENCHES for m in ("Acc (%)", "TC", "APT")]
    lines = ["\t".join(header)]
    for model in models:
        lines.append("\t".join([model] + sum((cells(model, b) for b, _ in BENCHES), [])))
    text = "\n".join(lines)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
