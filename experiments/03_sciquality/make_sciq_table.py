"""
Paper-layout tables of the scientific-quality evaluation (DeepSeek judge, 300
held-out prompts, strict both-orders rule): one row per model (base,
CollabLLM-science, CollabSciReM per family, then the Llama-3.1-8B ablations),
every row a head-to-head against CollabSciReM. Three views, each for the main
and the ablation rows:

  CSR view    CollabSciReM's win rate vs the row model (wins + ties/2), 95% CI;
              >50% = CollabSciReM judged better
  model view  the row model's own win rate vs CollabSciReM (= 100% - CSR view)
  counts      per dimension: prompts CollabSciReM won / tied / the row model won

    python experiments/03_sciquality/make_sciq_table.py     # writes results/tables/sci_quality_tables.tsv
"""

import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_full import FLIP, ci, lc_stats, load, pair_scores
from common import COMPARISONS, DIMENSIONS, RIGOR, conv_path, read_jsonl

OUT = "results/tables/sci_quality_tables.tsv"
LABELS = {"methodological_soundness": "Methodological soundness", "justification": "Justification",
          "logical_consistency": "Logical consistency", "factual_correctness": "Factual correctness",
          "novelty": "Novelty", "responsiveness": "Responsiveness", "overall": "Overall"}
FAMILIES = [("llama3b", "Llama-3.2-3B-Instruct"), ("llama8b", "Llama-3.1-8B-Instruct"),
            ("qwen3b", "Qwen-2.5-3B-Instruct"), ("qwen7b", "Qwen-2.5-7B-Instruct")]
ABLATIONS = [("ablation2", "R_gen + R_struct"), ("ablation3", "R_gen + R_sci"),
             ("ablation4", "R_struct + R_sci"), ("sensitivity", "CollabSciReM, sensitivity")]
RATE_COLS = [LABELS[d] for d in DIMENSIONS] + ["Rigor composite", "Rigor, length-controlled"]


def tokens(model):
    return f"{st.mean(t for r in read_jsonl(conv_path(model, 'eval')) for t in r['assistant_tokens'] if t):.0f}"


def stats(comp, rng):
    """Rates (mean, lo, hi) per column for CollabSciReM, the length-control flag, and W/T/L counts."""
    x, y, _ = COMPARISONS[comp]
    ids, last = load(comp)
    rates = []
    for d in DIMENSIONS:
        s = [v[0] for v in pair_scores(ids, last, [d]).values()]
        rates.append((st.mean(s), *ci(s, rng)))
    rig = pair_scores(ids, last, RIGOR)
    s = [v[0] for v in rig.values()]
    rates.append((st.mean(s), *ci(s, rng)))
    _, icept, lo, hi, _, share = lc_stats(x, y, rig, rng)
    rates.append((icept, lo, hi))
    counts = []
    for d in DIMENSIONS:
        w = t = l = 0
        for i in ids:
            a, b = last[(i, "XY")]["verdict"][d], FLIP[last[(i, "YX")]["verdict"][d]]
            if a == b == "A":
                w += 1
            elif a == b == "B":
                l += 1
            else:
                t += 1
        counts.append((w, t, l))
    return {"tokens": tokens(y), "rates": rates, "extrapolated": share < 0.05, "counts": counts}


def fmt(m, lo, hi):
    return f"{m:.1%} [{lo:.1%}, {hi:.1%}]"


def rate_cells(s, flip):
    cells = []
    for k, (m, lo, hi) in enumerate(s["rates"]):
        if flip:
            m, lo, hi = 1 - m, 1 - hi, 1 - lo
        cell = fmt(m, lo, hi)
        if k == len(s["rates"]) - 1 and s["extrapolated"]:
            cell += " (extrapolated)"
        cells.append(cell)
    return [s["tokens"]] + cells


def count_cells(s):
    return [s["tokens"]] + [str(v) for wtl in s["counts"] for v in wtl]


def block(title, group_row, col_row, rows):
    return [f"## {title}", "\t" + "\t".join(group_row), "Model\t" + "\t".join(col_row)] + \
           [label + "\t" + "\t".join(cells) for label, cells in rows] + [""]


def main():
    rng = random.Random(2026)
    main_s, abl_s = [], []
    for fam, base in FAMILIES:
        main_s += [(base, stats(f"{fam}-vs-base", rng)),
                   ("  CollabLLMSci", stats(f"{fam}-vs-collabllm", rng)),
                   ("  CollabSciReM", tokens(f"{fam}-collabscirem"))]
    abl_s = [(label, stats(f"llama8b-vs-{k}", rng)) for k, label in ABLATIONS]
    abl_s.append(("CollabSciReM", tokens("llama8b-collabscirem")))

    def rows(entries, cells_fn, n_ref):
        return [(label, cells_fn(s) if isinstance(s, dict) else [s] + ["reference"] * n_ref)
                for label, s in entries]

    rate_groups = ["Tokens/turn"] + RATE_COLS
    count_groups = ["Tokens/turn"] + [g for d in DIMENSIONS for g in (LABELS[d], "", "")]
    count_cols = ["Tokens/turn"] + ["CollabSciReM wins", "Ties", "This model wins"] * len(DIMENSIONS)
    src = "DeepSeek judge, 300 held-out prompts"
    lines = []
    for part, entries in (("main", main_s), ("ablation / sensitivity, Llama-3.1-8B", abl_s)):
        lines += block(f"Scientific quality — {part} (CollabSciReM win rate vs this model; >50% = CollabSciReM "
                       f"judged better; {src})", rate_groups, rate_groups,
                       rows(entries, lambda s: rate_cells(s, False), len(RATE_COLS)))
    for part, entries in (("main", main_s), ("ablation / sensitivity, Llama-3.1-8B", abl_s)):
        lines += block(f"Sci quality model view — {part} (this model's win rate vs CollabSciReM; >50% = this "
                       f"model judged better; {src})", rate_groups, rate_groups,
                       rows(entries, lambda s: rate_cells(s, True), len(RATE_COLS)))
    for part, entries in (("main", main_s), ("ablation / sensitivity, Llama-3.1-8B", abl_s)):
        lines += block(f"Sci quality counts — {part} (number of the 300 prompts won by each side; tie = tie "
                       f"or the judge's verdict changed when the order was swapped; {src})",
                       count_groups, count_cols, rows(entries, count_cells, len(count_cols) - 1))
    text = "\n".join(lines)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
