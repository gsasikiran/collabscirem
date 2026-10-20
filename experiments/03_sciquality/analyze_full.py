"""
SciQuality results on the 300 held-out prompts (DeepSeek-V4-Pro judge). Win rate = CollabSciReM's mean pair score (win 1, tie 0.5,
loss 0), 95% CI by bootstrap over prompts.

  strict          CollabSciReM wins a dimension only if it wins in both orders
                  (primary; reported in the paper)
  order-averaged  mean of its scores in the two orders (robustness check)

Writes results/tables/sci_quality_results.md and prints it.
"""

import math
import os
import random
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import COMPARISONS, DIMENSIONS, JUDGMENTS, RIGOR, conv_path, read_jsonl

JUDGE = "deepseek"
OUT = "results/tables/sci_quality_results.md"
FLIP = {"A": "B", "B": "A", "tie": "tie"}
SCORE = {"A": 1.0, "B": 0.0, "tie": 0.5}
FAMILIES = ("llama3b", "llama8b", "qwen3b", "qwen7b")
B = 10000


def load(comp):
    last = {}
    for r in read_jsonl(os.path.join(JUDGMENTS, "eval", JUDGE, f"{comp}.jsonl")):
        if r.get("verdict"):
            last[(r["id"], r["order"])] = r
    ids = sorted({i for i, _ in last if (i, "XY") in last and (i, "YX") in last})
    return ids, last


def pair_scores(ids, last, dims):
    """Per prompt: (strict, averaged) score of X over the given dimensions."""
    out = {}
    for i in ids:
        xy, yx = last[(i, "XY")]["verdict"], last[(i, "YX")]["verdict"]
        s, a = [], []
        for d in dims:
            v1, v2 = xy[d], FLIP[yx[d]]
            s.append(1.0 if v1 == v2 == "A" else 0.0 if v1 == v2 == "B" else 0.5)
            a.append((SCORE[v1] + SCORE[v2]) / 2)
        out[i] = (st.mean(s), st.mean(a))
    return out


def ci(values, rng):
    means = sorted(st.mean(rng.choices(values, k=len(values))) for _ in range(B))
    return means[int(0.025 * B)], means[int(0.975 * B)]


def length_controlled(comp, x, y, scores, rng):
    """Markdown row for lc_stats."""
    raw, icept, lo, hi, slope, share = lc_stats(x, y, scores, rng)
    return f"| {comp} | {raw:.1%} | {icept:.1%} [{lo:.1%}, {hi:.1%}] | {slope:+.1%} | {share:.0%} |"


def lc_stats(x, y, scores, rng):
    """OLS of the strict rigor score on log(tokens x / tokens y) per prompt. Returns raw mean, intercept
    (win rate at equal length) with bootstrap 95% CI, slope per doubling, share of prompts where x is longer."""
    total = lambda m: {r["id"]: sum(t or 0 for t in r["assistant_tokens"]) for r in read_jsonl(conv_path(m, "eval"))}
    tx, ty = total(x), total(y)
    ids = [i for i in scores if tx.get(i) and ty.get(i)]
    lr = np.array([math.log(tx[i] / ty[i]) for i in ids])
    s = np.array([scores[i][0] for i in ids])

    def fit(idx):
        slope, icept = np.polyfit(lr[idx], s[idx], 1)
        return icept, slope

    icept, slope = fit(np.arange(len(ids)))
    boots = sorted(fit(np.array(rng.choices(range(len(ids)), k=len(ids))))[0] for _ in range(2000))
    return s.mean(), icept, boots[50], boots[1949], slope * math.log(2), (lr > 0).mean()


def verdict(m, lo, hi):
    if m > 0.5 and lo > 0.5:
        return "supported"
    if m < 0.5 and hi < 0.5:
        return "contradicted"
    return "inconclusive"


def main():
    rng = random.Random(2026)
    lines = ["# Scientific-quality evaluation — results", "",
             f"Judge: `deepseek/deepseek-v4-pro-0813` (single judge, see the 2026-10-07 amendment). "
             "Win rate = CollabSciReM's share of wins (ties count half); 50% = parity; 95% bootstrap CI over prompts.",
             ""]
    rig = {}
    summary = ["| Comparison | Hyp. | n | tokens/turn (CSR vs other) | order consistency | rigor strict | rigor order-avg | overall strict |",
               "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    detail = []
    lc_rows = []
    cost = 0.0
    for comp, (x, y, hyp) in COMPARISONS.items():
        ids, last = load(comp)
        cost += sum(r.get("cost") or 0 for r in read_jsonl(os.path.join(JUDGMENTS, "eval", JUDGE, f"{comp}.jsonl")))
        if not ids:
            summary.append(f"| {comp} | {hyp} | 0 | | | | | |")
            continue
        tok = lambda m: st.mean(t for r in read_jsonl(conv_path(m, "eval")) for t in r["assistant_tokens"] if t)
        cons = st.mean(last[(i, "XY")]["verdict"][d] == FLIP[last[(i, "YX")]["verdict"][d]]
                       for i in ids for d in DIMENSIONS)
        r = pair_scores(ids, last, RIGOR)
        o = pair_scores(ids, last, ["overall"])
        rig[comp] = r
        rs, ra, os_ = [v[0] for v in r.values()], [v[1] for v in r.values()], [v[0] for v in o.values()]
        (l1, h1), (l2, h2), (l3, h3) = ci(rs, rng), ci(ra, rng), ci(os_, rng)
        summary.append(f"| {comp} | {hyp} | {len(ids)} | {tok(x):.0f} vs {tok(y):.0f} | {cons:.0%} | "
                       f"{st.mean(rs):.1%} [{l1:.1%}, {h1:.1%}] | {st.mean(ra):.1%} [{l2:.1%}, {h2:.1%}] | "
                       f"{st.mean(os_):.1%} [{l3:.1%}, {h3:.1%}] |")
        detail += [f"### {comp} ({hyp}): {x} vs {y}, n = {len(ids)}", "",
                   "| Dimension | strict | order-averaged |", "| --- | --- | --- |"]
        for d in DIMENSIONS:
            p = pair_scores(ids, last, [d])
            s, a = [v[0] for v in p.values()], [v[1] for v in p.values()]
            (sl, sh), (al, ah) = ci(s, rng), ci(a, rng)
            detail.append(f"| {d} | {st.mean(s):.1%} [{sl:.1%}, {sh:.1%}] | {st.mean(a):.1%} [{al:.1%}, {ah:.1%}] |")
        detail.append("")
        lc_rows.append(length_controlled(comp, x, y, r, rng))

    lines += ["## Summary", ""] + summary + [""]
    lines += ["## Length-controlled rigor win rate (robustness check)", "",
              "Strict rigor score regressed (OLS) on log(CollabSciReM tokens / other tokens) per prompt; the "
              "intercept is the win rate at equal length. 'CSR longer' = share of prompts where CollabSciReM's "
              "conversation is the longer one; near 0% means the equal-length value is an extrapolation.", "",
              "| Comparison | raw | length-controlled [95% CI] | slope per doubling | CSR longer |",
              "| --- | --- | --- | --- | --- |"] + [l for l in lc_rows if l] + [""]

    # Pre-registered hypothesis checks on the strict rigor composite.
    hyp_lines = ["## Hypotheses (strict rigor composite)", ""]
    fams = [f"{f}-vs-collabllm" for f in FAMILIES if f"{f}-vs-collabllm" in rig]
    if len(fams) == 4:
        common_ids = [{i: v[0] for i, v in rig[c].items()} for c in fams]
        def pooled(sample):
            return st.mean(st.mean(s) for s in sample)
        point = pooled([list(d.values()) for d in common_ids])
        boots = sorted(pooled([rng.choices(list(d.values()), k=len(d)) for d in common_ids]) for _ in range(B))
        lo, hi = boots[int(0.025 * B)], boots[int(0.975 * B)]
        n_dir = sum(st.mean(d.values()) > 0.5 for d in common_ids)
        v = verdict(point, lo, hi)
        if v == "supported" and n_dir < 3:
            v = "inconclusive"
        hyp_lines.append(f"* **H1** CollabSciReM vs CollabLLM-science, pooled over families: {point:.1%} "
                         f"[{lo:.1%}, {hi:.1%}], direction favours CollabSciReM in {n_dir}/4 families → **{v}**")
    for comp, label in (("llama8b-vs-ablation2", "H2 vs R_gen + R_struct (Llama-8B)"),):
        if comp in rig:
            s = [v[0] for v in rig[comp].values()]
            lo, hi = ci(s, rng)
            hyp_lines.append(f"* **{label}**: {st.mean(s):.1%} [{lo:.1%}, {hi:.1%}] → **{verdict(st.mean(s), lo, hi)}**")
    for f in FAMILIES:
        comp = f"{f}-vs-base"
        if comp in rig:
            s = [v[0] for v in rig[comp].values()]
            lo, hi = ci(s, rng)
            hyp_lines.append(f"* **H3** vs base, {f}: {st.mean(s):.1%} [{lo:.1%}, {hi:.1%}] → **{verdict(st.mean(s), lo, hi)}**")
    lines += hyp_lines + ["", f"Judge cost: ${cost:.2f}", "", "## Per-dimension results", ""] + detail

    text = "\n".join(lines)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
