"""
Named reward settings for ranking candidate responses in SFT and offline IPO
(paper Section 3.3, Table 3, and the ablations in Section 6 / Table 2).

Each preset is a weighted sum over the per-metric rewards stored in every row of the
training corpus (`row["rewards"][metric]`, a list over forward-sampled sessions that is
averaged). A metric missing from a row raises instead of silently counting as 0.

    --reward collabscirem   R_gen + R_sci + R_struct; equals the stored `score` / rewards.MR
    --reward score          use the row's precomputed `score` as is
"""

import numpy as np

# Table 3, CollabSciReM column
GEN = {"accuracy": 1.3, "completeness": 0.9, "clarity": 0.4, "interactivity": 0.3}
SCI = {"justification": 0.8, "logical_flow": 0.6, "novelty": 0.3}
STRUCT = {"format": 0.2, "token_amount": -0.4}
FULL = {**GEN, **SCI, **STRUCT}

PRESETS = {
    # Main models (Table 1, Table 6, Figure 2)
    "collabscirem": FULL,
    # Science-agnostic baseline CollabLLMSci: Table 3, CollabLLM column
    "collabllm": {"accuracy": 1.0, "interactivity": 0.5, "token_amount": -0.5},
    # Ablations (Table 2): one reward family removed at a time
    "gen_struct": {**GEN, **STRUCT},      # without R_sci
    "gen_sci": {**GEN, **SCI},            # without R_struct
    "struct_sci": {**SCI, **STRUCT},      # without R_gen
    # Table 2, "CollabLLMSci | gamma_CSciReM": the CollabLLM metrics with the
    # CollabSciReM weights (the reported model was trained without the token term)
    "collabllm_sciw": {"accuracy": 1.3, "interactivity": 0.3},
    # Table 2, sensitivity: all CollabSciReM weights scaled by 0.7
    "collabscirem_x0.7": {m: 0.7 * w for m, w in FULL.items()},
    # User study (Section 7): general-reward variant, Table 3 "User study" column
    "general": {"accuracy": 1.0, "interactivity": 0.5},
}


def _mean(x):
    a = np.asarray(x, dtype=float)
    return float(a.mean()) if a.size else 0.0


def make_score_fn(name):
    """Score function for a preset, or None for "score" (use the stored field)."""
    if name == "score":
        return None
    weights = PRESETS[name]

    def score_fn(row):
        rewards = row["rewards"]
        missing = [m for m in weights if m not in rewards]
        if missing:
            raise KeyError(f"reward preset {name!r} needs {missing}, row has {sorted(rewards)}")
        return sum(w * _mean(rewards[m]) for m, w in weights.items())

    return score_fn
