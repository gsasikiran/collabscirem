"""
Select the held-out research questions for the scientific-quality evaluation
(paper Section 4).

NOTE: selection needs the full processed OSSAS corpus (32.7K rows), which is not part of
this supplementary; its output for the 300 evaluation prompts is results/sciquality/prompts.jsonl.

Candidates: rows 700+ of data/ossas_papers (training
used rows 0-699), SCIENTIFIC_TEXT, both fields present, research question 50-600
tokens. Candidates whose research question has cosine > 0.90 to any training
research question (Qwen3-Embedding-0.6B) are dropped, as are repeated titles.
330 prompts are drawn with seed 2026, stratified by discipline (the part of
`field_subfield` before the dash); the first 30 of a seeded shuffle form the pilot
set, the other 300 the evaluation set.

    python experiments/03_sciquality/select_prompts.py      # writes results/sciquality/prompts.jsonl
"""

import json
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np
import tiktoken
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

SEED = 2026
N_TOTAL, N_PILOT = 330, 30
POOL = 3000             # stratified candidate pool that is embedded and deduplicated
SIM_MAX = 0.90
OUT = "results/sciquality/prompts.jsonl"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def discipline(field_subfield):
    head = re.split(r"\s+[—–-]\s+", field_subfield or "")[0].strip()
    return head or "Unspecified"


def stratified(items, key, n, rng):
    """Proportional allocation (largest remainder), seeded sample within each stratum."""
    groups = defaultdict(list)
    for it in items:
        groups[key(it)].append(it)
    total = len(items)
    quotas = {g: n * len(v) / total for g, v in groups.items()}
    alloc = {g: int(q) for g, q in quotas.items()}
    for g in sorted(quotas, key=lambda g: quotas[g] - alloc[g], reverse=True)[: n - sum(alloc.values())]:
        alloc[g] += 1
    picked = []
    for g in sorted(groups):
        picked += rng.sample(groups[g], min(alloc[g], len(groups[g])))
    return picked


@torch.no_grad()
def embed(texts, tok, model, bs=32):
    out = []
    for i in range(0, len(texts), bs):
        batch = tok(texts[i:i + bs], padding=True, truncation=True, max_length=512, return_tensors="pt")
        h = model(**batch).last_hidden_state
        # Qwen3-Embedding: last-token pooling with left padding.
        emb = torch.nn.functional.normalize(h[:, -1], dim=-1)
        out.append(emb.float().numpy())
    return np.concatenate(out)


def main():
    torch.set_num_threads(16)
    enc = tiktoken.get_encoding("cl100k_base")
    rows = load_dataset("data/ossas_papers")["train"]

    train_q = [r["research_question_and_hypothesis"] or "" for r in rows.select(range(700))]
    train_titles = {(r["title"] or "").strip().lower() for r in rows.select(range(700))}

    cands, seen = [], set()
    for i in range(700, len(rows)):
        r = rows[i]
        q, m = r["research_question_and_hypothesis"], r["methodological_details"]
        if r["article_classification"] != "SCIENTIFIC_TEXT" or not q or not m:
            continue
        title = (r["title"] or "").strip().lower()
        if title in train_titles or title in seen:
            continue
        if not 50 <= len(enc.encode(q)) <= 600:
            continue
        seen.add(title)
        cands.append({"row": i, "title": r["title"], "discipline": discipline(r["field_subfield"]),
                      "field_subfield": r["field_subfield"], "research_question": q,
                      "reference_methodology": m})
    print(f"{len(cands)} candidates after filters")

    rng = random.Random(SEED)
    pool = stratified(cands, lambda c: c["discipline"], POOL, rng)

    tok = AutoTokenizer.from_pretrained(EMBED_MODEL, padding_side="left")
    model = AutoModel.from_pretrained(EMBED_MODEL, torch_dtype=torch.float32).eval()
    e_train = embed(train_q, tok, model)
    e_pool = embed([c["research_question"] for c in pool], tok, model)
    max_sim = (e_pool @ e_train.T).max(axis=1)
    kept = [c | {"max_sim_to_train": round(float(s), 4)} for c, s in zip(pool, max_sim) if s <= SIM_MAX]
    print(f"pool {len(pool)}, dropped {len(pool) - len(kept)} with cosine > {SIM_MAX} to training")

    chosen = stratified(kept, lambda c: c["discipline"], N_TOTAL, rng)
    rng.shuffle(chosen)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for k, c in enumerate(chosen):
            f.write(json.dumps({"id": f"sciq-{c['row']}", "split": "pilot" if k < N_PILOT else "eval", **c}) + "\n")
    print(f"wrote {len(chosen)} prompts to {OUT}")
    print("disciplines:", Counter(c["discipline"] for c in chosen).most_common(15))
    print(f"max cosine to training: max {max(c['max_sim_to_train'] for c in chosen):.3f}, "
          f"median {np.median([c['max_sim_to_train'] for c in chosen]):.3f}")


if __name__ == "__main__":
    main()
