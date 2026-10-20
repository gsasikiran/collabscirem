# 3. SciQuality

Paper: Section 4 ("SciQuality"), Figure 2, Appendix E (judge prompt and rubric in Table 7).

SciQuality holds 300 held-out papers, disjoint from the 700 training papers. For each paper, a
gpt-4o-mini researcher simulator holds a four-turn conversation with each model, seeded with
the paper's research question. All models get the same fixed opening message. DeepSeek-V4-Pro
(`deepseek/deepseek-v4-pro-0813`, OpenRouter, temperature 0) then compares two conversations
pairwise on seven dimensions, once in each response order. A model wins a prompt only if it is
preferred in both orders. The script reports win rates with 95% bootstrap CIs.

```bash
# full pipeline: generate (vLLM, one GPU), judge, table
bash experiments/03_sciquality/run.sh

# tables only, from the released conversations and judgments (no GPU, no API)
python experiments/03_sciquality/make_sciq_table.py     # -> results/tables/sci_quality_tables.tsv
python experiments/03_sciquality/analyze_full.py        # -> results/tables/sci_quality_results.md
```

`run.sh` writes to `outputs/sciquality/` (`SCIQ_OUT`), so the released files in
`results/sciquality/` are never overwritten. To build tables from your own run, set
`SCIQ_OUT=outputs/sciquality` for the table scripts too.

| File | Role |
|---|---|
| `common.py` | models (16), comparisons (12), judge, dimensions |
| `select_prompts.py` | how the 300 held-out prompts were selected (output: `results/sciquality/prompts.jsonl`) |
| `generate.py` | four-turn conversations against a vLLM endpoint |
| `judge.py` | pairwise judging in both orders |
| `make_sciq_table.py`, `analyze_full.py` | win rates, CIs, response lengths |
