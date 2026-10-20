# 5. MINT

Paper: Section 5, Tables 1a, 2a and 6a (MINT columns).

MINT (Wang et al., 2024b) runs on its TheoremQA, MATH and MMLU subsets with MINT's default
setting: k = 5 interaction turns, tools enabled, and gpt-4o-mini textual feedback without
ground truth. The model under test is served by vLLM with a 4,096-token cap per turn. Each model
runs with seeds 1, 2 and 3. Reported metrics are SR@5 (problems solved within five turns), TC
(tokens per response) and APT (SR@5 / TC × 1000).

```bash
bash experiments/05_mint/setup.sh     # clone MINT @ 3f7f12c, apply mint-bench.patch, create .venv-mint
bash experiments/05_mint/run.sh       # needs one GPU and OPENAI_API_KEY
```

`run.sh` ends by writing `results/tables/mint_table_mt4096{,_ablation}.{md,tex}`.

`mint-bench.patch` adds the following to MINT:

* an OpenAI-compatible vLLM backend
* per-run seeds
* the per-turn token cap
* a no-tool mode, which is not used in the paper
