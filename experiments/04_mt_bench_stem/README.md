# 4. MT-Bench-STEM

Paper: Section 5, Tables 1a, 2a and 6a (MT-Bench-STEM columns).

The benchmark is the 20 two-turn questions from the STEM and Math categories of MT-Bench
(`data/mt_bench_stem.json`). Each model answers with seeds 42, 43 and 44. DeepSeek-V4-Pro scores
every response from 1 to 10 on overall quality, accuracy, format, usefulness and relevance.
TC is the average number of tokens per turn.

```bash
bash experiments/04_mt_bench_stem/run.sh       # generation (GPU) + judging (OPENROUTER_API_KEY) + table

# table only, from the released judge scores
python experiments/04_mt_bench_stem/make_mtbench_table.py    # -> results/tables/mtbench_tables.tsv
```

| File | Role |
|---|---|
| `download_mt_bench.py` | builds `data/mt_bench_stem.json` (already included) |
| `run_mt_bench_eval.py` | two-turn generation, writes `<model>_dialogues.json` |
| `score_mt_bench_with_judge.py` | judge scoring, writes `<model>_scores_deepseek.json` |
| `make_mtbench_table.py` | mean ± sample std over seeds; `--dir` selects the score directory |
| `../../collabscirem/benchmarks/mt_bench_stem.py`, `prompts/mt_bench_stem.txt` | generation loop and judge prompt |
