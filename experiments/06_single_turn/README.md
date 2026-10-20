# 6. Single-turn scientific reasoning

Paper: Section 5, Tables 1b, 2b and 6b.

The benchmarks are GPQA-Diamond, MMLU-Validation and TheoremQA, with image-based instances
excluded. Every model is evaluated zero-shot with free generation, using the same prompt,
decoding parameters, 4,096-token budget and answer-extraction function. We report pass@1
accuracy, tokens per response (TC) and accuracy per token (APT = Acc / TC × 1000), each as
mean ± sample std over seeds 42–44.

```bash
bash experiments/06_single_turn/run.sh              # one GPU; MODELS=main|ablation|all
# or one run:
python experiments/06_single_turn/run_single_turn_eval.py --model outputs/merged/llama-3.1-8B-collabscirem \
    --benchmark gpqa --seed 42 --out outputs/single_turn/llama-3.1-8B-collabscirem/gpqa_seed42.json
python experiments/06_single_turn/make_single_turn_table.py   # -> results/tables/single_turn_table.tsv
```

The benchmark loaders, prompts and answer extraction live in
[`collabscirem/benchmarks/`](../../collabscirem/benchmarks).
