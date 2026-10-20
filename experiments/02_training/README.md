# 2. Training: SFT → offline IPO → merge

Paper: Section 4 ("Fine-Tuning Base Models"), Appendix A (Table 3), Appendix B (Table 4).

All models are trained on the same corpus (`data/collabscirem_train`) with the same recipe.
They differ only in the **reward preset** that ranks candidate responses:

* SFT keeps, for every conversation, the final-turn candidate with the highest reward.
* IPO pairs the highest- and lowest-reward candidates of each turn. Pairs whose gap is below 0.02 are dropped.

Presets are defined in [`collabscirem/training/reward_presets.py`](../../collabscirem/training/reward_presets.py):

| Model | Preset | Weights (Table 3) |
|---|---|---|
| CollabSciReM | `collabscirem` | acc 1.3, compl 0.9, clarity 0.4, inter 0.3, just 0.8, logic 0.6, novelty 0.3, format 0.2, tokens −0.4 |
| CollabLLMSci | `collabllm` | acc 1.0, inter 0.5, tokens −0.5 |

The ablation presets are listed in [`../07_ablations`](../07_ablations).

```bash
# one model
bash experiments/02_training/train.sh collabscirem meta-llama/Llama-3.1-8B-Instruct llama-3.1-8B-collabscirem

# the eight main models (4 base models x {CollabSciReM, CollabLLMSci})
bash experiments/02_training/train_all.sh
```

`train.sh` merges the **last** IPO checkpoint into `outputs/merged/<run-name>`. Every
evaluation uses that merged model. Don't use the top-level adapter in
`outputs/offline_dpo/<run-name>/`: `load_best_model_at_end` makes it the best checkpoint by eval loss, not the last one.

| Setting | SFT | IPO |
|---|---|---|
| LoRA r / α / targets | 32 / 16 / all linear | 32 / 16 / all linear |
| Quantization | 4-bit NF4 | none (bf16) |
| Learning rate, schedule | 1e-5, cosine | 5e-6, cosine |
| Epochs | 7 | 8 |
| Batch per device × grad. accum. | 4 × 4 | 4 × 4 |
| Loss | cross-entropy | IPO, β = 0.2 |
| Max sequence length | 4096 | 4096 |

The reported models were trained with one process per run (`NPROC=1`, effective batch 16).
Set `NPROC=4` to spread a run over 4 GPUs; this changes the effective batch size.
