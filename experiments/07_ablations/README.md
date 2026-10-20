# 7. Ablations and sensitivity

Paper: Section 6, Table 2. All runs fine-tune Llama-3.1-8B-Instruct with the recipe from
[`../02_training`](../02_training) and change only the reward preset.

| Table 2 row | Preset | Run name |
|---|---|---|
| R_gen + R_struct (no R_sci) | `gen_struct` | `llama-3.1-8B-ablation-gen_struct` |
| R_gen + R_sci (no R_struct) | `gen_sci` | `llama-3.1-8B-ablation-gen_sci` |
| R_struct + R_sci (no R_gen) | `struct_sci` | `llama-3.1-8B-ablation-struct_sci` |
| CollabLLMSci with CollabSciReM weights | `collabllm_sciw` | `llama-3.1-8B-collabllm-sciw` |
| CollabSciReM, all weights × 0.7 | `collabscirem_x0.7` | `llama-3.1-8B-sensitivity` |

```bash
bash experiments/07_ablations/train_ablations.sh

# evaluate them with the same benchmark scripts
MODELS=ablation bash experiments/06_single_turn/run.sh
MODELS=ablation bash experiments/05_mint/run.sh
MODELS=ablation bash experiments/04_mt_bench_stem/run.sh
KEYS="llama8b-ablation2 llama8b-ablation3 llama8b-ablation4 llama8b-sensitivity" \
    bash experiments/03_sciquality/run.sh
```
