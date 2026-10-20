# CollabSciReM

Code, data and evaluation outputs for **CollabSciReM: Aligning Language Models for Scientific
Conversations via Science-Aware Reward Decomposition**.

CollabSciReM is a goal-conditioned, session-level reward for multi-turn research conversations.
It decomposes the reward into three parts:

* **Conversational quality** (`R_gen`): accuracy, completeness, clarity and interactivity.
* **Scientific rigor** (`R_sci`): justification, logical flow and novelty.
* **Structural efficiency** (`R_struct`): format and a token penalty.

We fine-tune Llama and Qwen models on 8.62K synthetic research conversations with SFT followed
by offline IPO. We evaluate them on SciQuality, a new pairwise judged benchmark built from 300
held-out papers, and on MT-Bench-STEM, MINT, GPQA-Diamond, MMLU-Validation and TheoremQA.

## Repository layout

```
collabscirem/                 library
├── reward.py, simulation.py, synthetic.py   multiturn-aware reward, forward sampling (§3.2)
├── metrics/                  the nine reward metrics and critic prompts (§3.3, Appendix F.3)
├── training/reward_presets.py  reward decomposition and weights (Table 3, ablations)
├── datasets/                 multi-turn corpus → SFT / preference pairs
├── data_sources/             scientific papers → seed prompts
├── prompts/                  user simulator and assistant prompts (Appendix F.1–F.2)
└── benchmarks/               GPQA-D, MMLU, TheoremQA, MT-Bench-STEM loaders and prompts
experiments/                  one folder per experiment in the paper
├── 01_data_generation/       synthetic conversations + reward annotation (§4, Fig. 4)
├── 02_training/              SFT → offline IPO → merge (§4, Appendix B)
├── 03_sciquality/            SciQuality pairwise evaluation (§4, Fig. 2, Appendix E)
├── 04_mt_bench_stem/         MT-Bench-STEM (§5, Tables 1a/6a)
├── 05_mint/                  MINT (§5, Tables 1a/6a)
├── 06_single_turn/           GPQA-D, MMLU-Valid, TheoremQA (§5, Tables 1b/6b)
├── 07_ablations/             reward ablations and sensitivity (§6, Table 2)
├── 08_user_study/            expert user study (§7, Appendix C)
├── models.sh                 every evaluated model and its path
├── serve_vllm.sh, with_vllm.sh  vLLM helpers for the multi-turn evaluations
data/                         training corpus, source papers, MT-Bench-STEM questions
results/                      released evaluation outputs and tables
```

Each `experiments/*/README.md` maps its scripts to the paper and gives the exact commands.

## Setup

```bash
git clone <this-repo> && cd CollabSciReM
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # pinned; tested with Python 3.13, PyTorch 2.8, vLLM 0.11, TRL 0.24
pip install -e .
cp .env.example .env                # OPENAI_API_KEY, OPENROUTER_API_KEY
```

Run every command from the repository root. Outputs go to `outputs/`.

## Released model and dataset

* **Model:** [gsasikiran/collabscirem-Llama-3.1-8B-Instruct](https://huggingface.co/gsasikiran/collabscirem-Llama-3.1-8B-Instruct)
  is Llama-3.1-8B-Instruct trained with CollabSciReM (SFT followed by offline IPO).
* **Dataset:** [gsasikiran/collab-scirem](https://huggingface.co/datasets/gsasikiran/collab-scirem)
  holds the synthetic multi-turn research conversations used for training.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("gsasikiran/collabscirem-Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("gsasikiran/collabscirem-Llama-3.1-8B-Instruct")
dataset = load_dataset("gsasikiran/collab-scirem")
```

## Reproducing the paper

```bash
# 1. data (optional: the corpus is already in data/)
bash experiments/01_data_generation/run.sh

# 2. training: the 8 main models (Llama-3.2-3B, Llama-3.1-8B, Qwen-2.5-3B, Qwen-2.5-7B
#    x {CollabSciReM, CollabLLMSci}) and the Llama-3.1-8B ablations
bash experiments/02_training/train_all.sh
bash experiments/07_ablations/train_ablations.sh

# 3. evaluation
bash experiments/03_sciquality/run.sh          # Figure 2
bash experiments/04_mt_bench_stem/run.sh       # Tables 1a, 2a, 6a
bash experiments/05_mint/setup.sh && bash experiments/05_mint/run.sh
bash experiments/06_single_turn/run.sh         # Tables 1b, 2b, 6b
```


### Tables from the released outputs (no GPU, no API)

```bash
python experiments/03_sciquality/make_sciq_table.py        # Figure 2  -> results/tables/sci_quality_tables.tsv
python experiments/04_mt_bench_stem/make_mtbench_table.py  # MT-Bench-STEM -> results/tables/mtbench_tables.tsv
```

* `results/sciquality/conversations/eval/<model>.jsonl` holds the four-turn conversations on all 300 held-out papers for 16 models: base, CollabLLMSci and CollabSciReM for each family, plus the Llama-3.1-8B ablations.
* `results/sciquality/judgments/eval/deepseek/<comparison>.jsonl` holds the judge verdicts in both orders, with reasons and listed factual errors.
* `results/mt_bench_stem/` holds the dialogues for seeds 42–44 and their per-turn judge scores.

## Acknowledgements

The multi-turn simulation, multiturn-aware reward and training scaffolding in this repository
are adapted from [CollabLLM](https://github.com/Wuyxin/collabllm) (Wu et al., 2025), released
under the MIT License.

## License

The code is released under MIT (see `LICENSE`). Data derived from OSSAS keeps the license of the source corpus.

## Citation

If you use this code, the released model
([collabscirem-Llama-3.1-8B-Instruct](https://huggingface.co/gsasikiran/collabscirem-Llama-3.1-8B-Instruct))
or the dataset ([collab-scirem](https://huggingface.co/datasets/gsasikiran/collab-scirem)), please cite:

```bibtex
@inproceedings{collabscirem2026,
    title  = {CollabSciReM: Aligning Language Models for Scientific Conversations via
              Science-Aware Reward Decomposition},
    author = {Sasi Kiran Gaddipati, Gollam Rabby, Sören Auer},
    year   = {2026}
}
```
