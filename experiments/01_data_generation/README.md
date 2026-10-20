# 1. Synthetic data generation and reward annotation

Paper: Section 4 ("Synthetic Data Generation", "Reward Annotation"), Figure 1, Figure 4, Appendix F.

Each of the 700 training papers in `data/ossas_papers` (rows 0–699) seeds a research
conversation. A simulated researcher (gpt-4o-mini) asks about the paper's research question,
and the assistant (gpt-4o-mini) proposes methodology. At every assistant turn, three
candidate responses are sampled. For each candidate, the conversation is rolled forward
(window of up to 4 turns), and the critic (gpt-5-mini) scores the nine metrics of Table 3 on the
continuations. The stored per-metric rewards are the multiturn-aware rewards of Eq. 5.

```bash
bash experiments/01_data_generation/run.sh      # needs OPENAI_API_KEY in .env
```

The output goes to `outputs/multiturn_data/`. The released result of this step is
`data/collabscirem_train/` (8,624 rows; the forward-sampled sessions are dropped to keep it
small). Training needs only the stored rewards, so **you can skip this step**.

| File | Role |
|---|---|
| `build_dataset.py` | generation and annotation driver |
| `../../collabscirem/data_sources/summarized_papers.py` | turns structured papers into seed prompts |
| `../../collabscirem/synthetic.py`, `simulation.py`, `reward.py` | forward sampling and the multiturn-aware reward |
| `../../collabscirem/metrics/` | the nine reward metrics and their critic prompts (Appendix F.3) |
| `../../collabscirem/prompts/` | user simulator and assistant prompts (Appendix F.1–F.2) |
