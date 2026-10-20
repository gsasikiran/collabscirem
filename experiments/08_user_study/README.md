# 8. User study

Paper: Section 7, Figure 3, Appendix C (Table 5, Figure 5).

In a blind study on Prolific, 25 domain experts each held a scientific dialogue of at least
three turns with three anonymized models. They then rated each model on logical justification
(1–10), interaction quality (1–10) and novelty (1–5), and reported whether it hallucinated. The
three models were:

| Model in the study | How to obtain it |
|---|---|
| Base model | `meta-llama/Llama-3.1-8B-Instruct` |
| General-reward variant | `bash experiments/02_training/train.sh general meta-llama/Llama-3.1-8B-Instruct llama-3.1-8B-general` (accuracy 1.0, interactivity 0.5, no token penalty; Table 3, "User study" column) |
| CollabSciReM | `outputs/merged/llama-3.1-8B-collabscirem` |

The anonymized interaction logs and ratings go in this directory. Participant identifiers
have been removed.
