"""Shared configuration for the SciQuality evaluation (paper Section 4, Appendix E)."""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RELEASED = os.path.join(ROOT, "results", "sciquality")
# Inputs (the 300 held-out prompts and the fixed researcher openings) always come from the
# released files. Conversations and judgments are read from / written to SCIQ_OUT, which
# defaults to the released outputs; run.sh sets it to outputs/sciquality for a fresh run.
PROMPTS = os.path.join(RELEASED, "prompts.jsonl")
OPENINGS = os.path.join(RELEASED, "openings.jsonl")
OUT = os.path.abspath(os.environ.get("SCIQ_OUT", RELEASED))
CONVS = os.path.join(OUT, "conversations")
JUDGMENTS = os.path.join(OUT, "judgments")

# key -> model path; served under the name "sciq-<key>".
# Fine-tuned models are the merged last IPO checkpoints (see README, "Training").
MODELS = {
    "llama3b-base": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3b-collabllm": "outputs/merged/llama-3.2-3B-collabllmsci",
    "llama3b-collabscirem": "outputs/merged/llama-3.2-3B-collabscirem",
    "llama8b-base": "meta-llama/Llama-3.1-8B-Instruct",
    "llama8b-collabllm": "outputs/merged/llama-3.1-8B-collabllmsci",
    "llama8b-collabscirem": "outputs/merged/llama-3.1-8B-collabscirem",
    "qwen3b-base": "Qwen/Qwen2.5-3B-Instruct",
    "qwen3b-collabllm": "outputs/merged/qwen-2.5-3B-collabllmsci",
    "qwen3b-collabscirem": "outputs/merged/qwen-2.5-3B-collabscirem",
    "qwen7b-base": "Qwen/Qwen2.5-7B-Instruct",
    "qwen7b-collabllm": "outputs/merged/qwen-2.5-7B-collabllmsci",
    "qwen7b-collabscirem": "outputs/merged/qwen-2.5-7B-collabscirem",
    "llama8b-ablation2": "outputs/merged/llama-3.1-8B-ablation-gen_struct",
    "llama8b-ablation3": "outputs/merged/llama-3.1-8B-ablation-gen_sci",
    "llama8b-ablation4": "outputs/merged/llama-3.1-8B-ablation-struct_sci",
    "llama8b-sensitivity": "outputs/merged/llama-3.1-8B-sensitivity",
}

# name -> (model X = CollabSciReM, model Y, group)
COMPARISONS = {}
for fam in ("llama3b", "llama8b", "qwen3b", "qwen7b"):
    COMPARISONS[f"{fam}-vs-collabllm"] = (f"{fam}-collabscirem", f"{fam}-collabllm", "H1")
    COMPARISONS[f"{fam}-vs-base"] = (f"{fam}-collabscirem", f"{fam}-base", "H3")
COMPARISONS["llama8b-vs-ablation2"] = ("llama8b-collabscirem", "llama8b-ablation2", "H2")
for k in ("ablation3", "ablation4", "sensitivity"):
    COMPARISONS[f"llama8b-vs-{k}"] = ("llama8b-collabscirem", f"llama8b-{k}", "exploratory")

# judge key -> OpenRouter model id
JUDGES = {
    "deepseek": "deepseek/deepseek-v4-pro-0813",
}

DIMENSIONS = ["methodological_soundness", "justification", "logical_consistency", "factual_correctness",
              "novelty", "responsiveness", "overall"]
RIGOR = DIMENSIONS[:4]


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def load_prompts(split):
    return [p for p in read_jsonl(PROMPTS) if split == "all" or p["split"] == split]


# Evaluation runs: run 1 is the original (seed 42); runs 2-3 reuse the same prompts
# and openings with another assistant seed (the simulator samples at temperature 0.7).
RUN_SEEDS = {1: 42, 2: 43, 3: 44}


def run_dir(split, run=1):
    return split if run == 1 else f"{split}_r{run}"


def conv_path(model_key, split, run=1):
    return os.path.join(CONVS, run_dir(split, run), f"{model_key}.jsonl")


def format_conversation(messages):
    role = {"user": "Researcher", "assistant": "Assistant"}
    return "\n\n".join(f"[{role[m['role']]}]\n{m['content']}" for m in messages if m["role"] in role)
