"""
MT-Bench STEM Subset Evaluation
================================
Inference-only pass: runs the model on MT-Bench STEM/math questions and saves
all dialogues (questions + model answers) to a JSON file.

Scoring is done separately offline via experiments/04_mt_bench_stem/score_mt_bench_with_judge.py,
which reads the dialogue file and calls a judge LLM (e.g. gpt-4o-mini) from
a machine that has internet access.

Offline workflow
----------------
1. With internet access (already done; the file ships in data/):
       python experiments/04_mt_bench_stem/download_mt_bench.py
          → writes data/mt_bench_stem.json

2. GPU (no internet needed):
       python experiments/04_mt_bench_stem/run_mt_bench_eval.py
          → reads  data/mt_bench_stem.json
          → writes outputs/mt_bench_stem/<model>_dialogues.json

3. With internet access (judge API; the paper uses DeepSeek-V4-Pro via OpenRouter):
       python experiments/04_mt_bench_stem/score_mt_bench_with_judge.py \\
           --dialogues outputs/mt_bench_stem/<model>_dialogues.json \\
           --judge openrouter/deepseek/deepseek-v4-pro-0813 --suffix _deepseek
          → writes outputs/mt_bench_stem/<model>_scores_deepseek.json
"""

import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple

# MT-Bench categories that constitute the "STEM subset"
_STEM_CATEGORIES = {"stem", "math"}

# ---------------------------------------------------------------------------
# Fallback: publicly known MT-Bench STEM questions (question IDs 141-150).
# Used only when all HuggingFace dataset loading attempts fail.
# ---------------------------------------------------------------------------
_FALLBACK_STEM_QUESTIONS: List[Dict] = [
    {
        "question_id": 141,
        "category": "stem",
        "turns": [
            "What are the main differences between a Gram-positive and Gram-negative bacterium?",
            "What are some effective antibiotics for treating infections caused by Gram-positive bacteria, and how do they work?"
        ]
    },
    {
        "question_id": 142,
        "category": "stem",
        "turns": [
            "How do vaccines work to protect against infectious diseases? Discuss the role of the immune system.",
            "Can you explain the difference between sterilizing immunity and protection against severe disease?"
        ]
    },
    {
        "question_id": 143,
        "category": "stem",
        "turns": [
            "What is the main difference between a nuclear fission and fusion reaction? And why is fusion power generation so difficult to achieve?",
            "Suppose you're asked to design a fusion reactor. What would be the key engineering challenges and how would you approach solving them?"
        ]
    },
    {
        "question_id": 144,
        "category": "stem",
        "turns": [
            "Explain the concept of natural selection and how it contributes to the evolution and adaptation of species.",
            "Discuss the role of genetic drift in the process of evolution, and how does it interact with natural selection?"
        ]
    },
    {
        "question_id": 145,
        "category": "stem",
        "turns": [
            "How can the application of CRISPR-Cas9 technology in gene editing transform the field of medicine, and what are its potential ethical implications?",
            "What are the potential risks of using CRISPR-Cas9 in gene therapy, and how can these risks be mitigated?"
        ]
    },
    {
        "question_id": 146,
        "category": "stem",
        "turns": [
            "Describe the process and challenges of drug development, from target identification to clinical trials.",
            "Could you elaborate on the ethical considerations involved in clinical trials for new drugs, including the potential risks for trial participants?"
        ]
    },
    {
        "question_id": 147,
        "category": "stem",
        "turns": [
            "How do mRNA vaccines like the ones used for COVID-19 work? Discuss the role of mRNA and the immune system.",
            "Can you explain more about how the immune system recognizes foreign pathogens and creates an immune response?"
        ]
    },
    {
        "question_id": 148,
        "category": "stem",
        "turns": [
            "What are the primary differences between plant and animal cells? Please provide a detailed comparison.",
            "Why do plant cells have cell walls and chloroplasts, but animal cells do not? Discuss the evolutionary reasons and functional implications of these differences."
        ]
    },
    {
        "question_id": 149,
        "category": "stem",
        "turns": [
            "What are the major branches of academic study of climate change? And what are the key findings and debates within those branches?",
            "Can you explain the role of carbon capture and storage in mitigating climate change, and what are its limitations?"
        ]
    },
    {
        "question_id": 150,
        "category": "stem",
        "turns": [
            "What are the main challenges and advancements in quantum computing, and how might this technology impact fields such as cryptography and drug discovery?",
            "Given the current state of quantum computing, what do you think are the most promising near-term applications, and why?"
        ]
    },
]


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------

def _load_from_philschmid(categories: set) -> List[Dict]:
    """Load MT-Bench questions from philschmid/mt-bench (recommended)."""
    ds = load_dataset("philschmid/mt-bench", split="train")
    questions = []
    for row in ds:
        if row.get("category", "").lower() in categories:
            questions.append({
                "question_id": row["question_id"],
                "category": row["category"],
                "turns": list(row["turns"]),
            })
    return questions


def _load_from_lmsys_judgments(categories: set) -> List[Dict]:
    """
    Fallback: extract unique questions from lmsys/mt_bench_human_judgments.
    The dataset contains one row per (question_id, model, turn); we reconstruct
    the 2-turn conversation by aggregating over turns.
    """
    ds = load_dataset("lmsys/mt_bench_human_judgments", split="human")
    by_qid: Dict[int, Dict] = {}
    for row in ds:
        cat = row.get("category", "").lower()
        if cat not in categories:
            continue
        qid = int(row["question_id"])
        turn = int(row.get("turn", 1))
        # Prefer 'question' column; fall back to 'user_prompt'
        question_text = row.get("question") or row.get("user_prompt") or ""
        if qid not in by_qid:
            by_qid[qid] = {"question_id": qid, "category": cat, "turns": ["", ""]}
        if question_text and turn in (1, 2):
            by_qid[qid]["turns"][turn - 1] = question_text

    return [q for q in by_qid.values() if q["turns"][0]]


def load_mt_bench_stem_questions(
    categories: Optional[set] = None,
    dataset_path: Optional[str] = None,
    dataset_id: Optional[str] = None,
) -> List[Dict]:
    """
    Load MT-Bench questions for the specified categories.

    Priority order:
    1. dataset_path — local JSON file (offline-safe, produced by download_mt_bench.py)
    2. dataset_id   — custom HuggingFace dataset ID (requires internet)
    3. philschmid/mt-bench (requires internet)
    4. lmsys/mt_bench_human_judgments (requires internet)
    5. Built-in fallback STEM questions

    Each question dict has: question_id, category, turns (list[str, str]).
    """
    if categories is None:
        categories = _STEM_CATEGORIES

    # --- 1. Local JSON file (preferred for offline/SLURM use) ---
    if dataset_path:
        path = Path(dataset_path)
        if path.exists():
            with open(path) as f:
                questions = json.load(f)
            questions = [q for q in questions if q.get("category", "").lower() in categories]
            print(f"[MT-Bench] Loaded {len(questions)} questions from local file '{dataset_path}'.")
            return questions
        else:
            print(f"[MT-Bench] WARNING: dataset_path '{dataset_path}' not found. Falling back to HuggingFace.")

    # --- 2-4. HuggingFace sources (require internet) ---
    sources = []
    if dataset_id:
        sources.append(("custom", dataset_id))
    sources += [("philschmid", None), ("lmsys", None)]

    for source, _ in sources:
        try:
            if source == "custom":
                ds = load_dataset(dataset_id, split="train")
                questions = [
                    {"question_id": r["question_id"], "category": r["category"], "turns": list(r["turns"])}
                    for r in ds if r.get("category", "").lower() in categories
                ]
            elif source == "philschmid":
                questions = _load_from_philschmid(categories)
            else:
                questions = _load_from_lmsys_judgments(categories)

            if questions:
                print(f"[MT-Bench] Loaded {len(questions)} STEM questions from '{source}'.")
                return questions
        except Exception as e:
            print(f"[MT-Bench] Could not load from '{source}': {e}")

    # --- 5. Built-in fallback ---
    print("[MT-Bench] All HuggingFace sources failed. Using built-in fallback questions.")
    return [q for q in _FALLBACK_STEM_QUESTIONS if q["category"].lower() in categories]


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def _generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict],
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
) -> Tuple[str, int]:
    """Run one forward pass and return (decoded_text, num_new_tokens)."""
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return decoded.strip(), len(new_tokens)


def _load_model_and_tokenizer(model_id: str):
    """
    Load a model + tokenizer, transparently handling three cases:

    1. A plain HuggingFace model id or local HF model directory.
    2. A local PEFT/LoRA adapter directory (contains ``adapter_config.json``).
       The base model named in the adapter config is loaded first, then the
       adapter is applied on top.

    Returns ``(model, tokenizer)`` with the model in eval mode.
    """
    adapter_config = Path(model_id) / "adapter_config.json"

    if adapter_config.exists():
        from peft import PeftModel

        with open(adapter_config) as f:
            base_model_id = json.load(f)["base_model_name_or_path"]
        print(f"[MT-Bench] Detected LoRA adapter. Base model: {base_model_id}")

        # Tokenizer usually ships with the adapter (it was saved during training);
        # fall back to the base model if it is missing.
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, model_id)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Main evaluation function  (inference-only, no judge)
# ---------------------------------------------------------------------------

def run_mt_bench_stem_evaluation(
    model_id: str,
    categories: Optional[set] = None,
    dataset_path: Optional[str] = None,
    dataset_id: Optional[str] = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    seed: int = 42,
    model=None,
    tokenizer=None,
    system_prompt: Optional[str] = None,
) -> Tuple[List[Dict], float]:
    """
    Run one MT-Bench STEM inference pass (a single seed) and return the
    dialogues so the caller can aggregate several seeds into one file for
    later scoring with experiments/04_mt_bench_stem/score_mt_bench_with_judge.py.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, local HF model directory, or a local PEFT/LoRA
        adapter directory (auto-detected via ``adapter_config.json``).
    categories : set, optional
        MT-Bench categories to include. Defaults to {"stem", "math"}.
    dataset_path : str, optional
        Path to a local JSON file produced by experiments/04_mt_bench_stem/download_mt_bench.py.
        Use this for offline / SLURM environments. When omitted the function
        tries to load from HuggingFace (requires internet).
    dataset_id : str, optional
        Override the HuggingFace dataset ID (ignored when dataset_path is set).
    max_new_tokens : int
        Maximum tokens to generate per turn. Default 1024.
    temperature : float
        Sampling temperature. Default 0.7.
    seed : int
        Random seed. Default 42.
    model, tokenizer : optional
        Pre-loaded model/tokenizer to reuse across seeds (avoids reloading the
        weights once per seed). When omitted they are loaded from ``model_id``.

    Returns
    -------
    (dialogues, avg_token_count)
        ``dialogues`` is a list of per-question entries (question + answers +
        per-turn token counts). Scoring is done offline by the judge script.
    """
    set_seed(seed)

    if categories is None:
        categories = _STEM_CATEGORIES

    # ---- Load questions ----
    questions = load_mt_bench_stem_questions(
        categories=categories,
        dataset_path=dataset_path,
        dataset_id=dataset_id,
    )
    if not questions:
        print("[MT-Bench] No questions loaded. Returning empty dialogues.")
        return [], 0.0

    # ---- Load model and tokenizer (unless supplied by the caller) ----
    if model is None or tokenizer is None:
        try:
            model, tokenizer = _load_model_and_tokenizer(model_id)
        except Exception as e:
            print(f"[MT-Bench] Error loading model '{model_id}': {e}")
            return [], 0.0

    print(
        f"\n🚀 Starting MT-Bench STEM inference for {model_id} "
        f"on {len(questions)} questions (seed={seed}, temperature={temperature}) ..."
    )

    dialogues: List[Dict] = []
    all_token_counts: List[int] = []

    for q in tqdm(questions, desc=f"MT-Bench STEM (seed={seed})"):
        turns = q["turns"]
        if not turns:
            continue

        system_msg = {
            "role": "system",
            "content": system_prompt or (
                "You are a knowledgeable scientist and engineer. "
                "Provide clear, accurate, and detailed answers."
            ),
        }

        # ---- Turn 1 ----
        messages_t1 = [system_msg, {"role": "user", "content": turns[0]}]
        answer1, tokens1 = _generate_response(
            model, tokenizer, messages_t1, max_new_tokens, temperature
        )
        all_token_counts.append(tokens1)

        entry: Dict = {
            "model_id": model_id,
            "question_id": q["question_id"],
            "category": q["category"],
            "seed": seed,
            "turn1": {"question": turns[0], "answer": answer1, "tokens": tokens1},
        }

        # ---- Turn 2 (if available) ----
        if len(turns) >= 2 and turns[1]:
            messages_t2 = messages_t1 + [
                {"role": "assistant", "content": answer1},
                {"role": "user", "content": turns[1]},
            ]
            answer2, tokens2 = _generate_response(
                model, tokenizer, messages_t2, max_new_tokens, temperature
            )
            all_token_counts.append(tokens2)
            entry["turn2"] = {"question": turns[1], "answer": answer2, "tokens": tokens2}

        dialogues.append(entry)

    avg_token_count = sum(all_token_counts) / max(len(all_token_counts), 1)

    print(
        f"\n📊 MT-Bench STEM Inference Complete for {model_id} (seed={seed})\n"
        f"   Questions  : {len(dialogues)}\n"
        f"   Avg tokens : {avg_token_count:.1f}"
    )

    return dialogues, avg_token_count
