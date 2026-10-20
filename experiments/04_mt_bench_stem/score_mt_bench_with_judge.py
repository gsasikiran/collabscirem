"""
Score MT-Bench-STEM dialogues with a judge LLM (litellm model string).

The paper uses DeepSeek-V4-Pro through OpenRouter (OPENROUTER_API_KEY):

    python experiments/04_mt_bench_stem/score_mt_bench_with_judge.py \
        --dialogues outputs/mt_bench_stem/*_dialogues.json \
        --judge openrouter/deepseek/deepseek-v4-pro-0813 --workers 20 --suffix _deepseek

Writes <model>_scores_deepseek.json next to each dialogue file (per-seed scores and the
across-seed mean/std of every dimension and of the token count).
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import litellm


# ---------------------------------------------------------------------------
# Judge prompt (must stay in sync with collabscirem/benchmarks/prompts/mt_bench_stem.txt)
# ---------------------------------------------------------------------------
_PROMPT_PATH = Path(__file__).resolve().parents[2] / "collabscirem" / "benchmarks" / "prompts" / "mt_bench_stem.txt"
with open(_PROMPT_PATH) as _f:
    _JUDGE_PROMPT = _f.read().strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_turn1_messages(question: str, answer: str) -> List[Dict]:
    return [
        {"role": "system", "content": _JUDGE_PROMPT},
        {
            "role": "user",
            "content": (
                f"[Question]\n{question}\n\n"
                f"[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]"
            ),
        },
    ]


def _build_turn2_messages(
    question1: str, answer1: str, question2: str, answer2: str
) -> List[Dict]:
    return [
        {"role": "system", "content": _JUDGE_PROMPT},
        {
            "role": "user",
            "content": (
                "<|The Start of Assistant's Conversation with User|>\n"
                f"### User:\n{question1}\n\n"
                f"### Assistant:\n{answer1}\n\n"
                f"### User:\n{question2}\n\n"
                f"### Assistant:\n{answer2}\n"
                "<|The End of Assistant's Conversation with User|>"
            ),
        },
    ]


# Dimensions the judge rates, in report order.
_DIMENSIONS = ("accuracy", "format", "usefulness", "relevance", "overall")


def _extract_scores(judgment: str) -> Dict[str, Optional[float]]:
    """
    Parse the judge output into per-dimension scores.

    Expects lines of the form ``Accuracy: [[7]]`` / ``Format: [[8]]`` /
    ``Overall: [[7]]`` (case-insensitive). Falls back to a bare ``[[N]]`` or
    ``Rating: N`` for the overall score to stay compatible with older single
    score judgments.
    """
    scores: Dict[str, Optional[float]] = {d: None for d in _DIMENSIONS}

    for dim in _DIMENSIONS:
        m = re.search(rf"{dim}\s*:?\s*\[\[(\d+(?:\.\d+)?)\]\]", judgment, re.IGNORECASE)
        if m:
            scores[dim] = float(m.group(1))

    # Backward-compatible fallback for a single overall rating.
    if scores["overall"] is None:
        m = re.search(r"\[\[(\d+(?:\.\d+)?)\]\]", judgment)
        if m is None:
            m = re.search(r"[Rr]ating[:\s]+(\d+(?:\.\d+)?)", judgment)
        if m:
            scores["overall"] = float(m.group(1))

    return scores


_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _is_reasoning_model(judge_model: str) -> bool:
    name = judge_model.rsplit("/", 1)[-1]
    return name.startswith(_REASONING_MODEL_PREFIXES)


def _call_judge(
    judge_model: str, messages: List[Dict], max_retries: int = 3
) -> Dict[str, Optional[float]]:
    """Call the judge and return per-dimension scores (accuracy/format/overall)."""
    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=judge_model,
                messages=messages,
                temperature=1.0,
            )
            if judge_model.startswith("openrouter/"):
                # OpenRouter judges (e.g. DeepSeek, as in the scientific-quality
                # evaluation) reason before answering; leave the output uncapped.
                kwargs["temperature"] = 0.0
            elif _is_reasoning_model(judge_model):
                # Reasoning models spend part of max_tokens on hidden
                # reasoning_tokens before emitting visible text, so a small
                # budget (e.g. 512) can be exhausted by reasoning alone,
                # leaving an empty response. Give it more headroom and keep
                # reasoning effort low since this is a simple 1-10 grading task.
                kwargs["max_tokens"] = 2048
                kwargs["reasoning_effort"] = "minimal"
            else:
                kwargs["max_tokens"] = 512

            response = litellm.completion(**kwargs)
            judgment = response.choices[0].message.content or ""
            scores = _extract_scores(judgment)
            # Accept the judgment once at least the overall score parsed.
            if scores["overall"] is not None:
                return scores
            print(f"  [Judge] Could not parse score on attempt {attempt + 1}: {judgment[:100]!r}")
        except Exception as e:
            print(f"  [Judge] Attempt {attempt + 1} failed: {e}")
    return {d: None for d in _DIMENSIONS}


# ---------------------------------------------------------------------------
# Core scoring logic
# ---------------------------------------------------------------------------

def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _std(values: List[float]) -> float:
    """Population standard deviation (matches numpy default ddof=0)."""
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5


def _normalize_runs(data) -> tuple:
    """
    Accept either the current multi-seed format::

        {"model_id", "temperature", "seeds", "runs": [{"seed", "avg_token_count", "dialogues"}]}

    or the legacy flat list of dialogue entries, and return
    ``(model_id, temperature, runs)`` where each run is a dict with keys
    ``seed``, ``avg_token_count`` and ``dialogues``.
    """
    if isinstance(data, dict) and "runs" in data:
        return data.get("model_id", "unknown"), data.get("temperature"), data["runs"]

    # Legacy format: a flat list of dialogue entries = a single (unknown-seed) run.
    if isinstance(data, list):
        model_id = data[0].get("model_id", "unknown") if data else "unknown"
        run = {"seed": None, "avg_token_count": None, "dialogues": data}
        return model_id, None, [run]

    raise ValueError("Unrecognized dialogue file format.")


def _avg_dims(dicts: List[Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    """Average a list of per-dimension score dicts, dimension by dimension."""
    out: Dict[str, Optional[float]] = {}
    for dim in _DIMENSIONS:
        vals = [d[dim] for d in dicts if d.get(dim) is not None]
        out[dim] = _mean(vals)
    return out


def _fmt_dims(dims: Dict[str, Optional[float]]) -> str:
    parts = []
    for d in _DIMENSIONS:
        v = dims.get(d)
        parts.append(f"{d}={v:.1f}" if v is not None else f"{d}=N/A")
    return ", ".join(parts)


def _score_run(run: Dict, judge_model: str, workers: int = 1) -> Dict:
    """Judge every dialogue in a single seed run; return scored entries + per-dimension means."""
    seed = run.get("seed")
    dialogues = run.get("dialogues", [])
    print(f"\n--- Scoring seed {seed} ({len(dialogues)} dialogues) ---")

    # All judge calls of the run are independent, so they can run concurrently.
    calls = {}
    for k, entry in enumerate(dialogues):
        t1 = entry["turn1"]
        calls[(k, 1)] = _build_turn1_messages(t1["question"], t1["answer"])
        if "turn2" in entry:
            t2 = entry["turn2"]
            calls[(k, 2)] = _build_turn2_messages(t1["question"], t1["answer"], t2["question"], t2["answer"])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = dict(zip(calls, pool.map(lambda m: _call_judge(judge_model, m), calls.values())))

    scored_entries = []
    per_question_dims: List[Dict[str, Optional[float]]] = []

    for k, entry in enumerate(dialogues):
        qid = entry["question_id"]
        category = entry["category"]
        score_t1 = results[(k, 1)]
        print(f"  [seed {seed}] Q{qid} [{category}] Turn 1: {_fmt_dims(score_t1)}")
        score_t2 = results.get((k, 2))
        if score_t2 is not None:
            print(f"  [seed {seed}] Q{qid} [{category}] Turn 2: {_fmt_dims(score_t2)}")

        # Per-question, per-dimension average across the (up to two) turns.
        turn_dims = [s for s in [score_t1, score_t2] if s is not None]
        avg_q = _avg_dims(turn_dims) if turn_dims else {d: None for d in _DIMENSIONS}
        per_question_dims.append(avg_q)

        scored_entries.append({
            **entry,
            "score_turn1": score_t1,
            "score_turn2": score_t2,
            "score_avg": avg_q,
        })

    mean_scores = _avg_dims(per_question_dims)
    return {
        "seed": seed,
        "avg_token_count": run.get("avg_token_count"),
        "mean_scores": mean_scores,
        "num_questions": len(dialogues),
        "per_question": scored_entries,
    }


def score_dialogues(
    dialogues_path: str,
    judge_model: str = "gpt-4o-mini",
    output_path: Optional[str] = None,
    workers: int = 1,
    suffix: str = "",
) -> Dict:
    """
    Score every seed run in the dialogue file and save results.

    Returns a summary dict with per-seed scores and across-seed mean/std of
    both the judge score and the average token count.
    """
    path = Path(dialogues_path)
    if not path.exists():
        print(f"ERROR: file not found: {dialogues_path}")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    model_id, temperature, runs = _normalize_runs(data)
    if not runs:
        print(f"WARNING: {dialogues_path} has no runs.")
        return {}

    print(
        f"\n=== Scoring {len(runs)} seed run(s) for {model_id} "
        f"using {judge_model} ==="
    )

    scored_runs = [_score_run(run, judge_model, workers) for run in runs]

    # ---- Across-seed aggregation (mean/std per dimension) ----
    scores_stats: Dict[str, Dict[str, Optional[float]]] = {}
    for dim in _DIMENSIONS:
        seed_vals = [r["mean_scores"].get(dim) for r in scored_runs]
        seed_vals = [v for v in seed_vals if v is not None]
        scores_stats[dim] = {"mean": _mean(seed_vals), "std": _std(seed_vals)}

    seed_tokens = [r["avg_token_count"] for r in scored_runs if r["avg_token_count"] is not None]
    mean_token_count = _mean(seed_tokens)
    std_token_count = _std(seed_tokens)

    tok_str = f"{mean_token_count:.1f} ± {std_token_count:.1f}" if mean_token_count is not None else "N/A"
    dim_lines = []
    for dim in _DIMENSIONS:
        m, s = scores_stats[dim]["mean"], scores_stats[dim]["std"]
        val = f"{m:.2f} ± {s:.2f} /10" if m is not None else "N/A"
        dim_lines.append(f"   {dim.capitalize():<14}: {val}")
    print(
        f"\n📊 Final MT-Bench STEM Results for {model_id}\n"
        f"   Judge model   : {judge_model}\n"
        f"   Seeds         : {[r['seed'] for r in scored_runs]}\n"
        + "\n".join(dim_lines) + "\n"
        f"   Avg tokens/turn: {tok_str}"
    )

    summary = {
        "model_id": model_id,
        "judge_model": judge_model,
        "temperature": temperature,
        "seeds": [r["seed"] for r in scored_runs],
        # Per-dimension mean/std across seeds:
        # accuracy, format, usefulness, relevance, overall.
        **{dim: scores_stats[dim] for dim in _DIMENSIONS},
        "mean_token_count": mean_token_count,
        "std_token_count": std_token_count,
        "per_seed_scores": [r["mean_scores"] for r in scored_runs],
        "per_seed_token_count": [r["avg_token_count"] for r in scored_runs],
        "runs": scored_runs,
    }

    # ---- Save results ----
    if output_path is None:
        output_path = str(path).replace("_dialogues.json", f"_scores{suffix}.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   Scores saved → {output_path}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score MT-Bench STEM dialogue files with an LLM judge."
    )
    parser.add_argument(
        "--dialogues",
        nargs="+",
        required=True,
        help="One or more dialogue JSON files produced by run_benchmark_eval.py.",
    )
    parser.add_argument(
        "--judge",
        default="gpt-4o-mini",
        help="litellm model string for the judge (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for scores JSON. Auto-derived from input name when omitted.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Concurrent judge calls")
    parser.add_argument("--suffix", default="", help="Output file suffix, e.g. '_deepseek' -> <model>_scores_deepseek.json")
    args = parser.parse_args()

    for dialogue_file in args.dialogues:
        out = args.output
        if out is None and len(args.dialogues) > 1:
            # When processing multiple files, always auto-derive names
            out = None
        score_dialogues(dialogue_file, judge_model=args.judge, output_path=out,
                        workers=args.workers, suffix=args.suffix)


if __name__ == "__main__":
    main()
