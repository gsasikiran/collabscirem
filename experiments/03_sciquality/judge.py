"""
Pairwise, blind judging for SciQuality (paper Appendix E). Every comparison is judged in
both orders (X as A, then X as B) by an OpenRouter model. Resumes by
(prompt, order); a failed or unparseable call is recorded and retried on re-run.

    python experiments/03_sciquality/judge.py --judge deepseek --split eval --comparisons qwen7b-vs-collabllm
"""

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (COMPARISONS, DIMENSIONS, JUDGES, JUDGMENTS, RUN_SEEDS, conv_path, format_conversation, run_dir,
                    load_prompts, read_jsonl)

LOGGER = logging.getLogger("sciq.judge")

JUDGE_PROMPT = """You are an expert scientific reviewer. Two AI research assistants (A and B) each held a \
four-turn conversation with a researcher who is planning a study. The researcher's later messages differ \
slightly because they react to each assistant; judge only the assistants' contributions.

Researcher's research question:
<<<
{question}
>>>

Reference methodology used by the original study (one valid approach, not the only acceptable one: do not \
penalize sound alternatives, but use it to check domain correctness and feasibility):
<<<
{reference}
>>>

Conversation with assistant A:
<<<
{conv_a}
>>>

Conversation with assistant B:
<<<
{conv_b}
>>>

Compare A and B on each dimension and pick "A", "B" or "tie" (tie only if they are genuinely equivalent). \
Judge substance, not length or formatting: more text is better only if it adds valid scientific content.

1. methodological_soundness: validity of the proposed study design - controls, confounds, measurement, \
analysis, feasibility.
2. justification: specific, checkable reasons for each design choice. A citation earns credit only if it \
identifies a real, relevant work you are confident exists; invented, vague or unverifiable references count \
against the response.
3. logical_consistency: no contradictions within or across turns; later turns build coherently on earlier ones.
4. factual_correctness: technical and factual accuracy; list concrete errors for each side.
5. novelty: non-obvious, useful ideas beyond a textbook answer.
6. responsiveness: addresses the researcher's evolving requests and questions.
7. overall: which assistant a domain expert would rather collaborate with.

Respond with a single JSON object and nothing else, in this form:
{{"methodological_soundness": {{"winner": "A", "reason": "one sentence"}},
  "justification": {{"winner": "B", "reason": "one sentence"}},
  "logical_consistency": {{"winner": "tie", "reason": "one sentence"}},
  "factual_correctness": {{"winner": "A", "reason": "one sentence"}},
  "novelty": {{"winner": "A", "reason": "one sentence"}},
  "responsiveness": {{"winner": "B", "reason": "one sentence"}},
  "overall": {{"winner": "A", "reason": "one sentence"}},
  "factual_errors": {{"A": ["..."], "B": ["..."]}}}}"""


def parse(text):
    """Verdict dict or None. Accepts a JSON object anywhere in the text (e.g. inside a code fence)."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    verdict = {}
    for d in DIMENSIONS:
        w = str((obj.get(d) or {}).get("winner", "")).strip().strip('"').upper()
        if w not in ("A", "B", "TIE"):
            return None
        verdict[d] = w.lower() if w == "TIE" else w
    errs = obj.get("factual_errors") or {}
    verdict["factual_errors"] = {"A": list(errs.get("A") or []), "B": list(errs.get("B") or [])}
    verdict["reasons"] = {d: (obj.get(d) or {}).get("reason", "") for d in DIMENSIONS}
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, choices=sorted(JUDGES))
    ap.add_argument("--split", required=True, choices=["pilot", "eval"])
    ap.add_argument("--run", type=int, default=1, choices=sorted(RUN_SEEDS))
    ap.add_argument("--comparisons", nargs="+", default=sorted(COMPARISONS))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--model-id", default=None, help="Override the OpenRouter model id")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model_id = args.model_id or JUDGES[args.judge]
    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],
                           timeout=900)
    prompts = {p["id"]: p for p in load_prompts(args.split)}

    jobs = []
    for comp in args.comparisons:
        x, y, _ = COMPARISONS[comp]
        cx = {r["id"]: r for r in read_jsonl(conv_path(x, args.split, args.run))}
        cy = {r["id"]: r for r in read_jsonl(conv_path(y, args.split, args.run))}
        out = os.path.join(JUDGMENTS, run_dir(args.split, args.run), args.judge, f"{comp}.jsonl")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        done = {(r["id"], r["order"]) for r in read_jsonl(out) if r.get("verdict")}
        missing = [pid for pid in prompts if pid not in cx or pid not in cy]
        if missing:
            LOGGER.warning("%s: %d prompts lack a conversation for one side, skipped", comp, len(missing))
        for pid in prompts:
            if pid in missing:
                continue
            for order in ("XY", "YX"):
                if (pid, order) not in done:
                    jobs.append((comp, pid, order, cx[pid], cy[pid], out))
    LOGGER.info("%s (%s): %d calls to make", args.judge, model_id, len(jobs))

    lock = threading.Lock()

    def work(job):
        comp, pid, order, rx, ry, out = job
        a, b = (rx, ry) if order == "XY" else (ry, rx)
        p = prompts[pid]
        content = JUDGE_PROMPT.format(question=p["research_question"], reference=p["reference_methodology"],
                                      conv_a=format_conversation(a["messages"]),
                                      conv_b=format_conversation(b["messages"]))
        rec = {"id": pid, "comparison": comp, "order": order, "A": a["model"], "B": b["model"],
               "judge": args.judge, "judge_model": model_id}
        for attempt in range(4):
            try:
                r = client.chat.completions.create(model=model_id, temperature=0,
                                                   messages=[{"role": "user", "content": content}],
                                                   extra_body={"usage": {"include": True}})
                text = r.choices[0].message.content or ""
                u = r.usage.model_dump() if r.usage else {}
                rec |= {"raw": text, "usage": u, "cost": u.get("cost"), "verdict": parse(text)}
                if rec["verdict"]:
                    break
                LOGGER.warning("%s %s %s: unparseable output, retry %d", comp, pid, order, attempt + 1)
            except Exception as e:
                rec |= {"error": f"{type(e).__name__}: {e}", "verdict": None}
                LOGGER.warning("%s %s %s: %s, retry %d", comp, pid, order, type(e).__name__, attempt + 1)
                time.sleep(15 * (attempt + 1))
        with lock, open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    cost = 0.0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(work, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            cost += rec.get("cost") or 0.0
            if i % 20 == 0 or i == len(futs):
                LOGGER.info("%s: %d/%d calls, cost so far $%.2f", args.judge, i, len(futs), cost)


if __name__ == "__main__":
    main()
