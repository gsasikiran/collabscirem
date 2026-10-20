"""
Generate research conversations for SciQuality (paper Section 4).

A gpt-4o-mini researcher (a persona written for this study, not the training-time
user prompt) plans a study on a held-out research question and talks to the
model under test, served by vLLM, for exactly 4 user turns. The researcher's
opening message is generated once per prompt and reused for every model.
Needs OPENAI_API_KEY (simulator) and a vLLM endpoint written by experiments/serve_vllm.sh;
resumes by prompt id.

    python experiments/03_sciquality/generate.py --model llama8b-collabscirem --split eval
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODELS, OPENINGS, RUN_SEEDS, conv_path, load_prompts, read_jsonl

LOGGER = logging.getLogger("sciq.generate")
N_USER_TURNS = 4

RESEARCHER_PROMPT = """You are role-playing a researcher who is talking to an AI research assistant. \
You are planning a study and want help designing its methodology.

Your research topic (private notes, written by you):
<<<
{question}
>>>

How to write your messages:
- Write only your next message to the assistant, in the first person, as the researcher. 2-5 sentences.
- Never write the methodology yourself and never answer as the assistant.
- {turn_instruction}
- Be a demanding but realistic collaborator: ask for specifics (design, controls, sample, measures, \
analysis, feasibility), question choices that seem weak, unclear or unjustified, or ask the assistant \
to develop one part further. Do not simply praise or thank.
- Output the message text only, without quotes or a speaker label."""

FIRST_TURN = ("This is your first message: describe in your own words what you want to study and ask "
              "for help designing the study. Do not paste your notes verbatim.")
LATER_TURN = ("Respond to the assistant's latest reply in the conversation so far, shown below, and "
              "move the planning forward.")


def call(client, retries=6, **kw):
    for attempt in range(retries):
        try:
            return client.chat.completions.create(**kw)
        except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError,
                openai.RateLimitError) as e:
            LOGGER.warning("call failed (%s), retry %d", type(e).__name__, attempt + 1)
            time.sleep(10 * (attempt + 1))
    return client.chat.completions.create(**kw)


def researcher_message(sim, prompt, messages):
    if not messages:
        content = RESEARCHER_PROMPT.format(question=prompt["research_question"], turn_instruction=FIRST_TURN)
    else:
        history = "\n\n".join(f"{'You' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in messages)
        content = (RESEARCHER_PROMPT.format(question=prompt["research_question"], turn_instruction=LATER_TURN)
                   + f"\n\nConversation so far:\n<<<\n{history}\n>>>")
    out = call(sim, model="gpt-4o-mini", temperature=0.7, max_tokens=400,
               messages=[{"role": "user", "content": content}])
    return out.choices[0].message.content.strip()


def ensure_openings(sim, prompts, workers):
    have = {r["id"]: r["opening"] for r in read_jsonl(OPENINGS)}
    todo = [p for p in prompts if p["id"] not in have]
    lock = threading.Lock()

    def work(p):
        msg = researcher_message(sim, p, [])
        with lock, open(OPENINGS, "a") as f:
            f.write(json.dumps({"id": p["id"], "opening": msg}) + "\n")
        return p["id"], msg

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for pid, msg in pool.map(work, todo):
            have[pid] = msg
    return have


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--split", required=True, choices=["pilot", "eval"])
    ap.add_argument("--run", type=int, default=1, choices=sorted(RUN_SEEDS))
    ap.add_argument("--openings-only", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sim = openai.OpenAI(timeout=120)
    prompts = load_prompts(args.split)
    os.makedirs(os.path.dirname(OPENINGS), exist_ok=True)
    openings = ensure_openings(sim, prompts, args.workers)
    if args.openings_only:
        return

    name = f"sciq-{args.model}"
    ep = os.path.join(os.path.dirname(OPENINGS), "..", "..", "outputs", "vllm_endpoints")
    base_url = open(os.path.join(ep, f"{name}.url")).read().strip()
    key = open(os.path.join(ep, f"{name}.key")).read().strip()
    model_client = openai.OpenAI(base_url=base_url, api_key=key, timeout=900)

    out = conv_path(args.model, args.split, args.run)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {r["id"] for r in read_jsonl(out)}
    todo = [p for p in prompts if p["id"] not in done]
    LOGGER.info("%s %s: %d done, %d to run", args.model, args.split, len(done), len(todo))
    lock = threading.Lock()

    def work(p):
        messages, tokens, truncated = [{"role": "user", "content": openings[p["id"]]}], [], 0
        for turn in range(N_USER_TURNS):
            if turn:
                messages.append({"role": "user", "content": researcher_message(sim, p, messages)})
            r = call(model_client, model=name, messages=messages, temperature=0.6, top_p=0.9,
                     max_tokens=2048, seed=RUN_SEEDS[args.run])
            choice = r.choices[0]
            messages.append({"role": "assistant", "content": choice.message.content or ""})
            tokens.append(r.usage.completion_tokens if r.usage else None)
            truncated += choice.finish_reason == "length"
        rec = {"id": p["id"], "model": args.model, "messages": messages,
               "assistant_tokens": tokens, "truncated_turns": truncated}
        with lock, open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return p["id"]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(work, p): p["id"] for p in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                fut.result()
                if i % 10 == 0 or i == len(futs):
                    LOGGER.info("%s: %d/%d", args.model, i, len(futs))
            except Exception:
                LOGGER.exception("prompt %s failed (left for a re-run)", futs[fut])


if __name__ == "__main__":
    main()
