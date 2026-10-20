"""
MINT (Wang et al., ICLR 2024) on the scientific reasoning subsets.

The model under test is served by vLLM (experiments/serve_vllm.sh); this script runs
MINT's interactive loop against it with gpt-4o-mini as the feedback provider.

Setting (MINT default): k = 5 interaction turns, at most 2 proposed solutions,
tools enabled, textual feedback without ground truth ("no_GT"). Every model
uses add_system_message=True so the proactive-prompting baseline differs only
by the extra system prompt.

Usage (OPENAI_API_KEY set). The paper uses --max-tokens 4096 and seeds 1-3:
    python experiments/05_mint/run_mint_eval.py --served-name qwen-2.5-7B-collabscirem --max-tokens 4096 --seed 1

No-tool ablation (--no-tools): same loop, budget and feedback, but code is never
executed; a turn without <solution> is a text reasoning turn. Output goes to
<served-name>-notool[-seed<N>], so it never mixes with the tool runs.

--max-tokens N (default 512, MINT's value) caps each
model turn; any other value adds "mt<N>" to the run tag.

Results: third_party/mint-bench/data/outputs/<served-name>[-<run-tag>]/.../results.jsonl
"""

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MINT_ROOT = os.path.join(ROOT, "third_party", "mint-bench")
MINT_PYTHON = os.path.join(MINT_ROOT, ".venv-mint", "bin", "python")  # openai==0.28 environment
TASKS = ["theoremqa", "math", "mmlu"]

ENV_CONFIG = {"max_steps": 5, "use_tools": True, "max_propose_solution": 2, "count_down": True}
FEEDBACK_TYPE = {"pseudo_human_feedback": "no_GT", "feedback_form": "textual"}


def build_config(task, served_name, run_tag, api_base, feedback_model, extra_system_prompt,
                 temperature, top_p, use_tools=True, max_tokens=512):
    sys.path.insert(0, MINT_ROOT)
    from mint.configs.generate_config import generate_config_json  # noqa: E402

    agent_config = {
        "model_name": served_name,
        "chat_mode": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "openai.api_base": api_base,
        "add_system_message": True,
    }
    if extra_system_prompt:
        agent_config["extra_system_prompt"] = extra_system_prompt
    agent_info = {"agent_class": "VLLMAgent", "config": agent_config}
    feedback_info = {"agent_class": "OpenAIFeedbackAgent", "model_name": feedback_model}

    cwd = os.getcwd()
    os.chdir(MINT_ROOT)  # MINT writes output dirs relative to its root
    try:
        config = generate_config_json(
            task_name=task,
            agent_model_info=agent_info,
            feedback_type=dict(FEEDBACK_TYPE),
            feedback_model_info=feedback_info,
            env_config=dict(ENV_CONFIG, use_tools=use_tools),
        )
    finally:
        os.chdir(cwd)
    if run_tag:
        # keep runs with different prompts apart; model_name must stay the served name
        config["output_dir"] = config["output_dir"].replace(served_name, f"{served_name}-{run_tag}", 1)
    # the served model name is used by vLLM; keep the key out of the config file
    return config


def main():
    parser = argparse.ArgumentParser(description="Run MINT reasoning subsets against a vLLM-served model")
    parser.add_argument("--served-name", required=True, help="Name used with experiments/serve_vllm.sh")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--feedback-model", default="gpt-4o-mini")
    parser.add_argument("--system-prompt-file", default=None)
    parser.add_argument("--run-tag", default="")
    # Sampled decoding with a per-run seed, so the three seeds give a meaningful mean and std.
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=None, help="Per-run seed; output goes to <name>__seed<N>")
    parser.add_argument("--no-tools", action="store_true", help="Ablation: disable code execution")
    parser.add_argument("--max-tokens", type=int, default=512, help="Per-turn output cap for the model under test")
    parser.add_argument("--debug", action="store_true", help="MINT debug mode (3 tasks per subset)")
    args = parser.parse_args()
    if args.max_tokens != 512:
        args.run_tag = f"mt{args.max_tokens}-{args.run_tag}" if args.run_tag else f"mt{args.max_tokens}"
    if args.no_tools:
        args.run_tag = f"notool-{args.run_tag}" if args.run_tag else "notool"
    if args.seed is not None:
        # Output dir becomes <served-name>-seed<N> (or <served-name>-<tag>-seed<N>).
        args.run_tag = f"{args.run_tag}-seed{args.seed}" if args.run_tag else f"seed{args.seed}"

    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Set OPENAI_API_KEY (the feedback provider is an OpenAI model).")
    endpoints = os.path.join(ROOT, "outputs", "vllm_endpoints")
    with open(os.path.join(endpoints, f"{args.served_name}.url")) as f:
        api_base = f.read().strip()
    with open(os.path.join(endpoints, f"{args.served_name}.key")) as f:
        vllm_key = f.read().strip()

    extra_system_prompt = None
    if args.system_prompt_file:
        with open(args.system_prompt_file) as f:
            extra_system_prompt = f.read().strip()

    config_dir = os.path.join(ROOT, "outputs", "mint_configs")
    os.makedirs(config_dir, exist_ok=True)
    env = dict(os.environ, VLLM_API_KEY=vllm_key, PYTHONPATH=MINT_ROOT)
    if args.seed is not None:
        env["MINT_SEED"] = str(args.seed)

    for task in args.tasks:
        config = build_config(task, args.served_name, args.run_tag, api_base, args.feedback_model,
                              extra_system_prompt, args.temperature, args.top_p,
                              use_tools=not args.no_tools, max_tokens=args.max_tokens)
        tag = args.served_name + (f"-{args.run_tag}" if args.run_tag else "")
        config_path = os.path.join(config_dir, f"{tag}_{task}.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"=== {task}: {config_path} -> {config['output_dir']}")
        cmd = [MINT_PYTHON, "-m", "mint.main", "--exp_config", config_path] + (["--debug"] if args.debug else [])
        subprocess.run(cmd, cwd=MINT_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
