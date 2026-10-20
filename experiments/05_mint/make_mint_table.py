"""
MINT table (Tables 1, 2 and 6). For each model and seed, pooled over TheoremQA / MATH /
MMLU (each problem weighted equally):

    SR@k    share of problems solved within the first k steps of the k=5 run
    ΔSR     SR@5 - SR@1  (how much the model gains from interaction)
    Tok     agent completion tokens per problem

Values are mean ± sample std over seeds 1-3. Against the family base model, an exact
McNemar test on SR@5 uses seed-matched pairs pooled over seeds.

The paper's runs use a 4096-token cap per turn (run directories <name>-mt4096-seed<N>):

    python experiments/05_mint/make_mint_table.py --variant mt4096
    python experiments/05_mint/make_mint_table.py --variant mt4096 --ablation
"""

import argparse
import glob
import json
import math
import os
from math import comb

# (served name, label, family, is_base); served names as in experiments/serve_vllm.sh
ROWS = [
    ("Llama-3.2-3B-Instruct", "Llama-3.2-3B-Instruct", "Llama-3.2-3B", True),
    ("llama-3.2-3B-collabllmsci", "\\quad + CollabLLMSci", "Llama-3.2-3B", False),
    ("llama-3.2-3B-collabscirem", "\\quad + CollabSciReM", "Llama-3.2-3B", False),
    ("Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct", "Llama-3.1-8B", True),
    ("llama-3.1-8B-collabllmsci", "\\quad + CollabLLMSci", "Llama-3.1-8B", False),
    ("llama-3.1-8B-collabscirem", "\\quad + CollabSciReM", "Llama-3.1-8B", False),
    ("Qwen2.5-3B-Instruct", "Qwen-2.5-3B-Instruct", "Qwen-2.5-3B", True),
    ("qwen-2.5-3B-collabllmsci", "\\quad + CollabLLMSci", "Qwen-2.5-3B", False),
    ("qwen-2.5-3B-collabscirem", "\\quad + CollabSciReM", "Qwen-2.5-3B", False),
    ("Qwen2.5-7B-Instruct", "Qwen-2.5-7B-Instruct", "Qwen-2.5-7B", True),
    ("qwen-2.5-7B-collabllmsci", "\\quad + CollabLLMSci", "Qwen-2.5-7B", False),
    ("qwen-2.5-7B-collabscirem", "\\quad + CollabSciReM", "Qwen-2.5-7B", False),
]
# Llama-3.1-8B ablation / sensitivity rows; --ablation
ABLATION_ROWS = [
    ("Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct", "Llama-3.1-8B", True),
    ("llama-3.1-8B-collabscirem", "\\quad + CollabSciReM", "Llama-3.1-8B", False),
    ("llama-3.1-8B-ablation-gen_struct", "\\quad + $R_{gen}+R_{struct}$", "Llama-3.1-8B", False),
    ("llama-3.1-8B-ablation-gen_sci", "\\quad + $R_{gen}+R_{sci}$", "Llama-3.1-8B", False),
    ("llama-3.1-8B-ablation-struct_sci", "\\quad + $R_{struct}+R_{sci}$", "Llama-3.1-8B", False),
    ("llama-3.1-8B-collabllm-sciw", "\\quad + CollabLLMSci $\\mid \\gamma_{CSciReM}$", "Llama-3.1-8B", False),
    ("llama-3.1-8B-sensitivity", "\\quad + sensitivity", "Llama-3.1-8B", False),
]
SEEDS = (1, 2, 3)
TASKS = {"theoremqa": 49, "math": 100, "mmlu": 76}
MAX_K = 5


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def mean_std(vals):
    m = sum(vals) / len(vals)
    s = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
    return m, s


def load_seed(root, run, seed, suffix=""):
    """Per-problem outcomes for one model and seed, or None if any task is incomplete."""
    problems = {}
    for task, n_expected in TASKS.items():
        paths = glob.glob(os.path.join(root, f"{run}{suffix}-seed{seed}", "**", "reasoning", task, "results.jsonl"),
                          recursive=True)
        if not paths:
            return None
        rows = [json.loads(l) for l in open(paths[0]) if l.strip()]
        # MINT appends and resumes by task id, so keep the last record per problem.
        by_id = {r["task"]["task_id"]: r for r in rows}
        if len(by_id) < n_expected:
            return None   # still running: never average in a partial task
        for tid, r in by_id.items():
            st = r["state"]
            steps = sum(1 for h in st.get("history", []) if h.get("role") == "assistant")
            problems[(task, tid)] = {
                "success": bool(st.get("success")),
                "steps": steps,
                "tokens": (st.get("token_counter") or {}).get("completion_tokens", 0),
                "error": bool(st.get("error")),
                # no-tool ablation only: turns that tried <execute> anyway
                "exec_attempts": (st.get("agent_action_count") or {}).get("execute_attempt", 0),
            }
    return problems


def seed_metrics(problems):
    n = len(problems)
    sr = {k: 100 * sum(p["success"] and p["steps"] <= k for p in problems.values()) / n
          for k in range(1, MAX_K + 1)}
    per_task = {t: 100 * sum(p["success"] for (tt, _), p in problems.items() if tt == t) / TASKS[t]
                for t in TASKS}
    total_tokens = sum(p["tokens"] for p in problems.values())
    total_steps = sum(p["steps"] for p in problems.values())
    # TC as in the paper: generated tokens per response. tok_problem is the whole
    # multi-turn session, i.e. the real cost of solving one MINT problem.
    tc = total_tokens / total_steps if total_steps else 0.0
    tok_problem = total_tokens / n
    return {
        "sr": sr, "delta": sr[MAX_K] - sr[1],
        "tokens": tok_problem,
        "tok_problem": tok_problem,
        "tc": tc,
        "apt": (sr[MAX_K] / tc * 1000) if tc else 0.0,                   # paper: Acc / TC x 1000
        "apt_problem": (sr[MAX_K] / tok_problem * 1000) if tok_problem else 0.0,
        "steps": total_steps / n,
        "errors": sum(p["error"] for p in problems.values()),
        "exec_attempt_rate": 100 * sum(p["exec_attempts"] > 0 for p in problems.values()) / n,
        "per_task": per_task,
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize the MINT matrix")
    parser.add_argument("--root", default="third_party/mint-bench/data/outputs")
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument("--variant", default="", help="Run-dir suffix, e.g. 'notool' for <name>-notool-seed<N>")
    parser.add_argument("--ablation", action="store_true", help="Llama-3.1-8B ablation rows instead of the main matrix")
    args = parser.parse_args()
    suffix = f"-{args.variant}" if args.variant else ""
    rows = ABLATION_ROWS if args.ablation else ROWS
    if args.out_prefix is None:
        args.out_prefix = ("results/tables/mint_table" + (f"_{args.variant}" if args.variant else "")
                           + ("_ablation" if args.ablation else ""))

    raw, agg = {}, {}
    for run, _, _, _ in rows:
        seeds = {s: load_seed(args.root, run, s, suffix) for s in SEEDS}
        raw[run] = {s: p for s, p in seeds.items() if p is not None}
        if not raw[run]:
            continue
        ms = {s: seed_metrics(p) for s, p in raw[run].items()}
        a = {"n_seeds": len(ms), "seeds": ms}
        for k in range(1, MAX_K + 1):
            a[f"sr{k}"] = mean_std([m["sr"][k] for m in ms.values()])
        a["delta"] = mean_std([m["delta"] for m in ms.values()])
        for key in ("tokens", "tok_problem", "tc", "apt", "apt_problem", "steps"):
            a[key] = mean_std([m[key] for m in ms.values()])
        for t in TASKS:
            a[t] = mean_std([m["per_task"][t] for m in ms.values()])
        a["errors"] = sum(m["errors"] for m in ms.values())
        agg[run] = a

    incomplete = [(run, len(raw.get(run, {}))) for run, _, _, _ in rows if len(raw.get(run, {})) < len(SEEDS)]
    if incomplete:
        print("INCOMPLETE (seeds with every task finished):")
        for run, n in incomplete:
            print(f"    {run}: {n}/{len(SEEDS)}")
        print()

    base_of = {fam: run for run, _, fam, is_base in rows if is_base}
    tests = {}
    for run, _, fam, is_base in rows:
        b = base_of[fam]
        if is_base or run not in agg or b not in agg:
            continue
        common = sorted(set(raw[run]) & set(raw[b]))
        gained = lost = 0
        for s in common:
            for key in set(raw[run][s]) & set(raw[b][s]):
                x, y = raw[run][s][key]["success"], raw[b][s][key]["success"]
                gained += x and not y
                lost += y and not x
        # Direction per seed, for the pre-registered "at least 2 of 3 seeds" rule.
        m, mb = agg[run]["seeds"], agg[b]["seeds"]
        h1 = sum(m[s]["sr"][MAX_K] > mb[s]["sr"][MAX_K] for s in common)
        h2 = sum(m[s]["delta"] > mb[s]["delta"] for s in common)
        h3 = sum(m[s]["tokens"] < mb[s]["tokens"] and m[s]["sr"][MAX_K] >= mb[s]["sr"][MAX_K] for s in common)
        tests[run] = {"gained": gained, "lost": lost, "p": mcnemar_exact(gained, lost),
                      "h1": h1, "h2": h2, "h3": h3, "n": len(common)}

    best = {}
    for run, _, fam, _ in rows:
        if run in agg and (fam not in best or agg[run]["sr5"][0] > agg[best[fam]]["sr5"][0]):
            best[fam] = run

    def pm(ms, fmt="{:.1f}"):
        return f"{fmt.format(ms[0])} ± {fmt.format(ms[1])}"

    md = ["| Model | SR@5 / Acc (↑) | TC (↓) | APT (↑) | SR@1 | ΔSR (↑) | Tok/problem (↓) | TheoremQA | MATH | MMLU | p (SR@5) |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for run, name, fam, is_base in rows:
        label = name.replace("\\quad + ", "&nbsp;&nbsp;+ ")
        if run not in agg:
            md.append(f"| {label} | — | — | — | — | — | — | — | — | — | — |")
            continue
        a = agg[run]
        sr5 = pm(a["sr5"])
        if best.get(fam) == run:
            sr5 = f"**{sr5}**"
        p = "—" if is_base or run not in tests else f"{tests[run]['p']:.3f}"
        md.append(f"| {label} | {sr5} | {pm(a['tc'], '{:.0f}')} | {pm(a['apt'], '{:.1f}')} | "
                  f"{pm(a['sr1'])} | {pm(a['delta'])} | {pm(a['tok_problem'], '{:.0f}')} | "
                  f"{pm(a['theoremqa'])} | {pm(a['math'])} | {pm(a['mmlu'])} | {p} |")
    md_text = "\n".join(md)

    def tpm(ms, fmt="{:.1f}"):
        return f"{fmt.format(ms[0])}\\textsubscript{{$\\pm$ {fmt.format(ms[1])}}}"

    # Column order mirrors Table 1 of the paper: Acc, TC, APT; the multi-turn
    # columns (SR@1, delta-SR) follow.
    tex = [r"\begin{tabular}{lccccc}", r"\toprule",
           r"\textbf{Model} & \textbf{Acc} ($\uparrow$) & \textbf{TC} ($\downarrow$) & "
           r"\textbf{APT} ($\uparrow$) & \textbf{SR@1} & "
           r"$\boldsymbol{\Delta}$\textbf{SR} ($\uparrow$) \\", r"\midrule"]
    prev = None
    for run, name, fam, _ in rows:
        if prev is not None and fam != prev:
            tex.append(r"\midrule")
        prev = fam
        if run not in agg:
            tex.append(f"{name} & -- & -- & -- & -- & -- \\\\")
            continue
        a = agg[run]
        sr5 = tpm(a["sr5"])
        if best.get(fam) == run:
            sr5 = f"\\textbf{{{sr5}}}"
        tex.append(f"{name} & {sr5} & {tpm(a['tc'], '{:.0f}')} & {tpm(a['apt'], '{:.1f}')} & "
                   f"{tpm(a['sr1'])} & {tpm(a['delta'])} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]

    print(md_text)
    print("\nHypothesis checks vs family base (seeds in favour / seeds compared; supported needs >= 2):")
    for run, _, _, _ in rows:
        if run in tests:
            t = tests[run]
            print(f"  {run:42s} H1 {t['h1']}/{t['n']}  H2 {t['h2']}/{t['n']}  H3 {t['h3']}/{t['n']}   "
                  f"SR@5 McNemar +{t['gained']}/-{t['lost']} p={t['p']:.4f}")
    if args.variant:
        print("\nProblems with an <execute> attempt (no-tool runs only; % , mean over seeds):")
        for run in agg:
            rates = [m["exec_attempt_rate"] for m in agg[run]["seeds"].values()]
            print(f"  {run:42s} {sum(rates) / len(rates):5.1f}")
    errs = {run: agg[run]["errors"] for run in agg if agg[run]["errors"]}
    if errs:
        print("\nProblems that ended in an agent/harness error (counted as failures):")
        for run, e in errs.items():
            print(f"  {run}: {e}")

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    with open(args.out_prefix + ".md", "w") as f:
        f.write(md_text + "\n")
    with open(args.out_prefix + ".tex", "w") as f:
        f.write("\n".join(tex) + "\n")
    print(f"\nWrote {args.out_prefix}.md and {args.out_prefix}.tex")


if __name__ == "__main__":
    main()
