# Scientific-quality evaluation — results

Judge: `deepseek/deepseek-v4-pro-0813` (single judge, see the 2026-10-07 amendment). Win rate = CollabSciReM's share of wins (ties count half); 50% = parity; 95% bootstrap CI over prompts.

## Summary

| Comparison | Hyp. | n | tokens/turn (CSR vs other) | order consistency | rigor strict | rigor order-avg | overall strict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama3b-vs-collabllm | H1 | 300 | 584 vs 555 | 52% | 54.8% [51.4%, 58.3%] | 55.2% [51.6%, 58.8%] | 56.5% [52.3%, 60.7%] |
| llama3b-vs-base | H3 | 300 | 584 vs 943 | 50% | 53.3% [49.8%, 56.9%] | 53.2% [49.6%, 56.9%] | 53.7% [49.7%, 57.7%] |
| llama8b-vs-collabllm | H1 | 300 | 474 vs 474 | 58% | 64.2% [60.5%, 68.0%] | 64.8% [61.0%, 68.5%] | 65.3% [61.2%, 69.3%] |
| llama8b-vs-base | H3 | 300 | 474 vs 963 | 45% | 45.4% [41.9%, 48.8%] | 45.2% [41.7%, 48.7%] | 45.5% [41.8%, 49.3%] |
| qwen3b-vs-collabllm | H1 | 300 | 388 vs 351 | 52% | 52.1% [48.5%, 55.5%] | 52.7% [49.0%, 56.4%] | 52.7% [48.3%, 57.0%] |
| qwen3b-vs-base | H3 | 300 | 388 vs 1199 | 44% | 34.3% [31.3%, 37.2%] | 34.1% [31.0%, 37.2%] | 34.5% [31.3%, 37.8%] |
| qwen7b-vs-collabllm | H1 | 300 | 542 vs 499 | 55% | 61.9% [58.3%, 65.5%] | 62.3% [58.7%, 66.0%] | 61.8% [57.7%, 65.8%] |
| qwen7b-vs-base | H3 | 300 | 542 vs 1306 | 49% | 34.3% [31.2%, 37.3%] | 33.4% [30.2%, 36.5%] | 33.5% [30.2%, 36.8%] |
| llama8b-vs-ablation2 | H2 | 300 | 474 vs 484 | 55% | 59.7% [56.1%, 63.2%] | 60.0% [56.2%, 63.7%] | 60.8% [56.7%, 65.0%] |
| llama8b-vs-ablation3 | exploratory | 300 | 474 vs 1512 | 52% | 37.3% [34.1%, 40.7%] | 37.2% [33.9%, 40.5%] | 36.7% [33.0%, 40.3%] |
| llama8b-vs-ablation4 | exploratory | 300 | 474 vs 511 | 59% | 63.5% [59.7%, 67.3%] | 64.1% [60.2%, 67.9%] | 64.7% [60.5%, 68.8%] |
| llama8b-vs-sensitivity | exploratory | 300 | 474 vs 486 | 54% | 49.3% [45.4%, 53.2%] | 49.3% [45.2%, 53.3%] | 49.3% [45.0%, 53.5%] |

## Length-controlled rigor win rate (robustness check)

Strict rigor score regressed (OLS) on log(CollabSciReM tokens / other tokens) per prompt; the intercept is the win rate at equal length. 'CSR longer' = share of prompts where CollabSciReM's conversation is the longer one; near 0% means the equal-length value is an extrapolation.

| Comparison | raw | length-controlled [95% CI] | slope per doubling | CSR longer |
| --- | --- | --- | --- | --- |
| llama3b-vs-collabllm | 54.8% | 58.2% [54.3%, 61.9%] | -46.4% | 72% |
| llama3b-vs-base | 53.3% | 29.9% [18.9%, 41.8%] | -34.2% | 0% |
| llama8b-vs-collabllm | 64.2% | 64.2% [60.7%, 67.9%] | -22.1% | 48% |
| llama8b-vs-base | 45.4% | 16.0% [0.0%, 32.7%] | -29.1% | 0% |
| qwen3b-vs-collabllm | 52.1% | 49.2% [44.5%, 54.1%] | +20.5% | 89% |
| qwen3b-vs-base | 34.3% | 24.1% [-1.9%, 49.3%] | -6.3% | 0% |
| qwen7b-vs-collabllm | 61.9% | 62.5% [57.8%, 67.0%] | -5.4% | 82% |
| qwen7b-vs-base | 34.3% | 11.8% [-8.2%, 31.5%] | -17.8% | 0% |
| llama8b-vs-ablation2 | 59.7% | 58.9% [55.2%, 62.6%] | -23.4% | 37% |
| llama8b-vs-ablation3 | 37.3% | -8.1% [-35.8%, 21.3%] | -27.2% | 0% |
| llama8b-vs-ablation4 | 63.5% | 62.8% [57.7%, 67.6%] | -6.7% | 17% |
| llama8b-vs-sensitivity | 49.3% | 48.9% [44.6%, 52.9%] | -11.6% | 37% |

## Hypotheses (strict rigor composite)

* **H1** CollabSciReM vs CollabLLM-science, pooled over families: 58.2% [56.5%, 60.0%], direction favours CollabSciReM in 4/4 families → **supported**
* **H2 vs R_gen + R_struct (Llama-8B)**: 59.7% [56.1%, 63.2%] → **supported**
* **H3** vs base, llama3b: 53.3% [49.8%, 56.8%] → **inconclusive**
* **H3** vs base, llama8b: 45.4% [42.0%, 48.8%] → **contradicted**
* **H3** vs base, qwen3b: 34.3% [31.3%, 37.3%] → **contradicted**
* **H3** vs base, qwen7b: 34.3% [31.3%, 37.4%] → **contradicted**

Judge cost: $95.40

## Per-dimension results

### llama3b-vs-collabllm (H1): llama3b-collabscirem vs llama3b-collabllm, n = 300

| Dimension | strict | order-averaged |
| --- | --- | --- |
| methodological_soundness | 56.0% [51.7%, 60.3%] | 56.5% [52.2%, 60.8%] |
| justification | 54.2% [50.3%, 58.0%] | 54.0% [49.8%, 58.0%] |
| logical_consistency | 53.3% [49.7%, 57.0%] | 54.4% [50.5%, 58.2%] |
| factual_correctness | 55.7% [51.5%, 59.8%] | 56.0% [51.8%, 60.1%] |
| novelty | 53.2% [49.7%, 56.8%] | 54.8% [50.8%, 58.7%] |
| responsiveness | 56.2% [52.2%, 60.2%] | 56.6% [52.4%, 60.6%] |
| overall | 56.5% [52.2%, 60.7%] | 56.5% [52.3%, 60.7%] |

### llama3b-vs-base (H3): llama3b-collabscirem vs llama3b-base, n = 300

| Dimension | strict | order-averaged |
| --- | --- | --- |
| methodological_soundness | 54.5% [50.5%, 58.5%] | 54.5% [50.5%, 58.5%] |
| justification | 51.3% [47.3%, 55.3%] | 51.7% [47.6%, 55.9%] |
| logical_consistency | 53.3% [49.5%, 57.0%] | 53.7% [49.8%, 57.6%] |
| factual_correctness | 54.0% [50.0%, 58.0%] | 53.2% [49.2%, 57.2%] |
| novelty | 51.7% [48.0%, 55.3%] | 52.4% [48.6%, 56.3%] |
| responsiveness | 48.7% [44.5%, 52.8%] | 48.2% [44.0%, 52.5%] |
| overall | 53.7% [49.7%, 57.7%] | 53.8% [49.7%, 57.8%] |

### llama8b-vs-collabllm (H1): llama8b-collabscirem vs llama8b-collabllm, n = 300

| Dimension | strict | order-averaged |
| --- | --- | --- |
| methodological_soundness | 65.0% [61.0%, 68.8%] | 65.2% [61.1%, 69.1%] |
| justification | 61.8% [58.0%, 65.7%] | 63.3% [59.4%, 67.2%] |
| logical_consistency | 65.5% [61.7%, 69.3%] | 66.0% [62.2%, 69.9%] |
| factual_correctness | 64.7% [60.7%, 68.7%] | 64.6% [60.6%, 68.5%] |
| novelty | 62.8% [59.0%, 66.7%] | 63.2% [59.4%, 67.1%] |
| responsiveness | 64.8% [60.8%, 68.7%] | 65.4% [61.3%, 69.5%] |
| overall | 65.3% [61.3%, 69.2%] | 65.3% [61.2%, 69.3%] |

### llama8b-vs-base (H3): llama8b-collabscirem vs llama8b-base, n = 300

| Dimension | strict | order-averaged |
| --- | --- | --- |
| methodological_soundness | 45.7% [41.8%, 49.5%] | 45.7% [41.9%, 49.5%] |
| justification | 43.8% [40.0%, 47.7%] | 43.9% [40.1%, 47.8%] |
| logical_consistency | 45.0% [41.3%, 48.5%] | 44.0% [40.4%, 47.6%] |
| factual_correctness | 47.0% [43.2%, 50.7%] | 47.1% [43.3%, 50.9%] |
| novelty | 46.0% [42.3%, 49.7%] | 45.3% [41.7%, 49.0%] |
| responsiveness | 42.3% [38.5%, 46.2%] | 42.0% [38.1%, 45.8%] |
| overall | 45.5% [41.7%, 49.3%] | 45.5% [41.8%, 49.3%] |

### qwen3b-vs-collabllm (H1): qwen3b-collabscirem vs qwen3b-collabllm, n = 300

| Dimension | strict | order-averaged |
| --- | --- | --- |
| methodological_soundness | 52.2% [48.0%, 56.3%] | 52.2% [47.9%, 56.4%] |
| justification | 51.8% [48.3%, 55.2%] | 53.4% [49.6%, 57.2%] |
| logical_consistency | 51.0% [47.3%, 54.7%] | 51.6% [47.7%, 55.7%] |
| factual_correctness | 53.3% [49.2%, 57.5%] | 53.5% [49.3%, 57.7%] |
| novelty | 52.5% [49.3%, 55.8%] | 53.4% [49.7%, 57.0%] |
| responsiveness | 54.5% [50.2%, 58.8%] | 54.7% [50.2%, 59.0%] |
| overall | 52.7% [48.3%, 57.0%] | 52.8% [48.5%, 57.1%] |

### qwen3b-vs-base (H3): qwen3b-collabscirem vs qwen3b-base, n = 300

| Dimension | strict | order-averaged |
| --- | --- | --- |
| methodological_soundness | 34.2% [31.0%, 37.5%] | 34.3% [31.1%, 37.7%] |
| justification | 33.2% [30.0%, 36.3%] | 32.6% [29.3%, 35.9%] |
| logical_consistency | 31.5% [28.3%, 34.7%] | 31.0% [27.7%, 34.3%] |
| factual_correctness | 38.3% [35.0%, 41.8%] | 38.3% [35.0%, 41.7%] |
| novelty | 33.3% [30.5%, 36.2%] | 32.0% [29.1%, 34.9%] |
| responsiveness | 26.0% [23.0%, 29.2%] | 26.0% [23.0%, 29.0%] |
| overall | 34.5% [31.3%, 37.8%] | 34.6% [31.2%, 37.9%] |

### qwen7b-vs-collabllm (H1): qwen7b-collabscirem vs qwen7b-collabllm, n = 300

| Dimension | strict | order-averaged |
| --- | --- | --- |
| methodological_soundness | 63.8% [59.8%, 67.8%] | 63.8% [59.8%, 67.8%] |
| justification | 61.5% [57.7%, 65.3%] | 62.4% [58.4%, 66.4%] |
| logical_consistency | 61.7% [58.0%, 65.3%] | 62.6% [58.8%, 66.2%] |
| factual_correctness | 60.5% [56.3%, 64.5%] | 60.5% [56.4%, 64.6%] |
| novelty | 63.5% [59.8%, 67.2%] | 64.2% [60.5%, 67.8%] |
| responsiveness | 64.3% [60.3%, 68.2%] | 64.9% [60.8%, 69.0%] |
| overall | 61.8% [57.7%, 66.0%] | 61.8% [57.7%, 66.0%] |

### qwen7b-vs-base (H3): qwen7b-collabscirem vs qwen7b-base, n = 300

| Dimension | strict | order-averaged |
| --- | --- | --- |
| methodological_soundness | 32.7% [29.2%, 36.0%] | 32.7% [29.2%, 36.2%] |
| justification | 31.3% [27.8%, 34.8%] | 29.9% [26.5%, 33.5%] |
| logical_consistency | 35.7% [32.3%, 39.0%] | 33.2% [29.8%, 36.8%] |
| factual_correctness | 37.5% [34.0%, 41.2%] | 37.6% [34.1%, 41.2%] |
| novelty | 32.3% [29.2%, 35.5%] | 30.8% [27.6%, 34.2%] |
| responsiveness | 26.8% [23.7%, 30.2%] | 25.8% [22.5%, 29.1%] |
| overall | 33.5% [30.0%, 37.0%] | 33.5% [30.0%, 37.1%] |

### llama8b-vs-ablation2 (H2): llama8b-collabscirem vs llama8b-ablation2, n = 300

| Dimension | strict | order-averaged |
| --- | --- | --- |
| methodological_soundness | 60.3% [56.2%, 64.5%] | 60.4% [56.2%, 64.5%] |
| justification | 60.7% [56.7%, 64.7%] | 60.3% [56.2%, 64.4%] |
| logical_consistency | 57.3% [53.7%, 61.0%] | 58.5% [54.7%, 62.3%] |
| factual_correctness | 60.3% [56.3%, 64.3%] | 60.6% [56.5%, 64.6%] |
| novelty | 58.3% [54.3%, 62.3%] | 58.1% [53.8%, 62.3%] |
| responsiveness | 61.0% [57.0%, 64.8%] | 61.1% [57.1%, 65.1%] |
| overall | 60.8% [56.7%, 65.0%] | 60.8% [56.6%, 65.0%] |

### llama8b-vs-ablation3 (exploratory): llama8b-collabscirem vs llama8b-ablation3, n = 300

| Dimension | strict | order-averaged |
| --- | --- | --- |
| methodological_soundness | 36.7% [33.0%, 40.2%] | 36.6% [32.9%, 40.1%] |
| justification | 30.5% [27.0%, 34.2%] | 30.5% [26.9%, 34.2%] |
| logical_consistency | 38.0% [34.5%, 41.7%] | 37.5% [33.8%, 41.1%] |
| factual_correctness | 44.2% [40.3%, 48.0%] | 44.1% [40.2%, 47.9%] |
| novelty | 28.0% [24.8%, 31.2%] | 27.0% [23.8%, 30.2%] |
| responsiveness | 21.7% [18.7%, 24.8%] | 21.4% [18.4%, 24.6%] |
| overall | 36.7% [33.0%, 40.3%] | 36.7% [33.0%, 40.3%] |

### llama8b-vs-ablation4 (exploratory): llama8b-collabscirem vs llama8b-ablation4, n = 300

| Dimension | strict | order-averaged |
| --- | --- | --- |
| methodological_soundness | 65.0% [60.8%, 69.0%] | 65.1% [60.8%, 69.3%] |
| justification | 64.2% [60.2%, 68.0%] | 65.3% [61.3%, 69.3%] |
| logical_consistency | 61.0% [57.0%, 64.8%] | 62.0% [58.1%, 66.1%] |
| factual_correctness | 63.8% [59.7%, 67.8%] | 64.0% [59.8%, 68.1%] |
| novelty | 62.0% [58.0%, 65.8%] | 62.7% [58.7%, 66.7%] |
| responsiveness | 62.0% [58.0%, 66.0%] | 62.7% [58.6%, 66.8%] |
| overall | 64.7% [60.5%, 68.7%] | 64.7% [60.5%, 68.8%] |

### llama8b-vs-sensitivity (exploratory): llama8b-collabscirem vs llama8b-sensitivity, n = 300

| Dimension | strict | order-averaged |
| --- | --- | --- |
| methodological_soundness | 49.5% [45.3%, 53.7%] | 49.4% [45.2%, 53.6%] |
| justification | 49.2% [45.2%, 53.2%] | 49.3% [45.1%, 53.5%] |
| logical_consistency | 49.3% [45.3%, 53.3%] | 49.2% [45.2%, 53.2%] |
| factual_correctness | 49.2% [44.8%, 53.5%] | 49.3% [45.1%, 53.5%] |
| novelty | 49.3% [45.5%, 53.3%] | 49.4% [45.4%, 53.4%] |
| responsiveness | 48.7% [44.5%, 52.8%] | 48.8% [44.8%, 52.9%] |
| overall | 49.3% [45.2%, 53.5%] | 49.3% [45.2%, 53.7%] |

