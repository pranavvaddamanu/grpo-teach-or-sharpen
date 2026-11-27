# Experiment log

Every run in order, what went wrong, and what each one taught. Most of the early work was about getting a measurement that could be trusted.

## Run 0: the parser bug

**Setup.** Prompt asked for `<think>`/`<answer>` tags; the correctness reward required an English `<answer>` tag.

**Result.** Strict accuracy 0.41 → 0.51 after training. Response length collapsed from 56 to 22 characters, the model stopped reasoning, and pass@8 fell.

**Diagnosis.** When Qwen reasoned, it often switched to Chinese tags (`<思考>`, `<答案>`). The parser only accepted English tags, so correct reasoning was scored wrong, while bare `<answer>N</answer>` responses were scored normally. The reward penalized reasoning, and GRPO removed it within 30 steps. After the collapse, 80–100% of groups had identical rewards, so training stalled.

**Lesson.** A reward that is wrong about *what* it rewards steers RL in that direction quickly. Read raw outputs before trusting any number.

## Runs 1–2: prompt fixes

- A revised prompt and a parser accepting both tag languages were written but **not applied**: the hand-edited cell never ran. The "new" baseline reproduced Run 0 within noise.
- Adding a worked example to the prompt made the model **drop tags entirely**. It reasoned correctly in English, but strict accuracy was 0.000.

**Lesson (engineering).** Ship complete notebooks rather than cell-by-cell edits, and add a check before training that prints the prompt and output formats.

## Run 3 (`default`): first clean result

**Setup.** Pre-fill `<think>\n`; lenient correctness reward plus a 0.1 format reward; train on 2–3 operand problems.

**Baseline under fair measurement.** Strict 0.003, lenient **0.86**. The base model writes correct reasoning but closes with `</thought>` instead of `</think><answer>`. Every earlier baseline had mostly measured formatting.

**Result (lenient).**

| Split | Greedy | pass@1 | pass@8 |
|---|---|---|---|
| In-dist. | 0.860 → 0.913 | 0.713 → 0.902 | 0.973 → 0.940 |
| Longer | 0.687 → 0.630 | 0.470 → 0.606 | 0.873 → 0.803 |
| 3-digit | 0.473 → 0.480 | 0.357 → 0.472 | 0.603 → 0.547 |

Format compliance went from 0% to 100%.

**Diagnosis.** Sharpening: problems where the 8 samples disagree fell from 79% to 10% in-distribution, and coverage within 8 samples lost 61 problems while gaining 13. On long expressions, responses shrank by ~36% and the model merged operations into single lines, introducing errors. Training reward saturated near 0.9–1.0 by step 50, after which most groups gave no signal.

**Hypotheses.** (1) GRPO optimizes single-sample reward, which collapses diversity. (2) Short training problems reward brevity. (3) Nothing anchors the model to the base distribution.

## Experiments: testing the hypotheses

| Experiment | Change | Outcome | Conclusion |
|---|---|---|---|
| `hard_train` | Train on 3–4 operands | Longer reasoning; greedy and pass@8 up on every split; 5-operand 0.65 → 0.75; coverage +64 / −15 | Run 3's sharpening came from training on problems that were too easy |
| `format_only` | Format reward only | Empty think blocks, accuracy 0.86 → 0.08 | Correctness drives the gains; a format reward alone is exploited |
| `kl` | β = 0.04 | Diversity kept, no long-expression drop, smaller gains, format never learned | Anchoring prevents damage at the cost of gains (confounded by format) |

## Seeds and pass@64: what replicated

Two more training seeds each for `default` and `hard_train` (same data and test sets), plus pass@64 on 100 problems per OOD split for base vs `hard_train` seed 1.

**Replicated.**
- `hard_train` improves longer expressions in all 3 seeds: 0.81, 0.83, 0.78 vs base 0.69 (each p < 0.001), and the 5-operand subset 0.75, 0.81, 0.74 vs 0.65.
- Across all 6 trained runs, reasoning length on longer expressions correlates with accuracy there (r = 0.92).

**Did not replicate.**
- Run 3's pass@8 drop on every split: seeds 1 and 2 of `default` held or raised pass@8.
- `hard_train`'s gains on 3-digit numbers (0.55 in seed 0; 0.41 and 0.46 in seeds 1–2) and its in-distribution edge.
- `default` on longer expressions is highly seed-dependent: 0.63, 0.80, 0.65.

**Format learning is stochastic.** 3 of 6 runs learned `</think><answer>`. In the `default` runs, the two that learned it had the shortest reasoning and the lowest long-expression accuracy. That's suggestive, not established, at this sample size.

**pass@64.** On longer expressions, base and `hard_train` solve nearly the same problems given 64 tries (0.96 vs 0.98; 4 vs 2 unique, p = 0.69), so the gain there is reliability (pass@1 0.46 → 0.82). On 3-digit numbers, `hard_train` solves 16 problems the base never does vs 5 the other way (p = 0.03): modest evidence of expanded coverage, from one seed.

**Lesson.** Several conclusions from single runs were wrong or overstated. The robust result is narrower: harder training problems reliably improve generalization to longer problems, and that improvement mostly comes from reliability.

## Next

- A KL run that still learns the format, to remove the confound.
- Control format learning directly (e.g. a larger format reward, or SFT on the format first) to test whether learning `</think><answer>` causes the reasoning compression.
- More seeds for the 3-digit comparison and pass@64.

## Infrastructure notes

- Colab's free tier disconnects and enforces GPU quotas; Kaggle's *Save & Run All* runs in the background and was more reliable.
- On a T4, `torch.cuda.is_bf16_supported()` reports emulated support; checking compute capability avoids slow emulated runs and training crashes.
- New Kaggle notebooks default to no accelerator; the notebook now refuses to run without a GPU.
- Evaluations are cached per split and checkpoints are kept, so interrupted sessions resume.
