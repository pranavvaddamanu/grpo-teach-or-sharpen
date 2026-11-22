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

## Next

- Two more seeds each for `default` and `hard_train` (mean ± std).
- pass@64 for base vs `hard_train` on the OOD splits, to separate new ability from reliability.
- A KL run that still learns the format, to remove the confound.

## Infrastructure notes

- Colab's free tier disconnects and enforces GPU quotas; Kaggle's *Save & Run All* runs in the background and was more reliable.
- On a T4, `torch.cuda.is_bf16_supported()` reports emulated support; checking compute capability avoids slow emulated runs and training crashes.
- New Kaggle notebooks default to no accelerator; the notebook now refuses to run without a GPU.
- Evaluations are cached per split and checkpoints are kept, so interrupted sessions resume.
