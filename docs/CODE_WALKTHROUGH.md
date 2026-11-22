# Code walkthrough

How the project works, from data generation to the statistics. Line references are to `rlmath.py` and the cells of `notebooks/grpo_arithmetic.ipynb`.

## 1. `rlmath.py`

The notebook writes this file in its `%%writefile rlmath.py` cell, so the notebook runs standalone. It has five parts.

### 1.1 Data generation

```python
@dataclass(frozen=True)
class Level:
    name: str
    n_operands: tuple[int, int]   # inclusive range, e.g. (2, 3)
    digits: tuple[int, int]       # digits per operand, e.g. (1, 2)
    ops: str = "+-*"
    max_mults: int = 1            # caps multiplications so answers stay small

LEVELS = {
    "train":      Level("train",      (2, 3), (1, 2)),
    "ood_length": Level("ood_length", (4, 5), (1, 2), max_mults=2),
    "ood_digits": Level("ood_digits", (2, 3), (3, 3)),
    "hard":       Level("hard",       (3, 4), (1, 2), max_mults=2),
}
```

A `Level` describes a difficulty: how many numbers, how many digits each, and how many multiplications are allowed.

- **`make_expression(rng, level)`** builds a random expression such as `36 - 8 * 7` and computes its answer with Python's `eval`, which applies standard operator precedence (−20 here). An `assert` ensures only digits, spaces and `+ - *` ever reach `eval`.
- **`make_split(level, n, seed, exclude)`** draws `n` unique expressions from a seeded RNG, skipping anything in `exclude`. The same seed always gives the same problems.
- **`build_datasets(n_train, n_eval, seed)`** produces five splits:

| Split | Level | Seed | Excludes |
|---|---|---|---|
| `train` | train | 0 | — |
| `test_id` | train | 1 | `train` |
| `test_ood_length` | ood_length | 2 | — |
| `test_ood_digits` | ood_digits | 3 | — |
| `train_hard` | hard | 10 | all test splits |

The `exclude` sets prevent leakage. `train_hard` is generated last, so adding it left the test sets identical to earlier runs, and every run is evaluated on exactly the same 900 problems.

### 1.2 Prompt

```python
SYSTEM_PROMPT = ("You are a careful math assistant. Respond in English only. "
                 "Reason step by step inside <think> </think> tags, then give only the final "
                 "integer inside <answer> </answer> tags.")
PREFILL = "<think>\n"

def render_prompt(tok, row):
    return tok.apply_chat_template(row["prompt"], tokenize=False,
                                   add_generation_prompt=True) + PREFILL
```

- `apply_chat_template` formats the system and user messages in Qwen's chat format and ends with the assistant-turn marker.
- `+ PREFILL` writes the first token of the response for the model, so generation continues *inside* the think block. The prompt alone didn't produce a reliable format.
- The same function builds prompts for training and for evaluation, so both see identical inputs.

### 1.3 Parsing

```python
_ANSWER_RE = re.compile(r"<(?:answer|答案)>\s*(.*?)\s*(?:</(?:answer|答案)>|$)", re.S)
_FORMAT_RE = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.S)
```

- `_ANSWER_RE` finds an answer block. It accepts the Chinese tag `答案` and a missing closing tag, because Qwen produces both.
- `_FORMAT_RE` is the strict format check used by the format reward.

Two ways to read an answer:

| Function | Rule | Measures |
|---|---|---|
| `extract_answer` | Last integer inside an answer tag; `None` if there is no tag | Format compliance + math |
| `extract_answer_lenient` | The tagged answer if present, else the last integer in the response | Math only |

`completion_text(c)` re-attaches `PREFILL`, since the trainer passes only the generated tokens. `output_style(text)` labels each response `format_ok`, `answer_tag_only` or `no_answer_tag`.

### 1.4 Rewards

```python
def correctness_reward(prompts, completions, answer, **kw):
    return [1.0 if extract_answer_lenient(completion_text(c)) == int(a) else 0.0
            for c, a in zip(completions, answer)]

def format_reward(prompts, completions, **kw):
    return [0.1 if _FORMAT_RE.search(completion_text(c)) else 0.0 for c in completions]
```

- TRL calls each reward function with the prompts, the completions and every extra dataset column as a keyword argument. Here that's `answer`, the correct answer. Each function returns one float per completion, and TRL sums them, so the maximum reward is 1.1.
- Correctness uses the **lenient** parser. With the strict parser, ~99% of base-model responses score 0, so almost every group of 8 would score all zeros and give GRPO no gradient.
- `correctness_reward_strict` is kept for ablations.

### 1.5 Evaluation

```python
def generate(model, tok, rows, n_samples=1, temperature=0.0, max_new_tokens=256, batch_size=32):
```
- **Left padding.** Prompts in a batch have different lengths, and generation continues from the right edge, so padding goes on the left.
- **`num_return_sequences=n_samples`** returns several samples per prompt in one call.
- **Temperature 0** is greedy decoding (one deterministic answer). **Temperature 0.8** samples, which is what pass@k needs.
- The prompt tokens are sliced off the output and `PREFILL` is prepended to each decoded response.

```python
def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
```
The unbiased estimator from the Codex paper: with `n` samples of which `c` are correct, the probability that a random size-`k` subset contains a correct one is 1 − C(n−c, k) / C(n, k). The product form avoids large factorials.

```python
def evaluate(model, tok, rows, n_samples=8, ks=(1, 2, 4, 8), temperature=0.8, ...):
```
1. **Greedy pass:** one answer per problem gives `greedy_acc` (strict), `lenient_acc`, `format_rate`, `answer_tag_rate` and `mean_chars`.
2. **Sampled pass:** 8 answers per problem give `n_correct` and `n_correct_lenient` per problem, then `pass@k` and `lenient_pass@k`.
3. **Records:** per-problem text, prediction and correctness. These make the paired tests and error analysis possible.

## 2. The notebook

### 2.1 Setup

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
%pip install -q "trl>=0.19" peft accelerate datasets matplotlib
%pip uninstall -y -q torchao
```
- Restricting to one GPU avoids the trainer's automatic multi-GPU mode on Kaggle's 2×T4, which breaks GRPO generation. This must run before `torch` is imported.
- `torchao` is uninstalled because Colab and Kaggle preinstall a version that current `peft` rejects.

The next cell mounts Google Drive on Colab, so results survive disconnects, and otherwise falls back to a local `runs/` folder. Kaggle ships a stub `google.colab` whose `mount()` raises, which is why the code catches every exception.

### 2.2 Configuration

```python
CFG = dict(n_train=4000, n_eval=300, max_steps=250, lr=2e-5, beta=0.0,
           num_generations=8, per_device_bs=8, grad_accum=2,
           max_completion_length=256, temperature=1.0, lora_r=16, lora_alpha=32, eval_samples=8)
use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
assert SMOKE or DEVICE == "cuda"
```

| Key | Meaning |
|---|---|
| `max_steps` | Optimizer updates |
| `lr` | Learning rate; LoRA tolerates a higher rate than full fine-tuning |
| `beta` | KL penalty weight toward the base model (0 = none) |
| `num_generations` | Group size: samples per prompt |
| `per_device_bs × grad_accum` | 16 completions per update = **2 prompts × 8 samples** |
| `max_completion_length` | Token limit per response |
| `temperature` | Sampling temperature *during training*, which keeps the group diverse |

250 steps × 2 prompts = **500 training problems seen**, one-eighth of the 4,000 generated.

`torch.cuda.is_bf16_supported()` returns `True` on a T4 through slow emulation, so the code checks for compute capability ≥ 8.0 instead (the T4 is 7.5). The `assert` stops the run immediately if no GPU is attached.

### 2.3 Baseline

`run_eval(model, tag, out_dir)` evaluates every test split, **saves after each split and skips any split already on disk**, so an interrupted session resumes where it stopped. The next cell prints the prompt and the output-style counts as a check before training.

### 2.4 Experiments

```python
EXPERIMENTS = {
    "default":       dict(),
    "hard_train":    dict(train_split="train_hard"),
    "format_only":   dict(rewards=["format"]),
    "kl":            dict(beta=0.04),
    "hard_train_s1": dict(train_split="train_hard", seed=1), ...
}
RUN_LIST = [...]
```
Each experiment overrides one setting of the defaults. `run_experiment(name)`:

1. **Merges configs:** `{**CFG, defaults, **EXPERIMENTS[name]}`, with later keys winning.
2. **Builds the dataset:** rows of `{"prompt": render_prompt(...), "answer": ...}`. The `answer` column reaches the reward functions.
3. **Configures training** with `GRPOConfig`: constant learning rate after 10 warmup steps, fp16 mixed precision on the T4, gradient checkpointing to save memory, and a checkpoint every 50 steps (all kept).
4. **Adds LoRA** with `LoraConfig(r=16, lora_alpha=32, target_modules=[q,k,v,o,gate,up,down]_proj)`. Each targeted weight W gets a trainable low-rank update B·A. That's about 1% of the parameters; the base weights stay frozen.
5. **Trains** with `GRPOTrainer(...).train(resume_from_checkpoint=<checkpoint exists>)`.
6. **Saves** the adapter and `train_log.json`, then calls `merge_and_unload()`, which folds B·A into W for faster generation.
7. **Evaluates** with `run_eval`, cached per experiment folder.

Experiments that already have a `final_adapter` skip training. Finished experiments from earlier sessions in the same output folder are loaded for comparison.

### 2.5 Analysis cells

- **Comparison table:** lenient greedy accuracy, pass@1, pass@8, format rate and mean characters per model and split, plus the 5-operand slice, which is unseen even by `hard_train`.
- **Paired tests:** *fixed* (base wrong → model right) and *broken* (base right → model wrong), compared with an exact McNemar test (a binomial test on discordant pairs). The same test is applied to coverage: *new* (base solved 0/8 → model ≥1/8) vs *lost*. Paired tests use the fact that both models answer the same problems, which makes them more sensitive than comparing two accuracy numbers.
- **pass@k plot** and **training curves** (reward, response length, share of groups with no signal, KL).
- **Seed table:** mean ± std across seeds 0–2. Seed-0 numbers from earlier sessions are entered in `PRIOR`.
- **pass@64:** base vs `hard_train_s1` on 100 problems per OOD split. It skips itself if the session has run past 8 hours.

## 3. One GRPO step

1. Sample 2 prompts from the training set.
2. Generate 8 completions per prompt at temperature 1.0 (16 total).
3. Score each completion: correctness (0/1) plus format (0/0.1).
4. **Advantage** within each group: (reward − group mean) / group std. If 6 of 8 are correct, correct answers get roughly +0.5 and wrong ones roughly −1.6. If all 8 score the same, every advantage is 0 and the group contributes no gradient. The `frac_reward_zero_std` curve tracks this.
5. **Loss:** −advantage × log-probability of each generated token, averaged over all tokens in the batch (TRL's default `loss_type="dapo"`). Minimizing it raises the probability of above-average completions and lowers below-average ones.
6. **Clipping** (ε = 0.2) limits the policy ratio. With TRL's default of one optimization pass per generation batch, the ratio is 1, so clipping never activates and the update is a plain policy gradient with a group baseline.
7. **KL term** (only when β > 0): β × the divergence from the base model. With LoRA, the reference model is the same network with the adapter disabled, so it costs an extra forward pass but no extra memory.
8. Update the LoRA weights.

GRPO replaces PPO's learned value network (critic) with the group mean as the baseline. Dropping the critic is what makes RL fit on a 16 GB T4.

## 4. `analysis/`

- `analyze.py` reproduces the paired analysis from any two `*_eval.json` files: McNemar on greedy and coverage, paired-bootstrap 95% CIs for pass@1 and pass@8 changes, sample disagreement, length and accuracy by operand count.
- `make_figures.py` regenerates every figure in the README from `results/`.
