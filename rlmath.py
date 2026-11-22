"""rlmath: GRPO with verifiable rewards on synthetic arithmetic.

Data generation with in-distribution / out-of-distribution splits,
reward functions, and pass@k evaluation. Shared by the Colab notebook.
"""
from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, asdict

import numpy as np

# ---------------------------------------------------------------- data

SYSTEM_PROMPT = (
    "You are a careful math assistant. Respond in English only. "
    "Reason step by step inside <think> </think> tags, then give only the final "
    "integer inside <answer> </answer> tags."
)
# Every response is started with this text, so the model continues from inside the
# think block. Prompting alone was unreliable: Qwen switched to Chinese tags (Run 0)
# or dropped tags entirely when shown an example (Run 2). Same trick as TinyZero.
PREFILL = "<think>\n"


@dataclass(frozen=True)
class Level:
    name: str
    n_operands: tuple[int, int]   # inclusive range
    digits: tuple[int, int]       # inclusive range of digits per operand
    ops: str = "+-*"
    max_mults: int = 1            # cap multiplications so answers stay sane


# Train on easy-to-medium; hold out longer expressions and bigger numbers.
LEVELS = {
    "train":      Level("train",      (2, 3), (1, 2)),
    "ood_length": Level("ood_length", (4, 5), (1, 2), max_mults=2),
    "ood_digits": Level("ood_digits", (2, 3), (3, 3)),
    # Harder training distribution: 3-4 operands. Overlaps test_ood_length at 4 operands,
    # so 5-operand problems are the clean OOD slice for this experiment.
    "hard": Level("hard", (3, 4), (1, 2), max_mults=2),
}


def _rand_operand(rng: random.Random, digits: tuple[int, int]) -> int:
    d = rng.randint(*digits)
    return rng.randint(10 ** (d - 1) if d > 1 else 1, 10 ** d - 1)


def make_expression(rng: random.Random, level: Level) -> tuple[str, int]:
    n = rng.randint(*level.n_operands)
    nums = [_rand_operand(rng, level.digits) for _ in range(n)]
    ops, mults = [], 0
    for _ in range(n - 1):
        choices = [o for o in level.ops if not (o == "*" and mults >= level.max_mults)]
        o = rng.choice(choices)
        mults += o == "*"
        ops.append(o)
    expr = str(nums[0])
    for o, x in zip(ops, nums[1:]):
        expr += f" {o} {x}"
    # Only digits, spaces and + - * reach eval.
    assert re.fullmatch(r"[\d +\-*]+", expr)
    return expr, int(eval(expr))


def make_split(level: Level, n: int, seed: int, exclude: set[str] | None = None) -> list[dict]:
    rng = random.Random(seed)
    exclude = exclude or set()
    seen, rows = set(), []
    tries = 0
    while len(rows) < n:
        tries += 1
        if tries > n * 50:
            raise RuntimeError(f"could not draw {n} unique problems for {level.name}")
        expr, ans = make_expression(rng, level)
        if expr in seen or expr in exclude:
            continue
        seen.add(expr)
        rows.append({"expression": expr, "answer": ans, "level": level.name,
                     "n_operands": expr.count(" ") // 2 + 1})
    return rows


def to_prompt(row: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Compute: {row['expression']}"},
    ]


def render_prompt(tok, row: dict) -> str:
    """Chat-formatted prompt string ending with PREFILL. Used for both training and eval."""
    return tok.apply_chat_template(row["prompt"], tokenize=False,
                                   add_generation_prompt=True) + PREFILL


def build_datasets(n_train=4000, n_eval=300, seed=0) -> dict[str, list[dict]]:
    train = make_split(LEVELS["train"], n_train, seed)
    train_exprs = {r["expression"] for r in train}
    splits = {
        "train": train,
        "test_id": make_split(LEVELS["train"], n_eval, seed + 1, exclude=train_exprs),
        "test_ood_length": make_split(LEVELS["ood_length"], n_eval, seed + 2),
        "test_ood_digits": make_split(LEVELS["ood_digits"], n_eval, seed + 3),
    }
    # Built after the test sets (which stay identical to earlier runs) and excluding
    # every test expression, so no test problem appears in the hard training set.
    test_exprs = {r["expression"] for k, v in splits.items() if k.startswith("test") for r in v}
    splits["train_hard"] = make_split(LEVELS["hard"], n_train, seed + 10, exclude=test_exprs)
    for rows in splits.values():
        for r in rows:
            r["prompt"] = to_prompt(r)
    return splits


# ------------------------------------------------------------- parsing

# Qwen sometimes switches tags to Chinese (<思考>/<答案>) and may omit the closing tag.
# The parser accepts both, so a correct answer is never scored wrong because of tag style
# (that bug made Run 0 punish reasoning and collapse to bare answers).
_ANSWER_RE = re.compile(r"<(?:answer|答案)>\s*(.*?)\s*(?:</(?:answer|答案)>|$)", re.S)
# The format reward stays strict: English think block followed by answer block.
_FORMAT_RE = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.S)


def completion_text(c) -> str:
    """Full response text: PREFILL plus what the model generated.

    TRL passes completions as strings for string prompts (what we use) and as
    [{'role':..,'content':..}] for chat prompts; both are handled.
    """
    if isinstance(c, list):
        c = c[0]["content"] if c else ""
    return c if c.startswith(PREFILL) else PREFILL + c


def extract_answer(text: str) -> int | None:
    m = _ANSWER_RE.findall(text)
    if not m:
        return None
    nums = re.findall(r"-?\d[\d,]*", m[-1])
    if not nums:
        return None
    try:
        return int(nums[-1].replace(",", ""))
    except ValueError:
        return None


def extract_answer_lenient(text: str) -> int | None:
    """Tagged answer if present, else the last integer (used by correctness_reward).

    Lets us separate "got the math wrong" from "got the format wrong".
    """
    a = extract_answer(text)
    if a is not None:
        return a
    nums = re.findall(r"-?\d[\d,]*", text)
    try:
        return int(nums[-1].replace(",", "")) if nums else None
    except ValueError:
        return None


def output_style(text: str) -> str:
    """Coarse label of how the model formatted its response."""
    if _FORMAT_RE.search(text):
        return "format_ok"
    if _ANSWER_RE.search(text):
        return "answer_tag_only"
    return "no_answer_tag"


# ------------------------------------------------------------- rewards
# Signature required by TRL: (prompts, completions, **dataset_columns) -> list[float]

def correctness_reward(prompts, completions, answer, **kw) -> list[float]:
    """Lenient: tagged answer if present, else the last integer in the response.

    The strict tag-only version scored correct math as wrong whenever the model closed
    its reasoning with </thought> instead of </think><answer>, which happened on 99% of
    base-model outputs (Run 3: strict 0.003 vs lenient 0.86). Format is taught separately
    by format_reward, so correctness measures math only.
    """
    return [1.0 if extract_answer_lenient(completion_text(c)) == int(a) else 0.0
            for c, a in zip(completions, answer)]


def correctness_reward_strict(prompts, completions, answer, **kw) -> list[float]:
    """Tag-only version, kept for the ablation."""
    return [1.0 if extract_answer(completion_text(c)) == int(a) else 0.0
            for c, a in zip(completions, answer)]


def format_reward(prompts, completions, **kw) -> list[float]:
    return [0.1 if _FORMAT_RE.search(completion_text(c)) else 0.0 for c in completions]


# ---------------------------------------------------------- evaluation

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator from Chen et al. (2021), Codex paper."""
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def generate(model, tok, rows, n_samples=1, temperature=0.0, max_new_tokens=256,
             batch_size=32) -> list[list[str]]:
    """Return n_samples completions per row (HF generate, left-padded batches)."""
    import torch
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    texts = [render_prompt(tok, r) for r in rows]
    out: list[list[str]] = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True).to(model.device)
        gen_kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                      num_return_sequences=n_samples)
        if temperature > 0:
            gen_kw.update(do_sample=True, temperature=temperature, top_p=1.0)
        else:
            gen_kw.update(do_sample=False)
        with torch.no_grad():
            g = model.generate(**enc, **gen_kw)
        dec = [PREFILL + d for d in
               tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
        for j in range(0, len(dec), n_samples):
            out.append(dec[j:j + n_samples])
    return out


def evaluate(model, tok, rows, n_samples=8, ks=(1, 2, 4, 8), temperature=0.8,
             max_new_tokens=256, batch_size=32, greedy=True) -> dict:
    """Greedy accuracy + sampled pass@k, plus per-row records for error analysis."""
    res: dict = {"n": len(rows)}
    records = [dict(expression=r["expression"], answer=r["answer"],
                    n_operands=r["n_operands"]) for r in rows]
    if greedy:
        g = generate(model, tok, rows, 1, 0.0, max_new_tokens, batch_size)
        for rec, (txt,) in zip(records, g):
            pred = extract_answer(txt)
            rec.update(greedy_text=txt, greedy_pred=pred, greedy_correct=pred == rec["answer"],
                       greedy_correct_lenient=extract_answer_lenient(txt) == rec["answer"],
                       greedy_format=bool(_FORMAT_RE.search(txt)), greedy_len=len(txt),
                       style=output_style(txt))
        res["greedy_acc"] = float(np.mean([r["greedy_correct"] for r in records]))
        res["lenient_acc"] = float(np.mean([r["greedy_correct_lenient"] for r in records]))
        res["format_rate"] = float(np.mean([r["greedy_format"] for r in records]))
        res["answer_tag_rate"] = float(np.mean([r["style"] != "no_answer_tag" for r in records]))
        res["mean_chars"] = float(np.mean([r["greedy_len"] for r in records]))
    if n_samples:
        s = generate(model, tok, rows, n_samples, temperature, max_new_tokens,
                     max(1, batch_size // n_samples))
        for rec, samples in zip(records, s):
            rec["n_correct"] = sum(extract_answer(t) == rec["answer"] for t in samples)
            rec["n_correct_lenient"] = sum(extract_answer_lenient(t) == rec["answer"] for t in samples)
        for k in ks:
            if k <= n_samples:
                res[f"pass@{k}"] = float(np.mean(
                    [pass_at_k(n_samples, r["n_correct"], k) for r in records]))
        # Format-independent capability: the fairer measure for "teach vs elicit".
        for k in ks:
            if k <= n_samples:
                res[f"lenient_pass@{k}"] = float(np.mean(
                    [pass_at_k(n_samples, r["n_correct_lenient"], k) for r in records]))
    res["records"] = records
    return res


def summarize(results: dict[str, dict]) -> str:
    """results: {split_name: evaluate(...) output} -> markdown table."""
    keys = [k for k in next(iter(results.values())) if k not in ("records", "n")]
    lines = ["| split | " + " | ".join(keys) + " |",
             "|---" * (len(keys) + 1) + "|"]
    for split, r in results.items():
        lines.append(f"| {split} | " + " | ".join(f"{r[k]:.3f}" for k in keys) + " |")
    return "\n".join(lines)


def save_results(path: str, results: dict) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=1, default=str)
