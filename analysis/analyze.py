"""Paired analysis of a trained model against the base model.

Usage:
    python analysis/analyze.py --base runs/.../base_eval.json --model runs/.../grpo_eval.json \
        [--label GRPO] [--plot pass_at_k.png]

Both files are outputs of rlmath.evaluate() saved by the notebook. All metrics are
lenient (math only), since strict metrics mostly measure whether the tags were learned.

Reports, per test split:
  * greedy accuracy, with McNemar's exact test on problems fixed vs broken
  * coverage within n samples (solved at least once), with McNemar on new vs lost
  * pass@1 and pass@n differences with 95% paired bootstrap confidence intervals
  * share of problems whose samples disagree (a diversity measure)
  * mean response length, and accuracy by number of operands
"""
import argparse
import json
from math import comb

import numpy as np


def mcnemar(a: int, b: int) -> float:
    """Exact two-sided McNemar test (binomial on the discordant pairs)."""
    n = a + b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def paired_bootstrap(diff: np.ndarray, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
    return diff.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def analyze(base: dict, model: dict, label: str):
    for split in base:
        b, g = base[split]["records"], model[split]["records"]
        assert [x["expression"] for x in b] == [y["expression"] for y in g], "problems differ"
        bc = np.array([x["greedy_correct_lenient"] for x in b])
        gc = np.array([y["greedy_correct_lenient"] for y in g])
        fixed, broken = int((~bc & gc).sum()), int((bc & ~gc).sum())

        nb = np.array([x["n_correct_lenient"] for x in b])
        ng = np.array([y["n_correct_lenient"] for y in g])
        # samples per problem = largest k reported for lenient pass@k
        n = max(int(k.split("@")[1]) for k in base[split] if k.startswith("lenient_pass@"))
        new, lost = int(((nb == 0) & (ng > 0)).sum()), int(((nb > 0) & (ng == 0)).sum())

        print(f"\n== {split}  ({label} vs base, {len(b)} problems)")
        print(f"  greedy accuracy   {bc.mean():.3f} -> {gc.mean():.3f}   "
              f"fixed={fixed} broken={broken}  McNemar p={mcnemar(fixed, broken):.3g}")
        print(f"  solved within {n}  {np.mean(nb > 0):.3f} -> {np.mean(ng > 0):.3f}   "
              f"new={new} lost={lost}  McNemar p={mcnemar(new, lost):.3g}")
        for name, f in [("pass@1", lambda c: c / n), (f"pass@{n}", lambda c: (c > 0).astype(float))]:
            m, lo, hi = paired_bootstrap(f(ng) - f(nb))
            print(f"  delta {name:8s} {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
        mixed = lambda c: np.mean((c > 0) & (c < n))
        print(f"  samples disagree  {mixed(nb):.2f} -> {mixed(ng):.2f}   (share of problems)")
        print(f"  mean chars        {np.mean([x['greedy_len'] for x in b]):.0f} -> "
              f"{np.mean([y['greedy_len'] for y in g]):.0f}")
        by = {}
        for x, y in zip(b, g):
            by.setdefault(x["n_operands"], ([], []))
            by[x["n_operands"]][0].append(x["greedy_correct_lenient"])
            by[x["n_operands"]][1].append(y["greedy_correct_lenient"])
        print("  by operands       " + ", ".join(
            f"{k}: {np.mean(v[0]):.2f}->{np.mean(v[1]):.2f}" for k, v in sorted(by.items())))


def plot_pass_at_k(base: dict, model: dict, label: str, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [1, 2, 4, 8]
    colors = {"base": "#2a78d6", label: "#eb6834"}
    fig, axes = plt.subplots(1, len(base), figsize=(4 * len(base), 3.4), sharey=True)
    for ax, split in zip(np.atleast_1d(axes), base):
        for res, lab in [(base, "base"), (model, label)]:
            v = [res[split][f"lenient_pass@{k}"] for k in ks]
            ax.plot(ks, v, marker="o", ms=7, lw=2, color=colors[lab], mec="white", mew=1.5, label=lab)
        ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels(ks)
        ax.set_title(split, loc="left"); ax.set_xlabel("k"); ax.grid(axis="y", alpha=.3)
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("pass@k (math only)")
    np.atleast_1d(axes)[0].legend(frameon=False)
    plt.tight_layout(); plt.savefig(path, dpi=150)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="GRPO")
    ap.add_argument("--plot")
    a = ap.parse_args()
    base, model = json.load(open(a.base)), json.load(open(a.model))
    analyze(base, model, a.label)
    if a.plot:
        plot_pass_at_k(base, model, a.label, a.plot)
