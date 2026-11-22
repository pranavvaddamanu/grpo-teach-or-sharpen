"""Regenerates the README figures from results/summary.json and results/train_logs_experiments.json."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
S = json.load(open(ROOT / "results/summary.json"))
FIG = ROOT / "results/figures"

MODELS = ["base", "default", "hard_train", "kl", "format_only"]
LABELS = {"base": "base", "default": "default (2–3 operands)", "hard_train": "hard_train (3–4 operands)",
          "kl": "kl (β=0.04)", "format_only": "format_only"}
COLORS = {"base": "#9a9a94", "default": "#2a78d6", "hard_train": "#eb6834",
          "kl": "#1baf7a", "format_only": "#eda100"}
SPLITS = {"test_id": "In-distribution", "test_ood_length": "OOD: longer\nexpressions",
          "test_ood_digits": "OOD: 3-digit\nnumbers"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#999", "xtick.color": "#444", "ytick.color": "#555"})

# Figure 1: greedy accuracy and pass@8 by experiment
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
w = 0.16
for ax, (col, title) in zip(axes, [(0, "Greedy accuracy"), (2, "pass@8 (solved within 8 tries)")]):
    x = np.arange(len(SPLITS))
    for i, m in enumerate(MODELS):
        v = [S["models"][m][s][col] for s in SPLITS]
        pos = x + (i - 2) * (w + 0.012)
        ax.bar(pos, v, width=w, color=COLORS[m], label=LABELS[m], edgecolor="white", linewidth=1.5)
        for p, val in zip(pos, v):
            ax.text(p, val + 0.015, f"{val:.2f}", ha="center", va="bottom", fontsize=7, color="#333")
    ax.set_xticks(x); ax.set_xticklabels(SPLITS.values())
    ax.set_title(title, loc="left", fontsize=11, color="#222"); ax.set_ylim(0, 1.08)
    ax.grid(axis="y", color="#e6e6e6", lw=.8); ax.set_axisbelow(True)
axes[0].set_ylabel("accuracy (math only)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=5)
fig.suptitle("Training on harder problems improves every split; training on easy problems trades pass@8 for consistency",
             x=0.01, ha="left", fontsize=12, color="#222")
plt.tight_layout(rect=(0, 0.07, 1, 1)); plt.savefig(FIG / "experiments_accuracy.png", dpi=150); plt.close()

# Figure 2: response length on longer expressions
fig, ax = plt.subplots(figsize=(7.5, 3.4))
v = [S["models"][m]["test_ood_length"][3] for m in MODELS]
ax.barh([LABELS[m] for m in MODELS][::-1], v[::-1], color=[COLORS[m] for m in MODELS][::-1],
        edgecolor="white", height=0.6)
for i, val in enumerate(v[::-1]):
    ax.text(val + 5, i, f"{val:.0f}", va="center", fontsize=9, color="#333")
ax.set_xlabel("mean response length on longer expressions (characters)")
fig.suptitle("RL on easy problems compressed reasoning;\nRL on harder problems lengthened it", x=0.01, ha="left", fontsize=11)
ax.grid(axis="x", color="#e6e6e6", lw=.8); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig(FIG / "response_length.png", dpi=150); plt.close()

# Figure 3: training dynamics
L = json.load(open(ROOT / "results/train_logs_experiments.json"))
keys = [("completions/mean_length", "response length (tokens)"), ("entropy", "entropy"),
        ("frac_reward_zero_std", "groups with no learning signal")]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
for name, logs in L.items():
    logs = [l for l in logs if "reward" in l]
    for ax, (k, t) in zip(axes, keys):
        ax.plot([l["step"] for l in logs], [l[k] for l in logs], lw=2, color=COLORS[name], label=LABELS[name])
for ax, (k, t) in zip(axes, keys):
    ax.set_title(t, loc="left", fontsize=10.5); ax.set_xlabel("training step")
    ax.grid(axis="y", color="#e6e6e6", lw=.8)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=3)
plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig(FIG / "training_dynamics.png", dpi=150); plt.close()
print("figures written to", FIG)
