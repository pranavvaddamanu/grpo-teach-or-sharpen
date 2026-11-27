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

# Figure 4: seeds. One dot per training seed, bar = mean; dashed line = base model.
if "seeds" in S:
    R = S["seeds"]["runs"]
    panels = [("test_id", "In-distribution"), ("test_ood_length", "OOD: longer expressions"),
              ("test_ood_digits", "OOD: 3-digit numbers"), ("five_op", "OOD: 5-operand subset")]
    groups = [("default", "default\n(2–3 operands)", COLORS["default"]),
              ("hard_train", "hard_train\n(3–4 operands)", COLORS["hard_train"])]
    base = {"test_id": .860, "test_ood_length": .687, "test_ood_digits": .473, "five_op": .654}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    for ax, (key, title) in zip(axes, panels):
        for i, (g, lab, c) in enumerate(groups):
            vals = [R[f"{g}_s{s}"][key] if key == "five_op" else R[f"{g}_s{s}"][key][0] for s in range(3)]
            ax.bar(i, np.mean(vals), width=0.55, color=c, alpha=0.25, edgecolor="none")
            ax.scatter([i - 0.1, i, i + 0.1], vals, color=c, s=46, zorder=3, edgecolor="white", linewidth=1.2)
            ax.text(i, max(vals) + 0.035, f"{np.mean(vals):.2f} ± {np.std(vals, ddof=1):.2f}",
                    ha="center", fontsize=8.5, color="#333")
        ax.axhline(base[key], color="#888", ls="--", lw=1.2)
        ax.text(-0.57, base[key] + 0.012, f"base {base[key]:.2f}", ha="left", fontsize=8, color="#666")
        ax.set_xticks([0, 1]); ax.set_xticklabels([g[1] for g in groups], fontsize=8.5)
        ax.set_title(title, loc="left", fontsize=10.5); ax.set_xlim(-0.6, 1.6); ax.set_ylim(0.3, 1.05)
        ax.grid(axis="y", color="#e6e6e6", lw=.8); ax.set_axisbelow(True)
    axes[0].set_ylabel("greedy accuracy (math only)")
    fig.suptitle("Across 3 seeds, training on harder problems reliably helps on longer expressions; other differences are within seed noise",
                 x=0.01, ha="left", fontsize=11.5)
    plt.tight_layout(); plt.savefig(FIG / "seeds.png", dpi=150); plt.close()
    print("seeds figure written")

# Figure 5: reasoning length vs accuracy on longer expressions, one point per trained run.
if "seeds" in S:
    fig, ax = plt.subplots(figsize=(6.4, 4))
    xs, ys = [], []
    for name, r in S["seeds"]["runs"].items():
        g = "hard_train" if name.startswith("hard") else "default"
        x, y = r["test_ood_length"][3], r["test_ood_length"][0]
        xs.append(x); ys.append(y)
        learned = r["format_rate"] > 0.5
        ax.scatter(x, y, s=80, color=COLORS[g] if learned else "white", edgecolor=COLORS[g], linewidth=2, zorder=3)
        ax.annotate(f"s{name[-1]}", (x, y), xytext=(7, -3), textcoords="offset points", fontsize=8, color="#555")
    ax.scatter(343, .687, s=80, color=COLORS["base"], zorder=3)
    ax.annotate("base", (343, .687), xytext=(7, -3), textcoords="offset points", fontsize=8, color="#555")
    r = np.corrcoef(xs, ys)[0, 1]
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=COLORS["default"], ms=8, label="default"),
         Line2D([], [], marker="o", ls="", color=COLORS["hard_train"], ms=8, label="hard_train"),
         Line2D([], [], marker="o", ls="", color="white", mec="#555", mew=1.5, ms=8, label="hollow: format not learned")]
    ax.legend(handles=h, frameon=False, fontsize=8.5, loc="lower right")
    ax.set_xlabel("mean response length on longer expressions (characters)")
    ax.set_ylabel("greedy accuracy, longer expressions")
    ax.set_title(f"Longer reasoning, higher accuracy (6 trained runs, r = {r:.2f})", loc="left", fontsize=10.5)
    ax.grid(color="#eee", lw=.8); ax.set_axisbelow(True)
    plt.tight_layout(); plt.savefig(FIG / "length_vs_accuracy.png", dpi=150); plt.close()
    print("length figure written")
