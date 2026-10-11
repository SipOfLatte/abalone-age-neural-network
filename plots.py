"""
Figures for the report.
Run model.py first to generate the results in the results/ folder, then run this to produce the
figures in the figures/ folder. The figures are saved as PDF (for LaTeX) and PNG (for quick preview).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")                     # draw to files only, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import model as m

RESULTS, FIGURES = "results", "figures"

# The four series colours are always used in this order, so a given
# setting keeps the same colour from one figure to the next.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = [BLUE, ORANGE, AQUA, YELLOW]
STYLES = ["-", "--", "-.", ":"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"   # text, axis text, gridlines
# Light-to-dark blue for the confusion matrix, and blue-grey-red for the
# correlation matrix (negative, zero, positive).
SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"])
DIV = LinearSegmentedColormap.from_list(
    "div", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a72a2a"])

SHORT = ["C1 (≤7)", "C2 (7–10]", "C3 (10–15]", "C4 (>15)"]   # class labels on axes
SINGLE, DOUBLE = 3.5, 7.16      # width of one IEEE column and of the full page 

# Times-like serif text at the sizes used in the IEEE template, so the figures
# match the body text of the report.
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.frameon": False,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "grid.linestyle": "-", "axes.axisbelow": True,
    "lines.linewidth": 1.4, "lines.solid_capstyle": "round",
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

FEATURE_LABELS = {"Length": "Length", "Diameter": "Diameter", "Height": "Height",
                  "Whole_weight": "Whole weight", "Shucked_weight": "Shucked weight",
                  "Viscera_weight": "Viscera weight", "Shell_weight": "Shell weight"}


def save(fig, name):
    #Save a figure as PDF then close it."""
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES, f"{name}.{ext}"))
    plt.close(fig)
    print("saved", name)


def load_history(name):
    #Per-epoch history saved by model.py
    return np.load(os.path.join(RESULTS, f"history_{name}.npy"))


def band(ax, hist, col, color, label, ls="-"):
    """
    Plot the mean learning curve over the runs, with a shaded 95% CI band. 
    col picks the quantity and the band is only drawn when there is more than one run to average.
    """
    epochs = np.arange(1, hist.shape[1] + 1)
    mean = hist[:, :, col].mean(0)
    ax.plot(epochs, mean, color=color, ls=ls, label=label)
    if hist.shape[0] > 1:
        ci = np.array([m.mean_ci(hist[:, e, col])[1] for e in range(hist.shape[1])])
        ax.fill_between(epochs, mean - ci, mean + ci, color=color, alpha=0.12,
                        lw=0)


def ci_plot(ax, x, rows, logx=False):
    #Train and test accuracy
    for i, (split, label) in enumerate([("train", "Train"), ("test", "Test")]):
        ax.errorbar(x, rows[f"{split}_acc_mean"], yerr=rows[f"{split}_acc_ci"],
                    color=SERIES[i], ls=STYLES[i], marker="os"[i], ms=4,
                    mec="white", mew=0.8, capsize=2.5, elinewidth=0.9,
                    label=label)
    if logx:
        ax.set_xscale("log")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="lower right")


def class_boxplot(ax, df, col):
    # Box plot of one feature for each of the four age classes
    groups = [df.loc[df["Class"] == c, col] for c in range(4)]
    bp = ax.boxplot(groups, widths=0.5, patch_artist=True,
                    medianprops={"color": INK, "lw": 1},
                    whiskerprops={"color": INK2, "lw": 0.7},
                    capprops={"color": INK2, "lw": 0.7},
                    flierprops={"marker": "o", "ms": 1.5, "mfc": INK2,
                                "mec": "none", "alpha": 0.5})
    for patch in bp["boxes"]:
        patch.set(facecolor="#cde2fb", edgecolor=BLUE, lw=0.8)
    ax.set_xticks(range(1, 5), ["C1", "C2", "C3", "C4"])
    ax.grid(axis="x", visible=False)



# Data analysis

def eda_figures(df):
    """Class distribution, feature distributions, correlations and the
    overlap between classes (the Step 1 visualisations)."""

    # Figure: age histogram with the three class cut-offs, next to the number
    # of abalone in each class. Ages are x.5 values, so bins of width 1 centred
    # on them put the cut-offs at 7, 10 and 15 exactly on bin edges.
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.2),
                               gridspec_kw={"width_ratios": [1.6, 1]})
    bins = np.arange(df["Age"].min() - 0.5, df["Age"].max() + 1.5, 1)
    a.hist(df["Age"], bins=bins, color=BLUE, edgecolor="white", lw=0.6)
    for cut in (7, 10, 15):
        a.axvline(cut, color=INK2, lw=0.8, ls="--")
    for x, lab in zip((5, 8.5, 12.5, 23), ["C1", "C2", "C3", "C4"]):
        a.text(x, a.get_ylim()[1] * 0.93, lab, ha="center", color=INK2,
               fontsize=7.5)
    a.set_xlabel("Age (years) = rings + 1.5")
    a.set_ylabel("Number of abalone")
    a.set_title("(a) Distribution of age with class boundaries")

    counts = df["Class"].value_counts().sort_index()
    bars = b.bar(range(4), counts.values, width=0.55, color=BLUE)
    for rect, n in zip(bars, counts.values):          # count and share above each bar
        b.text(rect.get_x() + rect.get_width() / 2, n + 25,
               f"{n}\n({n / len(df):.1%})", ha="center", va="bottom",
               fontsize=6.5, color=INK)
    b.set_xticks(range(4), SHORT)
    b.set_ylim(0, counts.max() * 1.22)
    b.set_ylabel("Number of abalone")
    b.set_title("(b) Class distribution")
    b.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_class_distribution")

    # Figure: a histogram for each of the seven measurements, plus the count
    # of each sex, to show skewness and range before standardising.
    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE, 3.1))
    for ax, col in zip(axes.flat, m.NUMERIC):
        ax.hist(df[col], bins=30, color=BLUE, edgecolor="white", lw=0.4)
        ax.set_title(FEATURE_LABELS[col])
        ax.tick_params(labelsize=6)
    sex = df["Sex"].value_counts().reindex(["M", "F", "I"])
    ax = axes.flat[-1]
    ax.bar(["Male", "Female", "Infant"], sex.values, width=0.55, color=BLUE)
    for i, n in enumerate(sex.values):
        ax.text(i, n + 20, str(n), ha="center", fontsize=6.5)
    ax.set_ylim(0, sex.max() * 1.18)
    ax.set_title("Sex")
    ax.grid(axis="x", visible=False)
    ax.tick_params(labelsize=6)
    for ax in axes[:, 0]:
        ax.set_ylabel("Count")
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    save(fig, "fig_feature_distributions")

    # Figure: (a) correlation matrix of the measurements and age, and
    # (b)-(e) box plots of four measurements by class.
  
    fig = plt.figure(figsize=(DOUBLE, 3.2))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.45, 1, 1], wspace=0.42, hspace=0.55)

    ax = fig.add_subplot(gs[:, 0])
    corr = df[m.NUMERIC + ["Age"]].corr()
    names = [FEATURE_LABELS.get(c, c).replace(" weight", " wt.") for c in corr.columns]
    im = ax.imshow(corr.values, cmap=DIV, vmin=-1, vmax=1)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.2,
                    color="white" if abs(v) > 0.75 else INK)   # keep text readable on dark cells
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(names)), names, fontsize=6.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    ax.set_title("(a) Pearson correlation")

    box_features = ["Shell_weight", "Diameter", "Whole_weight", "Height"]
    for k, col in enumerate(box_features):
        ax = fig.add_subplot(gs[k // 2, 1 + k % 2])
        class_boxplot(ax, df, col)
        ax.set_title(f"({'bcde'[k]}) {FEATURE_LABELS[col]}")
        ax.tick_params(labelsize=6)     # units (original / 200) are given in the caption
    save(fig, "fig_corr_boxes")


# ---------------------------------------------------------------------------
# Convergence trial and Steps 2 to 6
# ---------------------------------------------------------------------------
def convergence_figure():
    """Train and test loss over 1000 epochs for the three trial settings, used
    to choose the 300-epoch budget. Skipped if model.py was run without
    --convergence."""
    trials = [("trial_sgd_1x10_lr0.01", "SGD, $\\eta$ = 0.01"),
              ("trial_sgd_1x10_lr0.1", "SGD, $\\eta$ = 0.1"),
              ("trial_adam_1x10_lr0.001", "Adam, $\\eta$ = 0.001")]
    if not os.path.exists(os.path.join(RESULTS, f"history_{trials[0][0]}.npy")):
        return
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.0), sharey=True)
    for ax, (name, title) in zip(axes, trials):
        h = load_history(name)
        band(ax, h, 0, BLUE, "Train loss")
        band(ax, h, 1, ORANGE, "Test loss", ls="--")
        # Mark the chosen budget and the epoch where the mean test loss is lowest.
        best = h[:, :, 1].mean(0).argmin() + 1
        ax.axvline(m.EPOCHS, color=INK2, lw=0.8)
        ax.text(m.EPOCHS + 15, 0.86, f"budget\n{m.EPOCHS}", fontsize=6,
                color=INK2, va="top")
        ax.plot(best, h[:, best - 1, 1].mean(), "o", color=ORANGE, ms=4,
                mec="white", mew=0.8)
        ax.set_title(f"{title} (min test loss: epoch {best})")
        ax.set_xlabel("Epoch")
        ax.set_ylim(0.55, 0.9)
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend(loc="upper right")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_convergence")


def step_figures(summary):
    #Figures for Steps 2 to 5, built from summary.csv and the histories
    s2 = summary[summary.step == "step2"]
    s3 = summary[summary.step == "step3"]

    # Figure: (a) accuracy against the number of hidden neurons (Step 2),
    # (b) accuracy against the learning rate (Step 3), and (c) how test accuracy develops during training for four of those learning rates
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(DOUBLE, 2.15))
    neurons = [int(h.strip("(),")) for h in s2.hidden]
    ci_plot(a, neurons, s2)
    a.set_xticks(neurons)
    a.set_xlabel("Neurons in the hidden layer")
    a.set_title(f"(a) Hidden neurons (SGD, $\\eta$ = {s2.lr.iloc[0]})")

    ci_plot(b, s3.lr.values, s3, logx=True)
    b.legend(loc="lower center")
    b.set_xticks(s3.lr.values, [f"{v:g}" for v in s3.lr.values], fontsize=6)
    b.minorticks_off()
    b.set_xlabel("Learning rate $\\eta$ (log scale)")
    h_opt = s3.hidden.iloc[0].strip("(),")
    b.set_title(f"(b) Learning rate (SGD, {h_opt} neurons)")

    shown = [v for v in s3.lr.values if v in (0.001, 0.01, 0.1, 0.2)]
    for i, lr in enumerate(shown):
        name = s3.loc[s3.lr == lr, "config"].iloc[0]
        band(c, load_history(name), 3, SERIES[i], f"$\\eta$ = {lr:g}", ls=STYLES[i])
    c.set_xlabel("Epoch")
    c.set_ylabel("Test accuracy")
    c.set_ylim(0.5, 0.74)
    c.legend(loc="lower right", ncol=2, columnspacing=0.8, handlelength=1.6)
    c.set_title("(c) Test accuracy during training")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_step2_step3")

    # Figure: (a) Step 4, learning curves with one and two hidden layers, and
    # (b) Step 5, train and test loss for SGD and Adam.
    s4 = summary[summary.step == "step4"]
    s5 = summary[summary.step == "step5"]
    one = s3.loc[s3.test_acc_mean.idxmax(), "config"]   # best one-layer model from Step 3
    two = s4.config.iloc[0]
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.2))
    for i, (name, lab) in enumerate([(one, "One hidden layer"),
                                     (two, "Two hidden layers")]):
        h = load_history(name)
        band(a, h, 2, SERIES[i], f"{lab} (train)", ls="-")
        band(a, h, 3, SERIES[i], f"{lab} (test)", ls="--")
    a.set_xlabel("Epoch")
    a.set_ylabel("Accuracy")
    a.set_ylim(0.6, 0.78)
    a.legend(loc="lower right")
    a.set_title("(a) Step 4: one vs two hidden layers (SGD)")

    # Colour shows the optimiser and line style shows train or test, so the legend lists those two things separately instead of all six curves.
    handles = []
    for i, name in enumerate(s5.config):
        solver, _, lr = name.split("_")             # e.g. "adam_1x20_lr0.001"
        lab = f"{'SGD' if solver == 'sgd' else 'Adam'}, $\\eta$ = {lr[2:]}"
        h = load_history(name)
        band(b, h, 0, SERIES[i], None, ls="-")
        band(b, h, 1, SERIES[i], None, ls="--")
        handles.append(Line2D([], [], color=SERIES[i], label=lab))
    handles += [Line2D([], [], color=INK2, ls="-", label="Train"),
                Line2D([], [], color=INK2, ls="--", label="Test")]
    b.set_xlabel("Epoch")
    b.set_ylabel("Cross-entropy loss")
    b.set_ylim(0.55, 0.95)
    b.legend(handles=handles, loc="upper right", ncol=2)
    b.set_title("(b) Step 5: SGD vs Adam")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_step4_step5")


def final_figures():
    """Step 6: confusion matrix, per-class ROC curves and learning curves of
    the final model, from final_model.json and its training history."""
    with open(os.path.join(RESULTS, "final_model.json")) as f:
        res = json.load(f)

    """ # Confusion matrix. Colour shows the share of each true class (row), so
    # the small classes are as easy to read as the large ones; each cell also
    # prints the raw count."""
    cm = np.array(res["confusion_matrix"])
    pct = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(SINGLE, 2.7))
    im = ax.imshow(pct, cmap=SEQ, vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.1%}", ha="center",
                    va="center", fontsize=7,
                    color="white" if pct[i, j] > 0.55 else INK)
    ax.set_xticks(range(4), SHORT)
    ax.set_yticks(range(4), SHORT)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                      format=matplotlib.ticker.PercentFormatter(1.0))
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    ax.set_title("(a) Confusion matrix on the test set (row %)")
    save(fig, "fig_confusion_matrix")

    # One ROC curve per class (that class against the other three). The dotted diagonal is what a random classifier would give.
    fig, ax = plt.subplots(figsize=(SINGLE, 2.8))
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=":")
    for c in range(4):
        r = res["roc"][str(c)]
        ax.plot(r["fpr"], r["tpr"], color=SERIES[c], ls=STYLES[c],
                label=f"{SHORT[c]}: AUC = {r['auc']:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect("equal")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    ax.set_title("(b) One-vs-rest ROC curves (test set)")
    save(fig, "fig_roc")

    # Learning curves of the final model, to check it is not overfitting.
    hidden = res["hidden"]
    name = (f"final_{res['solver']}_{len(hidden)}x{hidden[0]}_lr{res['lr']}")
    h = load_history(name)
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.0))
    band(a, h, 0, BLUE, "Train")
    band(a, h, 1, ORANGE, "Test", ls="--")
    a.set_ylabel("Cross-entropy loss")
    a.set_ylim(0.55, 0.9)
    band(b, h, 2, BLUE, "Train")
    band(b, h, 3, ORANGE, "Test", ls="--")
    b.set_ylabel("Accuracy")
    b.set_ylim(0.6, 0.78)
    for ax, t in ((a, "(a) Loss"), (b, "(b) Accuracy")):
        ax.set_xlabel("Epoch")
        ax.set_title(t)
        ax.legend(loc="best")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_final_curves")


def main():
    os.makedirs(FIGURES, exist_ok=True)
    # The exploratory plots use the same cleaning and class labels as model.py.
    df, _ = m.clean_data(m.load_data("data/abalone.data"))
    df = m.add_age_class(df)
    eda_figures(df)
    convergence_figure()
    summary = pd.read_csv(os.path.join(RESULTS, "summary.csv"))
    step_figures(summary)
    final_figures()


if __name__ == "__main__":
    main()
