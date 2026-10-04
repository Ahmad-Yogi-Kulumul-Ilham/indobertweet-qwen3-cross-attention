"""Gambar untuk naskah (PNG 300 dpi) -> paper/figures/ (atau paper/figures_en/ dengan --en).

    python src/figures.py         # label bahasa Indonesia, desimal koma
    python src/figures.py --en    # label bahasa Inggris, desimal titik (naskah JMASIF)
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

LANG = "en" if "--en" in sys.argv else "id"
EN = {
    "Depresi (tweet)": "Depression (tweets)", "Distorsi kognitif": "Cognitive distortion",
    "TF-IDF + LogReg": "TF-IDF + LogReg", "Ensemble rata-rata": "Probability-averaging ensemble",
    "Fusion v1 (gate)": "Fusion v1 (gated)", "Fusion v2 (usulan)": "Fusion v2 (proposed)",
    "Depresi": "Depression", "Tidak depresi": "Not depressed", "Distorsi": "Distorted", "Tidak": "Not distorted",
    "Prediksi": "Predicted", "Label sebenarnya": "True label",
    "Teks\n(tweet /\nkalimat)": "Text\n(tweet /\nsentence)", "IndoBERTweet\n(fine-tune penuh)\n": "IndoBERTweet\n(full fine-tuning)\n",
    "Proyeksi\n512": "Projection\n512", "Cross-attention\ndua arah\n(A→B, B→A)\n+ residual, LN":
        "Bidirectional\ncross-attention\n(A→B, B→A)\n+ residual, LN",
    "Klasifier\n": "Classifier\n", "Klasifier aux\n": "Auxiliary classifier\n",
    "branch dropout\n(p = 0,15)": "branch dropout\n(p = 0.15)",
    "gambar1_arsitektur.png": "fig1_architecture.png", "gambar2_perbandingan_model.png": "fig2_model_comparison.png",
    "gambar3_confusion_matrix.png": "fig3_confusion_matrix.png",
}


def tr(s):
    return EN.get(s, s) if LANG == "en" else s


def dec(s):
    """Desimal koma untuk naskah Indonesia, titik untuk naskah Inggris."""
    return s if LANG == "en" else s.replace(".", ",")


OUT = Path("paper/figures_en" if LANG == "en" else "paper/figures")
TASKS = {"depresi": tr("Depresi (tweet)"), "distortion": tr("Distorsi kognitif")}
MODELS = [  # (run, label)
    ("tfidf_logreg", tr("TF-IDF + LogReg")),
    ("qwen3_lora", "Qwen3-1.7B (QLoRA)"),
    ("indobertweet", "IndoBERTweet"),
    ("ensemble_avg", tr("Ensemble rata-rata")),
    ("fusion_cross_v1", tr("Fusion v1 (gate)")),
    ("fusion_cross_v2", tr("Fusion v2 (usulan)")),
]
LABEL_ID = {"depresi": [tr("Depresi"), tr("Tidak depresi")], "distortion": [tr("Distorsi"), tr("Tidak")]}

BLUE, GRAY, INK, MUTED, GRID = "#2a78d6", "#b9b8b2", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
})


def fold_results(task, run):
    return [json.load(open(f, encoding="utf-8")) for f in sorted(Path("results", task, run).glob("fold*.json"))]


def fig_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, (task, title) in zip(axes, TASKS.items()):
        means, stds = [], []
        for run, _ in MODELS:
            v = np.array([r["test_metrics"]["macro_f1"] for r in fold_results(task, run)]) * 100
            means.append(v.mean())
            stds.append(v.std(ddof=1))
        y = np.arange(len(MODELS))
        colors = [BLUE if run == "fusion_cross_v2" else GRAY for run, _ in MODELS]
        ax.barh(y, means, xerr=stds, color=colors, height=0.6,
                error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2})
        for yi, m, s in zip(y, means, stds):
            ax.text(m + s + 0.4, yi, dec(f"{m:.2f}"), va="center", fontsize=8, color=INK)
        ax.set_xlim(66, 82)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: dec(f"{v:g}")))
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Macro-F1 (%)")
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_yticks(np.arange(len(MODELS)), [lab for _, lab in MODELS])
    fig.tight_layout()
    fig.savefig(OUT / tr("gambar2_perbandingan_model.png"), dpi=300)
    plt.close(fig)


def fig_confusion():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))
    for ax, (task, title) in zip(axes, TASKS.items()):
        labels = json.load(open(Path("data", task, "labels.json"), encoding="utf-8"))["labels"]
        cm = sum(np.array(r["test_metrics"]["confusion_matrix"]) for r in fold_results(task, "fusion_cross_v2"))
        pct = cm / cm.sum(axis=1, keepdims=True) * 100
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        names = [LABEL_ID[task][0] if "tidak" not in l else LABEL_ID[task][1] for l in labels]
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, dec(f"{pct[i, j]:.1f}") + f"%\n(n={cm[i, j]})", ha="center", va="center", fontsize=8,
                        color="white" if pct[i, j] > 55 else INK)
        ax.set_xticks(range(len(names)), names)
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel(tr("Prediksi"))
        ax.set_ylabel(tr("Label sebenarnya"))
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / tr("gambar3_confusion_matrix.png"), dpi=300)
    plt.close(fig)


def box(ax, x, y, w, h, text, fill="#f3f2ee", edge=MUTED, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=fill, edgecolor=edge, linewidth=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, color=INK,
            fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                 color=MUTED, linewidth=1))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.1, 2.0, 1.3, 1.0, tr("Teks\n(tweet /\nkalimat)"))
    box(ax, 2.0, 3.2, 2.2, 1.1, tr("IndoBERTweet\n(fine-tune penuh)\n") + r"$H \in \mathbb{R}^{L\times768}$")
    box(ax, 2.0, 0.7, 2.2, 1.1, "Qwen3-1.7B\n(4-bit + LoRA)\n" r"$H \in \mathbb{R}^{L\times2048}$")
    box(ax, 4.7, 3.35, 1.4, 0.8, tr("Proyeksi\n512"))
    box(ax, 4.7, 0.85, 1.4, 0.8, tr("Proyeksi\n512"))
    box(ax, 6.6, 1.6, 1.9, 1.8, tr("Cross-attention\ndua arah\n(A→B, B→A)\n+ residual, LN"), fill="#e3eefb",
        edge=BLUE)
    box(ax, 9.0, 2.05, 1.3, 0.9, "Mean pool\n+ concat", fill="#e3eefb", edge=BLUE)
    box(ax, 10.7, 2.05, 1.2, 0.9, tr("Klasifier\n") + r"($\mathcal{L}_{main}$)", bold=True)
    box(ax, 6.5, 4.05, 2.1, 0.75, tr("Klasifier aux\n") + r"($\mathcal{L}_{aux}$)", fill="#ffffff")
    box(ax, 6.5, 0.05, 2.1, 0.75, tr("Klasifier aux\n") + r"($\mathcal{L}_{aux}$)", fill="#ffffff")
    arrow(ax, 1.4, 2.7, 2.0, 3.6)
    arrow(ax, 1.4, 2.3, 2.0, 1.3)
    arrow(ax, 4.2, 3.75, 4.7, 3.75)
    arrow(ax, 4.2, 1.25, 4.7, 1.25)
    arrow(ax, 6.1, 3.6, 6.6, 3.0)
    arrow(ax, 6.1, 1.4, 6.6, 2.0)
    arrow(ax, 6.1, 4.0, 6.6, 4.4)
    arrow(ax, 6.1, 1.0, 6.6, 0.45)
    arrow(ax, 8.5, 2.5, 9.0, 2.5)
    arrow(ax, 10.3, 2.5, 10.7, 2.5)
    ax.text(5.4, 2.5, tr("branch dropout\n(p = 0,15)"), ha="center", va="center", fontsize=7, color=MUTED,
            style="italic")
    weight = "0.3" if LANG == "en" else "0{,}3"
    ax.text(10.4, 3.7, r"$\mathcal{L} = \mathcal{L}_{main} + " + weight + r"\cdot\overline{\mathcal{L}}_{aux}$",
            ha="center", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / tr("gambar1_arsitektur.png"), dpi=300)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_architecture()
    fig_comparison()
    fig_confusion()
    print("Gambar tersimpan di", OUT)


if __name__ == "__main__":
    main()
