"""
EDA/plots/p5_body_shape_eda.py
================================
Body Shape EDA — two publication figures:

  Figure 5A:  PCA of shape parameters → 2D scatter (Z_shape)
  Figure 5B:  Per-coefficient histogram grid (10 subplots, one per β_j)

Usage:
    python EDA/plots/p5_body_shape_eda.py \
        --features eda_cache/*.npz --labels ... --out_dir figures/body_shape
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_style import apply_paper_style, save_fig, add_stat_box, PALETTE, DATASET_COLORS

apply_paper_style()

BETA_LABELS = [f"β_{j}" for j in range(10)]


# ─────────────────────────────────────────────────────────────────────────────
# 5A — PCA of β
# ─────────────────────────────────────────────────────────────────────────────

def plot_shape_pca(
    datasets: Dict[str, np.ndarray],   # {name: betas (N,10)}
    out_dir: str = "figures/body_shape",
):
    """
    Z_shape = PCA(β).  Scatter reveals body-cluster spread.
    Ellipses (1σ) mark each dataset's cluster.
    """
    from matplotlib.patches import Ellipse

    labels_all, mats = [], []
    for name, B in datasets.items():
        labels_all.extend([name] * len(B))
        mats.append(B)

    if not mats:
        return

    X = np.concatenate(mats, axis=0).astype(np.float32)
    X = np.nan_to_num(X)

    pca = PCA(n_components=2, random_state=42)
    Z   = pca.fit_transform(X)
    ev  = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(7, 6))

    for i, name in enumerate(datasets.keys()):
        mask = np.array(labels_all) == name
        Zm   = Z[mask]
        c    = DATASET_COLORS.get(name, PALETTE[i])
        ax.scatter(Zm[:, 0], Zm[:, 1], s=10, alpha=0.35, linewidths=0, color=c)

        # 1-σ ellipse
        if len(Zm) >= 3:
            mu  = Zm.mean(0)
            cov = np.cov(Zm.T)
            ev2, evec = np.linalg.eigh(cov)
            order = ev2.argsort()[::-1]
            ev2 = ev2[order]; evec = evec[:, order]
            angle = np.degrees(np.arctan2(*evec[:, 0][::-1]))
            w = 2 * np.sqrt(ev2[0]) * 1.0
            h = 2 * np.sqrt(ev2[1]) * 1.0
            ell = Ellipse(xy=mu, width=w, height=h, angle=angle,
                          edgecolor=c, facecolor="none",
                          linewidth=1.8, label=name, zorder=3)
            ax.add_patch(ell)
            ax.scatter(*mu, color=c, s=60, marker="x", linewidths=2, zorder=4)

    ax.set_xlabel(f"PC-1  ({ev[0]:.1f}% var)", fontsize=12)
    ax.set_ylabel(f"PC-2  ({ev[1]:.1f}% var)", fontsize=12)
    ax.set_title("Body Shape Diversity — PCA(β) with 1σ Ellipses",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Dataset", framealpha=0.9,
              ncol=2 if len(datasets) > 5 else 1)
    ax.autoscale_view()
    plt.tight_layout()
    save_fig(fig, Path(out_dir), "shape_pca")


# ─────────────────────────────────────────────────────────────────────────────
# 5B — Per-coefficient histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_shape_coefficient_histograms(
    datasets: Dict[str, np.ndarray],   # {name: betas (N,10)}
    out_dir: str = "figures/body_shape",
):
    """
    2-row × 5-col grid of KDE plots, one cell per shape coefficient β_j.
    All datasets overlaid in each cell.
    """
    fig = plt.figure(figsize=(16, 7))
    gs  = gridspec.GridSpec(2, 5, figure=fig, hspace=0.45, wspace=0.35)

    for j in range(10):
        row, col = divmod(j, 5)
        ax = fig.add_subplot(gs[row, col])

        for i, (name, B) in enumerate(datasets.items()):
            vals = B[:, j]
            vals = vals[np.isfinite(vals)]
            if len(vals) < 3:
                continue
            c = DATASET_COLORS.get(name, PALETTE[i])
            sns.kdeplot(vals, ax=ax, fill=True, alpha=0.3, color=c,
                        linewidth=1.3, label=name if j == 0 else "_nolegend_")
            ax.axvline(vals.mean(), color=c, linestyle="--", linewidth=0.9, alpha=0.8)

        ax.set_title(BETA_LABELS[j], fontsize=10, fontweight="bold")
        ax.set_xlabel("Value", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        # SMPL prior ≈ N(0,1)
        x_rng = np.linspace(-3, 3, 100)
        ax.plot(x_rng, np.exp(-0.5 * x_rng ** 2) / np.sqrt(2 * np.pi),
                "k:", linewidth=0.8, label="N(0,1) prior" if j == 0 else "_nolegend_")

    # Shared legend from first axes
    handles, lbls = fig.get_axes()[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper right", fontsize=9,
               title="Dataset", framealpha=0.9, ncol=2)
    fig.suptitle("SMPL Shape Coefficient Distributions (β₀…β₉)",
                 fontsize=14, fontweight="bold", y=1.01)
    save_fig(fig, Path(out_dir), "shape_coefficient_histograms")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--labels",   nargs="+", required=True)
    p.add_argument("--out_dir",  default="figures/body_shape")
    args = p.parse_args()

    betas = {}
    for f, lbl in zip(args.features, args.labels):
        d       = dict(np.load(f, allow_pickle=True))
        betas[lbl] = d["betas"]

    plot_shape_pca(betas, args.out_dir)
    plot_shape_coefficient_histograms(betas, args.out_dir)


if __name__ == "__main__":
    _cli()
