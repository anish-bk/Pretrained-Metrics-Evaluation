"""
EDA/plots/p8_meta_correlation.py
==================================
Meta-EDA — Complexity Correlation Matrix

Constructs a per-image feature vector:
    x_i = [pose_norm, O_i, H_bg, L_i, ||β_i||, ||f_i||, ||g_i||]

Computes Pearson correlation matrix → seaborn heatmap.

Also produces a scatter-matrix (pairplot) for visual inspection of
pairwise relationships.

Usage:
    python EDA/plots/p8_meta_correlation.py \
        --features eda_cache/viton_features.npz \
        --label VITON --out_dir figures/meta
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_style import apply_paper_style, save_fig, PALETTE, DATASET_COLORS

apply_paper_style()

FEATURE_NAMES = [
    "Pose Complexity\n(||v_i||)",
    "Occlusion\nRatio O_i",
    "BG Entropy\nH_bg",
    "Luminance\nL_i",
    "Shape Norm\n||β_i||",
    "Face Embed\n||f_i||",
    "Garment\nEmbed ||g_i||",
]

FEATURE_NAMES_SHORT = [
    "Pose", "Occlusion", "BG_Entropy",
    "Luminance", "Shape", "Face", "Garment",
]


# ─────────────────────────────────────────────────────────────────────────────
# Build feature matrix from cache
# ─────────────────────────────────────────────────────────────────────────────

def _build_feature_matrix(d: dict) -> np.ndarray:
    """
    d: loaded .npz dict.
    Returns (N, 7) float32 matrix — one row per image.
    """
    pose_norm    = np.linalg.norm(d["pose_vecs"],    axis=1)  # (N,)
    occ          = d["occ_ratios"].astype(np.float32)
    bg_ent       = d["bg_entropy"].astype(np.float32)
    lum          = d["lum_mean"].astype(np.float32)
    shape_norm   = np.linalg.norm(d["betas"],         axis=1).astype(np.float32)
    face_norm    = np.linalg.norm(d["face_embs"],     axis=1).astype(np.float32)
    garment_norm = np.linalg.norm(d["garment_embs"], axis=1).astype(np.float32)

    X = np.stack(
        [pose_norm, occ, bg_ent, lum, shape_norm, face_norm, garment_norm],
        axis=1
    )
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Main correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_matrix(
    datasets: Dict[str, np.ndarray],   # {name: feature_matrix (N,7)}
    out_dir: str = "figures/meta",
):
    """
    For each dataset: full Pearson correlation heatmap (7×7).
    If one dataset → single panel.
    If multiple    → subplot grid (max 5 per row) + pooled panel.
    """
    n = len(datasets)
    pooled_X = np.concatenate(list(datasets.values()), axis=0)

    # ── Per-dataset panels ─────────────────────────────────────────────────
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(5.5 * cols, 5 * rows + 0.5))
    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    axes = axes.flatten()

    def _corr_heatmap(X, ax, title, ds_name=""):
        df = pd.DataFrame(X, columns=FEATURE_NAMES_SHORT)
        corr = df.corr(method="pearson")
        mask = np.zeros_like(corr, dtype=bool)
        # Don't mask — show full matrix including redundant lower triangle
        # so readers can verify symmetry in the paper
        sns.heatmap(
            corr, ax=ax, vmin=-1, vmax=1, center=0,
            cmap="RdBu_r",
            annot=True, fmt=".2f", annot_kws={"size": 8},
            linewidths=0.4, linecolor="#dddddd",
            square=True, cbar_kws={"shrink": 0.8},
            xticklabels=FEATURE_NAMES_SHORT,
            yticklabels=FEATURE_NAMES_SHORT,
        )
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)

    for i, (name, X) in enumerate(datasets.items()):
        X_clean = np.nan_to_num(X)
        _corr_heatmap(X_clean, axes[i], name)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Per-Image Complexity Feature Correlation Matrix\n"
                 "Corr(x_i) — Pearson",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(fig, Path(out_dir), "correlation_matrix_per_dataset")

    # ── Pooled (all datasets) ──────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 6.5))
    _corr_heatmap(np.nan_to_num(pooled_X), ax2, "All Datasets Pooled")
    fig2.suptitle("Pooled Complexity Correlation Matrix",
                  fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(fig2, Path(out_dir), "correlation_matrix_pooled")


# ─────────────────────────────────────────────────────────────────────────────
# Scatter-matrix (pairplot)
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter_matrix(
    datasets: Dict[str, np.ndarray],   # {name: feature_matrix (N,7)}
    out_dir: str = "figures/meta",
    max_per_ds: int = 300,
):
    """
    Seaborn pairplot with dataset hue.
    Subsampled to max_per_ds per dataset for readability.
    """
    rows_list = []
    for name, X in datasets.items():
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), min(max_per_ds, len(X)), replace=False)
        Xs  = np.nan_to_num(X[idx])
        df  = pd.DataFrame(Xs, columns=FEATURE_NAMES_SHORT)
        df["Dataset"] = name
        rows_list.append(df)

    df_all = pd.concat(rows_list, ignore_index=True)
    palette = {n: DATASET_COLORS.get(n, PALETTE[i])
               for i, n in enumerate(datasets.keys())}

    g = sns.pairplot(
        df_all, hue="Dataset", plot_kws=dict(alpha=0.4, s=8, linewidth=0),
        diag_kind="kde", diag_kws=dict(fill=True, alpha=0.4),
        palette=palette, corner=True,
    )
    g.fig.suptitle("Pairwise Scatter Matrix — Complexity Features",
                   fontsize=13, fontweight="bold", y=1.01)
    save_fig(g.fig, Path(out_dir), "scatter_matrix")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--labels",   nargs="+", required=True)
    p.add_argument("--out_dir",  default="figures/meta")
    p.add_argument("--no_pairplot", action="store_true",
                   help="Skip slow pairplot (useful for large datasets)")
    args = p.parse_args()

    Xs = {}
    for f, lbl in zip(args.features, args.labels):
        d     = dict(np.load(f, allow_pickle=True))
        Xs[lbl] = _build_feature_matrix(d)

    plot_correlation_matrix(Xs, args.out_dir)
    if not args.no_pairplot:
        plot_scatter_matrix(Xs, args.out_dir)


if __name__ == "__main__":
    _cli()
