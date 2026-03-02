"""
EDA/plots/p4_illumination_eda.py
=================================
Illumination EDA — two publication figures:

  Figure 4A:  Luminance spectrum (distribution of mean-L per image)
  Figure 4B:  PCA of illumination maps → 2D scatter revealing lighting modes

Usage:
    python EDA/plots/p4_illumination_eda.py \
        --features eda_cache/*.npz --labels ... --out_dir figures/illumination
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_style import apply_paper_style, save_fig, add_stat_box, PALETTE, DATASET_COLORS

apply_paper_style()


# ─────────────────────────────────────────────────────────────────────────────
# 4A — Luminance spectrum
# ─────────────────────────────────────────────────────────────────────────────

def plot_luminance_spectrum(
    datasets: Dict[str, np.ndarray],   # {name: lum_mean (N,)}
    datasets_grad: Dict[str, np.ndarray],  # {name: lum_grad_var (N,)}
    out_dir: str = "figures/illumination",
):
    """
    Panel 1: overlaid KDE of mean-luminance L_i
    Panel 2: gradient variance distribution (C_light proxy)
    Vertical shaded bands show dark (<0.3) / mid / bright (>0.7) regimes.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for i, (name, lum) in enumerate(datasets.items()):
        c = DATASET_COLORS.get(name, PALETTE[i])
        lum = lum[np.isfinite(lum)]
        sns.kdeplot(lum, ax=ax1, fill=True, alpha=0.3, color=c,
                    linewidth=1.5, label=name)
        ax1.axvline(lum.mean(), color=c, linestyle="--", linewidth=1.1)

        gv = datasets_grad.get(name, np.array([]))
        gv = gv[np.isfinite(gv)]
        if len(gv) > 2:
            sns.kdeplot(gv, ax=ax2, fill=True, alpha=0.3, color=c,
                        linewidth=1.5, label=name)

    # Regime bands for luminance
    for ax, (xl, xr, label, col) in [
        (ax1, (0.0, 0.3, "Dark", "#BBDEFB")),
        (ax1, (0.3, 0.7, "Mid",  "#E8F5E9")),
        (ax1, (0.7, 1.0, "Bright","#FFF9C4")),
    ]:
        ax.axvspan(xl, xr, alpha=0.12, color=col)
        ax.text((xl + xr) / 2, ax.get_ylim()[1] * 0.92, label,
                ha="center", fontsize=8, color="grey", style="italic")

    ax1.set_xlabel("Mean Luminance L_i (CIE-LAB, normalised)", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Luminance Spectrum", fontsize=12, fontweight="bold")
    ax1.set_xlim(0, 1)
    ax1.legend(title="Dataset", framealpha=0.9)

    ax2.set_xlabel("Gradient Variance  Var(||∇I||)", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("Illumination Gradient Variance (C_light)", fontsize=12, fontweight="bold")
    ax2.legend(title="Dataset", framealpha=0.9)

    fig.suptitle("Illumination Complexity Analysis", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, Path(out_dir), "luminance_spectrum")


# ─────────────────────────────────────────────────────────────────────────────
# 4B — PCA of illumination maps
# ─────────────────────────────────────────────────────────────────────────────

def plot_illumination_pca(
    datasets: Dict[str, np.ndarray],   # {name: lum_maps (N, H, W)}
    out_dir: str = "figures/illumination",
    n_components: int = 2,
):
    """
    Flatten each downsampled L map → PCA → 2D scatter.
    Reveals distinct lighting modes (indoor, outdoor, dramatic, flat).
    """
    labels_all, mats = [], []
    for name, maps in datasets.items():
        N = len(maps)
        flat = maps.reshape(N, -1)   # (N, H*W)
        labels_all.extend([name] * N)
        mats.append(flat)

    if not mats:
        return

    X = np.concatenate(mats, axis=0).astype(np.float32)
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.5, posinf=1.0, neginf=0.0)

    pca = PCA(n_components=2, random_state=42)
    Z   = pca.fit_transform(X)   # (N_total, 2)
    ev  = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(datasets.keys()):
        mask = np.array(labels_all) == name
        ax.scatter(Z[mask, 0], Z[mask, 1], s=10, alpha=0.4, linewidths=0,
                   color=DATASET_COLORS.get(name, PALETTE[i]), label=name)

    ax.set_xlabel(f"PC-1  ({ev[0]:.1f}% var)", fontsize=12)
    ax.set_ylabel(f"PC-2  ({ev[1]:.1f}% var)", fontsize=12)
    ax.set_title("PCA of Illumination Maps (Z_light)", fontsize=13, fontweight="bold")
    ax.legend(title="Dataset", markerscale=3, framealpha=0.9,
              ncol=2 if len(datasets) > 5 else 1)

    # Eigenvalue spectrum inset
    n_show = min(10, pca.n_components_)
    inset = ax.inset_axes([0.7, 0.02, 0.28, 0.28])
    ev_full = pca.explained_variance_ratio_[:n_show] * 100
    inset.bar(range(1, n_show + 1), ev_full, color="#90CAF9", edgecolor="white")
    inset.set_title("Var %", fontsize=7)
    inset.set_xlabel("PC", fontsize=7)
    inset.tick_params(labelsize=6)

    plt.tight_layout()
    save_fig(fig, Path(out_dir), "illumination_pca")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--labels",   nargs="+", required=True)
    p.add_argument("--out_dir",  default="figures/illumination")
    args = p.parse_args()

    lums = {}; grads = {}; maps = {}
    for f, lbl in zip(args.features, args.labels):
        d = dict(np.load(f, allow_pickle=True))
        lums[lbl]  = d["lum_mean"]
        grads[lbl] = d["lum_grad_var"]
        maps[lbl]  = d["lum_maps"]

    plot_luminance_spectrum(lums, grads, args.out_dir)
    plot_illumination_pca(maps, args.out_dir)


if __name__ == "__main__":
    _cli()
