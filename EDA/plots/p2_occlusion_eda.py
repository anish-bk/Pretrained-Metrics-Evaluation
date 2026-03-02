"""
EDA/plots/p2_occlusion_eda.py
==============================
Occlusion EDA — two publication figures:

  Figure 2A:  Occlusion ratio histogram (per dataset) with baseline overlay
  Figure 2B:  Spatial occlusion heatmap (mean occlusion mask aggregated)

Usage:
    python EDA/plots/p2_occlusion_eda.py \
        --features eda_cache/viton_features.npz eda_cache/dresscode_features.npz \
        --labels VITON DressCode --out_dir figures/occlusion
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_style import apply_paper_style, save_fig, add_stat_box, PALETTE, DATASET_COLORS

apply_paper_style()


# ─────────────────────────────────────────────────────────────────────────────
# 2A — Occlusion ratio histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_occlusion_histogram(
    datasets: Dict[str, np.ndarray],   # {name: occ_ratios (N,)}
    out_dir: str = "figures/occlusion",
    bins: int = 40,
):
    """
    Overlaid histogram + KDE of occlusion ratios for each dataset.
    Vertical line at the mean per dataset.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_hist, ax_kde = axes

    for i, (name, occ) in enumerate(datasets.items()):
        color = DATASET_COLORS.get(name, PALETTE[i % len(PALETTE)])
        occ = occ[np.isfinite(occ)]
        ax_hist.hist(occ, bins=bins, histtype="stepfilled",
                     alpha=0.45, color=color, label=name, density=True)
        ax_hist.axvline(occ.mean(), color=color, linestyle="--", linewidth=1.2, alpha=0.9)

        sns.kdeplot(occ, ax=ax_kde, fill=True, alpha=0.30,
                    color=color, linewidth=1.5, label=name)
        ax_kde.axvline(occ.mean(), color=color, linestyle="--", linewidth=1.2, alpha=0.9)

    for ax, title in zip(
        [ax_hist, ax_kde],
        ["Occlusion Ratio — Histogram (density)", "Occlusion Ratio — KDE"]
    ):
        ax.set_xlabel("Occlusion Ratio O_i", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(-0.02, 1.02)
        ax.legend(title="Dataset", framealpha=0.9)

    fig.suptitle("Garment Occlusion Distribution", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, Path(out_dir), "occlusion_histogram")


# ─────────────────────────────────────────────────────────────────────────────
# 2B — Spatial occlusion heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_occlusion_heatmap(
    datasets: Dict[str, np.ndarray],   # {name: occ_maps (N, H, W)}
    out_dir: str = "figures/occlusion",
):
    """
    For each dataset: M_occ = mean(OcclusionMask_i).
    Displayed as a spatial heatmap to reveal WHERE occlusions occur.
    """
    n = len(datasets)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    axes = axes.flatten()

    cmap = "hot"

    for i, (name, maps) in enumerate(datasets.items()):
        mean_map = maps.mean(axis=0)    # (H, W)
        ax = axes[i]
        im = ax.imshow(mean_map, cmap=cmap, vmin=0.0, vmax=min(mean_map.max(), 0.6),
                       aspect="auto", origin="upper")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Occlusion freq.")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Width →"); ax.set_ylabel("Height ↓")
        ax.set_xticks([]); ax.set_yticks([])

        # Annotate max region
        peak_y, peak_x = np.unravel_index(mean_map.argmax(), mean_map.shape)
        ax.plot(peak_x, peak_y, "c+", markersize=12, markeredgewidth=2)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Spatial Occlusion Heatmap (M_occ = mean over dataset)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(fig, Path(out_dir), "occlusion_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--labels",   nargs="+", required=True)
    p.add_argument("--out_dir",  default="figures/occlusion")
    args = p.parse_args()

    assert len(args.features) == len(args.labels)
    occ_ratios = {}; occ_maps = {}
    for f, lbl in zip(args.features, args.labels):
        d = dict(np.load(f, allow_pickle=True))
        occ_ratios[lbl] = d["occ_ratios"]
        occ_maps[lbl]   = d["occ_maps"]

    plot_occlusion_histogram(occ_ratios, args.out_dir)
    plot_occlusion_heatmap(occ_maps,   args.out_dir)


if __name__ == "__main__":
    _cli()
