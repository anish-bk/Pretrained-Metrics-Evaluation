"""
EDA/plots/p3_background_eda.py
================================
Background Complexity EDA — two publication figures:

  Figure 3A:  Background entropy histogram  (vs baseline overlay)
  Figure 3B:  Entropy vs Object-count scatter  (clean-studio vs complex)

Usage:
    python EDA/plots/p3_background_eda.py \
        --features eda_cache/*.npz --labels ... --out_dir figures/background
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_style import apply_paper_style, save_fig, add_stat_box, PALETTE, DATASET_COLORS

apply_paper_style()


# ─────────────────────────────────────────────────────────────────────────────
# 3A — Background entropy histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_bg_entropy_histogram(
    datasets: Dict[str, np.ndarray],   # {name: bg_entropy (N,)}
    out_dir: str = "figures/background",
    bins: int = 40,
):
    """Overlaid entropy histogram + KDE. Vertical mean lines per dataset."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for i, (name, ent) in enumerate(datasets.items()):
        c = DATASET_COLORS.get(name, PALETTE[i])
        ent = ent[np.isfinite(ent)]
        ax1.hist(ent, bins=bins, density=True, histtype="stepfilled",
                 alpha=0.4, color=c, label=name)
        ax1.axvline(ent.mean(), color=c, linestyle="--", linewidth=1.3, alpha=0.9)
        sns.kdeplot(ent, ax=ax2, fill=True, alpha=0.3, color=c,
                    linewidth=1.5, label=name)
        ax2.axvline(ent.mean(), color=c, linestyle="--", linewidth=1.2)

    for ax, title in zip(
        [ax1, ax2],
        ["Background Texture Entropy — Histogram", "Background Texture Entropy — KDE"],
    ):
        ax.set_xlabel("H_bg (bits)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(title="Dataset", framealpha=0.9)

    fig.suptitle("Background Complexity — Texture Entropy",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, Path(out_dir), "bg_entropy_histogram")


# ─────────────────────────────────────────────────────────────────────────────
# 3B — Entropy vs Object density scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_entropy_vs_objects(
    datasets_ent: Dict[str, np.ndarray],   # {name: bg_entropy (N,)}
    datasets_obj: Dict[str, np.ndarray],   # {name: bg_obj_count (N,)}
    out_dir: str = "figures/background",
):
    """
    2D scatter: x = #objects, y = entropy.
    Colour = dataset. Annotates low/high complexity quadrants.
    Adds marginal histograms (seaborn JointGrid) per dataset.
    """
    # ── Composite scatter ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))

    all_ents = []; all_objs = []
    for i, (name, ent) in enumerate(datasets_ent.items()):
        obj = datasets_obj.get(name, np.zeros(len(ent)))
        mask = np.isfinite(ent) & np.isfinite(obj)
        ent_m = ent[mask]; obj_m = obj[mask]
        c = DATASET_COLORS.get(name, PALETTE[i])
        ax.scatter(obj_m, ent_m, s=14, alpha=0.4, color=c,
                   linewidths=0, label=name)
        all_ents.extend(ent_m.tolist())
        all_objs.extend(obj_m.tolist())

    # Quadrant annotation
    ent_med = np.median(all_ents)
    obj_med = np.median(all_objs)
    for (tx, ty, label) in [
        (0.02, 0.98, "Simple BG\n(low objects,\nlow entropy)"),
        (0.98, 0.98, "Complex BG\n(many objects,\nhigh entropy)"),
        (0.02, 0.02, "Flat BG\n(low objects,\nhigh entropy)"),
        (0.98, 0.02, "Cluttered BG\n(many objects,\nlow entropy)"),
    ]:
        ax.text(tx, ty, label, transform=ax.transAxes,
                ha="left" if tx < 0.5 else "right",
                va="top"  if ty > 0.5 else "bottom",
                fontsize=8, color="grey", style="italic")

    # Median crosshairs
    ax.axhline(ent_med, color="grey", linestyle=":", linewidth=1, alpha=0.7)
    ax.axvline(obj_med, color="grey", linestyle=":", linewidth=1, alpha=0.7)

    ax.set_xlabel("Background Object Count  (#objects)", fontsize=12)
    ax.set_ylabel("Background Texture Entropy  H_bg (bits)", fontsize=12)
    ax.set_title("Background Complexity: Entropy vs. Object Density",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Dataset", markerscale=2, framealpha=0.9,
              loc="center right")
    plt.tight_layout()
    save_fig(fig, Path(out_dir), "bg_entropy_vs_objects")

    # ── Per-dataset marginal joint plot ────────────────────────────────────
    import itertools
    names = list(datasets_ent.keys())
    if len(names) == 1:
        name = names[0]
        ent  = datasets_ent[name]
        obj  = datasets_obj.get(name, np.zeros(len(ent)))
        mask = np.isfinite(ent) & np.isfinite(obj)
        jg = sns.jointplot(
            x=obj[mask], y=ent[mask],
            kind="scatter", alpha=0.4, s=10,
            marginal_kws=dict(fill=True, alpha=0.5),
            color=DATASET_COLORS.get(name, PALETTE[0]),
        )
        jg.ax_joint.set_xlabel("#Objects in BG")
        jg.ax_joint.set_ylabel("Entropy H_bg")
        jg.figure.suptitle(name, y=1.01, fontsize=12, fontweight="bold")
        save_fig(jg.figure, Path(out_dir), f"bg_joint_{name}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--labels",   nargs="+", required=True)
    p.add_argument("--out_dir",  default="figures/background")
    args = p.parse_args()

    ents = {}; objs = {}
    for f, lbl in zip(args.features, args.labels):
        d = dict(np.load(f, allow_pickle=True))
        ents[lbl] = d["bg_entropy"]
        objs[lbl] = d["bg_obj_count"].astype(float)

    plot_bg_entropy_histogram(ents, args.out_dir)
    plot_entropy_vs_objects(ents, objs, args.out_dir)


if __name__ == "__main__":
    _cli()
