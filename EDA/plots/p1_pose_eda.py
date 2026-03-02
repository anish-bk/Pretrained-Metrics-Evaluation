"""
EDA/plots/p1_pose_eda.py
=========================
Pose EDA — two publication figures:

  Figure 1A:  Pose UMAP / t-SNE scatter
              Input : pose_vecs (N×34)
              Output: 2-D scatter coloured by dataset

  Figure 1B:  Joint angle distribution (violin + KDE)
              Input : angles (N×8)  — 8 limb triplet angles per image
              Output: one violin per limb, overlaid KDE

Usage (standalone):
    python EDA/plots/p1_pose_eda.py \
        --features eda_cache/viton_features.npz \
        --label VITON --out_dir figures/pose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))  # EDA/
from plot_style import apply_paper_style, save_fig, add_stat_box, PALETTE, DATASET_COLORS

apply_paper_style()

LIMB_NAMES = [
    "L-Elbow", "R-Elbow",
    "L-Knee",  "R-Knee",
    "L-Shoulder", "R-Shoulder",
    "L-Torso", "R-Torso",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1A — UMAP scatter of pose vectors
# ─────────────────────────────────────────────────────────────────────────────

def plot_pose_umap(
    datasets: Dict[str, np.ndarray],   # {dataset_name: pose_vecs (N,34)}
    out_dir: str = "figures/pose",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    use_tsne: bool = False,
):
    """
    Jointly embeds normalised pose vectors from multiple datasets.
    Colours each point by its source dataset.
    """
    labels, vecs = [], []
    for name, V in datasets.items():
        labels.extend([name] * len(V))
        vecs.append(V)

    V_all = np.concatenate(vecs, axis=0)  # (N_total, 34)
    # Normalise
    mu = V_all.mean(0); sig = V_all.std(0) + 1e-8
    V_norm = (V_all - mu) / sig

    print(f"  [PoseEDA] Embedding {len(V_norm)} poses …")

    if use_tsne:
        from sklearn.manifold import TSNE
        Z = TSNE(n_components=2, random_state=42, perplexity=min(30, len(V_norm)//4)).fit_transform(V_norm)
        method_name = "t-SNE"
    else:
        try:
            import umap
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                                random_state=42, n_jobs=1)
            Z = reducer.fit_transform(V_norm)
            method_name = "UMAP"
        except ImportError:
            from sklearn.decomposition import PCA
            Z = PCA(n_components=2, random_state=42).fit_transform(V_norm)
            method_name = "PCA (umap not installed)"

    fig, ax = plt.subplots(figsize=(7, 6))
    unique_ds = list(datasets.keys())
    for i, name in enumerate(unique_ds):
        mask = np.array(labels) == name
        ax.scatter(
            Z[mask, 0], Z[mask, 1],
            s=8, alpha=0.5, linewidths=0,
            color=DATASET_COLORS.get(name, PALETTE[i % len(PALETTE)]),
            label=name,
        )

    ax.set_title(f"Pose Distribution — {method_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"{method_name}-1")
    ax.set_ylabel(f"{method_name}-2")
    ax.legend(markerscale=3, framealpha=0.9, loc="best",
              title="Dataset", ncol=2 if len(unique_ds) > 5 else 1)
    ax.set_xticks([]); ax.set_yticks([])

    save_fig(fig, Path(out_dir), "pose_umap")


# ─────────────────────────────────────────────────────────────────────────────
# 1B — Joint angle distributions
# ─────────────────────────────────────────────────────────────────────────────

def plot_joint_angle_distributions(
    datasets: Dict[str, np.ndarray],   # {dataset_name: angles (N,8)}
    out_dir: str = "figures/pose",
):
    """
    One violin per joint angle, side-by-side for each dataset.
    Also plots a KDE overlay for each dataset in a secondary panel.
    """
    n_joints = len(LIMB_NAMES)

    # ── (A) Multi-dataset violin per joint ────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=False)
    axes = axes.flatten()

    unique_ds = list(datasets.keys())
    for j, limb in enumerate(LIMB_NAMES):
        ax = axes[j]
        data_per_ds = {name: datasets[name][:, j] * (180 / np.pi)  # convert to degrees
                       for name in unique_ds}
        df_list = [
            {"Angle (°)": v, "Dataset": name}
            for name, arr in data_per_ds.items()
            for v in arr
        ]
        import pandas as pd
        df = pd.DataFrame(df_list)
        palette = {n: DATASET_COLORS.get(n, "#8888ff") for n in unique_ds}
        if len(df) > 0:
            sns.violinplot(data=df, x="Dataset", y="Angle (°)",
                           palette=palette, ax=ax, cut=0,
                           inner="quartile", linewidth=0.8)
        ax.set_title(limb, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        add_stat_box(ax, df["Angle (°)"].values if len(df) else np.array([]))

    fig.suptitle("Joint Angle Distribution per Limb (degrees)", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(fig, Path(out_dir), "joint_angle_violin")

    # ── (B) KDE overlay per dataset (all joints combined) ─────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for i, (name, angles) in enumerate(datasets.items()):
        all_angles = (angles * (180 / np.pi)).flatten()
        all_angles = all_angles[np.isfinite(all_angles)]
        if len(all_angles) < 5:
            continue
        sns.kdeplot(
            all_angles, ax=ax2, fill=True, alpha=0.25, linewidth=1.5,
            color=DATASET_COLORS.get(name, PALETTE[i]),
            label=name,
        )
    ax2.set_xlabel("Joint Angle (degrees)", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("Joint Angle KDE — All Limbs Combined", fontsize=13, fontweight="bold")
    ax2.legend(title="Dataset", framealpha=0.9)
    plt.tight_layout()
    save_fig(fig2, Path(out_dir), "joint_angle_kde")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--features",  nargs="+", required=True, help=".npz feature files")
    p.add_argument("--labels",    nargs="+", required=True, help="Dataset label per file")
    p.add_argument("--out_dir",   default="figures/pose")
    p.add_argument("--tsne",      action="store_true", help="Use t-SNE instead of UMAP")
    args = p.parse_args()

    assert len(args.features) == len(args.labels), "features and labels must match"
    datasets_pose = {}
    datasets_angles = {}
    for fpath, label in zip(args.features, args.labels):
        d = dict(np.load(fpath, allow_pickle=True))
        datasets_pose[label]   = d["pose_vecs"]
        datasets_angles[label] = d["angles"]

    plot_pose_umap(datasets_pose, args.out_dir, use_tsne=args.tsne)
    plot_joint_angle_distributions(datasets_angles, args.out_dir)


if __name__ == "__main__":
    _cli()
