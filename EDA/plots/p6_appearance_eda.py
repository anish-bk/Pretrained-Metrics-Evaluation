"""
EDA/plots/p6_appearance_eda.py
================================
Appearance / Face Diversity EDA — two publication figures:

  Figure 6A:  UMAP of face embeddings (coloured by dataset)
  Figure 6B:  Pairwise cosine distance histogram (higher mean → more diversity)

Usage:
    python EDA/plots/p6_appearance_eda.py \
        --features eda_cache/*.npz --labels ... --out_dir figures/appearance
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_style import apply_paper_style, save_fig, add_stat_box, PALETTE, DATASET_COLORS

apply_paper_style()


# ─────────────────────────────────────────────────────────────────────────────
# 6A — UMAP of face embeddings
# ─────────────────────────────────────────────────────────────────────────────

def plot_face_umap(
    datasets: Dict[str, np.ndarray],   # {name: face_embs (N,D)}
    out_dir: str = "figures/appearance",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    max_per_ds: int = 2000,
):
    """
    Joint UMAP of face embeddings from all datasets.
    Optionally coloured by a secondary scalar (luminance, bg_entropy).
    """
    labels_all, embs = [], []
    for name, E in datasets.items():
        # Subsample for speed
        idx = np.random.default_rng(42).choice(len(E), min(max_per_ds, len(E)), replace=False)
        labels_all.extend([name] * len(idx))
        embs.append(E[idx])

    if not embs:
        return

    E_all = np.concatenate(embs, axis=0).astype(np.float32)
    E_all = np.nan_to_num(E_all)
    # L2-normalise
    E_all = E_all / (np.linalg.norm(E_all, axis=1, keepdims=True) + 1e-12)

    print(f"  [AppearanceEDA] Embedding {len(E_all)} face vectors …")

    try:
        import umap
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                            random_state=42, n_jobs=1, metric="cosine")
        Z = reducer.fit_transform(E_all)
        method = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        Z = TSNE(n_components=2, random_state=42,
                 metric="cosine",
                 perplexity=min(30, len(E_all) // 4)).fit_transform(E_all)
        method = "t-SNE"

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(datasets.keys()):
        mask = np.array(labels_all) == name
        ax.scatter(Z[mask, 0], Z[mask, 1], s=8, alpha=0.45, linewidths=0,
                   color=DATASET_COLORS.get(name, PALETTE[i]), label=name)

    ax.set_title(f"Face Embedding {method} — Appearance Diversity",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel(f"{method}-1"); ax.set_ylabel(f"{method}-2")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(title="Dataset", markerscale=3, framealpha=0.9,
              ncol=2 if len(datasets) > 5 else 1)
    plt.tight_layout()
    save_fig(fig, Path(out_dir), "appearance_umap")


# ─────────────────────────────────────────────────────────────────────────────
# 6B — Pairwise cosine distance distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_pairwise_distance_distribution(
    datasets: Dict[str, np.ndarray],   # {name: face_embs (N,D)}
    out_dir: str = "figures/appearance",
    max_pairs: int = 5000,
):
    """
    For each dataset: sample N pairs, compute cosine distance, plot KDE.
    Vertical line = mean pairwise distance (D_face).
    Higher mean → more diverse faces.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    diversity_scores = {}

    for i, (name, E) in enumerate(datasets.items()):
        E = E.astype(np.float32)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        N = len(E)

        # Sample pairs without replacement
        rng = np.random.default_rng(42)
        n_pairs = min(max_pairs, N * (N - 1) // 2)
        idxA = rng.integers(0, N, n_pairs)
        idxB = rng.integers(0, N, n_pairs)
        same = idxA == idxB
        idxB[same] = (idxA[same] + 1) % N

        cos_sim = (E[idxA] * E[idxB]).sum(axis=1)
        cos_dist = 1.0 - cos_sim

        c = DATASET_COLORS.get(name, PALETTE[i])
        sns.kdeplot(cos_dist, ax=ax, fill=True, alpha=0.3, color=c,
                    linewidth=1.5, label=f"{name}  (D={cos_dist.mean():.3f})")
        ax.axvline(cos_dist.mean(), color=c, linestyle="--", linewidth=1.3)
        diversity_scores[name] = float(cos_dist.mean())

    ax.set_xlabel("Pairwise Cosine Distance  (1 − cos_sim)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Face Diversity — Pairwise Cosine Distance Distribution",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, 2.0)
    ax.legend(title="Dataset  (D_face=mean)", framealpha=0.9,
              fontsize=9, loc="upper left")

    # Bar chart inset: D_face per dataset
    names  = list(diversity_scores.keys())
    scores = [diversity_scores[n] for n in names]
    colors = [DATASET_COLORS.get(n, "#8888ff") for n in names]
    inset  = ax.inset_axes([0.62, 0.52, 0.36, 0.44])
    bars   = inset.bar(range(len(names)), scores, color=colors, edgecolor="white", linewidth=0.5)
    inset.set_xticks(range(len(names)))
    inset.set_xticklabels([n[:6] for n in names], rotation=45, fontsize=6, ha="right")
    inset.set_ylabel("D_face", fontsize=7)
    inset.set_title("Diversity score", fontsize=7)
    inset.tick_params(axis="y", labelsize=6)

    plt.tight_layout()
    save_fig(fig, Path(out_dir), "appearance_pairwise_distance")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--labels",   nargs="+", required=True)
    p.add_argument("--out_dir",  default="figures/appearance")
    args = p.parse_args()

    face_embs = {}
    for f, lbl in zip(args.features, args.labels):
        d           = dict(np.load(f, allow_pickle=True))
        face_embs[lbl] = d["face_embs"]

    plot_face_umap(face_embs, args.out_dir)
    plot_pairwise_distance_distribution(face_embs, args.out_dir)


if __name__ == "__main__":
    _cli()
