"""
EDA/plots/p7_garment_eda.py
=============================
Garment Texture Diversity EDA — two publication figures:

  Figure 7A:  UMAP of garment (CLIP) embeddings — style/pattern/colour clusters
  Figure 7B:  Embedding covariance eigenvalue spectrum
              Fast eigenvalue decay → low diversity
              Slow decay           → high diversity (broad manifold)

Usage:
    python EDA/plots/p7_garment_eda.py \
        --features eda_cache/*.npz --labels ... --out_dir figures/garment
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
# 7A — UMAP of garment embeddings
# ─────────────────────────────────────────────────────────────────────────────

def plot_garment_umap(
    datasets: Dict[str, np.ndarray],   # {name: garment_embs (N,D)}
    out_dir: str = "figures/garment",
    n_neighbors: int = 15,
    min_dist: float = 0.05,
    max_per_ds: int = 3000,
):
    """
    Joint embedding of garment CLIP features.
    Clusters reveal style groups (patterns, solids, formals, casuals).
    """
    labels_all, embs = [], []
    for name, E in datasets.items():
        idx = np.random.default_rng(0).choice(len(E), min(max_per_ds, len(E)), replace=False)
        labels_all.extend([name] * len(idx))
        embs.append(E[idx])

    if not embs:
        return

    E_all = np.concatenate(embs, axis=0).astype(np.float32)
    E_all = np.nan_to_num(E_all)
    E_all = E_all / (np.linalg.norm(E_all, axis=1, keepdims=True) + 1e-12)

    print(f"  [GarmentEDA] Embedding {len(E_all)} garment vectors …")
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                            random_state=0, n_jobs=1, metric="cosine")
        Z = reducer.fit_transform(E_all)
        method = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        Z = TSNE(n_components=2, random_state=0, metric="cosine",
                 perplexity=min(30, len(E_all) // 4)).fit_transform(E_all)
        method = "t-SNE"

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(datasets.keys()):
        mask = np.array(labels_all) == name
        ax.scatter(Z[mask, 0], Z[mask, 1], s=9, alpha=0.45, linewidths=0,
                   color=DATASET_COLORS.get(name, PALETTE[i]), label=name)

    ax.set_title(f"Garment Diversity — {method} of CLIP Embeddings",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel(f"{method}-1"); ax.set_ylabel(f"{method}-2")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(title="Dataset", markerscale=3, framealpha=0.9,
              ncol=2 if len(datasets) > 5 else 1)
    plt.tight_layout()
    save_fig(fig, Path(out_dir), "garment_umap")


# ─────────────────────────────────────────────────────────────────────────────
# 7B — Eigenvalue spectrum of Cov(g)
# ─────────────────────────────────────────────────────────────────────────────

def plot_eigenvalue_spectrum(
    datasets: Dict[str, np.ndarray],   # {name: garment_embs (N,D)}
    out_dir: str = "figures/garment",
    top_k: int = 50,
):
    """
    For each dataset: eigenvalues of sample covariance of garment embeddings.
    Normalised (cumulative explained variance) and plotted as line curves.
    Slow decay = high diversity.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for i, (name, E) in enumerate(datasets.items()):
        E = E.astype(np.float32)
        E = np.nan_to_num(E)
        mu  = E.mean(0, keepdims=True)
        Ec  = E - mu
        # Use economy SVD for large D
        U, s, Vt = np.linalg.svd(Ec, full_matrices=False)
        evs  = s ** 2 / max(len(E) - 1, 1)      # eigenvalues
        evs  = evs[:top_k]
        evs_norm = evs / evs.sum()               # normalised
        cumvar   = np.cumsum(evs_norm) * 100     # cumulative %

        c = DATASET_COLORS.get(name, PALETTE[i])
        ks = np.arange(1, len(evs) + 1)

        ax1.plot(ks, evs_norm, "-o", markersize=3, color=c, linewidth=1.5, label=name)
        ax2.plot(ks, cumvar,   "-o", markersize=3, color=c, linewidth=1.5, label=name)

    ax1.set_xlabel("Eigenvalue Index k", fontsize=12)
    ax1.set_ylabel("Normalised Eigenvalue", fontsize=12)
    ax1.set_title("Cov(g) Eigenvalue Spectrum\n(slow decay → high diversity)",
                  fontsize=12, fontweight="bold")
    ax1.set_yscale("log")
    ax1.legend(title="Dataset", framealpha=0.9)
    ax1.grid(True, which="both", alpha=0.3)

    ax2.set_xlabel("Number of Components k", fontsize=12)
    ax2.set_ylabel("Cumulative Variance Explained (%)", fontsize=12)
    ax2.set_title("Cumulative Explained Variance", fontsize=12, fontweight="bold")
    ax2.axhline(90, color="grey", linestyle=":", linewidth=1.0, label="90% threshold")
    ax2.axhline(99, color="silver", linestyle=":", linewidth=1.0, label="99% threshold")
    ax2.legend(title="Dataset", framealpha=0.9)
    ax2.set_ylim(0, 101)

    fig.suptitle("Garment Embedding Covariance — Eigenvalue Analysis",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, Path(out_dir), "garment_eigenvalue_spectrum")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--labels",   nargs="+", required=True)
    p.add_argument("--out_dir",  default="figures/garment")
    args = p.parse_args()

    garment_embs = {}
    for f, lbl in zip(args.features, args.labels):
        d               = dict(np.load(f, allow_pickle=True))
        garment_embs[lbl] = d["garment_embs"]

    plot_garment_umap(garment_embs, args.out_dir)
    plot_eigenvalue_spectrum(garment_embs, args.out_dir)


if __name__ == "__main__":
    _cli()
