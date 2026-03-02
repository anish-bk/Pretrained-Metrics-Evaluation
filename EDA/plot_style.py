"""
EDA/plot_style.py
==================
Shared publication-quality matplotlib/seaborn configuration.
Import this module first in any plotting script.

Provides:
  - apply_paper_style()  → sets RC params for IEEE/CVPR figures
  - PALETTE              → colour palette (one colour per dataset, up to 10)
  - DATASET_COLORS       → dict mapping dataset name → hex colour
  - save_fig()           → saves PDF + high-DPI PNG side-by-side
  - add_stat_annotation() → adds μ ± σ text box to an Axes
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette — 10 visually distinct colours for 10 datasets
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = [
    "#2196F3",  # blue          viton
    "#E91E63",  # pink          viton_hd
    "#4CAF50",  # green         dresscode
    "#FF9800",  # orange        mpv
    "#9C27B0",  # purple        deepfashion_tryon
    "#00BCD4",  # cyan          acgpn
    "#F44336",  # red           cp_vton
    "#8BC34A",  # light-green   hr_vton
    "#FF5722",  # deep-orange   ladi_vton
    "#607D8B",  # blue-grey     ovnet
]

DATASET_ORDER = [
    "viton", "viton_hd", "dresscode", "mpv", "deepfashion_tryon",
    "acgpn", "cp_vton", "hr_vton", "ladi_vton", "ovnet",
]
DATASET_COLORS: Dict[str, str] = {
    name: PALETTE[i % len(PALETTE)]
    for i, name in enumerate(DATASET_ORDER)
}


# ─────────────────────────────────────────────────────────────────────────────
# RC params
# ─────────────────────────────────────────────────────────────────────────────

def apply_paper_style(font_scale: float = 1.1):
    """
    Apply a clean, publication-ready style.
    Call once at the top of each plotting script.
    """
    sns.set_theme(style="whitegrid", font_scale=font_scale)
    matplotlib.rcParams.update({
        # Font
        "font.family":       "DejaVu Sans",
        "axes.titlesize":    13,
        "axes.labelsize":    12,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":   10,
        "legend.title_fontsize": 11,
        # Lines
        "axes.linewidth":    0.8,
        "grid.linewidth":    0.5,
        "grid.alpha":        0.4,
        # Figure
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.05,
        # Colours
        "axes.prop_cycle":   matplotlib.cycler(color=PALETTE),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Save helper
# ─────────────────────────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, out_dir: Path, stem: str):
    """Save figure as both PDF and PNG (300 dpi)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p)
        print(f"  Saved → {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Stat annotation
# ─────────────────────────────────────────────────────────────────────────────

def add_stat_box(ax: plt.Axes, values: np.ndarray, x: float = 0.97, y: float = 0.95):
    """Add μ ± σ text box in the upper-right corner of ax."""
    mu  = np.nanmean(values)
    sig = np.nanstd(values)
    text = f"μ = {mu:.3f}\nσ = {sig:.3f}"
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8),
    )
