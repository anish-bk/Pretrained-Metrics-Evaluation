"""
metrics/distribution_metrics.py
---------------------------------
Dataset-level distributional metrics:
  - FID  (Fréchet Inception Distance)   via clean-fid / torch-fidelity
  - IS   (Inception Score)              via torch-fidelity
  - KID  (Kernel Inception Distance)    via torch-fidelity

Usage pattern (see evaluate.py):
    dm = DistributionMetrics(device)
    for pred, gt in dataloader:
        dm.update(pred, gt)
    results = dm.compute()
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Helper: save tensor batch to temp folder as PNG files
# ─────────────────────────────────────────────────────────────────────────────

def _save_batch_to_dir(batch: torch.Tensor, directory: Path, offset: int = 0):
    """Save (B, C, H, W) float [0,1] tensor images as PNG files."""
    directory.mkdir(parents=True, exist_ok=True)
    for i, img_t in enumerate(batch):
        pil = TF.to_pil_image(img_t.clamp(0, 1).cpu())
        pil.save(directory / f"{offset + i:06d}.png")


# ─────────────────────────────────────────────────────────────────────────────
# DistributionMetrics
# ─────────────────────────────────────────────────────────────────────────────

class DistributionMetrics:
    """
    Accumulates predicted and ground-truth images across batches then
    computes FID, IS, and KID once the full dataset has been seen.
    """

    def __init__(self, device: str = "cpu", img_size: int = 299):
        self.device = device
        self.img_size = img_size
        self._tmp = tempfile.mkdtemp(prefix="tryon_eval_")
        self._pred_dir = Path(self._tmp) / "pred"
        self._gt_dir   = Path(self._tmp) / "gt"
        self._pred_dir.mkdir()
        self._gt_dir.mkdir()
        self._count = 0

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        """Add a batch of (B, C, H, W) images in [0, 1]."""
        _save_batch_to_dir(pred, self._pred_dir, offset=self._count)
        _save_batch_to_dir(gt,   self._gt_dir,   offset=self._count)
        self._count += pred.shape[0]

    def compute(self) -> Dict[str, float]:
        """
        Compute FID, IS, KID from accumulated images.
        Returns a dict with keys: fid, is_mean, is_std, kid_mean, kid_std
        """
        try:
            from torch_fidelity import calculate_metrics
        except ImportError:
            raise ImportError(
                "torch-fidelity not installed. Run: pip install torch-fidelity"
            )

        gpu = self.device.startswith("cuda")
        metrics = calculate_metrics(
            input1=str(self._pred_dir),   # generated images
            input2=str(self._gt_dir),     # reference / real images
            cuda=gpu,
            isc=True,    # Inception Score
            fid=True,    # FID
            kid=True,    # KID
            verbose=False,
        )

        return {
            "fid":      float(metrics["frechet_inception_distance"]),
            "is_mean":  float(metrics["inception_score_mean"]),
            "is_std":   float(metrics["inception_score_std"]),
            "kid_mean": float(metrics["kernel_inception_distance_mean"]),
            "kid_std":  float(metrics["kernel_inception_distance_std"]),
        }

    def cleanup(self):
        """Remove temporary directories."""
        shutil.rmtree(self._tmp, ignore_errors=True)

    def __del__(self):
        self.cleanup()
