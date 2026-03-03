"""
metrics/image_metrics.py
--------------------------
Per-batch and per-dataset image-quality metrics:
  - PSNR
  - SSIM
  - Masked SSIM
  - LPIPS
"""

import torch
import numpy as np
from typing import Optional

# scikit-image for SSIM/PSNR
from skimage.metrics import structural_similarity as ski_ssim
from skimage.metrics import peak_signal_noise_ratio as ski_psnr


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_numpy_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert (B, C, H, W) or (C, H, W) float [0,1] tensor → uint8 HWC numpy."""
    if tensor.ndim == 4:
        # take first image in batch for single-image calls; handled in batch mode below
        tensor = tensor[0]
    arr = tensor.detach().cpu().permute(1, 2, 0).numpy()  # HWC
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def _to_numpy_float(tensor: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) float [0,1] tensor → float HWC numpy."""
    if tensor.ndim == 4:
        tensor = tensor[0]
    return tensor.detach().cpu().permute(1, 2, 0).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Individual metric functions (operate on batches; return per-image lists)
# ─────────────────────────────────────────────────────────────────────────────

def compute_psnr_batch(pred: torch.Tensor, gt: torch.Tensor) -> list:
    """
    Args:
        pred, gt : (B, C, H, W) float tensors in [0, 1]
    Returns:
        List[float] of per-image PSNR values
    """
    results = []
    B = pred.shape[0]
    for i in range(B):
        p = np.clip(pred[i].detach().cpu().permute(1, 2, 0).numpy(), 0, 1)
        g = np.clip(gt[i].detach().cpu().permute(1, 2, 0).numpy(),   0, 1)
        psnr = ski_psnr(g, p, data_range=1.0)
        results.append(float(psnr))
    return results


def compute_ssim_batch(pred: torch.Tensor, gt: torch.Tensor) -> list:
    """
    Returns:
        List[float] of per-image SSIM values
    """
    results = []
    B = pred.shape[0]
    for i in range(B):
        p = np.clip(pred[i].detach().cpu().permute(1, 2, 0).numpy(), 0, 1)
        g = np.clip(gt[i].detach().cpu().permute(1, 2, 0).numpy(),   0, 1)
        # channel_axis added in skimage ≥ 0.19
        val = ski_ssim(g, p, data_range=1.0, channel_axis=-1)
        results.append(float(val))
    return results


def compute_masked_ssim_batch(
    pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor
) -> list:
    """
    Masked SSIM: SSIM computed only over the masked region.
    mask : (B, 1, H, W) binary float tensor
    Returns:
        List[float]
    """
    results = []
    B = pred.shape[0]
    for i in range(B):
        p   = np.clip(pred[i].detach().cpu().permute(1, 2, 0).numpy(), 0, 1)  # HWC
        g   = np.clip(gt[i].detach().cpu().permute(1, 2, 0).numpy(),   0, 1)
        m   = mask[i, 0].detach().cpu().numpy()  # HW binary

        # Expand mask to 3 channels
        m3  = np.stack([m, m, m], axis=-1) > 0.5   # HWC bool

        # Replace non-mask pixels with identical values so they don't contribute
        p_m = np.where(m3, p, g)

        # Window size must be ≤ min(H,W)
        win_size = min(7, p.shape[0], p.shape[1])
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            results.append(float("nan"))
            continue

        val = ski_ssim(g, p_m, data_range=1.0, channel_axis=-1, win_size=win_size)
        results.append(float(val))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# LPIPS
# ─────────────────────────────────────────────────────────────────────────────

class LPIPSMetric:
    """Wrapper around the lpips library; supports batched computation."""

    def __init__(self, net: str = "alex", device: str = "cpu"):
        import lpips
        self.loss_fn = lpips.LPIPS(net=net).to(device)
        self.loss_fn.eval()
        self.device = device

    @torch.no_grad()
    def compute_batch(self, pred: torch.Tensor, gt: torch.Tensor) -> list:
        """Expects (B, C, H, W) in [0, 1]; normalises to [-1, 1] internally."""
        pred = (pred * 2 - 1).to(self.device)
        gt   = (gt   * 2 - 1).to(self.device)
        scores = self.loss_fn(pred, gt)      # (B, 1, 1, 1)
        return scores.view(-1).cpu().tolist()
