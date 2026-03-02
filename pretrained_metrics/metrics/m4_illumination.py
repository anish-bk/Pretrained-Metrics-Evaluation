"""
metrics/m4_illumination.py
===========================
Metric 4 — Illumination Complexity
-------------------------------------

Step 1: Extract luminance channel (L) from CIE-LAB colour space.
    L_i = mean(L channel)  → overall brightness
    Var(L_i) across dataset → lighting variance

Step 2: Sobel illumination gradient
    G_i = ||∇I||   (magnitude of Sobel gradient over L channel)
    C_light = E[Var(G_i)]

No pretrained model is required — this is a signal-processing metric.

Input
------
person_imgs : torch.Tensor  (B, 3, H, W)  float32  [0, 1]

Returns (compute())
--------------------
dict with:
    luminance_mean_global       : mean L across all images
    luminance_var_global        : variance of per-image mean-L
    illumination_gradient_mean  : E[Var(G_i)]  ← C_light
    illumination_complexity     : luminance_var_global + illumination_gradient_mean
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rgb_to_lab_l(imgs: torch.Tensor) -> np.ndarray:
    """
    imgs : (B, 3, H, W)  float32  [0,1]
    Returns (B,) array of mean-L luminance per image.
    Also returns (B, H, W) array of L channels.
    """
    B = imgs.shape[0]
    mean_L   = np.zeros(B, dtype=np.float32)
    L_maps   = []

    for i in range(B):
        # Convert tensor → uint8 BGR for OpenCV
        rgb_np = imgs[i].permute(1, 2, 0).numpy()          # (H,W,3) float [0,1]
        rgb_u8 = (np.clip(rgb_np, 0, 1) * 255).astype(np.uint8)
        bgr_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
        lab    = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB)   # L: 0-255
        L      = lab[:, :, 0].astype(np.float32) / 255.0   # [0,1]
        mean_L[i] = L.mean()
        L_maps.append(L)

    return mean_L, L_maps


def _sobel_gradient_variance(L_map: np.ndarray) -> float:
    """
    L_map : (H, W) float [0,1]
    Returns Var(||∇I||) over spatial pixels.
    """
    gx = cv2.Sobel(L_map, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L_map, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(mag.var())


# ─────────────────────────────────────────────────────────────────────────────
# IlluminationMetrics
# ─────────────────────────────────────────────────────────────────────────────

class IlluminationMetrics:

    def __init__(self):
        self._mean_L: List[float]     = []
        self._grad_var: List[float]   = []

    # ------------------------------------------------------------------ #
    def update(self, person_imgs: torch.Tensor):
        """person_imgs : (B, 3, H, W)  float32  [0,1]"""
        mean_L, L_maps = _rgb_to_lab_l(person_imgs.cpu())

        for i in range(len(mean_L)):
            self._mean_L.append(float(mean_L[i]))
            self._grad_var.append(_sobel_gradient_variance(L_maps[i]))

    # ------------------------------------------------------------------ #
    def compute(self) -> Dict[str, float]:
        if not self._mean_L:
            return {k: float("nan") for k in [
                "luminance_mean_global", "luminance_var_global",
                "illumination_gradient_mean", "illumination_complexity",
            ]}

        arr_L  = np.array(self._mean_L)
        arr_gv = np.array(self._grad_var)

        lum_mean = float(arr_L.mean())
        lum_var  = float(arr_L.var())
        grad_mean = float(arr_gv.mean())

        return {
            "luminance_mean_global":      lum_mean,
            "luminance_var_global":       lum_var,
            "illumination_gradient_mean": grad_mean,
            "illumination_complexity":    lum_var + grad_mean,
        }

    def reset(self):
        self._mean_L.clear()
        self._grad_var.clear()
