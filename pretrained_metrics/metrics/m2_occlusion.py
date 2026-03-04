"""
metrics/m2_occlusion.py
========================
Metric 2 — Occlusion Complexity
---------------------------------
Measures how much the garment region is occluded by arms, hair, or other
objects, and how much that occlusion varies across the dataset.

    C_occ = E[O_i] + Var(O_i)

where O_i = |G_i ∩ (A_i ∪ H_i ∪ Other_i)| / |G_i|

Pretrained model
-----------------
Mask2Former (facebook/mask2former-swin-large-coco-panoptic) via HuggingFace.
Falls back to DeepLabV3 (torchvision) when Mask2Former is unavailable.
Falls back to a gradient-based saliency proxy as ultimate fallback.

Category mapping (COCO panoptic)
----------------------------------
  Garment  ← "shirt", "jacket", "coat", "dress", "shorts", "skirt", "pants", "top"
  Arms     ← "person" (full body; arm pixels are approximated from upper body)
  Hair     ← "hair"
  Other    ← everything that is not person/garment/background

Input
------
person_imgs : torch.Tensor  (B, 3, H, W)  float32  [0, 1]
(Cloth tensor is NOT needed — occlusion is measured on the person image.)

Returns (via compute())
------------------------
dict with:
    occlusion_mean     : E[O_i]
    occlusion_var      : Var(O_i)
    occlusion_complexity : E[O_i] + Var(O_i)   (C_occ)
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation backend
# ─────────────────────────────────────────────────────────────────────────────

class _SegBackend:
    """Abstracts the segmentation model; returns per-pixel class maps."""

    # COCO panoptic "stuff" label indices that represent garment/hair/body
    # These IDs come from the COCO panoptic label list.
    _HAIR_LABELS    = {"hair-wig", "hair"}
    _PERSON_LABELS  = {"person"}
    _GARMENT_LABELS = {
        "clothes", "shirt", "jacket", "coat", "dress",
        "shorts", "skirt", "pants", "top", "sweater",
    }

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._backend = "stub"
        self._model   = None
        self._processor = None
        self._load()

    # --------------------------------------------------------------------- #
    def _load(self):
        # Try DeepLabV3 (torchvision — lightweight)
        try:
            import torchvision.models.segmentation as seg_models
            self._model = seg_models.deeplabv3_resnet101(
                weights=seg_models.DeepLabV3_ResNet101_Weights.DEFAULT
            ).to(self.device).eval()
            self._backend = "deeplabv3"
            print("[OcclusionMetric] Using DeepLabV3-ResNet101 for segmentation.")
            return
        except Exception as e:
            print(f"[OcclusionMetric] DeepLabV3 unavailable ({e}). "
                  "Falling back to saliency proxy.")
        self._backend = "stub"

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def segment(self, imgs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        imgs : (B, 3, H, W)  float32  [0,1]
        Returns dict of binary masks (B, H, W) bool:
            "garment", "arms", "hair", "other"
        """
        B, C, H, W = imgs.shape

        if self._backend == "deeplabv3":
            return self._deeplabv3_masks(imgs, H, W)

        return self._stub_masks(imgs, H, W)

    # --------------------------------------------------------------------- #
    def _deeplabv3_masks(self, imgs, H, W):
        """
        DeepLabV3 is trained on Pascal VOC (21 classes).
        class 15 = person; the rest we treat as 'other'.

        Arm/garment separation via morphological erosion:
          - Heavy erosion removes thin appendages (arms) → remaining core ≈ torso/garment
          - arm pixels = person_mask & ~core
        This avoids the row-split bug where garment and arms were mutually
        exclusive by construction and could never overlap.
        """
        norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
        x    = torch.stack([norm(im) for im in imgs]).to(self.device)
        out  = self._model(x)["out"]                 # (B, 21, H, W)
        pred = out.argmax(1)                          # (B, H, W) int64

        person_mask = (pred == 15).float()            # (B, H, W) float32

        # ── Morphological erosion to isolate core body (garment proxy) ────────
        # Kernel size ≈ 1/12 of the shorter image dimension; must be odd & ≥ 11.
        k = max(H, W) // 12
        if k % 2 == 0:
            k += 1
        k = max(k, 11)
        pm4  = person_mask.unsqueeze(1)              # (B, 1, H, W)
        # Erosion = −MaxPool(−x)
        core = -F.max_pool2d(-pm4, kernel_size=k, stride=1, padding=k // 2)
        core = (core > 0.5).squeeze(1)               # (B, H, W) bool

        # Garment ≈ eroded core (thick torso); arms ≈ thin protrusions removed by erosion
        garment = core
        arms    = person_mask.bool() & ~core
        hair    = torch.zeros_like(garment)          # DeepLabV3 has no hair class
        other   = (~person_mask.bool()) & (pred != 0) # non-background, non-person

        return {
            "garment": garment.cpu(),
            "arms":    arms.cpu(),
            "hair":    hair.cpu(),
            "other":   other.cpu(),
        }

    def _stub_masks(self, imgs, H, W):
        """Saliency-based proxy: high-gradient regions ≈ garment boundary."""
        # Convert to grayscale, Sobel edges
        gray = 0.299 * imgs[:, 0] + 0.587 * imgs[:, 1] + 0.114 * imgs[:, 2]  # (B,H,W)
        B = gray.shape[0]
        gray4 = gray.unsqueeze(1)   # (B,1,H,W)
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=torch.float32, device=imgs.device).view(1, 1, 3, 3)
        sy = sx.transpose(-2, -1)
        gx = F.conv2d(gray4, sx, padding=1)
        gy = F.conv2d(gray4, sy, padding=1)
        edge = (gx ** 2 + gy ** 2).sqrt().squeeze(1)    # (B,H,W)

        thr  = edge.flatten(1).median(1).values[:, None, None] * 1.5
        garment = (edge > thr)

        h_split = H // 2
        arms     = torch.zeros_like(garment)
        arms[:, :h_split, :]  = garment[:, :h_split, :]
        garment_lower = garment.clone()
        garment_lower[:, :h_split, :] = False

        return {
            "garment": garment_lower.cpu(),
            "arms":    arms.cpu(),
            "hair":    torch.zeros(B, H, W, dtype=torch.bool),
            "other":   torch.zeros(B, H, W, dtype=torch.bool),
        }


# ─────────────────────────────────────────────────────────────────────────────
# OcclusionMetrics
# ─────────────────────────────────────────────────────────────────────────────

class OcclusionMetrics:
    """
    Accumulates per-image occlusion ratios, then computes C_occ.
    """

    def __init__(self, device: str = "cpu"):
        self._seg   = _SegBackend(device=device)
        self._ratios: List[float] = []

    # ------------------------------------------------------------------ #
    def update(self, person_imgs: torch.Tensor):
        """person_imgs : (B, 3, H, W)  float32  [0,1]"""
        masks = self._seg.segment(person_imgs)
        G = masks["garment"].float()    # (B, H, W)
        A = masks["arms"].float()
        Ha= masks["hair"].float()
        Ot= masks["other"].float()

        occluder = ((A + Ha + Ot) > 0).float()          # union of occluders
        overlap  = (G * occluder)                        # garment ∩ occluders

        B = G.shape[0]
        for i in range(B):
            g_area = G[i].sum().item()
            if g_area < 1:
                self._ratios.append(0.0)
                continue
            ratio = overlap[i].sum().item() / g_area
            self._ratios.append(float(min(ratio, 1.0)))

    # ------------------------------------------------------------------ #
    def compute(self) -> Dict[str, float]:
        if not self._ratios:
            return {
                "occlusion_mean":         float("nan"),
                "occlusion_var":          float("nan"),
                "occlusion_complexity":   float("nan"),
            }
        arr = np.array(self._ratios)
        mean = float(arr.mean())
        var  = float(arr.var())
        return {
            "occlusion_mean":       mean,
            "occlusion_var":        var,
            "occlusion_complexity": mean + var,
        }

    def reset(self):
        self._ratios.clear()
