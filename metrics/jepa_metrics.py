"""
metrics/jepa_metrics.py
------------------------
JEPA (Joint Embedding Predictive Architecture) Embedding Metrics
-----------------------------------------------------------------
We use a pretrained I-JEPA (Image JEPA) or VideoMAE-based encoder
to extract context and target patch embeddings, then measure:

  1. Embedding Prediction Error (EPE):
        Mean squared error between the predicted patch embedding (from the
        JEPA predictor operating on the person image context) and the actual
        target embedding extracted from the try-on result.

  2. Embedding Trace (Tr(Σ)):
        Trace of the covariance matrix of the target embeddings, i.e., the total
        variance / spread in the embedding space.  A high trace indicates
        high diversity / complexity of the predictions.

Architecture used by default
-----------------------------
We load the ViT-B/16 Vision Transformer (pretrained on ImageNet-1k via timm).
In a real JEPA setup you'd use official I-JEPA weights; here we use ViT as a
proxy encoder and a linear layer as the predictor.

Swap in official I-JEPA weights by replacing the encoder load in _load_model().
"""

from __future__ import annotations

import math
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torchvision.transforms as T


# ─────────────────────────────────────────────────────────────────────────────
# ViT-based JEPA surrogate encoder
# ─────────────────────────────────────────────────────────────────────────────

class JEPAEncoder(nn.Module):
    """
    Wraps a pretrained ViT to act as a JEPA-style context / target encoder.
    Returns CLS token + patch tokens: (B, N+1, D).
    """

    def __init__(self, model_name: str = "vit_base_patch16_224", device: str = "cpu"):
        super().__init__()
        self.device = device
        try:
            import timm
            self.backbone = timm.create_model(
                model_name, pretrained=True, num_classes=0
            ).to(device).eval()
            self.embed_dim = self.backbone.embed_dim
            self._available = True
        except Exception as e:
            print(f"[JEPAEncoder] Could not load ViT ({e}). Using random projection.")
            self._available = False
            self.embed_dim = 768
            self.backbone = None

        # Simple linear predictor (context → target prediction)
        self.predictor = nn.Linear(self.embed_dim, self.embed_dim).to(device)

    @torch.no_grad()
    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: (B, C, H, W) in [0,1].
        Returns (B, D) global average-pooled embedding.
        """
        imgs = imgs.to(self.device)
        # Normalise to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        imgs = (imgs - mean) / std

        if self._available:
            # Resize to model input size
            imgs = T.functional.resize(imgs, [224, 224])
            feats = self.backbone.forward_features(imgs)   # (B, N+1, D)
            # Global average pool over patch tokens (skip CLS at index 0)
            return feats[:, 1:, :].mean(dim=1)             # (B, D)
        else:
            # Fallback: random projection
            B, C, H, W = imgs.shape
            flat = imgs.view(B, -1)[:, :self.embed_dim]
            pad  = max(0, self.embed_dim - flat.shape[1])
            flat = torch.cat([flat, flat.new_zeros(B, pad)], dim=1)
            return flat


# ─────────────────────────────────────────────────────────────────────────────
# JEPA Metrics class
# ─────────────────────────────────────────────────────────────────────────────

class JEPAMetrics:
    """
    Computes:
      - Embedding Prediction Error (EPE) per image
      - Trace of the embedding covariance (trace_sigma) per dataset

    Usage:
        jepa = JEPAMetrics(device='cuda')
        for batch in loader:
            person, pred_tryon, gt = batch
            epe_list  = jepa.compute_epe_batch(person, pred_tryon)   # per image
            jepa.update_embeddings(pred_tryon)                        # accumulate for trace
        trace_sigma = jepa.compute_embedding_trace()
        jepa.reset()
    """

    def __init__(self, device: str = "cpu"):
        self.encoder  = JEPAEncoder(device=device)
        self.predictor = self.encoder.predictor
        self.device    = device
        self._all_embeddings: List[torch.Tensor] = []

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def compute_epe_batch(
        self, person: torch.Tensor, pred_tryon: torch.Tensor
    ) -> List[float]:
        """
        Embedding Prediction Error:
          context_emb = encoder(person)
          predicted_target = predictor(context_emb)
          actual_target    = encoder(pred_tryon)
          EPE = MSE(predicted_target, actual_target) per image.

        Returns:
            List[float] of per-image EPE values.
        """
        ctx_emb   = self.encoder.encode(person)         # (B, D)
        pred_emb  = self.predictor(ctx_emb)              # (B, D)  ← predicted target
        tgt_emb   = self.encoder.encode(pred_tryon)     # (B, D)  ← actual embedding

        mse = ((pred_emb - tgt_emb) ** 2).mean(dim=1)   # (B,)
        return mse.cpu().tolist()

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def update_embeddings(self, imgs: torch.Tensor):
        """Accumulate target embeddings from try-on results for trace computation."""
        emb = self.encoder.encode(imgs)    # (B, D)
        self._all_embeddings.append(emb.cpu())

    # ------------------------------------------------------------------ #
    def compute_embedding_trace(self) -> float:
        """
        Trace of covariance matrix Σ of accumulated target embeddings.
        Tr(Σ) = sum of variances across embedding dimensions.
        High trace ⟹ high spread/diversity in the embedding space.
        """
        if not self._all_embeddings:
            return float("nan")

        E = torch.cat(self._all_embeddings, dim=0)   # (N, D)
        E = E - E.mean(dim=0, keepdim=True)           # centre
        cov = (E.T @ E) / max(E.shape[0] - 1, 1)     # (D, D)
        trace = cov.diagonal().sum().item()
        return float(trace)

    def reset(self):
        self._all_embeddings = []
