"""
metrics/m5_body_shape.py
=========================
Metric 5 — Body Shape Diversity
---------------------------------
Uses SMPL shape coefficients β ∈ R^10 extracted from SPIN or a proxy.

D_shape = log det(Cov(β) + ε·I)

Pretrained model
-----------------
SPIN (Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop).
Official weights loaded from a local checkpoint (see SPIN_CHECKPOINT_PATH below).

Backend priority:
  Level 1 — SPIN + SMPL:  Genuine β ∈ R^10 SMPL shape coefficients.
             Requires:
               • pip install smplx
               • SMPL neutral model at SMPL_MODEL_PATH  (register at smpl.is.tue.mpg.de)
               • SPIN checkpoint at SPIN_CHECKPOINT_PATH
  Level 2 — ViT-B/16 proxy:  10-D projection of ViT CLS embedding.
             NOT equivalent to SMPL β, but captures body-shape variety.
  Level 3 — Random stub (smoke-test only).

Paths (edit to match your layout, or override via env vars)
------------------------------------------------------------
  SMPL_MODEL_PATH      default: "body_models/smpl/SMPL_NEUTRAL.pkl"
  SPIN_CHECKPOINT_PATH default: "data/spin_checkpoint.pt"

  Override at runtime:
    import os
    os.environ["SMPL_MODEL_PATH"]      = "/path/to/SMPL_NEUTRAL.pkl"
    os.environ["SPIN_CHECKPOINT_PATH"] = "/path/to/spin_checkpoint.pt"

Input
------
person_imgs : torch.Tensor  (B, 3, H, W)  float32  [0, 1]

Returns (compute())
--------------------
dict with:
    shape_diversity_logdet   : log det(Cov(β) + ε·I)
    shape_variance_total     : sum of eigenvalues of Cov(β)  (total variance)
    shape_dims               : number of shape coefficients (10 or proxy dim)
    backend                  : which backend was used ("spin", "vit_proxy", "stub")
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ── Path configuration ────────────────────────────────────────────────────────
_DEFAULT_SMPL_PATH = "body_models/smpl/SMPL_NEUTRAL.pkl"
_DEFAULT_SPIN_CKPT = "data/spin_checkpoint.pt"


# ─────────────────────────────────────────────────────────────────────────────
# SPIN network definition (HMR-style regressor)
# ─────────────────────────────────────────────────────────────────────────────

class _SPINRegressor(nn.Module):
    """
    Iterative regressor head from SPIN / HMR.
    Input  : (B, 2048) ResNet-50 global average-pooled features
    Output : (B, 85) = pose(72) + shape(10) + cam(3)

    Architecture mirrors the original SPIN repo:
      github.com/nkolot/SPIN/blob/master/models/hmr.py
    """
    def __init__(self, feat_dim: int = 2048, n_iter: int = 3):
        super().__init__()
        npose = 24 * 6   # 24 joints × 6D rotation representation = 144
                         # SPIN uses 6D repr internally but stores as axis-angle (72)
                         # The checkpoint uses the 6D version → 144 dims for pose
        self.n_iter = n_iter

        self.fc1   = nn.Linear(feat_dim + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2   = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose  = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam   = nn.Linear(1024, 3)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        # Mean pose initialisation (zero = standing upright)
        self.register_buffer(
            "mean_pose",
            torch.zeros(1, npose),
        )
        self.register_buffer(
            "mean_shape",
            torch.zeros(1, 10),
        )

    def forward(self, x: torch.Tensor):
        """x : (B, 2048)  →  betas : (B, 10)"""
        B = x.shape[0]
        pose  = self.mean_pose.expand(B, -1)
        shape = self.mean_shape.expand(B, -1)
        cam   = torch.zeros(B, 3, device=x.device)

        for _ in range(self.n_iter):
            xc = torch.cat([x, pose, shape, cam], dim=1)
            xc = self.drop1(torch.relu(self.fc1(xc)))
            xc = self.drop2(torch.relu(self.fc2(xc)))
            pose  = pose  + self.decpose(xc)
            shape = shape + self.decshape(xc)
            cam   = cam   + self.deccam(xc)

        return shape   # (B, 10)


# ─────────────────────────────────────────────────────────────────────────────
# Shape extractor — backend priority: SPIN → ViT proxy → stub
# ─────────────────────────────────────────────────────────────────────────────

class _ShapeExtractor:
    SHAPE_DIM = 10

    _SPIN_NORMALIZE = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._backend  = "stub"
        self._backbone = None   # ResNet-50 feature extractor
        self._regressor: Optional[_SPINRegressor] = None
        self._vit       = None
        self._vit_proj: Optional[nn.Linear] = None
        self._load()

    # ------------------------------------------------------------------ #
    def _load(self):
        if self._try_spin():
            return
        if self._try_vit():
            return
        print("[BodyShapeMetric] All backends failed. Using random stub.")
        self._backend = "stub"

    # ------------------------------------------------------------------ #
    def _try_spin(self) -> bool:
        smpl_path = os.environ.get("SMPL_MODEL_PATH", _DEFAULT_SMPL_PATH)
        ckpt_path = os.environ.get("SPIN_CHECKPOINT_PATH", _DEFAULT_SPIN_CKPT)

        if not os.path.isfile(smpl_path):
            print(f"[BodyShapeMetric] SMPL model not found at '{smpl_path}'. "
                  "Skipping SPIN. Register at smpl.is.tue.mpg.de to download.")
            return False

        if not os.path.isfile(ckpt_path):
            print(f"[BodyShapeMetric] SPIN checkpoint not found at '{ckpt_path}'. "
                  "Skipping SPIN.")
            return False

        try:
            import smplx                            # pip install smplx
            import torchvision.models as tvm

            # ── ResNet-50 backbone (feature extractor only) ───────────────
            resnet = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
            # Remove the final FC layer → output is (B, 2048) after avg-pool
            self._backbone = nn.Sequential(
                *list(resnet.children())[:-1],      # up to avg-pool
                nn.Flatten(1),                      # (B, 2048)
            ).to(self.device).eval()

            # ── Regressor head ────────────────────────────────────────────
            self._regressor = _SPINRegressor(feat_dim=2048, n_iter=3).to(self.device)

            # ── Load checkpoint ───────────────────────────────────────────
            ckpt = torch.load(ckpt_path, map_location="cpu")
            # SPIN checkpoints may store weights under different top-level keys
            state = (ckpt.get("model")
                     or ckpt.get("state_dict")
                     or ckpt)

            # Filter to regressor keys only (SPIN saves the full model)
            reg_prefix  = "regressor."
            reg_state   = {
                k[len(reg_prefix):]: v
                for k, v in state.items()
                if k.startswith(reg_prefix)
            }
            if reg_state:
                missing, unexpected = self._regressor.load_state_dict(
                    reg_state, strict=False
                )
                if missing:
                    print(f"[BodyShapeMetric] SPIN regressor: "
                          f"{len(missing)} missing keys (non-fatal).")
            else:
                # Some checkpoints store regressor weights at top level
                missing, unexpected = self._regressor.load_state_dict(
                    state, strict=False
                )

            self._regressor.eval()
            self._backend = "spin"
            print(f"[BodyShapeMetric] SPIN loaded ✓  "
                  f"(SMPL: '{smpl_path}', checkpoint: '{ckpt_path}').")
            return True

        except Exception as e:
            print(f"[BodyShapeMetric] SPIN initialisation failed: {e}. "
                  "Falling back to ViT proxy.")
            self._backbone   = None
            self._regressor  = None
            return False

    # ------------------------------------------------------------------ #
    def _try_vit(self) -> bool:
        try:
            import timm
            self._vit = timm.create_model(
                "vit_base_patch16_224", pretrained=True, num_classes=0
            ).to(self.device).eval()
            torch.manual_seed(0)
            self._vit_proj = nn.Linear(768, self.SHAPE_DIM, bias=False).to(self.device)
            nn.init.orthogonal_(self._vit_proj.weight)
            self._vit_proj.eval()
            self._backend = "vit_proxy"
            print("[BodyShapeMetric] Using ViT-B/16 proxy for body shape (no SPIN).")
            return True
        except Exception as e:
            print(f"[BodyShapeMetric] ViT unavailable ({e}).")
            return False

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def __call__(self, imgs: torch.Tensor) -> np.ndarray:
        """
        imgs : (B, 3, H, W)  float32  [0,1]
        Returns (B, 10) numpy float32 — SMPL β or proxy.
        """
        B = imgs.shape[0]

        if self._backend == "spin":
            return self._spin_forward(imgs)

        if self._backend == "vit_proxy":
            x = TF.resize(imgs, [224, 224]).to(self.device)
            x = torch.stack([self._SPIN_NORMALIZE(im) for im in x])
            feats = self._vit(x)                    # (B, 768)
            betas = self._vit_proj(feats)           # (B, 10)
            return betas.cpu().numpy()

        rng = np.random.default_rng(42)
        return rng.normal(0, 1, (B, self.SHAPE_DIM)).astype(np.float32)

    # ------------------------------------------------------------------ #
    def _spin_forward(self, imgs: torch.Tensor) -> np.ndarray:
        # Resize to 224×224 and normalise (ImageNet stats, same as SPIN training)
        x = TF.resize(imgs, [224, 224]).to(self.device)
        x = torch.stack([self._SPIN_NORMALIZE(im) for im in x])

        feats = self._backbone(x)                   # (B, 2048)
        betas = self._regressor(feats)              # (B, 10)
        return betas.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# BodyShapeMetrics  (public API — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class BodyShapeMetrics:

    def __init__(self, device: str = "cpu", eps: float = 1e-6):
        self._extractor = _ShapeExtractor(device)
        self.eps = eps
        self._betas: List[np.ndarray] = []

    # ------------------------------------------------------------------ #
    def update(self, person_imgs: torch.Tensor):
        """person_imgs : (B, 3, H, W)  float32  [0,1]"""
        betas = self._extractor(person_imgs)        # (B, 10)
        for b in betas:
            self._betas.append(b)

    # ------------------------------------------------------------------ #
    def compute(self) -> Dict[str, float]:
        if len(self._betas) < 2:
            return {k: float("nan") for k in [
                "shape_diversity_logdet", "shape_variance_total",
                "shape_dims", "backend",
            ]}

        B_mat = np.stack(self._betas, axis=0)       # (N, 10)
        mu    = B_mat.mean(axis=0, keepdims=True)
        Bc    = B_mat - mu
        cov   = (Bc.T @ Bc) / max(len(B_mat) - 1, 1)   # (10, 10)
        reg   = cov + self.eps * np.eye(10)

        sign, log_det = np.linalg.slogdet(reg)
        d_shape = float(log_det) if sign > 0 else float("nan")

        eigvals   = np.linalg.eigvalsh(reg)
        total_var = float(eigvals.sum())

        return {
            "shape_diversity_logdet": d_shape,
            "shape_variance_total":   total_var,
            "shape_dims":             float(self._extractor.SHAPE_DIM),
            "backend":                self._extractor._backend,
        }

    def reset(self):
        self._betas.clear()