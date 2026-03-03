"""
metrics/vlm_score.py
=====================
Multi-Dimensional VLM Plausibility Score for Virtual Try-On.

Four targeted sub-scores (all 1–10 scale, higher = better)
-----------------------------------------------------------
S1  Garment Fidelity
    How accurately is the original garment texture, colour, and pattern
    preserved on the person?  Seams, logos, and fine details matter.

S2  Geometric Naturalness
    Does the garment fabric drape, fold, and deform the way real clothing
    would?  No unnatural stretching, clipping, or rigidity.

S3  Identity & Body Preservation
    Has the person's face, skin tone, hair, body proportions, and pose
    been fully preserved — as if they were photographed wearing the outfit?

S4  In-the-Wild Scene Coherence
    Does the result look believable in its real-world context (lighting,
    shadows, background, viewpoint)?  Key for in-the-wild / unconstrained
    try-on images where the background is diverse, not a studio backdrop.

Weighted average
----------------
  VLM_score = w1*S1 + w2*S2 + w3*S3 + w4*S4

Default weights (tunable via constructor):
  w1=0.30  (garment fidelity   — most critical for try-on quality)
  w2=0.25  (geometric realism)
  w3=0.25  (identity preservation)
  w4=0.20  (scene coherence    — extra credit for in-the-wild)

Backend
-------
Primary  : BLIP-2 (Salesforce/blip2-opt-2.7b via HuggingFace Transformers)
Fallback : InstructBLIP if BLIP-2 fails to load
Stub     : neutral 5.0 per sub-score if no VLM can be loaded

Each sub-score is obtained with a fresh prompting call so the VLM gives
focused answers rather than a single coarse rating.

Usage
-----
    from metrics.vlm_score import VLMScoreMetric

    metric = VLMScoreMetric(device=\"cuda\")

    # evaluate.py per-batch call:
    detailed = metric.compute_batch(pred_tensor)
    # Returns list[dict]:
    # [{'s1': 8.0, 's2': 7.5, 's3': 9.0, 's4': 6.0, 'vlm_score': 7.75}, ...]

    # Back-compat: scalar list
    scalars = metric.compute_batch_scalar(pred_tensor)
    # Returns list[float]: [7.75, ...]
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import torch
from PIL import Image
import torchvision.transforms.functional as TF


# ─────────────────────────────────────────────────────────────────────────────
# Sub-score prompts
# ─────────────────────────────────────────────────────────────────────────────

_PROMPTS: List[Tuple[str, str]] = [
    (
        "s1",
        "Look at this virtual try-on image. On a scale from 1 to 10, how well "
        "are the garment's original texture, colour, pattern, seams, and fine "
        "details preserved on the person? 1 = heavily degraded or wrong, "
        "10 = perfectly faithful to the original garment. "
        "Reply with a single integer only.",
    ),
    (
        "s2",
        "Look at this virtual try-on image. On a scale from 1 to 10, how "
        "geometric naturalness: does the fabric drape, fold, and deform the "
        "way real clothing would on a person? 1 = unrealistic stiff or warped "
        "garment, 10 = completely natural 3-D cloth behaviour. "
        "Reply with a single integer only.",
    ),
    (
        "s3",
        "Look at this virtual try-on image. On a scale from 1 to 10, how well "
        "is the person's identity preserved — face, skin tone, hair, body "
        "proportions, and pose — compared to a real photo of the same person? "
        "1 = severe artefacts or identity loss, 10 = indistinguishable from "
        "a real photograph. Reply with a single integer only.",
    ),
    (
        "s4",
        "Look at this virtual try-on image taken in a real-world (in-the-wild) "
        "setting. On a scale from 1 to 10, how coherent is the result with its "
        "scene — lighting, shadows, background, and viewpoint? "
        "1 = clearly composited / lighting mismatch, "
        "10 = perfectly photorealistic and scene-consistent. "
        "Reply with a single integer only.",
    ),
]

# Default sub-score weights (must sum to 1.0)
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "s1": 0.30,   # garment fidelity      — most critical
    "s2": 0.25,   # geometric naturalness
    "s3": 0.25,   # identity preservation
    "s4": 0.20,   # scene coherence (in-the-wild)
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: clamp-parse a float from VLM text
# ─────────────────────────────────────────────────────────────────────────────

def _parse_score(text: str, fallback: float = 5.0) -> float:
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if nums:
        return float(max(1.0, min(10.0, float(nums[0]))))
    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# VLMScoreMetric
# ─────────────────────────────────────────────────────────────────────────────

class VLMScoreMetric:
    """
    Multi-dimensional VLM plausibility scorer for virtual try-on.

    Parameters
    ----------
    model_name  : HuggingFace model ID (default: BLIP-2 opt-2.7b).
    device      : 'cpu' | 'cuda' | 'cuda:N'.
    weights     : Dict mapping sub-score keys to weights.
                  Keys: 's1', 's2', 's3', 's4'.  Must sum to 1.0.
    vlm_batch   : Max images per VLM forward pass (reduce if OOM).
    neutral     : Fallback score per sub-dimension when VLM unavailable.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        device: str = "cpu",
        weights: Dict[str, float] | None = None,
        vlm_batch: int = 2,
        neutral: float = 5.0,
    ):
        self.device    = device
        self.weights   = weights or dict(_DEFAULT_WEIGHTS)
        self.vlm_batch = vlm_batch
        self.neutral   = neutral

        # Validate weights
        wsum = sum(self.weights.values())
        if abs(wsum - 1.0) > 1e-4:
            raise ValueError(
                f"VLMScoreMetric weights must sum to 1.0, got {wsum:.4f}"
            )

        self._model     = None
        self._processor = None
        self._backend   = "stub"
        self._load_model(model_name)

    # ── Model loading ────────────────────────────────────────────────────── #

    def _load_model(self, model_name: str):
        # ── Try BLIP-2 ────────────────────────────────────────────────────
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            print(f"[VLMScore] Loading BLIP-2: {model_name} …")
            self._processor = Blip2Processor.from_pretrained(model_name)
            self._model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                device_map=self.device,
            )
            self._model.eval()
            self._backend = "blip2"
            print("[VLMScore] BLIP-2 loaded.")
            return
        except Exception as e:
            print(f"[VLMScore] BLIP-2 unavailable ({e}). Trying InstructBLIP …")

        # ── Try InstructBLIP fallback ────────────────────────────────────
        try:
            from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
            fb = "Salesforce/instructblip-vicuna-7b"
            print(f"[VLMScore] Loading InstructBLIP: {fb} …")
            self._processor = InstructBlipProcessor.from_pretrained(fb)
            self._model = InstructBlipForConditionalGeneration.from_pretrained(
                fb,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                device_map=self.device,
            )
            self._model.eval()
            self._backend = "instructblip"
            print("[VLMScore] InstructBLIP loaded.")
            return
        except Exception as e:
            print(f"[VLMScore] InstructBLIP unavailable ({e}). Using stub.")

        self._backend = "stub"

    # ── Internal VLM call for a single sub-score ─────────────────────────── #

    @torch.no_grad()
    def _score_sub(self, pil_images: List[Image.Image], prompt: str) -> List[float]:
        """Run one prompt against a mini-batch of PIL images → list of scores."""
        if self._backend == "stub":
            return [self.neutral] * len(pil_images)

        try:
            inputs = self._processor(
                images=pil_images,
                text=[prompt] * len(pil_images),
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            generated = self._model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
            )
            texts = self._processor.batch_decode(generated, skip_special_tokens=True)
            return [_parse_score(t, self.neutral) for t in texts]

        except Exception as e:
            print(f"[VLMScore] Inference error ({e}). Using neutral score.")
            return [self.neutral] * len(pil_images)

    # ── Public batch scoring ─────────────────────────────────────────────── #

    def compute_batch(self, pred: torch.Tensor) -> List[Dict[str, float]]:
        """
        Score a batch of try-on result images on all 4 sub-dimensions and
        compute the weighted average VLM score.

        Parameters
        ----------
        pred : torch.Tensor  (B, 3, H, W) float32 in [0, 1]

        Returns
        -------
        list[dict] — one dict per image::

            {
              's1'        : float  garment fidelity        [1–10]
              's2'        : float  geometric naturalness   [1–10]
              's3'        : float  identity preservation   [1–10]
              's4'        : float  scene coherence         [1–10]
              'vlm_score' : float  weighted average        [1–10]
            }
        """
        B       = pred.shape[0]
        pil_list = [TF.to_pil_image(img.clamp(0, 1).cpu()) for img in pred]

        # Collect all sub-scores: {key: [score_per_image]}
        sub_scores: Dict[str, List[float]] = {}
        for key, prompt in _PROMPTS:
            all_s: List[float] = []
            for start in range(0, B, self.vlm_batch):
                chunk = pil_list[start: start + self.vlm_batch]
                all_s.extend(self._score_sub(chunk, prompt))
            sub_scores[key] = all_s

        # Assemble per-image dicts
        results: List[Dict[str, float]] = []
        for i in range(B):
            row: Dict[str, float] = {k: sub_scores[k][i] for k in sub_scores}
            row["vlm_score"] = sum(
                self.weights[k] * row[k] for k in self.weights
            )
            results.append(row)

        return results

    def compute_batch_scalar(self, pred: torch.Tensor) -> List[float]:
        """
        Convenience wrapper — returns only the weighted VLM score per image.

        Back-compat alias so evaluate.py can call this with no changes.
        """
        return [r["vlm_score"] for r in self.compute_batch(pred)]

    # ── Describe the scoring ─────────────────────────────────────────────── #

    def describe(self) -> str:
        lines = [
            f"VLMScoreMetric  backend={self._backend}",
            f"  Sub-dimensions & weights:",
            f"    S1 Garment Fidelity       w={self.weights['s1']:.2f}",
            f"    S2 Geometric Naturalness  w={self.weights['s2']:.2f}",
            f"    S3 Identity Preservation  w={self.weights['s3']:.2f}",
            f"    S4 Scene Coherence        w={self.weights['s4']:.2f}",
            f"  Weighted sum → VLM_score ∈ [1, 10]  (higher = better)",
        ]
        return "\n".join(lines)
