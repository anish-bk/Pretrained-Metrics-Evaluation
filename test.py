"""
test.py
========
Model-load verification test suite.
Tests that every pretrained model used in the pipeline can be loaded
and that a single forward pass produces a sane output shape.

Run:
    python test.py
    python test.py --device cuda
    python test.py --skip detr blip2   # skip slow / network-heavy models
    python test.py --quick             # skip all models > 1 GB
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pretrained_metrics"))


# ─────────────────────────────────────────────────────────────────────────────
# Test registry
# ─────────────────────────────────────────────────────────────────────────────

TESTS: List[Dict] = []   # populated by @register


def register(name: str, heavy: bool = False):
    """Decorator to register a test function."""
    def _wrap(fn: Callable):
        TESTS.append({"name": name, "fn": fn, "heavy": heavy})
        return fn
    return _wrap


# ─────────────────────────────────────────────────────────────────────────────
# Shared dummy tensors
# ─────────────────────────────────────────────────────────────────────────────

def _rand(B=2, C=3, H=256, W=192, device="cpu") -> torch.Tensor:
    return torch.rand(B, C, H, W, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

@register("PyTorch + CUDA")
def test_torch(device):
    v = torch.__version__
    assert len(v) > 0, "torch version empty"
    if device == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but not available"
    t = torch.rand(4, 4, device=device)
    assert t.shape == (4, 4)
    return f"torch {v}  |  device={device}"


@register("torchvision")
def test_torchvision(device):
    import torchvision
    v = torchvision.__version__
    from torchvision.models import resnet18, ResNet18_Weights
    m = resnet18(weights=None).to(device).eval()
    with torch.no_grad():
        out = m(_rand(2, 3, 224, 224, device))
    assert out.shape == (2, 1000)
    return f"torchvision {v}  |  ResNet18 forward OK"


@register("timm (ViT-B/16)")
def test_timm(device):
    import timm
    v = timm.__version__
    m = timm.create_model(
        "vit_base_patch16_224", pretrained=False, num_classes=0
    ).to(device).eval()
    with torch.no_grad():
        out = m(torch.rand(2, 3, 224, 224, device=device))
    assert out.shape[0] == 2
    return f"timm {v}  |  ViT-B/16 output: {tuple(out.shape)}"


@register("DeepLabV3 (person segmentation)")
def test_deeplabv3(device):
    from torchvision.models.segmentation import (
        deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
    )
    m = deeplabv3_resnet101(
        weights=DeepLabV3_ResNet101_Weights.DEFAULT
    ).to(device).eval()
    with torch.no_grad():
        out = m(_rand(2, 3, 520, 390, device))["out"]
    assert out.shape[:2] == (2, 21), f"unexpected shape {out.shape}"
    return f"DeepLabV3-ResNet101 output: {tuple(out.shape)}"


@register("M1 — Pose (_KeypointExtractor)")
def test_pose_extractor(device):
    from pretrained_metrics.metrics.m1_pose import _KeypointExtractor, _normalise_pose
    kp = _KeypointExtractor(device)
    imgs = _rand(2, 3, 256, 192, device)
    raw = kp(imgs)
    assert raw.shape == (2, 17, 2), f"unexpected kp shape {raw.shape}"
    normed, valid = _normalise_pose(raw)
    assert normed.shape == (2, 17, 2)
    return f"Keypoint shape: {raw.shape}  |  backend: {kp._backend}"


@register("M2 — Occlusion (_SegBackend)")
def test_seg_backend(device):
    from pretrained_metrics.metrics.m2_occlusion import _SegBackend
    sg = _SegBackend(device)
    imgs = _rand(2, 3, 256, 192)
    masks = sg.segment(imgs)
    assert "garment" in masks and "arms" in masks
    assert masks["garment"].shape == (2, 256, 192)
    return f"SegBackend backend: {sg._backend}  |  garment mask: {masks['garment'].shape}"


@register("M3 — Background (PersonSegmenter + ObjectDetector)")
def test_background(device):
    from pretrained_metrics.metrics.m3_background import _PersonSegmenter, _ObjectDetector
    ps  = _PersonSegmenter(device)
    od  = _ObjectDetector(device)
    imgs = _rand(2, 3, 256, 192)
    pmasks = ps(imgs)
    assert pmasks.shape == (2, 256, 192)
    counts = od.count_objects(imgs, pmasks)
    assert len(counts) == 2
    return f"PersonSeg backend: {ps._model is not None}  |  ObjDet backend: {od._backend}"


@register("M4 — Illumination (LAB + Sobel, no model)")
def test_illumination(device):
    from pretrained_metrics.metrics.m4_illumination import _rgb_to_lab_l, _sobel_gradient_variance
    imgs = _rand(2, 3, 128, 96)
    mean_L, L_maps = _rgb_to_lab_l(imgs)
    assert mean_L.shape == (2,)
    gv = _sobel_gradient_variance(L_maps[0])
    assert np.isfinite(gv)
    return f"LAB-L mean: {mean_L.mean():.4f}  |  grad_var: {gv:.6f}"


@register("M5 — Body Shape (_ShapeExtractor)")
def test_shape(device):
    from pretrained_metrics.metrics.m5_body_shape import _ShapeExtractor
    se   = _ShapeExtractor(device)
    imgs = _rand(2, 3, 256, 192)
    betas = se(imgs)
    assert betas.shape == (2, 10), f"unexpected shape {betas.shape}"
    return f"ShapeExtractor backend: {se._backend}  |  β shape: {betas.shape}"


@register("M6 — Appearance (_FaceEmbedder)")
def test_face_embedder(device):
    from pretrained_metrics.metrics.m6_appearance import _FaceEmbedder
    fe   = _FaceEmbedder(device)
    imgs = _rand(2, 3, 256, 192)
    embs = fe(imgs)
    assert embs.shape[0] == 2, f"wrong batch dim {embs.shape}"
    return f"FaceEmbedder backend: {fe._backend}  |  embed shape: {embs.shape}"


@register("M7 — Garment Texture (_GarmentEncoder)")
def test_garment_encoder(device):
    from pretrained_metrics.metrics.m7_garment_texture import _GarmentEncoder
    ge   = _GarmentEncoder(device)
    imgs = _rand(2, 3, 256, 192)
    embs = ge(imgs)
    assert embs.shape[0] == 2
    return f"GarmentEncoder backend: {ge._backend}  |  embed shape: {embs.shape}"


@register("Full PoseMetrics.update+compute")
def test_pose_metrics(device):
    from pretrained_metrics.metrics.m1_pose import PoseMetrics
    m = PoseMetrics(device)
    for _ in range(3):
        m.update(_rand(4, 3, 256, 192))
    r = m.compute()
    assert "pose_diversity" in r
    return f"pose_diversity={r['pose_diversity']:.4f}  |  artic={r['pose_artic_complexity']:.4f}"


@register("Full OcclusionMetrics.update+compute")
def test_occ_metrics(device):
    from pretrained_metrics.metrics.m2_occlusion import OcclusionMetrics
    m = OcclusionMetrics(device)
    for _ in range(3):
        m.update(_rand(4, 3, 256, 192))
    r = m.compute()
    assert "occlusion_complexity" in r
    return f"occlusion_complexity={r['occlusion_complexity']:.4f}"


@register("Full BackgroundMetrics.update+compute")
def test_bg_metrics(device):
    from pretrained_metrics.metrics.m3_background import BackgroundMetrics
    m = BackgroundMetrics(device)
    for _ in range(2):
        m.update(_rand(4, 3, 256, 192))
    r = m.compute()
    assert "bg_entropy_mean" in r
    return f"bg_entropy_mean={r['bg_entropy_mean']:.4f}"


@register("Full IlluminationMetrics.update+compute")
def test_illum_metrics(device):
    from pretrained_metrics.metrics.m4_illumination import IlluminationMetrics
    m = IlluminationMetrics()
    for _ in range(3):
        m.update(_rand(4, 3, 256, 192))
    r = m.compute()
    assert "illumination_complexity" in r
    return f"illumination_complexity={r['illumination_complexity']:.6f}"


@register("Full BodyShapeMetrics.update+compute")
def test_shape_metrics(device):
    from pretrained_metrics.metrics.m5_body_shape import BodyShapeMetrics
    m = BodyShapeMetrics(device)
    for _ in range(3):
        m.update(_rand(4, 3, 256, 192))
    r = m.compute()
    assert "shape_diversity_logdet" in r
    return f"shape_diversity_logdet={r['shape_diversity_logdet']:.4f}"


@register("Full AppearanceMetrics.update+compute")
def test_appear_metrics(device):
    from pretrained_metrics.metrics.m6_appearance import AppearanceMetrics
    m = AppearanceMetrics(device)
    for _ in range(3):
        m.update(_rand(4, 3, 256, 192))
    r = m.compute()
    assert "appearance_diversity_mean" in r
    return f"appearance_diversity_mean={r['appearance_diversity_mean']:.4f}"


@register("Full GarmentTextureMetrics.update+compute")
def test_garment_metrics(device):
    from pretrained_metrics.metrics.m7_garment_texture import GarmentTextureMetrics
    m = GarmentTextureMetrics(device)
    for _ in range(3):
        m.update(_rand(4, 3, 256, 192))
    r = m.compute()
    assert "garment_diversity_logdet" in r
    return f"garment_diversity_logdet={r['garment_diversity_logdet']:.4f}"


@register("UnifiedComplexityIndex")
def test_unified_index(device):
    from pretrained_metrics.metrics.unified_index import UnifiedComplexityIndex
    uci = UnifiedComplexityIndex()
    fake_metrics = {
        "pose_diversity":            -14.0,
        "pose_artic_complexity":       0.32,
        "occlusion_complexity":        0.16,
        "illumination_complexity":     0.06,
        "shape_diversity_logdet":    -28.0,
        "appearance_diversity_mean":   0.42,
        "garment_diversity_logdet":  -18.0,
    }
    uci.add_dataset("test_dataset", fake_metrics)
    scores = uci.compute_scores()
    assert len(scores) == 1
    s = scores[0]["unified_score"]
    assert isinstance(s, float)
    return f"unified_score={s:.4f}"


@register("EDA synthetic dry-run (no dataset needed)", heavy=False)
def test_eda_dryrun(device):
    """Import all EDA plot modules and run the synthetic path."""
    from EDA.run_eda import _make_synthetic_data, run_all_plots
    import tempfile
    data = {
        "synth_A": _make_synthetic_data(30, seed=0),
        "synth_B": _make_synthetic_data(30, seed=1),
    }
    with tempfile.TemporaryDirectory() as tmp:
        run_all_plots(data, out_root=tmp,
                      skip_figures=[], no_pairplot=True)
    return "EDA dry-run: all 8 plot families generated (tmp dir, auto-cleaned)"


@register("DETR object detector", heavy=True)
def test_detr(device):
    from transformers import DetrImageProcessor, DetrForObjectDetection
    proc  = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50"
    ).to(device).eval()
    import torchvision.transforms.functional as TF
    from PIL import Image
    pil = Image.fromarray(np.random.randint(0, 255, (256, 192, 3), dtype=np.uint8))
    inp = proc(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inp)
    n = (out.logits.softmax(-1)[0, :, :-1].max(-1).values > 0.5).sum().item()
    return f"DETR loaded  |  high-conf detections: {n}"


@register("BLIP-2 VLM score (4 sub-scores + weighted avg)", heavy=True)
def test_blip2(device):
    from metrics.vlm_score import VLMScoreMetric
    m = VLMScoreMetric(device=device)
    imgs = _rand(1, 3, 256, 192)
    # Rich API: list of dicts
    detailed = m.compute_batch(imgs)
    assert len(detailed) == 1
    entry = detailed[0]
    assert all(k in entry for k in ("s1", "s2", "s3", "s4", "vlm_score")), \
        f"Missing keys in {entry}"
    # Scalar back-compat
    scalars = m.compute_batch_scalar(imgs)
    assert len(scalars) == 1 and isinstance(scalars[0], float)
    return (
        f"backend={m._backend}  |  "
        f"S1={entry['s1']:.1f}  S2={entry['s2']:.1f}  "
        f"S3={entry['s3']:.1f}  S4={entry['s4']:.1f}  "
        f"VLM_score={entry['vlm_score']:.2f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def _green(s): return f"\033[92m{s}\033[0m"
def _red(s):   return f"\033[91m{s}\033[0m"
def _yellow(s):return f"\033[93m{s}\033[0m"


def run_tests(device: str, skip: List[str], quick: bool) -> int:
    print("\n" + "═" * 72)
    print(f"  VTON Pipeline — Model Load & Forward-Pass Tests")
    print(f"  device={device}  |  skip={skip}  |  quick={quick}")
    print("═" * 72)

    passed = failed = skipped = 0
    for t in TESTS:
        name  = t["name"]
        heavy = t["heavy"]
        fn    = t["fn"]

        # Decide whether to skip
        slug = name.split()[0].lower()
        if any(s.lower() in name.lower() for s in skip):
            print(f"  {'SKIP':<8} {name}")
            skipped += 1
            continue
        if quick and heavy:
            print(f"  {'SKIP':<8} {name}  (--quick)")
            skipped += 1
            continue

        t0 = time.time()
        try:
            msg = fn(device)
            dt  = time.time() - t0
            print(f"  {_green('PASS'):<17} [{dt:5.1f}s]  {name}")
            if msg:
                print(f"           ↳ {msg}")
            passed += 1
        except Exception as e:
            dt = time.time() - t0
            print(f"  {_red('FAIL'):<17} [{dt:5.1f}s]  {name}")
            print(f"           ↳ {_red(type(e).__name__)}: {e}")
            if "--verbose" in sys.argv:
                traceback.print_exc()
            failed += 1

    print("\n" + "─" * 72)
    print(f"  Results:  {_green(str(passed))} passed  "
          f"{_red(str(failed))} failed  "
          f"{_yellow(str(skipped))} skipped  "
          f"/ {len(TESTS)} total")
    print("─" * 72 + "\n")
    return failed


def _parse():
    p = argparse.ArgumentParser(description="Run model-load tests")
    p.add_argument("--device",  type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip",    nargs="*", default=[],
                   help="Keywords to skip (case-insensitive substring match)")
    p.add_argument("--quick",   action="store_true",
                   help="Skip tests marked heavy=True (DETR, BLIP-2)")
    p.add_argument("--verbose", action="store_true",
                   help="Print full traceback on failure")
    return p.parse_args()


if __name__ == "__main__":
    args   = _parse()
    n_fail = run_tests(args.device, args.skip, args.quick)
    sys.exit(0 if n_fail == 0 else 1)
