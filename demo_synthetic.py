"""
demo_synthetic.py
-----------------
Quick sanity-check / smoke-test that runs the entire metric pipeline
on SYNTHETICALLY GENERATED random images (no real dataset required).

Run:
    python demo_synthetic.py

This verifies all metric modules are installed correctly and produces
a sample JSON output in ./results/demo_*.json.
"""

from __future__ import annotations
import sys, json, time, math, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from metrics.image_metrics      import compute_psnr_batch, compute_ssim_batch, compute_masked_ssim_batch, LPIPSMetric
from metrics.distribution_metrics import DistributionMetrics
from metrics.pose_error         import PoseErrorMetric
from metrics.vlm_score          import VLMScoreMetric
from metrics.jepa_metrics       import JEPAMetrics

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES  = 20
BATCH_SIZE = 4
IMG_SIZE   = (3, 256, 192)    # C, H, W


def make_batch(n=BATCH_SIZE):
    pred   = torch.rand(n, *IMG_SIZE)
    gt     = torch.rand(n, *IMG_SIZE)
    person = torch.rand(n, *IMG_SIZE)
    mask   = (torch.rand(n, 1, IMG_SIZE[1], IMG_SIZE[2]) > 0.5).float()
    return pred, gt, person, mask


def _mean(lst):
    lst = [v for v in lst if not math.isnan(v)]
    return float(np.mean(lst)) if lst else float("nan")


def main():
    print("=" * 60)
    print("  Virtual Try-On Metrics — Synthetic Smoke Test")
    print("=" * 60)

    lpips_m = LPIPSMetric(device=DEVICE)
    dist_m  = DistributionMetrics(device=DEVICE)
    pose_m  = PoseErrorMetric(device=DEVICE)
    vlm_m   = VLMScoreMetric(device=DEVICE)    # will use stub (no GPU BLIP2)
    jepa_m  = JEPAMetrics(device=DEVICE)

    acc = {k: [] for k in [
        "psnr", "ssim", "masked_ssim", "lpips",
        "pose_error",
        "vlm_s1", "vlm_s2", "vlm_s3", "vlm_s4", "vlm_score",
        "jepa_epe",
    ]}

    n_batches = N_SAMPLES // BATCH_SIZE
    for i in range(n_batches):
        pred, gt, person, mask = make_batch(BATCH_SIZE)
        print(f"  Batch {i+1}/{n_batches} …", end=" ")

        acc["psnr"].extend(compute_psnr_batch(pred, gt))
        acc["ssim"].extend(compute_ssim_batch(pred, gt))
        acc["masked_ssim"].extend(compute_masked_ssim_batch(pred, gt, mask))
        acc["lpips"].extend(lpips_m.compute_batch(pred, gt))
        acc["pose_error"].extend(pose_m.compute_batch(pred, gt))
        # VLM: list[dict] with keys s1, s2, s3, s4, vlm_score
        vlm_results = vlm_m.compute_batch(pred)
        for vr in vlm_results:
            acc["vlm_s1"].append(vr["s1"])
            acc["vlm_s2"].append(vr["s2"])
            acc["vlm_s3"].append(vr["s3"])
            acc["vlm_s4"].append(vr["s4"])
            acc["vlm_score"].append(vr["vlm_score"])
        acc["jepa_epe"].extend(jepa_m.compute_epe_batch(person, pred))
        jepa_m.update_embeddings(pred)
        dist_m.update(pred, gt)

        print("✓")

    print("\n  Computing FID / IS / KID …")
    try:
        dist_res = dist_m.compute()
    except Exception as e:
        print(f"  [WARN] {e}")
        dist_res = {"fid": float("nan"), "is_mean": float("nan"),
                    "is_std": float("nan"), "kid_mean": float("nan"),
                    "kid_std": float("nan")}
    dist_m.cleanup()

    jepa_trace = jepa_m.compute_embedding_trace()

    results = {
        "psnr":                           _mean(acc["psnr"]),
        "ssim":                           _mean(acc["ssim"]),
        "masked_ssim":                    _mean(acc["masked_ssim"]),
        "lpips":                          _mean(acc["lpips"]),
        "fid":                            dist_res["fid"],
        "is_mean":                        dist_res["is_mean"],
        "is_std":                         dist_res["is_std"],
        "kid_mean":                       dist_res["kid_mean"],
        "kid_std":                        dist_res["kid_std"],
        "pose_error_px":                  _mean(acc["pose_error"]),
        # VLM sub-scores + composite
        "vlm_s1_garment_fidelity":        _mean(acc["vlm_s1"]),
        "vlm_s2_geometric_naturalness":   _mean(acc["vlm_s2"]),
        "vlm_s3_identity_preservation":   _mean(acc["vlm_s3"]),
        "vlm_s4_scene_coherence":         _mean(acc["vlm_s4"]),
        "vlm_score":                      _mean(acc["vlm_score"]),
        "jepa_epe":                       _mean(acc["jepa_epe"]),
        "jepa_trace_cov":                 jepa_trace,
    }

    print("\n  ┌─────────────────────────────────────────┐")
    print("  │           Smoke-Test Results             │")
    print("  ├─────────────────────────────────────────┤")
    for k, v in results.items():
        val = "N/A" if (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"
        print(f"  │  {k:<28} {val:>10}  │")
    print("  └─────────────────────────────────────────┘")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"demo_{ts}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
