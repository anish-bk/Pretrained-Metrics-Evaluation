"""
config.py
---------
Centralized configuration for dataset paths and defaults.
"""

import os
from pathlib import Path

# Workspace Root
BASE_DIR = Path(__file__).parent.absolute()

# Dataset Root Directories
# Default paths can be overridden by environment variables
DATASET_ROOTS = {
    # Existing
    "viton":              os.getenv("VITON_ROOT", str(BASE_DIR / "VITON")),
    "vitonhd":            os.getenv("VITONHD_ROOT", str(BASE_DIR / "zalando-hd-resized")),
    "dresscode":          os.getenv("DRESSCODE_ROOT", str(BASE_DIR / "dresscode")),
    "mpv":                os.getenv("MPV_ROOT", str(BASE_DIR / "MPV")),
    "deepfashion":        os.getenv("DEEPFASHION_ROOT", str(BASE_DIR / "DeepFashion")),
    "laion_fashion":      "huggingface", # Special flag for LAION dataset
    
    # New / Specific request aliases
    "street_tryon":       os.getenv("STREET_TRYON_ROOT", str(BASE_DIR / "street_tryon")),
    "curvton":            os.getenv("CURVTON_ROOT", str(BASE_DIR / "curvton")),
    
    # Legacy / Compatibility
    "vton":               os.getenv("VITON_ROOT", str(BASE_DIR / "VITON")),
    "deepfashion_tryon":  os.getenv("DEEPFASHION_ROOT", str(BASE_DIR / "DeepFashion")),
}

# EDA & Metrics Defaults
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 4
DEFAULT_IMG_SIZE = (512, 384)

# Cache & Output Directories
CACHE_DIR = BASE_DIR / "eda_cache"
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"

def get_root(dataset_name: str) -> str:
    """Helper to get a dataset root directory by name."""
    name = dataset_name.lower().replace("-", "_")
    return DATASET_ROOTS.get(name, str(BASE_DIR / name))
