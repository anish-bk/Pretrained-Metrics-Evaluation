from .loaders import get_dataset, DATASET_REGISTRY
from .anish_loaders import (
    AnishDressCodeDataset, 
    AnishVITONHDDataset, 
    AnishLAIONDataset,
    anish_collate_fn
)

# Re-export standalone availability flag
from .loaders import _HAS_STANDALONE


# ── HuggingFace datasets bridge ───────────────────────────────────────────────
# The local `datasets/` package shadows the HuggingFace `datasets` package when
# Python resolves bare `import datasets` or `from datasets import ...`.
# This helper temporarily removes the workspace root from sys.path and clears the
# cached local module so the real HuggingFace package is found instead.

def _hf_import(symbol: str):
    """
    Import `symbol` from the *installed* HuggingFace `datasets` package,
    bypassing this local package that shadows it.

    Usage:
        load_dataset  = _hf_import("load_dataset")
        load_from_disk = _hf_import("load_from_disk")
    """
    import sys
    import pathlib

    workspace = str(pathlib.Path(__file__).parent.parent)

    # Save state
    _path_backup = sys.path[:]
    _mod_backup  = {k: v for k, v in sys.modules.items()
                    if k == "datasets" or k.startswith("datasets.")}

    # Remove local package traces
    sys.path = [p for p in sys.path if p not in (workspace, "")]
    for key in list(_mod_backup):
        sys.modules.pop(key, None)

    try:
        import datasets as _hf_ds
        return getattr(_hf_ds, symbol)
    except Exception as exc:
        raise ImportError(
            f"HuggingFace 'datasets' package not installed or '{symbol}' "
            f"unavailable ({exc})."
        ) from exc
    finally:
        sys.path[:] = _path_backup
        sys.modules.update(_mod_backup)         # restore local package
