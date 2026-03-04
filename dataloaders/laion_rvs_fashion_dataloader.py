"""
dataloaders/laion_rvs_fashion_dataloader.py
============================================
Dataset class for the LAION-RVS-Fashion dataset, loaded from HuggingFace Hub.

Streams examples from ``Slep/LAION-RVS-Fashion`` and materialises a local
buffer of ``limit`` samples so that standard integer indexing (__getitem__)
works with PyTorch DataLoader.

Output dict per sample (canonical format):
    person : Tensor (3, H, W) float32 [0, 1]
    cloth  : Tensor (3, H, W) float32 [0, 1]
    meta   : dict   {id, dataset}
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from pathlib import Path


class LAIONRVSFashionDataset(Dataset):
    """
    Streaming-to-buffered loader for LAION-RVS-Fashion.

    Parameters
    ----------
    split : str
        HuggingFace split name (``"train"``).
    limit : int
        Maximum number of examples to buffer.
    img_size : tuple (H, W)
        Output image resolution.
    """

    def __init__(
        self,
        split: str = "train",
        limit: int = 1000,
        img_size: tuple = (512, 384),
        local_dir: str | Path | None = None,
        **kwargs,
    ):
        import pandas as pd
        from huggingface_hub import HfFileSystem

        self.transform = T.Compose([T.Resize(img_size), T.ToTensor()])
        self.data = []

        # Prefer a local dataset if provided or available under
        # `benchmark_datasets/LAION-RVS-Fashion` (exported HF format).
        if local_dir is None:
            local_candidate = Path.cwd() / "benchmark_datasets" / "LAION-RVS-Fashion"
        else:
            local_candidate = Path(local_dir)

        if local_candidate.exists():
            # Try load_from_disk (HuggingFace save_to_disk format)
            try:
                from datasets import _hf_import
                load_from_disk = _hf_import("load_from_disk")
                ds = load_from_disk(str(local_candidate))
                try:
                    hf_ds = ds[split]
                except Exception:
                    hf_ds = ds
                for i in range(min(len(hf_ds), limit)):
                    self.data.append(hf_ds[i])
                return  # loaded from disk – done
            except Exception:
                pass  # fall through to pandas read from Hub

        # Primary path: read parquet files directly from HuggingFace Hub via
        # pandas + HfFileSystem.  This bypasses the datasets streaming/
        # fingerprinting machinery entirely, avoiding both:
        #   - CastError from distractors_metadata.parquet (schema mismatch), and
        #   - RuntimeError: RLock objects should only be shared between processes
        #     through inheritance (caused by sys.modules manipulation in _hf_import
        #     corrupting internal datasets state before dill fingerprinting).
        try:
            fs = HfFileSystem()
            files = sorted(
                f for f in fs.glob(
                    f"datasets/Slep/LAION-RVS-Fashion/data/{split}/*.parquet"
                )
                if not f.split("/")[-1].startswith("distractors")
            )
            for fpath in files:
                df = pd.read_parquet(f"hf://{fpath}")
                for _, row in df.iterrows():
                    self.data.append(row.to_dict())
                    if len(self.data) >= limit:
                        return
        except Exception as exc:
            import warnings
            warnings.warn(
                f"[LAIONRVSFashionDataset] Could not load LAION data: {exc}",
                stacklevel=2,
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        # Resolve image fields (HF datasets may use different column names)
        p_pil = item.get("person_image") or item.get("image")
        c_pil = item.get("cloth_image") or item.get("cloth")

        if p_pil is None:
            p_pil = Image.new("RGB", (512, 512), (128, 128, 128))
        if c_pil is None:
            c_pil = Image.new("RGB", (512, 512), (128, 128, 128))

        if not isinstance(p_pil, Image.Image):
            p_pil = Image.fromarray(np.array(p_pil))
        if not isinstance(c_pil, Image.Image):
            c_pil = Image.fromarray(np.array(c_pil))

        return {
            "person": self.transform(p_pil.convert("RGB")),
            "cloth": self.transform(c_pil.convert("RGB")),
            "meta": {"id": f"laion_{idx}", "dataset": "laion"},
        }


if __name__ == "__main__":
    from datasets import _hf_import
    load_dataset = _hf_import("load_dataset")

    dataset = load_dataset(
        "Slep/LAION-RVS-Fashion",
        streaming=True,
    )

    train = dataset["train"]

    for example in train:
        print(example)
        break
