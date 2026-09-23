"""Latent dataset: precomputed (puzzle -> solution) VAE latent pairs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .maze import MazeRecord


class LatentPairs(Dataset):
    """Reads the float16 memmaps written by `mazelora.precompute`.

    `cond` is the puzzle (the reference image), `target` is the solution the
    model must produce. Both are unpacked [16, H/8, W/8] latents; patchifying
    happens on GPU in the training step.

    Backends whose text encoder looks at the image (Qwen's VLM) also need the
    puzzle as pixels. Pass `condition_px` and each item carries a `cond_px`
    uint8 HWC tensor, decoded in the dataloader workers so it costs no GPU time.
    """

    def __init__(self, cache_dir: str | Path, split: str,
                 data_dir: str | Path | None = None, condition_px: int | None = None):
        d = Path(cache_dir) / split
        self.cond = np.load(d / "puzzle.npy", mmap_mode="r")
        self.target = np.load(d / "solution.npy", mmap_mode="r")
        assert len(self.cond) == len(self.target), "puzzle/solution count mismatch"
        self.meta = json.loads((Path(cache_dir) / "meta.json").read_text())
        self.condition_px = condition_px
        self.puzzle_dir = Path(data_dir) / split / "puzzle" if data_dir else None
        if condition_px and self.puzzle_dir is None:
            raise ValueError("condition_px needs data_dir to locate the puzzle PNGs")

    def __len__(self) -> int:
        return len(self.cond)

    def __getitem__(self, i: int) -> dict:
        item = {
            "cond": torch.from_numpy(np.array(self.cond[i])),
            "target": torch.from_numpy(np.array(self.target[i])),
            "index": i,
        }
        if self.condition_px:
            im = Image.open(self.puzzle_dir / f"{i:06d}.png").convert("RGB")
            im = im.resize((self.condition_px, self.condition_px), Image.BILINEAR)
            item["cond_px"] = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy())
        return item


def load_records(data_dir: str | Path, split: str) -> list[MazeRecord]:
    with open(Path(data_dir) / split / "manifest.jsonl") as f:
        return [MazeRecord.from_json(json.loads(line)) for line in f]
