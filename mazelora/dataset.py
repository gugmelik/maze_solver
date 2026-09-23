"""Latent dataset: precomputed (puzzle -> solution) VAE latent pairs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .maze import MazeRecord


class LatentPairs(Dataset):
    """Reads the float16 memmaps written by `mazelora.precompute`.

    `cond` is the puzzle (Kontext reference image), `target` is the solution
    the model must produce. Both are unpacked [16, H/8, W/8] latents; packing
    happens on GPU in the training step.
    """

    def __init__(self, cache_dir: str | Path, split: str):
        d = Path(cache_dir) / split
        self.cond = np.load(d / "puzzle.npy", mmap_mode="r")
        self.target = np.load(d / "solution.npy", mmap_mode="r")
        assert len(self.cond) == len(self.target), "puzzle/solution count mismatch"
        self.meta = json.loads((Path(cache_dir) / "meta.json").read_text())

    def __len__(self) -> int:
        return len(self.cond)

    def __getitem__(self, i: int) -> dict:
        return {
            "cond": torch.from_numpy(np.array(self.cond[i])),
            "target": torch.from_numpy(np.array(self.target[i])),
            "index": i,
        }


def load_records(data_dir: str | Path, split: str) -> list[MazeRecord]:
    with open(Path(data_dir) / split / "manifest.jsonl") as f:
        return [MazeRecord.from_json(json.loads(line)) for line in f]
