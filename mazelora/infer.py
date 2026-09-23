"""Inference wrapper: run a LoRA-tuned image-edit model on maze puzzles.

Backend-agnostic. Where the backend allows it, conditioning is fed as
*precomputed latents* rather than PIL images, so evaluation sees exactly what
training saw instead of whatever the pipeline's default resizing produces.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .backends import Backend, get_backend
from .maze import MazeRecord


class MazeSolver:
    def __init__(self, backend: Backend, solver, cache_dir: Path, size_px: int):
        self.backend = backend
        self.solver = solver
        self.cache_dir = Path(cache_dir)
        self.size_px = size_px
        self._latents: dict[str, np.ndarray | None] = {}

    # ---------------- constructors ----------------
    @staticmethod
    def _size_px(cache_dir: Path) -> int:
        return json.loads((Path(cache_dir) / "meta.json").read_text()).get("size_px", 512)

    @classmethod
    def from_live_transformer(cls, backend: Backend, transformer, model_id: str,
                              cache_dir: Path, device: str = "cuda",
                              dtype=torch.bfloat16):
        """Reuse an already-resident transformer (for in-training validation)."""
        size_px = cls._size_px(cache_dir)
        solver = backend.build_solver(transformer, model_id, Path(cache_dir),
                                      device, dtype, size_px)
        return cls(backend, solver, cache_dir, size_px)

    @classmethod
    def from_checkpoint(cls, backend_name: str, lora_dir: str | None, cache_dir: Path,
                        model_id: str | None = None, quantization: str = "nf4",
                        device: str = "cuda", dtype=torch.bfloat16,
                        lora_scale: float = 1.0):
        """Load a base model and optionally apply a trained LoRA.

        `lora_dir=None` gives the untuned baseline, the reference point for
        every number the evaluation reports.
        """
        backend = get_backend(backend_name)
        model_id = model_id or backend.default_model_id
        size_px = cls._size_px(cache_dir)
        transformer = backend.load_transformer(model_id, quantization, dtype, device)
        solver = backend.build_solver(transformer, model_id, Path(cache_dir),
                                      device, dtype, size_px)
        if lora_dir:
            solver.load_lora(lora_dir, lora_scale)
        return cls(backend, solver, cache_dir, size_px)

    # ---------------- generation ----------------
    def set_lora_scale(self, scale: float):
        self.solver.set_lora_scale(scale)

    @torch.no_grad()
    def solve_images(self, images: list[Image.Image], num_steps: int = 28,
                     guidance_scale: float | None = None, seed: int | None = 0):
        g = self.backend.eval_guidance if guidance_scale is None else guidance_scale
        return self.solver.generate_from_images(images, num_steps, g, seed)

    def _cached(self, split: str) -> np.ndarray | None:
        if split not in self._latents:
            p = self.cache_dir / split / "puzzle.npy"
            self._latents[split] = np.load(p, mmap_mode="r") if p.exists() else None
        return self._latents[split]

    @torch.no_grad()
    def solve_records(self, records: list[MazeRecord], split_dir: Path, split: str = "eval",
                      num_steps: int = 28, guidance_scale: float | None = None,
                      seed: int | None = 0, batch_size: int = 1,
                      progress: bool = False) -> list[Image.Image]:
        g = self.backend.eval_guidance if guidance_scale is None else guidance_scale
        lat = self._cached(split)
        split_dir = Path(split_dir)
        out: list[Image.Image] = []
        rng = range(0, len(records), batch_size)
        if progress:
            from tqdm.auto import tqdm
            rng = tqdm(rng, desc=f"generating ({num_steps} steps)")
        for i in rng:
            chunk = records[i:i + batch_size]
            pil = [Image.open(split_dir / "puzzle" / f"{r.id}.png").convert("RGB")
                   for r in chunk]
            s = None if seed is None else seed + i
            if lat is not None:
                cond = torch.from_numpy(np.stack([np.array(lat[int(r.id)]) for r in chunk]))
                out += self.solver.generate_from_latents(cond, num_steps, g, s, images=pil)
            else:
                out += self.solver.generate_from_images(pil, num_steps, g, s)
        return out
