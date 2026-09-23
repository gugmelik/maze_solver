"""Inference wrapper: run a LoRA-tuned FLUX.1-Kontext on maze puzzles.

Text encoders are never instantiated -- the single cached prompt embedding is
fed to the pipeline directly, which keeps ~10 GB of T5 off the card.

Where possible we feed *precomputed* VAE latents rather than PIL images. That
skips the pipeline's auto-resize (512x512 is not one of Kontext's preferred
resolutions, so it would silently rescale) and guarantees the conditioning is
bit-identical to what the model saw during training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .flux_utils import MODEL_ID, load_prompt_cache, load_transformer
from .maze import MazeRecord


class MazeSolver:
    def __init__(self, pipe, prompt_embeds, pooled_embeds, cache_dir: Path | None = None,
                 size_px: int = 512):
        self.pipe = pipe
        self.prompt_embeds = prompt_embeds
        self.pooled_embeds = pooled_embeds
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.size_px = size_px
        self._latent_cache: dict[str, np.ndarray] = {}
        pipe.set_progress_bar_config(disable=True)

    # ---------------- constructors ----------------
    @staticmethod
    def _build_pipe(transformer, model_id, device, dtype=torch.bfloat16):
        from diffusers import FluxKontextPipeline
        pipe = FluxKontextPipeline.from_pretrained(
            model_id, transformer=transformer,
            text_encoder=None, text_encoder_2=None,
            tokenizer=None, tokenizer_2=None,
            torch_dtype=dtype,
        )
        pipe.vae.to(device)          # the DiT may be bnb-quantized and un-movable
        return pipe

    @classmethod
    def from_live_transformer(cls, transformer, model_id: str, cache_dir: Path,
                              device: str = "cuda", dtype=torch.bfloat16):
        """Reuse an already-resident transformer (used for in-training validation)."""
        pipe = cls._build_pipe(transformer, model_id, device, dtype)
        pe, pooled, _ = load_prompt_cache(Path(cache_dir) / "prompt.safetensors", device, dtype)
        return cls(pipe, pe.unsqueeze(0), pooled.unsqueeze(0), cache_dir)

    @classmethod
    def from_checkpoint(cls, lora_dir: str | None, cache_dir: Path,
                        model_id: str = MODEL_ID, quantization: str = "nf4",
                        device: str = "cuda", dtype=torch.bfloat16,
                        lora_scale: float = 1.0):
        """Load the base model and (optionally) apply a trained LoRA.

        `lora_dir=None` gives the untuned baseline, which is the reference point
        for every number the evaluation reports.
        """
        transformer = load_transformer(model_id, quantization, dtype, device)
        pipe = cls._build_pipe(transformer, model_id, device, dtype)
        if lora_dir:
            pipe.load_lora_weights(str(lora_dir), adapter_name="maze")
            pipe.set_adapters(["maze"], adapter_weights=[lora_scale])
        pe, pooled, _ = load_prompt_cache(Path(cache_dir) / "prompt.safetensors", device, dtype)
        return cls(pipe, pe.unsqueeze(0), pooled.unsqueeze(0), cache_dir)

    # ---------------- generation ----------------
    def _generate(self, cond, n: int, num_steps: int, guidance_scale: float,
                  seed: int | None, extra: dict) -> list[Image.Image]:
        device = self.pipe.vae.device
        gen = None
        if seed is not None:
            gen = [torch.Generator(device=device).manual_seed(seed + i) for i in range(n)]
        out = self.pipe(
            image=cond,
            prompt_embeds=self.prompt_embeds.expand(n, -1, -1),
            pooled_prompt_embeds=self.pooled_embeds.expand(n, -1),
            height=self.size_px, width=self.size_px,
            num_inference_steps=num_steps, guidance_scale=guidance_scale,
            generator=gen, output_type="pil", **extra,
        )
        return out.images

    @torch.no_grad()
    def solve_latents(self, cond_latents: torch.Tensor, num_steps: int = 28,
                      guidance_scale: float = 2.5, seed: int | None = 0):
        device = self.pipe.vae.device
        cond = cond_latents.to(device, self.pipe.vae.dtype)
        return self._generate(cond, cond.shape[0], num_steps, guidance_scale, seed, {})

    @torch.no_grad()
    def solve_images(self, images: list[Image.Image], num_steps: int = 28,
                     guidance_scale: float = 2.5, seed: int | None = 0):
        imgs = [im.convert("RGB").resize((self.size_px, self.size_px), Image.BILINEAR)
                for im in images]
        # _auto_resize=False: keep our 512x512 grid instead of snapping to a
        # "preferred Kontext resolution", which would break the decoder geometry.
        return self._generate(imgs, len(imgs), num_steps, guidance_scale, seed,
                              {"_auto_resize": False})

    def _cached_latents(self, split: str) -> np.ndarray | None:
        if self.cache_dir is None:
            return None
        if split not in self._latent_cache:
            p = self.cache_dir / split / "puzzle.npy"
            self._latent_cache[split] = np.load(p, mmap_mode="r") if p.exists() else None
        return self._latent_cache[split]

    @torch.no_grad()
    def solve_records(self, records: list[MazeRecord], split_dir: Path, split: str = "eval",
                      num_steps: int = 28, guidance_scale: float = 2.5,
                      seed: int | None = 0, batch_size: int = 1,
                      progress: bool = False) -> list[Image.Image]:
        """Solve a list of maze records, preferring cached latents for conditioning."""
        lat = self._cached_latents(split)
        split_dir = Path(split_dir)
        out: list[Image.Image] = []
        rng = range(0, len(records), batch_size)
        if progress:
            from tqdm.auto import tqdm
            rng = tqdm(rng, desc=f"generating ({num_steps} steps)")
        for i in rng:
            chunk = records[i:i + batch_size]
            if lat is not None:
                cond = torch.from_numpy(np.stack([np.array(lat[int(r.id)]) for r in chunk]))
                out += self.solve_latents(cond, num_steps, guidance_scale,
                                          None if seed is None else seed + i)
            else:
                pil = [Image.open(split_dir / "puzzle" / f"{r.id}.png") for r in chunk]
                out += self.solve_images(pil, num_steps, guidance_scale,
                                         None if seed is None else seed + i)
        return out
