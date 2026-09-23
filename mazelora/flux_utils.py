"""Shared FLUX.1-Kontext plumbing: NF4 loading, latent packing, prompt cache.

The instruction prompt is identical for every maze, so we encode it exactly
once and reuse the cached embeddings everywhere. That keeps the 9.5 GB T5-XXL
text encoder out of both the training and the evaluation process.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
VAE_SCALE = 8            # FLUX VAE spatial compression
LATENT_CHANNELS = 16


def bnb_config(mode: str):
    """Diffusers/transformers bitsandbytes config for `nf4`, `int8` or `none`."""
    if mode in (None, "none", "bf16"):
        return None, None
    from diffusers import BitsAndBytesConfig as DiffusersBnB
    from transformers import BitsAndBytesConfig as TransformersBnB
    if mode == "nf4":
        kw = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                  bnb_4bit_compute_dtype=torch.bfloat16,
                  bnb_4bit_use_double_quant=True)
    elif mode == "int8":
        kw = dict(load_in_8bit=True)
    else:
        raise ValueError(f"unknown quantization mode: {mode}")
    return DiffusersBnB(**kw), TransformersBnB(**kw)


def load_transformer(model_id: str = MODEL_ID, quantization: str = "nf4",
                     dtype=torch.bfloat16, device: str = "cuda"):
    """Load the 12B Kontext DiT, 4-bit quantized so it fits alongside activations."""
    from diffusers import FluxTransformer2DModel
    dcfg, _ = bnb_config(quantization)
    kwargs = dict(subfolder="transformer", torch_dtype=dtype)
    if dcfg is not None:
        kwargs["quantization_config"] = dcfg
    tr = FluxTransformer2DModel.from_pretrained(model_id, **kwargs)
    if dcfg is None:
        tr = tr.to(device)
    return tr


def load_vae(model_id: str = MODEL_ID, dtype=torch.bfloat16, device: str = "cuda"):
    from diffusers import AutoencoderKL
    return AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype).to(device)


# --------------------------------------------------------------------------- #
# latent packing (mirrors FluxKontextPipeline exactly)
# --------------------------------------------------------------------------- #
def pack(latents: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] -> [B, (H/2)*(W/2), C*4] patchified sequence."""
    from diffusers import FluxKontextPipeline
    b, c, h, w = latents.shape
    return FluxKontextPipeline._pack_latents(latents, b, c, h, w)


def unpack(latents: torch.Tensor, height_px: int, width_px: int) -> torch.Tensor:
    from diffusers import FluxKontextPipeline
    return FluxKontextPipeline._unpack_latents(latents, height_px, width_px, VAE_SCALE)


def latent_ids(h_lat: int, w_lat: int, device, dtype, is_reference: bool = False):
    """Positional ids for a packed latent grid. Kontext marks the reference
    image by setting the first id channel to 1 instead of 0."""
    from diffusers import FluxKontextPipeline
    ids = FluxKontextPipeline._prepare_latent_image_ids(1, h_lat // 2, w_lat // 2, device, dtype)
    if is_reference:
        ids = ids.clone()
        ids[..., 0] = 1
    return ids


def vae_encode(vae, images: torch.Tensor, generator=None) -> torch.Tensor:
    """images in [-1, 1], [B,3,H,W] -> scaled latents [B,16,H/8,W/8]."""
    posterior = vae.encode(images.to(vae.dtype)).latent_dist
    lat = posterior.sample(generator) if generator is not None else posterior.mode()
    return (lat - vae.config.shift_factor) * vae.config.scaling_factor


def vae_decode(vae, latents: torch.Tensor) -> torch.Tensor:
    lat = latents / vae.config.scaling_factor + vae.config.shift_factor
    return vae.decode(lat.to(vae.dtype)).sample


# --------------------------------------------------------------------------- #
# prompt cache
# --------------------------------------------------------------------------- #
def save_prompt_cache(path: Path, prompt: str, prompt_embeds, pooled, text_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"prompt_embeds": prompt_embeds.squeeze(0).cpu().contiguous(),
         "pooled_prompt_embeds": pooled.squeeze(0).cpu().contiguous(),
         "text_ids": text_ids.cpu().contiguous()},
        str(path), metadata={"prompt": prompt},
    )


def load_prompt_cache(path: Path, device, dtype):
    d = load_file(str(path))
    return (d["prompt_embeds"].to(device, dtype),
            d["pooled_prompt_embeds"].to(device, dtype),
            d["text_ids"].to(device, dtype))
