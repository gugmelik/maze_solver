"""Backend interface: everything that differs between image-edit base models.

The maze task, the dataset, the decoder, the metrics and the report are all
model-agnostic. Only these pieces change per model, so they live behind one
small interface and the rest of the codebase never branches on model identity.

Both supported models are rectified-flow image-edit transformers that condition
by concatenating the reference image's latents onto the noisy target along the
*sequence* axis, so the training objective below is genuinely shared. What
differs is conditioning plumbing: FLUX gets a cached text embedding plus
explicit positional ids, Qwen gets per-sample embeddings from a vision-language
encoder that looks at the maze itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch

#: Checkpoint filename. Always passed explicitly when loading: diffusers'
#: model-level `load_lora_adapter` leaves `use_safetensors=None`, and its file
#: lookup is gated on `(use_safetensors and weight_name is None) or
#: weight_name.endswith(".safetensors")` -- so with both unset it skips
#: safetensors entirely and fails looking for a .bin that was never written.
LORA_WEIGHTS_FILE = "pytorch_lora_weights.safetensors"


def save_lora(transformer, ck_dir: Path) -> Path:
    """Write the adapter to `<ck_dir>/pytorch_lora_weights.safetensors`."""
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import save_file
    ck_dir = Path(ck_dir)
    ck_dir.mkdir(parents=True, exist_ok=True)
    sd = {f"transformer.{k}": v.to(torch.float32).cpu().contiguous()
          for k, v in get_peft_model_state_dict(transformer).items()}
    path = ck_dir / LORA_WEIGHTS_FILE
    save_file(sd, str(path))
    return path


def load_lora(transformer, ck_dir, adapter_name: str = "default"):
    """Attach a saved adapter to a transformer, inferring rank and targets."""
    transformer.load_lora_adapter(str(ck_dir), prefix="transformer",
                                  weight_name=LORA_WEIGHTS_FILE,
                                  adapter_name=adapter_name)


# --------------------------------------------------------------------------- #
# shared rectified-flow maths
# --------------------------------------------------------------------------- #
def pack(latents: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] -> [B, (H/2)(W/2), C*4].

    FLUX and QwenImage use byte-identical patchify logic, so this is shared.
    """
    b, c, h, w = latents.shape[0], latents.shape[-3], latents.shape[-2], latents.shape[-1]
    latents = latents.reshape(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(b, (h // 2) * (w // 2), c * 4)


def get_sigmas(scheduler, timesteps, device, n_dim: int, dtype):
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_t = scheduler.timesteps.to(device)
    idx = [(schedule_t == t).nonzero().item() for t in timesteps]
    sigma = sigmas[idx].flatten()
    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def sample_noisy(target_packed: torch.Tensor, cfg, scheduler):
    """Draw a timestep per sample and interpolate toward noise.

    Returns (noisy, noise, sigmas, timesteps).
    """
    from diffusers.training_utils import compute_density_for_timestep_sampling
    device, bsz = target_packed.device, target_packed.shape[0]
    n_train = scheduler.config.num_train_timesteps

    noise = torch.randn_like(target_packed)
    u = compute_density_for_timestep_sampling(
        cfg.weighting_scheme, bsz, cfg.logit_mean, cfg.logit_std, cfg.mode_scale)
    idx = (u * n_train).long().clamp(0, n_train - 1)
    timesteps = scheduler.timesteps[idx].to(device)
    sigmas = get_sigmas(scheduler, timesteps, device, target_packed.ndim, target_packed.dtype)
    noisy = (1.0 - sigmas) * target_packed + sigmas * noise
    return noisy, noise, sigmas, timesteps


def flow_loss(pred, noise, target_packed, sigmas, cfg):
    """Rectified flow: supervise the straight-line velocity `noise - clean`."""
    from diffusers.training_utils import compute_loss_weighting_for_sd3
    flow_target = (noise - target_packed).float()
    weighting = compute_loss_weighting_for_sd3(cfg.weighting_scheme, sigmas).float()
    return (weighting * (pred.float() - flow_target) ** 2).mean()


def bnb_config(mode: str):
    """bitsandbytes configs for diffusers and transformers, or (None, None)."""
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


# --------------------------------------------------------------------------- #
class Backend(ABC):
    """One image-edit base model, wired for maze training and evaluation."""

    name: str
    default_model_id: str
    default_lora_targets: list[str]
    latent_channels: int = 16
    vae_scale: int = 8
    #: pixel size of the extra condition image a backend needs at train time
    #: (None when text conditioning does not depend on the image)
    condition_px: int | None = None
    #: recommended inference guidance, and the value baked in during training
    train_guidance: float = 1.0
    eval_guidance: float = 2.5

    # ---- loading -------------------------------------------------------- #
    @abstractmethod
    def load_transformer(self, model_id: str, quantization: str, dtype, device): ...

    @abstractmethod
    def load_vae(self, model_id: str, dtype, device): ...

    @abstractmethod
    def encode_images(self, vae, images: torch.Tensor) -> torch.Tensor:
        """images in [-1,1], [B,3,H,W] -> normalised latents [B,C,h,w]."""

    # ---- caching -------------------------------------------------------- #
    def precompute_extra(self, cache_dir: Path, model_id: str, quantization: str,
                         prompt: str, device: str) -> None:
        """Optional one-off cache (e.g. a text embedding). Default: nothing."""

    # ---- training ------------------------------------------------------- #
    @abstractmethod
    def make_context(self, cache_dir: Path, model_id: str, quantization: str,
                     device, dtype, size_px: int) -> dict:
        """Build whatever conditioning state the training step reuses."""

    @abstractmethod
    def loss(self, transformer, batch: dict, ctx: dict, cfg, scheduler,
             device, dtype) -> torch.Tensor: ...

    # ---- inference ------------------------------------------------------ #
    @abstractmethod
    def build_solver(self, transformer, model_id: str, cache_dir: Path,
                     device, dtype, size_px: int): ...

    def scheduler(self, model_id: str):
        from diffusers import FlowMatchEulerDiscreteScheduler
        return FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
