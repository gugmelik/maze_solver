"""Qwen-Image-Edit-2511 backend (20B DiT, ~54 GB download).

Three things differ from FLUX in ways that matter here:

1. **Text conditioning sees the image.** The text encoder is Qwen2.5-VL, and
   `_get_qwen_prompt_embeds` feeds the maze itself (downscaled to ~384px)
   through the vision tower. So the prompt embedding is *per sample* and cannot
   be cached once the way FLUX's can. Caching it for 20k mazes would cost
   ~37 GB, so instead the VLM stays resident in 4-bit (~4 GB) and runs inside
   the training step on a condition image the dataloader prepares.
2. **No guidance embedding and no pooled projection** (`guidance_embeds: false`),
   and positions come from `img_shapes` -- a list of (frames, h/2, w/2) per
   image -- rather than an explicit id tensor.
3. **The VAE is a 3D (video) autoencoder** with per-channel latent statistics
   rather than a single scale/shift pair. We add and strip the temporal axis so
   the on-disk latent cache has the same layout as the FLUX one.

Generation uses a small Euler loop rather than `QwenImageEditPlusPipeline`,
because that pipeline hard-codes the reference image to 1024x1024 (`VAE_IMAGE_SIZE`)
and refuses batch sizes above 1. We train at 512, so the loop keeps evaluation
geometry identical to training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .base import Backend, bnb_config, flow_loss, load_lora, pack, sample_noisy

CONDITION_PX = 384          # what the vision tower sees; matches CONDITION_IMAGE_SIZE


class QwenEditBackend(Backend):
    name = "qwen_edit"
    default_model_id = "Qwen/Qwen-Image-Edit-2511"
    # QwenImage blocks are dual-stream: to_* drives the image tokens and add_*_proj
    # / to_add_out drive the text tokens. Adapting only the image half would leave
    # the instruction pathway frozen, so both are targeted.
    default_lora_targets = ["to_q", "to_k", "to_v", "to_out.0",
                            "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]
    latent_channels = 16
    vae_scale = 8
    condition_px = CONDITION_PX
    train_guidance = 1.0        # unused: guidance_embeds is False
    eval_guidance = 4.0         # true-CFG scale, Qwen's usual default

    # ---- loading -------------------------------------------------------- #
    def load_transformer(self, model_id, quantization, dtype, device):
        from diffusers import QwenImageTransformer2DModel
        dcfg, _ = bnb_config(quantization)
        kw = dict(subfolder="transformer", torch_dtype=dtype)
        if dcfg is not None:
            kw["quantization_config"] = dcfg
        tr = QwenImageTransformer2DModel.from_pretrained(model_id, **kw)
        return tr if dcfg is not None else tr.to(device)

    def load_vae(self, model_id, dtype, device):
        from diffusers import AutoencoderKLQwenImage
        return AutoencoderKLQwenImage.from_pretrained(
            model_id, subfolder="vae", torch_dtype=dtype).to(device)

    @staticmethod
    def _latent_stats(vae, device, dtype):
        z = vae.config.z_dim
        mean = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1).to(device, dtype)
        std = torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1).to(device, dtype)
        return mean, std

    def encode_images(self, vae, images):
        """[B,3,H,W] in [-1,1] -> [B,16,h,w]; the temporal axis is added and stripped."""
        x = images.to(vae.dtype).unsqueeze(2)                 # [B,3,1,H,W]
        lat = vae.encode(x).latent_dist.mode()                # [B,16,1,h,w]
        mean, std = self._latent_stats(vae, lat.device, lat.dtype)
        return ((lat - mean) / std).squeeze(2)

    def decode_latents(self, vae, latents):
        """[B,16,h,w] -> image tensor in [-1,1]."""
        mean, std = self._latent_stats(vae, latents.device, vae.dtype)
        x = latents.to(vae.dtype).unsqueeze(2) * std + mean
        return vae.decode(x, return_dict=False)[0][:, :, 0]

    # ---- shared VLM plumbing -------------------------------------------- #
    @staticmethod
    def _load_text_pipe(model_id, quantization, device, dtype):
        """QwenImageEditPlusPipeline with no DiT/VAE, purely to reuse its exact
        prompt template and `_get_qwen_prompt_embeds` implementation."""
        from diffusers import QwenImageEditPlusPipeline
        from transformers import Qwen2_5_VLForConditionalGeneration
        _, tcfg = bnb_config(quantization)
        kw = dict(subfolder="text_encoder", torch_dtype=dtype)
        if tcfg is not None:
            kw.update(quantization_config=tcfg, device_map=device)
        te = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kw)
        if tcfg is None:
            te = te.to(device)
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_id, transformer=None, vae=None, text_encoder=te, torch_dtype=dtype)
        return pipe

    @staticmethod
    def _encode_batch(text_pipe, prompt: str, images: list[Image.Image], device):
        """Per-sample VLM encoding, right-padded into one batch.

        `_get_qwen_prompt_embeds` treats a *list* of images as several references
        for a single prompt, so each maze is encoded on its own and the results
        are padded here.

        Note that `encode_prompt` returns `None` for the mask whenever it would
        be all ones -- which is always the case for a single un-padded prompt.
        We only build a real mask when padding actually makes one necessary, and
        otherwise pass `None` through, exactly as the pipeline does.
        """
        embeds, lengths = [], []
        for im in images:
            e, _ = text_pipe.encode_prompt(prompt=[prompt], image=im, device=device,
                                           num_images_per_prompt=1,
                                           max_sequence_length=1024)
            embeds.append(e[0])
            lengths.append(e.shape[1])

        L = max(lengths)
        if len(set(lengths)) == 1:
            return torch.stack(embeds), None      # no padding -> no mask needed

        out = embeds[0].new_zeros(len(embeds), L, embeds[0].shape[-1])
        mask = torch.zeros(len(embeds), L, dtype=torch.long, device=embeds[0].device)
        for i, (e, n) in enumerate(zip(embeds, lengths)):
            out[i, :n] = e
            mask[i, :n] = 1
        return out, mask

    @staticmethod
    def _img_shapes(bsz, size_px, vae_scale):
        s = size_px // vae_scale // 2
        return [[(1, s, s), (1, s, s)] for _ in range(bsz)]

    # ---- training ------------------------------------------------------- #
    def make_context(self, cache_dir, model_id, quantization, device, dtype, size_px):
        import json
        meta = json.loads((Path(cache_dir) / "meta.json").read_text())
        print(f"loading Qwen2.5-VL text encoder ({quantization}); it stays resident "
              f"because prompt embeddings depend on each maze")
        return {
            "text_pipe": self._load_text_pipe(model_id, quantization, device, dtype),
            "prompt_text": meta["prompt"],
            "size_px": size_px,
        }

    def loss(self, transformer, batch, ctx, cfg, scheduler, device, dtype):
        target = batch["target"].to(device, dtype, non_blocking=True)
        cond = batch["cond"].to(device, dtype, non_blocking=True)
        tp, cp = pack(target), pack(cond)
        bsz = tp.shape[0]

        pil = [Image.fromarray(x.numpy()) for x in batch["cond_px"]]
        with torch.no_grad():
            embeds, mask = self._encode_batch(ctx["text_pipe"], ctx["prompt_text"],
                                              pil, device)
        embeds = embeds.to(dtype)

        noisy, noise, sigmas, timesteps = sample_noisy(tp, cfg, scheduler)
        hidden = torch.cat([noisy, cp], dim=1)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=hidden.is_cuda):
            pred = transformer(
                hidden_states=hidden,
                timestep=timesteps.to(hidden.dtype) / 1000,
                guidance=None,                       # guidance_embeds is False
                encoder_hidden_states=embeds,
                encoder_hidden_states_mask=mask,
                img_shapes=self._img_shapes(bsz, ctx["size_px"], self.vae_scale),
                return_dict=False,
            )[0]
        return flow_loss(pred[:, : tp.shape[1]], noise, tp, sigmas, cfg)

    # ---- inference ------------------------------------------------------ #
    def build_solver(self, transformer, model_id, cache_dir, device, dtype, size_px):
        return _QwenSolver(self, transformer, model_id, cache_dir, device, dtype, size_px)


class _QwenSolver:
    """Minimal flow-matching Euler sampler, kept at the training resolution."""

    def __init__(self, backend, transformer, model_id, cache_dir, device, dtype, size_px):
        import json
        self.b = backend
        self.transformer = transformer
        self.vae = backend.load_vae(model_id, dtype, device)
        self.vae.eval()
        self.scheduler = backend.scheduler(model_id)
        self.text_pipe = backend._load_text_pipe(model_id, "nf4", device, dtype)
        self.prompt = json.loads((Path(cache_dir) / "meta.json").read_text())["prompt"]
        self.size_px = size_px
        self.device = device
        self.dtype = dtype
        self._lora_scale = 1.0

    def load_lora(self, lora_dir, scale):
        # the transformer infers rank and target modules from the state dict
        load_lora(self.transformer, lora_dir, adapter_name="maze")
        self.set_lora_scale(scale)

    def set_lora_scale(self, scale):
        self._lora_scale = scale
        self.transformer.set_adapters(["maze"], [scale])

    def _condition(self, images: list[Image.Image]):
        cond_pil = [im.convert("RGB").resize((CONDITION_PX, CONDITION_PX), Image.BILINEAR)
                    for im in images]
        embeds, mask = self.b._encode_batch(self.text_pipe, self.prompt, cond_pil, self.device)
        return embeds.to(self.dtype), mask

    @torch.no_grad()
    def _sample(self, cond_latents, embeds, mask, num_steps, true_cfg,
                neg=None, neg_mask=None, generator=None):
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
            calculate_shift, retrieve_timesteps)
        dev, dt = self.device, self.dtype
        cp = pack(cond_latents.to(dev, dt))
        bsz, seq = cp.shape[0], cp.shape[1]

        latents = torch.randn((bsz, seq, cp.shape[-1]), device=dev, dtype=dt,
                              generator=generator)
        sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        mu = calculate_shift(seq,
                             self.scheduler.config.get("base_image_seq_len", 256),
                             self.scheduler.config.get("max_image_seq_len", 4096),
                             self.scheduler.config.get("base_shift", 0.5),
                             self.scheduler.config.get("max_shift", 1.15))
        timesteps, num_steps = retrieve_timesteps(self.scheduler, num_steps, dev,
                                                  sigmas=sigmas, mu=mu)
        shapes = self.b._img_shapes(bsz, self.size_px, self.b.vae_scale)
        self.scheduler.set_begin_index(0)

        for t in timesteps:
            inp = torch.cat([latents, cp], dim=1)
            ts = t.expand(bsz).to(dt)
            pred = self.transformer(hidden_states=inp, timestep=ts / 1000, guidance=None,
                                    encoder_hidden_states=embeds,
                                    encoder_hidden_states_mask=mask,
                                    img_shapes=shapes, return_dict=False)[0][:, :seq]
            if true_cfg > 1.0 and neg is not None:
                npred = self.transformer(hidden_states=inp, timestep=ts / 1000, guidance=None,
                                         encoder_hidden_states=neg,
                                         encoder_hidden_states_mask=neg_mask,
                                         img_shapes=shapes, return_dict=False)[0][:, :seq]
                pred = npred + true_cfg * (pred - npred)
            latents = self.scheduler.step(pred, t, latents, return_dict=False)[0]

        from diffusers import QwenImageEditPlusPipeline as P
        lat = P._unpack_latents(latents, self.size_px, self.size_px, self.b.vae_scale)
        img = self.b.decode_latents(self.vae, lat.squeeze(2))
        img = ((img.float() / 2 + 0.5).clamp(0, 1) * 255).round().byte()
        return [Image.fromarray(x.permute(1, 2, 0).cpu().numpy()) for x in img]

    def _gen(self, seed):
        return None if seed is None else torch.Generator(device=self.device).manual_seed(seed)

    @torch.no_grad()
    def generate_from_images(self, images, num_steps, guidance, seed):
        embeds, mask = self._condition(images)
        neg = neg_mask = None
        if guidance > 1.0:
            cond_pil = [im.convert("RGB").resize((CONDITION_PX, CONDITION_PX), Image.BILINEAR)
                        for im in images]
            neg, neg_mask = self.b._encode_batch(self.text_pipe, " ", cond_pil, self.device)
            neg = neg.to(self.dtype)
        px = [im.convert("RGB").resize((self.size_px, self.size_px), Image.BILINEAR)
              for im in images]
        arr = np.stack([np.asarray(p, dtype=np.float32) for p in px])
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).to(self.device) / 127.5 - 1.0
        cond_lat = self.b.encode_images(self.vae, t)
        return self._sample(cond_lat, embeds, mask, num_steps, guidance,
                            neg, neg_mask, self._gen(seed))

    @torch.no_grad()
    def generate_from_latents(self, cond_latents, num_steps, guidance, seed, images=None):
        if images is None:
            raise ValueError(
                "Qwen conditions its text encoder on the image, so generation needs the "
                "puzzle image as well as its latents.")
        embeds, mask = self._condition(images)
        neg = neg_mask = None
        if guidance > 1.0:
            cond_pil = [im.convert("RGB").resize((CONDITION_PX, CONDITION_PX), Image.BILINEAR)
                        for im in images]
            neg, neg_mask = self.b._encode_batch(self.text_pipe, " ", cond_pil, self.device)
            neg = neg.to(self.dtype)
        return self._sample(cond_latents, embeds, mask, num_steps, guidance,
                            neg, neg_mask, self._gen(seed))
