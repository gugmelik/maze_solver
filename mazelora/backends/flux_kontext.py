"""FLUX.1-Kontext-dev backend (12B DiT, ~34 GB download).

Text conditioning is image-independent: T5-XXL sees only the instruction, which
is identical for every maze. So the prompt is encoded **once** into a cached
embedding and the 9.5 GB text tower never loads again -- not during training,
not during evaluation.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from .base import Backend, bnb_config, flow_loss, pack, sample_noisy


class FluxKontextBackend(Backend):
    name = "flux_kontext"
    default_model_id = "black-forest-labs/FLUX.1-Kontext-dev"
    default_lora_targets = ["to_q", "to_k", "to_v", "to_out.0"]
    latent_channels = 16
    vae_scale = 8
    condition_px = None          # T5 never sees the image
    train_guidance = 1.0
    eval_guidance = 2.5

    # ---- loading -------------------------------------------------------- #
    def load_transformer(self, model_id, quantization, dtype, device):
        from diffusers import FluxTransformer2DModel
        dcfg, _ = bnb_config(quantization)
        kw = dict(subfolder="transformer", torch_dtype=dtype)
        if dcfg is not None:
            kw["quantization_config"] = dcfg
        tr = FluxTransformer2DModel.from_pretrained(model_id, **kw)
        return tr if dcfg is not None else tr.to(device)

    def load_vae(self, model_id, dtype, device):
        from diffusers import AutoencoderKL
        return AutoencoderKL.from_pretrained(model_id, subfolder="vae",
                                             torch_dtype=dtype).to(device)

    def encode_images(self, vae, images):
        lat = vae.encode(images.to(vae.dtype)).latent_dist.mode()
        return (lat - vae.config.shift_factor) * vae.config.scaling_factor

    # ---- caching -------------------------------------------------------- #
    def precompute_extra(self, cache_dir, model_id, quantization, prompt, device):
        out = Path(cache_dir) / "prompt.safetensors"
        if out.exists():
            print(f"prompt cache exists, skipping: {out}")
            return
        import gc
        from diffusers import FluxKontextPipeline
        from transformers import T5EncoderModel

        _, tcfg = bnb_config(quantization)
        print(f"loading text encoders (one-shot, t5 quantization={quantization})...")
        te2_kw = dict(subfolder="text_encoder_2", torch_dtype=torch.bfloat16)
        if tcfg is not None:
            te2_kw.update(quantization_config=tcfg, device_map=device)
        te2 = T5EncoderModel.from_pretrained(model_id, **te2_kw)

        pipe = FluxKontextPipeline.from_pretrained(
            model_id, transformer=None, vae=None, text_encoder_2=te2,
            torch_dtype=torch.bfloat16)
        if tcfg is None:
            pipe.to(device)
        else:
            pipe.text_encoder.to(device)   # bnb modules are already placed

        embeds, pooled, text_ids = pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, device=device,
            num_images_per_prompt=1, max_sequence_length=512)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_file({"prompt_embeds": embeds.squeeze(0).cpu().contiguous(),
                   "pooled_prompt_embeds": pooled.squeeze(0).cpu().contiguous(),
                   "text_ids": text_ids.cpu().contiguous()},
                  str(out), metadata={"prompt": prompt})
        print(f"prompt embeds {tuple(embeds.shape)} -> {out}")
        del pipe, te2
        gc.collect()
        torch.cuda.empty_cache()

    # ---- training ------------------------------------------------------- #
    @staticmethod
    def _latent_ids(h_lat, w_lat, device, dtype, is_reference):
        from diffusers import FluxKontextPipeline
        ids = FluxKontextPipeline._prepare_latent_image_ids(
            1, h_lat // 2, w_lat // 2, device, dtype)
        if is_reference:
            ids = ids.clone()
            ids[..., 0] = 1     # Kontext tags the reference image this way
        return ids

    def make_context(self, cache_dir, model_id, quantization, device, dtype, size_px):
        d = load_file(str(Path(cache_dir) / "prompt.safetensors"))
        h = w = size_px // self.vae_scale
        return {
            "prompt": d["prompt_embeds"].to(device, dtype).unsqueeze(0),
            "pooled": d["pooled_prompt_embeds"].to(device, dtype).unsqueeze(0),
            "text_ids": d["text_ids"].to(device, dtype),
            "img_ids": torch.cat([self._latent_ids(h, w, device, dtype, False),
                                  self._latent_ids(h, w, device, dtype, True)], dim=0),
        }

    def loss(self, transformer, batch, ctx, cfg, scheduler, device, dtype):
        target = batch["target"].to(device, dtype, non_blocking=True)
        cond = batch["cond"].to(device, dtype, non_blocking=True)
        tp, cp = pack(target), pack(cond)
        bsz = tp.shape[0]

        noisy, noise, sigmas, timesteps = sample_noisy(tp, cfg, scheduler)
        hidden = torch.cat([noisy, cp], dim=1)
        guidance = (torch.full([bsz], cfg.guidance_scale, device=device, dtype=torch.float32)
                    if transformer.config.guidance_embeds else None)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=hidden.is_cuda):
            pred = transformer(
                hidden_states=hidden,
                timestep=timesteps.to(hidden.dtype) / 1000,
                guidance=guidance,
                pooled_projections=ctx["pooled"].expand(bsz, -1),
                encoder_hidden_states=ctx["prompt"].expand(bsz, -1, -1),
                txt_ids=ctx["text_ids"],
                img_ids=ctx["img_ids"],
                return_dict=False,
            )[0]
        return flow_loss(pred[:, : tp.shape[1]], noise, tp, sigmas, cfg)

    # ---- inference ------------------------------------------------------ #
    def build_solver(self, transformer, model_id, cache_dir, device, dtype, size_px):
        return _FluxSolver(self, transformer, model_id, cache_dir, device, dtype, size_px)


class _FluxSolver:
    """Wraps FluxKontextPipeline with no text encoders and cached embeddings."""

    def __init__(self, backend, transformer, model_id, cache_dir, device, dtype, size_px):
        from diffusers import FluxKontextPipeline
        self.pipe = FluxKontextPipeline.from_pretrained(
            model_id, transformer=transformer,
            text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None,
            torch_dtype=dtype)
        self.pipe.vae.to(device)          # the DiT may be bnb-quantized and un-movable
        self.pipe.set_progress_bar_config(disable=True)
        d = load_file(str(Path(cache_dir) / "prompt.safetensors"))
        self.prompt = d["prompt_embeds"].to(device, dtype).unsqueeze(0)
        self.pooled = d["pooled_prompt_embeds"].to(device, dtype).unsqueeze(0)
        self.size_px = size_px
        self.device = device

    def load_lora(self, lora_dir, scale):
        self.pipe.load_lora_weights(str(lora_dir), adapter_name="maze")
        self.pipe.set_adapters(["maze"], adapter_weights=[scale])

    def set_lora_scale(self, scale):
        self.pipe.set_adapters(["maze"], adapter_weights=[scale])

    def _run(self, cond, n, num_steps, guidance, gen, extra):
        out = self.pipe(image=cond,
                        prompt_embeds=self.prompt.expand(n, -1, -1),
                        pooled_prompt_embeds=self.pooled.expand(n, -1),
                        height=self.size_px, width=self.size_px,
                        num_inference_steps=num_steps, guidance_scale=guidance,
                        generator=gen, output_type="pil", **extra)
        return out.images

    def _gen(self, n, seed):
        if seed is None:
            return None
        return [torch.Generator(device=self.pipe.vae.device).manual_seed(seed + i)
                for i in range(n)]

    @torch.no_grad()
    def generate_from_latents(self, cond_latents, num_steps, guidance, seed, images=None):
        # `images` is accepted for interface parity with backends whose text
        # encoder needs the picture; FLUX's cached T5 embedding does not.
        cond = cond_latents.to(self.pipe.vae.device, self.pipe.vae.dtype)
        n = cond.shape[0]
        return self._run(cond, n, num_steps, guidance, self._gen(n, seed), {})

    @torch.no_grad()
    def generate_from_images(self, images: list[Image.Image], num_steps, guidance, seed):
        imgs = [im.convert("RGB").resize((self.size_px, self.size_px), Image.BILINEAR)
                for im in images]
        # _auto_resize=False: 512x512 is not a "preferred Kontext resolution",
        # so the pipeline would otherwise rescale and break decoder geometry.
        return self._run(imgs, len(imgs), num_steps, guidance,
                         self._gen(len(imgs), seed), {"_auto_resize": False})
