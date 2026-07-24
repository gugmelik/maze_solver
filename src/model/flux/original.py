from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor, nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel

from src.model.distributed import SelectiveDistributedMixin
from src.utils.model_utils import MAP_DTYPE


class Flux(SelectiveDistributedMixin, nn.Module):
    distributed_model_attr = "transformer"
    frozen_module_attrs = ("vae", "text_encoder", "text_encoder_2")

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        accelerator,
        max_sequence_length: int = 512,
        guidance_scale: float = 3.5,
    ) -> None:
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.accelerator = accelerator
        self.device = accelerator.device
        self.weight_dtype = MAP_DTYPE[accelerator.mixed_precision]
        self.max_sequence_length = max_sequence_length
        self.guidance_scale = guidance_scale

        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer",
        )
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="text_encoder",
        )
        self.text_encoder_2 = T5EncoderModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="vae",
        )
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="scheduler",
        )
        self.transformer = FluxTransformer2DModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="transformer",
        )


    def load_train_base_model(self) -> None:
        # The baseline FLUX LoRA path uses a single base checkpoint
        # for both training and inference, so there is nothing to swap here.
        return None

    def prepare_for_training(self) -> None:
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.transformer.requires_grad_(True)

        self.transformer.to(dtype=self.weight_dtype)
        self.text_encoder.to(dtype=self.weight_dtype)
        self.text_encoder_2.to(dtype=self.weight_dtype)
        self.vae.to(dtype=self.weight_dtype)

    def get_trainable_params(self, config) -> list[dict[str, Any]]:
        transformer_params = filter(lambda p: p.requires_grad, self.transformer.parameters())
        return [
            {
                "params": transformer_params,
                "lr": config.lr_for_transformer,
                "name": "transformer_params",
            }
        ]

    def get_state_dict(
        self,
        model_state_dict: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        transformer_weights = model_state_dict
        if transformer_weights is None:
            transformer_weights = self.get_unwrapped_distributed_model().state_dict()
        return {
            "transformer_weights": transformer_weights,
        }

    def load_state_dict_(self, state_dict: dict[str, Any]) -> None:
        self.get_unwrapped_distributed_model().load_state_dict(
            state_dict["transformer_weights"]
        )

    def forward(self, pixel_values: Tensor, prompt: Sequence[str], *args, **kwargs) -> dict[str, Tensor]:
        with torch.no_grad():
            pixel_values = pixel_values.to(self.device, dtype=self.vae.dtype)
            model_input = self.vae.encode(pixel_values).latent_dist.sample()
            model_input = (model_input - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            model_input = model_input.to(dtype=self.weight_dtype)

            prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(prompt)
            latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                model_input.shape[0],
                model_input.shape[2],# // 2,
                model_input.shape[3], # // 2,
                self.device,
                self.weight_dtype,
            )

            noise = torch.randn_like(model_input)
            batch_size = model_input.shape[0]
            timestep_indices = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (batch_size,)
            )
            timesteps = self.noise_scheduler.timesteps[timestep_indices].to(device=model_input.device)
            sigmas = self.get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
            noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

            packed_noisy_model_input = FluxPipeline._pack_latents(
                noisy_model_input,
                batch_size=model_input.shape[0],
                num_channels_latents=model_input.shape[1],
                height=model_input.shape[2],
                width=model_input.shape[3],
            )

            guidance = None
            transformer_config = self.get_distributed_model_attr("config")
            if getattr(transformer_config, "guidance_embeds", False):
                guidance = torch.full(
                    (batch_size,),
                    self.guidance_scale,
                    device=model_input.device,
                )

        with self.accelerator.autocast():
            model_pred = self.transformer(
                hidden_states=packed_noisy_model_input,
                timestep=timesteps / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

            vae_scale_factor = 2 ** len(self.vae.config.block_out_channels)
            model_pred = FluxPipeline._unpack_latents(
                model_pred,
                height=pixel_values.shape[-2],
                width=pixel_values.shape[-1],
                vae_scale_factor=vae_scale_factor,
            )

        return {
            "model_pred": model_pred,
            "target": noise - model_input,
        }

    def get_sigmas(self, timesteps: Tensor, n_dim: int = 4, dtype: torch.dtype = torch.float32) -> Tensor:
        sigmas = self.noise_scheduler.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(self.device)
        step_indices = [(schedule_timesteps == timestep).nonzero().item() for timestep in timesteps]
        sigma = sigmas[step_indices].flatten()
        while sigma.ndim < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def encode_prompt(self, prompt: Sequence[str] | str) -> tuple[Tensor, Tensor, Tensor]:
        prompt = [prompt] if isinstance(prompt, str) else list(prompt)
        prompt_embeds = self._encode_prompt_with_t5(prompt)
        pooled_prompt_embeds = self._encode_prompt_with_clip(prompt)
        text_ids = torch.zeros(
            prompt_embeds.shape[1],
            3,
            device=self.device,
            dtype=self.weight_dtype,
        )
        return (
            prompt_embeds.to(device=self.device, dtype=self.weight_dtype),
            pooled_prompt_embeds.to(device=self.device, dtype=self.weight_dtype),
            text_ids,
        )

    def _encode_prompt_with_t5(self, prompt: Sequence[str]) -> Tensor:
        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        prompt_embeds = self.text_encoder_2(text_inputs.input_ids.to(self.device))[0]
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.text_encoder_2.dtype)
        return prompt_embeds

    def _encode_prompt_with_clip(self, prompt: Sequence[str]) -> Tensor:
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )
        prompt_embeds = self.text_encoder(
            text_inputs.input_ids.to(self.device),
            output_hidden_states=False,
        ).pooler_output
        return prompt_embeds.to(device=self.device, dtype=self.text_encoder.dtype)


class Flux_v2(Flux):
    def forward(self, pixel_values: Tensor, prompt: Sequence[str], *args, **kwargs) -> dict[str, Tensor]:
        with torch.no_grad():
            pixel_values = pixel_values.to(self.device, dtype=self.vae.dtype)
            model_input = self.vae.encode(pixel_values).latent_dist.sample()
            model_input = (model_input - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            model_input = model_input.to(dtype=self.weight_dtype)

            vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels))

            latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                model_input.shape[0],
                model_input.shape[2] // 2,
                model_input.shape[3] // 2,
                self.device,
                self.weight_dtype,
            )

            noise = torch.randn_like(model_input)
            bsz = model_input.shape[0]

            prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(prompt)

            u = compute_density_for_timestep_sampling(
                weighting_scheme=None,
                batch_size=bsz,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
            indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
            timesteps = self.noise_scheduler.timesteps[indices].to(
                device=model_input.device
            )
            
            sigmas = self.get_sigmas(
                timesteps, n_dim=model_input.ndim, dtype=model_input.dtype
            )
            noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

            packed_noisy_model_input = FluxPipeline._pack_latents(
                noisy_model_input,
                batch_size=model_input.shape[0],
                num_channels_latents=model_input.shape[1],
                height=model_input.shape[2],
                width=model_input.shape[3],
            )

            guidance = torch.tensor(
                [1], device=self.device
            )
            guidance = guidance.expand(model_input.shape[0])

        with self.accelerator.autocast():
            model_pred = self.transformer(
                hidden_states=packed_noisy_model_input,
                timestep=timesteps / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

            model_pred = FluxPipeline._unpack_latents(
                model_pred,
                height=model_input.shape[2] * vae_scale_factor,
                width=model_input.shape[3] * vae_scale_factor,
                vae_scale_factor=vae_scale_factor,
            )

        return {
            "model_pred": model_pred,
            "target": noise - model_input,
        }