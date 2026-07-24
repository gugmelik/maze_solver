from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, set_peft_model_state_dict

from diffusers.utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
)

from src.model.flux.original import Flux
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from diffusers import FluxPipeline

from src.model.distributed import get_adapter_state_dict
from src.utils.model_utils import MAP_DTYPE


class FluxLora(Flux):
    FORMAT_VERSION = 1
    checkpoint_trainable_only = True

    def __init__(
        self,
        rank: int,
        lora_modules: list[str],
        init_lora_weights: str,
        *args,
        **kwargs,
    ) -> None:
        self.lora_rank = rank
        self.lora_modules = lora_modules
        self.init_lora_weights = init_lora_weights
        super().__init__(*args, **kwargs)

    def prepare_for_training(self) -> None:
        super().prepare_for_training()
        self.transformer.requires_grad_(False)

        transformer_lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank,
            init_lora_weights=self.init_lora_weights,
            target_modules=self.lora_modules,
        )
        self.transformer.add_adapter(transformer_lora_config)

        for parameter in self.transformer.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.to(torch.float32)

    def get_trainable_params(self, config) -> list[dict[str, Any]]:
        lora_params = filter(lambda p: p.requires_grad, self.transformer.parameters())
        return [
            {
                "params": lora_params,
                "lr": config.lr_for_lora,
                "name": "lora_params",
            }
        ]

    def get_state_dict(
        self,
        model_state_dict: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        transformer = self.get_unwrapped_distributed_model()
        return {
            "transformer_lora_weights": convert_state_dict_to_diffusers(
                get_adapter_state_dict(
                    transformer,
                    model_state_dict,
                )
            ),
        }

    def load_state_dict_(self, state_dict: dict[str, Any]) -> None:
        transformer_state_dict = state_dict["transformer_lora_weights"]
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(
            self.get_unwrapped_distributed_model(),
            transformer_state_dict,
            adapter_name="default",
        )
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                raise ValueError(
                    "Loading FLUX LoRA weights from state_dict led to unexpected keys "
                    f"not found in the model: {unexpected_keys}"
                )


class FluxLora2(FluxLora):
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