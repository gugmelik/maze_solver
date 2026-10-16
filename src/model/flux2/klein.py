from __future__ import annotations

import copy
from typing import Any, Sequence

import torch
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline, Flux2Transformer2DModel
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from peft import LoraConfig, set_peft_model_state_dict
from diffusers.utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
from torch import Tensor, nn
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from src.model.distributed import SelectiveDistributedMixin, get_adapter_state_dict


class Flux2Klein(SelectiveDistributedMixin, nn.Module):
    """
    FLUX.2 [klein] 4B loaded in bf16 with no quantization config.
    """

    distributed_model_attr = "transformer"
    frozen_module_attrs = ("vae", "text_encoder")

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        accelerator,
        max_sequence_length: int = 128,
        guidance_scale: float = 1.0,
        text_encoder_out_layers: Sequence[int] = (9, 18, 27),
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.accelerator = accelerator
        self.device = accelerator.device
        self.weight_dtype = torch.bfloat16
        self.max_sequence_length = max_sequence_length
        self.guidance_scale = guidance_scale
        self.text_encoder_out_layers = tuple(text_encoder_out_layers)
        self.gradient_checkpointing = gradient_checkpointing

        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "quantization_config": None,
        }
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
        )
        self.text_encoder = Qwen3ForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            **load_kwargs,
        )
        self.vae = AutoencoderKLFlux2.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
            **load_kwargs,
        )
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="scheduler",
        )
        # Validation calls set_timesteps on the pipeline scheduler. Training
        # keeps its own 1000-step schedule so those indices stay in range.
        self.train_scheduler = copy.deepcopy(self.noise_scheduler)
        self.transformer = Flux2Transformer2DModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="transformer",
            **load_kwargs,
        )
        self._assert_bf16_unquantized()

    def _assert_bf16_unquantized(self) -> None:
        for name in ("transformer", "text_encoder", "vae"):
            module = getattr(self, name)
            if getattr(module, "is_loaded_in_4bit", False) or getattr(module, "is_loaded_in_8bit", False):
                raise RuntimeError(f"{name} was loaded quantized; expected full bf16 weights.")
            dtype = next(module.parameters()).dtype
            if dtype != torch.bfloat16:
                raise RuntimeError(f"{name} dtype is {dtype}, expected torch.bfloat16.")

    def load_train_base_model(self) -> None:
        return None

    def prepare_for_training(self) -> None:
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.transformer.requires_grad_(True)
        self.vae.eval()
        self.text_encoder.eval()
        if self.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

    def get_trainable_params(self, config) -> list[dict[str, Any]]:
        transformer_params = filter(lambda parameter: parameter.requires_grad, self.transformer.parameters())
        return [
            {
                "params": transformer_params,
                "lr": config.lr_for_transformer,
                "name": "transformer_params",
            }
        ]

    def get_state_dict(self, model_state_dict: dict[str, Any] | None = None) -> dict[str, Any]:
        transformer_weights = model_state_dict
        if transformer_weights is None:
            transformer_weights = self.get_unwrapped_distributed_model().state_dict()
        return {"transformer_weights": transformer_weights}

    def load_state_dict_(self, state_dict: dict[str, Any]) -> None:
        self.get_unwrapped_distributed_model().load_state_dict(state_dict["transformer_weights"])

    def encode_prompt(self, prompt: Sequence[str] | str) -> tuple[Tensor, Tensor]:
        prompt = [prompt] if isinstance(prompt, str) else list(prompt)
        prompt_embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            prompt=prompt,
            device=self.device,
            max_sequence_length=self.max_sequence_length,
            hidden_states_layers=list(self.text_encoder_out_layers),
        )
        text_ids = Flux2KleinPipeline._prepare_text_ids(prompt_embeds).to(self.device)
        return prompt_embeds.to(dtype=self.weight_dtype), text_ids

    def encode_vae_image(self, pixel_values: Tensor) -> Tensor:
        latents = self.vae.encode(pixel_values).latent_dist.mode()
        latents = Flux2KleinPipeline._patchify_latents(latents)
        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(device=latents.device, dtype=latents.dtype)
        bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(device=latents.device, dtype=latents.dtype)
        return (latents - bn_mean) / bn_std

    def get_sigmas(self, timesteps: Tensor, n_dim: int, dtype: torch.dtype) -> Tensor:
        sigmas = self.train_scheduler.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.train_scheduler.timesteps.to(self.device)
        step_indices = [(schedule_timesteps == timestep).nonzero().item() for timestep in timesteps]
        sigma = sigmas[step_indices].flatten()
        while sigma.ndim < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def forward(
        self,
        pixel_values: Tensor,
        cond_pixel_values: Tensor,
        prompt: Sequence[str],
        *args,
        **kwargs,
    ) -> dict[str, Tensor]:
        with torch.no_grad():
            pixel_values = pixel_values.to(self.device, dtype=self.vae.dtype)
            cond_pixel_values = cond_pixel_values.to(self.device, dtype=self.vae.dtype)
            model_input = self.encode_vae_image(pixel_values).to(dtype=self.weight_dtype)
            cond_input = self.encode_vae_image(cond_pixel_values).to(dtype=self.weight_dtype)
            prompt_embeds, text_ids = self.encode_prompt(prompt)

            noise = torch.randn_like(model_input)
            batch_size = model_input.shape[0]
            timesteps_u = compute_density_for_timestep_sampling(
                weighting_scheme="none",
                batch_size=batch_size,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
            indices = (timesteps_u * self.train_scheduler.config.num_train_timesteps).long()
            timesteps = self.train_scheduler.timesteps[indices].to(device=model_input.device)
            sigmas = self.get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
            noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

            model_input_ids = Flux2KleinPipeline._prepare_latent_ids(model_input).to(device=model_input.device)
            cond_input_ids = Flux2KleinPipeline._prepare_image_ids([cond_input[0:1]]).to(device=cond_input.device)
            cond_input_ids = cond_input_ids.expand(cond_input.shape[0], -1, -1)

            packed_noisy = Flux2KleinPipeline._pack_latents(noisy_model_input)
            packed_cond = Flux2KleinPipeline._pack_latents(cond_input)
            target_tokens = packed_noisy.shape[1]
            target_ids = model_input_ids.shape[1]
            packed_hidden = torch.cat([packed_noisy, packed_cond], dim=1)
            packed_ids = torch.cat([model_input_ids, cond_input_ids], dim=1)

            guidance = None
            if getattr(self.get_distributed_model_attr("config"), "guidance_embeds", False):
                guidance = torch.full((batch_size,), self.guidance_scale, device=model_input.device)
            loss_weight = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)

        with self.accelerator.autocast():
            model_pred = self.transformer(
                hidden_states=packed_hidden,
                timestep=timesteps / 1000,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=packed_ids,
                return_dict=False,
            )[0]
            model_pred = model_pred[:, :target_tokens, :]
            model_input_ids = packed_ids[:, :target_ids, :]
            model_pred = Flux2KleinPipeline._unpack_latents_with_ids(model_pred, model_input_ids)

        return {
            "model_pred": model_pred,
            "target": noise - model_input,
            "loss_weight": loss_weight,
        }


class Flux2KleinLora(Flux2Klein):
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
        target_modules = list(self.lora_modules)
        for name, module in self.transformer.named_modules():
            if name.endswith("attn.to_out") and isinstance(module, nn.Linear):
                target_modules.append(name)
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank,
            init_lora_weights=self.init_lora_weights,
            target_modules=target_modules,
        )
        self.transformer.add_adapter(lora_config)
        if self.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
        for parameter in self.transformer.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.to(torch.float32)
        self._assert_bf16_unquantized_base()

    def _assert_bf16_unquantized_base(self) -> None:
        frozen = [parameter for parameter in self.transformer.parameters() if not parameter.requires_grad]
        if not frozen:
            raise RuntimeError("FLUX.2 Klein transformer has no frozen base weights.")
        dtype = frozen[0].dtype
        if dtype != torch.bfloat16:
            raise RuntimeError(f"Frozen Klein transformer weights are {dtype}, expected torch.bfloat16.")

    def get_trainable_params(self, config) -> list[dict[str, Any]]:
        lora_params = filter(lambda parameter: parameter.requires_grad, self.transformer.parameters())
        return [
            {
                "params": lora_params,
                "lr": config.lr_for_lora,
                "name": "lora_params",
            }
        ]

    def get_state_dict(self, model_state_dict: dict[str, Any] | None = None) -> dict[str, Any]:
        transformer = self.get_unwrapped_distributed_model()
        return {
            "transformer_lora_weights": convert_state_dict_to_diffusers(
                get_adapter_state_dict(transformer, model_state_dict)
            ),
        }

    def load_state_dict_(self, state_dict: dict[str, Any]) -> None:
        transformer_state_dict = convert_unet_state_dict_to_peft(state_dict["transformer_lora_weights"])
        incompatible_keys = set_peft_model_state_dict(
            self.get_unwrapped_distributed_model(),
            transformer_state_dict,
            adapter_name="default",
        )
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                raise ValueError(f"Unexpected FLUX.2 Klein LoRA keys: {unexpected_keys}")
