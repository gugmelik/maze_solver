import torch
from torch import nn
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from src.utils.model_utils import MAP_DTYPE
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from src.model.distributed import SelectiveDistributedMixin


class SDXL(SelectiveDistributedMixin, nn.Module):
    distributed_model_attr = "unet"
    frozen_module_attrs = ("vae", "text_encoder", "text_encoder_2")

    def __init__(self, pretrained_model_name_or_path, accelerator):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        self.weight_dtype = MAP_DTYPE[accelerator.mixed_precision]
        self.accelerator = accelerator
        self.device = accelerator.device

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer",
            use_fast=False,
        )

        self.tokenizer_2 = AutoTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            use_fast=False,
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="text_encoder"
        )
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="text_encoder_2"
        )

        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="scheduler"
        )

        self.vae = AutoencoderKL.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="vae",
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_model_name_or_path, 
            subfolder="unet",
        )
        
    def load_train_base_model(self):
        pass

    def prepare_for_training(self):
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.unet.requires_grad_(True)
        

        self.unet.to(dtype=self.weight_dtype)
        self.vae.to(dtype=torch.float32)
        self.text_encoder.to(dtype=self.weight_dtype)
        self.text_encoder_2.to(dtype=self.weight_dtype)

    def get_trainable_params(self, config):
        unet_params = filter(lambda p: p.requires_grad, self.unet.parameters())
        trainable_params = [
            {'params': unet_params, 'lr': config.lr_for_unet, 'name': 'unet_params'},
        ]
        return trainable_params

    def get_state_dict(self, model_state_dict=None):
        unet_weights = model_state_dict
        if unet_weights is None:
            unet_weights = self.get_unwrapped_distributed_model().state_dict()
        return {
            'unet_weights': unet_weights,
        }

    def load_state_dict_(self, state_dict):
        self.get_unwrapped_distributed_model().load_state_dict(
            state_dict['unet_weights']
        )

    def forward(self, pixel_values, prompt, original_sizes, crop_top_lefts, target_sizes, do_cfg=False, *args, **kwargs):
        with torch.no_grad():
            pixel_values = pixel_values.to(self.device, self.vae.dtype)
            model_input = self.vae.encode(pixel_values).latent_dist.sample()
            model_input = model_input.to(self.weight_dtype) * self.vae.config.scaling_factor
                        
            # Sample noise that we'll add to the latents
            noise = torch.randn_like(model_input)

            bsz = model_input.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device 
            )
            timesteps = timesteps.long()

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = self.noise_scheduler.add_noise(model_input, noise, timesteps)
            
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
            add_time_ids = [
                self._get_add_time_ids(
                    s, c, t, 
                    dtype=noisy_model_input.dtype, 
                    text_encoder_projection_dim=text_encoder_projection_dim
                ) for s, c, t in zip(original_sizes, crop_top_lefts, target_sizes)
            ]
            add_time_ids = torch.cat(add_time_ids).to(model_input.device)
            
            #*********************************************************
            prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                do_cfg=do_cfg
            )

            #*********************************************************

            add_text_embeds = pooled_prompt_embeds
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        with self.accelerator.autocast():
            model_pred = self.unet(
                noisy_model_input,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

        target = noise
                
        return {
            'model_pred': model_pred,
            'target': target,
        }


    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        # From pipeline.StableDiffusionXLPipeline._get_add_time_ids
        add_time_ids = list(tuple(original_size) + crops_coords_top_left + target_size)

        unet = self.get_unwrapped_distributed_model()
        passed_add_embed_dim = (
            unet.config.addition_time_embed_dim * len(add_time_ids)
            + text_encoder_projection_dim
        )
        expected_add_embed_dim = unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids
        
    def encode_prompt(self, prompt, do_cfg=False):
        prompt = prompt if not do_cfg else [""] * len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        text_inputs_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs_2.input_ids

        prompt_embeds = self.text_encoder(
            text_input_ids.to(self.text_encoder.device), 
            output_hidden_states=True, 
        )
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]

        prompt_embeds_2 = self.text_encoder_2(
            text_input_ids_2.to(self.text_encoder_2.device), 
            output_hidden_states=True, 
        )

        pooled_prompt_embeds_2 = prompt_embeds_2[0]
        prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]


        prompt_embeds = torch.concat([prompt_embeds, prompt_embeds_2], dim=-1)
        bs_embed = prompt_embeds.shape[0]
        pooled_prompt_embeds = pooled_prompt_embeds_2.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds