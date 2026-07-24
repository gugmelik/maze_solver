from __future__ import annotations

import torch

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import FluxPipeline as DiffusersFluxPipeline


class FluxPipeline:
    @staticmethod
    def from_pretrained(model, accelerator, *args, **kwargs):
        if "torch_dtype" in kwargs and isinstance(kwargs["torch_dtype"], str):
            kwargs["torch_dtype"] = getattr(torch, kwargs["torch_dtype"])
        pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path")

        transformer = model.get_unwrapped_distributed_model()
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="scheduler",
        )

        pipeline = DiffusersFluxPipeline.from_pretrained(
            pretrained_model_name_or_path,
            scheduler=scheduler,
            tokenizer=model.tokenizer,
            tokenizer_2=model.tokenizer_2,
            text_encoder=model.text_encoder,
            text_encoder_2=model.text_encoder_2,
            transformer=transformer,
            vae=model.vae,
            *args,
            **kwargs,
        )
        object.__setattr__(
            pipeline,
            "transformer",
            model.get_pipeline_distributed_model(),
        )
        pipeline.set_progress_bar_config(disable=True)
        return pipeline
