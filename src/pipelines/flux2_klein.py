from __future__ import annotations

import torch
from diffusers import Flux2KleinPipeline as DiffusersFlux2KleinPipeline


class Flux2KleinPipeline:
    @staticmethod
    def from_pretrained(model, accelerator, *args, **kwargs):
        if "torch_dtype" in kwargs and isinstance(kwargs["torch_dtype"], str):
            kwargs["torch_dtype"] = getattr(torch, kwargs["torch_dtype"])
        pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path")
        kwargs.setdefault("torch_dtype", torch.bfloat16)

        pipeline = DiffusersFlux2KleinPipeline.from_pretrained(
            pretrained_model_name_or_path,
            tokenizer=model.tokenizer,
            text_encoder=model.text_encoder,
            transformer=model.get_unwrapped_distributed_model(),
            vae=model.vae,
            scheduler=model.noise_scheduler,
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
