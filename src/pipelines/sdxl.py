import torch
from diffusers.pipelines import StableDiffusionXLPipeline
from diffusers import DDIMScheduler

class SDXLPipeline:
    @staticmethod
    def from_pretrained(model, accelerator, *args, **kwargs):
        if "torch_dtype" in kwargs:
            kwargs["torch_dtype"] = getattr(torch,  kwargs["torch_dtype"])
        
        unet = model.get_unwrapped_distributed_model()
        scheduler = DDIMScheduler.from_pretrained(
            kwargs["pretrained_model_name_or_path"], 
            subfolder="scheduler"
        )
        
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            scheduler=scheduler,
            tokenizer=model.tokenizer,
            tokenizer_2=model.tokenizer_2,
            text_encoder=model.text_encoder,
            text_encoder_2=model.text_encoder_2,
            unet=unet,
            vae=model.vae,
            *args, 
            **kwargs
        )
        object.__setattr__(
            pipeline,
            "unet",
            model.get_pipeline_distributed_model(),
        )
        pipeline.set_progress_bar_config(disable=True)
        return pipeline
        