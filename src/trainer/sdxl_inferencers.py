import torch

from src.trainer.base_inferencer import BaseInferencer


class SDXLInferencer(BaseInferencer):
    @torch.no_grad()
    def process_evaluation_batch(self, batch, eval_metrics):
        batch = self.transform_batch(batch)
        seed = self.config.validation_args.get("seed", 0)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        generated_images = self.pipe(
            prompt=batch['prompt'],
            generator=generator,
            **self.config.validation_args
        ).images

        batch['generated'] = generated_images

        for metric in self.metrics:
            metric_result = metric(**batch)
            for k, v in metric_result.items():
                eval_metrics.update(k, v)
                
        return batch

    
    def _evaluation_epoch(self, *args, **kwargs):
        self.pipe.vae.to(dtype=torch.float16)
        try:
            return super()._evaluation_epoch(*args, **kwargs)
        finally:
            self.pipe.vae.to(dtype=torch.float32)


class PhotomakerInferencer(BaseInferencer):
    def _log_batch(self, batch_idx, batch, mode):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        prompt = batch['prompt']
        id = batch['id']
        generated_img = batch['generated'][0].resize((256, 256))

        cutted_prompt = prompt.replace(" ", "_")[:30]
        image_name = f"{mode}_images/{id}/{cutted_prompt}"
        self.writer.add_image(image_name, generated_img)
        
            
    def _evaluation_epoch(self, *args, **kwargs):
        self.accelerator.wait_for_everyone()
        self.model.load_inference_base_model()
        self.accelerator.wait_for_everyone()
        self.pipe.vae.to(dtype=torch.float16)
        try:
            return super()._evaluation_epoch(*args, **kwargs)
        finally:
            self.pipe.vae.to(dtype=torch.float32)
            self.accelerator.wait_for_everyone()
            self.model.load_train_base_model()
            self.accelerator.wait_for_everyone()
    

    @torch.no_grad()
    def process_evaluation_batch(self, batch, eval_metrics):
        batch = self.transform_batch(batch)
        seed = self.config.validation_args.get("seed", 0)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        generated_images = self.pipe(
            prompt=batch['prompt'],
            input_id_images=batch['ref_images'],
            generator=generator,
            **self.config.validation_args
        ).images

        batch['generated'] = generated_images

        for metric in self.metrics:
            metric_result = metric(**batch)
            for k, v in metric_result.items():
                eval_metrics.update(k, v)
                
        return batch
