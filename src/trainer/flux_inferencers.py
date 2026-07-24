from __future__ import annotations

import torch

from src.trainer.base_inferencer import BaseInferencer


class FluxInferencer(BaseInferencer):
    @torch.no_grad()
    def process_evaluation_batch(self, batch, eval_metrics):
        batch = self.transform_batch(batch)
        seed = self.config.validation_args.get("seed", 0)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        generated_images = self.pipe(
            prompt=batch["prompt"],
            generator=generator,
            **self.config.validation_args,
        ).images

        batch["generated"] = generated_images
        for metric in self.metrics:
            metric_result = metric(**batch)
            for key, value in metric_result.items():
                eval_metrics.update(key, value)

        return batch
