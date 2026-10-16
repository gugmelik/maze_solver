from __future__ import annotations

import torch
from PIL import Image

from src.metrics.tracker import MetricTracker
from src.trainer.flux_trainers import FluxTrainer


def _first(value):
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim == 4:
        image = image[0]
    image = image.detach().float().cpu()
    if image.min() < 0:
        image = image * 0.5 + 0.5
    array = (image.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    return Image.fromarray(array)


class Flux2KleinTrainer(FluxTrainer):
    def process_evaluation_batch(self, batch, eval_metrics):
        batch = self.transform_batch(batch)
        seed = self.config.validation_args.get("seed", 0)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        pipeline_args = {
            key: value
            for key, value in self.config.validation_args.items()
            if key != "seed"
        }
        condition = _tensor_to_pil(batch["cond_pixel_values"])
        generated_images = self.pipe(
            prompt=_first(batch["prompt"]),
            image=condition,
            generator=generator,
            **pipeline_args,
        ).images

        batch["generated"] = generated_images
        for metric in self.metrics:
            metric_result = metric(**batch)
            for key, value in metric_result.items():
                eval_metrics.update(key, value)
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        if mode == "train":
            return
        # Avoid flooding W&B when validating on the full Maze test suites.
        if batch_idx >= 2:
            return
        generated_img = batch["generated"][0].resize((256, 256))
        sample_id = batch.get("id")
        if isinstance(sample_id, (list, tuple)):
            sample_id = sample_id[0]
        suffix = str(sample_id) if sample_id else f"sample_{batch_idx:03d}"
        image_name = f"{mode}_images/{suffix}"
        self.writer.add_image(image_name, generated_img)
