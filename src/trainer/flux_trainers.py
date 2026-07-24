from __future__ import annotations

import torch

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class FluxTrainer(BaseTrainer):
    def process_batch(self, batch, train_metrics: MetricTracker):
        batch = self.transform_batch(batch)

        with self.accelerator.accumulate(self.model.get_distributed_model()):
            output = self.model(**batch)
            batch.update(output)

            all_losses = self.criterion(**batch)
            batch.update(all_losses)

            if self.is_train:
                assert torch.isfinite(batch["loss"])
                self.accelerator.backward(batch["loss"])
                if self.accelerator.sync_gradients:
                    self._clip_grad_norm()
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self._log_grad_norms(train_metrics)
                    self.optimizer.zero_grad()

            for loss_name in self.config.writer.loss_names:
                if loss_name in batch:
                    batch[loss_name] = self.accelerator.gather(batch[loss_name]).mean()
                    train_metrics.update(loss_name, batch[loss_name].item())
        return batch

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

    def _log_batch(self, batch_idx, batch, mode="train"):
        if mode == "train":
            return

        prompt = batch["prompt"]
        generated_img = batch["generated"][0].resize((256, 256))
        cutted_prompt = prompt.replace(" ", "_").replace(",", "")[:40]
        image_name = f"{mode}_images/{cutted_prompt or batch_idx}"
        self.writer.add_image(image_name, generated_img)
