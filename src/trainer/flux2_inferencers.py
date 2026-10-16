from __future__ import annotations

import torch

from src.trainer.flux2_trainers import Flux2KleinTrainer, _first, _tensor_to_pil
from src.trainer.base_inferencer import BaseInferencer


class Flux2KleinInferencer(BaseInferencer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._artifact_counters = {}

    @torch.no_grad()
    def process_evaluation_batch(self, batch, eval_metrics):
        return Flux2KleinTrainer.process_evaluation_batch(self, batch, eval_metrics)

    def _handle_evaluation_artifacts(self, epoch, part, batch, *, keep):
        if not keep:
            return

        generated_images = batch.get("generated")
        if not generated_images:
            return

        output_dir = self._get_partition_output_dir(epoch, part)
        output_dir.mkdir(parents=True, exist_ok=True)

        sample_id = batch.get("id")
        if isinstance(sample_id, (list, tuple)):
            sample_id = sample_id[0]
        if sample_id:
            stem = str(sample_id)
            generated_images[0].save(output_dir / f"{stem}.png")
            return

        # Prompt text is shared across Maze samples, so keep a local counter.
        key = (epoch, part)
        sample_idx = self._artifact_counters.get(key, 0)
        self._artifact_counters[key] = sample_idx + 1
        stem = f"sample_{sample_idx:03d}"
        generated_images[0].save(output_dir / f"{stem}_generated.jpg")

        if "cond_pixel_values" in batch:
            _tensor_to_pil(batch["cond_pixel_values"]).save(output_dir / f"{stem}_condition.jpg")
        if "pixel_values" in batch:
            _tensor_to_pil(batch["pixel_values"]).save(output_dir / f"{stem}_target.jpg")
