from __future__ import annotations

import numpy as np
from PIL import Image

from src.metrics.base_metric import BaseMetric


def _as_uint8(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    tensor = image.detach().float().cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]
    if float(tensor.min()) < 0:
        tensor = tensor * 0.5 + 0.5
    return (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def _red_mask(image: np.ndarray) -> np.ndarray:
    red = image[..., 0] > 160
    green = image[..., 1] < 90
    blue = image[..., 2] < 90
    return red & green & blue


def _mask_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def _pixel_match(prediction: np.ndarray, target: np.ndarray, tolerance: int = 24) -> float:
    difference = np.abs(prediction.astype(np.int16) - target.astype(np.int16)).max(axis=-1)
    return float((difference <= tolerance).mean())


class EndoCoTAccuracy(BaseMetric):
    """
    Validation accuracy against the ground-truth edited image.

    Maze samples count as correct when the generated red path overlaps the
    ground-truth path. Other EndoCoT tasks count as correct when enough pixels
    match the target image.
    """

    def __init__(self, maze_iou_threshold: float = 0.5, pixel_threshold: float = 0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maze_iou_threshold = maze_iou_threshold
        self.pixel_threshold = pixel_threshold

    def to_cuda(self):
        return None

    def to_cpu(self):
        return None

    def __call__(self, generated, pixel_values, task, **batch):
        prediction = generated[0] if isinstance(generated, (list, tuple)) else generated
        task_name = task[0] if isinstance(task, (list, tuple)) else task
        target = _as_uint8(pixel_values)
        prediction = prediction.convert("RGB").resize((target.shape[1], target.shape[0]), Image.Resampling.BILINEAR)
        prediction = np.asarray(prediction)

        if str(task_name).lower() == "maze":
            score = _mask_iou(_red_mask(prediction), _red_mask(target))
            correct = score >= self.maze_iou_threshold
        else:
            score = _pixel_match(prediction, target)
            correct = score >= self.pixel_threshold
        return {"accuracy": float(correct)}
