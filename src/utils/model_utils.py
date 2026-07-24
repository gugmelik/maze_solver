import torch
import numpy as np
from transformers import PretrainedConfig
from scipy.ndimage import distance_transform_edt
from contextlib import contextmanager


MAP_DTYPE = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "no": torch.float32
}

@contextmanager
def freeze_params(module):
    old = [p.requires_grad for p in module.parameters()]
    for p in module.parameters():
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, r in zip(module.parameters(), old):
            p.requires_grad_(r)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def cos_sim(arr1, arr2):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    return arr1 @ arr2 / np.linalg.norm(arr1) / np.linalg.norm(arr2)


def get_bigger_mask(mask_np, p):
    area = int(mask_np.sum())
    if area < 10:
        print("SMALL AREA")
        return mask_np
    total = mask_np.size
    target = int(np.ceil(area * (1.0 + p)))
    target = min(target, total)

    assert target > area, (target, area)
    dist = distance_transform_edt(~mask_np)

    need = target - area  
    bg_dist = dist[~mask_np]  

    if need >= bg_dist.size:
        out = np.ones_like(mask_np, dtype=bool)
    else:

        r = float(sorted(bg_dist)[need - 1])
        eps = 1e-6 
        out = dist <= (r + eps)
    return out


class BaseTimer:
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def stop(self):
        self.end.record()
        torch.cuda.synchronize()
        return self.start.elapsed_time(self.end) / 1000