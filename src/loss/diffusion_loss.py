import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    def forward(self, model_pred, target, **batch):
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return {'loss': loss}


class MaskedDiffusionLoss(nn.Module):
    def forward(self, model_pred, target, is_masked_loss, bbox, **batch):
        if is_masked_loss:
            loss = 0
            for i, box in enumerate(bbox):
                scaled_box = np.array(box) / 8
                scaled_box = scaled_box.astype(np.int32)
                
                model_pred_i = model_pred[i, :, scaled_box[1]:scaled_box[3], scaled_box[0]:scaled_box[2]]
                target_i = target[i, :, scaled_box[1]:scaled_box[3], scaled_box[0]:scaled_box[2]]
                
                loss = loss + F.mse_loss(model_pred_i.float(), target_i.float(), reduction="mean")
            
            loss = loss / len(bbox)
            result = {"loss": loss, "masked_mse": loss}
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            result = {"loss": loss, "mse": loss}
            
        return result

