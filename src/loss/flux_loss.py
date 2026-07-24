from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FluxFlowMatchLoss(nn.Module):
    def forward(
        self,
        model_pred: Tensor,
        target: Tensor,
        loss_weight: Optional[Tensor] = None,
        **batch,
    ) -> dict[str, Tensor]:
        loss = (model_pred.float() - target.float()) ** 2

        if loss_weight is not None:
            weight = loss_weight.float()
            while weight.ndim < loss.ndim:
                weight = weight.unsqueeze(-1)
            loss = loss * weight

        loss = loss.reshape(loss.shape[0], -1).mean(dim=1).mean()
        return {"loss": loss, "mse": loss}


class FluxFlowMatchCrossEntropyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.md_loss = FluxFlowMatchLoss()

    def forward(self, **batch):
        if batch["do_embeds_step"]:
            ce_target = batch["ce_target"]
            ce_pred = batch["ce_pred"]
            
            ce_target = ce_target.long()
            loss_ce = F.cross_entropy(ce_pred.float(), ce_target)
            
            with torch.no_grad():
                labels_pred = ce_pred.argmax(dim=-1)
                acc = (labels_pred == ce_target).float().mean()
            loss = loss_ce  
            result = {'loss': loss, "ce_loss": loss, "acc": acc}
        else:
            result = self.md_loss(**batch)
        return result
