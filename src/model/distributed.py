from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import torch
from torch import nn


def get_adapter_state_dict(
    module: nn.Module,
    state_dict: dict[str, Any] | None = None,
    *,
    adapter_name: str = "default",
) -> dict[str, Any]:
    """
    Extract PEFT adapter weights without mistaking an FSDP2 mesh for TP.

    PEFT 0.19 detects FSDP2's internal ``_tp_plan`` as tensor parallelism and
    tries to import a newer Transformers TP integration. The supplied state is
    already materialized, so TP gathering must not run here.
    """
    from peft.utils import get_peft_model_state_dict

    had_tp_plan = hasattr(module, "_tp_plan")
    tp_plan = getattr(module, "_tp_plan", None)
    if had_tp_plan:
        module._tp_plan = None
    try:
        return get_peft_model_state_dict(
            module,
            state_dict=state_dict,
            adapter_name=adapter_name,
        )
    finally:
        if had_tp_plan:
            module._tp_plan = tp_plan


def materialize_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """
    Convert a possibly sharded state dict into ordinary CPU tensors.

    FSDP2 exposes parameters as DTensors. Calling ``full_tensor`` only for the
    selected trainable parameters keeps LoRA checkpoints small and avoids
    gathering the frozen diffusion backbone.
    """

    def materialize(value):
        if isinstance(value, Mapping):
            return {key: materialize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(materialize(item) for item in value)
        if hasattr(value, "full_tensor"):
            value = value.full_tensor()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        return value

    return {key: materialize(value) for key, value in state_dict.items()}


class DistributedModuleProxy(nn.Module):
    """
    Expose Diffusers model attributes while forwarding through a prepared model.

    DDP does not proxy attributes such as ``config`` and ``dtype``. Diffusers
    pipelines require those attributes, but their forward must still go through
    the DDP/FSDP-prepared module.
    """

    def __init__(self, prepared_module: nn.Module, original_module: nn.Module):
        super().__init__()
        self.prepared_module = prepared_module
        object.__setattr__(self, "_original_module", original_module)

    def forward(self, *args, **kwargs):
        return self.prepared_module(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "_original_module"), name)


class SelectiveDistributedMixin:
    """Common contract for wrapping only the trainable diffusion backbone."""

    distributed_model_attr: ClassVar[str]
    frozen_module_attrs: ClassVar[tuple[str, ...]] = ()
    checkpoint_trainable_only: ClassVar[bool] = False

    def get_distributed_model(self) -> nn.Module:
        return getattr(self, self.distributed_model_attr)

    def set_distributed_model(self, module: nn.Module) -> None:
        setattr(self, self.distributed_model_attr, module)

    def is_distributed_model_prepared(self) -> bool:
        return any(
            prepared_model is self.get_distributed_model()
            for prepared_model in getattr(self.accelerator, "_models", ())
        )

    def get_unwrapped_distributed_model(self) -> nn.Module:
        return self.accelerator.unwrap_model(
            self.get_distributed_model(),
            keep_fp32_wrapper=False,
        )

    def get_pipeline_distributed_model(self) -> nn.Module:
        prepared_module = self.get_distributed_model()
        original_module = self.get_unwrapped_distributed_model()
        if prepared_module is original_module:
            return original_module
        return DistributedModuleProxy(prepared_module, original_module)

    def get_distributed_model_attr(self, name: str):
        return getattr(self.get_unwrapped_distributed_model(), name)

    def move_frozen_modules_to_device(self, device: torch.device) -> None:
        for name in self.frozen_module_attrs:
            module = getattr(self, name, None)
            if isinstance(module, nn.Module):
                module.to(device)

    def get_trainable_model_state_dict(self) -> dict[str, Any]:
        """
        Return only raw trainable parameter entries from the prepared model.

        The raw keys are retained for distributed resume. Model-specific
        ``get_state_dict`` methods convert this state to the portable checkpoint
        format used by existing inference code.
        """
        module = self.get_unwrapped_distributed_model()
        return self.select_trainable_model_state_dict(module.state_dict())

    def select_trainable_model_state_dict(
        self,
        state_dict: Mapping[str, Any],
    ) -> dict[str, Any]:
        module = self.get_unwrapped_distributed_model()
        trainable_names = {
            name
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }
        selected_state = {
            name: value
            for name, value in state_dict.items()
            if name in trainable_names
        }
        return materialize_state_dict(selected_state)

    def load_prepared_model_state_dict(
        self,
        state_dict: dict[str, Any],
        *,
        strict: bool = False,
        broadcast_from_rank0: bool = False,
        ignore_frozen_params: bool | None = None,
    ):
        """
        Load a full-tensor state dict into DDP, FSDP1, FSDP2, or one process.
        """
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )

        broadcast_from_rank0 = (
            broadcast_from_rank0
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            ignore_frozen_params=(
                self.checkpoint_trainable_only
                if ignore_frozen_params is None
                else ignore_frozen_params
            ),
            strict=strict,
            broadcast_from_rank0=broadcast_from_rank0,
        )
        return set_model_state_dict(
            self.get_distributed_model(),
            state_dict,
            options=options,
        )
