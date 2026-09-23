"""Model backends. Pick one with `--backend`."""

from __future__ import annotations

from .base import (LORA_WEIGHTS_FILE, Backend, bnb_config, flow_loss, get_sigmas,
                   load_lora, pack, sample_noisy, save_lora)
from .flux_kontext import FluxKontextBackend
from .qwen_edit import QwenEditBackend

_CLASSES = [FluxKontextBackend, QwenEditBackend]
BACKENDS = {c.name: c for c in _CLASSES}


def get_backend(name: str) -> Backend:
    try:
        return BACKENDS[name]()
    except KeyError:
        raise SystemExit(
            f"unknown backend {name!r}; choose from {', '.join(sorted(BACKENDS))}") from None


def backend_names() -> list[str]:
    return sorted(BACKENDS)


__all__ = ["Backend", "BACKENDS", "get_backend", "backend_names", "pack",
           "sample_noisy", "flow_loss", "get_sigmas", "bnb_config",
           "save_lora", "load_lora", "LORA_WEIGHTS_FILE"]
