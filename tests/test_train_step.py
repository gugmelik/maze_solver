"""Verify the training step without downloading 34 GB of FLUX weights.

Builds a randomly-initialised, deliberately tiny FluxTransformer2DModel with the
same interface as FLUX.1-Kontext-dev and pushes one real step through
`flow_matching_loss`. Catches the things that actually break silently: wrong
sequence concatenation, a missing reference-image tag, LoRA attached to nothing,
or gradients leaking into the frozen base.

Needs a GPU but only ~50 MB of VRAM.

    python tests/test_train_step.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mazelora.flux_utils import latent_ids, pack
from mazelora.train import flow_matching_loss

FAILS: list[str] = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def main():
    if not torch.cuda.is_available():
        print("no CUDA device; skipping")
        return 0
    from diffusers import FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig

    dev, H = "cuda", 16
    tr = FluxTransformer2DModel(
        patch_size=1, in_channels=64, num_layers=2, num_single_layers=2,
        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=4096,
        pooled_projection_dim=768, guidance_embeds=True, axes_dims_rope=(4, 6, 6),
    ).to(dev, torch.bfloat16)
    tr.requires_grad_(False)
    tr.enable_gradient_checkpointing()
    tr.add_adapter(LoraConfig(r=8, lora_alpha=8, init_lora_weights="gaussian",
                              target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    cast_training_params(tr, dtype=torch.float32)
    named = [(n, p) for n, p in tr.named_parameters() if p.requires_grad]

    print("\nLoRA attachment")
    check("adapter found target modules", len(named) > 0, f"{len(named)} tensors")
    check("only LoRA params are trainable", all("lora" in n for n, _ in named))
    check("adapters are fp32 under bf16 autocast",
          all(p.dtype == torch.float32 for _, p in named))

    tgt = torch.randn(2, 16, H, H, device=dev, dtype=torch.bfloat16)
    cnd = torch.randn(2, 16, H, H, device=dev, dtype=torch.bfloat16)
    tp, cp = pack(tgt), pack(cnd)
    ids = torch.cat([latent_ids(H, H, dev, torch.bfloat16, False),
                     latent_ids(H, H, dev, torch.bfloat16, True)], dim=0)

    print("\nKontext conditioning layout")
    check("packing is [B, (H/2)(W/2), C*4]", tuple(tp.shape) == (2, (H // 2) ** 2, 64),
          str(tuple(tp.shape)))
    check("img_ids cover target + reference", ids.shape[0] == 2 * tp.shape[1])
    check("reference half tagged with id 0 == 1",
          sorted(ids[:, 0].unique().tolist()) == [0.0, 1.0] and ids[tp.shape[1]:, 0].min() == 1)

    ctx = {"prompt": torch.randn(1, 512, 4096, device=dev, dtype=torch.bfloat16),
           "pooled": torch.randn(1, 768, device=dev, dtype=torch.bfloat16),
           "text_ids": torch.zeros(512, 3, device=dev, dtype=torch.bfloat16),
           "img_ids": ids}
    cfg = types.SimpleNamespace(weighting_scheme="logit_normal", logit_mean=0.0,
                                logit_std=1.0, mode_scale=1.29, guidance_scale=1.0)
    sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0,
                                            use_dynamic_shifting=True)

    print("\nloss and gradients")
    torch.manual_seed(0)
    loss = flow_matching_loss(tr, tp, cp, ctx, cfg, sched)
    check("loss is finite", bool(torch.isfinite(loss)), f"{loss.item():.4f}")
    loss.backward()
    check("frozen base receives no gradient",
          all(p.grad is None for n, p in tr.named_parameters() if "lora" not in n))

    opt = torch.optim.AdamW([p for _, p in named], lr=5e-3)
    opt.step()
    opt.zero_grad()
    flow_matching_loss(tr, tp, cp, ctx, cfg, sched).backward()
    live = [n for n, p in named if p.grad is not None and p.grad.abs().sum() > 0]
    # lora_B is zero-initialised, so lora_A only receives gradient from step 2 on
    check("every LoRA tensor trains after the first update",
          len(live) == len(named), f"{len(live)}/{len(named)}")

    print("\noptimisation actually descends")
    # Re-seed before every call so the sampled timestep and noise are identical
    # each iteration: otherwise we would be measuring the variance of the
    # flow-matching estimator, not whether the adapter is learning.
    losses = []
    for _ in range(120):
        opt.zero_grad()
        torch.manual_seed(0)
        l = flow_matching_loss(tr, tp, cp, ctx, cfg, sched)
        l.backward()
        opt.step()
        losses.append(l.item())
    first, last = losses[0], losses[-1]
    # a 2-layer toy model with 7k adapter params cannot drive this far down;
    # we only assert a real, monotone descent, not a particular magnitude
    check("loss decreases on a fixed batch+timestep", last < first,
          f"{first:.4f} -> {last:.4f} ({100*(1-last/first):.1f}% down)")
    check("descent is monotone, not noise",
          sum(b < a for a, b in zip(losses, losses[1:])) > 0.9 * (len(losses) - 1),
          f"{sum(b < a for a, b in zip(losses, losses[1:]))}/{len(losses)-1} steps improved")

    print(f"\n{'ALL CHECKS PASSED' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
