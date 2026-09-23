"""Verify each backend's training step without downloading tens of GB of weights.

For every registered backend we build a randomly-initialised, deliberately tiny
transformer with the same interface as the real one and push a real batch
through `backend.loss`. This catches what breaks silently: wrong sequence
concatenation, a missing reference-image tag, LoRA attached to nothing,
gradients leaking into the frozen base, or a backend wired to the wrong
conditioning signature.

Qwen's vision-language text encoder is stubbed (it is 8 GB and orthogonal to
what is under test); everything else is the real code path.

Needs a GPU but only ~100 MB of VRAM.

    python tests/test_train_step.py
"""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mazelora.backends import get_backend, load_lora, pack, save_lora

FAILS: list[str] = []
LAT = 16                      # latent grid side -> 128px equivalent, 64 tokens


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def cfg():
    return types.SimpleNamespace(weighting_scheme="logit_normal", logit_mean=0.0,
                                 logit_std=1.0, mode_scale=1.29, guidance_scale=1.0)


class _StubTextPipe:
    """Stands in for Qwen2.5-VL.

    Mirrors the real `encode_prompt` contract exactly, including the easily
    missed part: it returns `None` for the mask whenever the mask would be all
    ones, which for a single un-padded prompt is always. `vary` makes the
    sequence length differ per call, to exercise the padding branch.
    """

    def __init__(self, dim, device, dtype, seq=37, vary=False):
        self.dim, self.device, self.dtype, self.seq = dim, device, dtype, seq
        self.vary = vary
        self.calls = 0

    def encode_prompt(self, prompt, image=None, device=None, num_images_per_prompt=1,
                      max_sequence_length=1024):
        assert image is not None, "Qwen must pass the maze image to its text encoder"
        seq = self.seq + (self.calls * 3 if self.vary else 0)
        self.calls += 1
        e = torch.randn(1, seq, self.dim, device=self.device, dtype=self.dtype)
        return e, None      # the real pipeline drops an all-ones mask


def tiny_flux(device, dtype):
    from diffusers import FluxTransformer2DModel
    tr = FluxTransformer2DModel(
        patch_size=1, in_channels=64, num_layers=2, num_single_layers=2,
        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=4096,
        pooled_projection_dim=768, guidance_embeds=True, axes_dims_rope=(4, 6, 6),
    ).to(device, dtype)
    b = get_backend("flux_kontext")
    ctx = {"prompt": torch.randn(1, 512, 4096, device=device, dtype=dtype),
           "pooled": torch.randn(1, 768, device=device, dtype=dtype),
           "text_ids": torch.zeros(512, 3, device=device, dtype=dtype),
           "img_ids": torch.cat([b._latent_ids(LAT, LAT, device, dtype, False),
                                 b._latent_ids(LAT, LAT, device, dtype, True)], dim=0)}
    return b, tr, ctx


def tiny_qwen(device, dtype):
    from diffusers import QwenImageTransformer2DModel
    tr = QwenImageTransformer2DModel(
        patch_size=2, in_channels=64, out_channels=16, num_layers=2,
        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=3584,
        guidance_embeds=False, axes_dims_rope=(4, 6, 6), zero_cond_t=True,
    ).to(device, dtype)
    b = get_backend("qwen_edit")
    ctx = {"text_pipe": _StubTextPipe(3584, device, dtype),
           "prompt_text": "solve the maze", "size_px": LAT * 8}
    return b, tr, ctx


BUILDERS = {"flux_kontext": tiny_flux, "qwen_edit": tiny_qwen}


def _img():
    from PIL import Image
    return Image.new("RGB", (32, 32))


def batch(backend, device, bsz=2):
    b = {"target": torch.randn(bsz, 16, LAT, LAT, dtype=torch.bfloat16),
         "cond": torch.randn(bsz, 16, LAT, LAT, dtype=torch.bfloat16)}
    if backend.condition_px:
        b["cond_px"] = torch.randint(0, 255, (bsz, 32, 32, 3), dtype=torch.uint8)
    return b


def run_backend(name, device, dtype):
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig

    print(f"\n{'='*12} backend: {name} {'='*12}")
    backend, tr, ctx = BUILDERS[name](device, dtype)
    tr.requires_grad_(False)
    tr.enable_gradient_checkpointing()
    tr.add_adapter(LoraConfig(r=8, lora_alpha=8, init_lora_weights="gaussian",
                              target_modules=backend.default_lora_targets))
    cast_training_params(tr, dtype=torch.float32)
    named = [(n, p) for n, p in tr.named_parameters() if p.requires_grad]

    check("adapter found the backend's target modules", len(named) > 0,
          f"{len(named)} tensors from {backend.default_lora_targets}")
    check("only LoRA params are trainable", all("lora" in n for n, _ in named))
    check("adapters are fp32 under bf16 autocast",
          all(p.dtype == torch.float32 for _, p in named))

    tp = pack(torch.randn(2, 16, LAT, LAT))
    check("packing is [B, (H/2)(W/2), C*4]", tuple(tp.shape) == (2, (LAT // 2) ** 2, 64),
          str(tuple(tp.shape)))

    sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0,
                                            use_dynamic_shifting=True)
    bt = batch(backend, device)
    torch.manual_seed(0)
    loss = backend.loss(tr, bt, ctx, cfg(), sched, device, dtype)
    check("loss is finite", bool(torch.isfinite(loss)), f"{loss.item():.4f}")
    loss.backward()
    check("frozen base receives no gradient",
          all(p.grad is None for n, p in tr.named_parameters() if "lora" not in n))
    if name == "qwen_edit":
        check("text encoder was given the maze image", ctx["text_pipe"].calls == 2,
              f"{ctx['text_pipe'].calls} per-sample encodes for batch of 2")
        # regression: encode_prompt returns None for an all-ones mask, and the
        # equal-length case must pass that None straight through
        e, m = backend._encode_batch(ctx["text_pipe"], "p",
                                     [_img(), _img()], device)
        check("equal-length prompts -> embeds stacked, mask left as None",
              m is None and tuple(e.shape[:2]) == (2, ctx["text_pipe"].seq),
              f"embeds {tuple(e.shape)}, mask {m}")
        vary = _StubTextPipe(3584, device, dtype, vary=True)
        e, m = backend._encode_batch(vary, "p", [_img(), _img(), _img()], device)
        ok = (m is not None and e.shape[0] == 3 and m.shape == e.shape[:2]
              and m.sum(1).tolist() == [37, 40, 43] and e.shape[1] == 43)
        check("unequal-length prompts -> right-padded with a real mask", ok,
              f"embeds {tuple(e.shape)}, mask rows {None if m is None else m.sum(1).tolist()}")

    opt = torch.optim.AdamW([p for _, p in named], lr=5e-3)
    opt.step()
    opt.zero_grad()
    backend.loss(tr, bt, ctx, cfg(), sched, device, dtype).backward()
    # lora_B is zero-initialised, so lora_A only receives gradient from step 2 on
    dead = {n for n, p in named if p.grad is None or p.grad.abs().sum() == 0}
    # QwenImage is dual-stream and the final block's text output is discarded, so
    # that block's text-query and text-out projections are structurally inert.
    # With 60 real layers this wastes 4 of ~960 adapter tensors.
    expected_dead: set[str] = set()
    if hasattr(tr, "transformer_blocks"):
        last = f"transformer_blocks.{len(tr.transformer_blocks) - 1}."
        expected_dead = {n for n in dead if n.startswith(last)
                         and (".add_q_proj." in n or ".to_add_out." in n)}
    check("every LoRA tensor trains, bar the final block's discarded text outputs",
          dead == expected_dead,
          f"{len(named)-len(dead)}/{len(named)} live"
          + (f", inert: {sorted(x.split('.attn.')[-1].split('.lora')[0] for x in dead)}"
             if dead else ""))

    # Re-seed before every call so the sampled timestep and noise are identical
    # each iteration: otherwise we would be measuring the variance of the
    # flow-matching estimator, not whether the adapter is learning.
    losses = []
    for _ in range(80):
        opt.zero_grad()
        torch.manual_seed(0)
        l = backend.loss(tr, bt, ctx, cfg(), sched, device, dtype)
        l.backward()
        opt.step()
        losses.append(l.item())
    improved = sum(b < a for a, b in zip(losses, losses[1:]))
    check("loss decreases on a fixed batch+timestep", losses[-1] < losses[0],
          f"{losses[0]:.4f} -> {losses[-1]:.4f} ({100*(1-losses[-1]/losses[0]):.1f}% down)")
    check("descent is monotone, not noise", improved > 0.9 * (len(losses) - 1),
          f"{improved}/{len(losses)-1} steps improved")

    # Checkpoint round-trip. Saving is only half the job: diffusers' model-level
    # loader skips safetensors unless the filename is passed explicitly, so a
    # checkpoint can be written perfectly and still be unloadable.
    from peft.utils import get_peft_model_state_dict
    with tempfile.TemporaryDirectory() as td:
        path = save_lora(tr, td)
        check("checkpoint written as safetensors",
              path.name == "pytorch_lora_weights.safetensors" and path.exists(),
              path.name)

        torch.manual_seed(1)
        before_loss = backend.loss(tr, bt, ctx, cfg(), sched, device, dtype).item()
        before = {k: v.detach().clone() for k, v in get_peft_model_state_dict(tr).items()}

        tr.delete_adapters(["default"])
        gone = not any("lora" in n for n, _ in tr.named_parameters())
        check("adapter can be detached", gone)

        load_lora(tr, td)
        after = get_peft_model_state_dict(tr)
        same_keys = set(after) == set(before)
        same_vals = same_keys and all(
            torch.allclose(after[k].float().cpu(), before[k].float().cpu(), atol=1e-6)
            for k in before)
        check("reloaded adapter matches what was saved", same_keys and same_vals,
              f"{len(before)} tensors"
              + ("" if same_keys else f", key mismatch: {sorted(set(before) ^ set(after))[:3]}"))

        torch.manual_seed(1)
        after_loss = backend.loss(tr, bt, ctx, cfg(), sched, device, dtype).item()
        check("reloaded model reproduces the same loss",
              abs(after_loss - before_loss) < 1e-4,
              f"{before_loss:.6f} vs {after_loss:.6f}")


def pick_device():
    """Use the GPU when it has room, otherwise fall back to CPU.

    These are toy models, so CPU is only a few seconds slower -- and the card
    may well be busy with a real training run.
    """
    if not torch.cuda.is_available():
        return "cpu", torch.float32
    try:
        # even creating the CUDA context needs a few hundred MB, so this whole
        # probe has to tolerate a card that is already full
        free, _ = torch.cuda.mem_get_info()
        if free < 2 * 2**30:
            print(f"only {free/2**30:.1f} GB free on the GPU; running on CPU")
            return "cpu", torch.float32
        return "cuda", torch.bfloat16
    except Exception as e:
        print(f"GPU unavailable ({type(e).__name__}); running on CPU")
        return "cpu", torch.float32


def main():
    device, dtype = pick_device()
    print(f"device: {device} ({dtype})")
    for name in BUILDERS:
        run_backend(name, device, dtype)
        if device == "cuda":
            torch.cuda.empty_cache()
    print(f"\n{'ALL CHECKS PASSED' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
