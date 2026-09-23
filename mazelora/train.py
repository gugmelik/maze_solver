"""LoRA fine-tuning of FLUX.1-Kontext-dev on maze solving. Single 24 GB GPU.

Memory plan for an A5000:
  * DiT quantized to NF4                              ~7 GB
  * LoRA adapters kept in fp32, everything else frozen  <1 GB
  * gradient checkpointing on the DiT blocks           trades compute for ~8 GB
  * 8-bit AdamW                                        negligible state
  * text encoders + VAE never loaded (cached latents)  saves ~10 GB
Leaving roughly 10 GB of headroom for activations at batch size 1-2.

Objective: rectified-flow matching, identical to the diffusers FLUX reference
scripts -- predict (noise - target) at a logit-normally sampled sigma.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import LatentPairs
from .flux_utils import MODEL_ID, load_prompt_cache, load_transformer, pack, latent_ids


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default=None, help="YAML file of defaults")
    p.add_argument("--data", type=str, default="data/maze5")
    p.add_argument("--cache", type=str, default="cache/maze5")
    p.add_argument("--output", type=str, default="outputs/baseline")
    p.add_argument("--model_id", type=str, default=MODEL_ID)
    p.add_argument("--quantization", type=str, default="nf4", choices=["nf4", "int8", "none"])

    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_targets", nargs="+",
                   default=["to_q", "to_k", "to_v", "to_out.0"])

    p.add_argument("--max_steps", type=int, default=6000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--guidance_scale", type=float, default=1.0,
                   help="value fed to the distilled guidance embedding during training")
    p.add_argument("--weighting_scheme", type=str, default="logit_normal",
                   choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    p.add_argument("--logit_mean", type=float, default=0.0)
    p.add_argument("--logit_std", type=float, default=1.0)
    p.add_argument("--mode_scale", type=float, default=1.29)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                   action="store_false")

    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--validate_every", type=int, default=1000)
    p.add_argument("--validate_n", type=int, default=16,
                   help="eval samples generated for in-training validation")
    p.add_argument("--validate_steps", type=int, default=20, help="denoising steps at validation")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--resume", type=str, default=None, help="checkpoint dir to resume from")

    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--no_wandb", dest="wandb", action="store_false",
                   help="disable W&B even if the config file enables it")
    p.add_argument("--wandb_project", type=str, default="maze-diff-reasoning")
    p.add_argument("--wandb_entity", type=str, default=None,
                   help="W&B team; omit to use your default entity")
    p.add_argument("--wandb_run_name", type=str, default=None)
    return p


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    known, _ = pre.parse_known_args(argv)
    parser = build_parser()
    if known.config:
        cfg = yaml.safe_load(Path(known.config).read_text()) or {}
        unknown = set(cfg) - {a.dest for a in parser._actions}
        if unknown:
            raise SystemExit(f"unknown keys in {known.config}: {sorted(unknown)}")
        parser.set_defaults(**cfg)          # CLI flags still win over the YAML
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
def get_sigmas(scheduler, timesteps, device, n_dim: int, dtype):
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_t = scheduler.timesteps.to(device)
    idx = [(schedule_t == t).nonzero().item() for t in timesteps]
    sigma = sigmas[idx].flatten()
    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def flow_matching_loss(transformer, target_packed, cond_packed, ctx, cfg, scheduler):
    """One rectified-flow training step, Kontext style.

    The reference (puzzle) latents are appended to the noisy target along the
    *sequence* axis and tagged via `img_ids[..., 0] == 1`; only the target half
    of the prediction is supervised. Target is the straight-line velocity
    `noise - clean`, i.e. the model learns to point from noise back to the
    solved maze.
    """
    from diffusers.training_utils import (compute_density_for_timestep_sampling,
                                          compute_loss_weighting_for_sd3)
    device, bsz = target_packed.device, target_packed.shape[0]
    n_train = scheduler.config.num_train_timesteps

    noise = torch.randn_like(target_packed)
    u = compute_density_for_timestep_sampling(
        cfg.weighting_scheme, bsz, cfg.logit_mean, cfg.logit_std, cfg.mode_scale)
    idx = (u * n_train).long().clamp(0, n_train - 1)
    timesteps = scheduler.timesteps[idx].to(device)
    sigmas = get_sigmas(scheduler, timesteps, device, target_packed.ndim, target_packed.dtype)
    noisy = (1.0 - sigmas) * target_packed + sigmas * noise

    hidden = torch.cat([noisy, cond_packed], dim=1)
    guidance = (torch.full([bsz], cfg.guidance_scale, device=device, dtype=torch.float32)
                if transformer.config.guidance_embeds else None)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=hidden.is_cuda):
        pred = transformer(
            hidden_states=hidden,
            timestep=timesteps.to(hidden.dtype) / 1000,
            guidance=guidance,
            pooled_projections=ctx["pooled"].expand(bsz, -1),
            encoder_hidden_states=ctx["prompt"].expand(bsz, -1, -1),
            txt_ids=ctx["text_ids"],
            img_ids=ctx["img_ids"],
            return_dict=False,
        )[0]
    pred = pred[:, : target_packed.shape[1]].float()

    flow_target = (noise - target_packed).float()
    weighting = compute_loss_weighting_for_sd3(cfg.weighting_scheme, sigmas).float()
    return (weighting * (pred - flow_target) ** 2).mean()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    dtype = torch.bfloat16

    out = Path(args.output)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2))

    # ---------------- data ----------------
    train_ds = LatentPairs(args.cache, "train")
    size_px = train_ds.meta["size_px"]
    h_lat = w_lat = size_px // 8
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True,
                        persistent_workers=args.num_workers > 0)
    print(f"train samples: {len(train_ds)}  latent {tuple(train_ds.cond.shape[1:])}")

    prompt_embeds, pooled_embeds, text_ids = load_prompt_cache(
        Path(args.cache) / "prompt.safetensors", device, dtype)
    prompt_embeds = prompt_embeds.unsqueeze(0)
    pooled_embeds = pooled_embeds.unsqueeze(0)

    # ---------------- model ----------------
    print(f"loading transformer ({args.quantization})...")
    transformer = load_transformer(args.model_id, args.quantization, dtype, device)
    transformer.requires_grad_(False)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    from peft import LoraConfig
    from diffusers.training_utils import cast_training_params
    lora_cfg = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha,
                          lora_dropout=args.lora_dropout, init_lora_weights="gaussian",
                          target_modules=args.lora_targets)
    transformer.add_adapter(lora_cfg)
    cast_training_params(transformer, dtype=torch.float32)   # fp32 adapters, bf16 autocast
    params = [p for p in transformer.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params)/1e6:.2f}M "
          f"across {len(params)} tensors")

    if args.resume:
        from diffusers import FluxKontextPipeline
        state = FluxKontextPipeline.lora_state_dict(args.resume)
        state = {k.removeprefix("transformer."): v for k, v in state.items()
                 if k.startswith("transformer.")}
        from peft.utils import set_peft_model_state_dict
        set_peft_model_state_dict(transformer, state)
        print(f"resumed LoRA weights from {args.resume}")

    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(params, lr=args.lr, betas=(0.9, 0.999),
                                    weight_decay=1e-4, eps=1e-8)
    from diffusers.optimization import get_scheduler
    lr_sched = get_scheduler(args.lr_scheduler, optimizer=optimizer,
                             num_warmup_steps=args.warmup_steps,
                             num_training_steps=args.max_steps)

    from diffusers import FlowMatchEulerDiscreteScheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model_id, subfolder="scheduler")
    sched_copy = copy.deepcopy(noise_scheduler)

    start_step = 0
    if args.resume and (Path(args.resume) / "state.pt").exists():
        st = torch.load(Path(args.resume) / "state.pt", map_location="cpu")
        optimizer.load_state_dict(st["optimizer"])
        lr_sched.load_state_dict(st["lr_scheduler"])
        start_step = st["step"]
        print(f"resumed optimizer at step {start_step}")

    # positional ids are constant for a fixed resolution -> build once
    tgt_ids = latent_ids(h_lat, w_lat, device, dtype, is_reference=False)
    cond_ids = latent_ids(h_lat, w_lat, device, dtype, is_reference=True)
    img_ids = torch.cat([tgt_ids, cond_ids], dim=0)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_run_name, config=vars(args), dir=str(out))
        print(f"wandb: {run.url}")

    def save_checkpoint(step: int):
        from diffusers import FluxKontextPipeline
        from peft.utils import get_peft_model_state_dict
        ck = out / "checkpoints" / f"step-{step:06d}"
        ck.mkdir(parents=True, exist_ok=True)
        lora_sd = get_peft_model_state_dict(transformer)
        FluxKontextPipeline.save_lora_weights(str(ck), transformer_lora_layers=lora_sd,
                                              safe_serialization=True)
        torch.save({"optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_sched.state_dict(), "step": step},
                   ck / "state.pt")
        (out / "checkpoints" / "latest").write_text(ck.name)
        return ck

    def run_validation(step: int):
        from .infer import MazeSolver
        from .metrics import aggregate, score_sample
        from .dataset import load_records
        from .maze import render_record
        transformer.eval()
        try:
            solver = MazeSolver.from_live_transformer(
                transformer, args.model_id, Path(args.cache), device)
            recs = load_records(args.data, "eval")[: args.validate_n]
            imgs = solver.solve_records(recs, Path(args.data) / "eval",
                                        num_steps=args.validate_steps,
                                        guidance_scale=2.5, seed=args.seed)
            rows = [score_sample(im, r, render_record(r, True))
                    for im, r in zip(imgs, recs)]
            agg = aggregate(rows)
            print(f"  [val @ {step}] solved={agg['solved']:.3f} "
                  f"f1={agg['edge_f1']:.3f} struct={agg['structure_acc']:.3f}")
            vdir = out / "validation" / f"step-{step:06d}"
            vdir.mkdir(parents=True, exist_ok=True)
            for im, r in zip(imgs[:8], recs[:8]):
                im.save(vdir / f"{r.id}.png")
            (vdir / "metrics.json").write_text(json.dumps(agg, indent=2))
            if run is not None:
                import wandb
                run.log({f"val/{k}": v for k, v in agg.items()
                         if isinstance(v, (int, float))}, step=step)
                run.log({"val/samples": [
                    wandb.Image(im, caption=f"{r.id} solved={row['solved']}")
                    for im, r, row in zip(imgs[:8], recs[:8], rows[:8])]}, step=step)
                # accuracy vs difficulty is the signal worth watching; it is a
                # dict, so it would otherwise be dropped by the scalar filter
                by_len = agg.get("solved_by_path_len") or {}
                if by_len:
                    tbl = wandb.Table(columns=["path_len", "n", "solved"],
                                      data=[[int(k), v["n"], v["solved"]]
                                            for k, v in sorted(by_len.items(), key=lambda x: int(x[0]))])
                    run.log({"val/solved_by_path_len": wandb.plot.bar(
                        tbl, "path_len", "solved",
                        title="Solved rate by shortest-path length")}, step=step)
            del solver
            torch.cuda.empty_cache()
        finally:
            transformer.train()

    # ---------------- train ----------------
    transformer.train()
    step = start_step
    micro = 0
    running, t0 = [], time.time()
    bar = tqdm(total=args.max_steps, initial=start_step, desc="train")
    done = False
    while not done:
        for batch in loader:
            target_lat = batch["target"].to(device, dtype, non_blocking=True)
            cond_lat = batch["cond"].to(device, dtype, non_blocking=True)

            ctx = {"prompt": prompt_embeds, "pooled": pooled_embeds,
                   "text_ids": text_ids, "img_ids": img_ids}
            loss = flow_matching_loss(transformer, pack(target_lat), pack(cond_lat),
                                      ctx, args, sched_copy)

            (loss / args.grad_accum).backward()
            running.append(loss.detach().item())
            micro += 1

            if micro % args.grad_accum:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
            lr_sched.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            bar.update(1)

            if step % args.log_every == 0:
                avg = float(np.mean(running[-args.log_every * args.grad_accum:]))
                sps = args.log_every / max(time.time() - t0, 1e-6)
                mem = torch.cuda.max_memory_allocated() / 2**30
                bar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr_sched.get_last_lr()[0]:.2e}",
                                mem=f"{mem:.1f}G")
                if run is not None:
                    run.log({"train/loss": avg, "train/lr": lr_sched.get_last_lr()[0],
                             "train/grad_norm": float(grad_norm),
                             "train/steps_per_sec": sps, "train/vram_gb": mem}, step=step)
                t0 = time.time()

            if args.validate_every and step % args.validate_every == 0:
                run_validation(step)
                t0 = time.time()
            if args.save_every and step % args.save_every == 0:
                print(f"  saved {save_checkpoint(step)}")
                t0 = time.time()
            if step >= args.max_steps:
                done = True
                break

    bar.close()
    ck = save_checkpoint(step)
    print(f"\nfinished at step {step}\nfinal checkpoint: {ck}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
