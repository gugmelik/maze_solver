"""LoRA fine-tuning of an image-edit diffusion model on maze solving.

Pick the base model with `--backend`:

    --backend flux_kontext    FLUX.1-Kontext-dev, 12B, ~34 GB download
    --backend qwen_edit       Qwen-Image-Edit-2511, 20B, ~54 GB download

Both train the same way -- rectified flow matching on a reference-conditioned
transformer -- so results are directly comparable. What differs per model lives
in `mazelora/backends/`.

Memory plan for a 24 GB card:
  * DiT quantized to NF4                                ~7 GB (12B) / ~12 GB (20B)
  * LoRA adapters in fp32, everything else frozen        <1 GB
  * gradient checkpointing on the DiT blocks             trades compute for ~8 GB
  * 8-bit AdamW                                          negligible state
  * VAE never loaded; text encoder loaded only if the
    backend's text conditioning depends on the image     0 GB (FLUX) / ~4 GB (Qwen)
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

from .backends import backend_names, get_backend, load_lora, save_lora
from .dataset import LatentPairs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default=None, help="YAML file of defaults")
    p.add_argument("--backend", type=str, default="flux_kontext", choices=backend_names(),
                   help="which base model to fine-tune")
    p.add_argument("--data", type=str, default="data/maze5")
    p.add_argument("--cache", type=str, default="cache/maze5")
    p.add_argument("--output", type=str, default=None,
                   help="default: outputs/<backend>")
    p.add_argument("--model_id", type=str, default=None, help="override the backend default")
    p.add_argument("--quantization", type=str, default="nf4", choices=["nf4", "int8", "none"])

    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_targets", nargs="+", default=None,
                   help="default: the backend's attention projections")

    p.add_argument("--max_steps", type=int, default=6000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--guidance_scale", type=float, default=None,
                   help="value fed to a distilled guidance embedding, where the "
                        "model has one; default: the backend's")
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
    p.add_argument("--validate_n", type=int, default=16)
    p.add_argument("--validate_steps", type=int, default=20)
    p.add_argument("--validate_guidance", type=float, default=None)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--resume", type=str, default=None, help="checkpoint dir to resume from")

    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--no_wandb", dest="wandb", action="store_false",
                   help="disable W&B even if the config file enables it")
    p.add_argument("--wandb_project", type=str, default="maze-diff-reasoning")
    p.add_argument("--wandb_entity", type=str, default=None)
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
    args = parser.parse_args(argv)

    backend = get_backend(args.backend)
    if args.lora_targets is None:
        args.lora_targets = list(backend.default_lora_targets)
    if args.model_id is None:
        args.model_id = backend.default_model_id
    if args.guidance_scale is None:
        args.guidance_scale = backend.train_guidance
    if args.output is None:
        args.output = f"outputs/{backend.name}"
    return args, backend


def main():
    args, backend = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device, dtype = "cuda", torch.bfloat16

    cache = Path(args.cache) / backend.name
    if not (cache / "meta.json").exists():
        raise SystemExit(
            f"no cache for backend {backend.name!r} at {cache}.\n"
            f"run: python -m mazelora.precompute --backend {backend.name} "
            f"--data {args.data} --cache {args.cache}")

    out = Path(args.output)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2))

    # ---------------- data ----------------
    train_ds = LatentPairs(cache, "train", data_dir=args.data,
                           condition_px=backend.condition_px)
    size_px = train_ds.meta["size_px"]
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True,
                        persistent_workers=args.num_workers > 0)
    print(f"backend: {backend.name} ({args.model_id})")
    print(f"train samples: {len(train_ds)}  latent {tuple(train_ds.cond.shape[1:])}")

    # ---------------- model ----------------
    print(f"loading transformer ({args.quantization})...")
    transformer = backend.load_transformer(args.model_id, args.quantization, dtype, device)
    transformer.requires_grad_(False)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig
    transformer.add_adapter(LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian", target_modules=args.lora_targets))
    if args.resume:
        load_lora(transformer, args.resume)
        print(f"resumed LoRA weights from {args.resume}")

    # Order matters: loading re-injects the adapter at the model's dtype (bf16),
    # so the fp32 cast and the parameter list must both come *after* any resume.
    # Casting first would silently train a resumed run in bf16, and collecting
    # `params` first could hand the optimizer stale tensors.
    cast_training_params(transformer, dtype=torch.float32)   # fp32 adapters, bf16 autocast
    params = [p for p in transformer.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params)/1e6:.2f}M "
          f"across {len(params)} tensors")

    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(params, lr=args.lr, betas=(0.9, 0.999),
                                    weight_decay=1e-4, eps=1e-8)
    from diffusers.optimization import get_scheduler
    lr_sched = get_scheduler(args.lr_scheduler, optimizer=optimizer,
                             num_warmup_steps=args.warmup_steps,
                             num_training_steps=args.max_steps)

    sched_copy = copy.deepcopy(backend.scheduler(args.model_id))
    ctx = backend.make_context(cache, args.model_id, args.quantization,
                               device, dtype, size_px)

    start_step = 0
    if args.resume and (Path(args.resume) / "state.pt").exists():
        st = torch.load(Path(args.resume) / "state.pt", map_location="cpu")
        optimizer.load_state_dict(st["optimizer"])
        lr_sched.load_state_dict(st["lr_scheduler"])
        start_step = st["step"]
        print(f"resumed optimizer at step {start_step}")

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_run_name or f"{backend.name}-r{args.lora_rank}",
                         config=vars(args), dir=str(out))
        print(f"wandb: {run.url}")

    def save_checkpoint(step: int):
        ck = out / "checkpoints" / f"step-{step:06d}"
        save_lora(transformer, ck)
        torch.save({"optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_sched.state_dict(), "step": step},
                   ck / "state.pt")
        (out / "checkpoints" / "latest").write_text(ck.name)
        return ck

    def run_validation(step: int):
        from .dataset import load_records
        from .infer import MazeSolver
        from .maze import render_record
        from .metrics import aggregate, score_sample
        transformer.eval()
        try:
            solver = MazeSolver.from_live_transformer(
                backend, transformer, args.model_id, cache, device, dtype)
            recs = load_records(args.data, "eval")[: args.validate_n]
            imgs = solver.solve_records(recs, Path(args.data) / "eval", "eval",
                                        num_steps=args.validate_steps,
                                        guidance_scale=args.validate_guidance,
                                        seed=args.seed)
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
                                      data=[[int(k), v["n"], v["solved"]] for k, v in
                                            sorted(by_len.items(), key=lambda x: int(x[0]))])
                    run.log({"val/solved_by_path_len": wandb.plot.bar(
                        tbl, "path_len", "solved",
                        title="Solved rate by shortest-path length")}, step=step)
            del solver
            torch.cuda.empty_cache()
        finally:
            transformer.train()

    # ---------------- train ----------------
    transformer.train()
    step, micro = start_step, 0
    running, t0 = [], time.time()
    bar = tqdm(total=args.max_steps, initial=start_step, desc="train")
    done = False
    while not done:
        for batch in loader:
            loss = backend.loss(transformer, batch, ctx, args, sched_copy, device, dtype)
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
