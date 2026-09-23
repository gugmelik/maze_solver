"""Generate solutions for the held-out split, score them, write an HTML report.

    python -m mazelora.evaluate --lora outputs/baseline/checkpoints/step-006000
    python -m mazelora.evaluate --lora none --out outputs/base_model   # untuned reference

Outputs under --out:
    metrics.json   aggregate numbers (feed back in via --compare_to)
    rows.jsonl     per-sample scores
    samples/       generated PNGs
    report.html    open this
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .dataset import load_records
from .flux_utils import MODEL_ID
from .maze import render_record
from .metrics import aggregate, score_sample
from .report import build_report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lora", type=str, default=None,
                    help="checkpoint dir, or 'none' for the untuned base model")
    ap.add_argument("--lora_scale", type=float, default=1.0)
    ap.add_argument("--data", type=str, default="data/maze5")
    ap.add_argument("--cache", type=str, default="cache/maze5")
    ap.add_argument("--split", type=str, default="eval")
    ap.add_argument("--out", type=str, default=None,
                    help="default: <lora>/eval or outputs/base_model")
    ap.add_argument("--model_id", type=str, default=MODEL_ID)
    ap.add_argument("--quantization", type=str, default="nf4", choices=["nf4", "int8", "none"])
    ap.add_argument("--n", type=int, default=100, help="how many held-out mazes to score")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gallery", type=int, default=48, help="samples embedded in the report")
    ap.add_argument("--compare_to", type=str, default=None,
                    help="another run's metrics.json, shown as a side-by-side delta")
    ap.add_argument("--save_images", action="store_true", default=True)
    args = ap.parse_args()

    lora = None if args.lora in (None, "none", "None", "") else args.lora
    out = Path(args.out) if args.out else (
        Path(lora) / "eval" if lora else Path("outputs/base_model"))
    out.mkdir(parents=True, exist_ok=True)

    records = load_records(args.data, args.split)[: args.n]
    print(f"evaluating {len(records)} mazes from {args.data}/{args.split}")

    from .infer import MazeSolver
    print(f"loading model ({'LoRA: ' + str(lora) if lora else 'base, no LoRA'})...")
    solver = MazeSolver.from_checkpoint(lora, Path(args.cache), args.model_id,
                                        args.quantization, lora_scale=args.lora_scale)

    t0 = time.time()
    preds = solver.solve_records(records, Path(args.data) / args.split, args.split,
                                 num_steps=args.steps, guidance_scale=args.guidance,
                                 seed=args.seed, batch_size=args.batch_size, progress=True)
    dt = time.time() - t0
    print(f"generated {len(preds)} images in {dt:.0f}s ({dt/max(len(preds),1):.1f}s each)")

    if args.save_images:
        (out / "samples").mkdir(exist_ok=True)
        for im, r in zip(preds, records):
            im.save(out / "samples" / f"{r.id}.png")

    rows, images = [], {}
    for im, rec in zip(preds, records):
        gt = render_record(rec, with_path=True)
        rows.append(score_sample(im, rec, gt))
        if len(images) < args.gallery:
            images[rec.id] = (render_record(rec, with_path=False), im, gt)

    agg = aggregate(rows)
    agg["seconds_per_image"] = dt / max(len(preds), 1)
    (out / "metrics.json").write_text(json.dumps(agg, indent=2))
    with open(out / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    baseline = None
    if args.compare_to and Path(args.compare_to).exists():
        baseline = json.loads(Path(args.compare_to).read_text())

    meta = {
        "checkpoint": str(lora) if lora else "base model (no LoRA)",
        "model": args.model_id, "quantization": args.quantization,
        "split": f"{args.data}/{args.split}", "samples": len(records),
        "denoise steps": args.steps, "guidance": args.guidance,
        "lora scale": args.lora_scale if lora else "-",
        "seed": args.seed, "sec / image": f"{agg['seconds_per_image']:.1f}",
        "generated": time.strftime("%Y-%m-%d %H:%M"),
    }
    # sort failures first so the gallery opens on the interesting cases
    gallery_rows = sorted(rows, key=lambda r: (r["solved"], r["id"]))
    report = build_report(out / "report.html", agg, gallery_rows, images, meta,
                          baseline, args.gallery)

    print("\n=== results ===")
    for k in ("solved", "exact_edges", "edge_f1", "edge_iou", "structure_acc",
              "endpoints_ok", "wall_violations", "red_pixel_iou"):
        if k in agg:
            print(f"  {k:18s} {agg[k]:.4f}")
    print(f"\nreport -> {report.resolve()}")


if __name__ == "__main__":
    main()
