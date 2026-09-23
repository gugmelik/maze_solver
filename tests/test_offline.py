"""Model-free checks: maze generation, the image decoder, and scoring.

Runs in seconds and needs no GPU or model weights. Run it after touching
anything in maze.py / decode.py / metrics.py -- if the decoder drifts, every
number the evaluation reports drifts with it.

    python tests/test_offline.py
"""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mazelora.decode import record_from_image
from mazelora.maze import render_maze, render_record, sample_maze, shortest_path_bfs
from mazelora.metrics import aggregate, score_sample

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def degrade(img, blur=0.0, noise=0.0, shift=0, quality=0):
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if shift:
        img = img.transform(img.size, Image.AFFINE, (1, 0, shift, 0, 1, -shift),
                            resample=Image.BICUBIC, fillcolor=(0, 0, 0))
    if noise:
        a = np.asarray(img).astype(np.int16)
        a = a + np.random.RandomState(0).normal(0, noise, a.shape)
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if quality:
        import io
        b = io.BytesIO(); img.save(b, "JPEG", quality=quality)
        img = Image.open(b).convert("RGB")
    return img


def main():
    rng = random.Random(42)
    recs = [sample_maze(5, 6, rng, sample_id=f"{i:06d}") for i in range(40)]

    print("\nmaze generation")
    check("solution paths are wall-legal",
          all(shortest_path_bfs(r.grid, r.start, r.goal) == r.path for r in recs))
    check("move strings match path length",
          all(len(r.udrl) == len(r.path) - 1 for r in recs))
    check("min path length respected", all(len(r.path) >= 6 for r in recs))

    print("\ndecoder: ground-truth solution must score perfectly")
    rows = [score_sample(render_record(r, True), r, render_record(r, True)) for r in recs]
    agg = aggregate(rows)
    check("solved == 1.0", agg["solved"] == 1.0, f"{agg['solved']:.3f}")
    check("exact edge set == 1.0", agg["exact_edges"] == 1.0, f"{agg['exact_edges']:.3f}")
    check("decoded moves == true moves", all(r["pred_udrl"] == r["gt_udrl"] for r in rows))
    check("no wall violations", agg["wall_violations"] == 0.0)
    check("structure fully preserved", agg["structure_acc"] == 1.0)

    print("\ndecoder: negatives must not score")
    empty = aggregate([score_sample(render_record(r, False), r) for r in recs])
    check("unsolved puzzle -> solved == 0", empty["solved"] == 0.0)
    check("unsolved puzzle -> no route found", empty["route_found"] == 0.0)
    wrong = aggregate([score_sample(render_maze(r.grid, r.start, r.goal, o.path), r)
                       for r, o in zip(recs, recs[1:] + recs[:1])])
    check("foreign path -> solved == 0", wrong["solved"] == 0.0)
    check("foreign path -> wall crossings flagged", wrong["wall_violations"] > 0.5,
          f"{wrong['wall_violations']:.2f} per maze")

    print("\ndecoder: robustness to generation artefacts")
    for label, kw in [("blur 1 / noise 8 / shift 2", dict(blur=1.0, noise=8, shift=2)),
                      ("blur 2.5 / noise 20 / shift 5", dict(blur=2.5, noise=20, shift=5)),
                      ("blur 4 / noise 30 / shift 8 / jpeg 60",
                       dict(blur=4.0, noise=30, shift=8, quality=60))]:
        a = aggregate([score_sample(degrade(render_record(r, True), **kw), r) for r in recs])
        check(f"still solves under {label}", a["solved"] == 1.0, f"{a['solved']:.3f}")

    print("\nmaze recovery from a puzzle image alone (used by the app)")
    rec2 = [record_from_image(degrade(render_record(r, False), blur=3.0), 5) for r in recs]
    check("walls recovered", all(a is not None and a.walls == b.walls
                                 for a, b in zip(rec2, recs)))
    check("shortest path recovered", all(a is not None and a.udrl == b.udrl
                                         for a, b in zip(rec2, recs)))

    print("\nreport renders")
    from mazelora.report import build_report
    imgs = {r.id: (render_record(r, False), render_record(r, True), render_record(r, True))
            for r in recs[:6]}
    with tempfile.TemporaryDirectory() as td:
        p = build_report(Path(td) / "r.html", agg, rows[:6], imgs, {"checkpoint": "test"},
                         baseline_agg=None, gallery_n=6)
        html = p.read_text()
    check("report is self-contained html", html.startswith("<!doctype html>")
          and "data:image/png;base64," in html and len(html) > 10_000)

    print(f"\n{'ALL CHECKS PASSED' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
