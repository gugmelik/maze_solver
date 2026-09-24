"""Inference-time scaling: Best-of-N sampling with a rule-based verifier.

The maze task has a property most generative tasks lack: **verification is free
and exact, and needs no ground truth**. Given only the puzzle image we can
recover the walls and the two endpoints (`decode.record_from_image`), then check
whether a candidate's red route actually connects start to goal without crossing
a wall. So Best-of-N here is a real deployable procedure, not just an oracle
bound: sample N candidates, keep the first one the verifier accepts.

Two curves are reported, and the gap between them is the interesting part:

  * **Best-of-N (verified)** -- what you could actually ship. Selection uses the
    puzzle image alone. This is the honest number.
  * **pass@N (oracle)** -- was *any* of the N candidates correct, judged against
    ground truth. The ceiling that a perfect selector would reach.

The verifier deliberately checks **legality only**. It never consults the BFS
solution, even though `record_from_image` could compute one -- that would make
the experiment circular (you could just draw the answer). Legality is a
checkable property of a candidate; it is the diffusion-model analogue of running
a unit test rather than reading the answer key.

    python -m mazelora.best_of_n --backend qwen_edit \
        --lora outputs/qwen_edit/checkpoints/step-006000 --n_mazes 40 --n_samples 8
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .backends import backend_names, get_backend
from .dataset import load_records
from .decode import decode_against, record_from_image
from .maze import Geometry, MazeRecord, render_record
from .metrics import gt_open_edges, score_sample


# --------------------------------------------------------------------------- #
# verification -- uses the puzzle image only, never the ground-truth record
# --------------------------------------------------------------------------- #
@dataclass
class Verdict:
    accepted: bool          # legal route start->goal, nothing drawn through a wall
    route_found: bool
    wall_violations: int
    extra_edges: int        # red drawn off the chosen route
    structure_acc: float    # how much of the maze the model preserved
    route_len: int


def verify(candidate: Image.Image, seen: MazeRecord) -> Verdict:
    """Score a candidate against the maze as *decoded from the puzzle image*.

    `seen` comes from `record_from_image`, so this function never touches the
    real answer -- it only asks "is what was drawn a legal solution of the maze
    I can see?".
    """
    d = decode_against(candidate, seen)
    open_edges = gt_open_edges(seen)
    violations = sum(1 for e in d.drawn_edges if e not in open_edges)
    route_edges = ({frozenset((a, b)) for a, b in zip(d.route, d.route[1:])}
                   if d.route else set())
    legal = bool(d.route) and all(e in open_edges for e in route_edges)

    all_edges = {frozenset((a, b)) for a, b in Geometry(seen.size).internal_edges()}
    judged = all_edges - d.drawn_edges
    struct = (sum(1 for e in judged if (e in d.open_edges) == (e in open_edges)) / len(judged)
              if judged else 1.0)

    return Verdict(
        accepted=bool(legal and violations == 0),
        route_found=bool(d.route),
        wall_violations=violations,
        extra_edges=len(d.drawn_edges - route_edges),
        structure_acc=struct,
        route_len=len(d.route) if d.route else 0,
    )


def rank_key(v: Verdict):
    """Sort ascending; the first element is the pick.

    Accepted candidates win outright. Among them, prefer the cleanest drawing
    (no stray red) and then the better-preserved maze. Among rejects, prefer one
    that at least connected the endpoints, then fewer illegal crossings.
    """
    return (not v.accepted,
            v.wall_violations,
            not v.route_found,
            v.extra_edges,
            -v.structure_acc)


def select(verdicts: list[Verdict]) -> int:
    """Index of the chosen candidate. Ties break toward the earlier sample, so
    Best-of-1 is exactly the plain single-sample baseline."""
    return min(range(len(verdicts)), key=lambda i: (rank_key(verdicts[i]), i))


# --------------------------------------------------------------------------- #
def scaling_curve(per_maze: list[dict], n_max: int) -> dict:
    """Solved rate as a function of the sample budget, for both selectors."""
    out = {"n": [], "best_of_n": [], "pass_at_n": [], "accepted_rate": []}
    for n in range(1, n_max + 1):
        bon, oracle, acc = [], [], []
        for m in per_maze:
            v = [Verdict(**x) for x in m["verdicts"][:n]]
            pick = select(v)
            bon.append(m["solved"][pick])
            oracle.append(any(m["solved"][:n]))
            acc.append(any(x.accepted for x in v))
        out["n"].append(n)
        out["best_of_n"].append(float(np.mean(bon)))
        out["pass_at_n"].append(float(np.mean(oracle)))
        out["accepted_rate"].append(float(np.mean(acc)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", type=str, default="qwen_edit", choices=backend_names())
    ap.add_argument("--lora", type=str, default=None,
                    help="checkpoint dir, or 'none' for the untuned base model")
    ap.add_argument("--lora_scale", type=float, default=1.0)
    ap.add_argument("--data", type=str, default="data/maze5")
    ap.add_argument("--cache", type=str, default="cache/maze5")
    ap.add_argument("--split", type=str, default="eval")
    ap.add_argument("--out", type=str, default=None,
                    help="default: <lora>/best_of_n")
    ap.add_argument("--model_id", type=str, default=None)
    ap.add_argument("--quantization", type=str, default="nf4",
                    choices=["nf4", "int8", "none"])
    ap.add_argument("--n_mazes", type=int, default=40, help="held-out mazes to test")
    ap.add_argument("--n_samples", type=int, default=8, help="candidates per maze (N)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gallery", type=int, default=12, help="mazes shown in the report")
    ap.add_argument("--save_images", action="store_true", default=True)
    args = ap.parse_args()

    backend = get_backend(args.backend)
    model_id = args.model_id or backend.default_model_id
    guidance = backend.eval_guidance if args.guidance is None else args.guidance
    cache = Path(args.cache) / backend.name
    if not (cache / "meta.json").exists():
        raise SystemExit(f"no cache for backend {args.backend!r} at {cache}; "
                         f"run mazelora.precompute --backend {args.backend} first")

    lora = None if args.lora in (None, "none", "None", "") else args.lora
    out = Path(args.out) if args.out else (
        Path(lora) / "best_of_n" if lora else Path(f"outputs/{backend.name}_base/best_of_n"))
    out.mkdir(parents=True, exist_ok=True)

    records = load_records(args.data, args.split)[: args.n_mazes]
    split_dir = Path(args.data) / args.split
    print(f"Best-of-N: {len(records)} mazes x {args.n_samples} candidates "
          f"= {len(records) * args.n_samples} generations")

    from .infer import MazeSolver
    print(f"loading {backend.name} ({'LoRA: ' + str(lora) if lora else 'base model'})...")
    solver = MazeSolver.from_checkpoint(args.backend, lora, cache, model_id,
                                        args.quantization, lora_scale=args.lora_scale)

    # The verifier only ever sees the puzzle, so decode each maze from its image.
    # Where that fails we fall back to the true record and flag it, rather than
    # silently dropping the maze from the denominator.
    seen: list[MazeRecord] = []
    n_fallback = 0
    for r in records:
        s = record_from_image(Image.open(split_dir / "puzzle" / f"{r.id}.png"), r.size,
                              sample_id=r.id)
        if s is None:
            s, n_fallback = r, n_fallback + 1
        seen.append(s)
    if n_fallback:
        print(f"  warning: {n_fallback} puzzles did not decode; used the true record")

    # N independent passes over the maze set: one seed offset per candidate slot.
    candidates: list[list[Image.Image]] = [[] for _ in records]
    t0 = time.time()
    for k in range(args.n_samples):
        print(f"  candidate {k+1}/{args.n_samples}")
        imgs = solver.solve_records(records, split_dir, args.split,
                                    num_steps=args.steps, guidance_scale=guidance,
                                    seed=args.seed + k * 100_003,
                                    batch_size=args.batch_size, progress=True)
        for i, im in enumerate(imgs):
            candidates[i].append(im)
    dt = time.time() - t0
    total = len(records) * args.n_samples
    print(f"generated {total} images in {dt:.0f}s ({dt/max(total,1):.1f}s each)")

    per_maze = []
    for rec, sn, cands in zip(records, seen, candidates):
        gt = render_record(rec, with_path=True)
        verdicts = [verify(c, sn) for c in cands]
        solved = [bool(score_sample(c, rec, gt)["solved"]) for c in cands]
        pick = select(verdicts)
        per_maze.append({
            "id": rec.id, "path_len": len(rec.path),
            "verdicts": [asdict(v) for v in verdicts],
            "solved": solved, "pick": pick,
            "solved_at_1": solved[0], "solved_best_of_n": solved[pick],
            "pass_at_n": any(solved),
        })
        if args.save_images:
            d = out / "samples" / rec.id
            d.mkdir(parents=True, exist_ok=True)
            for k, c in enumerate(cands):
                c.save(d / f"cand{k}{'_picked' if k == pick else ''}.png")

    curve = scaling_curve(per_maze, args.n_samples)

    # How trustworthy is the verifier? Of the candidates it accepted, how many
    # were genuinely correct -- and did it ever reject a correct one?
    acc_and_solved = acc_total = solved_total = missed = 0
    for m in per_maze:
        for v, s in zip(m["verdicts"], m["solved"]):
            acc_total += v["accepted"]
            solved_total += s
            acc_and_solved += v["accepted"] and s
            missed += s and not v["accepted"]

    agg = {
        "n_mazes": len(records), "n_samples": args.n_samples,
        "backend": backend.name, "checkpoint": str(lora) if lora else "base model",
        "steps": args.steps, "guidance": guidance, "seconds_per_image": dt / max(total, 1),
        "solved_at_1": float(np.mean([m["solved_at_1"] for m in per_maze])),
        "solved_best_of_n": float(np.mean([m["solved_best_of_n"] for m in per_maze])),
        "pass_at_n": float(np.mean([m["pass_at_n"] for m in per_maze])),
        "verifier_precision": (acc_and_solved / acc_total) if acc_total else None,
        "verifier_recall": (acc_and_solved / solved_total) if solved_total else None,
        "verifier_false_accepts": acc_total - acc_and_solved,
        "verifier_missed_correct": missed,
        "puzzles_not_decoded": n_fallback,
        "curve": curve,
    }
    (out / "metrics.json").write_text(json.dumps(agg, indent=2))
    with open(out / "rows.jsonl", "w") as f:
        for m in per_maze:
            f.write(json.dumps(m) + "\n")

    from .bon_report import build_bon_report
    gallery = {}
    order = sorted(per_maze, key=lambda m: (m["solved_at_1"], not m["solved_best_of_n"]))
    for m in order[: args.gallery]:
        rec = next(r for r in records if r.id == m["id"])
        i = records.index(rec)
        gallery[m["id"]] = (render_record(rec, with_path=False), candidates[i],
                            render_record(rec, with_path=True))
    report = build_bon_report(out / "report.html", agg, order, gallery, args.gallery)

    print("\n=== inference-time scaling ===")
    print(f"  solved @ 1 sample     {agg['solved_at_1']*100:6.1f}%")
    print(f"  Best-of-{args.n_samples} (verified)  {agg['solved_best_of_n']*100:6.1f}%")
    print(f"  pass@{args.n_samples} (oracle)      {agg['pass_at_n']*100:6.1f}%")
    if agg["verifier_precision"] is not None:
        print(f"  verifier precision    {agg['verifier_precision']*100:6.1f}%  "
              f"({agg['verifier_false_accepts']} false accepts)")
    print(f"\nreport -> {report.resolve()}")


if __name__ == "__main__":
    main()
