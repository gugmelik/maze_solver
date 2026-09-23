"""Build a train/eval maze dataset of (puzzle -> solution) image pairs.

Each sample is a triple:
    puzzle/<id>.png    maze + yellow start + blue goal, no path   (model input)
    solution/<id>.png  same maze with the red shortest path       (model target)
    manifest.jsonl     symbolic record: walls, start, goal, path, U/D/L/R string

Mazes are deduplicated on (walls, start, goal) across *both* splits, so nothing
in the eval set was memorised from training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import random
from pathlib import Path

from tqdm import tqdm

from .maze import render_record, sample_maze


def _fingerprint(rec) -> str:
    blob = json.dumps([rec.walls, list(rec.start), list(rec.goal)], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()


def _worker(job):
    idx, size, min_len, seed = job
    rec = sample_maze(size, min_len, random.Random(seed + idx), sample_id=f"{idx:06d}")
    return _fingerprint(rec), rec.to_json()


def _render_worker(job):
    split_dir, rec_json, size_px = job
    from .maze import MazeRecord
    rec = MazeRecord.from_json(rec_json)
    render_record(rec, with_path=False, size_px=size_px).save(
        Path(split_dir) / "puzzle" / f"{rec.id}.png")
    render_record(rec, with_path=True, size_px=size_px).save(
        Path(split_dir) / "solution" / f"{rec.id}.png")
    return rec.id


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=5, help="maze grid dimension (n x n)")
    ap.add_argument("--min_len", type=int, default=6, help="minimum cells on the solution path")
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--eval", type=int, default=1000)
    ap.add_argument("--out", type=str, default="data/maze5")
    ap.add_argument("--size_px", type=int, default=512)
    ap.add_argument("--workers", type=int, default=min(16, mp.cpu_count()))
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    out = Path(args.out)
    total = args.train + args.eval

    # 1. sample unique mazes (over-sample to absorb duplicate rejections)
    seen, records = set(), []
    attempt, batch = 0, max(total // 4, 1000)
    with mp.Pool(args.workers) as pool, tqdm(total=total, desc="sampling mazes") as bar:
        while len(records) < total:
            jobs = [(attempt + i, args.size, args.min_len, args.seed) for i in range(batch)]
            attempt += batch
            for fp, rec_json in pool.imap_unordered(_worker, jobs, chunksize=32):
                if fp in seen:
                    continue
                seen.add(fp)
                records.append(rec_json)
                bar.update(1)
                if len(records) >= total:
                    break

    # 2. deterministic shuffle + split, then renumber ids per split
    random.Random(args.seed).shuffle(records)
    splits = {"train": records[:args.train], "eval": records[args.train:total]}

    meta = {"size": args.size, "min_len": args.min_len, "size_px": args.size_px,
            "seed": args.seed, "counts": {k: len(v) for k, v in splits.items()}}
    out.mkdir(parents=True, exist_ok=True)

    for split, recs in splits.items():
        sdir = out / split
        (sdir / "puzzle").mkdir(parents=True, exist_ok=True)
        (sdir / "solution").mkdir(parents=True, exist_ok=True)
        for i, rj in enumerate(recs):
            rj["id"] = f"{i:06d}"
        jobs = [(str(sdir), rj, args.size_px) for rj in recs]
        with mp.Pool(args.workers) as pool:
            list(tqdm(pool.imap_unordered(_render_worker, jobs, chunksize=16),
                      total=len(jobs), desc=f"rendering {split}"))
        with open(sdir / "manifest.jsonl", "w") as f:
            for rj in recs:
                f.write(json.dumps(rj) + "\n")

    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\ndone -> {out.resolve()}")
    for k, v in meta["counts"].items():
        print(f"  {k:6s} {v}")


if __name__ == "__main__":
    main()
