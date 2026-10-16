#!/usr/bin/env python3
"""Score Klein generations with the official EndoCoT Maze path metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


WALL_MASKS = {"N": 1, "S": 2, "W": 4, "E": 8}
MOVES = {
    "U": (-1, 0, "N"),
    "D": (1, 0, "S"),
    "L": (0, -1, "W"),
    "R": (0, 1, "E"),
}


def parse_maze_txt(txt_path: Path):
    with txt_path.open(encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    size = int(lines[0])
    start = tuple(map(int, lines[1].split()))
    end = tuple(map(int, lines[2].split()))
    grid = [list(map(int, lines[3 + row].split())) for row in range(size)]
    return {"size": size, "start": start, "end": end, "grid": grid}


def extract_actions(img_path: Path, maze_info: dict) -> str:
    rows = maze_info["size"]
    cols = maze_info["size"]
    start_pos = maze_info["start"]
    grid_walls = maze_info["grid"]

    image = Image.open(img_path).convert("RGB").resize((512, 512), Image.BILINEAR)
    pixels = np.asarray(image)
    red = pixels[:, :, 0].astype(float)
    green = pixels[:, :, 1].astype(float)
    blue = pixels[:, :, 2].astype(float)
    red_mask = (red > 100) & (red > green * 1.2) & (red > blue * 1.2)

    cell_h = 512 // rows
    cell_w = 512 // cols
    path_grid = np.zeros((rows, cols), dtype=bool)
    for row in range(rows):
        for col in range(cols):
            y0, y1 = int(row * cell_h), int((row + 1) * cell_h)
            x0, x1 = int(col * cell_w), int((col + 1) * cell_w)
            margin = int(cell_h * 0.2) if cell_h > 10 else 2
            sub = red_mask[y0 + margin : y1 - margin, x0 + margin : x1 - margin]
            if sub.size > 0 and float(np.mean(sub)) > 0.1:
                path_grid[row, col] = True

    actions = []
    visited = {start_pos}
    curr_r, curr_c = start_pos
    directions = [("R", MOVES["R"]), ("D", MOVES["D"]), ("L", MOVES["L"]), ("U", MOVES["U"])]
    for _ in range(rows * cols * 2):
        found = False
        wall_val = grid_walls[curr_r][curr_c]
        for act, (dr, dc, wall_dir) in directions:
            nr, nc = curr_r + dr, curr_c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if wall_val & WALL_MASKS[wall_dir]:
                continue
            if path_grid[nr, nc] and (nr, nc) not in visited:
                visited.add((nr, nc))
                actions.append(act)
                curr_r, curr_c = nr, nc
                found = True
                break
        if not found:
            break
    return "".join(actions)


def verify_path(maze_info: dict, path: str) -> bool:
    grid = maze_info["grid"]
    size = maze_info["size"]
    current = maze_info["start"]
    cell_walls = []
    for row in grid:
        cell_walls.append(
            [
                {
                    "N": (value & 1) != 0,
                    "S": (value & 2) != 0,
                    "W": (value & 4) != 0,
                    "E": (value & 8) != 0,
                }
                for value in row
            ]
        )
    for move in path.replace(",", "").replace(" ", ""):
        if move not in MOVES:
            continue
        dr, dc, wall = MOVES[move]
        r, c = current
        if cell_walls[r][c][wall]:
            return False
        nr, nc = r + dr, c + dc
        if not (0 <= nr < size and 0 <= nc < size):
            return False
        current = (nr, nc)
    return current == maze_info["end"]


def prefix_overlap(gt_path: str, pred_path: str) -> float:
    if not gt_path:
        return 0.0
    overlap = 0
    for left, right in zip(gt_path, pred_path):
        if left != right:
            break
        overlap += 1
    return overlap / len(gt_path)


def score_level(test_dir: Path, result_dir: Path) -> dict:
    txt_files = sorted(test_dir.glob("*.txt"), key=lambda path: [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)])
    predictions = {}
    correct = 0
    valid = 0
    for txt_path in txt_files:
        sample_id = txt_path.stem
        maze = parse_maze_txt(txt_path)
        image_path = result_dir / f"{sample_id}.png"
        if not image_path.is_file():
            continue
        path = extract_actions(image_path, maze)
        predictions[f"{sample_id}.png"] = path
        valid += 1
        if verify_path(maze, path):
            correct += 1

    result_dir.mkdir(parents=True, exist_ok=True)
    pred_json = result_dir / "0_result.json"
    pred_json.write_text(json.dumps(predictions, indent=4), encoding="utf-8")

    gt_json = test_dir / "path.json"
    overlap_sum = 0.0
    overlap_count = 0
    if gt_json.is_file():
        gt_data = json.loads(gt_json.read_text(encoding="utf-8"))
        for key, gt_path in gt_data.items():
            clean = key.replace(".png", "")
            pred = predictions.get(clean) or predictions.get(f"{clean}.png")
            if pred is None:
                continue
            overlap_sum += prefix_overlap(gt_path, pred)
            overlap_count += 1

    accuracy = (correct / valid) if valid else 0.0
    overlap = (overlap_sum / overlap_count) if overlap_count else 0.0
    metrics = {
        "valid": valid,
        "correct": correct,
        "path_accuracy": accuracy,
        "prefix_overlap": overlap,
        "pred_json": str(pred_json),
    }
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--gen-root", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[8, 16, 32])
    args = parser.parse_args()

    summary = {}
    checkpoint_dirs = sorted(
        args.gen_root.glob("checkpoint_*"),
        key=lambda path: int(path.name.split("_")[1]),
    )
    for checkpoint_dir in checkpoint_dirs:
        epoch = int(checkpoint_dir.name.split("_")[1])
        summary[str(epoch)] = {}
        for level in args.levels:
            part_name = f"maze_{level}_test"
            result_dir = checkpoint_dir / part_name
            test_dir = args.test_root / f"{level}_test"
            if not result_dir.is_dir() or not test_dir.is_dir():
                continue
            metrics = score_level(test_dir, result_dir)
            summary[str(epoch)][str(level)] = metrics
            print(
                f"epoch={epoch} level={level} "
                f"path_acc={metrics['path_accuracy']:.3f} "
                f"overlap={metrics['prefix_overlap']:.3f} "
                f"correct={metrics['correct']}/{metrics['valid']}"
            )

    out_path = args.gen_root / "maze_test_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
