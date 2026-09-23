"""Per-sample and aggregate scoring for generated maze solutions."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .decode import Decoded, decode_against
from .maze import Geometry, MazeRecord


def gt_open_edges(rec: MazeRecord) -> set[frozenset]:
    grid, n, out = rec.grid, rec.size, set()
    for r in range(n):
        for c in range(n):
            if c < n - 1 and not grid[r][c]["E"]:
                out.add(frozenset(((r, c), (r, c + 1))))
            if r < n - 1 and not grid[r][c]["S"]:
                out.add(frozenset(((r, c), (r + 1, c))))
    return out


def _prf(pred: set, gt: set) -> tuple[float, float, float, float]:
    inter = len(pred & gt)
    union = len(pred | gt)
    p = inter / len(pred) if pred else 0.0
    r = inter / len(gt) if gt else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1, (inter / union if union else 1.0)


def score_sample(img: Image.Image, rec: MazeRecord,
                 gt_img: Image.Image | None = None) -> dict:
    """Score one generated image against its maze record.

    `solved` is the headline metric: the model drew a connected red route from
    start to goal that never crosses a wall. Because a DFS maze is *perfect*
    (exactly one simple route between any two cells), a wall-legal route is
    necessarily the shortest one, so `solved` also implies path optimality.
    """
    d: Decoded = decode_against(img, rec)
    gt_edges = rec.path_edges
    open_edges = gt_open_edges(rec)

    violations = sorted(
        tuple(sorted(tuple(x) for x in e)) for e in d.drawn_edges if e not in open_edges
    )
    route_edges = (
        {frozenset((a, b)) for a, b in zip(d.route, d.route[1:])} if d.route else set()
    )
    route_legal = bool(d.route) and all(e in open_edges for e in route_edges)

    p, r, f1, iou = _prf(d.drawn_edges, gt_edges)

    # structure preservation: did the model redraw the same maze walls?
    # judged only on edges it did not paint over, so the path cannot confound it.
    n_struct = n_struct_ok = 0
    all_edges = {frozenset((a, b)) for a, b in Geometry(rec.size).internal_edges()}
    for e in all_edges - d.drawn_edges:
        n_struct += 1
        pred_open = e in d.open_edges
        if pred_open == (e in open_edges):
            n_struct_ok += 1

    out = {
        "id": rec.id,
        "size": rec.size,
        "gt_udrl": rec.udrl,
        "pred_udrl": d.udrl,
        "path_len": len(rec.path),
        "solved": bool(route_legal),
        "route_found": bool(d.route),
        "exact_edges": d.drawn_edges == gt_edges,
        "edge_precision": p,
        "edge_recall": r,
        "edge_f1": f1,
        "edge_iou": iou,
        "n_drawn_edges": len(d.drawn_edges),
        "n_gt_edges": len(gt_edges),
        "wall_violations": len(violations),
        "violation_edges": violations[:8],
        "endpoints_ok": bool(d.start == rec.start and d.goal == rec.goal),
        "start_ok": bool(d.start == rec.start),
        "goal_ok": bool(d.goal == rec.goal),
        "structure_acc": (n_struct_ok / n_struct) if n_struct else 1.0,
    }

    if gt_img is not None:
        from .decode import _masks
        gt_red = _masks(np.asarray(gt_img.convert("RGB")))[0]
        pr = d.red_mask
        if pr.shape != gt_red.shape:
            gt_red = np.asarray(Image.fromarray(gt_red.astype(np.uint8) * 255)
                                .resize(pr.shape[::-1], Image.NEAREST)) > 127
        inter = np.logical_and(pr, gt_red).sum()
        union = np.logical_or(pr, gt_red).sum()
        out["red_pixel_iou"] = float(inter / union) if union else 1.0
    return out


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = ["solved", "route_found", "exact_edges", "edge_precision", "edge_recall",
            "edge_f1", "edge_iou", "endpoints_ok", "start_ok", "goal_ok",
            "structure_acc", "wall_violations", "red_pixel_iou"]
    agg = {"n": len(rows)}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r and r[k] is not None]
        if vals:
            agg[k] = float(np.mean(vals))

    by_len: dict[int, list[dict]] = {}
    for r in rows:
        by_len.setdefault(r["path_len"], []).append(r)
    agg["solved_by_path_len"] = {
        str(L): {"n": len(v), "solved": float(np.mean([x["solved"] for x in v]))}
        for L, v in sorted(by_len.items())
    }
    return agg
