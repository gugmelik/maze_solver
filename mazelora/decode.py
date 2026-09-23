"""Decode a (possibly model-generated) maze image back into a symbolic solution.

The trick that makes this robust: we sample *edge midpoints* rather than cell
interiors. At the midpoint of the boundary between two adjacent cells the
renderer produces exactly one of three well-separated appearances:

    wall present      -> black
    passage, no path  -> white / light grey hairline
    passage, on path  -> red

So a single sampler yields both the drawn path (as an edge set, which converts
straight to U/D/L/R moves) and the maze structure the model reproduced. Cell
interiors would instead be occluded by the yellow/blue endpoint dots.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .maze import Geometry, MazeRecord, path_to_udrl

# pixel classification thresholds (generous: model output is not pixel-exact)
RED_MARGIN = 60      # R must exceed max(G,B) by this much
RED_MIN = 100
DARK_MAX = 80        # all channels below -> wall
LIGHT_MIN = 140      # all channels above -> open passage

# per-edge decision thresholds, as a fraction of the sampled patch
RED_FRAC = 0.30
DARK_FRAC = 0.50


def _masks(rgb: np.ndarray):
    r, g, b = rgb[..., 0].astype(np.int16), rgb[..., 1].astype(np.int16), rgb[..., 2].astype(np.int16)
    red = (r - np.maximum(g, b) > RED_MARGIN) & (r > RED_MIN)
    dark = (np.maximum(np.maximum(r, g), b) < DARK_MAX)
    light = (np.minimum(np.minimum(r, g), b) > LIGHT_MIN)
    return red, dark, light


def _patch_bounds(g: Geometry, a, b) -> tuple[int, int, int, int]:
    """Small window centred on the a|b boundary, inset to stay clear of corners."""
    cx, cy = g.edge_midpoint(a, b)
    horizontal = a[0] == b[0]          # cells side by side -> vertical boundary
    ex = g.cell * (0.10 if horizontal else 0.15)
    ey = g.cell * (0.15 if horizontal else 0.10)
    x0, x1 = int(round(cx - ex)), int(round(cx + ex))
    y0, y1 = int(round(cy - ey)), int(round(cy + ey))
    return max(x0, 0), max(y0, 0), min(x1, g.size_px), min(y1, g.size_px)


@dataclass
class Decoded:
    size: int
    drawn_edges: set = field(default_factory=set)   # frozenset({a,b}) carrying red
    open_edges: set = field(default_factory=set)    # frozenset({a,b}) that look like passages
    wall_edges: set = field(default_factory=set)    # frozenset({a,b}) that look like walls
    start: tuple | None = None                      # cell of the yellow marker
    goal: tuple | None = None                       # cell of the blue marker
    route: list | None = None                       # start->goal route through drawn_edges
    udrl: str | None = None
    red_mask: np.ndarray | None = None


def _marker_cell(mask: np.ndarray, g: Geometry):
    if mask.sum() < 20:
        return None
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    r = min(g.n - 1, max(0, int(cy // g.cell)))
    c = min(g.n - 1, max(0, int(cx // g.cell)))
    return (r, c)


def _route(drawn: set, start, goal):
    """Shortest start->goal route using only drawn edges (None if disconnected)."""
    if start is None or goal is None:
        return None
    adj: dict = {}
    for e in drawn:
        a, b = tuple(e)
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    if start not in adj and start != goal:
        return None
    prev, q = {start: None}, deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            out = []
            while cur is not None:
                out.append(cur)
                cur = prev[cur]
            return out[::-1]
        for nxt in adj.get(cur, ()):
            if nxt not in prev:
                prev[nxt] = cur
                q.append(nxt)
    return None


def decode_image(img: Image.Image, size: int, size_px: int | None = None) -> Decoded:
    """Decode without any ground-truth knowledge except the grid dimension."""
    img = img.convert("RGB")
    if size_px is None:
        size_px = img.size[0]
    if img.size != (size_px, size_px):
        img = img.resize((size_px, size_px), Image.BILINEAR)
    rgb = np.asarray(img)
    red, dark, light = _masks(rgb)
    g = Geometry(size, size_px)

    d = Decoded(size=size, red_mask=red)
    for a, b in g.internal_edges():
        x0, y0, x1, y1 = _patch_bounds(g, a, b)
        pr, pd = red[y0:y1, x0:x1], dark[y0:y1, x0:x1]
        n = max(pr.size, 1)
        red_frac, dark_frac = pr.sum() / n, pd.sum() / n
        e = frozenset((a, b))
        if red_frac >= RED_FRAC:
            d.drawn_edges.add(e)
            d.open_edges.add(e)          # the model drew a path through it
        elif dark_frac >= DARK_FRAC:
            d.wall_edges.add(e)
        else:
            d.open_edges.add(e)

    r, gr, b = rgb[..., 0].astype(np.int16), rgb[..., 1].astype(np.int16), rgb[..., 2].astype(np.int16)
    d.start = _marker_cell((r > 150) & (gr > 150) & (b < 110), g)
    d.goal = _marker_cell((b > 150) & (r < 110) & (gr < 130), g)
    d.route = _route(d.drawn_edges, d.start, d.goal)
    d.udrl = path_to_udrl(d.route) if d.route else None
    return d


def decode_against(img: Image.Image, rec: MazeRecord) -> Decoded:
    """Decode, then re-resolve the route using the *true* start/goal cells.

    Using the ground-truth endpoints separates two failure modes: 'the model
    moved the markers' (caught by `endpoints_ok`) and 'the model drew a bad
    path' (caught by the route metrics).
    """
    d = decode_image(img, rec.size)
    d.route = _route(d.drawn_edges, rec.start, rec.goal)
    d.udrl = path_to_udrl(d.route) if d.route else None
    return d


def record_from_image(img: Image.Image, size: int, sample_id: str = "uploaded"):
    """Recover a full MazeRecord (walls, endpoints, true shortest path) from a
    *puzzle* image alone.

    This is what lets the app score a maze it has never seen -- an upload, or a
    puzzle generated outside this repo -- with the same metrics as the eval set.
    Returns None if the image does not decode into a solvable maze.
    """
    from .maze import (MazeRecord, WALL_E, WALL_N, WALL_S, WALL_W,
                       path_to_udrl, shortest_path_bfs, wall_codes_to_grid)

    d = decode_image(img, size)
    if d.start is None or d.goal is None or d.start == d.goal:
        return None

    codes = [[0] * size for _ in range(size)]
    for r in range(size):
        for c in range(size):
            v = 0
            if r == 0 or frozenset(((r, c), (r - 1, c))) in d.wall_edges: v |= WALL_N
            if r == size - 1 or frozenset(((r, c), (r + 1, c))) in d.wall_edges: v |= WALL_S
            if c == 0 or frozenset(((r, c), (r, c - 1))) in d.wall_edges: v |= WALL_W
            if c == size - 1 or frozenset(((r, c), (r, c + 1))) in d.wall_edges: v |= WALL_E
            codes[r][c] = v

    try:
        path = shortest_path_bfs(wall_codes_to_grid(codes), d.start, d.goal)
    except ValueError:
        return None
    return MazeRecord(id=sample_id, size=size, start=d.start, goal=d.goal,
                      path=path, udrl=path_to_udrl(path), walls=codes)
