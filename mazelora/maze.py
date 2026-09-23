"""Maze generation, rendering and wall codec.

The grid/BFS/rendering logic is ported from DiffThinker's `Maze/gen_image.py`
(https://github.com/lcqysl/DiffThinker) so that our ground-truth images are
pixel-identical to the reference task. Reorganised here into a library with
deterministic seeding and a structured maze record.

Rendering conventions (these are what the evaluator decodes):
  * background / wall          -> black
  * cell interior + passages   -> white
  * passage separator hairline -> light grey (224,224,224)
  * start cell marker          -> yellow dot
  * goal cell marker           -> blue dot
  * solution                   -> red polyline through cell centres
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

from PIL import Image, ImageDraw

# Wall bit codec used by the .txt format of the reference repo.
WALL_N, WALL_S, WALL_W, WALL_E = 1, 2, 4, 8

IMAGE_SIZE = 512
GRID_COLOR = (224, 224, 224)
START_COLOR = "yellow"
GOAL_COLOR = "blue"
PATH_COLOR = "red"


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def _make_empty_grid(n: int):
    return [[{"N": True, "E": True, "S": True, "W": True, "visited": False}
             for _ in range(n)] for _ in range(n)]


def _neighbors_of(r: int, c: int, n: int):
    res = []
    if r > 0:     res.append(("N", r - 1, c))
    if c < n - 1: res.append(("E", r, c + 1))
    if r < n - 1: res.append(("S", r + 1, c))
    if c > 0:     res.append(("W", r, c - 1))
    return res


def _remove_wall(grid, r, c, d):
    if d == "N":   grid[r][c]["N"], grid[r - 1][c]["S"] = False, False
    elif d == "S": grid[r][c]["S"], grid[r + 1][c]["N"] = False, False
    elif d == "E": grid[r][c]["E"], grid[r][c + 1]["W"] = False, False
    elif d == "W": grid[r][c]["W"], grid[r][c - 1]["E"] = False, False


def gen_maze_dfs(n: int, rng: random.Random):
    """Randomised-DFS perfect maze: exactly one simple path between any two cells."""
    grid = _make_empty_grid(n)
    sr, sc = rng.randrange(n), rng.randrange(n)
    grid[sr][sc]["visited"] = True
    stack = [(sr, sc)]
    while stack:
        r, c = stack[-1]
        unvisited = [(d, nr, nc) for (d, nr, nc) in _neighbors_of(r, c, n)
                     if not grid[nr][nc]["visited"]]
        if unvisited:
            d, nr, nc = rng.choice(unvisited)
            _remove_wall(grid, r, c, d)
            grid[nr][nc]["visited"] = True
            stack.append((nr, nc))
        else:
            stack.pop()
    for row in grid:
        for cell in row:
            cell.pop("visited", None)
    return grid


def shortest_path_bfs(grid, start, end):
    n = len(grid)
    q = deque([(start, [start])])
    seen = {start}
    while q:
        (r, c), path = q.popleft()
        if (r, c) == end:
            return [tuple(p) for p in path]
        cell = grid[r][c]
        for dr, dc, wall in ((-1, 0, "N"), (1, 0, "S"), (0, -1, "W"), (0, 1, "E")):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and not cell[wall] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append(((nr, nc), path + [(nr, nc)]))
    raise ValueError("no path exists")


def path_to_udrl(path: Sequence[tuple[int, int]]) -> str:
    moves = []
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        if r2 < r1:   moves.append("U")
        elif r2 > r1: moves.append("D")
        elif c2 < c1: moves.append("L")
        else:         moves.append("R")
    return "".join(moves)


def grid_to_wall_codes(grid) -> list[list[int]]:
    n = len(grid)
    out = []
    for r in range(n):
        row = []
        for c in range(n):
            cell, v = grid[r][c], 0
            if cell["N"]: v |= WALL_N
            if cell["S"]: v |= WALL_S
            if cell["W"]: v |= WALL_W
            if cell["E"]: v |= WALL_E
            row.append(v)
        out.append(row)
    return out


def wall_codes_to_grid(codes: Sequence[Sequence[int]]):
    n = len(codes)
    return [[{"N": bool(codes[r][c] & WALL_N), "S": bool(codes[r][c] & WALL_S),
              "W": bool(codes[r][c] & WALL_W), "E": bool(codes[r][c] & WALL_E)}
             for c in range(n)] for r in range(n)]


@dataclass
class MazeRecord:
    """Everything needed to render, score and re-derive a maze sample."""
    id: str
    size: int
    start: tuple[int, int]
    goal: tuple[int, int]
    path: list[tuple[int, int]]
    udrl: str
    walls: list[list[int]]

    def to_json(self) -> dict:
        d = asdict(self)
        d["start"] = list(self.start)
        d["goal"] = list(self.goal)
        d["path"] = [list(p) for p in self.path]
        return d

    @staticmethod
    def from_json(d: dict) -> "MazeRecord":
        return MazeRecord(
            id=d["id"], size=d["size"],
            start=tuple(d["start"]), goal=tuple(d["goal"]),
            path=[tuple(p) for p in d["path"]], udrl=d["udrl"], walls=d["walls"],
        )

    @property
    def grid(self):
        return wall_codes_to_grid(self.walls)

    @property
    def path_edges(self) -> set[frozenset]:
        return {frozenset((a, b)) for a, b in zip(self.path, self.path[1:])}


def sample_maze(size: int, min_len: int, rng: random.Random,
                max_tries: int = 500, sample_id: str = "") -> MazeRecord:
    """Rejection-sample a maze whose unique start->goal path has >= min_len cells."""
    for _ in range(max_tries):
        grid = gen_maze_dfs(size, rng)
        cells = [(r, c) for r in range(size) for c in range(size)]
        start, goal = rng.sample(cells, 2)
        try:
            path = shortest_path_bfs(grid, start, goal)
        except ValueError:
            continue
        if len(path) >= min_len:
            return MazeRecord(id=sample_id, size=size, start=start, goal=goal,
                              path=path, udrl=path_to_udrl(path),
                              walls=grid_to_wall_codes(grid))
    raise RuntimeError(f"could not sample a {size}x{size} maze with path >= {min_len}")


# --------------------------------------------------------------------------- #
# geometry (shared by renderer and decoder)
# --------------------------------------------------------------------------- #
class Geometry:
    """Pixel geometry of a rendered maze. The decoder relies on this too."""

    def __init__(self, n: int, size_px: int = IMAGE_SIZE):
        self.n = n
        self.size_px = size_px
        self.cell = size_px / n
        self.wall_w = self.cell / 4.0
        self.half_wall = self.wall_w / 2.0
        self.grid_w = max(1, int(self.cell / 16.0))
        self.dot_radius = max(2, int((self.cell - self.wall_w) * 0.25))

    def center(self, rc: tuple[int, int]) -> tuple[float, float]:
        r, c = rc
        return (c * self.cell + self.cell / 2, r * self.cell + self.cell / 2)

    def edge_midpoint(self, a: tuple[int, int], b: tuple[int, int]) -> tuple[float, float]:
        """Pixel midpoint of the boundary shared by two 4-adjacent cells."""
        (ax, ay), (bx, by) = self.center(a), self.center(b)
        return ((ax + bx) / 2, (ay + by) / 2)

    def internal_edges(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        out = []
        for r in range(self.n):
            for c in range(self.n):
                if c < self.n - 1: out.append(((r, c), (r, c + 1)))
                if r < self.n - 1: out.append(((r, c), (r + 1, c)))
        return out


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def render_maze(grid, start, goal, path: Iterable | None = None,
                size_px: int = IMAGE_SIZE) -> Image.Image:
    n = len(grid)
    g = Geometry(n, size_px)
    img = Image.new("RGB", (size_px, size_px), "black")
    draw = ImageDraw.Draw(img)
    cs, ww, hw = g.cell, g.wall_w, g.half_wall

    for r in range(n):
        for c in range(n):
            x1, y1 = c * cs + hw, r * cs + hw
            x2, y2 = (c + 1) * cs - hw, (r + 1) * cs - hw
            draw.rectangle([(x1, y1), (x2, y2)], fill="white")
            cell = grid[r][c]
            if not cell["S"] and r < n - 1:
                draw.rectangle([(x1, y2), (x2, y2 + ww)], fill="white")
            if not cell["E"] and c < n - 1:
                draw.rectangle([(x2, y1), (x2 + ww, y2)], fill="white")

    for r in range(n):
        for c in range(n):
            if r < n - 1 and not grid[r][c]["S"]:
                y = (r + 1) * cs
                draw.line([(c * cs + hw, y), ((c + 1) * cs - hw, y)],
                          fill=GRID_COLOR, width=g.grid_w)
            if c < n - 1 and not grid[r][c]["E"]:
                x = (c + 1) * cs
                draw.line([(x, r * cs + hw), (x, (r + 1) * cs - hw)],
                          fill=GRID_COLOR, width=g.grid_w)

    if path is not None:
        pts = [g.center(rc) for rc in path]
        draw.line(pts, fill=PATH_COLOR, width=max(1, int(ww)), joint="curve")

    def dot(rc, color):
        cx, cy = g.center(rc)
        rad = g.dot_radius
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=color)

    dot(start, START_COLOR)
    dot(goal, GOAL_COLOR)
    return img


def render_record(rec: MazeRecord, with_path: bool, size_px: int = IMAGE_SIZE) -> Image.Image:
    return render_maze(rec.grid, rec.start, rec.goal,
                       path=rec.path if with_path else None, size_px=size_px)
