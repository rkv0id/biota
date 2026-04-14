"""Build the initial state for an ecosystem simulation.

Spawn positions are generated via Poisson disk sampling: a minimum separation
distance between creature centers is enforced, but positions are otherwise
random. This gives more naturalistic initial conditions than a regular grid
while guaranteeing creatures don't overlap or start too close.

The algorithm is Bridson's dart-throwing variant (O(n)):
  - Place a random seed point
  - Maintain an active list of candidate points
  - For each candidate, attempt k=30 random annular samples at distance
    [min_dist, 2*min_dist]
  - Accept samples that are at least min_dist from all existing points
  - Continue until n points are placed or the active list is exhausted

If Poisson disk sampling fails to place n creatures (grid too small for the
requested n and min_dist), the function falls back to a jittered grid so the
run still proceeds rather than crashing.
"""

import numpy as np
import torch

from biota.ecosystem.result import SpawnConfig


def _poisson_disk_sample(
    n: int,
    grid: int,
    min_dist: float,
    margin: int,
    rng: np.random.Generator,
    max_attempts: int = 30,
) -> list[tuple[int, int]]:
    """Bridson's Poisson disk sampling on a square grid.

    Places up to n points such that no two are closer than min_dist.
    Points are kept at least margin pixels from the grid edge so creatures
    are not cut off by the wall border.

    Returns a list of (y, x) integer coordinates. May return fewer than n
    points if the grid is too crowded.
    """
    lo, hi = margin, grid - margin
    if lo >= hi:
        return []

    # Background grid for O(1) neighbour lookup
    cell = min_dist / np.sqrt(2.0)
    bg: dict[tuple[int, int], tuple[float, float]] = {}

    def grid_key(py: float, px: float) -> tuple[int, int]:
        return int((py - lo) / cell), int((px - lo) / cell)

    def too_close(py: float, px: float) -> bool:
        gy, gx = grid_key(py, px)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nb = (gy + dy, gx + dx)
                if nb in bg:
                    ey, ex = bg[nb]
                    if (py - ey) ** 2 + (px - ex) ** 2 < min_dist**2:
                        return True
        return False

    # Seed point
    sy = float(rng.uniform(lo, hi))
    sx = float(rng.uniform(lo, hi))
    points: list[tuple[float, float]] = [(sy, sx)]
    bg[grid_key(sy, sx)] = (sy, sx)
    active = [0]

    while active and len(points) < n:
        idx = int(rng.integers(0, len(active)))
        cy, cx = points[active[idx]]
        placed = False
        for _ in range(max_attempts):
            angle = float(rng.uniform(0, 2 * np.pi))
            dist = float(rng.uniform(min_dist, 2 * min_dist))
            ny = cy + dist * np.sin(angle)
            nx = cx + dist * np.cos(angle)
            if not (lo <= ny < hi and lo <= nx < hi):
                continue
            if too_close(ny, nx):
                continue
            points.append((ny, nx))
            bg[grid_key(ny, nx)] = (ny, nx)
            active.append(len(points) - 1)
            placed = True
            if len(points) >= n:
                break
        if not placed:
            active.pop(idx)

    return [(round(y), round(x)) for y, x in points[:n]]


def _jittered_grid_fallback(
    n: int,
    grid: int,
    margin: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Jittered grid fallback when Poisson disk can't place n points.

    Divides the usable area into n cells and places one point per cell with
    uniform random jitter. Always places exactly n points.
    """
    lo, hi = margin, grid - margin
    span = hi - lo
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    cell_h = span / rows
    cell_w = span / cols
    positions: list[tuple[int, int]] = []
    for i in range(n):
        r, c = divmod(i, cols)
        cy = lo + (r + rng.random()) * cell_h
        cx = lo + (c + rng.random()) * cell_w
        positions.append((int(np.clip(round(cy), lo, hi - 1)), int(np.clip(round(cx), lo, hi - 1))))
    return positions[:n]


def compute_spawn_positions(
    spawn: SpawnConfig,
    grid_h: int,
    grid_w: int,
) -> list[tuple[int, int]]:
    """Return (y, x) spawn center coordinates using Poisson disk sampling.

    Falls back to jittered grid if Poisson disk cannot place all n creatures.
    Always returns exactly spawn.n positions.
    """
    rng = np.random.default_rng(spawn.seed)
    margin = spawn.patch + 2  # keep patches off the wall border
    # Use the smaller dimension for the Poisson disk to guarantee min_dist
    grid_min = min(grid_h, grid_w)

    positions = _poisson_disk_sample(
        n=spawn.n,
        grid=grid_min,
        min_dist=float(spawn.min_dist),
        margin=margin,
        rng=rng,
    )

    if len(positions) < spawn.n:
        positions = _jittered_grid_fallback(spawn.n, grid_min, margin, rng)

    # Scale positions to actual grid_h x grid_w if rectangular
    if grid_h != grid_w:
        positions = [(int(y * grid_h / grid_min), int(x * grid_w / grid_min)) for y, x in positions]

    return positions


def build_initial_state(
    spawn: SpawnConfig,
    grid_h: int,
    grid_w: int,
    device: str,
) -> torch.Tensor:
    """Build a (grid_h, grid_w, 1) initial state with n creatures on a shared grid.

    Spawn positions are determined by Poisson disk sampling. Each creature gets
    an independent random patch of size spawn.patch x spawn.patch centered at
    its position. Patch i uses RNG seed spawn.seed + 1000 + i so position
    sampling and patch initialization use independent random streams.

    Args:
        spawn:   Spawn configuration (n, min_dist, patch, seed).
        grid:    Ecosystem grid side length.
        device:  Torch device for the output tensor.

    Returns:
        state:     (grid, grid, 1) float32 tensor on device.
    """
    positions = compute_spawn_positions(spawn, grid_h, grid_w)
    state = torch.zeros((grid_h, grid_w, 1), dtype=torch.float32, device=device)
    patch = spawn.patch

    for i, (cy, cx) in enumerate(positions):
        rng = np.random.default_rng(spawn.seed + 1000 + i)
        patch_data = rng.random((patch, patch), dtype=np.float32)

        y0 = max(0, cy - patch // 2)
        x0 = max(0, cx - patch // 2)
        y1 = min(grid_h, y0 + patch)
        x1 = min(grid_w, x0 + patch)

        ph = y1 - y0
        pw = x1 - x0
        patch_tensor = torch.from_numpy(patch_data[:ph, :pw]).to(device)
        state[y0:y1, x0:x1, 0] = patch_tensor

    return state
