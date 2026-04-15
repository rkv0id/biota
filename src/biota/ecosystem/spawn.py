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

Per-creature count (n) is passed separately from SpawnConfig because in v2.2.0
different sources in a heterogeneous run contribute different numbers of
creatures. SpawnConfig only carries values shared across all sources.
"""

import numpy as np
import torch

from biota.ecosystem.config import SpawnConfig


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
    n: int,
    grid_h: int,
    grid_w: int,
    patch_override: int | None = None,
) -> list[tuple[int, int]]:
    """Return (y, x) spawn center coordinates using Poisson disk sampling.

    Falls back to jittered grid if Poisson disk cannot place all n creatures.
    Always returns exactly n positions.

    patch_override is used by the multi-species spawn path to compute the
    border margin from the largest per-species patch rather than spawn.patch.
    Single-species callers leave it None.
    """
    rng = np.random.default_rng(spawn.seed)
    effective_patch = patch_override if patch_override is not None else spawn.patch
    margin = effective_patch + 2  # keep patches off the wall border
    # Use the smaller dimension for the Poisson disk to guarantee min_dist
    grid_min = min(grid_h, grid_w)

    positions = _poisson_disk_sample(
        n=n,
        grid=grid_min,
        min_dist=float(spawn.min_dist),
        margin=margin,
        rng=rng,
    )

    if len(positions) < n:
        positions = _jittered_grid_fallback(n, grid_min, margin, rng)

    # Scale positions to actual grid_h x grid_w if rectangular
    if grid_h != grid_w:
        positions = [(int(y * grid_h / grid_min), int(x * grid_w / grid_min)) for y, x in positions]

    return positions


def build_initial_state(
    spawn: SpawnConfig,
    n: int,
    grid_h: int,
    grid_w: int,
    device: str,
    patch_override: int | None = None,
) -> torch.Tensor:
    """Build a (grid_h, grid_w, 1) initial state with n creatures on a shared grid.

    Spawn positions are determined by Poisson disk sampling. Each creature gets
    an independent random patch (uniform random in [0, 1) per cell) centered
    at its position. Patch i uses RNG seed spawn.seed + 1000 + i so position
    sampling and patch initialization use independent random streams.

    patch_override lets callers supply a per-source patch size (used by the
    homogeneous-run path when CreatureSource.patch is set) without mutating
    SpawnConfig. When None, falls back to spawn.patch.

    Args:
        spawn:          Spawn layout parameters (min_dist, patch, seed).
        n:              Number of creatures to place.
        grid_h:         Ecosystem grid height.
        grid_w:         Ecosystem grid width.
        device:         Torch device for the output tensor.
        patch_override: Optional override for per-creature patch side length.

    Returns:
        state: (grid_h, grid_w, 1) float32 tensor on device.
    """
    patch = patch_override if patch_override is not None else spawn.patch
    positions = compute_spawn_positions(spawn, n, grid_h, grid_w, patch_override=patch)
    state = torch.zeros((grid_h, grid_w, 1), dtype=torch.float32, device=device)

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


def build_initial_state_multi_species(
    spawn: SpawnConfig,
    counts: list[int],
    grid_h: int,
    grid_w: int,
    device: str,
    patches: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build mass and species-weight initial states for a heterogeneous run.

    Total creature count is sum(counts). Spawn positions are generated in one
    Poisson disk pass against the union, then sliced per-species in order
    (species 0 takes the first counts[0] positions, species 1 the next
    counts[1], and so on). This keeps spawn behaviour deterministic and
    reproducible from a single seed regardless of how many species are
    involved.

    Each spawned patch sets the mass channel (uniform random in [0, 1) per
    cell, same as build_initial_state) and pins species ownership to a
    one-hot for that species' index in the counts list.

    patches is an optional per-species list of patch sizes; if omitted, every
    species uses spawn.patch. The Poisson disk margin uses max(patches) so
    the largest creature still fits inside the wall border. patches must be
    the same length as counts.

    Returns:
        mass:    (H, W, 1) float32 tensor on device.
        weights: (H, W, S) float32 tensor on device, S = len(counts). Where
                 mass > 0, the weight vector is one-hot for the owning
                 species. Elsewhere it is all zeros.
    """
    if not counts:
        raise ValueError("counts must contain at least one species count")
    if any(c <= 0 for c in counts):
        raise ValueError(f"counts must all be positive, got {counts}")

    if patches is None:
        patches = [spawn.patch] * len(counts)
    if len(patches) != len(counts):
        raise ValueError(
            f"patches length {len(patches)} does not match counts length {len(counts)}"
        )
    if any(p <= 0 for p in patches):
        raise ValueError(f"patches must all be positive, got {patches}")

    total_n = sum(counts)
    s = len(counts)
    # Use the largest per-species patch for the Poisson disk margin so the
    # biggest creature still fits inside the wall border. min_dist itself is
    # not adjusted since it controls inter-creature spacing, not edge buffer.
    max_patch = max(patches)
    positions = compute_spawn_positions(spawn, total_n, grid_h, grid_w, patch_override=max_patch)
    mass = torch.zeros((grid_h, grid_w, 1), dtype=torch.float32, device=device)
    weights = torch.zeros((grid_h, grid_w, s), dtype=torch.float32, device=device)

    # Walk positions in order, assigning each to its species via the prefix
    # sums of counts. species_of[i] is the species index for creature i.
    species_of: list[int] = []
    for sp_idx, c in enumerate(counts):
        species_of.extend([sp_idx] * c)

    for i, ((cy, cx), sp_idx) in enumerate(zip(positions, species_of, strict=True)):
        rng = np.random.default_rng(spawn.seed + 1000 + i)
        patch = patches[sp_idx]
        patch_data = rng.random((patch, patch), dtype=np.float32)

        y0 = max(0, cy - patch // 2)
        x0 = max(0, cx - patch // 2)
        y1 = min(grid_h, y0 + patch)
        x1 = min(grid_w, x0 + patch)
        ph = y1 - y0
        pw = x1 - x0

        patch_tensor = torch.from_numpy(patch_data[:ph, :pw]).to(device)
        mass[y0:y1, x0:x1, 0] = patch_tensor
        # One-hot weight for this species, but only where the patch deposited
        # mass. This keeps weights at 0 in cells the patch didn't touch.
        weights[y0:y1, x0:x1, sp_idx] = (patch_tensor > 0).to(torch.float32)

    return mass, weights
