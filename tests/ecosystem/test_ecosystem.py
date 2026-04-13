"""Tests for the ecosystem simulation module.

Covers Poisson disk sampling, spawn state construction, and the full
run_ecosystem pipeline with a synthetic creature.
"""

import json
import math
import tempfile
from pathlib import Path

import numpy as np

from biota.ecosystem.result import EcosystemConfig, EcosystemMeasures, SpawnConfig
from biota.ecosystem.spawn import (
    _jittered_grid_fallback,  # type: ignore[reportPrivateUsage]
    _poisson_disk_sample,  # type: ignore[reportPrivateUsage]
    build_initial_state,
    compute_spawn_positions,
)
from biota.search.result import RolloutResult

# === helpers ===


def _spawn(n: int = 4, min_dist: int = 60, patch: int = 16, seed: int = 0) -> SpawnConfig:
    return SpawnConfig(n=n, min_dist=min_dist, patch=patch, seed=seed)


def _synthetic_creature() -> RolloutResult:
    """Minimal RolloutResult with 3-kernel params - enough to build a FlowLenia sim."""
    k = 3
    return RolloutResult(
        params={
            "R": 8.0,
            "r": [0.5] * k,
            "m": [0.15] * k,
            "s": [0.015] * k,
            "h": [0.5] * k,
            "a": [[0.5, 0.5, 0.5]] * k,
            "b": [[0.5, 0.5, 0.5]] * k,
            "w": [[0.5, 0.5, 0.5]] * k,
        },
        seed=0,
        descriptors=(0.3, 0.5, 0.6),
        quality=0.8,
        rejection_reason=None,
        thumbnail=np.zeros((16, 32, 32), dtype=np.uint8),
        parent_cell=None,
        created_at=0.0,
        compute_seconds=1.0,
    )


# === Poisson disk sampling ===


def test_poisson_disk_returns_n_points_on_large_grid() -> None:
    rng = np.random.default_rng(42)
    pts = _poisson_disk_sample(n=6, grid=512, min_dist=60.0, margin=20, rng=rng)
    assert len(pts) == 6


def test_poisson_disk_min_distance_respected() -> None:
    rng = np.random.default_rng(7)
    min_dist = 60.0
    pts = _poisson_disk_sample(n=4, grid=512, min_dist=min_dist, margin=20, rng=rng)
    for i, (y1, x1) in enumerate(pts):
        for j, (y2, x2) in enumerate(pts):
            if i >= j:
                continue
            dist = math.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)
            assert dist >= min_dist - 1.0, f"points {i} and {j} too close: {dist:.1f} < {min_dist}"


def test_poisson_disk_points_within_grid_and_margin() -> None:
    rng = np.random.default_rng(0)
    margin = 20
    grid = 256
    pts = _poisson_disk_sample(n=3, grid=grid, min_dist=40.0, margin=margin, rng=rng)
    for y, x in pts:
        assert margin <= y < grid - margin
        assert margin <= x < grid - margin


def test_poisson_disk_returns_fewer_on_crowded_grid() -> None:
    # Impossible to fit 20 points with min_dist=100 on a 128 grid
    rng = np.random.default_rng(0)
    pts = _poisson_disk_sample(n=20, grid=128, min_dist=100.0, margin=10, rng=rng)
    assert len(pts) < 20


def test_jittered_grid_fallback_returns_exactly_n() -> None:
    rng = np.random.default_rng(0)
    pts = _jittered_grid_fallback(n=7, grid=256, margin=10, rng=rng)
    assert len(pts) == 7


def test_compute_spawn_positions_returns_n() -> None:
    spawn = _spawn(n=4, min_dist=60)
    pts = compute_spawn_positions(spawn, grid=512)
    assert len(pts) == 4


def test_compute_spawn_positions_reproducible() -> None:
    spawn = _spawn(n=4, min_dist=60, seed=99)
    pts1 = compute_spawn_positions(spawn, grid=512)
    pts2 = compute_spawn_positions(spawn, grid=512)
    assert pts1 == pts2


def test_compute_spawn_positions_seed_matters() -> None:
    pts1 = compute_spawn_positions(_spawn(seed=0), grid=512)
    pts2 = compute_spawn_positions(_spawn(seed=1), grid=512)
    assert pts1 != pts2


# === build_initial_state ===


def test_build_initial_state_shape() -> None:
    import torch

    spawn = _spawn(n=4, patch=12)
    state = build_initial_state(spawn, grid=256, device="cpu")
    assert state.shape == (256, 256, 1)
    assert state.dtype == torch.float32


def test_build_initial_state_has_nonzero_mass() -> None:
    spawn = _spawn(n=4, patch=12)
    state = build_initial_state(spawn, grid=256, device="cpu")
    assert float(state.sum()) > 0.0


def test_build_initial_state_mass_bounded() -> None:
    # Each patch is uniform [0, 1) so total mass <= n * patch^2
    spawn = _spawn(n=4, patch=12)
    state = build_initial_state(spawn, grid=256, device="cpu")
    max_possible = 4 * 12 * 12
    assert float(state.sum()) <= max_possible


def test_build_initial_state_reproducible() -> None:
    spawn = _spawn(n=3, seed=42)
    s1 = build_initial_state(spawn, grid=256, device="cpu")
    s2 = build_initial_state(spawn, grid=256, device="cpu")
    assert (s1 == s2).all()


# === EcosystemResult.to_summary_dict ===


def test_to_summary_dict_is_json_serializable() -> None:
    from biota.ecosystem.result import EcosystemResult

    spawn = _spawn(n=2)
    config = EcosystemConfig(
        source_run_id="test-run",
        source_coords=(5, 8, 3),
        grid=128,
        steps=10,
        snapshot_every=5,
        spawn=spawn,
    )
    measures = EcosystemMeasures(
        initial_mass=10.0,
        final_mass=9.8,
        mass_history=[10.0, 9.9, 9.8],
        peak_mass=10.0,
        min_mass=9.8,
        mass_turnover=0.01,
        snapshot_steps=[5, 10],
    )
    result = EcosystemResult(
        config=config,
        run_id="eco-test",
        run_dir="/tmp/eco-test",
        measures=measures,
        elapsed_seconds=1.0,
    )
    d = result.to_summary_dict()
    json_str = json.dumps(d)
    assert "test-run" in json_str
    assert "eco-test" in json_str


# === run_ecosystem integration ===


def test_run_ecosystem_creates_output_files() -> None:
    from biota.ecosystem.run import run_ecosystem

    spawn = _spawn(n=2, min_dist=30, patch=8)
    config = EcosystemConfig(
        source_run_id="test-run",
        source_coords=(5, 8, 3),
        grid=64,
        steps=5,
        snapshot_every=5,
        spawn=spawn,
        device="cpu",
    )
    creature = _synthetic_creature()

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_ecosystem(config, creature, output_root=tmpdir)
        run_dir = Path(result.run_dir)

        assert (run_dir / "summary.json").exists()
        assert (run_dir / "config.json").exists()
        assert (run_dir / "trajectory.npy").exists()
        assert (run_dir / "frames").is_dir()


def test_run_ecosystem_trajectory_shape() -> None:
    from biota.ecosystem.run import run_ecosystem

    spawn = _spawn(n=2, min_dist=20, patch=6)
    config = EcosystemConfig(
        source_run_id="test-run",
        source_coords=(1, 2, 3),
        grid=64,
        steps=10,
        snapshot_every=5,
        spawn=spawn,
        device="cpu",
    )
    creature = _synthetic_creature()

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_ecosystem(config, creature, output_root=tmpdir)
        traj = np.load(Path(result.run_dir) / "trajectory.npy")
        # 2 snapshots: step 5 and step 10
        assert traj.shape == (2, 64, 64)
        assert traj.dtype == np.float32


def test_run_ecosystem_summary_json_valid() -> None:
    from biota.ecosystem.run import run_ecosystem

    spawn = _spawn(n=2, min_dist=20, patch=6)
    config = EcosystemConfig(
        source_run_id="my-run",
        source_coords=(0, 1, 2),
        grid=64,
        steps=5,
        snapshot_every=5,
        spawn=spawn,
        device="cpu",
    )
    creature = _synthetic_creature()

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_ecosystem(config, creature, output_root=tmpdir)
        summary = json.loads((Path(result.run_dir) / "summary.json").read_text())
        assert summary["source_run_id"] == "my-run"
        assert summary["source_coords"] == [0, 1, 2]
        assert summary["measures"]["initial_mass"] >= 0.0
        assert len(summary["measures"]["snapshot_steps"]) == 1


def test_run_ecosystem_measures_plausible() -> None:
    from biota.ecosystem.run import run_ecosystem

    spawn = _spawn(n=2, min_dist=20, patch=8)
    config = EcosystemConfig(
        source_run_id="test-run",
        source_coords=(3, 3, 3),
        grid=64,
        steps=10,
        snapshot_every=10,
        spawn=spawn,
        device="cpu",
    )
    creature = _synthetic_creature()

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_ecosystem(config, creature, output_root=tmpdir)
        m = result.measures
        assert m.initial_mass > 0.0
        assert m.peak_mass >= m.initial_mass or m.peak_mass >= m.final_mass
        assert m.min_mass <= m.initial_mass
        assert m.mass_turnover >= 0.0
        assert len(m.mass_history) == config.steps + 1
