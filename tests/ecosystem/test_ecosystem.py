"""Tests for the ecosystem simulation module.

Covers Poisson disk sampling, spawn state construction, summary_dict
serialization, and the full run_ecosystem pipeline end-to-end via a real
on-disk archive pickle.

run_ecosystem now loads its creatures from disk (per CreatureSource entries
in the config) rather than taking a pre-loaded RolloutResult. To exercise it
the tests build a tiny real Archive with one cell, pickle it, and point the
config at that directory.
"""

import json
import math
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from biota.ecosystem.config import CreatureSource, EcosystemConfig, SpawnConfig
from biota.ecosystem.result import EcosystemMeasures, EcosystemResult
from biota.ecosystem.spawn import (
    _jittered_grid_fallback,  # type: ignore[reportPrivateUsage]
    _poisson_disk_sample,  # type: ignore[reportPrivateUsage]
    build_initial_state,
    compute_spawn_positions,
)
from biota.search.archive import Archive
from biota.search.result import RolloutResult

# === helpers ===


def _spawn(min_dist: int = 60, patch: int = 16, seed: int = 0) -> SpawnConfig:
    return SpawnConfig(min_dist=min_dist, patch=patch, seed=seed)


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


def _write_archive_with_creature(
    root: Path,
    run_id: str,
    coords: tuple[int, int, int],
    creature: RolloutResult,
) -> Path:
    """Write a pickled Archive containing exactly one creature at coords.

    The bin counts are chosen so that the creature's descriptors bin cleanly
    into the requested coords. For the synthetic creature descriptors
    (0.3, 0.5, 0.6) this just means the default (32, 32, 16) bins at
    (9, 16, 9). Tests that need other coords should pass descriptors that
    bin into those coords or use a different archive setup.
    """
    archive = Archive()
    # Force the bin math by going directly at the dict; Archive's try_insert
    # would bin on descriptors and we'd be tied to the default descriptors.
    archive._cells[coords] = creature  # pyright: ignore[reportPrivateUsage]
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "archive.pkl", "wb") as f:
        pickle.dump(archive, f)
    return run_dir


def _single_source_config(
    archive_dir: Path,
    run_id: str = "test-run",
    coords: tuple[int, int, int] = (5, 8, 3),
    n: int = 2,
    grid_h: int = 64,
    grid_w: int = 64,
    steps: int = 5,
    snapshot_every: int = 5,
    output_format: str = "gif",
) -> EcosystemConfig:
    return EcosystemConfig(
        name="test-experiment",
        sources=(CreatureSource(archive_dir=str(archive_dir), run_id=run_id, coords=coords, n=n),),
        grid_h=grid_h,
        grid_w=grid_w,
        steps=steps,
        snapshot_every=snapshot_every,
        spawn=SpawnConfig(min_dist=20, patch=6, seed=0),
        device="cpu",
        border="wall",
        output_format=output_format,
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
    pts = compute_spawn_positions(_spawn(min_dist=60), n=4, grid_h=512, grid_w=512)
    assert len(pts) == 4


def test_compute_spawn_positions_reproducible() -> None:
    spawn = _spawn(min_dist=60, seed=99)
    pts1 = compute_spawn_positions(spawn, n=4, grid_h=512, grid_w=512)
    pts2 = compute_spawn_positions(spawn, n=4, grid_h=512, grid_w=512)
    assert pts1 == pts2


def test_compute_spawn_positions_seed_matters() -> None:
    pts1 = compute_spawn_positions(_spawn(seed=0), n=4, grid_h=512, grid_w=512)
    pts2 = compute_spawn_positions(_spawn(seed=1), n=4, grid_h=512, grid_w=512)
    assert pts1 != pts2


# === build_initial_state ===


def test_build_initial_state_shape() -> None:
    import torch

    state = build_initial_state(_spawn(patch=12), n=4, grid_h=256, grid_w=256, device="cpu")
    assert state.shape == (256, 256, 1)
    assert state.dtype == torch.float32


def test_build_initial_state_has_nonzero_mass() -> None:
    state = build_initial_state(_spawn(patch=12), n=4, grid_h=256, grid_w=256, device="cpu")
    assert float(state.sum()) > 0.0


def test_build_initial_state_mass_bounded() -> None:
    # Each patch is uniform [0, 1) so total mass <= n * patch^2
    state = build_initial_state(_spawn(patch=12), n=4, grid_h=256, grid_w=256, device="cpu")
    max_possible = 4 * 12 * 12
    assert float(state.sum()) <= max_possible


def test_build_initial_state_reproducible() -> None:
    spawn = _spawn(seed=42)
    s1 = build_initial_state(spawn, n=3, grid_h=256, grid_w=256, device="cpu")
    s2 = build_initial_state(spawn, n=3, grid_h=256, grid_w=256, device="cpu")
    assert (s1 == s2).all()


# === build_initial_state_multi_species ===


def test_build_initial_state_multi_species_shapes() -> None:
    import torch

    from biota.ecosystem.spawn import build_initial_state_multi_species

    mass, weights = build_initial_state_multi_species(
        _spawn(min_dist=20, patch=8), counts=[3, 2], grid_h=128, grid_w=128, device="cpu"
    )
    assert mass.shape == (128, 128, 1)
    assert weights.shape == (128, 128, 2)
    assert mass.dtype == torch.float32
    assert weights.dtype == torch.float32


def test_build_initial_state_multi_species_weights_one_hot() -> None:
    """Each cell with mass should be owned by exactly one species at init."""
    from biota.ecosystem.spawn import build_initial_state_multi_species

    mass, weights = build_initial_state_multi_species(
        _spawn(min_dist=20, patch=8, seed=11),
        counts=[2, 2],
        grid_h=128,
        grid_w=128,
        device="cpu",
    )
    mass_present = mass[:, :, 0] > 0
    if mass_present.any():
        # At every mass-bearing cell, weights should sum to 1 and be one-hot.
        sums = weights.sum(dim=-1)[mass_present]
        assert (sums - 1.0).abs().max().item() < 1e-6
        # One-hot means max value per cell is 1.0
        max_per_cell = weights.max(dim=-1).values[mass_present]
        assert (max_per_cell - 1.0).abs().max().item() < 1e-6


def test_build_initial_state_multi_species_rejects_empty_counts() -> None:
    from biota.ecosystem.spawn import build_initial_state_multi_species

    try:
        build_initial_state_multi_species(_spawn(), counts=[], grid_h=64, grid_w=64, device="cpu")
    except ValueError as exc:
        assert "at least one" in str(exc)
        return
    raise AssertionError("empty counts did not raise")


def test_build_initial_state_multi_species_rejects_zero_count() -> None:
    from biota.ecosystem.spawn import build_initial_state_multi_species

    try:
        build_initial_state_multi_species(
            _spawn(), counts=[2, 0, 1], grid_h=64, grid_w=64, device="cpu"
        )
    except ValueError as exc:
        assert "positive" in str(exc)
        return
    raise AssertionError("zero count did not raise")


def test_build_initial_state_multi_species_per_species_patches() -> None:
    """Per-species patches change initial mass per species proportional to area.

    Two species, same count, but species 1 uses a 4x larger patch (16 vs 8).
    Initial mass for species 1's region should be roughly (16/8)^2 = 4x larger
    than species 0's, modulo random patch fill (uniform [0,1) per cell -> mean 0.5).
    """
    from biota.ecosystem.spawn import build_initial_state_multi_species

    mass, weights = build_initial_state_multi_species(
        _spawn(min_dist=30, patch=8, seed=42),
        counts=[2, 2],
        grid_h=128,
        grid_w=128,
        device="cpu",
        patches=[8, 16],
    )
    # Mass owned by species 0 vs species 1.
    species0_mass = float((mass[:, :, 0] * weights[:, :, 0]).sum())
    species1_mass = float((mass[:, :, 0] * weights[:, :, 1]).sum())
    ratio = species1_mass / species0_mass
    # Expected ratio is 4 (area scales as patch^2). Allow wide tolerance for
    # random fill noise: any value between 2 and 6 confirms the override took.
    assert 2.0 < ratio < 6.0, (
        f"expected species1/species0 mass ratio ~4 (patch area scaling), got {ratio:.2f}"
    )
    # Without the override, the ratio should be ~1 since both species would
    # use the spawn-level patch=8.
    mass_default, weights_default = build_initial_state_multi_species(
        _spawn(min_dist=30, patch=8, seed=42),
        counts=[2, 2],
        grid_h=128,
        grid_w=128,
        device="cpu",
    )
    s0_default = float((mass_default[:, :, 0] * weights_default[:, :, 0]).sum())
    s1_default = float((mass_default[:, :, 0] * weights_default[:, :, 1]).sum())
    default_ratio = s1_default / s0_default
    assert 0.5 < default_ratio < 2.0, (
        f"without override, ratio should be ~1, got {default_ratio:.2f}"
    )
    # Sanity: shapes unchanged
    assert mass.shape == mass_default.shape
    assert weights.shape == weights_default.shape


def test_build_initial_state_multi_species_rejects_patches_length_mismatch() -> None:
    from biota.ecosystem.spawn import build_initial_state_multi_species

    try:
        build_initial_state_multi_species(
            _spawn(),
            counts=[2, 2],
            grid_h=64,
            grid_w=64,
            device="cpu",
            patches=[8, 16, 32],  # length 3 vs counts length 2
        )
    except ValueError as exc:
        assert "length" in str(exc)
        return
    raise AssertionError("patches length mismatch did not raise")


def test_run_ecosystem_homogeneous_respects_source_patch() -> None:
    """A homogeneous run with source.patch override produces different initial
    mass than the same run without override."""
    import json
    import tempfile
    from pathlib import Path

    from biota.ecosystem.run import run_ecosystem

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        archive_dir = tmp / "archive"
        _write_archive_with_creature(archive_dir, "test-run", (5, 8, 3), _synthetic_creature())

        # Baseline: no per-source patch
        baseline_cfg = _single_source_config(
            archive_dir, run_id="test-run", coords=(5, 8, 3), grid_h=128, grid_w=128
        )
        baseline_result = run_ecosystem(baseline_cfg, output_root=tmp / "eco-baseline")
        baseline_mass = baseline_result.measures.initial_mass

        # Override: same config but source has a doubled patch
        override_cfg = EcosystemConfig(
            name=baseline_cfg.name,
            sources=(
                CreatureSource(
                    archive_dir=baseline_cfg.sources[0].archive_dir,
                    run_id=baseline_cfg.sources[0].run_id,
                    coords=baseline_cfg.sources[0].coords,
                    n=baseline_cfg.sources[0].n,
                    patch=baseline_cfg.spawn.patch * 2,
                ),
            ),
            grid_h=baseline_cfg.grid_h,
            grid_w=baseline_cfg.grid_w,
            steps=baseline_cfg.steps,
            snapshot_every=baseline_cfg.snapshot_every,
            spawn=baseline_cfg.spawn,
            device=baseline_cfg.device,
            border=baseline_cfg.border,
            output_format=baseline_cfg.output_format,
        )
        override_result = run_ecosystem(override_cfg, output_root=tmp / "eco-override")
        override_mass = override_result.measures.initial_mass

        # Doubled patch ~> 4x area -> ~4x mass (uniform random fill, mean 0.5)
        ratio = override_mass / baseline_mass
        assert 2.5 < ratio < 5.5, (
            f"expected ~4x initial mass with doubled patch, got ratio {ratio:.2f}"
        )

        # config.json and summary.json should record the resolved patch
        override_summary = json.loads((Path(override_result.run_dir) / "summary.json").read_text())
        assert override_summary["sources"][0]["patch"] == baseline_cfg.spawn.patch * 2
        baseline_summary = json.loads((Path(baseline_result.run_dir) / "summary.json").read_text())
        assert baseline_summary["sources"][0]["patch"] == baseline_cfg.spawn.patch


def test_run_ecosystem_accepts_preloaded_creatures() -> None:
    """run_ecosystem(creatures=...) bypasses disk archive load.

    This is the path the cluster Ray dispatcher takes: driver loads creatures
    locally, passes them via Ray task payload so worker nodes don't need
    filesystem access to the archive directory. The result should be
    bit-identical to the disk-load path given the same creatures.
    """
    import tempfile
    from pathlib import Path

    from biota.ecosystem.run import load_creature, run_ecosystem

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        archive_dir = tmp / "archive"
        _write_archive_with_creature(archive_dir, "test-run", (5, 8, 3), _synthetic_creature())

        cfg = _single_source_config(
            archive_dir, run_id="test-run", coords=(5, 8, 3), grid_h=64, grid_w=64
        )

        # Load via disk (standard sequential path)
        disk_result = run_ecosystem(cfg, output_root=tmp / "disk")

        # Load via pre-loaded creatures
        creature = load_creature(cfg.sources[0])
        preloaded_result = run_ecosystem(cfg, output_root=tmp / "preloaded", creatures=[creature])

        # Same simulation = same initial mass and same trajectory
        assert disk_result.measures.initial_mass == preloaded_result.measures.initial_mass
        assert disk_result.measures.final_mass == preloaded_result.measures.final_mass
        assert disk_result.measures.peak_mass == preloaded_result.measures.peak_mass


def test_run_ecosystem_rejects_creatures_length_mismatch() -> None:
    """creatures must match config.sources length."""
    import tempfile
    from pathlib import Path

    from biota.ecosystem.run import load_creature, run_ecosystem

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        archive_dir = tmp / "archive"
        _write_archive_with_creature(archive_dir, "test-run", (5, 8, 3), _synthetic_creature())

        cfg = _single_source_config(
            archive_dir, run_id="test-run", coords=(5, 8, 3), grid_h=64, grid_w=64
        )

        creature = load_creature(cfg.sources[0])
        # Two creatures for one source: mismatch.
        with pytest.raises(ValueError, match=r"creatures length .* does not match"):
            run_ecosystem(cfg, output_root=tmp / "bad", creatures=[creature, creature])


def test_compute_ecosystem_returns_artifacts_without_io() -> None:
    """compute_ecosystem returns (result, artifacts) and writes nothing to disk.

    This is the worker-side entry point for the Ray dispatcher: workers run
    compute_ecosystem and ship the artifacts dict back to the driver, which
    materializes locally. The contract is "pure compute, no I/O" so the
    worker filesystem never gets touched.
    """
    import tempfile
    from pathlib import Path

    from biota.ecosystem.run import compute_ecosystem

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        archive_dir = tmp / "archive"
        _write_archive_with_creature(archive_dir, "test-run", (5, 8, 3), _synthetic_creature())

        cfg = _single_source_config(
            archive_dir, run_id="test-run", coords=(5, 8, 3), grid_h=64, grid_w=64
        )

        # Use an output_root that does not exist; if compute_ecosystem writes
        # anything the directory creation would itself be I/O we don't want.
        nonexistent_out = tmp / "should-not-exist"
        result, artifacts = compute_ecosystem(cfg, output_root=nonexistent_out)

        # Nothing materialized
        assert not nonexistent_out.exists(), "compute_ecosystem must not touch disk"

        # Required artifacts present
        assert "config.json" in artifacts
        assert "summary.json" in artifacts
        assert "trajectory.npy" in artifacts
        # GIF is the default output_format for _single_source_config
        assert "ecosystem.gif" in artifacts

        # Bytes are non-empty and parseable where applicable
        config_dict = json.loads(artifacts["config.json"].decode("utf-8"))
        assert config_dict["name"] == cfg.name
        summary_dict = json.loads(artifacts["summary.json"].decode("utf-8"))
        assert summary_dict["run_id"] == result.run_id
        assert summary_dict["measures"]["initial_mass"] == result.measures.initial_mass


def test_compute_ecosystem_then_materialize_matches_run_ecosystem() -> None:
    """Worker (compute) + driver (materialize) produces the same files
    on disk as the sequential run_ecosystem path. This is the round-trip
    invariant the cluster dispatcher relies on."""
    import tempfile
    from pathlib import Path

    from biota.ecosystem.run import compute_ecosystem, materialize_outputs, run_ecosystem

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        archive_dir = tmp / "archive"
        _write_archive_with_creature(archive_dir, "test-run", (5, 8, 3), _synthetic_creature())

        cfg = _single_source_config(
            archive_dir, run_id="test-run", coords=(5, 8, 3), grid_h=64, grid_w=64
        )

        # Path A: compute then materialize (the cluster dispatcher path)
        result_a, artifacts = compute_ecosystem(cfg, output_root=tmp / "compute-then-materialize")
        run_dir_a = Path(result_a.run_dir)
        run_dir_a.mkdir(parents=True, exist_ok=True)
        materialize_outputs(run_dir_a, artifacts)

        # Path B: run_ecosystem (the sequential CLI path)
        result_b = run_ecosystem(cfg, output_root=tmp / "run-ecosystem")
        run_dir_b = Path(result_b.run_dir)

        # Both produce the same set of artifact filenames on disk.
        files_a = {p.relative_to(run_dir_a).as_posix() for p in run_dir_a.rglob("*") if p.is_file()}
        files_b = {p.relative_to(run_dir_b).as_posix() for p in run_dir_b.rglob("*") if p.is_file()}
        assert files_a == files_b, (
            f"file sets differ: only-a={files_a - files_b} only-b={files_b - files_a}"
        )

        # config.json bytes are identical (same config -> same serialization).
        assert (run_dir_a / "config.json").read_bytes() == (run_dir_b / "config.json").read_bytes()


# === EcosystemResult.to_summary_dict ===


def test_to_summary_dict_is_json_serializable() -> None:
    config = EcosystemConfig(
        name="smoke",
        sources=(CreatureSource(archive_dir="archive", run_id="test-run", coords=(5, 8, 3), n=2),),
        grid_h=128,
        grid_w=128,
        steps=10,
        snapshot_every=5,
        spawn=SpawnConfig(min_dist=40, patch=16, seed=0),
        device="cpu",
        border="wall",
        output_format="gif",
    )
    measures = EcosystemMeasures(
        initial_mass=10.0,
        final_mass=9.8,
        mass_history=[10.0, 9.9, 9.8],
        peak_mass=10.0,
        min_mass=9.8,
        mass_turnover=0.01,
        snapshot_steps=[5, 10],
        species_mass_history=[[10.0, 9.9, 9.8]],
        species_territory_history=[[100.0, 99.0, 98.0]],
        interaction_coefficients=[],
        outcome_label="",
    )
    result = EcosystemResult(
        config=config,
        run_id="eco-test",
        run_dir="/tmp/eco-test",
        measures=measures,
        elapsed_seconds=1.0,
    )
    d = result.to_summary_dict()
    # Round-trip through JSON to verify nothing non-serializable slipped in.
    js = json.dumps(d)
    roundtrip = json.loads(js)
    assert roundtrip["run_id"] == "eco-test"
    assert roundtrip["name"] == "smoke"
    assert roundtrip["mode"] == "homogeneous"
    assert roundtrip["sources"] == [
        {
            "archive_dir": "archive",
            "run": "test-run",
            "cell": [5, 8, 3],
            "n": 2,
            "patch": 16,  # falls back to spawn.patch since CreatureSource.patch=None
        }
    ]


def test_to_summary_dict_marks_heterogeneous_mode() -> None:
    config = EcosystemConfig(
        name="het",
        sources=(
            CreatureSource(archive_dir="archive", run_id="run-a", coords=(1, 2, 3), n=4),
            CreatureSource(archive_dir="archive", run_id="run-b", coords=(4, 5, 6), n=4),
        ),
        grid_h=128,
        grid_w=128,
        steps=10,
        snapshot_every=5,
        spawn=SpawnConfig(min_dist=40, patch=16, seed=0),
        device="cpu",
        border="torus",
        output_format="gif",
    )
    measures = EcosystemMeasures(
        initial_mass=1.0,
        final_mass=1.0,
        mass_history=[1.0],
        peak_mass=1.0,
        min_mass=1.0,
        mass_turnover=0.0,
        snapshot_steps=[],
        species_mass_history=[[1.0]],
        species_territory_history=[[50.0]],
        interaction_coefficients=[],
        outcome_label="",
    )
    result = EcosystemResult(
        config=config, run_id="het-test", run_dir="/tmp", measures=measures, elapsed_seconds=0.0
    )
    d = result.to_summary_dict()
    assert d["mode"] == "heterogeneous"
    assert len(d["sources"]) == 2  # pyright: ignore[reportArgumentType]


# === run_ecosystem integration (homogeneous path) ===


def test_run_ecosystem_creates_output_files_gif_mode() -> None:
    from biota.ecosystem.run import run_ecosystem

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        archive_dir = tmp / "archive"
        output_dir = tmp / "eco"
        _write_archive_with_creature(archive_dir, "test-run", (5, 8, 3), _synthetic_creature())

        config = _single_source_config(archive_dir, output_format="gif")
        result = run_ecosystem(config, output_root=output_dir)

        run_dir = Path(result.run_dir)
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "config.json").exists()
        assert (run_dir / "trajectory.npy").exists()
        assert (run_dir / "ecosystem.gif").exists()
        assert not (run_dir / "frames").exists()


def test_run_ecosystem_creates_output_files_frames_mode() -> None:
    from biota.ecosystem.run import run_ecosystem

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        archive_dir = tmp / "archive"
        output_dir = tmp / "eco"
        _write_archive_with_creature(archive_dir, "test-run-2", (5, 8, 3), _synthetic_creature())

        config = _single_source_config(archive_dir, run_id="test-run-2", output_format="frames")
        result = run_ecosystem(config, output_root=output_dir)

        run_dir = Path(result.run_dir)
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "frames").is_dir()
        assert not (run_dir / "ecosystem.gif").exists()


def test_run_ecosystem_trajectory_shape() -> None:
    from biota.ecosystem.run import run_ecosystem

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        archive_dir = tmp / "archive"
        output_dir = tmp / "eco"
        _write_archive_with_creature(archive_dir, "test-run", (1, 2, 3), _synthetic_creature())

        config = _single_source_config(archive_dir, coords=(1, 2, 3), steps=10, snapshot_every=5)
        result = run_ecosystem(config, output_root=output_dir)

        traj = np.load(Path(result.run_dir) / "trajectory.npy")
        assert traj.shape == (2, 64, 64)  # snapshots at step 5 and 10
        assert traj.dtype == np.float32


def test_run_ecosystem_summary_json_valid() -> None:
    from biota.ecosystem.run import run_ecosystem

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        archive_dir = tmp / "archive"
        output_dir = tmp / "eco"
        _write_archive_with_creature(archive_dir, "my-run", (0, 1, 2), _synthetic_creature())

        config = _single_source_config(
            archive_dir, run_id="my-run", coords=(0, 1, 2), steps=5, snapshot_every=5
        )
        result = run_ecosystem(config, output_root=output_dir)

        summary = json.loads((Path(result.run_dir) / "summary.json").read_text())
        assert summary["name"] == "test-experiment"
        assert summary["mode"] == "homogeneous"
        assert summary["sources"][0]["run"] == "my-run"
        assert summary["sources"][0]["cell"] == [0, 1, 2]
        assert summary["measures"]["initial_mass"] >= 0.0
        assert len(summary["measures"]["snapshot_steps"]) == 1
        # Homogeneous runs: interaction fields are present but empty.
        assert summary["measures"]["interaction_coefficients"] == []
        assert summary["measures"]["outcome_label"] == ""


def test_run_ecosystem_measures_plausible() -> None:
    from biota.ecosystem.run import run_ecosystem

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        archive_dir = tmp / "archive"
        output_dir = tmp / "eco"
        _write_archive_with_creature(archive_dir, "test-run", (3, 3, 3), _synthetic_creature())

        config = _single_source_config(
            archive_dir,
            coords=(3, 3, 3),
            steps=10,
            snapshot_every=10,
            n=2,
        )
        result = run_ecosystem(config, output_root=output_dir)

        m = result.measures
        assert m.initial_mass > 0.0
        assert m.peak_mass >= m.initial_mass or m.peak_mass >= m.final_mass
        assert m.min_mass <= m.initial_mass
        assert m.mass_turnover >= 0.0
        assert len(m.mass_history) == config.steps + 1


def test_run_ecosystem_heterogeneous_succeeds_end_to_end() -> None:
    """Heterogeneous configs run via the species-indexed path, write outputs."""
    from biota.ecosystem.run import run_ecosystem

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        archive_dir = tmp / "archive"
        # Two distinct creatures in two separate archive runs.
        _write_archive_with_creature(archive_dir, "run-a", (1, 1, 1), _synthetic_creature())
        _write_archive_with_creature(archive_dir, "run-b", (2, 2, 2), _synthetic_creature())

        config = EcosystemConfig(
            name="hetero-smoke",
            sources=(
                CreatureSource(archive_dir=str(archive_dir), run_id="run-a", coords=(1, 1, 1), n=2),
                CreatureSource(archive_dir=str(archive_dir), run_id="run-b", coords=(2, 2, 2), n=2),
            ),
            grid_h=64,
            grid_w=64,
            steps=10,
            snapshot_every=5,
            spawn=SpawnConfig(min_dist=15, patch=6, seed=0),
            device="cpu",
            border="wall",
            output_format="gif",
        )
        result = run_ecosystem(config, output_root=tmp / "eco")
        run_dir = Path(result.run_dir)
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "ecosystem.gif").exists()
        assert (run_dir / "trajectory.npy").exists()

        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["mode"] == "heterogeneous"
        assert len(summary["sources"]) == 2
        assert result.measures.initial_mass > 0.0
        # v3.0.0: interaction coefficients must be a 2x2 matrix of floats.
        ic = summary["measures"]["interaction_coefficients"]
        assert len(ic) == 2
        assert all(len(row) == 2 for row in ic)
        assert all(isinstance(v, float) for row in ic for v in row)
        # Outcome label must be one of the four recognised categories.
        label = summary["measures"]["outcome_label"]
        assert label in {"merger", "coexistence", "exclusion", "fragmentation"}


def test_run_ecosystem_heterogeneous_interaction_fields() -> None:
    """Heterogeneous runs populate interaction_coefficients and outcome_label
    on the result object and in summary.json. This test is the load-bearing
    integration check: if growth field capture, interaction measurement, or
    outcome classification break silently, the measures object will have
    wrong shapes or an empty label.
    """
    from biota.ecosystem.run import run_ecosystem

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        archive_dir = tmp / "archive"
        _write_archive_with_creature(archive_dir, "run-a", (0, 0, 0), _synthetic_creature())
        _write_archive_with_creature(archive_dir, "run-b", (1, 1, 1), _synthetic_creature())

        config = EcosystemConfig(
            name="interaction-test",
            sources=(
                CreatureSource(archive_dir=str(archive_dir), run_id="run-a", coords=(0, 0, 0), n=1),
                CreatureSource(archive_dir=str(archive_dir), run_id="run-b", coords=(1, 1, 1), n=1),
            ),
            grid_h=64,
            grid_w=64,
            # Enough steps and snapshots for interaction measurement to have data.
            steps=20,
            snapshot_every=10,
            spawn=SpawnConfig(min_dist=20, patch=8, seed=0),
            device="cpu",
            border="torus",
            output_format="gif",
        )
        result = run_ecosystem(config, output_root=tmp / "eco")

        m = result.measures
        # Interaction coefficients: S x S matrix (S=2 here).
        assert len(m.interaction_coefficients) == 2, "expected 2x2 coefficient matrix"
        assert all(len(row) == 2 for row in m.interaction_coefficients)
        # Each entry is a float (may be NaN if species never co-occurred, but
        # must not be None or the wrong type).
        for row in m.interaction_coefficients:
            for v in row:
                assert isinstance(v, float), f"non-float coefficient: {v!r}"
        # Outcome label is set and is one of the four valid values.
        assert m.outcome_label in {"merger", "coexistence", "exclusion", "fragmentation"}, (
            f"unexpected outcome label: {m.outcome_label!r}"
        )
        # Confirm the fields round-trip through summary.json.
        summary = json.loads((tmp / "eco" / result.run_id / "summary.json").read_text())
        assert summary["measures"]["interaction_coefficients"] == m.interaction_coefficients
        assert summary["measures"]["outcome_label"] == m.outcome_label


def test_run_ecosystem_heterogeneous_kernel_count_mismatch_raises() -> None:
    """All heterogeneous sources must have the same kernel count."""
    from biota.ecosystem.run import run_ecosystem

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        archive_dir = tmp / "archive"

        c1 = _synthetic_creature()  # k=3
        # Build a creature with a different kernel count.
        c2_params = dict(c1.params)
        k2 = 5
        c2_params["r"] = [0.5] * k2
        c2_params["m"] = [0.15] * k2
        c2_params["s"] = [0.015] * k2
        c2_params["h"] = [0.5] * k2
        c2_params["a"] = [[0.5, 0.5, 0.5]] * k2
        c2_params["b"] = [[0.5, 0.5, 0.5]] * k2
        c2_params["w"] = [[0.5, 0.5, 0.5]] * k2
        c2 = RolloutResult(
            params=c2_params,  # type: ignore[arg-type]
            seed=1,
            descriptors=(0.4, 0.5, 0.6),
            quality=0.7,
            rejection_reason=None,
            thumbnail=np.zeros((16, 32, 32), dtype=np.uint8),
            parent_cell=None,
            created_at=0.0,
            compute_seconds=1.0,
        )
        _write_archive_with_creature(archive_dir, "run-a", (1, 1, 1), c1)
        _write_archive_with_creature(archive_dir, "run-b", (2, 2, 2), c2)

        config = EcosystemConfig(
            name="kernel-mismatch",
            sources=(
                CreatureSource(archive_dir=str(archive_dir), run_id="run-a", coords=(1, 1, 1), n=2),
                CreatureSource(archive_dir=str(archive_dir), run_id="run-b", coords=(2, 2, 2), n=2),
            ),
            grid_h=64,
            grid_w=64,
            steps=5,
            snapshot_every=5,
            spawn=SpawnConfig(min_dist=15, patch=6, seed=0),
            device="cpu",
            border="wall",
            output_format="gif",
        )
        try:
            run_ecosystem(config, output_root=tmp / "eco")
        except ValueError as exc:
            assert "kernel count" in str(exc)
            return
    raise AssertionError("kernel count mismatch did not raise")


def test_run_ecosystem_missing_archive_raises() -> None:
    from biota.ecosystem.run import run_ecosystem

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config = _single_source_config(
            archive_dir=tmp / "does-not-exist", run_id="missing", coords=(1, 1, 1)
        )
        try:
            run_ecosystem(config, output_root=tmp / "eco")
        except FileNotFoundError as exc:
            assert "missing" in str(exc)
            return
    raise AssertionError("run_ecosystem did not raise for missing archive")


def test_run_ecosystem_missing_cell_raises() -> None:
    from biota.ecosystem.run import run_ecosystem

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        archive_dir = tmp / "archive"
        _write_archive_with_creature(archive_dir, "has-one-cell", (0, 0, 0), _synthetic_creature())

        config = _single_source_config(archive_dir, run_id="has-one-cell", coords=(99, 99, 99))
        try:
            run_ecosystem(config, output_root=tmp / "eco")
        except KeyError as exc:
            assert "99" in str(exc)
            return
    raise AssertionError("run_ecosystem did not raise for missing cell")
