"""Tests for biota.ecosystem.dispatch.

Two tiers:

1. Validation unit tests don't touch Ray and verify the public API rejects
   bad inputs cleanly with informative ValueError messages.

2. A real local-Ray smoke test (run_experiments_parallel_local_ray_smoke)
   actually starts Ray, submits two tiny experiments, and asserts both
   completed with the expected output structure. Marked with the
   `smoke_ray` marker so the default pytest run skips it; the smoke-ray
   targets in justfile pick it up.
"""

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from biota.ecosystem.config import CreatureSource, EcosystemConfig, SpawnConfig
from biota.ecosystem.dispatch import detect_gpu_count, run_experiments_parallel
from biota.search.archive import Archive
from biota.search.result import RolloutResult


def _synthetic_creature() -> RolloutResult:
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
        creature_id="",
        parent_id=None,
        created_at=0.0,
        compute_seconds=1.0,
    )


def _write_archive(root: Path, run_id: str, coords: tuple[int, int, int] | None = None) -> None:
    arc = Archive(n_centroids=4, similarity_epsilon=0.0)
    centroids = np.array(
        [[0.3, 0.5, 0.6], [5.0, 5.0, 5.0], [8.0, 2.0, 8.0], [2.0, 8.0, 2.0]],
        dtype=np.float64,
    )
    arc.attach_centroids(centroids)
    from dataclasses import replace as _replace

    creature = _replace(_synthetic_creature(), creature_id=f"{run_id}-0")
    arc.try_insert(creature)
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "archive.pkl", "wb") as f:
        pickle.dump(arc, f)


def _experiment(name: str, archive_dir: Path, run_id: str) -> EcosystemConfig:
    return EcosystemConfig(
        name=name,
        sources=(
            CreatureSource(
                archive_dir=str(archive_dir),
                run_id=run_id,
                coords=None,
                n=2,
                creature_id=f"{run_id}-0",
            ),
        ),
        grid_h=64,
        grid_w=64,
        steps=5,
        snapshot_every=5,
        spawn=SpawnConfig(min_dist=20, patch=8, seed=0),
        device="cpu",
        border="wall",
        output_format="gif",
    )


# === validation unit tests (no Ray) ===


def test_run_experiments_parallel_requires_ray_flag() -> None:
    """Calling without local_ray or ray_address must raise; sequential is the
    caller's job, not this module's."""
    with pytest.raises(ValueError, match="requires either local_ray=True or ray_address"):
        run_experiments_parallel((), Path("/tmp"), workers=1, gpu_fraction=1.0, local_ray=False)


def test_run_experiments_parallel_rejects_zero_workers() -> None:
    with pytest.raises(ValueError, match="workers must be >= 1"):
        run_experiments_parallel((), Path("/tmp"), workers=0, gpu_fraction=1.0, local_ray=True)


def test_run_experiments_parallel_rejects_negative_gpu_fraction() -> None:
    with pytest.raises(ValueError, match="gpu_fraction must be >= 0"):
        run_experiments_parallel((), Path("/tmp"), workers=1, gpu_fraction=-0.5, local_ray=True)


def test_detect_gpu_count_returns_nonnegative_int() -> None:
    """Detection should never crash; just returns 0 if nothing CUDA-capable."""
    n = detect_gpu_count()
    assert isinstance(n, int)
    assert n >= 0


# === local-Ray smoke test ===


@pytest.mark.smoke_ray
def test_run_experiments_parallel_local_ray_smoke() -> None:
    """Real local-Ray run with two experiments. Verifies end-to-end:
    Ray init, parallel task submission, result collection, output files
    written for each experiment, Ray teardown.

    Skipped by default (marked smoke_ray); justfile target picks it up.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        archive_dir = tmp / "archive"
        _write_archive(archive_dir, "run-a")
        _write_archive(archive_dir, "run-b")

        experiments = (
            _experiment("alpha", archive_dir, "run-a"),
            _experiment("beta", archive_dir, "run-b"),
        )
        output_root = tmp / "ecosystem"

        # CPU-only smoke: gpu_fraction=0 means no GPU resource reservation,
        # so Ray schedules tasks anywhere. Works on CI machines without CUDA
        # since config.device='cpu' on the experiments themselves.
        successes, failures = run_experiments_parallel(
            experiments,
            output_root,
            workers=2,
            gpu_fraction=0,  # CPU-only; bypasses GPU resource check
            local_ray=True,
        )

        assert len(failures) == 0, f"unexpected failures: {failures}"
        assert len(successes) == 2

        # Both run dirs should exist with the standard output files.
        eco_runs = sorted(output_root.iterdir())
        assert len(eco_runs) == 2
        for run_dir in eco_runs:
            assert (run_dir / "summary.json").exists()
            assert (run_dir / "config.json").exists()
            summary = json.loads((run_dir / "summary.json").read_text())
            assert "measures" in summary
            # Homogeneous experiments: interaction fields present but empty.
            assert summary["measures"]["interaction_coefficients"] == []
            assert summary["measures"]["outcome_label"] == ""

        # Names should be preserved in the run_dir paths.
        names = {r.name.split("-", 4)[-1] for r in eco_runs}
        assert names == {"alpha", "beta"}


@pytest.mark.smoke_ray
def test_run_experiments_parallel_isolates_failure() -> None:
    """One bad experiment should not abort the others; failure is reported.

    With v2.3.0+ creature resolution happens on the driver before Ray task
    submission, so a missing archive surfaces as a driver-side FileNotFoundError
    rather than a RayTaskError. Either way the dispatcher reports it in the
    failures list and continues with the rest.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        archive_dir = tmp / "archive"
        # Only run-a exists; run-b's load fails on the driver.
        _write_archive(archive_dir, "run-a")

        experiments = (
            _experiment("good", archive_dir, "run-a"),
            _experiment("bad", archive_dir, "run-missing"),
        )
        output_root = tmp / "ecosystem"

        successes, failures = run_experiments_parallel(
            experiments,
            output_root,
            workers=2,
            gpu_fraction=0,
            local_ray=True,
        )

        assert len(successes) == 1
        assert len(failures) == 1
        assert failures[0][0] == "bad"
        # FileNotFoundError from load_creature when archive run dir is missing.
        assert (
            isinstance(failures[0][1], FileNotFoundError)
            or "not found" in str(failures[0][1]).lower()
        )
