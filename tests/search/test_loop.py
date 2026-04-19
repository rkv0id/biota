"""Tests for the search loop.

Most tests share a module-scoped fixture that runs one search with budget=10
and captures all events plus the run directory. Individual tests then assert
against the cached results. This keeps total test time around 30 seconds
despite each rollout taking ~2 seconds at the cheap config.

The determinism test runs its own pair of searches because it needs to
compare two independent runs.

All tests use the default no-Ray mode so pytest never spins up Ray.
"""

import json
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypedDict

import pytest

from biota.ray_compat import shutdown
from biota.search.archive import Archive
from biota.search.loop import (
    CheckpointWritten,
    RolloutCompleted,
    SearchConfig,
    SearchEvent,
    SearchFinished,
    SearchStarted,
    search,
)
from biota.search.rollout import RolloutConfig
from biota.sim.flowlenia import Config as SimConfig


class SessionSearch(TypedDict):
    archive: Archive
    events: list[SearchEvent]
    run_dir: Path


def _cheap_config(budget: int = 10, random_phase_size: int = 5, **kwargs: Any) -> SearchConfig:
    defaults: dict[str, Any] = {"batch_size": 1, "workers": 1, "calibration": 5, "centroids": 16}
    defaults.update(kwargs)
    return SearchConfig(
        rollout=RolloutConfig(sim=SimConfig(grid_h=32, grid_w=32, kernels=10), steps=110),
        budget=budget,
        random_phase_size=random_phase_size,
        base_seed=42,
        checkpoint_every=3,
        **defaults,
    )


@pytest.fixture(autouse=True)
def clean_runtime() -> Iterator[None]:
    shutdown()
    yield
    shutdown()


@pytest.fixture(scope="module")
def session_search(tmp_path_factory: pytest.TempPathFactory) -> SessionSearch:
    """Run one search and cache its outputs for tests to share.

    Module-scoped so the ~20-second rollout cost is paid once per file.
    Tests assert against the cached events list, archive, and run directory.
    """
    tmp_path = tmp_path_factory.mktemp("session_search")
    events: list[SearchEvent] = []
    archive = search(
        config=_cheap_config(budget=10, random_phase_size=5),
        runs_root=tmp_path,
        on_event=events.append,
    )
    run_dir = next(tmp_path.iterdir())
    return SessionSearch(archive=archive, events=events, run_dir=run_dir)


# === lifecycle ===


def test_search_returns_archive(session_search: SessionSearch) -> None:
    assert isinstance(session_search["archive"], Archive)


def test_search_creates_run_dir(session_search: SessionSearch) -> None:
    run_dir = session_search["run_dir"]
    assert run_dir.is_dir()


def test_search_writes_manifest_and_config(session_search: SessionSearch) -> None:
    run_dir = session_search["run_dir"]
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "config.json").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert "run_id" in manifest
    assert "started_at" in manifest
    assert "config" in manifest


def test_search_writes_archive_checkpoint(session_search: SessionSearch) -> None:
    run_dir = session_search["run_dir"]
    pkl = run_dir / "archive.pkl"
    assert pkl.exists()
    with open(pkl, "rb") as f:
        loaded = pickle.load(f)
    assert isinstance(loaded, Archive)


def test_search_writes_events_log(session_search: SessionSearch) -> None:
    run_dir = session_search["run_dir"]
    events_path = run_dir / "events.jsonl"
    assert events_path.exists()
    lines = events_path.read_text().strip().split("\n")
    assert len(lines) == 10
    first = json.loads(lines[0])
    assert "seed" in first
    assert "quality" in first
    assert "insertion_status" in first


# === events ===


def test_search_emits_started_and_finished_events(session_search: SessionSearch) -> None:
    events = session_search["events"]
    started = [e for e in events if isinstance(e, SearchStarted)]
    finished = [e for e in events if isinstance(e, SearchFinished)]
    assert len(started) == 1
    assert len(finished) == 1
    assert finished[0].completed == 10


def test_search_emits_one_completed_event_per_rollout(session_search: SessionSearch) -> None:
    events = session_search["events"]
    completed = [e for e in events if isinstance(e, RolloutCompleted)]
    assert len(completed) == 10
    counts = [e.completed_count for e in completed]
    assert counts == sorted(counts)
    assert counts[-1] == 10


def test_search_emits_checkpoint_events(session_search: SessionSearch) -> None:
    events = session_search["events"]
    checkpoints = [e for e in events if isinstance(e, CheckpointWritten)]
    # checkpoint_every=3, budget=10 -> mid-run checkpoints at 3, 6, 9 plus
    # 1 final = at least 4
    assert len(checkpoints) >= 3


# === phase transitions ===


def test_random_phase_results_have_no_parent_id(session_search: SessionSearch) -> None:
    events = session_search["events"]
    completed = [e for e in events if isinstance(e, RolloutCompleted)]
    sorted_by_seed = sorted(completed, key=lambda e: e.result.seed)
    for event in sorted_by_seed[:5]:
        assert event.result.parent_id is None


def test_mutation_phase_runs_to_completion(session_search: SessionSearch) -> None:
    """Soft check: the loop ran all 10 rollouts (5 random + 5 mutation)
    without crashing, regardless of whether mutation samples got real parents
    or fell back to random."""
    events = session_search["events"]
    completed = [e for e in events if isinstance(e, RolloutCompleted)]
    assert len(completed) == 10


# === determinism (runs its own pair of searches) ===


def test_random_phase_is_deterministic(tmp_path: Path) -> None:
    """Same base_seed -> same sequence of random-phase results in default
    no-Ray mode. Runs two independent searches with random_phase_size == budget
    so no mutation-phase nondeterminism enters the picture."""
    events_a: list[SearchEvent] = []
    events_b: list[SearchEvent] = []
    cfg = _cheap_config(budget=5, random_phase_size=5)

    search(config=cfg, runs_root=tmp_path / "a", on_event=events_a.append)
    search(config=cfg, runs_root=tmp_path / "b", on_event=events_b.append)

    completed_a = [e for e in events_a if isinstance(e, RolloutCompleted)]
    completed_b = [e for e in events_b if isinstance(e, RolloutCompleted)]
    assert len(completed_a) == len(completed_b) == 5

    for a, b in zip(completed_a, completed_b, strict=True):
        assert a.result.params == b.result.params
        assert a.result.descriptors == b.result.descriptors
        assert a.result.quality == b.result.quality
        assert a.result.seed == b.result.seed


# === SearchConfig validation ===


def test_searchconfig_rejects_zero_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        _cheap_config(batch_size=0)


def test_searchconfig_rejects_zero_workers() -> None:
    with pytest.raises(ValueError, match="workers"):
        _cheap_config(workers=0)


def test_searchconfig_default_batch_size_is_one() -> None:
    cfg = _cheap_config()
    assert cfg.batch_size == 1


def test_searchconfig_default_workers_is_one() -> None:
    cfg = _cheap_config()
    assert cfg.workers == 1


# === Signal-only descriptor validation ===


def test_signal_only_descriptor_without_signal_field_raises() -> None:
    """Passing a signal-only descriptor without --signal-field raises ValueError."""
    cfg = _cheap_config(
        budget=5,
        random_phase_size=5,
        descriptor_names=("signal_field_variance", "gyradius", "spectral_entropy"),
        signal_field=False,
    )
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(ValueError, match="signal-enabled"):
        search(cfg, runs_root=tmpdir)


def test_signal_only_descriptor_with_signal_field_does_not_raise() -> None:
    """signal-only descriptor with signal_field=True passes validation."""
    cfg = _cheap_config(
        budget=5,
        random_phase_size=5,
        descriptor_names=("signal_field_variance", "gyradius", "spectral_entropy"),
        signal_field=True,
    )
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        archive = search(cfg, runs_root=tmpdir)
        assert archive is not None


def test_non_signal_descriptor_without_signal_field_does_not_raise() -> None:
    """Standard descriptors work without signal_field."""
    cfg = _cheap_config(budget=5, random_phase_size=5)
    assert not cfg.signal_field  # default is False
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        archive = search(cfg, runs_root=tmpdir)
        assert archive is not None
