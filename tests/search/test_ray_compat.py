"""Tests for biota.ray_compat in --no-ray mode.

These tests never import ray or spin up a Ray cluster. The Ray-mode code path
is exercised by the cluster smoke test in step 9-10, not by pytest.
"""

import pytest

from biota.ray_compat import (
    RolloutHandle,
    _is_attaching_to_existing_cluster,  # pyright: ignore[reportPrivateUsage]
    init,
    is_ray_active,
    shutdown,
    submit_rollout,
    wait_for_completed,
)
from biota.search.params import sample_random
from biota.search.rollout import RolloutConfig
from biota.sim.flowlenia import Config as SimConfig

CHEAP_CONFIG = RolloutConfig(sim=SimConfig(grid=32, kernels=10), steps=50)


@pytest.fixture(autouse=True)
def clean_runtime():
    """Ensure each test starts and ends with a clean ray_compat state."""
    shutdown()  # in case a previous test left state
    yield
    shutdown()


# === lifecycle ===


def test_shutdown_on_uninitialized_is_idempotent() -> None:
    shutdown()  # no error
    shutdown()  # still no error


def test_init_no_ray_marks_initialized() -> None:
    init(no_ray=True)
    assert is_ray_active() is False


def test_double_init_raises() -> None:
    init(no_ray=True)
    with pytest.raises(RuntimeError, match="already initialized"):
        init(no_ray=True)


def test_shutdown_clears_state() -> None:
    init(no_ray=True)
    shutdown()
    assert is_ray_active() is False
    # After shutdown, init can be called again
    init(no_ray=True)


# === cluster detection ===


def _always_false(_p: str) -> bool:
    return False


def _always_true(_p: str) -> bool:
    return True


def test_cluster_detection_false_when_no_env_no_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With neither RAY_ADDRESS nor the cookie file present, init() should
    take the fresh-local branch and pass num_cpus."""
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.setattr("os.path.exists", _always_false)
    assert _is_attaching_to_existing_cluster() is False


def test_cluster_detection_true_when_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """With RAY_ADDRESS set, init() should take the attach branch and NOT
    pass num_cpus (which would raise ValueError)."""
    monkeypatch.setenv("RAY_ADDRESS", "auto")
    monkeypatch.setattr("os.path.exists", _always_false)
    assert _is_attaching_to_existing_cluster() is True


def test_cluster_detection_true_when_cookie_file_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the ray_current_cluster file present (as 'ray start' would write),
    init() should take the attach branch even if RAY_ADDRESS is unset."""
    monkeypatch.delenv("RAY_ADDRESS", raising=False)

    def fake_exists(p: str) -> bool:
        return p == "/tmp/ray/ray_current_cluster"

    monkeypatch.setattr("os.path.exists", fake_exists)
    assert _is_attaching_to_existing_cluster() is True


def test_cluster_detection_env_takes_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If both signals are present, either one is enough; both -> True."""
    monkeypatch.setenv("RAY_ADDRESS", "auto")
    monkeypatch.setattr("os.path.exists", _always_true)
    assert _is_attaching_to_existing_cluster() is True


def test_submit_before_init_raises() -> None:
    params = sample_random(kernels=10, seed=0)
    with pytest.raises(RuntimeError, match="not initialized"):
        submit_rollout(params, seed=0, config=CHEAP_CONFIG)


def test_wait_before_init_raises() -> None:
    with pytest.raises(RuntimeError, match="not initialized"):
        wait_for_completed([])  # type: ignore[arg-type]


# === submit and wait in no_ray mode ===


def test_submit_returns_handle() -> None:
    init(no_ray=True)
    params = sample_random(kernels=10, seed=0)
    handle = submit_rollout(params, seed=42, config=CHEAP_CONFIG)
    assert isinstance(handle, RolloutHandle)


def test_submit_then_wait_returns_correct_seed_and_params() -> None:
    init(no_ray=True)
    params = sample_random(kernels=10, seed=0)
    handle = submit_rollout(params, seed=42, config=CHEAP_CONFIG)
    completed, _ = wait_for_completed([handle])
    assert len(completed) == 1
    assert completed[0].seed == 42
    assert completed[0].params == params


def test_submit_passes_through_parent_cell() -> None:
    init(no_ray=True)
    params = sample_random(kernels=10, seed=0)
    handle = submit_rollout(params, seed=0, config=CHEAP_CONFIG, parent_cell=(5, 10, 3))
    completed, _ = wait_for_completed([handle])
    assert completed[0].parent_cell == (5, 10, 3)


def test_wait_for_completed_drains_all_handles() -> None:
    init(no_ray=True)
    handles = [
        submit_rollout(sample_random(kernels=10, seed=i), seed=i, config=CHEAP_CONFIG)
        for i in range(3)
    ]
    completed, pending = wait_for_completed(handles, min_completed=1)
    assert len(completed) == 3
    assert len(pending) == 0
    # Check seeds round-trip correctly
    seeds = sorted(r.seed for r in completed)
    assert seeds == [0, 1, 2]


def test_wait_for_completed_empty_raises() -> None:
    init(no_ray=True)
    with pytest.raises(ValueError, match="at least one handle"):
        wait_for_completed([])


# === mini end-to-end search loop simulation ===


def test_mini_search_loop_no_ray() -> None:
    """Simulate the search loop's submit-then-drain pattern with 5 rollouts."""
    init(no_ray=True)

    handles: list[RolloutHandle] = []
    for i in range(5):
        params = sample_random(kernels=10, seed=i)
        handles.append(submit_rollout(params, seed=i, config=CHEAP_CONFIG))

    all_results = []
    while handles:
        completed, handles = wait_for_completed(handles, min_completed=1)
        all_results.extend(completed)

    assert len(all_results) == 5
    # All results have valid params and seeds
    for result in all_results:
        assert isinstance(result.seed, int)
        assert result.params is not None
        # Most will fail the persistent filter at this small config but they
        # should all return successfully (not raise)
        assert result.thumbnail.shape == (16, 32, 32)
