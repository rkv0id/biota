"""Tests for biota.ray_compat in default (no-ray) mode.

These tests never import ray or spin up a Ray cluster. The Ray-mode code path
(local_ray=True, ray_address=...) is exercised by the cluster smoke test in
step 9-10, not by pytest. The one exception is _build_ray_init_kwargs, which
is a pure function that just builds the kwargs dict to pass to ray.init() -
no ray import required, safe to unit-test directly.
"""

from typing import Any

import pytest

from biota.ray_compat import (
    RolloutHandle,
    _build_ray_init_kwargs,  # pyright: ignore[reportPrivateUsage]
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


def test_init_default_is_no_ray() -> None:
    """Default init() with no flags runs in no-Ray mode (synchronous)."""
    init()
    assert is_ray_active() is False


def test_double_init_raises() -> None:
    init()
    with pytest.raises(RuntimeError, match="already initialized"):
        init()


def test_shutdown_clears_state() -> None:
    init()
    shutdown()
    assert is_ray_active() is False
    # After shutdown, init can be called again
    init()


def test_init_rejects_conflicting_flags() -> None:
    """local_ray and ray_address are mutually exclusive."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        init(local_ray=True, ray_address="10.10.12.1:6379")


# === _build_ray_init_kwargs ===


def test_build_kwargs_local_ray_with_num_workers() -> None:
    """Local Ray with num_workers set: pass num_cpus to ray.init."""
    kwargs: dict[str, Any] = _build_ray_init_kwargs(num_workers=4, ray_address=None)
    assert kwargs == {"num_cpus": 4, "ignore_reinit_error": False}


def test_build_kwargs_local_ray_without_num_workers() -> None:
    """Local Ray without num_workers: pass num_cpus=None (Ray picks default)."""
    kwargs: dict[str, Any] = _build_ray_init_kwargs(num_workers=None, ray_address=None)
    assert kwargs == {"num_cpus": None, "ignore_reinit_error": False}


def test_build_kwargs_attach_ignores_num_workers() -> None:
    """Attach mode: pass address=, never num_cpus (even if num_workers is set).

    Passing num_cpus to a cluster-connecting ray.init() raises ValueError. The
    num_workers arg is silently dropped in this branch; the cluster's
    resources were already configured at 'ray start' time.
    """
    kwargs: dict[str, Any] = _build_ray_init_kwargs(num_workers=8, ray_address="10.10.12.1:6379")
    assert kwargs == {
        "address": "10.10.12.1:6379",
        "ignore_reinit_error": False,
    }
    assert "num_cpus" not in kwargs


def test_build_kwargs_attach_with_ray_client_url() -> None:
    """Ray Client URLs (ray://host:port) pass through verbatim."""
    kwargs: dict[str, Any] = _build_ray_init_kwargs(
        num_workers=None, ray_address="ray://head.example.com:10001"
    )
    assert kwargs == {
        "address": "ray://head.example.com:10001",
        "ignore_reinit_error": False,
    }


# === submit/wait state checks ===


def test_submit_before_init_raises() -> None:
    params = sample_random(kernels=10, seed=0)
    with pytest.raises(RuntimeError, match="not initialized"):
        submit_rollout(params, seed=0, config=CHEAP_CONFIG)


def test_wait_before_init_raises() -> None:
    with pytest.raises(RuntimeError, match="not initialized"):
        wait_for_completed([])  # type: ignore[arg-type]


# === submit and wait in default (no-Ray) mode ===


def test_submit_returns_handle() -> None:
    init()
    params = sample_random(kernels=10, seed=0)
    handle = submit_rollout(params, seed=42, config=CHEAP_CONFIG)
    assert isinstance(handle, RolloutHandle)


def test_submit_then_wait_returns_correct_seed_and_params() -> None:
    init()
    params = sample_random(kernels=10, seed=0)
    handle = submit_rollout(params, seed=42, config=CHEAP_CONFIG)
    completed, _ = wait_for_completed([handle])
    assert len(completed) == 1
    assert completed[0].seed == 42
    assert completed[0].params == params


def test_submit_passes_through_parent_cell() -> None:
    init()
    params = sample_random(kernels=10, seed=0)
    handle = submit_rollout(params, seed=0, config=CHEAP_CONFIG, parent_cell=(5, 10, 3))
    completed, _ = wait_for_completed([handle])
    assert completed[0].parent_cell == (5, 10, 3)


def test_wait_for_completed_drains_all_handles() -> None:
    init()
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
    init()
    with pytest.raises(ValueError, match="at least one handle"):
        wait_for_completed([])


# === mini end-to-end search loop simulation ===


def test_mini_search_loop_default() -> None:
    """Simulate the search loop's submit-then-drain pattern with 5 rollouts."""
    init()

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
