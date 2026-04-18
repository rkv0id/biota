"""Tests for biota.ray_compat in default (no-ray) mode.

These tests never import ray or spin up a Ray cluster. The Ray-mode code path
(local_ray=True, ray_address=...) is exercised by the cluster smoke test, not
by pytest. The pure-function helpers _build_ray_init_kwargs and
_num_gpus_for_device are unit-tested directly since they encapsulate branching
logic that has caused production bugs in the past.
"""

from typing import Any

import pytest

from biota.ray_compat import (
    RolloutHandle,
    _build_ray_init_kwargs,  # pyright: ignore[reportPrivateUsage]
    _num_gpus_for_device,  # pyright: ignore[reportPrivateUsage]
    init,
    is_ray_active,
    shutdown,
    submit_batch,
    wait_for_completed,
)
from biota.search.params import sample_random
from biota.search.rollout import (
    THUMBNAIL_FRAMES,
    THUMBNAIL_SIZE,
    RolloutConfig,
)
from biota.sim.flowlenia import Config as SimConfig

CHEAP_CONFIG = RolloutConfig(sim=SimConfig(grid_h=32, grid_w=32, kernels=10), steps=50)


@pytest.fixture(autouse=True)
def clean_runtime():
    """Ensure each test starts and ends with a clean ray_compat state."""
    shutdown()
    yield
    shutdown()


# === lifecycle ===


def test_shutdown_on_uninitialized_is_idempotent() -> None:
    shutdown()
    shutdown()


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
    init()


def test_init_rejects_conflicting_flags() -> None:
    """local_ray and ray_address are mutually exclusive."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        init(local_ray=True, ray_address="10.10.12.1:6379")


# === _build_ray_init_kwargs ===


def test_build_kwargs_local_ray() -> None:
    """Fresh local Ray: no address, no num_cpus (Ray detects core count)."""
    kwargs: dict[str, Any] = _build_ray_init_kwargs(ray_address=None)
    assert kwargs == {"ignore_reinit_error": False}
    assert "num_cpus" not in kwargs
    assert "address" not in kwargs


def test_build_kwargs_attach() -> None:
    """Attach mode: pass address= only, never num_cpus.

    Passing num_cpus to a cluster-connecting ray.init() raises ValueError.
    Worker count is fixed at 'ray start' time, not in the search CLI.
    """
    kwargs: dict[str, Any] = _build_ray_init_kwargs(ray_address="10.10.12.1:6379")
    assert kwargs == {
        "address": "10.10.12.1:6379",
        "ignore_reinit_error": False,
    }
    assert "num_cpus" not in kwargs


def test_build_kwargs_attach_with_ray_client_url() -> None:
    """Ray Client URLs (ray://host:port) pass through verbatim to ray.init.

    Connecting via ray:// requires pip install ray[client], which biota does
    not declare as a dependency. This test only verifies that the address
    normalization does not mangle the URL, not that the connection succeeds.
    """
    kwargs: dict[str, Any] = _build_ray_init_kwargs(ray_address="ray://head.example.com:10001")
    assert kwargs == {
        "address": "ray://head.example.com:10001",
        "ignore_reinit_error": False,
    }


# === _num_gpus_for_device ===


def test_num_gpus_for_cpu_is_zero() -> None:
    assert _num_gpus_for_device("cpu") == 0.0


def test_num_gpus_for_cuda_is_one() -> None:
    """CUDA batch tasks declare num_gpus=1 - the batch fills the whole GPU."""
    assert _num_gpus_for_device("cuda") == 1.0
    assert _num_gpus_for_device("cuda:0") == 1.0
    assert _num_gpus_for_device("cuda:3") == 1.0


def test_num_gpus_for_mps_is_zero() -> None:
    """MPS is macOS-local, not schedulable via Ray's resource system."""
    assert _num_gpus_for_device("mps") == 0.0


# === submit and wait - state checks ===


def test_submit_before_init_raises() -> None:
    params = sample_random(kernels=10, seed=0)
    with pytest.raises(RuntimeError, match="not initialized"):
        submit_batch([params], seeds=[0], config=CHEAP_CONFIG)


def test_wait_before_init_raises() -> None:
    with pytest.raises(RuntimeError, match="not initialized"):
        wait_for_completed([])  # type: ignore[arg-type]


# === submit_batch and wait_for_completed in default (no-Ray) mode ===


def test_submit_batch_returns_handle() -> None:
    init()
    params = sample_random(kernels=10, seed=0)
    handle = submit_batch([params], seeds=[42], config=CHEAP_CONFIG)
    assert isinstance(handle, RolloutHandle)


def test_submit_batch_single_returns_one_result() -> None:
    init()
    params = sample_random(kernels=10, seed=0)
    handle = submit_batch([params], seeds=[42], config=CHEAP_CONFIG)
    completed, pending = wait_for_completed([handle])
    assert len(completed) == 1  # one batch
    assert len(completed[0]) == 1  # one result in that batch
    assert completed[0][0].seed == 42
    assert completed[0][0].params == params
    assert len(pending) == 0


def test_submit_batch_multiple_returns_all_results() -> None:
    """A batch of B params produces a single handle resolving to B results."""
    init()
    B = 3
    params_list = [sample_random(kernels=10, seed=i) for i in range(B)]
    handle = submit_batch(params_list, seeds=list(range(B)), config=CHEAP_CONFIG)
    completed, pending = wait_for_completed([handle])
    assert len(completed) == 1  # one batch handle
    batch = completed[0]
    assert len(batch) == B
    assert sorted(r.seed for r in batch) == list(range(B))
    assert len(pending) == 0


def test_submit_batch_passes_through_parent_cells() -> None:
    init()
    params = sample_random(kernels=10, seed=0)
    handle = submit_batch([params], seeds=[0], config=CHEAP_CONFIG, parent_cells=[(5, 10, 3)])
    completed, _ = wait_for_completed([handle])
    assert completed[0][0].parent_cell == (5, 10, 3)


def test_submit_batch_default_parent_cells_none() -> None:
    """parent_cells defaults to [None]*B when not supplied."""
    init()
    params = sample_random(kernels=10, seed=0)
    handle = submit_batch([params], seeds=[0], config=CHEAP_CONFIG)
    completed, _ = wait_for_completed([handle])
    assert completed[0][0].parent_cell is None


def test_wait_for_completed_drains_multiple_handles() -> None:
    """Multiple single-element batch handles all drain in one wait call."""
    init()
    handles = [
        submit_batch([sample_random(kernels=10, seed=i)], seeds=[i], config=CHEAP_CONFIG)
        for i in range(3)
    ]
    completed, pending = wait_for_completed(handles, min_completed=1)
    # In no-ray mode all handles are already resolved
    assert len(completed) == 3
    assert len(pending) == 0
    seeds = sorted(r.seed for batch in completed for r in batch)
    assert seeds == [0, 1, 2]


def test_wait_for_completed_empty_raises() -> None:
    init()
    with pytest.raises(ValueError, match="at least one handle"):
        wait_for_completed([])


# === mini end-to-end search loop simulation ===


def test_mini_search_loop_batched() -> None:
    """Simulate the loop's submit-then-drain pattern with batch_size=2."""
    init()

    params_all = [sample_random(kernels=10, seed=i) for i in range(5)]
    handles: list[RolloutHandle] = [
        submit_batch(params_all[0:2], seeds=[0, 1], config=CHEAP_CONFIG),
        submit_batch(params_all[2:4], seeds=[2, 3], config=CHEAP_CONFIG),
        submit_batch(params_all[4:5], seeds=[4], config=CHEAP_CONFIG),
    ]

    all_results = []
    while handles:
        completed_batches, handles = wait_for_completed(handles, min_completed=1)
        for batch in completed_batches:
            all_results.extend(batch)

    assert len(all_results) == 5
    expected_size = min(THUMBNAIL_SIZE, CHEAP_CONFIG.sim.grid)
    for result in all_results:
        assert isinstance(result.seed, int)
        assert result.params is not None
        assert result.thumbnail.shape == (THUMBNAIL_FRAMES, expected_size, expected_size)
