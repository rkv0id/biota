# pyright: basic
"""Ray compatibility shim. The one file in the project that runs in pyright basic
mode. All Ray imports and decorators live here so the rest of the codebase can stay
strict.

Three runtime modes, selected by the init() flags:

- default (no flags): rollouts run synchronously in the driver process.
  submit_rollout builds and immediately resolves a RolloutHandle;
  wait_for_completed returns everything in one call.
- local_ray=True: a fresh Ray instance is started in the driver process,
  and rollouts run as Ray tasks. Use num_workers to cap parallelism.
- ray_address=<str>: attach to an already-running Ray cluster at the given
  address. Rollouts run as Ray tasks on whatever nodes the cluster has.

local_ray and ray_address are mutually exclusive.

The search loop calls the same five functions in all three modes:

    init(num_workers, local_ray=..., ray_address=...)
    handle = submit_rollout(params, seed, config, device, parent_id)
    completed, still_pending = wait_for_completed(handles, min_completed=1)
    is_ray_active()
    shutdown()

The exported types are properly annotated; callers get full pyright strictness
across the rest of the project. The internal Ray plumbing is the only thing
that escapes type checking.
"""

from dataclasses import dataclass
from typing import Any

from biota.search.result import ParamDict, RolloutResult
from biota.search.rollout import RolloutConfig, rollout_batch

# === module-level state (singleton because Ray itself is a singleton) ===

_state: dict[str, Any] = {
    "initialized": False,
    "mode": None,  # "ray" or "no_ray"
}


# === public API ===


@dataclass
class RolloutHandle:
    """Opaque handle to an in-flight batch. Treat as a black box.

    Resolves to a list[RolloutResult] - one result per rollout in the batch.
    """

    _result: list[RolloutResult] | None
    _ray_ref: Any  # ray.ObjectRef[list[RolloutResult]] in ray mode, None otherwise


def init(
    *,
    local_ray: bool = False,
    ray_address: str | None = None,
) -> None:
    """Initialize the runtime. Must be called once before any submit_batch.

    Three modes, selected by the flags:

    - **default (no flags)**: no Ray. Batches run synchronously in the driver
      process. Zero Ray startup overhead, ideal for laptop dev iteration and
      for CI where parallelism comes from pytest itself.

    - **local_ray=True**: start a fresh Ray instance in the driver process.
      Ray uses its default worker count (typically detected CPU count). To
      control the number of workers, start Ray manually before calling
      biota search with --local-ray.

    - **ray_address=<str>**: attach to an already-running Ray cluster at the
      given address. Value is passed verbatim to ray.init(address=...). Use
      "host:port" (e.g. "10.10.12.1:6379") for a GCS-level connection from
      a node on the same network as the cluster. The "ray://host:port" Ray
      Client form also works but requires the ray[client] extra, which biota
      does not declare as a dependency. Cluster resources were fixed at
      'ray start' time; passing num_cpus to a cluster-connecting ray.init()
      raises ValueError, so we never pass it.

    local_ray and ray_address are mutually exclusive. Passing both raises
    ValueError. Passing neither is the no-ray default.

    Calling init twice in the same process is an error - either shutdown
    first or trust that the existing runtime is correctly configured.
    """
    if _state["initialized"]:
        raise RuntimeError(
            "ray_compat already initialized; call shutdown() first if you need to reinitialize"
        )

    if local_ray and ray_address is not None:
        raise ValueError("local_ray and ray_address are mutually exclusive; pass one or the other")

    # Default: no Ray. Synchronous in the driver.
    if not local_ray and ray_address is None:
        _state["mode"] = "no_ray"
        _state["initialized"] = True
        return

    # Ray mode: import lazily so no-ray callers don't have to spin up Ray or
    # even resolve its import path.
    import ray

    if not ray.is_initialized():
        init_kwargs = _build_ray_init_kwargs(ray_address=ray_address)
        ray.init(**init_kwargs)
    _state["mode"] = "ray"
    _state["initialized"] = True


def _build_ray_init_kwargs(ray_address: str | None) -> dict[str, Any]:
    """Build the kwargs dict to pass to ray.init() based on our mode flags.

    Extracted as a pure function so it can be unit-tested without importing
    ray or spinning up a cluster. We had two real production bugs in this
    branching, so the small test surface is worth it.

    Rules:
    - If ray_address is set, pass address=<value> only (ray.init raises
      ValueError if num_cpus is also given when attaching to a cluster).
    - If ray_address is None (fresh local Ray), pass no num_cpus and let
      Ray detect the core count. Worker count is controlled at 'ray start'
      time, not in the search CLI.
    - Always pass ignore_reinit_error=False so double-init is a loud error.
    """
    if ray_address is not None:
        return {"address": ray_address, "ignore_reinit_error": False}
    return {"ignore_reinit_error": False}


def shutdown() -> None:
    """Tear down the runtime. Idempotent."""
    if not _state["initialized"]:
        return

    if _state["mode"] == "ray":
        import ray

        if ray.is_initialized():
            ray.shutdown()

    _state["mode"] = None
    _state["initialized"] = False


def is_ray_active() -> bool:
    """True if init was called with local_ray=True or ray_address set."""
    return _state["initialized"] and _state["mode"] == "ray"


def submit_batch(
    params_list: list[ParamDict],
    seeds: list[int],
    config: RolloutConfig,
    device: str = "cpu",
    parent_ids: list[str | None] | None = None,
) -> RolloutHandle:
    """Submit a batch of rollouts for execution and return an opaque handle.

    In no_ray mode this runs the batch synchronously and returns a handle
    that's already resolved. In ray mode this fires a Ray task and returns a
    handle wrapping the ObjectRef.

    The handle resolves to list[RolloutResult], one per element of params_list.
    CUDA batches declare num_gpus=1 per task - the batch fills the GPU directly
    via vectorized forward pass, so fractional GPU allocation is not needed.
    """
    _require_initialized()

    if _state["mode"] == "no_ray":
        results = rollout_batch(params_list, seeds, config, device=device, parent_ids=parent_ids)
        return RolloutHandle(_result=results, _ray_ref=None)

    # Ray mode
    ref = _get_batch_remote(device).remote(params_list, seeds, config, device, parent_ids)
    return RolloutHandle(_result=None, _ray_ref=ref)


def wait_for_completed(
    handles: list[RolloutHandle], min_completed: int = 1
) -> tuple[list[list[RolloutResult]], list[RolloutHandle]]:
    """Block until at least min_completed handles have results, then return.

    Returns (completed_batches, still_pending_handles). Each element of
    completed_batches is a list[RolloutResult] from one completed batch
    dispatch. The still_pending list contains handles to keep waiting on.

    In no_ray mode, every handle is already resolved, so this returns all of
    them in one call regardless of min_completed.

    Raises ValueError if handles is empty.
    """
    _require_initialized()
    if not handles:
        raise ValueError("wait_for_completed needs at least one handle")

    if _state["mode"] == "no_ray":
        batches = [h._result for h in handles if h._result is not None]
        return batches, []

    # Ray mode: call ray.wait on the full ref list, then collect completed batches.
    import ray

    refs = [h._ray_ref for h in handles]
    target = min(min_completed, len(refs))
    ready, not_ready = ray.wait(refs, num_returns=target)

    completed_batches: list[list[RolloutResult]] = ray.get(ready)
    not_ready_set = set(id(r) for r in not_ready)
    still_pending = [h for h in handles if id(h._ray_ref) in not_ready_set]
    return completed_batches, still_pending


# === internals ===


def _require_initialized() -> None:
    if not _state["initialized"]:
        raise RuntimeError("ray_compat not initialized; call init() first")


def _batch_remote_impl(
    params_list: list[ParamDict],
    seeds: list[int],
    config: RolloutConfig,
    device: str,
    parent_ids: list[str | None] | None,
) -> list[RolloutResult]:
    """The function body that becomes a Ray batch task.

    Lives outside the decorator so it's importable for testing without Ray.
    Returns list[RolloutResult], one per element of params_list.
    """
    return rollout_batch(params_list, seeds, config, device=device, parent_ids=parent_ids)


def _num_gpus_for_device(device: str) -> float:
    """Ray num_gpus for a batch task on the given device.

    CUDA: 1.0 - the batch fills the whole GPU via the vectorized forward pass.
    Fractional allocation (the old gpus_per_rollout approach) is superseded
    by batch_size: the batch itself controls GPU utilisation, so each task
    legitimately owns one GPU for its duration.

    CPU and MPS: 0.0. MPS is macOS-local and cannot be scheduled through
    Ray's resource system; it works only when the Ray worker happens to be on
    the same machine as the MPS device, which on macOS it always is.

    Pure function for testability.
    """
    return 1.0 if device.startswith("cuda") else 0.0


# Cached Ray remote wrappers, one per device class. Built lazily on first
# access via _get_batch_remote() so `import ray` is deferred until needed.
# In no_ray mode this cache is never populated.
_batch_remote_cache: dict[str, Any] = {}


def _get_batch_remote(device: str) -> Any:
    """Return the Ray-remote-decorated version of _batch_remote_impl for the
    given device class, building it on first call.

    Device handling:
    - "cpu" or "mps": ray.remote(num_gpus=0). No GPU resource declared.
    - "cuda" or "cuda:N": ray.remote(num_gpus=1). Task owns one GPU for the
      duration of the batch; batch_size controls utilisation within that GPU.
    """
    # Normalise to device class for the cache key
    key = "cuda" if device.startswith("cuda") else device
    if key not in _batch_remote_cache:
        import ray

        num_gpus = _num_gpus_for_device(device)
        _batch_remote_cache[key] = ray.remote(num_gpus=num_gpus)(_batch_remote_impl)
    return _batch_remote_cache[key]
