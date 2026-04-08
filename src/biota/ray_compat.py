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
    handle = submit_rollout(params, seed, config, device, parent_cell)
    completed, still_pending = wait_for_completed(handles, min_completed=1)
    is_ray_active()
    shutdown()

The exported types are properly annotated; callers get full pyright strictness
across the rest of the project. The internal Ray plumbing is the only thing
that escapes type checking.

See DECISIONS.md (2026-04-06) for why we firewall Ray here.
"""

from dataclasses import dataclass
from typing import Any

from biota.search.result import CellCoord, ParamDict, RolloutResult
from biota.search.rollout import RolloutConfig, rollout

# === module-level state (singleton because Ray itself is a singleton) ===

_state: dict[str, Any] = {
    "initialized": False,
    "mode": None,  # "ray" or "no_ray"
}


# === public API ===


@dataclass
class RolloutHandle:
    """Opaque handle to an in-flight rollout. Treat as a black box."""

    _result: RolloutResult | None
    _ray_ref: Any  # ray.ObjectRef[RolloutResult] in ray mode, None in no_ray mode


def init(
    num_workers: int | None = None,
    *,
    local_ray: bool = False,
    ray_address: str | None = None,
) -> None:
    """Initialize the runtime. Must be called once before any submit_rollout.

    Three modes, selected by the flags:

    - **default (no flags)**: no Ray. Rollouts run synchronously in the driver
      process. Zero Ray startup overhead, ideal for laptop dev iteration and
      for CI where parallelism comes from pytest itself.

    - **local_ray=True**: start a fresh Ray instance in the driver process.
      num_workers caps the parallelism if set; if None, Ray uses its default
      (typically detected CPU count).

    - **ray_address=<str>**: attach to an already-running Ray cluster at the
      given address. Value is passed verbatim to ray.init(address=...). Use
      "host:port" for a GCS-level connection from the same network, or
      "ray://host:port" for Ray Client protocol. num_workers is ignored when
      attaching because cluster resources were fixed at 'ray start' time;
      passing num_cpus to a cluster-connecting ray.init() raises ValueError.

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
        init_kwargs = _build_ray_init_kwargs(num_workers=num_workers, ray_address=ray_address)
        ray.init(**init_kwargs)
    _state["mode"] = "ray"
    _state["initialized"] = True


def _build_ray_init_kwargs(num_workers: int | None, ray_address: str | None) -> dict[str, Any]:
    """Build the kwargs dict to pass to ray.init() based on our mode flags.

    Extracted as a pure function so it can be unit-tested without importing
    ray or spinning up a cluster. The branching logic is small enough that
    directly testing it is overkill, but we had two real production bugs in
    this branching already (see DECISIONS.md 2026-04-07 and 2026-04-08), so
    the tiny test surface is worth it.

    Rules:
    - If ray_address is set, pass address=<value> and never num_cpus
      (ray.init raises ValueError if both are given when attaching).
    - If ray_address is None, pass num_cpus=num_workers for fresh local Ray.
    - Always pass ignore_reinit_error=False so double-init is a loud error.
    """
    if ray_address is not None:
        return {"address": ray_address, "ignore_reinit_error": False}
    return {"num_cpus": num_workers, "ignore_reinit_error": False}


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


def submit_rollout(
    params: ParamDict,
    seed: int,
    config: RolloutConfig,
    device: str = "cpu",
    parent_cell: CellCoord | None = None,
) -> RolloutHandle:
    """Submit one rollout for execution and return an opaque handle.

    In no_ray mode this runs the rollout synchronously and returns a handle
    that's already resolved. In ray mode this fires a Ray task and returns a
    handle wrapping the ObjectRef.

    The same handle type works for both modes; wait_for_completed knows how
    to drain it.
    """
    _require_initialized()

    if _state["mode"] == "no_ray":
        result = rollout(params, seed, config, device=device, parent_cell=parent_cell)
        return RolloutHandle(_result=result, _ray_ref=None)

    # Ray mode
    ref = _get_rollout_remote(device).remote(params, seed, config, device, parent_cell)
    return RolloutHandle(_result=None, _ray_ref=ref)


def wait_for_completed(
    handles: list[RolloutHandle], min_completed: int = 1
) -> tuple[list[RolloutResult], list[RolloutHandle]]:
    """Block until at least min_completed handles have results, then return.

    Returns (completed_results, still_pending_handles). The completed list
    contains the actual RolloutResult objects; the still_pending list
    contains handles to keep waiting on.

    In no_ray mode, every handle is already resolved, so this returns all of
    them in one call regardless of min_completed.

    Raises ValueError if handles is empty.
    """
    _require_initialized()
    if not handles:
        raise ValueError("wait_for_completed needs at least one handle")

    if _state["mode"] == "no_ray":
        results = [h._result for h in handles if h._result is not None]
        return results, []

    # Ray mode: split handles into ones that are resolved (cached) and not.
    # In practice, in ray mode every handle has a ref and no cached _result,
    # so we just call ray.wait on the full list.
    import ray

    refs = [h._ray_ref for h in handles]
    target = min(min_completed, len(refs))
    ready, not_ready = ray.wait(refs, num_returns=target)

    completed_results: list[RolloutResult] = ray.get(ready)
    not_ready_set = set(id(r) for r in not_ready)
    still_pending = [h for h in handles if id(h._ray_ref) in not_ready_set]
    return completed_results, still_pending


# === internals ===


def _require_initialized() -> None:
    if not _state["initialized"]:
        raise RuntimeError("ray_compat not initialized; call init() first")


def _rollout_remote_impl(
    params: ParamDict,
    seed: int,
    config: RolloutConfig,
    device: str,
    parent_cell: CellCoord | None,
) -> RolloutResult:
    """The function body that becomes a Ray task. Lives outside the decorator
    so it's importable for testing without needing Ray.
    """
    return rollout(params, seed, config, device=device, parent_cell=parent_cell)


def _num_gpus_for_device(device: str) -> int:
    """Map a torch device string to the Ray num_gpus resource count.

    CUDA devices need num_gpus=1 so Ray schedules the task onto a GPU-bearing
    worker and sets CUDA_VISIBLE_DEVICES correctly. Everything else (cpu, mps)
    uses num_gpus=0. Pure function for testability.
    """
    return 1 if device.startswith("cuda") else 0


# Cached Ray remote wrappers, one per device class. Built lazily on first
# access via _get_rollout_remote() so `import ray` is deferred until actually
# needed. Keyed by a normalized device tag so CPU and CUDA rollouts get
# different @ray.remote decorations (CUDA requires num_gpus=1 so Ray doesn't
# hide the GPU from the worker via CUDA_VISIBLE_DEVICES=""). In no_ray mode
# this cache is never populated.
_rollout_remote_cache: dict[str, Any] = {}


def _get_rollout_remote(device: str) -> Any:
    """Return the Ray-remote-decorated version of _rollout_remote_impl for the
    given device class, building it on first call. This is the only thing that
    triggers `import ray` in normal use.

    Device handling:

    - "cpu" -> ray.remote(num_gpus=0)(impl). CPU-only task, Ray keeps
      CUDA_VISIBLE_DEVICES clear.
    - "cuda" or "cuda:N" -> ray.remote(num_gpus=1)(impl). Task requires one
      GPU. Ray schedules onto a GPU-bearing worker and sets
      CUDA_VISIBLE_DEVICES to the assigned GPU index so torch sees it.
    - "mps" -> ray.remote(num_gpus=0)(impl). MPS is macOS-local; there's no
      way to schedule MPS tasks through Ray's resource system. MPS + Ray is
      effectively unsupported but we don't error out here; the rollout will
      work if the Ray worker happens to be on the same machine as an MPS
      device, which on macOS it always is.
    """
    key = "cuda" if device.startswith("cuda") else "cpu"
    if key not in _rollout_remote_cache:
        import ray

        num_gpus = _num_gpus_for_device(device)
        _rollout_remote_cache[key] = ray.remote(num_gpus=num_gpus)(_rollout_remote_impl)
    return _rollout_remote_cache[key]
