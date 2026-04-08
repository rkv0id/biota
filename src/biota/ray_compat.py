# pyright: basic
"""Ray compatibility shim. The one file in the project that runs in pyright basic
mode. All Ray imports and decorators live here so the rest of the codebase can stay
strict.

Two runtime modes:

- no_ray: rollouts run synchronously in the driver process. submit_rollout
  builds and immediately resolves a RolloutHandle; wait_for_completed returns
  everything in one call.
- ray:    rollouts run as Ray tasks on workers in the cluster. submit_rollout
  returns a handle wrapping a Ray ObjectRef; wait_for_completed uses ray.wait
  to drain completed handles as they finish.

The search loop calls the same five functions in both modes:

    init(num_workers, no_ray)
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


def init(num_workers: int | None = None, *, no_ray: bool = False) -> None:
    """Initialize the runtime. Must be called once before any submit_rollout.

    In no_ray mode, num_workers is ignored. In ray mode, num_workers caps the
    parallelism if set; if None, Ray uses its default (typically detected CPU
    count, or whatever the cluster's resources expose).

    Cluster connection: ray.init() may connect to an already-running Ray
    cluster instead of starting a fresh local one. This happens when:

    1. The RAY_ADDRESS env var is set (commonly to 'auto' or 'host:port').
    2. A running local cluster is detected via /tmp/ray/ray_current_cluster,
       which ray itself writes when 'ray start --head' runs on the machine.

    In either case we do NOT pass num_cpus, because the cluster's resources
    were already configured at 'ray start' time; passing num_cpus to a
    cluster-connecting ray.init() raises ValueError. num_workers is effectively
    ignored when attaching; the cluster decides its own resource counts.

    Calling init twice in the same process is an error - either shutdown
    first or trust that the existing runtime is correctly configured.
    """
    if _state["initialized"]:
        raise RuntimeError(
            "ray_compat already initialized; call shutdown() first if you need to reinitialize"
        )

    if no_ray:
        _state["mode"] = "no_ray"
        _state["initialized"] = True
        return

    # Ray mode: import lazily so unit tests in no_ray mode don't have to spin
    # up Ray or even resolve its import path
    import ray

    if not ray.is_initialized():
        if _is_attaching_to_existing_cluster():
            # Joining an existing cluster. Don't pass num_cpus; the cluster's
            # resources were set when 'ray start' was run on the nodes.
            ray.init(ignore_reinit_error=False)
        else:
            # Starting a fresh local Ray instance in the driver process.
            ray.init(num_cpus=num_workers, ignore_reinit_error=False)
    _state["mode"] = "ray"
    _state["initialized"] = True


def _is_attaching_to_existing_cluster() -> bool:
    """Return True if ray.init() will connect to an already-running cluster
    rather than starting a fresh local one.

    Mirrors Ray's own autodetection order:
    1. RAY_ADDRESS env var is set
    2. /tmp/ray/ray_current_cluster file exists (written by 'ray start')

    Used to decide whether to pass num_cpus to ray.init(): passing num_cpus
    when attaching to an existing cluster raises ValueError, so we only pass
    it when starting a fresh instance.
    """
    import os

    if os.environ.get("RAY_ADDRESS"):
        return True
    return os.path.exists("/tmp/ray/ray_current_cluster")


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
    """True if init was called with no_ray=False."""
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
    ref = _get_rollout_remote().remote(params, seed, config, device, parent_cell)
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


# Cached Ray remote wrapper. Built on first access via _get_rollout_remote()
# so `import ray` is deferred until actually needed. In no_ray mode this
# cache is never populated.
_rollout_remote_cache: Any = None


def _get_rollout_remote() -> Any:
    """Return the Ray-remote-decorated version of _rollout_remote_impl, building
    it on first call. This is the only thing that triggers `import ray` in
    normal use.
    """
    global _rollout_remote_cache
    if _rollout_remote_cache is None:
        import ray

        _rollout_remote_cache = ray.remote(_rollout_remote_impl)
    return _rollout_remote_cache
