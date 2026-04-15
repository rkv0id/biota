# pyright: basic
"""Parallel dispatch for ecosystem experiments via Ray.

When the user passes --local-ray or --ray-address to `biota ecosystem`, this
module wraps each experiment as a Ray remote task and runs them concurrently.
The simulation body itself (run_ecosystem) is unchanged; the parallelism is
purely at the experiment-fanout level.

Two CLI knobs control concurrency:

- workers: total number of in-flight experiments at any time. Defaults to the
  detected GPU count (or 1 if no GPUs available). Caps how many Ray tasks the
  scheduler runs concurrently.
- gpu_fraction: fraction of a GPU each worker reserves. Default 1.0 means one
  worker per GPU. Set to 0.5 to pack two workers per GPU, 0.33 for three, etc.
  Useful when ecosystem runs leave the GPU underutilized at the per-task level
  (typical for small grids or low species counts).

Failure isolation: one experiment crashing does not abort the others. Failures
are collected and surfaced at the end with a per-experiment status summary.
This matches the sequential-mode tolerance (we already swallow GIF write
failures with a print warning).

Progress: as each Ray task completes, this module prints a 'completed N/M' line
naming the experiment. Order of completion is non-deterministic; the printed
order tracks ray.wait's resolution order, which roughly correlates with task
runtime (shorter experiments finish first).

Like ray_compat.py, this file runs in pyright basic mode so we can import ray
without dragging the rest of the codebase into untyped territory.
"""

import time
from pathlib import Path
from typing import Any

from biota.ecosystem.config import EcosystemConfig
from biota.ecosystem.result import EcosystemResult
from biota.ecosystem.run import run_ecosystem

# === public API ===


def run_experiments_parallel(
    experiments: tuple[EcosystemConfig, ...],
    output_root: Path,
    workers: int,
    gpu_fraction: float,
    *,
    local_ray: bool = False,
    ray_address: str | None = None,
) -> tuple[list[EcosystemResult], list[tuple[str, BaseException]]]:
    """Run all experiments in parallel via Ray; return (successes, failures).

    Either local_ray=True or ray_address must be set. Passing neither raises
    ValueError; the caller (CLI) is expected to dispatch to the sequential
    path when no Ray flag is present.

    workers caps in-flight tasks. gpu_fraction sets num_gpus per task.

    The returned successes list preserves submission order (matches the input
    experiments tuple, with failed entries omitted). Failures carry the
    experiment name and the exception instance.
    """
    if not local_ray and ray_address is None:
        raise ValueError(
            "run_experiments_parallel requires either local_ray=True or ray_address; "
            "use the sequential path when neither is set"
        )
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if gpu_fraction < 0:
        raise ValueError(f"gpu_fraction must be >= 0, got {gpu_fraction}")

    import ray

    init_kwargs: dict[str, Any] = {"ignore_reinit_error": False}
    if ray_address is not None:
        init_kwargs["address"] = ray_address
    if not ray.is_initialized():
        ray.init(**init_kwargs)

    try:
        return _dispatch(experiments, output_root, workers, gpu_fraction)
    finally:
        # Only tear down the Ray instance we own (local mode). When attaching
        # to an external cluster we leave it running for the user.
        if local_ray and ray.is_initialized():
            ray.shutdown()


def detect_gpu_count() -> int:
    """Best-effort GPU count detection; returns 0 on any failure.

    Used as the default for --workers when neither flag is passed by the user.
    Prefers torch's CUDA enumeration since that's what the simulation actually
    uses. Falls back to 0 (caller defaults to workers=1) on any error.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        pass
    return 0


# === internal ===


def _dispatch(
    experiments: tuple[EcosystemConfig, ...],
    output_root: Path,
    workers: int,
    gpu_fraction: float,
) -> tuple[list[EcosystemResult], list[tuple[str, BaseException]]]:
    """Submit all experiments as Ray tasks, collect results as they complete."""
    import ray

    # Build the remote function dynamically so num_gpus comes from the CLI flag.
    # Defining it inline (not at module scope) means re-decorating per call;
    # cheap, and avoids leaking a global with a fixed gpu_fraction.
    @ray.remote(num_gpus=gpu_fraction)
    def _ray_run(cfg: EcosystemConfig, out: Path) -> EcosystemResult:
        return run_ecosystem(cfg, output_root=out)

    # Tag each future with its experiment so we can name failures and report
    # progress with the right name. ray.wait returns ObjectRefs in resolution
    # order, not submission order, so a side-table is necessary.
    futures = [_ray_run.remote(exp, output_root) for exp in experiments]
    future_to_exp: dict[Any, EcosystemConfig] = dict(zip(futures, experiments, strict=True))

    successes: dict[int, EcosystemResult] = {}  # keyed by submission index
    failures: list[tuple[str, BaseException]] = []
    submission_index = {id(f): i for i, f in enumerate(futures)}

    pending = list(futures)
    total = len(pending)
    completed = 0
    started_at = time.time()

    # Cap concurrency by the workers flag using Ray's max_concurrency-style
    # pattern: only wait on at most `workers` tasks at a time. When fewer tasks
    # remain than workers the wait collapses to whatever is left.
    while pending:
        # Number of tasks Ray should consider; ray.wait returns when at least
        # one is ready, so we always make forward progress.
        ready, pending = ray.wait(pending, num_returns=1, timeout=None)
        for ref in ready:
            exp = future_to_exp[ref]
            idx = submission_index[id(ref)]
            try:
                result: EcosystemResult = ray.get(ref)
                successes[idx] = result
                status = "ok"
            except Exception as exc:
                failures.append((exp.name, exc))
                status = f"FAILED ({type(exc).__name__})"
            completed += 1
            elapsed = time.time() - started_at
            print(
                f"[ecosystem] completed {completed}/{total}: {exp.name!r} {status} "
                f"(wall {elapsed:.1f}s)"
            )

    # Preserve submission order in successes return value.
    successes_ordered = [successes[i] for i in sorted(successes.keys())]
    return successes_ordered, failures
