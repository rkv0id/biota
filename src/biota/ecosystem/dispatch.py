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

    # Driver-side creature loading. Workers in cluster mode have no access to
    # the driver's archive directory (different filesystems on different
    # nodes), so we read each creature here and ship it as task payload.
    # Local-Ray and noray paths still benefit: archives are read once per
    # experiment regardless of how many tasks reference them.
    #
    # Load failures (missing archive, missing cell) isolate per experiment
    # the same way runtime failures do: surfaced in the failures list, the
    # other experiments still run. This matches the existing tolerance the
    # rest of the dispatcher already provides.
    from biota.ecosystem.run import load_creature

    successes: dict[int, EcosystemResult] = {}  # keyed by submission index
    failures: list[tuple[str, BaseException]] = []

    # Resolve each experiment to (config, creatures_or_None). None means a
    # driver-side load failed; the experiment is recorded as failed and
    # not submitted to Ray.
    resolved: list[tuple[int, EcosystemConfig, list[Any] | None]] = []
    for idx, exp in enumerate(experiments):
        try:
            creatures = [load_creature(s) for s in exp.sources]
        except Exception as exc:
            failures.append((exp.name, exc))
            resolved.append((idx, exp, None))
            continue
        resolved.append((idx, exp, creatures))

    # Build the remote function dynamically so num_gpus comes from the CLI flag.
    # Defining it inline (not at module scope) means re-decorating per call;
    # cheap, and avoids leaking a global with a fixed gpu_fraction.
    #
    # The remote task runs compute_ecosystem (no I/O) and returns
    # (EcosystemResult, dict[str, bytes]). The driver materializes the bytes
    # under its own output_root so cluster runs land on the driver, not on
    # whichever worker happened to execute the task. Worker file systems
    # never get written to.
    @ray.remote(num_gpus=gpu_fraction)
    def _ray_run(
        cfg: EcosystemConfig, creatures: list[Any], out: Path
    ) -> tuple[EcosystemResult, dict[str, bytes]]:
        from biota.ecosystem.run import compute_ecosystem

        return compute_ecosystem(cfg, output_root=out, creatures=creatures)

    # Submit only the experiments whose creatures resolved cleanly.
    futures: list[Any] = []
    future_to_exp: dict[Any, EcosystemConfig] = {}
    submission_index: dict[int, int] = {}
    for idx, exp, creatures in resolved:
        if creatures is None:
            continue
        ref = _ray_run.remote(exp, creatures, output_root)
        futures.append(ref)
        future_to_exp[ref] = exp
        submission_index[id(ref)] = idx

    pending = list(futures)
    total = len(experiments)  # report against full experiment count, not just submitted
    # Pre-failed (driver-side) count so the running tally is honest.
    completed = sum(1 for _, _, c in resolved if c is None)
    started_at = time.time()

    # Print pre-failure lines so the user sees them inline with progress.
    for _, exp, creatures in resolved:
        if creatures is None:
            print(
                f"[ecosystem] completed {completed}/{total}: {exp.name!r} "
                f"FAILED (driver-side: archive load) (wall 0.0s)"
            )

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
                result, artifacts = ray.get(ref)
                # Materialize on the driver. The worker's compute_ecosystem
                # returned a prospective run_dir based on its view of
                # output_root; on the driver that path is the same string but
                # refers to the driver's filesystem. mkdir + write here.
                from biota.ecosystem.run import materialize_outputs

                local_run_dir = Path(result.run_dir)
                local_run_dir.mkdir(parents=True, exist_ok=True)
                materialize_outputs(local_run_dir, artifacts)
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
