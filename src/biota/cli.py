"""Command-line interface for biota.

Two commands:

    biota search    Run a MAP-Elites search with configurable flags.
    biota doctor    Print runtime info (versions, devices, module health).

Most users only need `biota search --preset dev --no-ray` for quick local
runs; `biota search --preset standard` for a serious search. `biota doctor`
is a one-command install sanity check.

The CLI is thin: it parses flags into a SearchConfig and calls
biota.search.loop.search. All real logic lives in the search layer.
"""

import platform
import sys
from dataclasses import replace
from pathlib import Path

import typer

from biota.search.archive import InsertionStatus
from biota.search.loop import (
    CheckpointWritten,
    RolloutCompleted,
    SearchConfig,
    SearchEvent,
    SearchStarted,
    search,
)
from biota.search.rollout import RolloutConfig, dev_preset, pretty_preset, standard_preset
from biota.sim.flowlenia import Config as SimConfig

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _resolve_preset(name: str) -> RolloutConfig:
    presets = {
        "dev": dev_preset,
        "standard": standard_preset,
        "pretty": pretty_preset,
    }
    factory = presets.get(name)
    if factory is None:
        raise typer.BadParameter(
            f"unknown preset {name!r}. choose from: {', '.join(presets.keys())}"
        )
    return factory()


def _override_sim(rollout: RolloutConfig, grid: int | None, steps: int | None) -> RolloutConfig:
    """Apply --grid and --steps overrides on top of a preset."""
    new_sim = rollout.sim
    if grid is not None:
        new_sim = SimConfig(
            grid=grid,
            kernels=rollout.sim.kernels,
            dd=rollout.sim.dd,
            dt=rollout.sim.dt,
            sigma=rollout.sim.sigma,
            border=rollout.sim.border,
        )
    new_steps = steps if steps is not None else rollout.steps
    return replace(rollout, sim=new_sim, steps=new_steps)


# === search command ===


def _format_status(status: InsertionStatus) -> str:
    """Short tag for the progress line."""
    return {
        InsertionStatus.INSERTED: "inserted ",
        InsertionStatus.REPLACED: "replaced ",
        InsertionStatus.REJECTED_FILTER: "rej:filt ",
        InsertionStatus.REJECTED_QUALITY: "rej:qual ",
        InsertionStatus.REJECTED_SIMILARITY: "rej:sim  ",
    }[status]


def _print_event(event: SearchEvent, budget: int) -> None:
    """Print one line per event to stderr. Keeps stdout clean for scripting."""
    if isinstance(event, SearchStarted):
        print(
            f"[search] starting run {event.run_id} "
            f"budget={event.config.budget} "
            f"random_phase={event.config.random_phase_size} "
            f"device={event.config.device}",
            file=sys.stderr,
        )
    elif isinstance(event, RolloutCompleted):
        tag = _format_status(event.insertion_status)
        result = event.result
        q = f"q={result.quality:.3f}" if result.quality is not None else "q=  -  "
        reason = f"  {result.rejection_reason}" if result.rejection_reason else ""
        print(
            f"[{event.completed_count:4d}/{budget:4d}] seed={result.seed:<5d} {tag} {q}{reason}",
            file=sys.stderr,
        )
    elif isinstance(event, CheckpointWritten):
        print(
            f"[checkpoint] {event.path} ({event.archive_size} cells)",
            file=sys.stderr,
        )
    else:
        # SearchFinished - pyright narrows via exhaustiveness
        print(
            f"[done] {event.completed} rollouts, "
            f"{event.archive_size} archive cells, "
            f"{event.elapsed_seconds:.1f}s elapsed",
            file=sys.stderr,
        )


@app.command(name="search")
def search_cmd(
    preset: str = typer.Option(
        "standard", "--preset", help="Resolution preset: dev, standard, or pretty."
    ),
    budget: int = typer.Option(500, "--budget", help="Total number of rollouts."),
    random_phase: int = typer.Option(
        200, "--random-phase", help="Random sampling rollouts before mutation."
    ),
    max_concurrent: int = typer.Option(8, "--max-concurrent", help="Maximum in-flight rollouts."),
    no_ray: bool = typer.Option(
        False, "--no-ray", help="Bypass Ray; run synchronously in the driver."
    ),
    num_workers: int | None = typer.Option(
        None, "--num-workers", help="Ray worker cap (ignored with --no-ray)."
    ),
    device: str = typer.Option("cpu", "--device", help="Torch device: cpu, mps, or cuda."),
    base_seed: int = typer.Option(0, "--base-seed", help="Seed for reproducibility."),
    checkpoint_every: int = typer.Option(
        100, "--checkpoint-every", help="Checkpoint cadence in completed rollouts."
    ),
    runs_root: Path = typer.Option(
        Path("runs"), "--runs-root", help="Root directory for run subdirectories."
    ),
    grid: int | None = typer.Option(
        None, "--grid", help="Override preset grid size (for experimentation)."
    ),
    steps: int | None = typer.Option(
        None, "--steps", help="Override preset step count (for experimentation)."
    ),
) -> None:
    """Run a MAP-Elites search over Flow-Lenia parameters."""
    rollout_cfg = _override_sim(_resolve_preset(preset), grid=grid, steps=steps)
    config = SearchConfig(
        rollout=rollout_cfg,
        budget=budget,
        random_phase_size=random_phase,
        max_concurrent=max_concurrent,
        no_ray=no_ray,
        num_workers=num_workers,
        device=device,
        base_seed=base_seed,
        checkpoint_every=checkpoint_every,
    )

    def on_event(event: SearchEvent) -> None:
        _print_event(event, budget=budget)

    archive = search(config=config, runs_root=runs_root, on_event=on_event)

    # Final summary to stdout, one line, scriptable
    print(f"archive_size={len(archive)}")


# === doctor command ===


def _format_doctor() -> str:
    """Collect runtime info and format as a human-readable block."""
    lines: list[str] = []

    try:
        from importlib.metadata import version

        biota_version = version("biota")
    except Exception:
        biota_version = "unknown"
    lines.append(f"biota {biota_version}")

    py = sys.version_info
    plat = f"{platform.system().lower()} {platform.machine()}"
    lines.append(f"python {py.major}.{py.minor}.{py.micro} ({plat})")

    try:
        import torch

        lines.append(f"torch {torch.__version__}")
        lines.append("  cpu:  yes")
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        lines.append(f"  mps:  {'yes' if mps_available else 'no'}")
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            lines.append(f"  cuda: yes ({torch.cuda.device_count()} device(s))")
        else:
            lines.append("  cuda: no")
    except ImportError:
        lines.append("torch: NOT INSTALLED")

    try:
        import ray

        lines.append(f"ray {ray.__version__}")
    except ImportError:
        lines.append("ray: NOT INSTALLED")

    # biota.search module health - importlib avoids unused-import warnings
    import importlib

    search_modules = (
        "biota.search.archive",
        "biota.search.descriptors",
        "biota.search.loop",
        "biota.search.params",
        "biota.search.quality",
        "biota.search.result",
        "biota.search.rollout",
    )
    try:
        for mod in search_modules:
            importlib.import_module(mod)
        lines.append(f"biota.search: ok ({len(search_modules)} modules)")
    except ImportError as e:
        lines.append(f"biota.search: FAIL ({e})")

    # biota.ray_compat health in no_ray mode
    try:
        from biota.ray_compat import init, is_ray_active, shutdown

        init(no_ray=True)
        assert is_ray_active() is False
        shutdown()
        lines.append("biota.ray_compat: ok (no-ray init successful)")
    except Exception as e:
        lines.append(f"biota.ray_compat: FAIL ({e})")

    return "\n".join(lines)


@app.command()
def doctor() -> None:
    """Print detected runtime info (versions, devices, module health)."""
    print(_format_doctor())
