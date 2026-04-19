"""Command-line interface for biota.

Two commands:

    biota search    Run a MAP-Elites search with configurable flags.
    biota doctor    Print runtime info (versions, devices, module health).

Most users only need `biota search --preset dev` for quick local runs
(synchronous, no Ray). Add `--local-ray` for local parallelism, or
`--ray-address HOST` to attach to an existing cluster. `biota doctor`
is a one-command install sanity check.

The CLI is thin: it parses flags into a SearchConfig and calls
biota.search.loop.search. All real logic lives in the search layer.
"""

import importlib
import importlib.util
import platform
import sys
from dataclasses import replace
from pathlib import Path

import typer

from biota.search.archive import Archive
from biota.search.descriptors import DEFAULT_DESCRIPTORS, REGISTRY, Descriptor
from biota.search.loop import (
    CalibrationDoneFn,
    CalibrationProgressFn,
    CheckpointWritten,
    EventCallback,
    RolloutCompleted,
    SearchConfig,
    SearchEvent,
    SearchStarted,
    search,
)
from biota.search.rollout import (
    PRESET_CALIBRATION,
    SIGNAL_CALIBRATION_BONUS,
    SIGNAL_STEPS,
    RolloutConfig,
    dev_preset,
    pretty_preset,
    standard_preset,
)
from biota.sim.flowlenia import Config as SimConfig
from biota.viz.tty import SearchDisplay

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


def _override_sim(
    rollout: RolloutConfig,
    grid: int | None,
    steps: int | None,
    border: str | None,
) -> RolloutConfig:
    """Apply --grid, --steps, and --border overrides on top of a preset."""
    new_sim = rollout.sim
    effective_border = border if border is not None else rollout.sim.border
    if grid is not None or border is not None:
        new_sim = SimConfig(
            grid_h=grid if grid is not None else rollout.sim.grid_h,
            grid_w=grid if grid is not None else rollout.sim.grid_w,
            kernels=rollout.sim.kernels,
            dd=rollout.sim.dd,
            dt=rollout.sim.dt,
            sigma=rollout.sim.sigma,
            border=effective_border,
        )
    new_steps = steps if steps is not None else rollout.steps
    return replace(rollout, sim=new_sim, steps=new_steps)


# === search command ===


def _make_event_handler(
    display: SearchDisplay,
    live_archive: Archive,
) -> tuple[CalibrationProgressFn, CalibrationDoneFn, EventCallback]:
    """Build the two callbacks wired to a SearchDisplay.

    archive_ref is a single-element list mutated by the search loop so the
    event handler can read the live archive for descriptor coverage bars.
    """

    def on_cal_progress(completed: int, total: int, n_survivors: int) -> None:
        display.on_calibration_progress(completed, total, n_survivors)

    def on_cal_done(
        n_survivors: int,
        descriptor_names: tuple[str, str, str],
        axis_ranges: "list[tuple[float,float]] | None" = None,
    ) -> None:
        display.on_calibration_done(n_survivors, descriptor_names, axis_ranges)

    def on_event(event: SearchEvent) -> None:
        if isinstance(event, SearchStarted):
            display.on_search_started(event.run_id, event.config)
        elif isinstance(event, RolloutCompleted):
            result = event.result
            # Update descriptor coverage from the live archive
            if live_archive.calibrated:
                desc_values: list[list[float]] = [[], [], []]
                for _idx, r in live_archive.iter_occupied():
                    if r.descriptors is not None:
                        for i, v in enumerate(r.descriptors):
                            desc_values[i].append(float(v))
                display.on_archive_snapshot(
                    archive_size=len(live_archive),
                    fill_pct=live_archive.fill_fraction,
                    desc_values=desc_values,
                )
            display.on_rollout_completed(
                completed=event.completed_count,
                status=event.insertion_status.value,
                quality=result.quality,
                rejection_reason=result.rejection_reason,
                seed=result.seed,
                descriptors=result.descriptors,
            )
        elif isinstance(event, CheckpointWritten):
            display.on_checkpoint(str(event.path), event.archive_size)
        else:
            # SearchFinished -- pyright exhaustiveness
            display.on_search_finished(event.completed, event.archive_size, event.elapsed_seconds)

    return on_cal_progress, on_cal_done, on_event


RAY_DEFAULT_PORT = 6379


def load_descriptor_module(path: Path) -> None:
    """Load a user-supplied Python file and merge its DESCRIPTORS into REGISTRY.

    The file must define a module-level list named DESCRIPTORS. Every element
    must be a Descriptor instance with a callable compute field. Fails loudly
    on any violation rather than silently producing bad archives.
    """
    spec = importlib.util.spec_from_file_location("_biota_custom_descriptors", path)
    if spec is None or spec.loader is None:
        raise typer.BadParameter(
            f"cannot load descriptor module: {path}", param_hint="--descriptor-module"
        )
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception as exc:
        raise typer.BadParameter(
            f"error executing descriptor module {path}: {exc}", param_hint="--descriptor-module"
        ) from exc

    if not hasattr(mod, "DESCRIPTORS"):
        raise typer.BadParameter(
            f"{path} must define a list named DESCRIPTORS", param_hint="--descriptor-module"
        )
    descriptors = mod.DESCRIPTORS
    if not isinstance(descriptors, list):
        raise typer.BadParameter(
            f"{path}: DESCRIPTORS must be a list, got {type(descriptors).__name__}",
            param_hint="--descriptor-module",
        )
    for i, d in enumerate(descriptors):
        if not isinstance(d, Descriptor):
            raise typer.BadParameter(
                f"{path}: DESCRIPTORS[{i}] is {type(d).__name__}, expected Descriptor",
                param_hint="--descriptor-module",
            )
        if not callable(d.compute):
            raise typer.BadParameter(
                f"{path}: DESCRIPTORS[{i}] ({d.name!r}) has non-callable compute field",
                param_hint="--descriptor-module",
            )
        REGISTRY[d.name] = d


def _resolve_descriptor_names(names_str: str) -> tuple[str, str, str]:
    """Parse and validate three descriptor names from a comma-separated string."""
    names = [n.strip() for n in names_str.replace(",", " ").split() if n.strip()]
    if len(names) != 3:
        raise typer.BadParameter(
            f"--descriptors requires exactly 3 names, got {len(names)}: {names_str!r}",
            param_hint="--descriptors",
        )
    for name in names:
        if name not in REGISTRY:
            known = ", ".join(sorted(REGISTRY))
            raise typer.BadParameter(
                f"unknown descriptor {name!r}. known: {known}",
                param_hint="--descriptors",
            )
    return names[0], names[1], names[2]


def _normalize_ray_address(value: str | None) -> str | None:
    """Add the default port to a host-only ray_address if no port is given.

    Rules:
    - None -> None
    - "ray://..." -> passed through verbatim (Ray Client URL; requires ray[client])
    - "host:port" -> passed through verbatim
    - "host" -> "host:6379" (append default GCS port)
    """
    if value is None:
        return None
    if value.startswith("ray://"):
        return value
    if ":" in value:
        return value
    return f"{value}:{RAY_DEFAULT_PORT}"


@app.command(name="search")
def search_cmd(
    preset: str = typer.Option(
        "standard", "--preset", help="Resolution preset: dev, standard, or pretty."
    ),
    budget: int = typer.Option(500, "--budget", help="Total number of rollouts."),
    random_phase: int = typer.Option(
        200, "--random-phase", help="Random sampling rollouts before mutation."
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        help=(
            "Rollouts evaluated simultaneously per dispatch. Default 1 matches "
            "pre-v0.4.0 behaviour. On cuda/mps, values of 32-128 give meaningful "
            "speedup. On cpu, leave at 1."
        ),
    ),
    workers: int = typer.Option(
        1,
        "--workers",
        help=(
            "Concurrent batch dispatches in flight. Default 1 = synchronous "
            "MAP-Elites (maximally fresh archive). Higher values trade archive "
            "freshness for throughput on multi-node cluster setups."
        ),
    ),
    local_ray: bool = typer.Option(
        False,
        "--local-ray",
        help="Start a fresh local Ray instance. Default: no Ray (synchronous).",
    ),
    ray_address: str | None = typer.Option(
        None,
        "--ray-address",
        help=(
            "Attach to an existing Ray cluster at HOST[:PORT]. "
            f"Port defaults to {RAY_DEFAULT_PORT} if not given. "
            "Use 'ray://host:port' for Ray Client protocol (requires pip install ray[client]). "
            "Mutually exclusive with --local-ray."
        ),
    ),
    device: str = typer.Option("cpu", "--device", help="Torch device: cpu, mps, or cuda."),
    base_seed: int = typer.Option(0, "--base-seed", help="Seed for reproducibility."),
    checkpoint_every: int = typer.Option(
        100, "--checkpoint-every", help="Checkpoint cadence in completed rollouts."
    ),
    calibration: int | None = typer.Option(
        None,
        "--calibration",
        help=(
            "Calibration rollouts before search begins (not counted in budget). "
            "Defaults to the preset value (dev=50, standard=150, pretty=200) "
            "plus 50 when --signal-field is active. Override with an explicit N."
        ),
    ),
    centroids: int = typer.Option(
        1024, "--centroids", help="CVT archive capacity (number of Voronoi cells)."
    ),
    output_dir: Path = typer.Option(
        Path("archive"), "--output-dir", help="Root directory for run subdirectories."
    ),
    grid: int | None = typer.Option(
        None, "--grid", help="Override preset grid size (for experimentation)."
    ),
    steps: int | None = typer.Option(
        None, "--steps", help="Override preset step count (for experimentation)."
    ),
    border: str | None = typer.Option(
        None,
        "--border",
        help="Grid border: 'wall' (default) or 'torus' (wrapping edges).",
    ),
    descriptors: str = typer.Option(
        ",".join(DEFAULT_DESCRIPTORS),
        "--descriptors",
        help=(
            "Three descriptor names for the archive axes, comma-separated. "
            "Example: --descriptors velocity,gyradius,oscillation. "
            "Must all exist in the built-in registry or --descriptor-module."
        ),
    ),
    descriptor_module: Path | None = typer.Option(
        None,
        "--descriptor-module",
        help=(
            "Path to a Python file defining custom Descriptor objects. "
            "The file must define a list named DESCRIPTORS. "
            "Custom descriptors are merged into the registry before --descriptors is resolved."
        ),
    ),
    signal_field: bool = typer.Option(
        False,
        "--signal-field",
        help=(
            "Enable the signal field. Adds emission_vector, receptor_profile, and "
            "signal_kernel_* to the searchable parameter space. Produces a signal-enabled "
            "archive tagged in manifest.json. Signal archives are incompatible with "
            "non-signal archives in ecosystem runs."
        ),
    ),
) -> None:
    """Run a MAP-Elites search over Flow-Lenia parameters."""
    if local_ray and ray_address is not None:
        raise typer.BadParameter(
            "--local-ray and --ray-address are mutually exclusive; pass one or the other",
            param_hint="--local-ray / --ray-address",
        )

    if descriptor_module is not None:
        load_descriptor_module(descriptor_module)

    descriptor_names = _resolve_descriptor_names(descriptors)

    if border is not None and border not in ("wall", "torus"):
        raise typer.BadParameter(
            f"border must be 'wall' or 'torus', got {border!r}", param_hint="--border"
        )
    base_preset = _resolve_preset(preset)
    # When --signal-field is active and no explicit --steps override was given,
    # apply the per-preset signal step count (more steps for signal dynamics).
    effective_steps = steps
    if signal_field and steps is None and preset in SIGNAL_STEPS:
        effective_steps = SIGNAL_STEPS[preset]
    rollout_cfg = _override_sim(base_preset, grid=grid, steps=effective_steps, border=border)

    # Calibration: use explicit --calibration if given, otherwise derive from preset
    # and add the signal bonus when --signal-field is active.
    effective_calibration = calibration
    if effective_calibration is None:
        effective_calibration = PRESET_CALIBRATION.get(preset, 150)
        if signal_field:
            effective_calibration += SIGNAL_CALIBRATION_BONUS

    config = SearchConfig(
        rollout=rollout_cfg,
        budget=budget,
        random_phase_size=random_phase,
        batch_size=batch_size,
        workers=workers,
        local_ray=local_ray,
        ray_address=_normalize_ray_address(ray_address),
        device=device,
        base_seed=base_seed,
        checkpoint_every=checkpoint_every,
        descriptor_names=descriptor_names,
        signal_field=signal_field,
        calibration=effective_calibration,
        centroids=centroids,
    )

    from biota.search.archive import Archive as _Archive

    live_archive = _Archive(
        n_centroids=centroids,
        descriptor_names=descriptor_names,
    )
    display = SearchDisplay(
        budget=budget,
        calibration=effective_calibration,
        descriptor_names=descriptor_names,
        device=device,
        workers=workers,
    )
    # Pass the archive object so the event handler can read live state for coverage bars
    on_cal_progress, on_cal_done, on_event = _make_event_handler(display, live_archive)

    archive = search(
        config=config,
        runs_root=output_dir,
        on_event=on_event,
        on_calibration_progress=on_cal_progress,
        on_calibration_done=on_cal_done,
        archive=live_archive,
    )

    # Final summary to stdout, one line, scriptable
    print(f"archive_size={len(archive)}")


# === ecosystem command ===


@app.command()
def ecosystem(
    config: Path = typer.Option(
        ...,
        "--config",
        help="Path to a YAML file defining one or more ecosystem experiments.",
    ),
    archive_dir: Path = typer.Option(
        Path("archive"),
        "--archive-dir",
        help=(
            "Default archive directory. Individual sources in the config may "
            "override this with their own archive_dir."
        ),
    ),
    output_dir: Path = typer.Option(
        Path("ecosystem"),
        "--output-dir",
        help="Root directory for ecosystem run output.",
    ),
    device: str = typer.Option("cpu", "--device", help="Torch device: cpu, mps, or cuda."),
    local_ray: bool = typer.Option(
        False,
        "--local-ray",
        help=(
            "Start a fresh local Ray instance and run experiments in parallel. "
            "Mutually exclusive with --ray-address."
        ),
    ),
    ray_address: str | None = typer.Option(
        None,
        "--ray-address",
        help=(
            "Attach to an existing Ray cluster at HOST[:PORT] and run experiments "
            "in parallel. Use 'ray://host:port' for Ray Client protocol "
            "(requires pip install ray[client]). "
            "Mutually exclusive with --local-ray."
        ),
    ),
    workers: int | None = typer.Option(
        None,
        "--workers",
        help=(
            "Maximum experiments running concurrently when Ray is active. "
            "Defaults to the detected CUDA GPU count, or 1 if no GPUs found. "
            "Ignored without --local-ray or --ray-address."
        ),
    ),
    gpu_fraction: float | None = typer.Option(
        None,
        "--gpu-fraction",
        help=(
            "Fraction of a GPU each worker reserves. Defaults derive from "
            "--device: 1.0 for cuda (one worker per GPU), 0 for cpu and mps "
            "(no GPU reservation). Set explicitly to pack workers per GPU, "
            "e.g. 0.5 with --device cuda runs two workers per GPU. Ignored "
            "without Ray."
        ),
    ),
) -> None:
    """Run one or more ecosystem experiments declared in a YAML config."""
    from biota.ecosystem.config import ConfigError, load_config_file
    from biota.ecosystem.run import run_ecosystem

    if local_ray and ray_address is not None:
        raise typer.BadParameter(
            "--local-ray and --ray-address are mutually exclusive; pass one or the other",
            param_hint="--local-ray / --ray-address",
        )

    # Resolve --gpu-fraction from --device when not set explicitly. cpu and
    # mps both mean "Ray should not reserve GPUs" because the simulation
    # doesn't use them; cuda means "one worker per GPU" by default.
    if gpu_fraction is None:
        gpu_fraction = 1.0 if device == "cuda" else 0.0
    elif gpu_fraction < 0:
        raise typer.BadParameter(
            f"--gpu-fraction must be >= 0, got {gpu_fraction}",
            param_hint="--gpu-fraction",
        )
    elif device == "cuda" and gpu_fraction == 0:
        # The contradictory case: user asked for cuda but told Ray not to
        # reserve GPUs. Tasks would compete chaotically for whichever GPU
        # each happened to land on. Refuse rather than silently misbehave.
        raise typer.BadParameter(
            "--device cuda with --gpu-fraction 0 is contradictory; pass --device cpu "
            "for CPU-only Ray scheduling, or pass --gpu-fraction > 0 to reserve GPU(s).",
            param_hint="--gpu-fraction",
        )
    elif device != "cuda" and gpu_fraction > 0:
        # User passed --gpu-fraction X with --device cpu/mps. Ray would
        # reserve GPU resources that the simulation never uses, idling them.
        # Warn but don't block; sometimes users want this for cluster
        # scheduling reasons we can't predict.
        print(
            f"[ecosystem] note: --gpu-fraction {gpu_fraction} with --device {device} "
            f"reserves GPUs that the simulation will not use. Pass --gpu-fraction 0 "
            f"to release them.",
            file=sys.stderr,
        )

    try:
        experiments = load_config_file(
            config,
            default_archive_dir=str(archive_dir),
            default_device=device,
        )
    except ConfigError as exc:
        raise typer.BadParameter(str(exc), param_hint="--config") from exc

    print(
        f"[ecosystem] loaded {len(experiments)} experiment"
        f"{'s' if len(experiments) != 1 else ''} from {config}",
        file=sys.stderr,
    )

    use_ray = local_ray or ray_address is not None
    if use_ray:
        from biota.ecosystem.dispatch import detect_gpu_count, run_experiments_parallel

        if workers is None:
            workers = max(1, detect_gpu_count())
        if workers < 1:
            raise typer.BadParameter(
                f"--workers must be >= 1, got {workers}", param_hint="--workers"
            )
        print(
            f"[ecosystem] parallel dispatch: workers={workers}, "
            f"gpu_fraction={gpu_fraction} "
            f"({'local Ray' if local_ray else f'cluster {ray_address}'})",
            file=sys.stderr,
        )
        successes, failures = run_experiments_parallel(
            experiments,
            output_dir,
            workers=workers,
            gpu_fraction=gpu_fraction,
            local_ray=local_ray,
            ray_address=_normalize_ray_address(ray_address),
        )
        for result in successes:
            print(f"output={result.run_dir}")
        if failures:
            print(
                f"[ecosystem] {len(failures)} of {len(experiments)} experiments failed:",
                file=sys.stderr,
            )
            for name, exc in failures:
                print(f"  - {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            raise typer.Exit(code=1)
        return

    # Sequential path.
    if workers is not None:
        print(
            "[ecosystem] note: --workers ignored without --local-ray or --ray-address; "
            "running sequentially.",
            file=sys.stderr,
        )
    for i, exp in enumerate(experiments):
        print(
            f"[ecosystem] running experiment {i + 1}/{len(experiments)}: {exp.name!r} "
            f"({'heterogeneous' if exp.is_heterogeneous else 'homogeneous'}, "
            f"{len(exp.sources)} source{'s' if len(exp.sources) != 1 else ''})",
            file=sys.stderr,
        )
        result = run_ecosystem(exp, output_root=output_dir)
        print(f"output={result.run_dir}")


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

    # biota.ray_compat health in default (no-Ray) mode
    try:
        from biota.ray_compat import init, is_ray_active, shutdown

        init()
        assert is_ray_active() is False
        shutdown()
        lines.append("biota.ray_compat: ok (default init successful)")
    except Exception as e:
        lines.append(f"biota.ray_compat: FAIL ({e})")

    # biota.ecosystem module health
    ecosystem_modules = (
        "biota.ecosystem.result",
        "biota.ecosystem.spawn",
        "biota.ecosystem.run",
    )
    try:
        for mod in ecosystem_modules:
            importlib.import_module(mod)
        lines.append(f"biota.ecosystem: ok ({len(ecosystem_modules)} modules)")
    except ImportError as e:
        lines.append(f"biota.ecosystem: FAIL ({e})")

    return "\n".join(lines)


@app.command()
def doctor() -> None:
    """Print detected runtime info (versions, devices, module health)."""
    print(_format_doctor())
