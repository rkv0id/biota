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

from biota.search.archive import InsertionStatus
from biota.search.descriptors import DEFAULT_DESCRIPTORS, REGISTRY, Descriptor
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
    - "ray://..." -> passed through verbatim (Ray Client URL)
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
            "Use 'ray://host:port' for Ray Client protocol. "
            "Mutually exclusive with --local-ray."
        ),
    ),
    device: str = typer.Option("cpu", "--device", help="Torch device: cpu, mps, or cuda."),
    base_seed: int = typer.Option(0, "--base-seed", help="Seed for reproducibility."),
    checkpoint_every: int = typer.Option(
        100, "--checkpoint-every", help="Checkpoint cadence in completed rollouts."
    ),
    output_dir: Path = typer.Option(
        Path("archive-runs"), "--output-dir", help="Root directory for run subdirectories."
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
    rollout_cfg = _override_sim(_resolve_preset(preset), grid=grid, steps=steps, border=border)
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
    )

    def on_event(event: SearchEvent) -> None:
        _print_event(event, budget=budget)

    archive = search(config=config, runs_root=output_dir, on_event=on_event)

    # Final summary to stdout, one line, scriptable
    print(f"archive_size={len(archive)}")


# === ecosystem command ===


@app.command()
def ecosystem(
    run: str = typer.Option(
        ..., "--run", help="Source archive run id (directory name under --archive-dir)."
    ),
    cell: str = typer.Option(
        ...,
        "--cell",
        help="Archive cell coordinate as y,x,z (e.g. 10,15,8).",
    ),
    n: int = typer.Option(..., "--n", help="Number of creature copies to spawn."),
    grid: str = typer.Option(
        ...,
        "--grid",
        help="Grid dimensions. Square: 512. Rectangular: 768x512 (HxW).",
    ),
    steps: int = typer.Option(..., "--steps", help="Number of simulation steps."),
    snapshot_every: int = typer.Option(
        ..., "--snapshot-every", help="Capture a state snapshot every N steps."
    ),
    patch: int = typer.Option(
        ..., "--patch", help="Side length of the initial random patch per creature."
    ),
    min_dist: int = typer.Option(
        ...,
        "--min-dist",
        help="Minimum pixel distance between spawn centers (Poisson disk sampling).",
    ),
    output_format: str = typer.Option(
        "gif",
        "--output-format",
        help="Output format for simulation frames: 'gif' (default) or 'frames' (individual PNGs).",
    ),
    device: str = typer.Option("cpu", "--device", help="Torch device: cpu, mps, or cuda."),
    archive_dir: Path = typer.Option(
        Path("archive-runs"),
        "--archive-dir",
        help="Directory containing archive run subdirectories.",
    ),
    output_dir: Path = typer.Option(
        Path("ecosystem-runs"),
        "--output-dir",
        help="Root directory for ecosystem run output.",
    ),
    seed: int = typer.Option(0, "--seed", help="Base RNG seed for spawn positions and patches."),
    border: str = typer.Option(
        "torus",
        "--border",
        help="Grid border: 'wall' or 'torus' (default).",
    ),
) -> None:
    """Run a homogeneous ecosystem simulation from a single archive creature."""
    import pickle

    from biota.ecosystem.result import EcosystemConfig, SpawnConfig
    from biota.ecosystem.run import run_ecosystem
    from biota.search.archive import Archive

    # Parse --grid HxW or single integer
    try:
        if "x" in grid.lower():
            parts_g = grid.lower().split("x")
            if len(parts_g) != 2:
                raise ValueError
            grid_h, grid_w = int(parts_g[0]), int(parts_g[1])
        else:
            grid_h = grid_w = int(grid)
    except ValueError:
        raise typer.BadParameter(
            f"--grid must be an integer (512) or HxW (768x512). Got: {grid!r}",
            param_hint="--grid",
        ) from None

    # Parse --cell coordinate
    try:
        cell_parts = [int(x.strip()) for x in cell.split(",")]
        if len(cell_parts) != 3:
            raise ValueError
        coords: tuple[int, int, int] = (cell_parts[0], cell_parts[1], cell_parts[2])
    except ValueError:
        raise typer.BadParameter(
            f"--cell must be three comma-separated integers, e.g. 10,15,8. Got: {cell!r}",
            param_hint="--cell",
        ) from None

    if border not in ("wall", "torus"):
        raise typer.BadParameter(
            f"border must be 'wall' or 'torus', got {border!r}", param_hint="--border"
        )

    if output_format not in ("gif", "frames"):
        raise typer.BadParameter(
            f"output-format must be 'gif' or 'frames', got {output_format!r}",
            param_hint="--output-format",
        )

    # Load archive
    run_dir = archive_dir / run
    if not run_dir.exists():
        raise typer.BadParameter(f"run directory not found: {run_dir}", param_hint="--run")

    pkl_path = run_dir / "archive.pkl"
    if not pkl_path.exists():
        raise typer.BadParameter(f"no archive.pkl in {run_dir}", param_hint="--run")

    with open(pkl_path, "rb") as f:
        archive = pickle.load(f)

    if not isinstance(archive, Archive):
        raise typer.Exit(code=1)

    if coords not in archive:
        raise typer.BadParameter(
            f"cell {coords} not in archive. Use build_index.py to inspect occupied cells.",
            param_hint="--cell",
        )

    creature = archive[coords]

    spawn = SpawnConfig(n=n, min_dist=min_dist, patch=patch, seed=seed)
    config = EcosystemConfig(
        source_run_id=run,
        source_coords=coords,
        grid_h=grid_h,
        grid_w=grid_w,
        steps=steps,
        snapshot_every=snapshot_every,
        spawn=spawn,
        device=device,
        border=border,
        output_format=output_format,
    )

    result = run_ecosystem(config, creature, output_root=output_dir)
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
