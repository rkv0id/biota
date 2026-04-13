"""Ecosystem simulation runner.

Takes an EcosystemConfig and a RolloutResult (the source creature from the
archive), spawns N copies on a shared grid, runs the simulation, captures
snapshots and measures, writes output files, and returns an EcosystemResult.

Output directory structure:
    ecosystem-runs/<run_id>/
        summary.json        run metadata and measures
        trajectory.npy      state snapshots, shape (n_snapshots, H, W) float32
        frames/             PNG frames for every snapshot step
        config.json         full EcosystemConfig serialized

The simulation uses FlowLenia in single-step mode (not batched) since there
is only one state to evolve. The creature's params are used globally across
the entire grid.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from biota.ecosystem.result import (
    EcosystemConfig,
    EcosystemMeasures,
    EcosystemResult,
)
from biota.ecosystem.spawn import build_initial_state
from biota.search.result import RolloutResult
from biota.sim.flowlenia import Config as SimConfig
from biota.sim.flowlenia import FlowLenia, Params


def _make_eco_run_id(source_run_id: str, coords: tuple[int, int, int]) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    coord_str = f"{coords[0]}_{coords[1]}_{coords[2]}"
    # Take the adjective-noun suffix from the source run id if it matches
    # the standard format (timestamp-adj-noun), otherwise use first 8 chars
    parts = source_run_id.split("-")
    suffix = "-".join(parts[-2:]) if len(parts) >= 4 else source_run_id[:8]
    return f"{timestamp}-eco-{suffix}-{coord_str}"


def _save_frame_png(state: np.ndarray, path: Path) -> None:
    """Save a (H, W) float32 state as a magma-colorized PNG."""
    try:
        import imageio.v3 as iio

        from biota.viz.colormap import apply_magma

        # Use 99th percentile as the normalization ceiling so the creature
        # body maps to the full magma range rather than just the dark end.
        # On a large ecosystem grid the vast majority of pixels are near-zero,
        # so normalizing by max() compresses all creature structure into the
        # bottom few entries of the colormap.
        peak = float(np.percentile(state, 99.9))
        if peak > 0:
            normalized = (state / peak * 255).clip(0, 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(state, dtype=np.uint8)

        colored = apply_magma(normalized)  # (H, W, 3) RGB
        iio.imwrite(path, colored)
    except Exception:
        pass  # Frame saving is best-effort; don't abort the run


def run_ecosystem(
    config: EcosystemConfig,
    creature: RolloutResult,
    output_root: Path | str = "ecosystem-runs",
    on_event: object = None,
) -> EcosystemResult:
    """Run one ecosystem simulation and write output to disk.

    Args:
        config:      Complete ecosystem configuration.
        creature:    The archive RolloutResult to use as the creature source.
                     Its params are used globally across the ecosystem grid.
        output_root: Root directory for ecosystem run output.
        on_event:    Optional callback for progress events (reserved for v2.1).

    Returns:
        EcosystemResult with measures and paths to output files.
    """
    run_id = _make_eco_run_id(config.source_run_id, config.source_coords)
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = run_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    # Write config.json immediately so the run is identifiable even if it fails
    config_dict = {
        "source_run_id": config.source_run_id,
        "source_coords": list(config.source_coords),
        "grid": config.grid,
        "steps": config.steps,
        "snapshot_every": config.snapshot_every,
        "device": config.device,
        "spawn": {
            "n": config.spawn.n,
            "min_dist": config.spawn.min_dist,
            "patch": config.spawn.patch,
            "seed": config.spawn.seed,
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config_dict, indent=2))

    # Build sim using the creature's native kernel count
    n_kernels = len(creature.params["r"])
    sim_config = SimConfig(grid=config.grid, kernels=n_kernels)
    p = creature.params
    sim_params = Params(
        R=p["R"],
        r=torch.tensor(p["r"], dtype=torch.float32, device=config.device),
        m=torch.tensor(p["m"], dtype=torch.float32, device=config.device),
        s=torch.tensor(p["s"], dtype=torch.float32, device=config.device),
        h=torch.tensor(p["h"], dtype=torch.float32, device=config.device),
        a=torch.tensor(p["a"], dtype=torch.float32, device=config.device),
        b=torch.tensor(p["b"], dtype=torch.float32, device=config.device),
        w=torch.tensor(p["w"], dtype=torch.float32, device=config.device),
    )
    fl = FlowLenia(sim_config, sim_params, device=config.device)

    # Build initial state
    state = build_initial_state(config.spawn, config.grid, config.device)
    initial_mass = float(state.sum().item())

    # Simulation loop
    snapshots: list[np.ndarray] = []
    snapshot_steps: list[int] = []
    mass_history: list[float] = [initial_mass]

    started_at = time.time()

    for step in range(1, config.steps + 1):
        state = fl.step(state)
        mass = float(state.sum().item())
        mass_history.append(mass)

        if step % config.snapshot_every == 0 or step == config.steps:
            frame = state[:, :, 0].detach().cpu().numpy().astype(np.float32)
            snapshots.append(frame)
            snapshot_steps.append(step)
            _save_frame_png(frame, frames_dir / f"step_{step:06d}.png")

    elapsed = time.time() - started_at

    # Compute measures
    mass_arr = np.array(mass_history, dtype=np.float32)
    final_mass = mass_history[-1]
    peak_mass = float(mass_arr.max())
    min_mass = float(mass_arr.min())

    if initial_mass > 0 and len(mass_history) > 1:
        diffs = np.abs(np.diff(mass_arr))
        mass_turnover = float(diffs.mean() / initial_mass)
    else:
        mass_turnover = 0.0

    measures = EcosystemMeasures(
        initial_mass=initial_mass,
        final_mass=final_mass,
        mass_history=mass_history,
        peak_mass=peak_mass,
        min_mass=min_mass,
        mass_turnover=mass_turnover,
        snapshot_steps=snapshot_steps,
    )

    # Write trajectory.npy
    if snapshots:
        trajectory = np.stack(snapshots, axis=0)  # (n_snapshots, H, W)
        np.save(run_dir / "trajectory.npy", trajectory)

    # Write summary.json
    result = EcosystemResult(
        config=config,
        run_id=run_id,
        run_dir=str(run_dir),
        measures=measures,
        elapsed_seconds=elapsed,
    )
    (run_dir / "summary.json").write_text(json.dumps(result.to_summary_dict(), indent=2))

    n_creatures = config.spawn.n
    n_snapshots = len(snapshots)
    print(
        f"[ecosystem] {run_id}: {n_creatures} creatures, "
        f"{config.steps} steps, {n_snapshots} snapshots, "
        f"{elapsed:.1f}s"
    )

    return result
