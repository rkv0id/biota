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
    timestamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    coord_str = f"{coords[0]}_{coords[1]}_{coords[2]}"
    parts = source_run_id.split("-")
    suffix = "-".join(parts[-2:]) if len(parts) >= 4 else source_run_id[:8]
    return f"{timestamp}-eco-{suffix}-{coord_str}"


def _colorize_frame(state: np.ndarray) -> np.ndarray:
    """Convert a (H, W) float32 state to (H, W, 3) uint8 magma-colorized RGB."""
    from biota.viz.colormap import apply_magma

    peak = float(np.percentile(state, 99.9))
    if peak > 0:
        normalized = (state / peak * 255).clip(0, 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(state, dtype=np.uint8)
    return apply_magma(normalized)  # (H, W, 3)


def _save_frame_png(state: np.ndarray, path: Path) -> None:
    """Save a (H, W) float32 state as a magma-colorized PNG."""
    try:
        import imageio.v3 as iio

        iio.imwrite(path, _colorize_frame(state))
    except Exception:
        pass  # Frame saving is best-effort; don't abort the run


def _write_gif(frames: list[np.ndarray], path: Path, fps: int = 10) -> None:
    """Write a list of (H, W, 3) uint8 RGB frames as an animated GIF.

    Downsamples to at most 256px on the longest side when larger, preserving
    aspect ratio via pixel striding (no extra dependencies).
    """
    try:
        import imageio.v3 as iio

        if not frames:
            return

        h, w = frames[0].shape[:2]
        max_dim = 256
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            step_h = max(1, int(1.0 / scale))
            step_w = max(1, int(1.0 / scale))
            frames = [f[::step_h, ::step_w] for f in frames]

        duration_ms = 1000 // fps
        iio.imwrite(path, frames, extension=".gif", loop=0, duration=duration_ms)
    except Exception as exc:
        print(f"  [warning] GIF write failed: {exc}")


def run_ecosystem(
    config: EcosystemConfig,
    creature: RolloutResult,
    output_root: Path | str = "ecosystem-runs",
    on_event: object = None,
) -> EcosystemResult:
    """Run one ecosystem simulation and write output to disk.

    Writes trajectory.npy (raw float32 snapshots) always.
    Writes ecosystem.gif when output_format='gif' (default).
    Writes frames/ directory of PNGs when output_format='frames'.
    """
    run_id = _make_eco_run_id(config.source_run_id, config.source_coords)
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if config.output_format == "frames":
        frames_dir = run_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
    else:
        frames_dir = run_dir / "frames"  # not created, used as fallback path only

    # Write config.json immediately so the run is identifiable even if it fails
    config_dict = {
        "source_run_id": config.source_run_id,
        "source_coords": list(config.source_coords),
        "grid_h": config.grid_h,
        "grid_w": config.grid_w,
        "steps": config.steps,
        "snapshot_every": config.snapshot_every,
        "device": config.device,
        "border": config.border,
        "output_format": config.output_format,
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
    sim_config = SimConfig(
        grid_h=config.grid_h,
        grid_w=config.grid_w,
        kernels=n_kernels,
        border=config.border,
    )
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

    state = build_initial_state(config.spawn, config.grid_h, config.grid_w, config.device)
    initial_mass = float(state.sum().item())

    raw_snapshots: list[np.ndarray] = []
    rgb_frames: list[np.ndarray] = []
    snapshot_steps: list[int] = []
    mass_history: list[float] = [initial_mass]

    started_at = time.time()

    for step in range(1, config.steps + 1):
        state = fl.step(state)
        mass = float(state.sum().item())
        mass_history.append(mass)

        if step % config.snapshot_every == 0 or step == config.steps:
            frame = state[:, :, 0].detach().cpu().numpy().astype(np.float32)
            raw_snapshots.append(frame)
            snapshot_steps.append(step)
            colored = _colorize_frame(frame)
            rgb_frames.append(colored)
            if config.output_format == "frames":
                _save_frame_png(frame, frames_dir / f"step_{step:06d}.png")

    elapsed = time.time() - started_at

    if config.output_format == "gif" and rgb_frames:
        _write_gif(rgb_frames, run_dir / "ecosystem.gif")

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

    if raw_snapshots:
        trajectory = np.stack(raw_snapshots, axis=0)
        np.save(run_dir / "trajectory.npy", trajectory)

    result = EcosystemResult(
        config=config,
        run_id=run_id,
        run_dir=str(run_dir),
        measures=measures,
        elapsed_seconds=elapsed,
    )
    (run_dir / "summary.json").write_text(json.dumps(result.to_summary_dict(), indent=2))

    print(
        f"[ecosystem] {run_id}: {config.spawn.n} creatures, "
        f"{config.steps} steps, {len(raw_snapshots)} snapshots "
        f"({config.output_format}), {elapsed:.1f}s"
    )
    return result
