"""Ecosystem simulation runner.

Takes an EcosystemConfig (already validated from YAML), loads each source's
archive creature, spawns them on a shared grid, runs the simulation, captures
snapshots and measures, writes output files, and returns an EcosystemResult.

Output directory structure:
    ecosystem/<run_id>/
        summary.json        run metadata and measures
        trajectory.npy      mass snapshots, shape (n_snapshots, H, W) float32
        frames/             PNG frames per snapshot step (frames mode only)
        config.json         the resolved EcosystemConfig serialized

Homogeneous runs (one source) use FlowLenia's scalar step path unchanged from
v2.0.x. Heterogeneous runs (two or more sources) use LocalizedFlowLenia with
species-indexed parameter localization: each species keeps its full param set,
and per-cell weights track which species owns the local mass.
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from biota.ecosystem.config import CreatureSource, EcosystemConfig
from biota.ecosystem.result import EcosystemMeasures, EcosystemResult
from biota.ecosystem.spawn import build_initial_state, build_initial_state_multi_species
from biota.search.archive import Archive
from biota.search.result import RolloutResult
from biota.sim.flowlenia import Config as SimConfig
from biota.sim.flowlenia import FlowLenia, Params
from biota.sim.localized import LocalizedFlowLenia, LocalizedState


def _make_run_id(name: str) -> str:
    """Build a run id from the experiment name plus a millisecond-precision stamp."""
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in name).strip("-")
    if not safe:
        safe = "experiment"
    return f"{stamp}-{safe}"


def load_creature(source: CreatureSource) -> RolloutResult:
    """Read one archive and extract the RolloutResult at the requested cell.

    Public utility shared between the sequential runner and the Ray
    dispatcher. The dispatcher calls this on the driver before submitting
    tasks so worker nodes do not need filesystem access to the archive.
    """
    run_dir = Path(source.archive_dir) / source.run_id
    if not run_dir.exists():
        raise FileNotFoundError(
            f"archive run directory not found: {run_dir} "
            f"(source: archive_dir={source.archive_dir!r}, run={source.run_id!r})"
        )
    pkl_path = run_dir / "archive.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"no archive.pkl in {run_dir}")

    with open(pkl_path, "rb") as f:
        loaded = pickle.load(f)

    if not isinstance(loaded, Archive):
        raise TypeError(f"archive.pkl in {run_dir} is a {type(loaded).__name__}, expected Archive")
    archive: Archive = loaded

    if source.coords not in archive:
        raise KeyError(
            f"cell {source.coords} not present in archive {source.run_id}. "
            f"Use build_index.py to see occupied cells."
        )
    return archive[source.coords]


def _params_from_creature(creature: RolloutResult, device: str) -> Params:
    """Build a Params on `device` from a RolloutResult's ParamDict."""
    p = creature.params
    return Params(
        R=p["R"],
        r=torch.tensor(p["r"], dtype=torch.float32, device=device),
        m=torch.tensor(p["m"], dtype=torch.float32, device=device),
        s=torch.tensor(p["s"], dtype=torch.float32, device=device),
        h=torch.tensor(p["h"], dtype=torch.float32, device=device),
        a=torch.tensor(p["a"], dtype=torch.float32, device=device),
        b=torch.tensor(p["b"], dtype=torch.float32, device=device),
        w=torch.tensor(p["w"], dtype=torch.float32, device=device),
    )


def _build_sim_config(creature: RolloutResult, cfg: EcosystemConfig) -> SimConfig:
    """Construct the FlowLenia Config for an experiment.

    The kernel count comes from the first creature; in heterogeneous runs all
    creatures must use the same kernel count (FlowLenia kernel tensors share
    the K dimension across species in our LocalizedFlowLenia implementation).
    The validation against subsequent creatures happens at sim build time.
    """
    return SimConfig(
        grid_h=cfg.grid_h,
        grid_w=cfg.grid_w,
        kernels=len(creature.params["r"]),
        border=cfg.border,
    )


def _colorize_frame(state: np.ndarray, global_peak: float | None = None) -> np.ndarray:
    """Convert a (H, W) float32 mass snapshot to (H, W, 3) magma-colorized RGB."""
    from biota.viz.colormap import apply_magma

    peak = global_peak if global_peak is not None else float(np.percentile(state, 99.9))
    if peak > 0:
        normalized = (state / peak * 255).clip(0, 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(state, dtype=np.uint8)
    return apply_magma(normalized)


def _render_frame_png_bytes(state: np.ndarray) -> bytes | None:
    """Render a (H, W) float32 state to PNG bytes. Returns None on failure."""
    try:
        import io

        import imageio.v3 as iio

        buf = io.BytesIO()
        iio.imwrite(buf, _colorize_frame(state), extension=".png")
        return buf.getvalue()
    except Exception:
        return None


def _render_gif_bytes(frames: list[np.ndarray], fps: int = 10) -> bytes | None:
    """Render frames to GIF bytes. Returns None on failure or empty input.

    Pure compute: no I/O. The materialize step writes the bytes to disk.
    Same downsampling rule as the original on-disk writer (max 256px on the
    long axis) so worker-rendered output is bit-identical to local output.
    """
    try:
        import io

        import imageio.v3 as iio

        if not frames:
            return None

        h, w = frames[0].shape[:2]
        max_dim = 256
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            step_h = max(1, int(1.0 / scale))
            step_w = max(1, int(1.0 / scale))
            frames = [f[::step_h, ::step_w] for f in frames]

        duration_ms = 1000 // fps
        buf = io.BytesIO()
        iio.imwrite(buf, frames, extension=".gif", loop=0, duration=duration_ms)
        return buf.getvalue()
    except Exception as exc:
        print(f"  [warning] GIF render failed: {exc}")
        return None


def _compute_outputs(
    config: EcosystemConfig,
    run_id: str,
    run_dir: Path,
    raw_snapshots: list[np.ndarray],
    snapshot_steps: list[int],
    mass_history: list[float],
    initial_mass: float,
    elapsed: float,
) -> tuple[EcosystemResult, dict[str, bytes]]:
    """Compute the EcosystemResult and the artifact bytes dict.

    Pure compute: no I/O. The returned dict maps relative filename (under
    run_dir) to bytes ready to write. Used by both the local sequential path
    (which calls materialize_outputs immediately) and the cluster Ray path
    (which ships the dict back to the driver and materializes there).

    Per-frame PNG rendering for the 'frames' output_format is done here
    rather than streamed during the simulation loop, so cluster runs ship
    one bundled artifact dict rather than relying on per-step disk writes.
    """
    artifacts: dict[str, bytes] = {}

    rgb_frames: list[np.ndarray] = []
    if raw_snapshots:
        global_peak = float(np.percentile(np.stack(raw_snapshots), 99.9))
        rgb_frames = [_colorize_frame(f, global_peak) for f in raw_snapshots]

    if config.output_format == "gif" and rgb_frames:
        gif_bytes = _render_gif_bytes(rgb_frames)
        if gif_bytes is not None:
            artifacts["ecosystem.gif"] = gif_bytes

    if config.output_format == "frames":
        for frame, step in zip(raw_snapshots, snapshot_steps, strict=True):
            png_bytes = _render_frame_png_bytes(frame)
            if png_bytes is not None:
                artifacts[f"frames/step_{step:06d}.png"] = png_bytes

    mass_arr = np.array(mass_history, dtype=np.float32)
    final_mass = mass_history[-1] if mass_history else 0.0
    peak_mass = float(mass_arr.max()) if mass_arr.size else 0.0
    min_mass = float(mass_arr.min()) if mass_arr.size else 0.0

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
        import io

        trajectory = np.stack(raw_snapshots, axis=0)
        buf = io.BytesIO()
        np.save(buf, trajectory)
        artifacts["trajectory.npy"] = buf.getvalue()

    result = EcosystemResult(
        config=config,
        run_id=run_id,
        run_dir=str(run_dir),
        measures=measures,
        elapsed_seconds=elapsed,
    )
    artifacts["summary.json"] = json.dumps(result.to_summary_dict(), indent=2).encode("utf-8")
    return result, artifacts


def materialize_outputs(run_dir: Path, artifacts: dict[str, bytes]) -> None:
    """Write artifacts dict to disk under run_dir. Creates subdirs as needed.

    Public utility shared between the sequential runner (which calls
    compute_ecosystem then materialize_outputs in one go via run_ecosystem)
    and the Ray dispatcher (which calls compute_ecosystem on the worker,
    ships the bytes back to the driver, and materializes there). artifact
    keys are relative paths under run_dir.
    """
    for relpath, data in artifacts.items():
        target = run_dir / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def _config_json_bytes(config: EcosystemConfig) -> bytes:
    """Serialize the resolved config to JSON bytes (no I/O)."""
    mode = "heterogeneous" if config.is_heterogeneous else "homogeneous"
    config_dict: dict[str, object] = {
        "name": config.name,
        "grid_h": config.grid_h,
        "grid_w": config.grid_w,
        "steps": config.steps,
        "snapshot_every": config.snapshot_every,
        "device": config.device,
        "border": config.border,
        "output_format": config.output_format,
        "mode": mode,
        "sources": [
            {
                "archive_dir": s.archive_dir,
                "run": s.run_id,
                "cell": list(s.coords),
                "n": s.n,
                "patch": s.patch if s.patch is not None else config.spawn.patch,
            }
            for s in config.sources
        ],
        "spawn": {
            "min_dist": config.spawn.min_dist,
            "patch": config.spawn.patch,
            "seed": config.spawn.seed,
        },
    }
    return json.dumps(config_dict, indent=2).encode("utf-8")


def _write_config_json(config: EcosystemConfig, run_dir: Path) -> None:
    """Materialize config.json under run_dir."""
    (run_dir / "config.json").write_bytes(_config_json_bytes(config))


def _run_homogeneous(
    config: EcosystemConfig,
    creatures: list[RolloutResult] | None = None,
) -> tuple[list[np.ndarray], list[int], list[float], float]:
    """Scalar-path simulation loop. Returns (snapshots, snapshot_steps, mass_history, initial_mass)."""
    source = config.sources[0]
    creature = creatures[0] if creatures is not None else load_creature(source)
    sim_cfg = _build_sim_config(creature, config)
    sim_params = _params_from_creature(creature, config.device)
    fl = FlowLenia(sim_cfg, sim_params, device=config.device)

    state = build_initial_state(
        config.spawn,
        source.n,
        config.grid_h,
        config.grid_w,
        config.device,
        patch_override=source.patch,
    )
    initial_mass = float(state.sum().item())

    snapshots: list[np.ndarray] = []
    steps_taken: list[int] = []
    mass_history: list[float] = [initial_mass]

    for step in range(1, config.steps + 1):
        state = fl.step(state)
        mass_history.append(float(state.sum().item()))

        if step % config.snapshot_every == 0 or step == config.steps:
            frame = state[:, :, 0].detach().cpu().numpy().astype(np.float32)
            snapshots.append(frame)
            steps_taken.append(step)

    return snapshots, steps_taken, mass_history, initial_mass


def _run_heterogeneous(
    config: EcosystemConfig,
    creatures: list[RolloutResult] | None = None,
) -> tuple[list[np.ndarray], list[int], list[float], float]:
    """Species-indexed simulation loop via LocalizedFlowLenia."""
    if creatures is None:
        creatures = [load_creature(s) for s in config.sources]

    # All species must share the same kernel count: LocalizedFlowLenia's per-step
    # FFT path expects each species' fK to have the same K dim as the others.
    # The Plantec convention is one global K; biota's species-indexed extension
    # keeps each species' own kernels but they still need to be parallel-shaped.
    k_counts = [len(c.params["r"]) for c in creatures]
    if len(set(k_counts)) != 1:
        raise ValueError(
            f"all sources in a heterogeneous experiment must have the same kernel "
            f"count, got {k_counts} for sources {[s.run_id for s in config.sources]}"
        )

    sim_cfg = _build_sim_config(creatures[0], config)
    species_params = [_params_from_creature(c, config.device) for c in creatures]
    lfl = LocalizedFlowLenia(sim_cfg, species_params, device=config.device)

    counts = [s.n for s in config.sources]
    # Per-species patch: source override if set, otherwise the experiment default.
    patches = [s.patch if s.patch is not None else config.spawn.patch for s in config.sources]
    mass, weights = build_initial_state_multi_species(
        config.spawn,
        counts,
        config.grid_h,
        config.grid_w,
        config.device,
        patches=patches,
    )
    state = LocalizedState(mass=mass, weights=weights)
    initial_mass = float(state.mass.sum().item())

    snapshots: list[np.ndarray] = []
    steps_taken: list[int] = []
    mass_history: list[float] = [initial_mass]

    for step in range(1, config.steps + 1):
        state = lfl.step(state)
        mass_history.append(float(state.mass.sum().item()))

        if step % config.snapshot_every == 0 or step == config.steps:
            frame = state.mass[:, :, 0].detach().cpu().numpy().astype(np.float32)
            snapshots.append(frame)
            steps_taken.append(step)

    return snapshots, steps_taken, mass_history, initial_mass


def run_ecosystem(
    config: EcosystemConfig,
    output_root: Path | str = "ecosystem",
    creatures: list[RolloutResult] | None = None,
) -> EcosystemResult:
    """Run one ecosystem simulation and write output to disk.

    Homogeneous (one source): scalar FlowLenia step path, identical to v2.0.x.
    Heterogeneous (two or more sources): species-indexed LocalizedFlowLenia.

    Sequential CLI entry point. The Ray dispatch path uses compute_ecosystem
    instead, which separates simulation+rendering from disk materialization
    so workers can ship artifact bytes back to the driver instead of writing
    to a worker-local filesystem the driver has no access to.

    creatures, when provided, must be a list of RolloutResult objects in the
    same order as config.sources. The runner uses these directly instead of
    reading the archive from disk. When None, creatures are loaded from disk
    via load_creature.
    """
    result, artifacts = compute_ecosystem(config, output_root, creatures=creatures)
    run_dir = Path(result.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_config_json(config, run_dir)
    materialize_outputs(run_dir, artifacts)
    _print_run_summary(config, result, len(result.measures.snapshot_steps))
    return result


def compute_ecosystem(
    config: EcosystemConfig,
    output_root: Path | str = "ecosystem",
    creatures: list[RolloutResult] | None = None,
) -> tuple[EcosystemResult, dict[str, bytes]]:
    """Run one ecosystem simulation and return (result, artifacts) without
    materializing to disk.

    Used by the Ray dispatcher: the worker calls this, ships the
    (result, artifacts) pair back to the driver via Ray's ObjectStore, and
    the driver writes the artifacts to its own output_root.

    The returned artifacts dict maps relative filename (under run_dir) to
    bytes. It includes config.json, summary.json, the GIF or per-frame
    PNGs depending on output_format, and trajectory.npy. The driver
    materializes by writing each entry under its local run_dir.

    The returned EcosystemResult.run_dir holds a *prospective* path
    (output_root / run_id). The driver may rewrite this to its local path
    when materializing if its output_root differs from the worker's.
    """
    if creatures is not None and len(creatures) != len(config.sources):
        raise ValueError(
            f"creatures length {len(creatures)} does not match sources length {len(config.sources)}"
        )

    run_id = _make_run_id(config.name)
    run_dir = Path(output_root) / run_id

    started_at = time.time()
    if config.is_heterogeneous:
        snapshots, steps_taken, mass_history, initial_mass = _run_heterogeneous(
            config, creatures=creatures
        )
    else:
        snapshots, steps_taken, mass_history, initial_mass = _run_homogeneous(
            config, creatures=creatures
        )
    elapsed = time.time() - started_at

    result, artifacts = _compute_outputs(
        config=config,
        run_id=run_id,
        run_dir=run_dir,
        raw_snapshots=snapshots,
        snapshot_steps=steps_taken,
        mass_history=mass_history,
        initial_mass=initial_mass,
        elapsed=elapsed,
    )
    # config.json travels in the same artifact bundle so the driver
    # materializes it alongside everything else.
    artifacts["config.json"] = _config_json_bytes(config)
    return result, artifacts


def _print_run_summary(config: EcosystemConfig, result: EcosystemResult, n_snapshots: int) -> None:
    """Stdout one-liner summarizing a completed run."""
    mode = "heterogeneous" if config.is_heterogeneous else "homogeneous"
    n_creatures = sum(s.n for s in config.sources)
    print(
        f"[ecosystem] {result.run_id}: {mode}, "
        f"{n_creatures} creatures across {len(config.sources)} "
        f"species, {config.steps} steps, {n_snapshots} snapshots "
        f"({config.output_format}), {result.elapsed_seconds:.1f}s"
    )
