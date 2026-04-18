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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from biota.ecosystem.analytics import (
    HeteroSpatial,
    HomoSpatial,
    compute_signal_observables,
    compute_signal_observables_hetero,
    compute_spatial_observables_hetero,
    compute_spatial_observables_homo,
)
from biota.ecosystem.config import CreatureSource, EcosystemConfig
from biota.ecosystem.interaction import (
    OutcomeSequence,
    classify_outcome_hetero,
    classify_outcome_homo,
    compute_interaction_coefficients,
)
from biota.ecosystem.result import EcosystemMeasures, EcosystemResult
from biota.ecosystem.spawn import build_initial_state, build_initial_state_multi_species
from biota.search.archive import Archive
from biota.search.result import RolloutResult
from biota.sim.flowlenia import Config as SimConfig
from biota.sim.flowlenia import FlowLenia, Params
from biota.sim.localized import LocalizedFlowLenia, LocalizedState


@dataclass
class SimOutput:
    """Raw output from one simulation loop (homo or hetero).

    Uniform shape so _compute_outputs doesn't need to know which path ran.
    For homogeneous runs, ownership_snapshots is empty and
    species_mass_history has one row equal to total mass_history.
    """

    snapshots: list[np.ndarray]  # (H, W) float32 mass at snapshot steps
    snapshot_steps: list[int]
    mass_history: list[float]  # total mass per step, length = steps + 1
    initial_mass: float
    n_species: int
    species_mass_history: list[list[float]] = field(default_factory=list)
    # Effective area per species at each step: sum of ownership weights
    # across all cells. Measures territorial extent. For homo runs, has one
    # entry that tracks the number of cells with nonzero mass.
    species_territory_history: list[list[float]] = field(default_factory=list)
    # (H, W, S) float32 ownership at each snapshot step; empty for homo runs
    ownership_snapshots: list[np.ndarray] = field(default_factory=list)
    # Per-species growth fields at snapshot steps. Outer list indexed by
    # snapshot, inner list indexed by species: growth_snapshots[snap][s] is
    # the (H, W) float32 G_s_total for species s before ownership blending.
    # Empty for homo runs. Used to compute empirical interaction coefficients.
    growth_snapshots: list[list[np.ndarray]] = field(default_factory=list)
    # Signal field total mass per step (parallel to mass_history). Empty when
    # no signal field is active.
    signal_total_history: list[float] = field(default_factory=list)
    # Mean signal per channel at each snapshot step. Shape: (n_snapshots, C).
    # Empty when no signal field is active.
    signal_channel_snapshots: list[list[float]] = field(default_factory=list)
    # Per-species mean signal received per snapshot. Shape: (S, n_snapshots, C).
    # Computed as mean signal per channel over species s territory. Empty for
    # non-signal or homo runs.
    species_signal_received: list[list[list[float]]] = field(default_factory=list)
    # Downsampled spatial signal sum at each snapshot. Shape: (n_snapshots, H//4, W//4).
    # signal.sum(dim=-1) downsampled 4x -- total signal magnitude per spatial cell.
    # Used to generate the signal GIF. Empty when no signal field is active.
    signal_sum_snapshots: list[np.ndarray] = field(default_factory=list)


# Perceptually distinct hues for species coloring. Ordered so the first
# few are maximally distinguishable. When ownership blends two species
# the result is a weighted mix of these RGB triples.
SPECIES_PALETTE: list[tuple[int, int, int]] = [
    (255, 140, 50),  # warm orange
    (80, 180, 255),  # sky blue
    (180, 255, 80),  # lime green
    (255, 90, 180),  # hot pink
    (140, 100, 255),  # purple
    (255, 220, 60),  # gold
    (60, 220, 200),  # teal
    (255, 100, 100),  # coral
]


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


def validate_signal_consistency(creatures: list[RolloutResult]) -> None:
    """Raise if creatures mix signal-enabled and non-signal archives.

    All creatures in an ecosystem run must have the same signal field status.
    A mixed run is physically incoherent: the signal field either exists for
    all species or for none.
    """
    from biota.search.params import has_signal_field

    has_signal = [has_signal_field(c.params) for c in creatures]
    if any(has_signal) and not all(has_signal):
        non_signal_indices = [i for i, h in enumerate(has_signal) if not h]
        raise ValueError(
            f"ecosystem run mixes signal-enabled and non-signal creatures. "
            f"All sources must come from the same archive type. "
            f"Non-signal species at source indices: {non_signal_indices}. "
            f"Re-run with all sources from signal-enabled archives, or all from standard archives."
        )


def _params_from_creature(creature: RolloutResult, device: str) -> Params:
    """Build a Params on `device` from a RolloutResult's ParamDict."""
    from biota.search.params import has_signal_field

    p = creature.params
    base = Params(
        R=p["R"],
        r=torch.tensor(p["r"], dtype=torch.float32, device=device),
        m=torch.tensor(p["m"], dtype=torch.float32, device=device),
        s=torch.tensor(p["s"], dtype=torch.float32, device=device),
        h=torch.tensor(p["h"], dtype=torch.float32, device=device),
        a=torch.tensor(p["a"], dtype=torch.float32, device=device),
        b=torch.tensor(p["b"], dtype=torch.float32, device=device),
        w=torch.tensor(p["w"], dtype=torch.float32, device=device),
    )
    if not has_signal_field(p):
        return base
    return Params(
        R=base.R,
        r=base.r,
        m=base.m,
        s=base.s,
        h=base.h,
        a=base.a,
        b=base.b,
        w=base.w,
        emission_vector=torch.tensor(p["emission_vector"], dtype=torch.float32, device=device),  # type: ignore[typeddict-item]
        receptor_profile=torch.tensor(p["receptor_profile"], dtype=torch.float32, device=device),  # type: ignore[typeddict-item]
        emission_rate=float(p["emission_rate"]),  # type: ignore[typeddict-item]
        decay_rates=torch.tensor(p["decay_rates"], dtype=torch.float32, device=device),  # type: ignore[typeddict-item]
        alpha_coupling=float(p["alpha_coupling"]),  # type: ignore[typeddict-item]
        beta_modulation=float(p["beta_modulation"]),  # type: ignore[typeddict-item]
        signal_kernel_r=float(p["signal_kernel_r"]),  # type: ignore[typeddict-item]
        signal_kernel_a=torch.tensor(p["signal_kernel_a"], dtype=torch.float32, device=device),  # type: ignore[typeddict-item]
        signal_kernel_b=torch.tensor(p["signal_kernel_b"], dtype=torch.float32, device=device),  # type: ignore[typeddict-item]
        signal_kernel_w=torch.tensor(p["signal_kernel_w"], dtype=torch.float32, device=device),  # type: ignore[typeddict-item]
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


def _colorize_frame_species(
    mass: np.ndarray,
    ownership: np.ndarray,
    global_peak: float,
) -> np.ndarray:
    """Colorize a frame by species ownership.

    mass:      (H, W) float32 mass density.
    ownership: (H, W, S) float32 species ownership weights (simplex).
    global_peak: normalization ceiling for mass intensity.

    Returns (H, W, 3) uint8 RGB. Each pixel's hue is a weighted blend
    of the species palette colors by ownership, and brightness scales
    with mass intensity. Background (no mass) is black.
    """
    h, w = mass.shape
    n_species = ownership.shape[2]
    intensity = np.clip(mass / max(global_peak, 1e-8), 0.0, 1.0)  # (H, W)

    # Build per-pixel RGB by blending palette colors by ownership weight
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for s in range(n_species):
        color_idx = s % len(SPECIES_PALETTE)
        r, g, b = SPECIES_PALETTE[color_idx]
        weight = ownership[:, :, s]  # (H, W)
        rgb[:, :, 0] += weight * r
        rgb[:, :, 1] += weight * g
        rgb[:, :, 2] += weight * b

    # Scale by mass intensity
    rgb *= intensity[:, :, np.newaxis]
    return rgb.clip(0, 255).astype(np.uint8)


def _colorize_signal_frame(
    signal_sum: np.ndarray,
    global_peak: float,
    ownership: np.ndarray | None = None,
    n_species: int = 1,
) -> np.ndarray:
    """Colorize a downsampled (H, W) signal-sum frame.

    For homogeneous runs: teal colormap scaled by signal magnitude.
    For heterogeneous runs: species-colored by upsampled ownership blend.
    Returns (H, W, 3) uint8 RGB.
    """
    h, w = signal_sum.shape
    intensity = np.clip(signal_sum / max(global_peak, 1e-8), 0.0, 1.0)

    if ownership is not None and n_species > 1:
        # Downsample ownership to match signal_sum resolution.
        own_ds = ownership[::4, ::4, :]  # (H//4, W//4, S) approx
        own_ds = own_ds[:h, :w, :]  # clip to exact size
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        for s in range(min(n_species, own_ds.shape[2])):
            color_idx = s % len(SPECIES_PALETTE)
            r, g, b = SPECIES_PALETTE[color_idx]
            weight = own_ds[:, :, s]
            rgb[:, :, 0] += weight * r
            rgb[:, :, 1] += weight * g
            rgb[:, :, 2] += weight * b
        rgb *= intensity[:, :, np.newaxis]
    else:
        # Teal colormap: map intensity to teal (0,180,160) -> white (255,255,255)
        t = intensity
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        rgb[:, :, 0] = t * 80
        rgb[:, :, 1] = t * 220
        rgb[:, :, 2] = t * 200

    return rgb.clip(0, 255).astype(np.uint8)


def _render_rgb_png_bytes(rgb: np.ndarray) -> bytes | None:
    """Render an already-colorized (H, W, 3) uint8 frame to PNG bytes."""
    try:
        import io

        import imageio.v3 as iio

        buf = io.BytesIO()
        iio.imwrite(buf, rgb, extension=".png")
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
    sim: SimOutput,
    elapsed: float,
    creatures: list[RolloutResult] | None = None,
) -> tuple[EcosystemResult, dict[str, bytes]]:
    """Compute the EcosystemResult and the artifact bytes dict.

    Pure compute: no I/O. The returned dict maps relative filename (under
    run_dir) to bytes ready to write. Used by both the local sequential path
    (which calls materialize_outputs immediately) and the cluster Ray path
    (which ships the dict back to the driver and materializes there).

    Per-frame PNG rendering for the 'frames' output_format is done here
    rather than streamed during the simulation loop, so cluster runs ship
    one bundled artifact dict rather than relying on per-step disk writes.

    For heterogeneous runs with ownership data, frames are species-colored
    (each species gets a distinct hue, blended by ownership weight, scaled
    by mass intensity). For homogeneous runs, the standard magma colormap
    is used.
    """
    artifacts: dict[str, bytes] = {}
    has_ownership = len(sim.ownership_snapshots) > 0

    global_peak = 0.0
    if sim.snapshots:
        global_peak = float(np.percentile(np.stack(sim.snapshots), 99.9))

    # Build RGB frames: species-colored when ownership is available, magma otherwise.
    rgb_frames: list[np.ndarray] = []
    if sim.snapshots:
        if has_ownership:
            rgb_frames = [
                _colorize_frame_species(mass, own, global_peak)
                for mass, own in zip(sim.snapshots, sim.ownership_snapshots, strict=True)
            ]
        else:
            rgb_frames = [_colorize_frame(f, global_peak) for f in sim.snapshots]

    if config.output_format == "gif" and rgb_frames:
        gif_bytes = _render_gif_bytes(rgb_frames)
        if gif_bytes is not None:
            artifacts["ecosystem.gif"] = gif_bytes

    # Signal GIF: species-colored signal density, downsampled 4x.
    if config.output_format == "gif" and sim.signal_sum_snapshots:
        sig_peak = (
            float(np.percentile(np.stack(sim.signal_sum_snapshots), 99.5))
            if sim.signal_sum_snapshots
            else 1.0
        )
        if sig_peak <= 0:
            sig_peak = 1.0
        # Upsample each downsampled frame back to match mass frame size for
        # consistent GIF dimensions -- repeat pixels 4x.
        h_full, w_full = sim.snapshots[0].shape if sim.snapshots else (64, 64)
        sig_rgb_frames: list[np.ndarray] = []
        for i, sig_ds in enumerate(sim.signal_sum_snapshots):
            own = sim.ownership_snapshots[i] if sim.ownership_snapshots else None
            rgb_ds = _colorize_signal_frame(
                sig_ds, sig_peak, ownership=own, n_species=sim.n_species
            )
            # Upsample by 4x using repeat to match mass GIF dimensions.
            rgb_up = np.repeat(np.repeat(rgb_ds, 4, axis=0), 4, axis=1)
            rgb_up = rgb_up[:h_full, :w_full]
            sig_rgb_frames.append(rgb_up)
        sig_gif = _render_gif_bytes(sig_rgb_frames)
        if sig_gif is not None:
            artifacts["signal.gif"] = sig_gif

    if config.output_format == "frames":
        for i, (frame, step) in enumerate(zip(sim.snapshots, sim.snapshot_steps, strict=True)):
            if has_ownership:
                rgb = _colorize_frame_species(frame, sim.ownership_snapshots[i], global_peak)
            else:
                rgb = _colorize_frame(frame)
            png_bytes = _render_rgb_png_bytes(rgb)
            if png_bytes is not None:
                artifacts[f"frames/step_{step:06d}.png"] = png_bytes

    mass_arr = np.array(sim.mass_history, dtype=np.float32)
    final_mass = sim.mass_history[-1] if sim.mass_history else 0.0
    peak_mass = float(mass_arr.max()) if mass_arr.size else 0.0
    min_mass = float(mass_arr.min()) if mass_arr.size else 0.0

    if sim.initial_mass > 0 and len(sim.mass_history) > 1:
        diffs = np.abs(np.diff(mass_arr))
        mass_turnover = float(diffs.mean() / sim.initial_mass)
    else:
        mass_turnover = 0.0

    if has_ownership:
        hs: HeteroSpatial = compute_spatial_observables_hetero(sim.ownership_snapshots)
        interaction_coefficients = compute_interaction_coefficients(
            sim.ownership_snapshots,
            sim.growth_snapshots,
            interface_area=hs.species_interface_area,
        )
        outcome_seq: OutcomeSequence = classify_outcome_hetero(
            sim.species_territory_history,
            sim.ownership_snapshots,
            sim.snapshot_steps,
            hs.species_patch_count,
        )
        ho: HomoSpatial | None = None
    else:
        hs = HeteroSpatial()
        interaction_coefficients = []
        ho = compute_spatial_observables_homo(sim.snapshots)
        outcome_seq = classify_outcome_homo(
            sim.snapshot_steps,
            ho.patch_count_history,
            ho.patch_size_history,
            ho.initial_patch_sizes,
        )

    # Signal observables (empty when no signal field was active).
    sig_obs = compute_signal_observables(
        sim.signal_total_history,
        sim.mass_history,
        sim.signal_channel_snapshots,
    )
    # Hetero-specific signal observables: receptor alignment + emission-reception matrix.
    # Extract per-species emission_vector and receptor_profile from creatures when available.
    emission_vectors: list[list[float]] | None = None
    receptor_profiles: list[list[float]] | None = None
    if creatures is not None and has_ownership:
        from biota.search.params import has_signal_field

        if all(has_signal_field(c.params) for c in creatures):
            emission_vectors = [c.params["emission_vector"] for c in creatures]  # type: ignore[typeddict-item]
            receptor_profiles = [c.params["receptor_profile"] for c in creatures]  # type: ignore[typeddict-item]
    sig_obs_hetero = compute_signal_observables_hetero(
        sim.species_signal_received,
        species_params_emission_vector=emission_vectors,
        species_params_receptor_profile=receptor_profiles,
    )

    outcome_sequence = [
        [{"label": w.label, "from": w.from_step, "to": w.to_step} for w in series]
        for series in outcome_seq.series
    ]

    measures = EcosystemMeasures(
        initial_mass=sim.initial_mass,
        final_mass=final_mass,
        mass_history=sim.mass_history,
        peak_mass=peak_mass,
        min_mass=min_mass,
        mass_turnover=mass_turnover,
        snapshot_steps=sim.snapshot_steps,
        species_mass_history=sim.species_mass_history,
        species_territory_history=sim.species_territory_history,
        interaction_coefficients=interaction_coefficients,
        outcome_label=outcome_seq.final_label,
        outcome_sequence=outcome_sequence,
        species_patch_count=hs.species_patch_count,
        species_interface_area=hs.species_interface_area,
        species_com_distance=hs.species_com_distance,
        species_spatial_entropy=hs.species_spatial_entropy,
        contact_occurred=hs.contact_occurred,
        patch_count_history=ho.patch_count_history if ho else [],
        mass_spatial_entropy_history=ho.mass_spatial_entropy_history if ho else [],
        initial_patch_sizes=ho.initial_patch_sizes if ho else [],
        patch_size_history=ho.patch_size_history if ho else [],
        signal_total_history=sig_obs["signal_total_history"],  # type: ignore[arg-type]
        signal_mass_fraction=sig_obs["signal_mass_fraction"],  # type: ignore[arg-type]
        signal_channel_snapshots=sig_obs["signal_channel_snapshots"],  # type: ignore[arg-type]
        dominant_channel_history=sig_obs["dominant_channel_history"],  # type: ignore[arg-type]
        receptor_alignment=sig_obs_hetero["receptor_alignment"],  # type: ignore[arg-type]
        emission_reception_matrix=sig_obs_hetero["emission_reception_matrix"],  # type: ignore[arg-type]
    )

    if sim.snapshots:
        import io

        trajectory = np.stack(sim.snapshots, axis=0)
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
) -> SimOutput:
    """Scalar-path simulation loop. Returns SimOutput with n_species=1."""
    source = config.sources[0]
    creature = creatures[0] if creatures is not None else load_creature(source)
    validate_signal_consistency([creature])
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

    # Initialize signal field when creature has signal params.
    signal: torch.Tensor | None = None
    if sim_params.has_signal:
        signal = fl.make_initial_signal_field(seed=config.spawn.seed or 0)

    snapshots: list[np.ndarray] = []
    steps_taken: list[int] = []
    mass_history: list[float] = [initial_mass]
    signal_total_history: list[float] = [float(signal.sum().item()) if signal is not None else 0.0]
    signal_channel_snapshots: list[list[float]] = []
    signal_sum_snapshots: list[np.ndarray] = []

    for step in range(1, config.steps + 1):
        state, signal = fl.step(state, signal)
        mass_history.append(float(state.sum().item()))
        signal_total_history.append(float(signal.sum().item()) if signal is not None else 0.0)

        if step % config.snapshot_every == 0 or step == config.steps:
            frame = state[:, :, 0].detach().cpu().numpy().astype(np.float32)
            snapshots.append(frame)
            steps_taken.append(step)
            if signal is not None:
                ch_mean = signal.mean(dim=(0, 1)).detach().cpu().tolist()
                signal_channel_snapshots.append(ch_mean)
                sig_ds = signal.sum(dim=-1)[::4, ::4].detach().cpu().numpy().astype(np.float32)
                signal_sum_snapshots.append(sig_ds)

    return SimOutput(
        snapshots=snapshots,
        snapshot_steps=steps_taken,
        mass_history=mass_history,
        initial_mass=initial_mass,
        n_species=1,
        species_mass_history=[mass_history],
        species_territory_history=[],
        ownership_snapshots=[],
        signal_total_history=signal_total_history,
        signal_channel_snapshots=signal_channel_snapshots,
        signal_sum_snapshots=signal_sum_snapshots,
    )


def _run_heterogeneous(
    config: EcosystemConfig,
    creatures: list[RolloutResult] | None = None,
) -> SimOutput:
    """Species-indexed simulation loop via LocalizedFlowLenia."""
    if creatures is None:
        creatures = [load_creature(s) for s in config.sources]

    validate_signal_consistency(creatures)

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
    patches = [s.patch if s.patch is not None else config.spawn.patch for s in config.sources]
    mass, weights = build_initial_state_multi_species(
        config.spawn,
        counts,
        config.grid_h,
        config.grid_w,
        config.device,
        patches=patches,
    )

    # Initialize signal field when any species has signal params.
    signal: torch.Tensor | None = None
    if any(sp.has_signal for sp in species_params):
        # Use the first signal-capable species' FlowLenia instance to generate
        # the background field (make_initial_signal_field is param-independent
        # for everything except the grid size, which is shared).
        for sp_fl in lfl._species:  # pyright: ignore[reportPrivateUsage]
            if sp_fl.params.has_signal:
                signal = sp_fl.make_initial_signal_field(seed=config.spawn.seed or 0)
                break

    state = LocalizedState(mass=mass, weights=weights, signal=signal)
    initial_mass = float(state.mass.sum().item())
    n_species = len(creatures)

    snapshots: list[np.ndarray] = []
    ownership_snapshots: list[np.ndarray] = []
    growth_snapshots: list[list[np.ndarray]] = []
    steps_taken: list[int] = []
    mass_history: list[float] = [initial_mass]
    signal_total_history: list[float] = [float(signal.sum().item()) if signal is not None else 0.0]
    signal_channel_snapshots: list[list[float]] = []
    # species_signal_received[s] = list of (C,) mean signal per snapshot
    species_signal_received: list[list[list[float]]] = [[] for _ in range(n_species)]
    signal_sum_snapshots: list[np.ndarray] = []
    species_mass: list[list[float]] = []
    species_territory: list[list[float]] = []
    for s in range(n_species):
        sp_mass = float((state.mass[:, :, 0] * state.weights[:, :, s]).sum().item())
        species_mass.append([sp_mass])
        sp_territory = float(state.weights[:, :, s].sum().item())
        species_territory.append([sp_territory])

    for step in range(1, config.steps + 1):
        is_snapshot = step % config.snapshot_every == 0 or step == config.steps
        growth_tensors: list[torch.Tensor] = []
        if is_snapshot:
            state, growth_tensors = lfl.step_with_diagnostics(state)
        else:
            state = lfl.step(state)
        mass_history.append(float(state.mass.sum().item()))
        sig = state.signal
        signal_total_history.append(float(sig.sum().item()) if sig is not None else 0.0)
        for s in range(n_species):
            sp_mass = float((state.mass[:, :, 0] * state.weights[:, :, s]).sum().item())
            species_mass[s].append(sp_mass)
            sp_territory = float(state.weights[:, :, s].sum().item())
            species_territory[s].append(sp_territory)

        if is_snapshot:
            frame = state.mass[:, :, 0].detach().cpu().numpy().astype(np.float32)
            snapshots.append(frame)
            steps_taken.append(step)
            ownership_snapshots.append(state.weights.detach().cpu().numpy().astype(np.float32))
            growth_snapshots.append([g.numpy().astype(np.float32) for g in growth_tensors])
            if sig is not None:
                ch_mean = sig.mean(dim=(0, 1)).detach().cpu().tolist()
                signal_channel_snapshots.append(ch_mean)
                # Spatial signal sum downsampled 4x for the signal GIF.
                sig_ds = sig.sum(dim=-1)[::4, ::4].detach().cpu().numpy().astype(np.float32)
                signal_sum_snapshots.append(sig_ds)
                # Per-species: mean signal over cells where species has ownership > 0.1
                for s in range(n_species):
                    w_s = state.weights[:, :, s]  # (H, W)
                    territory_mask = (w_s > 0.1).float().unsqueeze(-1)  # (H, W, 1)
                    total_territory = territory_mask.sum().item()
                    if total_territory > 0:
                        sp_sig = (sig * territory_mask).sum(dim=(0, 1)) / total_territory
                        species_signal_received[s].append(sp_sig.detach().cpu().tolist())
                    else:
                        species_signal_received[s].append([0.0] * sig.shape[-1])

    return SimOutput(
        snapshots=snapshots,
        snapshot_steps=steps_taken,
        mass_history=mass_history,
        initial_mass=initial_mass,
        n_species=n_species,
        species_mass_history=species_mass,
        species_territory_history=species_territory,
        ownership_snapshots=ownership_snapshots,
        growth_snapshots=growth_snapshots,
        signal_total_history=signal_total_history,
        signal_channel_snapshots=signal_channel_snapshots,
        species_signal_received=species_signal_received,
        signal_sum_snapshots=signal_sum_snapshots,
    )


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
        sim = _run_heterogeneous(config, creatures=creatures)
    else:
        sim = _run_homogeneous(config, creatures=creatures)
    elapsed = time.time() - started_at

    result, artifacts = _compute_outputs(
        config=config,
        run_id=run_id,
        run_dir=run_dir,
        sim=sim,
        elapsed=elapsed,
        creatures=creatures,
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
