"""The rollout function: worker-side bridge between sim and search.

Takes plain-Python ParamDict + seed + RolloutConfig, runs Flow-Lenia on the
worker's device, captures the COM/bbox/thumbnail trace, evaluates the quality
function, and returns a RolloutResult that's safe to pickle and ship back to
the driver.

This module is the only place biota.search talks to biota.sim. Step 7
(ray_compat.py) wraps `rollout` in a Ray task when Ray mode is enabled;
the default no-Ray code path calls `rollout` directly. Tests use the
direct path so Ray never starts.

Per-step stats (COM and bbox-fraction) are computed on CPU after a small
device->host transfer. The overhead is ~10% on CPU device and a similar order
on MPS/CUDA - acceptable given how much simpler the code is than a fully
on-device branch-free version. Thumbnails are batched as on-device tensors
during the loop and quantized once at the end.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from biota.search.descriptors import (
    DEFAULT_DESCRIPTORS,
    Descriptor,
    RolloutTrace,
    resolve_descriptors,
)
from biota.search.params import sample_random
from biota.search.quality import RolloutEvaluation, evaluate
from biota.search.result import CellCoord, ParamDict, RolloutResult
from biota.sim.flowlenia import Config as SimConfig
from biota.sim.flowlenia import FlowLenia, Params

THUMBNAIL_FRAMES = 32
"""Default number of frames captured per rollout into the thumbnail buffer.
Chosen to give animations smooth motion without ballooning archive size.
At 32 frames per cell, a 250-cell archive with 192-pixel grayscale frames
weighs about 280 MB - still small enough to scp around without thinking."""

THUMBNAIL_SIZE = 192
"""Default target edge length for downsampled thumbnail frames, in pixels.
Clamped at call time to min(THUMBNAIL_SIZE, sim_grid) so dev preset (64-grid)
doesn't try to upsample - it naturally produces 64-pixel thumbnails, while
standard preset (192-grid) produces 192-pixel thumbnails at native resolution.
192 matches the standard preset's sim grid, which means click-to-detail in
the viewer shows the full sim resolution with no aliasing."""
TRACE_TAIL_STEPS = 100  # the persistent filter needs two 50-step windows


@dataclass(frozen=True)
class RolloutConfig:
    """Worker-side rollout configuration: simulation plus capture parameters."""

    sim: SimConfig
    steps: int
    patch_fraction: int = 3
    """Initial patch is grid // patch_fraction on each side, centered."""

    thumbnail_frames: int = THUMBNAIL_FRAMES
    thumbnail_size: int = THUMBNAIL_SIZE

    active_descriptors: tuple[Descriptor, Descriptor, Descriptor] = field(
        default_factory=lambda: resolve_descriptors(DEFAULT_DESCRIPTORS)
    )
    """The three active descriptor objects used to compute archive coordinates.
    Defaults to (velocity, gyradius, spectral_entropy) when not supplied."""


def dev_preset() -> RolloutConfig:
    """Small and fast: 96x96 grid, 200 steps. For iteration and smoke tests."""
    return RolloutConfig(sim=SimConfig(grid_h=96, grid_w=96), steps=200)


def standard_preset() -> RolloutConfig:
    """The v1 default: 192x192 grid, 300 steps. For real searches and benchmarks."""
    return RolloutConfig(sim=SimConfig(grid_h=192, grid_w=192), steps=300)


def pretty_preset() -> RolloutConfig:
    """The hero shot: 384x384 grid, 500 steps. For the M6 export bundle."""
    return RolloutConfig(sim=SimConfig(grid_h=384, grid_w=384), steps=500)


def _params_dict_to_tensors(params: ParamDict, device: str) -> Params:
    """Convert a pickle-safe ParamDict to a torch-tensor Params on the given device."""
    from biota.search.params import has_signal_field

    base = Params(
        R=params["R"],
        r=torch.tensor(params["r"], dtype=torch.float32, device=device),
        m=torch.tensor(params["m"], dtype=torch.float32, device=device),
        s=torch.tensor(params["s"], dtype=torch.float32, device=device),
        h=torch.tensor(params["h"], dtype=torch.float32, device=device),
        a=torch.tensor(params["a"], dtype=torch.float32, device=device),
        b=torch.tensor(params["b"], dtype=torch.float32, device=device),
        w=torch.tensor(params["w"], dtype=torch.float32, device=device),
    )
    if not has_signal_field(params):
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
        emission_vector=torch.tensor(
            params["emission_vector"],  # type: ignore[typeddict-item]
            dtype=torch.float32,
            device=device,
        ),
        receptor_profile=torch.tensor(
            params["receptor_profile"],  # type: ignore[typeddict-item]
            dtype=torch.float32,
            device=device,
        ),
        signal_kernel_r=float(params["signal_kernel_r"]),  # type: ignore[typeddict-item]
        signal_kernel_a=torch.tensor(
            params["signal_kernel_a"],  # type: ignore[typeddict-item]
            dtype=torch.float32,
            device=device,
        ),
        signal_kernel_b=torch.tensor(
            params["signal_kernel_b"],  # type: ignore[typeddict-item]
            dtype=torch.float32,
            device=device,
        ),
        signal_kernel_w=torch.tensor(
            params["signal_kernel_w"],  # type: ignore[typeddict-item]
            dtype=torch.float32,
            device=device,
        ),
    )


def _initial_state(grid: int, patch_fraction: int, seed: int, device: str) -> torch.Tensor:
    """Centered random patch on a zero grid, deterministic for a given seed.

    Matches the M0 fixture recipe: grid // patch_fraction square patch of
    uniform [0, 1) values at the grid center.
    """
    patch = grid // patch_fraction
    rng = np.random.default_rng(seed)
    patch_data = rng.random((patch, patch, 1), dtype=np.float32)
    state = torch.zeros((grid, grid, 1), dtype=torch.float32, device=device)
    start = (grid - patch) // 2
    end = start + patch
    state[start:end, start:end, :] = torch.from_numpy(patch_data).to(device)
    return state


def _step_stats(state: torch.Tensor) -> tuple[float, float, float, float]:
    """Compute COM_y, COM_x, bbox-fraction, and gyradius for one state.

    Transfers the state to CPU and uses numpy. The per-step transfer cost is
    ~10% overhead on CPU device and similar order on MPS/CUDA. Worth it for
    code clarity over a fully on-device version that would need branch-free
    bbox computation.

    Gyradius is the mass-weighted RMS distance from COM, tracked cheaply here
    since we already have COM and the mass array, for use by the
    morphological_instability and activity descriptors.
    """
    arr = state[:, :, 0].detach().cpu().numpy()
    total = float(arr.sum())
    if total <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    h, w = arr.shape

    ys, xs = np.indices(arr.shape)
    com_y = float((ys * arr).sum() / total)
    com_x = float((xs * arr).sum() / total)

    # Gyradius: mass-weighted RMS distance from COM
    dy = ys - com_y
    dx = xs - com_x
    sq_dist = dy * dy + dx * dx
    gyradius = float(np.sqrt(max(float((arr * sq_dist).sum() / total), 0.0)))

    peak = float(arr.max())
    if peak <= 0.0:
        return com_y, com_x, 0.0, gyradius
    threshold = 0.1 * peak
    mask = arr > threshold
    if not mask.any():
        return com_y, com_x, 0.0, gyradius
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    bbox_area = (int(rows[-1]) - int(rows[0]) + 1) * (int(cols[-1]) - int(cols[0]) + 1)
    return com_y, com_x, bbox_area / (h * w), gyradius


def _downsample_frame(state: torch.Tensor, target: int) -> torch.Tensor:
    """Average-pool a (H, W, 1) state to (target, target). Returns float32 on the same device."""
    flat = state[:, :, 0].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    pooled = F.adaptive_avg_pool2d(flat, output_size=(target, target))
    return pooled[0, 0]


def _empty_thumbnail(frames: int, size: int) -> np.ndarray:
    return np.zeros((frames, size, size), dtype=np.uint8)


def _quantize_thumbnail(frames: torch.Tensor) -> np.ndarray:
    """Convert (N, S, S) float32 frames into (N, S, S) uint8 with global normalization."""
    cpu = frames.detach().cpu().numpy().astype(np.float32)
    peak = float(cpu.max()) if cpu.size > 0 else 0.0
    if peak <= 0.0:
        return np.zeros_like(cpu, dtype=np.uint8)
    return np.clip(cpu / peak * 255.0, 0.0, 255.0).astype(np.uint8)


def rollout(
    params: ParamDict,
    seed: int,
    config: RolloutConfig,
    device: str = "cpu",
    parent_cell: CellCoord | None = None,
) -> RolloutResult:
    """Run one Flow-Lenia rollout end-to-end and return a RolloutResult.

    Always returns a result. If the sim produces a non-finite state (which
    should not happen with mass conservation but is checked defensively), or
    the quality function rejects, the result has quality=None and a populated
    rejection_reason.
    """
    started_at = time.perf_counter()
    created_at = time.time()

    # 1. Build sim from ParamDict
    sim_params = _params_dict_to_tensors(params, device=device)
    fl = FlowLenia(config.sim, sim_params, device=device)

    # 2. Initial state, deterministic for the seed
    state = _initial_state(
        grid=config.sim.grid_h,
        patch_fraction=config.patch_fraction,
        seed=seed,
        device=device,
    )
    initial_mass = float(state.sum().item())

    # Initialize signal field when creature has signal params.
    # The background field is spatially varied so receptors have something
    # to respond to during solo rollouts.
    signal: torch.Tensor | None = None
    initial_signal_mass = 0.0
    if sim_params.has_signal:
        signal = fl.make_initial_signal_field(seed=seed)
        initial_signal_mass = float(signal.sum().item())

    # Clamp the thumbnail target to the sim grid so dev preset (small grid)
    # doesn't try to upsample - if the configured thumbnail size is larger
    # than the sim grid, cap it at the grid itself. This lets us keep a
    # single sensible default (192) without per-preset overrides.
    thumbnail_size = min(config.thumbnail_size, config.sim.grid_h)

    # 3. Pick frame indices for the thumbnail (uniform across the full rollout)
    if config.thumbnail_frames > 0:
        indices = np.linspace(0, config.steps, config.thumbnail_frames)
        frame_indices = set(indices.round().astype(int).tolist())
    else:
        frame_indices = set()

    # 4. Per-step buffers (numpy on CPU since stats are scalars per step)
    history_len = config.steps + 1
    com_y_np = np.zeros(history_len, dtype=np.float32)
    com_x_np = np.zeros(history_len, dtype=np.float32)
    bbox_np = np.zeros(history_len, dtype=np.float32)
    gyradius_np = np.zeros(history_len, dtype=np.float32)
    thumb_buf: list[torch.Tensor] = []

    # 5. Run the loop, capturing stats and frames as we go
    for step in range(history_len):
        com_y, com_x, bbox, gyr = _step_stats(state)
        com_y_np[step] = com_y
        com_x_np[step] = com_x
        bbox_np[step] = bbox
        gyradius_np[step] = gyr
        if step in frame_indices:
            thumb_buf.append(_downsample_frame(state, thumbnail_size))
        if step < config.steps:
            state, signal = fl.step(state, signal)

    final_mass = float(state.sum().item())
    final_signal_mass = float(signal.sum().item()) if signal is not None else 0.0

    # 6. Final state to numpy for descriptor functions
    final_state_np = state[:, :, 0].detach().cpu().numpy().astype(np.float32)

    if not np.isfinite(final_mass) or not np.isfinite(com_y_np).all():
        # Defensive: NaN/inf in the state means the sim went off the rails.
        compute_seconds = time.perf_counter() - started_at
        thumbnail = (
            _quantize_thumbnail(torch.stack(thumb_buf, dim=0))
            if thumb_buf
            else _empty_thumbnail(config.thumbnail_frames, thumbnail_size)
        )
        return RolloutResult(
            params=params,
            seed=seed,
            descriptors=None,
            quality=None,
            rejection_reason="nan_state",
            thumbnail=thumbnail,
            parent_cell=parent_cell,
            created_at=created_at,
            compute_seconds=compute_seconds,
        )

    com_history = np.stack([com_y_np, com_x_np], axis=1)  # (steps+1, 2)

    # 7. Build the trace covering the last TRACE_TAIL_STEPS steps
    tail = min(TRACE_TAIL_STEPS, history_len)
    trace = RolloutTrace(
        com_history=com_history[-tail:].astype(np.float32),
        bbox_fraction_history=bbox_np[-tail:].astype(np.float32),
        gyradius_history=gyradius_np[-tail:].astype(np.float32),
        final_state=final_state_np,
        grid_size=config.sim.grid_h,
        total_steps=config.steps,
    )

    # 8. Evaluate quality
    eval_result = evaluate(
        RolloutEvaluation(
            initial_mass=initial_mass,
            final_mass=final_mass,
            trace=trace,
            initial_total=initial_mass + initial_signal_mass,
            final_signal_mass=final_signal_mass,
        ),
        active_descriptors=config.active_descriptors,
    )

    # 9. Build the thumbnail (with fallback if frame capture missed)
    if thumb_buf:
        thumbnail = _quantize_thumbnail(torch.stack(thumb_buf, dim=0))
    else:
        thumbnail = _empty_thumbnail(config.thumbnail_frames, thumbnail_size)

    compute_seconds = time.perf_counter() - started_at
    return RolloutResult(
        params=params,
        seed=seed,
        descriptors=eval_result.descriptors,
        quality=eval_result.quality,
        rejection_reason=eval_result.rejection_reason,
        thumbnail=thumbnail,
        parent_cell=parent_cell,
        created_at=created_at,
        compute_seconds=compute_seconds,
    )


def _step_stats_batch(
    states: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute COM, bbox-fraction, and gyradius for a batch of states.

    states: (B, H, W, 1) -> four (B,) float32 arrays:
        com_y, com_x, bbox_frac, gyradius.

    Transfers the whole batch to CPU once rather than one transfer per element.
    """
    arr = states[:, :, :, 0].detach().cpu().numpy().astype(np.float32)  # (B, H, W)
    B, h, w = arr.shape
    totals = arr.sum(axis=(1, 2))  # (B,)

    com_y = np.zeros(B, dtype=np.float32)
    com_x = np.zeros(B, dtype=np.float32)
    bbox_frac = np.zeros(B, dtype=np.float32)
    gyradius = np.zeros(B, dtype=np.float32)

    ys, xs = np.indices((h, w), dtype=np.float32)

    for i in range(B):
        total = float(totals[i])
        if total <= 0.0:
            continue
        a = arr[i]
        cy = float((ys * a).sum() / total)
        cx = float((xs * a).sum() / total)
        com_y[i] = cy
        com_x[i] = cx

        # Gyradius: mass-weighted RMS distance from COM
        dy = ys - cy
        dx = xs - cx
        sq_dist = dy * dy + dx * dx
        gyradius[i] = float(np.sqrt(max(float((a * sq_dist).sum() / total), 0.0)))

        peak = float(a.max())
        if peak <= 0.0:
            continue
        threshold = 0.1 * peak
        mask = a > threshold
        if not mask.any():
            continue
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        bbox_area = (int(rows[-1]) - int(rows[0]) + 1) * (int(cols[-1]) - int(cols[0]) + 1)
        bbox_frac[i] = bbox_area / (h * w)

    return com_y, com_x, bbox_frac, gyradius


def _downsample_frame_batch(states: torch.Tensor, target: int) -> list[torch.Tensor]:
    """Downsample a batch of states to target x target, one frame per element.

    states: (B, H, W, 1) -> list of B (target, target) float32 tensors on the same device.
    """
    B = states.shape[0]
    flat = states[:, :, :, 0].unsqueeze(1)  # (B, 1, H, W)
    pooled = F.adaptive_avg_pool2d(flat, output_size=(target, target))  # (B, 1, T, T)
    return [pooled[i, 0] for i in range(B)]


def _build_batched_fk(
    params_list: list[ParamDict], config: RolloutConfig, device: str
) -> tuple[torch.Tensor, FlowLenia]:
    """Build stacked FFT kernels for a batch of param sets.

    Returns (fK_batch, ref_fl) where fK_batch is (B, K, H, W) complex and
    ref_fl is a FlowLenia instance built from the first param set - used to
    borrow the param-independent tensors (Sobel kernels, position grid).

    Building B FlowLenia instances just to extract their _fK is slightly
    wasteful, but the kernel build is a one-time cost before the sim loop and
    the logic is already well-tested. Duplicating the kernel formula here
    would be worse.
    """
    fk_list: list[torch.Tensor] = []
    ref_fl: FlowLenia | None = None

    for p in params_list:
        tp = Params(
            R=p["R"],
            r=torch.tensor(p["r"], dtype=torch.float32, device=device),
            m=torch.tensor(p["m"], dtype=torch.float32, device=device),
            s=torch.tensor(p["s"], dtype=torch.float32, device=device),
            h=torch.tensor(p["h"], dtype=torch.float32, device=device),
            a=torch.tensor(p["a"], dtype=torch.float32, device=device),
            b=torch.tensor(p["b"], dtype=torch.float32, device=device),
            w=torch.tensor(p["w"], dtype=torch.float32, device=device),
        )
        fl = FlowLenia(config.sim, tp, device=device)
        fK_single, _, _, _ = fl.kernel_tensors()
        fk_list.append(fK_single)  # (K, H, W)
        if ref_fl is None:
            ref_fl = fl

    assert ref_fl is not None
    return torch.stack(fk_list, dim=0), ref_fl  # (B, K, H, W)


def _batched_sim_step(
    A: torch.Tensor,
    fK: torch.Tensor,
    m_batch: torch.Tensor,
    s_batch: torch.Tensor,
    h_batch: torch.Tensor,
    sobel_kx: torch.Tensor,
    sobel_ky: torch.Tensor,
    pos: torch.Tensor,
    config: SimConfig,
) -> torch.Tensor:
    """One Flow-Lenia step for B states with per-element params.

    A:      (B, H, W, 1)
    fK:     (B, K, H, W) complex
    m/s/h:  (B, K) - per-element growth function params
    sobel_kx, sobel_ky: (1, 1, 3, 3)
    pos:    (2, H, W)

    Returns (B, H, W, 1).
    """
    B = A.shape[0]
    A2 = A[:, :, :, 0]  # (B, H, W)

    fA = torch.fft.fft2(A2)
    U = torch.fft.ifft2(fK * fA.unsqueeze(1), dim=(-2, -1)).real  # (B, K, H, W)

    m = m_batch.view(B, -1, 1, 1)
    s = s_batch.view(B, -1, 1, 1)
    h = h_batch.view(B, -1, 1, 1)
    G = (torch.exp(-(((U - m) / s) ** 2) / 2.0) * 2.0 - 1.0) * h
    U_sum = G.sum(dim=1)  # (B, H, W)

    # Sobel: treat B as the spatial batch for F.conv2d
    U_4d = U_sum.unsqueeze(1)  # (B, 1, H, W)
    gx_U = F.conv2d(U_4d, sobel_kx, padding=1)[:, 0]
    gy_U = F.conv2d(U_4d, sobel_ky, padding=1)[:, 0]
    nabla_U = torch.stack([gy_U, gx_U], dim=1)  # (B, 2, H, W)

    A2_4d = A2.unsqueeze(1)
    gx_A = F.conv2d(A2_4d, sobel_kx, padding=1)[:, 0]
    gy_A = F.conv2d(A2_4d, sobel_ky, padding=1)[:, 0]
    nabla_A = torch.stack([gy_A, gx_A], dim=1)  # (B, 2, H, W)

    alpha = torch.clamp(A2**2, 0.0, 1.0).unsqueeze(1)  # (B, 1, H, W)
    F_flow = nabla_U * (1.0 - alpha) - nabla_A * alpha  # (B, 2, H, W)

    # Reintegration
    dd = config.dd
    sigma = config.sigma
    ma = dd - sigma
    pos_b = pos.unsqueeze(0)  # (1, 2, H, W)

    clipped_flow = torch.clamp(config.dt * F_flow, -ma, ma)
    mu = pos_b + clipped_flow
    grid_hw = torch.tensor([config.grid_h, config.grid_w], device=A.device, dtype=A.dtype).view(
        1, 2, 1, 1
    )
    if config.border == "torus":
        mu = torch.stack([mu[:, 0] % config.grid_h, mu[:, 1] % config.grid_w], dim=1)
    else:
        mu = torch.stack(
            [
                torch.clamp(mu[:, 0], sigma, config.grid_h - sigma),
                torch.clamp(mu[:, 1], sigma, config.grid_w - sigma),
            ],
            dim=1,
        )

    new_A2 = torch.zeros_like(A2)
    sigma_clip_max = min(1.0, 2.0 * sigma)
    denom = 4.0 * sigma * sigma

    for dx in range(-dd, dd + 1):
        for dy in range(-dd, dd + 1):
            Ar = torch.roll(A2, shifts=(dx, dy), dims=(1, 2))
            mur = torch.roll(mu, shifts=(dx, dy), dims=(2, 3))
            dpmu = torch.abs(pos_b - mur)
            if config.border == "torus":
                dpmu = torch.minimum(dpmu, grid_hw - dpmu)
            sz = 0.5 - dpmu + sigma
            sz_clipped = torch.clamp(sz, 0.0, sigma_clip_max)
            area = (sz_clipped[:, 0] * sz_clipped[:, 1]) / denom
            new_A2 = new_A2 + Ar * area

    return new_A2.unsqueeze(-1)


def rollout_batch(
    params_list: list[ParamDict],
    seeds: list[int],
    config: RolloutConfig,
    device: str = "cpu",
    parent_cells: list[CellCoord | None] | None = None,
) -> list[RolloutResult]:
    """Run B Flow-Lenia rollouts simultaneously as a single batched forward pass.

    One PyTorch forward pass evaluates all B parameter sets in parallel over a
    leading batch dimension. The sim loop runs in a single Python loop;
    PyTorch handles the batch parallelism internally (GPU vectorization or
    CPU thread-level parallelism).

    params_list and seeds must have the same length B. parent_cells defaults
    to [None] * B if not supplied.

    Returns a list of B RolloutResult objects in the same order as params_list.
    The interface mirrors rollout() - each result has quality/descriptors/
    rejection_reason set correctly regardless of whether the sim produced a
    valid creature.
    """
    B = len(params_list)
    assert len(seeds) == B, (
        f"params_list and seeds must be the same length, got {B} and {len(seeds)}"
    )
    if parent_cells is None:
        parent_cells_nn: list[CellCoord | None] = [None] * B
        parent_cells = parent_cells_nn

    started_at = time.perf_counter()
    created_at = time.time()

    thumbnail_size = min(config.thumbnail_size, config.sim.grid_h)

    # Frame capture indices shared across all elements
    if config.thumbnail_frames > 0:
        indices = np.linspace(0, config.steps, config.thumbnail_frames)
        frame_indices = set(indices.round().astype(int).tolist())
    else:
        frame_indices = set()

    # Build batched FFT kernels and borrow param-independent tensors from ref instance
    fK, ref_fl = _build_batched_fk(params_list, config, device)  # (B, K, H, W)
    _, sobel_kx, sobel_ky, pos = ref_fl.kernel_tensors()

    # Stack per-element growth params: m, s, h all (B, K)
    m_batch = torch.stack(
        [torch.tensor(p["m"], dtype=torch.float32, device=device) for p in params_list]
    )
    s_batch = torch.stack(
        [torch.tensor(p["s"], dtype=torch.float32, device=device) for p in params_list]
    )
    h_batch = torch.stack(
        [torch.tensor(p["h"], dtype=torch.float32, device=device) for p in params_list]
    )

    # Build initial states - one per seed, stacked to (B, H, W, 1)
    states = torch.stack(
        [
            _initial_state(
                grid=config.sim.grid_h,
                patch_fraction=config.patch_fraction,
                seed=seeds[i],
                device=device,
            )
            for i in range(B)
        ]
    )

    initial_masses = [float(states[i].sum().item()) for i in range(B)]

    # Per-element history buffers
    history_len = config.steps + 1
    com_y_hist = np.zeros((B, history_len), dtype=np.float32)
    com_x_hist = np.zeros((B, history_len), dtype=np.float32)
    bbox_hist = np.zeros((B, history_len), dtype=np.float32)
    gyradius_hist = np.zeros((B, history_len), dtype=np.float32)
    thumb_bufs: list[list[torch.Tensor]] = [[] for _ in range(B)]

    # Run the sim loop
    for step in range(history_len):
        com_y, com_x, bbox, gyr = _step_stats_batch(states)
        com_y_hist[:, step] = com_y
        com_x_hist[:, step] = com_x
        bbox_hist[:, step] = bbox
        gyradius_hist[:, step] = gyr

        if step in frame_indices:
            frames = _downsample_frame_batch(states, thumbnail_size)
            for i in range(B):
                thumb_bufs[i].append(frames[i])

        if step < config.steps:
            states = _batched_sim_step(
                states,
                fK,
                m_batch,
                s_batch,
                h_batch,
                sobel_kx,
                sobel_ky,
                pos,
                config.sim,
            )

    final_masses = [float(states[i].sum().item()) for i in range(B)]

    # Post-sim: per-element evaluation (all on CPU/numpy from here)
    results: list[RolloutResult] = []
    compute_seconds = time.perf_counter() - started_at

    for i in range(B):
        final_state_np = states[i, :, :, 0].detach().cpu().numpy().astype(np.float32)

        if thumb_bufs[i]:
            thumbnail = _quantize_thumbnail(torch.stack(thumb_bufs[i], dim=0))
        else:
            thumbnail = _empty_thumbnail(config.thumbnail_frames, thumbnail_size)

        com_history = np.stack([com_y_hist[i], com_x_hist[i]], axis=1)  # (steps+1, 2)

        if not np.isfinite(final_masses[i]) or not np.isfinite(com_history).all():
            results.append(
                RolloutResult(
                    params=params_list[i],
                    seed=seeds[i],
                    descriptors=None,
                    quality=None,
                    rejection_reason="nan_state",
                    thumbnail=thumbnail,
                    parent_cell=parent_cells[i],
                    created_at=created_at,
                    compute_seconds=compute_seconds,
                )
            )
            continue

        tail = min(TRACE_TAIL_STEPS, history_len)
        trace = RolloutTrace(
            com_history=com_history[-tail:].astype(np.float32),
            bbox_fraction_history=bbox_hist[i, -tail:].astype(np.float32),
            gyradius_history=gyradius_hist[i, -tail:].astype(np.float32),
            final_state=final_state_np,
            grid_size=config.sim.grid_h,
            total_steps=config.steps,
        )

        eval_result = evaluate(
            RolloutEvaluation(
                initial_mass=initial_masses[i],
                final_mass=final_masses[i],
                trace=trace,
            ),
            active_descriptors=config.active_descriptors,
        )

        results.append(
            RolloutResult(
                params=params_list[i],
                seed=seeds[i],
                descriptors=eval_result.descriptors,
                quality=eval_result.quality,
                rejection_reason=eval_result.rejection_reason,
                thumbnail=thumbnail,
                parent_cell=parent_cells[i],
                created_at=created_at,
                compute_seconds=compute_seconds,
            )
        )

    return results


# Re-export for convenience
__all__ = [
    "RolloutConfig",
    "dev_preset",
    "pretty_preset",
    "rollout",
    "rollout_batch",
    "sample_random",
    "standard_preset",
]
