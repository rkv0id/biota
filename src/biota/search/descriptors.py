"""Behavior descriptors for Flow-Lenia rollouts.

Descriptors are normalized scalar reductions over a RolloutTrace, each
returning a float in [0, 1]. They define the behavioral axes of the MAP-Elites
archive. The search always uses exactly three active descriptors.

Built-in descriptor library (15 total):

  velocity                mean COM displacement/step over trailing WINDOW steps
  gyradius                mass-weighted RMS distance from COM at final step
  spectral_entropy        Shannon entropy of radially-averaged FFT spectrum
  oscillation             variance of bbox fraction over trace tail
  compactness             mass inside bbox / total mass at final step
  mass_asymmetry          |mean COM_x drift - mean COM_y drift| / sum
  png_compressibility     PNG compressed / uncompressed size of final state
  rotational_symmetry     angular variance of radial mass profile at final step
  persistence_score       max drift of the three core descriptors across trace tail
  displacement_ratio      total displacement / total path length (0=orbiter, 1=glider)
  angular_velocity        mean absolute angular speed of COM motion, normalized
  growth_gradient         spatial gradient concentration relative to COM, normalized
  morphological_instability variance of gyradius over trace tail (shape stability)
  activity                mean absolute change in gyradius per step (internal work rate)
  spatial_entropy         Shannon entropy of coarse spatial mass distribution

Custom descriptors can be loaded from a user Python file via --descriptor-module;
see cli.py. The file must define a list named DESCRIPTORS containing Descriptor
objects. Loaded descriptors are merged into the registry at startup.

A Descriptor's compute function must return a raw float clipped to [0, 100].
CVT-MAP-Elites handles scale implicitly via centroid fitting; no per-descriptor
normalization is needed. The [0, 100] bound guards against numerical outliers
distorting centroid positions. Typical values are well below 10.
"""

import zlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from biota.search.result import Descriptors

WINDOW = 50
"""Number of trailing steps used to compute the velocity descriptor."""

VELOCITY_NORMALIZER = 0.02
"""Empirical maximum velocity in cells-per-step for the biota parameter
prior. Calibrated against an actual cluster run on standard preset where
the maximum observed velocity across the discovered population was 0.0077
cells/step. The 0.02 normalizer gives meaningful spread across the
[0, ~0.4] of normalized range with a comfortable safety margin against
slightly faster outliers.

This value is much smaller than the Leniabreeder paper's 0.5 because
biota's parameter prior produces creatures that move 50-100x slower than
their population. If a future change to the parameter prior or the sim
preset produces faster creatures and they all clip to 1.0 here, this
needs to be re-tuned upward."""

GYRADIUS_NORMALIZER_DIVISOR = 4.0
"""Gyradius is normalized by grid_size / GYRADIUS_NORMALIZER_DIVISOR. A
creature whose mass-weighted RMS distance from center equals grid_size/4
saturates the descriptor at 1.0; in practice creatures sit well below
that bound."""

SPECTRAL_RADIAL_BINS = 32
"""Number of radial bins used to coarse-grain the 2D FFT magnitude
spectrum into a 1D radial profile. The spectral entropy descriptor takes
Shannon entropy of this 1D profile (normalized to a probability mass)."""

SPECTRAL_ENTROPY_FLOOR = 0.55
"""Empirical lower bound of spectral entropy across biota's discovered
population at standard preset.

The raw spectral_entropy value (Shannon entropy of the radial FFT profile,
divided by log(SPECTRAL_RADIAL_BINS)) lives in [0, 1] in principle but
in practice biota's discovered creatures are all sharp-edged structured
solitons whose spectral entropy lands in roughly [0.55, 0.95]. Pure
Gaussians give 0.14-0.45 but the search never finds those, because the
parameter prior + survival filters exclude smooth diffuse blobs.

Without a remap, ~80% of the discovered population would crowd into the
top 4 of 16 structure bins (everything between [0.85, 0.95]). With the
remap (raw - FLOOR) / (1 - FLOOR), the same population spreads across
the full [0, 1] range and uses all 16 bins.

Calibrated against the iteration-2 cluster run (run id eager-bramble,
166 cells) which observed a minimum spectral entropy of 0.59 across the
discovered population. The 0.55 floor is set slightly below the minimum
with margin for slightly less-structured outliers."""


# === Descriptor dataclass ===


@dataclass(frozen=True)
class Descriptor:
    """A behavioral descriptor: metadata plus a pure compute function.

    The compute function receives a RolloutTrace and must return a raw float
    clipped to [0, 100]. CVT handles scale; no normalization is applied inside
    compute(). Typical observed ranges are well below 10 for all built-in
    descriptors.

    Attributes:
        name:            Full display name used in the UI ("spectral entropy").
        short_name:      Compact label for axis ticks ("spec.ent.").
        direction_label: What increasing values mean ("higher entropy").
        compute:         Pure function (RolloutTrace) -> float in [0, 1].
    """

    name: str
    short_name: str
    direction_label: str
    compute: Callable[["RolloutTrace"], float]
    signal_only: bool = False
    """When True, this descriptor requires a signal-enabled rollout.
    Passing a signal-only descriptor to a non-signal search raises a
    ValueError at startup (enforced in loop.py)."""


# === RolloutTrace ===


@dataclass(frozen=True)
class RolloutTrace:
    """Per-rollout history bundle, computed by the worker during simulation.

    All arrays are numpy, dtype float32, on CPU. The trace is large enough to
    cover the last 100 steps so the persistent filter can compare two adjacent
    50-step windows.

    Attributes:
        com_history: Center of mass at each step in the trailing window.
            Shape (T, 2) where T >= 2*WINDOW. Each row is (com_y, com_x) in
            grid units.
        bbox_fraction_history: Bbox area as a fraction of grid area at each
            step in the trailing window. Shape (T,). Used by the localized
            filter in quality.py, not by the descriptor functions in this
            module.
        gyradius_history: Mass-weighted RMS distance from COM at each step in
            the trailing window. Shape (T,). Tracked cheaply alongside COM
            during the rollout loop. Used by morphological_instability and
            activity descriptors.
        final_state: The simulation state at the final step. Shape
            (grid_size, grid_size), dtype float32.
        grid_size: The simulation grid edge length, used for descriptor
            normalization.
        total_steps: The total number of simulation steps in the rollout.
    """

    com_history: np.ndarray
    bbox_fraction_history: np.ndarray
    gyradius_history: np.ndarray
    final_state: np.ndarray
    grid_size: int
    total_steps: int
    midpoint_state: np.ndarray | None = None
    """State at the midpoint step (total_steps // 2). Used by the multi-
    point compactness term in the quality metric to penalise creatures that
    peak early and degrade. None when not captured (older code paths and
    rollout_batch, which does not support per-element midpoint capture)."""
    signal_emission_history: np.ndarray | None = None
    """Mean positive-growth emission activity per step in the trace tail.
    Shape (T,) float32. Proportional to how much signal the creature
    actually emitted at each step. None for non-signal rollouts."""
    signal_reception_history: np.ndarray | None = None
    """Mean absolute reception response |dot(convolved_signal, receptor)|
    per step in the trace tail. Shape (T,) float32. Measures how strongly
    the creature actually responded to the chemical environment.
    None for non-signal rollouts."""
    signal_retention: float | None = None
    """final_mass / initial_mass for signal rollouts. Measures how much
    mass the creature retained vs bled into the signal field.
    None for non-signal rollouts."""
    final_signal_state: np.ndarray | None = None
    """Final signal field (H, W, C) float32, summed to (H, W) for spatial
    descriptors. None for non-signal rollouts."""
    initial_signal_mass: float = 0.0
    """Total signal field mass at step 0. Zero for non-signal rollouts."""

    def slice(self, start: int, end: int) -> "RolloutTrace":
        """Return a trace covering steps [start, end) within the original window.

        Used by the persistent filter to compute descriptors at two different
        time points. The final_state is unchanged (it is still the last step
        of the original rollout) since the gyradius and spectral_entropy
        descriptors always read the final step.
        """
        return RolloutTrace(
            com_history=self.com_history[start:end],
            bbox_fraction_history=self.bbox_fraction_history[start:end],
            gyradius_history=self.gyradius_history[start:end],
            final_state=self.final_state,
            grid_size=self.grid_size,
            total_steps=self.total_steps,
            signal_emission_history=(
                self.signal_emission_history[start:end]
                if self.signal_emission_history is not None
                else None
            ),
            signal_reception_history=(
                self.signal_reception_history[start:end]
                if self.signal_reception_history is not None
                else None
            ),
            signal_retention=self.signal_retention,
            final_signal_state=self.final_signal_state,
            initial_signal_mass=self.initial_signal_mass,
        )


# === built-in compute functions ===


def compute_velocity(trace: RolloutTrace) -> float:
    """Mean COM step-delta over the last WINDOW steps, in cells/step.

    Raw value: typically 0.0-0.02 for biota's parameter prior. CVT handles
    scale; VELOCITY_NORMALIZER is kept as a reference for typical ranges.
    """
    coms = trace.com_history[-WINDOW:]
    if len(coms) < 2:
        return 0.0
    deltas = np.diff(coms, axis=0)
    speeds = np.linalg.norm(deltas, axis=1)
    mean_speed = float(speeds.mean())
    return float(np.clip(mean_speed, 0.0, 100.0))


def compute_gyradius(trace: RolloutTrace) -> float:
    """Mass-weighted RMS distance from center of mass, in grid units.

    This is Chan's `rm` from the original Lenia paper: a continuous, smooth
    measure of how spread-out the creature's mass is. Unlike the bounding
    box-based size measure used in M1, gyradius does not depend on a binary
    mass threshold and varies smoothly with the mass distribution.

    Returns 0.0 if the final state has no mass.
    """
    state = trace.final_state
    total_mass = float(state.sum())
    if total_mass <= 0.0:
        return 0.0

    h, w = state.shape
    y_coords, x_coords = np.indices((h, w), dtype=np.float32)
    com_y = float((state * y_coords).sum() / total_mass)
    com_x = float((state * x_coords).sum() / total_mass)

    dy = y_coords - com_y
    dx = x_coords - com_x
    sq_dist = dy * dy + dx * dx
    variance = float((state * sq_dist).sum() / total_mass)
    gyradius = float(np.sqrt(max(variance, 0.0)))

    return float(np.clip(gyradius, 0.0, 100.0))


def compute_spectral_entropy(trace: RolloutTrace) -> float:
    """Shannon entropy of the radially-averaged FFT magnitude spectrum.

    Captures spatial frequency content of the final-step mass distribution.
    Smooth blobs cluster near zero; sharp-edged structured creatures spread
    into the upper range. See SPECTRAL_ENTROPY_FLOOR for the empirical remap
    that prevents population crowding in the top few bins.

    Returns 0.0 for empty or constant final states.
    """
    state = trace.final_state
    if float(state.sum()) <= 0.0:
        return 0.0

    h, w = state.shape

    spectrum = np.fft.fft2(state.astype(np.float64))
    magnitude = np.abs(spectrum)
    magnitude[0, 0] = 0.0

    if float(magnitude.sum()) <= 0.0:
        return 0.0

    ky = np.fft.fftfreq(h) * h
    kx = np.fft.fftfreq(w) * w
    ky_grid, kx_grid = np.meshgrid(ky, kx, indexing="ij")
    radius = np.sqrt(ky_grid * ky_grid + kx_grid * kx_grid)

    r_max = float(np.sqrt((h / 2.0) ** 2 + (w / 2.0) ** 2))
    if r_max <= 0.0:
        return 0.0

    bin_indices = np.minimum(
        np.floor(radius / r_max * SPECTRAL_RADIAL_BINS).astype(np.int64),
        SPECTRAL_RADIAL_BINS - 1,
    )

    radial_profile = np.bincount(
        bin_indices.ravel(),
        weights=magnitude.ravel(),
        minlength=SPECTRAL_RADIAL_BINS,
    )

    total = float(radial_profile.sum())
    if total <= 0.0:
        return 0.0

    p = radial_profile / total
    nonzero = p[p > 0.0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    max_entropy = float(np.log(SPECTRAL_RADIAL_BINS))
    if max_entropy <= 0.0:
        return 0.0

    raw = entropy / max_entropy
    return float(np.clip(raw, 0.0, 100.0))


def compute_oscillation(trace: RolloutTrace) -> float:
    """Variance of bbox fraction over the trace tail, normalized to [0, 1].

    Pulsing/breathing creatures have a bbox fraction that oscillates as
    they expand and contract. Rigid translating creatures have near-zero
    variance. The normalizer (0.05) is empirical: a bbox fraction swinging
    between 0.1 and 0.3 has variance ~0.01; rigid creatures ~0.0001.
    Normalize by 0.05 so strongly oscillating creatures saturate at 1.0.
    """
    fractions = trace.bbox_fraction_history[-WINDOW:]
    if len(fractions) < 2:
        return 0.0
    variance = float(np.var(fractions))
    return float(np.clip(variance, 0.0, 100.0))


def compute_compactness(trace: RolloutTrace) -> float:
    """Fraction of total mass inside the final-step bounding box, in [0, 1].

    Uses a 10%-of-peak threshold to define the bbox, same as quality.py.
    Compact creatures with all mass inside a tight bbox score near 1.0.
    Diffuse creatures with mass scattered as background score near 0.0.

    This is the same compactness used as the quality metric in quality.py,
    promoted here to a behavioral axis so it can serve as a descriptor
    dimension separate from quality.
    """
    state = trace.final_state
    total_mass = float(state.sum())
    if total_mass <= 0.0:
        return 0.0

    peak = float(state.max())
    if peak <= 0.0:
        return 0.0

    threshold = 0.1 * peak
    mask = state > threshold
    if not mask.any():
        return 0.0

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    y_min, y_max = int(rows[0]), int(rows[-1]) + 1
    x_min, x_max = int(cols[0]), int(cols[-1]) + 1

    bbox_mass = float(state[y_min:y_max, x_min:x_max].sum())
    return float(np.clip(bbox_mass / total_mass, 0.0, 100.0))


def compute_mass_asymmetry(trace: RolloutTrace) -> float:
    """Directional bias of motion: |mean_x_drift - mean_y_drift| / sum.

    Straight-line movers have nearly all drift along one axis -> high
    asymmetry. Orbiters or creatures with circular motion have balanced
    x and y drift -> low asymmetry. Returns 0 if there is no net motion.

    Normalized to [0, 1] by construction: the numerator is bounded by
    the denominator when it is nonzero.
    """
    coms = trace.com_history[-WINDOW:]
    if len(coms) < 2:
        return 0.0
    deltas = np.diff(coms, axis=0)
    mean_x = float(np.abs(deltas[:, 1]).mean())
    mean_y = float(np.abs(deltas[:, 0]).mean())
    total = mean_x + mean_y
    if total <= 0.0:
        return 0.0
    return float(np.clip(abs(mean_x - mean_y) / total, 0.0, 100.0))


def compute_png_compressibility(trace: RolloutTrace) -> float:
    """PNG compressed size / raw uncompressed size of the final state image.

    Standard ALife complexity metric (Adachi et al. 2024, Michel et al. 2025).
    Smooth/boring states compress well -> low ratio. Noisy random states are
    nearly incompressible -> high ratio. Structured creatures with repetition
    and symmetry land in the middle.

    Implementation: quantize the float32 final state to uint8, then
    zlib-compress (PNG's compressor). Raw size is H*W bytes.
    """
    state = trace.final_state
    if float(state.sum()) <= 0.0:
        return 0.0
    peak = float(state.max())
    if peak <= 0.0:
        return 0.0
    quantized = (state / peak * 255).clip(0, 255).astype(np.uint8)
    raw_bytes = quantized.tobytes()
    compressed = zlib.compress(raw_bytes, level=6)
    return float(np.clip(len(compressed) / len(raw_bytes), 0.0, 100.0))


def compute_rotational_symmetry(trace: RolloutTrace) -> float:
    """Angular variance of the radial mass profile at the final step, in [0, 1].

    Low value -> mass distributed uniformly around the COM (rings, dots,
    radially symmetric solitons). High value -> mass concentrated in
    particular angular directions (L-shapes, dumbbells, asymmetric gliders).

    Computed by binning mass into 16 angular sectors around the COM, then
    taking the variance of the resulting distribution normalized against
    the maximum achievable variance (all mass in one sector).
    """
    state = trace.final_state
    total_mass = float(state.sum())
    if total_mass <= 0.0:
        return 0.0

    h, w = state.shape
    y_coords, x_coords = np.indices((h, w), dtype=np.float32)
    com_y = float((state * y_coords).sum() / total_mass)
    com_x = float((state * x_coords).sum() / total_mass)

    dy = y_coords - com_y
    dx = x_coords - com_x

    n_bins = 16
    angles = (np.arctan2(dy, dx) + np.pi) / (2.0 * np.pi)  # [0, 1)
    bin_indices = np.minimum(np.floor(angles * n_bins).astype(np.int64), n_bins - 1)
    radial_mass = np.bincount(bin_indices.ravel(), weights=state.ravel(), minlength=n_bins)

    total = float(radial_mass.sum())
    if total <= 0.0:
        return 0.0

    p = radial_mass / total
    variance = float(np.var(p))
    # Maximum variance: all mass in one bin -> p = [1, 0, ..., 0].
    # var([1, 0,...,0]) = mean_p * (1 - mean_p) where mean_p = 1/n_bins.
    mean_p = 1.0 / n_bins
    max_variance = mean_p * (1.0 - mean_p)
    if max_variance <= 0.0:
        return 0.0
    return float(np.clip(variance / max_variance, 0.0, 100.0))


def compute_persistence_score(trace: RolloutTrace) -> float:
    """Temporal stability of core descriptor values over the trace tail, in [0, 1].

    Continuous version of the binary persistent filter in quality.py. Computes
    velocity, gyradius, and spectral_entropy over two adjacent WINDOW-step
    slices and returns the maximum normalized drift, scaled against the
    PERSISTENT_DESCRIPTOR_DRIFT threshold (0.2).

    Low score -> creature is behaviorally consistent over time (stable).
    High score -> descriptors drift significantly (creature is changing).

    Always uses velocity/gyradius/spectral_entropy regardless of which three
    descriptors are active for the current search, so the measure stays
    anchored to the same observables as the quality filter.
    """
    total = len(trace.com_history)
    if total < 2 * WINDOW:
        return 0.0

    late_slice = trace.slice(total - WINDOW, total)
    early_slice = trace.slice(total - 2 * WINDOW, total - WINDOW)

    late = (
        compute_velocity(late_slice),
        compute_gyradius(late_slice),
        compute_spectral_entropy(late_slice),
    )
    early = (
        compute_velocity(early_slice),
        compute_gyradius(early_slice),
        compute_spectral_entropy(early_slice),
    )

    max_drift = max(abs(a - b) for a, b in zip(late, early, strict=True))
    return float(np.clip(max_drift, 0.0, 100.0))


ANGULAR_VELOCITY_NORMALIZER = 0.15
"""Empirical upper bound for mean angular speed (radians/step) in biota's
parameter prior. A creature traversing a full circle in 40 steps has angular
speed 2*pi/40 ~ 0.157 rad/step, so 0.15 places a full rotation near 1.0."""

GROWTH_GRADIENT_NORMALIZER = 0.5
"""Empirical upper bound for the mass-weighted mean spatial gradient magnitude,
normalized by grid_size. Calibrated so strongly driven asymmetric creatures
saturate near 1.0."""

MORPHOLOGICAL_INSTABILITY_NORMALIZER = 0.05
"""Empirical upper bound for gyradius variance over the trace tail, divided
by grid_size^2. Strongly deforming/pulsing creatures saturate near 1.0."""

ACTIVITY_NORMALIZER = 0.02
"""Empirical upper bound for mean absolute gyradius change per step, divided
by grid_size. Creatures with strong pulsing or morphological change saturate
near 1.0. Static creatures score near 0.0."""

SPATIAL_ENTROPY_BINS = 8
"""Number of spatial bins per axis for the coarse spatial entropy grid.
An 8x8 = 64-cell grid captures meaningful spatial organization."""


def compute_displacement_ratio(trace: "RolloutTrace") -> float:
    """Ratio of total displacement to total path length over the trace tail.

    A perfect translating glider travels in a straight line: displacement
    equals path length, ratio = 1.0. A pure orbiter returns to its start:
    displacement ~ 0 but path length > 0, ratio ~ 0.0. A stationary creature
    has both near zero; ratio defaults to 0.0.

    Separates true gliders from fast-moving orbiters, which velocity alone
    cannot distinguish (Plantec et al. use linear vs angular speed as
    separate optimization targets for exactly this reason).
    """
    coms = trace.com_history[-WINDOW:]
    if len(coms) < 2:
        return 0.0
    deltas = np.diff(coms, axis=0)
    step_distances = np.linalg.norm(deltas, axis=1)
    path_length = float(step_distances.sum())
    if path_length <= 0.0:
        return 0.0
    total_displacement = float(np.linalg.norm(coms[-1] - coms[0]))
    return float(np.clip(total_displacement / path_length, 0.0, 100.0))


def compute_angular_velocity(trace: "RolloutTrace") -> float:
    """Mean absolute angular speed of COM motion over the trace tail, in [0, 1].

    Measures how quickly the direction of COM motion rotates. Pure rotors and
    orbiters score high; straight-line translators and stationary creatures
    score near zero. Orthogonal to mass_asymmetry (which measures directional
    bias of displacement magnitude, not rotation rate).

    Used by Plantec et al. as an optimization target for finding rotors; here
    adapted as a MAP-Elites descriptor normalized by ANGULAR_VELOCITY_NORMALIZER
    (0.15 rad/step ~ one full orbit in 40 steps).
    """
    coms = trace.com_history[-WINDOW:]
    if len(coms) < 3:
        return 0.0
    deltas = np.diff(coms, axis=0)
    speeds = np.linalg.norm(deltas, axis=1)
    threshold = 1e-4
    moving = speeds > threshold
    if moving.sum() < 2:
        return 0.0
    angles = np.arctan2(deltas[:, 0], deltas[:, 1])
    angle_diffs = np.diff(angles)
    angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
    valid = moving[:-1] & moving[1:]
    if valid.sum() == 0:
        return 0.0
    mean_angular_speed = float(np.abs(angle_diffs[valid]).mean())
    return float(np.clip(mean_angular_speed, 0.0, 100.0))


def compute_growth_gradient(trace: "RolloutTrace") -> float:
    """Mass-weighted mean spatial gradient magnitude at the final step, in [0, 1].

    Approximates Chan's growth-centroid distance (dgm): measures how strongly
    the internal structure of the creature is driven by spatial gradients.
    Computed using numpy.gradient on the final state, weighted by mass.

    Low value: smooth internally consistent creature (rings, symmetric blobs).
    High value: creature with strong internal gradients - channels, labyrinths,
    sharp-edged structures. Complements spectral_entropy (frequency domain)
    with a spatial-domain edge/gradient density measure.
    """
    state = trace.final_state
    total_mass = float(state.sum())
    if total_mass <= 0.0:
        return 0.0
    gy, gx = np.gradient(state.astype(np.float64))
    grad_magnitude = np.sqrt(gy * gy + gx * gx).astype(np.float32)
    weighted_grad = float((state * grad_magnitude).sum() / total_mass)
    # GROWTH_GRADIENT_NORMALIZER is an absolute bound: a uniform block scores
    # ~0.065, a thin ring ~0.22, strongly noisy states ~0.5+. Using 0.5 as
    # the normalizer puts structured creatures in the mid-range.
    return float(np.clip(weighted_grad, 0.0, 100.0))


def compute_morphological_instability(trace: "RolloutTrace") -> float:
    """Variance of gyradius over the trace tail, normalized to [0, 1].

    Measures how much the creature's spread/size fluctuates over time.
    Low value: rigid creature that maintains its form - predictable in
    ecosystem contacts. High value: creature that constantly reshapes or pulses
    in size - unpredictable and potentially interesting in multi-body dynamics.

    Distinct from oscillation (bounding-box area variance) in using the
    mass-weighted gyradius, which is more sensitive to internal mass
    redistribution without threshold-based bounding box artifacts.
    """
    gyr = trace.gyradius_history[-WINDOW:]
    if len(gyr) < 2:
        return 0.0
    variance = float(np.var(gyr))
    return float(np.clip(variance, 0.0, 100.0))


def compute_activity(trace: "RolloutTrace") -> float:
    """Mean absolute change in gyradius per step, normalized to [0, 1].

    A per-creature adaptation of evolutionary activity (Michel et al. 2025):
    measures how much internal work is happening step to step. Static or
    rigidly translating creatures score near 0.0. Creatures that are pulsing,
    reshaping, or undergoing morphological change score near 1.0.

    Complementary to oscillation (bounding-box variance) and morphological_
    instability (gyradius variance): activity captures the rate of change,
    not just the total magnitude of variation.
    """
    gyr = trace.gyradius_history[-WINDOW:]
    if len(gyr) < 2:
        return 0.0
    step_changes = np.abs(np.diff(gyr))
    mean_change = float(step_changes.mean())
    return float(np.clip(mean_change, 0.0, 100.0))


def compute_spatial_entropy(trace: "RolloutTrace") -> float:
    """Shannon entropy of the coarse spatial mass distribution, in [0, 1].

    Divides the grid into SPATIAL_ENTROPY_BINS x SPATIAL_ENTROPY_BINS cells
    and computes the Shannon entropy of the resulting mass distribution.
    Adapted from Michel et al. 2025's multi-scale matter distribution metric,
    applied at a single coarse scale as a per-creature descriptor.

    Low value: mass concentrated in a small region (compact, localized).
    High value: mass spread uniformly across the grid (diffuse or multi-body).

    Distinct from compactness (threshold-based bounding box) and gyradius
    (distance-weighted from COM): spatial entropy captures distribution
    uniformity across the full grid without distance or threshold assumptions.
    """
    state = trace.final_state
    total_mass = float(state.sum())
    if total_mass <= 0.0:
        return 0.0
    h, w = state.shape
    n = SPATIAL_ENTROPY_BINS
    h_crop = (h // n) * n
    w_crop = (w // n) * n
    cropped = state[:h_crop, :w_crop]
    coarse = cropped.reshape(n, h_crop // n, n, w_crop // n).sum(axis=(1, 3))
    total = float(coarse.sum())
    if total <= 0.0:
        return 0.0
    p = coarse.ravel() / total
    nonzero = p[p > 0.0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    max_entropy = float(np.log(n * n))
    if max_entropy <= 0.0:
        return 0.0
    return float(np.clip(entropy / max_entropy, 0.0, 100.0))


# === registry ===


# === signal descriptors ===


def compute_signal_field_variance(trace: RolloutTrace) -> float:
    """Spatial variance of the total signal field at the end of the rollout.

    Measures how structured and localized the creature's chemical footprint
    is. High variance = signal is concentrated near the creature body (strong
    local emission, fast decay). Low variance = signal has diffused evenly
    across the grid or barely accumulated.

    Computed from final_signal_state summed across channels to a (H, W)
    scalar field. Returns 0.0 for non-signal rollouts.
    """
    if trace.final_signal_state is None:
        return 0.0
    # Sum across channels -> (H, W) total signal per cell
    field = trace.final_signal_state.sum(axis=-1)
    return float(np.var(field))


def compute_signal_mass_ratio(trace: RolloutTrace) -> float:
    """Ratio of final signal field mass to creature mass at step 0.

    Measures how much chemical substance has accumulated in the field
    relative to the creature's body. Varies with emission_rate, decay_rates,
    and how long the run lasted. A pure listener (no emission) scores near 0;
    a broadcaster scores high.

    Returns 0.0 for non-signal rollouts.
    """
    if trace.final_signal_state is None or trace.initial_signal_mass == 0.0:
        return 0.0
    final_signal = float(trace.final_signal_state.sum())
    # Normalize by initial signal mass so the ratio is relative to the
    # background field that was present at the start of the rollout.
    return float(np.clip(final_signal / max(trace.initial_signal_mass, 1e-9), 0.0, 100.0))


def compute_dominant_channel_fraction(trace: RolloutTrace) -> float:
    """Fraction of total signal mass carried by the dominant channel.

    Captures the effective emission_vector direction: a creature that
    strongly emits into one chemical channel scores near 1.0; one whose
    signal spreads evenly across all channels scores near 1/C (~0.25 for
    C=4 channels).

    High values = chemical specialists; low values = generalists.
    Returns 1/C (uniform prior) for non-signal rollouts or zero signal.
    """
    if trace.final_signal_state is None:
        return 0.0
    # Sum over spatial dims -> (C,) total per channel
    per_channel = trace.final_signal_state.sum(axis=(0, 1))
    total = float(per_channel.sum())
    if total <= 0.0:
        C = trace.final_signal_state.shape[-1]
        return 1.0 / max(C, 1)
    return float(per_channel.max() / total)


REGISTRY: dict[str, Descriptor] = {
    "velocity": Descriptor(
        name="velocity",
        short_name="vel.",
        direction_label="faster",
        compute=compute_velocity,
    ),
    "gyradius": Descriptor(
        name="gyradius",
        short_name="gyr.",
        direction_label="larger",
        compute=compute_gyradius,
    ),
    "spectral_entropy": Descriptor(
        name="spectral entropy",
        short_name="spec.ent.",
        direction_label="higher entropy",
        compute=compute_spectral_entropy,
    ),
    "oscillation": Descriptor(
        name="oscillation",
        short_name="osc.",
        direction_label="more pulsing",
        compute=compute_oscillation,
    ),
    "compactness": Descriptor(
        name="compactness",
        short_name="comp.",
        direction_label="more compact",
        compute=compute_compactness,
    ),
    "mass_asymmetry": Descriptor(
        name="mass asymmetry",
        short_name="asym.",
        direction_label="more asymmetric",
        compute=compute_mass_asymmetry,
    ),
    "png_compressibility": Descriptor(
        name="PNG compressibility",
        short_name="compress.",
        direction_label="less compressible",
        compute=compute_png_compressibility,
    ),
    "rotational_symmetry": Descriptor(
        name="rotational symmetry",
        short_name="rot.sym.",
        direction_label="more asymmetric",
        compute=compute_rotational_symmetry,
    ),
    "persistence_score": Descriptor(
        name="persistence score",
        short_name="persist.",
        direction_label="less stable",
        compute=compute_persistence_score,
    ),
    "displacement_ratio": Descriptor(
        name="displacement ratio",
        short_name="displ.",
        direction_label="more glider-like",
        compute=compute_displacement_ratio,
    ),
    "angular_velocity": Descriptor(
        name="angular velocity",
        short_name="ang.vel.",
        direction_label="faster rotation",
        compute=compute_angular_velocity,
    ),
    "growth_gradient": Descriptor(
        name="growth gradient",
        short_name="grad.",
        direction_label="higher gradient",
        compute=compute_growth_gradient,
    ),
    "morphological_instability": Descriptor(
        name="morphological instability",
        short_name="morph.",
        direction_label="more unstable",
        compute=compute_morphological_instability,
    ),
    "activity": Descriptor(
        name="activity",
        short_name="act.",
        direction_label="more active",
        compute=compute_activity,
    ),
    "spatial_entropy": Descriptor(
        name="spatial entropy",
        short_name="spat.ent.",
        direction_label="more diffuse",
        compute=compute_spatial_entropy,
    ),
    # --- signal-only descriptors ---
    "signal_field_variance": Descriptor(
        name="signal field variance",
        short_name="sig.var.",
        direction_label="more localized",
        compute=compute_signal_field_variance,
        signal_only=True,
    ),
    "signal_mass_ratio": Descriptor(
        name="signal mass ratio",
        short_name="sig.ratio",
        direction_label="more accumulated",
        compute=compute_signal_mass_ratio,
        signal_only=True,
    ),
    "dominant_channel_fraction": Descriptor(
        name="dominant channel fraction",
        short_name="dom.ch.",
        direction_label="more specialized",
        compute=compute_dominant_channel_fraction,
        signal_only=True,
    ),
}

DEFAULT_DESCRIPTORS: tuple[str, str, str] = ("velocity", "gyradius", "spectral_entropy")
"""The three descriptors active when --descriptors is not passed."""


def resolve_descriptors(names: tuple[str, str, str]) -> tuple[Descriptor, Descriptor, Descriptor]:
    """Look up three descriptor names in the registry.

    Raises ValueError with a clear message if any name is unknown.
    """
    result: list[Descriptor] = []
    for name in names:
        if name not in REGISTRY:
            known = ", ".join(sorted(REGISTRY))
            raise ValueError(f"unknown descriptor {name!r}. known: {known}")
        result.append(REGISTRY[name])
    return result[0], result[1], result[2]


# === compute API ===


def compute_descriptors(
    trace: RolloutTrace,
    active: tuple[Descriptor, Descriptor, Descriptor] | None = None,
) -> Descriptors | None:
    """Compute three normalized descriptors from a rollout trace.

    Returns None if the rollout is dead (zero mass at the final step).

    When active is None, uses the default three (velocity, gyradius,
    spectral_entropy). This backward-compatible default is used by
    quality.py's persistent filter, which always compares the same three
    core observables regardless of the search's active descriptor axes.
    """
    if float(trace.final_state.sum()) <= 0.0:
        return None
    if active is None:
        return (
            compute_velocity(trace),
            compute_gyradius(trace),
            compute_spectral_entropy(trace),
        )
    return (
        active[0].compute(trace),
        active[1].compute(trace),
        active[2].compute(trace),
    )
