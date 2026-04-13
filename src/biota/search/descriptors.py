"""Behavior descriptors for Flow-Lenia rollouts.

Descriptors are normalized scalar reductions over a RolloutTrace, each
returning a float in [0, 1]. They define the behavioral axes of the MAP-Elites
archive. The search always uses exactly three active descriptors.

Built-in descriptor library (9 total):

  velocity             mean COM displacement/step over trailing WINDOW steps
  gyradius             mass-weighted RMS distance from COM at final step
  spectral_entropy     Shannon entropy of radially-averaged FFT spectrum
  oscillation          variance of bbox fraction over trace tail
  compactness          mass inside bbox / total mass at final step
  mass_asymmetry       |mean COM_x drift - mean COM_y drift| / sum
  png_compressibility  PNG compressed / uncompressed size of final state
  rotational_symmetry  angular variance of radial mass profile at final step
  persistence_score    max drift of the three core descriptors across trace tail

Custom descriptors can be loaded from a user Python file via --descriptor-module;
see cli.py. The file must define a list named DESCRIPTORS containing Descriptor
objects. Loaded descriptors are merged into the registry at startup.

A Descriptor's compute function must return float in [0, 1]. Out-of-range
values are silently clipped during archive cell assignment. Validate your
normalizers against real rollout data before running a full search.
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

    The compute function receives a RolloutTrace and must return a float
    in [0, 1]. Values outside this range are clipped at archive cell
    assignment; they do not raise an error but will silently produce
    degenerate archive coverage.

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
        final_state: The simulation state at the final step. Shape
            (grid_size, grid_size), dtype float32.
        grid_size: The simulation grid edge length, used for descriptor
            normalization.
        total_steps: The total number of simulation steps in the rollout.
    """

    com_history: np.ndarray
    bbox_fraction_history: np.ndarray
    final_state: np.ndarray
    grid_size: int
    total_steps: int

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
            final_state=self.final_state,
            grid_size=self.grid_size,
            total_steps=self.total_steps,
        )


# === built-in compute functions ===


def compute_velocity(trace: RolloutTrace) -> float:
    """Mean COM step-delta over the last WINDOW steps, normalized to [0, 1].

    Velocity is the L2 norm of the per-step center-of-mass displacement,
    averaged across the trailing WINDOW steps. The normalizer is
    VELOCITY_NORMALIZER (0.02 cells/step), an empirical bound calibrated
    against an actual biota cluster run.
    """
    coms = trace.com_history[-WINDOW:]
    if len(coms) < 2:
        return 0.0
    deltas = np.diff(coms, axis=0)
    speeds = np.linalg.norm(deltas, axis=1)
    mean_speed = float(speeds.mean())
    return float(np.clip(mean_speed / VELOCITY_NORMALIZER, 0.0, 1.0))


def compute_gyradius(trace: RolloutTrace) -> float:
    """Mass-weighted RMS distance from center of mass, normalized to [0, 1].

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

    normalizer = trace.grid_size / GYRADIUS_NORMALIZER_DIVISOR
    if normalizer <= 0.0:
        return 0.0
    return float(np.clip(gyradius / normalizer, 0.0, 1.0))


def compute_spectral_entropy(trace: RolloutTrace) -> float:
    """Shannon entropy of the radially-averaged FFT magnitude spectrum, in [0, 1].

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
    remapped = (raw - SPECTRAL_ENTROPY_FLOOR) / (1.0 - SPECTRAL_ENTROPY_FLOOR)
    return float(np.clip(remapped, 0.0, 1.0))


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
    return float(np.clip(variance / 0.05, 0.0, 1.0))


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
    return float(np.clip(bbox_mass / total_mass, 0.0, 1.0))


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
    return float(np.clip(abs(mean_x - mean_y) / total, 0.0, 1.0))


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
    return float(np.clip(len(compressed) / len(raw_bytes), 0.0, 1.0))


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
    return float(np.clip(variance / max_variance, 0.0, 1.0))


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

    drift_threshold = 0.2  # matches PERSISTENT_DESCRIPTOR_DRIFT in quality.py
    max_drift = max(abs(a - b) for a, b in zip(late, early, strict=True))
    return float(np.clip(max_drift / drift_threshold, 0.0, 1.0))


# === registry ===


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
