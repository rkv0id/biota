"""Behavior descriptors for Flow-Lenia rollouts.

Three scalar reductions, all normalized to [0, 1]:

- velocity:        mean COM step delta over the last WINDOW steps,
                   normalized against an empirical 0.02 cells/step bound
                   (calibrated against the biota parameter prior, which
                   produces creatures roughly 50x slower than the
                   Leniabreeder population)
- gyradius:        mass-weighted RMS distance from center of mass at the
                   final step, normalized by grid_size / 4 (Chan's `rm`)
- spectral_entropy: Shannon entropy of the radially-averaged FFT magnitude
                   spectrum of the final state, normalized to [0, 1]

The rollout worker tracks COM and bbox-fraction during the sim, plus the
final state, bundles them into a RolloutTrace, and passes that to
compute_descriptors.

These descriptors replace the M1 (speed, size, structure) triple, which
was demonstrated to be degenerate (see DECISIONS.md 2026-04-09 "descriptor
degeneracy diagnosis"). They also replace an interim attempt that used
Chan's growth-centroid distance (dgm) for the third axis - the dgm
descriptor turned out to be structurally degenerate for the radially
symmetric solitons biota's parameter prior produces, since the growth-
field COM and mass COM mathematically coincide for any radially symmetric
creature. See DECISIONS.md 2026-04-09 "descriptor rework iteration".

Spectral entropy captures spatial frequency content directly: a smooth
disk has all energy concentrated in low-frequency bins (low entropy),
a dotted lattice has energy spread across high frequencies (high
entropy), a bullseye has periodic mid-frequency content. This is what
the M1 Shannon-entropy structure descriptor was trying to do but did
not achieve, because Shannon entropy on a 16x16 mass histogram does
not encode spatial arrangement.

Note: gyradius and spectral_entropy are computed at the final step
regardless of the trace slice. Only velocity is window-dependent. The
persistent filter in quality.py compares early-vs-late descriptor
drift; under this design, two of the three descriptors are window-
invariant by construction, so the persistent filter effectively becomes
a velocity-stability check. That is acceptable: a creature whose shape
is stable but whose velocity is changing is genuinely non-persistent.

If the rollout died (zero mass at the final step), compute_descriptors
returns None and the rollout will be flagged as 'dead' by the alive
filter downstream.
"""

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

    Computation:

        com = (sum(mass * x) / total_mass, sum(mass * y) / total_mass)
        gyradius = sqrt(sum(mass * |xy - com|^2) / total_mass)

    Normalized by grid_size / GYRADIUS_NORMALIZER_DIVISOR (default
    grid_size / 4). A creature whose mass-weighted RMS distance equals a
    quarter of the grid edge maps to 1.0; in practice creatures stay well
    below that.

    Returns 0.0 if the final state has no mass (the alive filter will catch
    this case downstream).
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
    A smooth disk has all energy concentrated in low-frequency bins
    (entropy near 0). A dotted texture or fine lattice has energy spread
    across many spatial frequencies (entropy near 1). A bullseye with
    periodic rings has energy concentrated in a few mid-frequency bins
    (intermediate entropy).

    Computation:

        F = fft2(state)
        magnitude = |F|
        zero out the DC component (total mass, dominates and carries no
            spatial information)
        radially average into SPECTRAL_RADIAL_BINS bins by 2D wavenumber
            sqrt(kx^2 + ky^2)
        normalize the radial profile to a probability distribution
        entropy = -sum(p * log(p)) for nonzero p
        normalized = entropy / log(SPECTRAL_RADIAL_BINS)

    Why radial average instead of full 2D entropy: radial averaging is
    rotation-invariant, which is the right symmetry for biota's
    population (most creatures are roughly radially symmetric solitons).
    Two creatures that differ only by rotation should have the same
    spectral entropy.

    Why exclude DC: the DC component is the total mass, which Flow-Lenia
    conserves. Including it would dominate the magnitude spectrum across
    all creatures and dampen the entropy signal.

    Returns 0.0 for empty or constant final states.
    """
    state = trace.final_state
    if float(state.sum()) <= 0.0:
        return 0.0

    h, w = state.shape

    # 2D FFT, magnitude, zero DC
    spectrum = np.fft.fft2(state.astype(np.float64))
    magnitude = np.abs(spectrum)
    magnitude[0, 0] = 0.0

    if float(magnitude.sum()) <= 0.0:
        return 0.0

    # Build the radial wavenumber map. fft2 returns frequencies in the
    # standard layout: 0, 1, ..., N/2, -N/2+1, ..., -1. Convert to centered
    # frequency indices via fftfreq * N (in cycles-per-grid units).
    ky = np.fft.fftfreq(h) * h  # (h,)
    kx = np.fft.fftfreq(w) * w  # (w,)
    ky_grid, kx_grid = np.meshgrid(ky, kx, indexing="ij")
    radius = np.sqrt(ky_grid * ky_grid + kx_grid * kx_grid)

    # Radial bins span [0, max possible radius]. Maximum radius for an HxW
    # grid is sqrt((h/2)^2 + (w/2)^2). We bin uniformly into
    # SPECTRAL_RADIAL_BINS bins.
    r_max = float(np.sqrt((h / 2.0) ** 2 + (w / 2.0) ** 2))
    if r_max <= 0.0:
        return 0.0

    bin_indices = np.minimum(
        np.floor(radius / r_max * SPECTRAL_RADIAL_BINS).astype(np.int64),
        SPECTRAL_RADIAL_BINS - 1,
    )

    # Sum magnitude into the radial bins
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
    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


def compute_descriptors(trace: RolloutTrace) -> Descriptors | None:
    """Compute the three normalized descriptors from a rollout trace.

    Returns None if the rollout is dead (zero mass at the final step), in
    which case the alive filter downstream will reject it with reason 'dead'.
    """
    if float(trace.final_state.sum()) <= 0.0:
        return None
    return (
        compute_velocity(trace),
        compute_gyradius(trace),
        compute_spectral_entropy(trace),
    )
