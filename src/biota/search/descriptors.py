"""Behavior descriptors for Flow-Lenia rollouts.

Three scalar reductions, all normalized to [0, 1]:

- velocity:  mean COM step delta over the last WINDOW steps, normalized
             against the empirical Leniabreeder bound (0.5 cells/step)
- gyradius:  mass-weighted RMS distance from center of mass at the final
             step, normalized by grid_size / 4 (Chan's `rm`)
- dgm:       distance between center of mass of the growth field and
             center of mass of the actual mass, at the final step,
             normalized by grid_size (Chan's growth-centroid distance)

The rollout worker tracks COM, bbox-fraction, and the final state during the
sim, plus computes the growth field once at the final step, bundles them into
a RolloutTrace, and passes that to compute_descriptors.

These descriptors replace an earlier (speed, size, structure) triple from
M1 that was demonstrated to be degenerate (see DECISIONS.md 2026-04-09
"descriptor degeneracy diagnosis"). The new design follows Chan (2019) and
Faldor & Cully Leniabreeder (2024).

Note: gyradius and dgm are computed at the final step regardless of the
trace slice. Only velocity is window-dependent. The persistent filter in
quality.py compares early-vs-late descriptor drift; under this design,
two of the three descriptors are window-invariant by construction, so
the persistent filter effectively becomes a velocity-stability check.
That is acceptable: a creature whose shape is stable but whose velocity
is changing (accelerating, decelerating, or rotating) is genuinely
non-persistent and should be rejected.

If the rollout died (zero mass at the final step), compute_descriptors
returns None and the rollout will be flagged as 'dead' by the alive
filter downstream.
"""

from dataclasses import dataclass

import numpy as np

from biota.search.result import Descriptors

WINDOW = 50
"""Number of trailing steps used to compute the velocity descriptor."""

VELOCITY_NORMALIZER = 0.5
"""Empirical maximum velocity in cells-per-step. Calibrated against the
Leniabreeder bound (Faldor & Cully 2024). Real Lenia solitons (e.g. Aquarium
at 0.12 cells/step) sit comfortably below 0.2; this normalizer gives the
descriptor meaningful spread in the bottom half of [0, 1] without pinning
fast creatures to 1.0."""

GYRADIUS_NORMALIZER_DIVISOR = 4.0
"""Gyradius is normalized by grid_size / GYRADIUS_NORMALIZER_DIVISOR. A
creature whose mass-weighted RMS distance from center equals grid_size/4
saturates the descriptor at 1.0; in practice, even diffuse creatures sit
well below that bound."""


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
        final_growth_field: The per-cell growth field at the final step (the
            same `U_sum` quantity that drives the flow term inside FlowLenia).
            Shape (grid_size, grid_size), dtype float32. Used by the dgm
            (growth-centroid-distance) descriptor.
        grid_size: The simulation grid edge length, used for descriptor
            normalization.
        total_steps: The total number of simulation steps in the rollout.
    """

    com_history: np.ndarray
    bbox_fraction_history: np.ndarray
    final_state: np.ndarray
    final_growth_field: np.ndarray
    grid_size: int
    total_steps: int

    def slice(self, start: int, end: int) -> "RolloutTrace":
        """Return a trace covering steps [start, end) within the original window.

        Used by the persistent filter to compute descriptors at two different
        time points. The final_state and final_growth_field are unchanged
        (they are still the last step of the original rollout) since the
        gyradius and dgm descriptors always read the final step.
        """
        return RolloutTrace(
            com_history=self.com_history[start:end],
            bbox_fraction_history=self.bbox_fraction_history[start:end],
            final_state=self.final_state,
            final_growth_field=self.final_growth_field,
            grid_size=self.grid_size,
            total_steps=self.total_steps,
        )


def compute_velocity(trace: RolloutTrace) -> float:
    """Mean COM step-delta over the last WINDOW steps, normalized to [0, 1].

    Velocity is the L2 norm of the per-step center-of-mass displacement,
    averaged across the trailing WINDOW steps. The normalizer is
    VELOCITY_NORMALIZER (0.5 cells/step), an empirical bound calibrated
    against the Leniabreeder paper. Real Lenia creatures top out around
    0.2 cells/step in practice; the choice of 0.5 keeps the descriptor's
    useful range in roughly [0, 0.4] of the normalized scale, which is
    enough resolution to discriminate stationary from drifting from
    fast-moving creatures.
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


def compute_growth_centroid_distance(trace: RolloutTrace) -> float:
    """Distance between growth-field COM and mass COM, normalized to [0, 1].

    This is Chan's growth-centroid distance (dgm). It captures whether the
    creature's growth pressure is aligned with its mass distribution
    (symmetric, stable creatures: dgm near 0) or asymmetric (structured or
    motile creatures: dgm > 0). Distinguishes a smooth uniform disk from a
    creature with internal pattern that the Shannon-entropy-based M1
    structure descriptor could not separate.

    Computation:

        com_mass = mass-weighted center of the final state
        com_growth = mass-weighted center of the growth field magnitude
        dgm = ||com_growth - com_mass|| / grid_size

    The growth field can have negative values (cells where mass should
    shrink), so we weight by its absolute value to get a center of "growth
    activity" rather than a signed sum. The result is normalized by
    grid_size: a creature whose growth and mass centers are a full grid
    edge apart would saturate at 1.0, which can only happen for
    pathological non-localized creatures that the localized filter
    rejects anyway.

    Returns 0.0 if either the mass or the growth-field absolute values are
    zero.
    """
    state = trace.final_state
    growth = trace.final_growth_field

    total_mass = float(state.sum())
    if total_mass <= 0.0:
        return 0.0

    growth_abs = np.abs(growth)
    total_growth = float(growth_abs.sum())
    if total_growth <= 0.0:
        return 0.0

    h, w = state.shape
    y_coords, x_coords = np.indices((h, w), dtype=np.float32)

    com_mass_y = float((state * y_coords).sum() / total_mass)
    com_mass_x = float((state * x_coords).sum() / total_mass)
    com_growth_y = float((growth_abs * y_coords).sum() / total_growth)
    com_growth_x = float((growth_abs * x_coords).sum() / total_growth)

    dy = com_growth_y - com_mass_y
    dx = com_growth_x - com_mass_x
    distance = float(np.sqrt(dy * dy + dx * dx))

    if trace.grid_size <= 0:
        return 0.0
    return float(np.clip(distance / trace.grid_size, 0.0, 1.0))


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
        compute_growth_centroid_distance(trace),
    )
