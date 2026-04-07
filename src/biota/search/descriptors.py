"""Behavior descriptors for Flow-Lenia rollouts.

Three scalar reductions, all normalized to [0, 1]:

- speed:     COM velocity magnitude averaged over last 50 steps
- size:      bbox-area-fraction averaged over last 50 steps
- structure: Shannon entropy of mass within the final-step bbox, 16x16 binning

The rollout worker tracks COM, bbox-fraction, and the final state during the
sim, bundles them into a RolloutTrace, and passes that to compute_descriptors.
The trace is small (a few hundred floats plus the final 96x96 grid) and lives
only inside the worker - it does not cross the driver/worker boundary.

If the rollout died (zero mass at the final step), compute_descriptors returns
None and the rollout will be flagged as 'dead' by the alive filter downstream.
"""

from dataclasses import dataclass

import numpy as np

from biota.search.result import Descriptors

WINDOW = 50
"""Number of trailing steps used to compute speed and size descriptors."""

STRUCTURE_BINS = 16
"""Side length of the structure-entropy histogram, in bins."""


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
            step in the trailing window. Shape (T,). Already computed against
            the per-step peak mass, so this is comparable across steps without
            re-normalization.
        final_state: The simulation state at the final step. Shape
            (grid_size, grid_size), dtype float32.
        grid_size: The simulation grid edge length, used for speed
            normalization.
        total_steps: The total number of simulation steps in the rollout, used
            for speed normalization.
    """

    com_history: np.ndarray
    bbox_fraction_history: np.ndarray
    final_state: np.ndarray
    grid_size: int
    total_steps: int

    def slice(self, start: int, end: int) -> "RolloutTrace":
        """Return a trace covering steps [start, end) within the original window.

        Used by the persistent filter to compute descriptors at two different
        time points. The final_state is unchanged (it's still the last step of
        the original rollout) since it's only used by the structure descriptor
        which always reads the final step.
        """
        return RolloutTrace(
            com_history=self.com_history[start:end],
            bbox_fraction_history=self.bbox_fraction_history[start:end],
            final_state=self.final_state,
            grid_size=self.grid_size,
            total_steps=self.total_steps,
        )


def compute_speed(trace: RolloutTrace) -> float:
    """Mean COM velocity magnitude over the last WINDOW steps, normalized to [0, 1].

    The normalizer is 0.5 * grid_size / total_steps, which is a generous upper
    bound: a creature traveling half the grid over the entire rollout would
    score 1.0. Real creatures rarely exceed 0.3.
    """
    coms = trace.com_history[-WINDOW:]
    if len(coms) < 2:
        return 0.0
    deltas = np.diff(coms, axis=0)
    speeds = np.linalg.norm(deltas, axis=1)
    mean_speed = float(speeds.mean())
    normalizer = 0.5 * trace.grid_size / max(trace.total_steps, 1)
    return float(np.clip(mean_speed / normalizer, 0.0, 1.0))


def compute_size(trace: RolloutTrace) -> float:
    """Mean bbox area as a fraction of grid area over the last WINDOW steps.

    Already in [0, 1] by construction; the clip is defensive.
    """
    fractions = trace.bbox_fraction_history[-WINDOW:]
    if len(fractions) == 0:
        return 0.0
    return float(np.clip(fractions.mean(), 0.0, 1.0))


def compute_structure(trace: RolloutTrace) -> float:
    """Shannon entropy of mass within the final-step bbox, 16x16 binning, normalized to [0, 1].

    Returns 0 if the final state has zero mass or a degenerate bbox. Returns
    1.0 if mass is uniformly distributed across all 256 bins. Natural log
    throughout.
    """
    state = trace.final_state
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
    if y_max - y_min < 1 or x_max - x_min < 1:
        return 0.0

    bbox = state[y_min:y_max, x_min:x_max]
    bbox_mass = float(bbox.sum())
    if bbox_mass <= 0.0:
        return 0.0

    # Bin the bbox into 16x16 sub-cells via histogram2d.
    h, w = bbox.shape
    y_coords, x_coords = np.indices(bbox.shape)
    weights = bbox.ravel()
    hist, _, _ = np.histogram2d(
        y_coords.ravel(),
        x_coords.ravel(),
        bins=[STRUCTURE_BINS, STRUCTURE_BINS],
        range=[[0, h], [0, w]],
        weights=weights,
    )

    p = hist.ravel()
    total = p.sum()
    if total <= 0.0:
        return 0.0
    p = p / total
    nonzero = p[p > 0.0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    max_entropy = float(np.log(STRUCTURE_BINS * STRUCTURE_BINS))
    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


def compute_descriptors(trace: RolloutTrace) -> Descriptors | None:
    """Compute the three normalized descriptors from a rollout trace.

    Returns None if the rollout is dead (zero mass at the final step), in
    which case the alive filter downstream will reject it with reason 'dead'.
    """
    if float(trace.final_state.sum()) <= 0.0:
        return None
    return (
        compute_speed(trace),
        compute_size(trace),
        compute_structure(trace),
    )
