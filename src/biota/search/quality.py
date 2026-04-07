"""Quality function: filter-then-rank.

Three filters must pass for a rollout to be considered alive and stable:

- alive:      final mass within [0.5, 2.0] * initial mass
- localized:  mean bbox-fraction over the last WINDOW steps < 0.6
- persistent: descriptors over the last 50 steps are within 0.2 (normalized
              units) of descriptors over the preceding 50 steps

Survivors are ranked by compactness: the fraction of total mass that lives
inside the bounding box at the final step. Compactness in [0, 1], where 1.0
means all mass is concentrated inside the bbox and 0.0 means it's all
scattered as background noise.

Structure is a descriptor axis (used by the archive for diversity), NOT a
quality multiplier. Quality and structure are decoupled.
"""

from dataclasses import dataclass

import numpy as np

from biota.search.descriptors import (
    WINDOW,
    RolloutTrace,
    compute_descriptors,
)
from biota.search.result import Descriptors

ALIVE_LOWER = 0.5
ALIVE_UPPER = 2.0
LOCALIZED_THRESHOLD = 0.6
PERSISTENT_DESCRIPTOR_DRIFT = 0.2


@dataclass(frozen=True)
class RolloutEvaluation:
    """Inputs the quality function needs to evaluate a rollout.

    Attributes:
        initial_mass: Total mass at step 0.
        final_mass: Total mass at the final step.
        trace: RolloutTrace covering the last >=100 steps. The trace must have
            at least 2*WINDOW steps for the persistent filter to be evaluable.
    """

    initial_mass: float
    final_mass: float
    trace: RolloutTrace


@dataclass(frozen=True)
class EvaluationResult:
    """Output of the quality function.

    descriptors is None only when the rollout is dead (zero mass at final step).
    quality is None when any filter rejected; rejection_reason names which one.
    """

    descriptors: Descriptors | None
    quality: float | None
    rejection_reason: str | None


def _alive(initial_mass: float, final_mass: float) -> bool:
    return ALIVE_LOWER * initial_mass <= final_mass <= ALIVE_UPPER * initial_mass


def _localized(trace: RolloutTrace) -> bool:
    fractions = trace.bbox_fraction_history[-WINDOW:]
    if len(fractions) == 0:
        return False
    return float(fractions.mean()) < LOCALIZED_THRESHOLD


def _persistent(
    trace: RolloutTrace, late_descriptors: Descriptors
) -> tuple[bool, Descriptors | None]:
    """Compare descriptors at two adjacent 50-step windows.

    Returns (is_persistent, early_descriptors). The early descriptors are
    returned for diagnostics and possible reuse, but the canonical descriptors
    used downstream are the late ones.
    """
    total = len(trace.com_history)
    if total < 2 * WINDOW:
        return False, None

    early_slice = trace.slice(total - 2 * WINDOW, total - WINDOW)
    early = compute_descriptors(early_slice)
    if early is None:
        return False, None

    drift = max(abs(a - b) for a, b in zip(late_descriptors, early, strict=True))
    return drift <= PERSISTENT_DESCRIPTOR_DRIFT, early


def _compactness(trace: RolloutTrace) -> float:
    """Fraction of total mass inside the final-step bounding box.

    The bbox uses the same threshold as the size descriptor (0.1 * peak).
    Returns 0 for dead or degenerate states.
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


def evaluate(eval_input: RolloutEvaluation) -> EvaluationResult:
    """Run the filter-then-rank quality pipeline on a rollout.

    Order matters: cheap filters first, then descriptor computation, then
    persistent filter, then compactness. Each gate produces a specific
    rejection reason if it fails.
    """
    if not _alive(eval_input.initial_mass, eval_input.final_mass):
        return EvaluationResult(descriptors=None, quality=None, rejection_reason="dead")

    if not _localized(eval_input.trace):
        return EvaluationResult(descriptors=None, quality=None, rejection_reason="exploded")

    descriptors = compute_descriptors(eval_input.trace)
    if descriptors is None:
        return EvaluationResult(descriptors=None, quality=None, rejection_reason="no_descriptors")

    is_persistent, _ = _persistent(eval_input.trace, descriptors)
    if not is_persistent:
        return EvaluationResult(descriptors=descriptors, quality=None, rejection_reason="unstable")

    quality = _compactness(eval_input.trace)
    return EvaluationResult(descriptors=descriptors, quality=quality, rejection_reason=None)
