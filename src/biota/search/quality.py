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
    Descriptor,
    RolloutTrace,
    compute_descriptors,
)
from biota.search.result import Descriptors

ALIVE_LOWER = 0.5
ALIVE_UPPER = 2.0
LOCALIZED_THRESHOLD = 0.6
PERSISTENT_DESCRIPTOR_DRIFT = 0.2
# For signal creatures, the alive filter uses a stricter mass floor: the creature's
# mass field must not collapse below this fraction of initial_mass even if total
# (mass + signal) is conserved. Higher value = stronger selection against mass bleed.
CREATURE_MASS_FLOOR = 0.2
# For non-signal rollouts the original looser floor is used (mass floor = 0 effectively,
# since the total-mass check already enforces >=0.5 * initial_mass).


@dataclass(frozen=True)
class RolloutEvaluation:
    """Inputs the quality function needs to evaluate a rollout.

    Attributes:
        initial_mass:  Total mass at step 0 (mass field only; signal field
                       starts from a known low background, not zero).
        final_mass:    Mass field total at the final step.
        trace:         RolloutTrace covering the last >=100 steps.
        initial_total: initial_mass + initial_signal_mass. Used by the alive
                       filter when signal field is active. Equal to initial_mass
                       for non-signal rollouts (signal starts at 0).
        final_signal_mass: Total signal field mass at the final step. Zero for
                       non-signal rollouts.
    """

    initial_mass: float
    final_mass: float
    trace: RolloutTrace
    initial_total: float = 0.0
    final_signal_mass: float = 0.0

    def __post_init__(self) -> None:
        # Default initial_total to initial_mass when not supplied (non-signal path).
        if self.initial_total == 0.0:
            object.__setattr__(self, "initial_total", self.initial_mass)


@dataclass(frozen=True)
class EvaluationResult:
    """Output of the quality function.

    descriptors is None only when the rollout is dead (zero mass at final step).
    quality is None when any filter rejected; rejection_reason names which one.
    """

    descriptors: Descriptors | None
    quality: float | None
    rejection_reason: str | None


def _alive(eval_input: RolloutEvaluation) -> bool:
    """Two-part alive check for signal-aware rollouts.

    1. Total conserved mass (mass + signal) stays within [0.5, 2.0] of
       the initial total. This handles the signal field correctly: a creature
       that emits into the signal field reduces its mass field mass but the
       total should remain conserved.

    2. Mass field alone must not collapse below CREATURE_MASS_FLOOR fraction
       of initial_mass. Without this, a pure emitter that converts all mass
       to signal passes condition 1 but has no creature left.

    For non-signal rollouts, initial_total == initial_mass and
    final_signal_mass == 0, so this reduces to the original check.
    """
    final_total = eval_input.final_mass + eval_input.final_signal_mass
    init = eval_input.initial_total
    if not (ALIVE_LOWER * init <= final_total <= ALIVE_UPPER * init):
        return False
    return not (
        eval_input.initial_mass > 0
        and eval_input.final_mass < CREATURE_MASS_FLOOR * eval_input.initial_mass
    )


def _localized(trace: RolloutTrace) -> bool:
    fractions = trace.bbox_fraction_history[-WINDOW:]
    if len(fractions) == 0:
        return False
    return float(fractions.mean()) < LOCALIZED_THRESHOLD


def _persistent(
    trace: RolloutTrace,
    late_descriptors: Descriptors,
    active: tuple[Descriptor, Descriptor, Descriptor] | None = None,
) -> tuple[bool, Descriptors | None]:
    """Compare descriptors at two adjacent 50-step windows.

    Returns (is_persistent, early_descriptors). The early descriptors are
    returned for diagnostics and possible reuse, but the canonical descriptors
    used downstream are the late ones.

    active is passed through to compute_descriptors so early and late are
    computed on the same descriptor axes.
    """
    total = len(trace.com_history)
    if total < 2 * WINDOW:
        return False, None

    early_slice = trace.slice(total - 2 * WINDOW, total - WINDOW)
    early = compute_descriptors(early_slice, active=active)
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


def evaluate(
    eval_input: RolloutEvaluation,
    active_descriptors: tuple[Descriptor, Descriptor, Descriptor] | None = None,
) -> EvaluationResult:
    """Run the filter-then-rank quality pipeline on a rollout.

    Order matters: cheap filters first, then descriptor computation, then
    persistent filter, then compactness. Each gate produces a specific
    rejection reason if it fails.

    active_descriptors controls which three descriptors are computed for the
    archive coordinates. When None, uses (velocity, gyradius, spectral_entropy).
    """
    if not _alive(eval_input):
        return EvaluationResult(descriptors=None, quality=None, rejection_reason="dead")

    if not _localized(eval_input.trace):
        return EvaluationResult(descriptors=None, quality=None, rejection_reason="exploded")

    descriptors = compute_descriptors(eval_input.trace, active=active_descriptors)
    if descriptors is None:
        return EvaluationResult(descriptors=None, quality=None, rejection_reason="no_descriptors")

    is_persistent, _ = _persistent(eval_input.trace, descriptors, active=active_descriptors)
    if not is_persistent:
        return EvaluationResult(descriptors=descriptors, quality=None, rejection_reason="unstable")

    compactness = _compactness(eval_input.trace)
    # Signal retention bonus: reward creatures that keep their mass field stable
    # under signal emission. Only applied when initial_total > initial_mass (i.e.
    # a signal run with a non-trivial background field) OR when final_signal_mass > 0.
    if eval_input.final_signal_mass > 0 or eval_input.initial_total > eval_input.initial_mass:
        retention = float(
            eval_input.final_mass / eval_input.initial_mass if eval_input.initial_mass > 0 else 1.0
        )
        retention = max(0.0, min(1.0, retention))
        quality = 0.7 * compactness + 0.3 * retention
    else:
        quality = compactness
    return EvaluationResult(descriptors=descriptors, quality=quality, rejection_reason=None)
