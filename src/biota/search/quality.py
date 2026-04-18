"""Quality function: filter-then-rank.

Three hard filters must pass for a rollout to qualify:

- alive:      conserved mass (mass + signal) within [0.5, 2.0] * initial total,
              AND mass field >= CREATURE_MASS_FLOOR * initial_mass (signal runs only)
- localized:  mean bbox-fraction over the last WINDOW steps < 0.6
- persistent: descriptor drift across adjacent 50-step windows <= 0.2

Survivors are ranked by a three-component quality score:

    q = w_c * compactness + w_s * stability + w_r * retention

where:

    compactness  = min(compact(state_T/2), compact(state_T))
                   Minimum compactness at the midpoint and final step.
                   Penalises creatures that peak early and degrade over time.
                   compact(s) = fraction of total mass inside the bounding box.

    stability    = clip(1 - drift / PERSISTENT_DESCRIPTOR_DRIFT, 0, 1)
                   Continuous version of the persistent filter. Creatures that
                   barely pass (drift near 0.2) score near 0; rock-stable ones
                   (drift near 0) score near 1.

    retention    = clip(final_mass / initial_mass, 0, 1)
                   Rewards creatures that keep their mass field intact.
                   For non-signal rollouts this is always ~1; for signal runs
                   it directly penalises runaway emission.

Weights:
    Non-signal:  w_c = 0.6,  w_s = 0.4,  w_r = 0.0
    Signal:      w_c = 0.5,  w_s = 0.3,  w_r = 0.2

The midpoint compactness term is the key addition over the old single-point
metric. Most Flow-Lenia solitons score >0.95 compactness at the final step,
making the old metric nearly constant across the population. Comparing with
the midpoint state catches creatures that start well-defined but become
increasingly diffuse or fragmented -- exactly the kind of instability that
matters for long ecosystem runs.
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
# Signal creatures: mass field must not drop below this fraction of initial_mass
# even when total (mass + signal) is conserved.
CREATURE_MASS_FLOOR = 0.2

# Quality component weights
_W_COMPACT_BASE = 0.6
_W_STABLE_BASE = 0.4
_W_COMPACT_SIG = 0.5
_W_STABLE_SIG = 0.3
_W_RETAIN_SIG = 0.2


@dataclass(frozen=True)
class RolloutEvaluation:
    """Inputs the quality function needs to evaluate a rollout.

    Attributes:
        initial_mass:      Total mass at step 0.
        final_mass:        Mass field total at the final step.
        trace:             RolloutTrace covering the last >=100 steps.
                           trace.midpoint_state is used for the two-point
                           compactness term when available.
        initial_total:     initial_mass + initial_signal_mass. Defaults to
                           initial_mass for non-signal rollouts.
        final_signal_mass: Total signal field mass at the final step.
                           Zero for non-signal rollouts.
    """

    initial_mass: float
    final_mass: float
    trace: RolloutTrace
    initial_total: float = 0.0
    final_signal_mass: float = 0.0

    def __post_init__(self) -> None:
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
    """Total-mass conservation check plus creature mass floor for signal runs."""
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
) -> tuple[bool, float, Descriptors | None]:
    """Compare descriptors at two adjacent 50-step windows.

    Returns (passes_filter, drift_value, early_descriptors).
    drift_value is in [0, inf); passes_filter is True when drift <= threshold.
    The continuous drift value is used by the stability quality component.
    """
    total = len(trace.com_history)
    if total < 2 * WINDOW:
        return False, float(PERSISTENT_DESCRIPTOR_DRIFT), None

    early_slice = trace.slice(total - 2 * WINDOW, total - WINDOW)
    early = compute_descriptors(early_slice, active=active)
    if early is None:
        return False, float(PERSISTENT_DESCRIPTOR_DRIFT), None

    drift = max(abs(a - b) for a, b in zip(late_descriptors, early, strict=True))
    return drift <= PERSISTENT_DESCRIPTOR_DRIFT, float(drift), early


def _compactness(state: np.ndarray) -> float:
    """Fraction of total mass inside the bounding box of a 2D state array.

    The bbox uses a threshold of 0.1 * peak mass. Returns 0 for dead or
    degenerate states.
    """
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

    Order: cheap filters first (alive, localized), then descriptor computation,
    then persistent filter, then multi-component quality score.

    Quality components:
        compactness  min(compact at midpoint, compact at final step)
        stability    1 - drift / PERSISTENT_DESCRIPTOR_DRIFT, clipped [0,1]
        retention    final_mass / initial_mass, clipped [0,1] (signal only)

    Weights (non-signal): compactness=0.6, stability=0.4
    Weights (signal):     compactness=0.5, stability=0.3, retention=0.2
    """
    if not _alive(eval_input):
        return EvaluationResult(descriptors=None, quality=None, rejection_reason="dead")

    if not _localized(eval_input.trace):
        return EvaluationResult(descriptors=None, quality=None, rejection_reason="exploded")

    descriptors = compute_descriptors(eval_input.trace, active=active_descriptors)
    if descriptors is None:
        return EvaluationResult(descriptors=None, quality=None, rejection_reason="no_descriptors")

    passes, drift, _ = _persistent(eval_input.trace, descriptors, active=active_descriptors)
    if not passes:
        return EvaluationResult(descriptors=descriptors, quality=None, rejection_reason="unstable")

    # --- Multi-component quality score ---

    # Compactness: min of midpoint and final to catch early-peak degraders.
    c_final = _compactness(eval_input.trace.final_state)
    if eval_input.trace.midpoint_state is not None:
        c_mid = _compactness(eval_input.trace.midpoint_state)
        compactness = min(c_mid, c_final)
    else:
        compactness = c_final

    # Stability: continuous version of the persistent filter.
    stability = float(np.clip(1.0 - drift / PERSISTENT_DESCRIPTOR_DRIFT, 0.0, 1.0))

    # Retention: mass field preservation (meaningful for signal runs).
    is_signal = (
        eval_input.final_signal_mass > 0 or eval_input.initial_total > eval_input.initial_mass
    )
    if is_signal:
        retention = float(
            np.clip(
                eval_input.final_mass / eval_input.initial_mass
                if eval_input.initial_mass > 0
                else 1.0,
                0.0,
                1.0,
            )
        )
        quality = (
            _W_COMPACT_SIG * compactness + _W_STABLE_SIG * stability + _W_RETAIN_SIG * retention
        )
    else:
        quality = _W_COMPACT_BASE * compactness + _W_STABLE_BASE * stability

    return EvaluationResult(descriptors=descriptors, quality=quality, rejection_reason=None)
