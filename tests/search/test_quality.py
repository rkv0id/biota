"""Tests for the filter-then-rank quality function."""

import numpy as np

from biota.search.descriptors import WINDOW, RolloutTrace
from biota.search.quality import (
    EvaluationResult,
    RolloutEvaluation,
    evaluate,
)

GRID = 96
STEPS = 300
TRACE_LEN = 2 * WINDOW  # 100, the minimum for the persistent filter


def _make_trace(
    *,
    com_path: np.ndarray | None = None,
    bbox_history: np.ndarray | None = None,
    final_state: np.ndarray | None = None,
) -> RolloutTrace:
    if com_path is None:
        com_path = np.full((TRACE_LEN, 2), 48.0, dtype=np.float32)
    if bbox_history is None:
        bbox_history = np.full(TRACE_LEN, 0.04, dtype=np.float32)
    if final_state is None:
        final_state = np.zeros((GRID, GRID), dtype=np.float32)
        final_state[40:50, 40:50] = 1.0
    # Quality tests don't exercise the dgm descriptor (they use zero or
    # uniform growth fields), so the default growth field is zeros - which
    # makes the dgm computation return 0.0 via the early-out for zero total
    # growth. Tests that need real growth-field behavior should override.
    growth = np.zeros_like(final_state)
    return RolloutTrace(
        com_history=com_path.astype(np.float32),
        bbox_fraction_history=bbox_history.astype(np.float32),
        final_state=final_state.astype(np.float32),
        final_growth_field=growth.astype(np.float32),
        grid_size=GRID,
        total_steps=STEPS,
    )


def _eval(
    *,
    initial_mass: float = 100.0,
    final_mass: float = 100.0,
    trace: RolloutTrace | None = None,
) -> EvaluationResult:
    return evaluate(
        RolloutEvaluation(
            initial_mass=initial_mass,
            final_mass=final_mass,
            trace=trace if trace is not None else _make_trace(),
        )
    )


# === alive filter ===


def test_alive_filter_rejects_dead() -> None:
    result = _eval(initial_mass=100.0, final_mass=10.0)
    assert result.quality is None
    assert result.rejection_reason == "dead"


def test_alive_filter_rejects_explosion_in_mass() -> None:
    result = _eval(initial_mass=100.0, final_mass=500.0)
    assert result.quality is None
    assert result.rejection_reason == "dead"


def test_alive_filter_passes_at_lower_bound() -> None:
    result = _eval(initial_mass=100.0, final_mass=50.0)
    assert result.rejection_reason != "dead"


def test_alive_filter_passes_at_upper_bound() -> None:
    result = _eval(initial_mass=100.0, final_mass=200.0)
    assert result.rejection_reason != "dead"


# === localized filter ===


def test_localized_filter_rejects_when_bbox_too_large() -> None:
    bbox = np.full(TRACE_LEN, 0.7, dtype=np.float32)
    result = _eval(trace=_make_trace(bbox_history=bbox))
    assert result.quality is None
    assert result.rejection_reason == "exploded"


def test_localized_filter_passes_when_bbox_small_enough() -> None:
    bbox = np.full(TRACE_LEN, 0.3, dtype=np.float32)
    result = _eval(trace=_make_trace(bbox_history=bbox))
    assert result.rejection_reason != "exploded"


# === persistent filter ===


def test_persistent_filter_rejects_drifting_speed() -> None:
    # COM stationary for the first 50 steps, then moves fast for the next 50
    coms = np.zeros((TRACE_LEN, 2), dtype=np.float32)
    coms[WINDOW:, 0] = np.arange(WINDOW) * 2.0  # 2 cells/step in the late window
    bbox = np.full(TRACE_LEN, 0.1, dtype=np.float32)
    state = np.zeros((GRID, GRID), dtype=np.float32)
    state[40:60, 40:60] = 1.0
    result = _eval(trace=_make_trace(com_path=coms, bbox_history=bbox, final_state=state))
    assert result.quality is None
    assert result.rejection_reason == "unstable"


def test_persistent_filter_passes_when_descriptors_stable() -> None:
    coms = np.zeros((TRACE_LEN, 2), dtype=np.float32)
    bbox = np.full(TRACE_LEN, 0.1, dtype=np.float32)
    result = _eval(trace=_make_trace(com_path=coms, bbox_history=bbox))
    assert result.rejection_reason != "unstable"


# === compactness ranker ===


def test_compactness_ranks_concentrated_mass_high() -> None:
    # Single dense blob, no scattered mass
    state = np.zeros((GRID, GRID), dtype=np.float32)
    state[40:60, 40:60] = 1.0
    result = _eval(trace=_make_trace(final_state=state))
    assert result.quality is not None
    assert result.quality > 0.95


def test_compactness_low_with_scattered_background() -> None:
    # A small dense blob plus scattered noise everywhere
    state = np.full((GRID, GRID), 0.05, dtype=np.float32)  # background
    state[45:55, 45:55] = 1.0  # blob
    result = _eval(trace=_make_trace(final_state=state))
    assert result.quality is not None
    assert result.quality < 0.6


# === full pipeline ===


def test_clean_creature_passes_all_filters() -> None:
    state = np.zeros((GRID, GRID), dtype=np.float32)
    state[40:55, 40:55] = 1.0
    result = _eval(
        initial_mass=100.0,
        final_mass=100.0,
        trace=_make_trace(final_state=state),
    )
    assert result.descriptors is not None
    assert result.quality is not None
    assert result.rejection_reason is None
    assert 0.0 <= result.quality <= 1.0
    for d in result.descriptors:
        assert 0.0 <= d <= 1.0


def test_short_trace_persistence_rejects() -> None:
    short_coms = np.zeros((WINDOW, 2), dtype=np.float32)
    short_bbox = np.full(WINDOW, 0.1, dtype=np.float32)
    state = np.zeros((GRID, GRID), dtype=np.float32)
    state[40:50, 40:50] = 1.0
    trace = RolloutTrace(
        com_history=short_coms,
        bbox_fraction_history=short_bbox,
        final_state=state,
        final_growth_field=np.zeros_like(state),
        grid_size=GRID,
        total_steps=STEPS,
    )
    result = _eval(trace=trace)
    assert result.rejection_reason == "unstable"
