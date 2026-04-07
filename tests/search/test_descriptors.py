"""Tests for descriptor functions.

Each test constructs a synthetic RolloutTrace where the expected descriptor
value is computable by hand, then asserts the function reproduces it.
"""

import numpy as np
import pytest

from biota.search.descriptors import (
    WINDOW,
    RolloutTrace,
    compute_descriptors,
    compute_size,
    compute_speed,
    compute_structure,
)

GRID = 96
STEPS = 300


def _make_trace(
    com_history: np.ndarray,
    bbox_fraction_history: np.ndarray,
    final_state: np.ndarray,
) -> RolloutTrace:
    return RolloutTrace(
        com_history=com_history.astype(np.float32),
        bbox_fraction_history=bbox_fraction_history.astype(np.float32),
        final_state=final_state.astype(np.float32),
        grid_size=GRID,
        total_steps=STEPS,
    )


# === speed ===


def test_speed_stationary_is_zero() -> None:
    coms = np.full((WINDOW + 1, 2), 48.0)
    bbox = np.full(WINDOW + 1, 0.1)
    state = np.zeros((GRID, GRID))
    state[40:50, 40:50] = 1.0
    trace = _make_trace(coms, bbox, state)
    assert compute_speed(trace) == 0.0


def test_speed_moves_one_cell_per_step_normalizes_correctly() -> None:
    # COM moves +1 cell in y per step. Mean speed = 1.0 cells/step.
    # Normalizer = 0.5 * 96 / 300 = 0.16.
    # Normalized = 1.0 / 0.16 = 6.25, clipped to 1.0.
    coms = np.zeros((WINDOW + 1, 2))
    coms[:, 0] = np.arange(WINDOW + 1)
    bbox = np.full(WINDOW + 1, 0.1)
    state = np.zeros((GRID, GRID))
    state[40:50, 40:50] = 1.0
    trace = _make_trace(coms, bbox, state)
    assert compute_speed(trace) == 1.0


def test_speed_subnormal_velocity() -> None:
    # COM moves 0.05 cells/step. Normalizer 0.16. Normalized 0.3125.
    coms = np.zeros((WINDOW + 1, 2))
    coms[:, 0] = np.arange(WINDOW + 1) * 0.05
    bbox = np.full(WINDOW + 1, 0.1)
    state = np.zeros((GRID, GRID))
    state[40:50, 40:50] = 1.0
    trace = _make_trace(coms, bbox, state)
    speed = compute_speed(trace)
    assert 0.30 < speed < 0.32


def test_speed_uses_only_last_window() -> None:
    # First half is fast (1 cell/step), last WINDOW is stationary.
    # Result should be 0.
    coms = np.zeros((WINDOW * 2 + 1, 2))
    coms[: WINDOW + 1, 0] = np.arange(WINDOW + 1)
    coms[WINDOW + 1 :, 0] = WINDOW  # stationary at the end
    bbox = np.full(WINDOW * 2 + 1, 0.1)
    state = np.zeros((GRID, GRID))
    state[40:50, 40:50] = 1.0
    trace = _make_trace(coms, bbox, state)
    assert compute_speed(trace) == 0.0


# === size ===


def test_size_constant_fraction() -> None:
    coms = np.zeros((WINDOW + 1, 2))
    bbox = np.full(WINDOW + 1, 0.25)
    state = np.zeros((GRID, GRID))
    state[40:60, 40:60] = 1.0
    trace = _make_trace(coms, bbox, state)
    assert compute_size(trace) == pytest.approx(0.25)


def test_size_changing_fraction_averages() -> None:
    # Half the window at 0.1, half at 0.3 -> mean 0.2
    coms = np.zeros((WINDOW + 1, 2))
    bbox = np.zeros(WINDOW + 1)
    bbox[: (WINDOW + 1) // 2] = 0.1
    bbox[(WINDOW + 1) // 2 :] = 0.3
    state = np.zeros((GRID, GRID))
    state[40:50, 40:50] = 1.0
    trace = _make_trace(coms, bbox, state)
    assert compute_size(trace) == pytest.approx(0.2, abs=0.01)


# === structure ===


def test_structure_uniform_bbox_is_max() -> None:
    # Uniform mass over a region larger than the binning grid -> max entropy
    state = np.zeros((GRID, GRID))
    state[20:60, 20:60] = 1.0
    trace = _make_trace(
        np.zeros((WINDOW + 1, 2)),
        np.full(WINDOW + 1, 0.1),
        state,
    )
    structure = compute_structure(trace)
    assert structure == pytest.approx(1.0, abs=0.01)


def test_structure_single_pixel_is_min() -> None:
    state = np.zeros((GRID, GRID))
    state[48, 48] = 1.0
    trace = _make_trace(
        np.zeros((WINDOW + 1, 2)),
        np.full(WINDOW + 1, 0.001),
        state,
    )
    assert compute_structure(trace) == 0.0


def test_structure_two_blobs_intermediate() -> None:
    # Two compact blobs in a region -> should be > 0 and < uniform
    state = np.zeros((GRID, GRID))
    state[30:35, 30:35] = 1.0
    state[60:65, 60:65] = 1.0
    trace = _make_trace(
        np.zeros((WINDOW + 1, 2)),
        np.full(WINDOW + 1, 0.1),
        state,
    )
    structure = compute_structure(trace)
    assert 0.0 < structure < 1.0


def test_structure_dead_state_is_zero() -> None:
    state = np.zeros((GRID, GRID))
    trace = _make_trace(
        np.zeros((WINDOW + 1, 2)),
        np.full(WINDOW + 1, 0.0),
        state,
    )
    assert compute_structure(trace) == 0.0


# === compute_descriptors orchestrator ===


def test_compute_descriptors_dead_returns_none() -> None:
    state = np.zeros((GRID, GRID))
    trace = _make_trace(
        np.zeros((WINDOW + 1, 2)),
        np.zeros(WINDOW + 1),
        state,
    )
    assert compute_descriptors(trace) is None


def test_compute_descriptors_returns_three_floats_in_unit_cube() -> None:
    state = np.zeros((GRID, GRID))
    state[30:50, 30:50] = 1.0
    trace = _make_trace(
        np.full((WINDOW + 1, 2), 40.0),
        np.full(WINDOW + 1, 0.04),
        state,
    )
    result = compute_descriptors(trace)
    assert result is not None
    assert len(result) == 3
    for value in result:
        assert 0.0 <= value <= 1.0


def test_slice_returns_subwindow() -> None:
    coms = np.arange(200, dtype=np.float32).reshape(100, 2)
    bbox = np.linspace(0.0, 1.0, 100, dtype=np.float32)
    state = np.zeros((GRID, GRID), dtype=np.float32)
    trace = _make_trace(coms, bbox, state)
    sliced = trace.slice(0, 50)
    assert len(sliced.com_history) == 50
    assert len(sliced.bbox_fraction_history) == 50
    np.testing.assert_array_equal(sliced.final_state, state)
    assert sliced.grid_size == GRID
    assert sliced.total_steps == STEPS
