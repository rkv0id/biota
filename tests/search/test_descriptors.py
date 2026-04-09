"""Tests for the behavior descriptor module.

These tests exist primarily to catch normalizer mistakes. The original
speed/size/structure descriptors from M1 were degenerate because their
normalizers were miscalibrated - a creature moving at realistic Lenia speeds
was binning to slot 0 out of 32 because the normalizer was theoretical
(0.5 * grid_size / steps) rather than empirical. The failure was invisible
until the search had produced a full archive and we could look at the cell
distribution. See DECISIONS.md 2026-04-09 "descriptor degeneracy diagnosis".

To prevent this class of bug going forward, each descriptor has calibration
tests of the form "a creature with observable X should produce normalized
descriptor value Y." If a normalizer is off by an order of magnitude, these
tests fail immediately.

All tests use synthetic RolloutTraces built directly (not from a real
rollout) so we can control the inputs precisely.
"""

import numpy as np
import pytest

from biota.search.descriptors import (
    GYRADIUS_NORMALIZER_DIVISOR,
    VELOCITY_NORMALIZER,
    WINDOW,
    RolloutTrace,
    compute_descriptors,
    compute_growth_centroid_distance,
    compute_gyradius,
    compute_velocity,
)

GRID = 96
STEPS = 300
TRACE_LEN = 2 * WINDOW


def _make_trace(
    *,
    com_path: np.ndarray | None = None,
    final_state: np.ndarray | None = None,
    final_growth_field: np.ndarray | None = None,
) -> RolloutTrace:
    if com_path is None:
        com_path = np.full((TRACE_LEN, 2), GRID / 2.0, dtype=np.float32)
    if final_state is None:
        # Default: a single lit pixel at the center of the grid. Passes the
        # zero-mass guard in compute_descriptors, gyradius is 0, dgm is 0.
        final_state = np.zeros((GRID, GRID), dtype=np.float32)
        final_state[GRID // 2, GRID // 2] = 1.0
    if final_growth_field is None:
        final_growth_field = np.zeros((GRID, GRID), dtype=np.float32)
    return RolloutTrace(
        com_history=com_path.astype(np.float32),
        bbox_fraction_history=np.full(TRACE_LEN, 0.01, dtype=np.float32),
        final_state=final_state.astype(np.float32),
        final_growth_field=final_growth_field.astype(np.float32),
        grid_size=GRID,
        total_steps=STEPS,
    )


# === compute_velocity ===


def test_velocity_stationary_is_zero() -> None:
    """Constant COM history -> zero deltas -> velocity 0."""
    coms = np.full((TRACE_LEN, 2), 48.0, dtype=np.float32)
    assert compute_velocity(_make_trace(com_path=coms)) == 0.0


def test_velocity_short_history_returns_zero() -> None:
    """Fewer than 2 points in the window -> no deltas -> returns 0.
    The short trace uses a one-row COM history which hits the early-out."""
    coms = np.array([[48.0, 48.0]], dtype=np.float32)
    assert compute_velocity(_make_trace(com_path=coms)) == 0.0


def test_velocity_calibration_slow_creature() -> None:
    """COM moving 0.05 cells/step along one axis should produce
    normalized velocity = 0.05 / VELOCITY_NORMALIZER = 0.1.

    This is the canonical 'slow Lenia creature' calibration. If the normalizer
    is off by 10x or 100x, this test fails loudly."""
    coms = np.zeros((TRACE_LEN, 2), dtype=np.float32)
    coms[:, 0] = np.arange(TRACE_LEN) * 0.05
    expected = 0.05 / VELOCITY_NORMALIZER
    assert compute_velocity(_make_trace(com_path=coms)) == pytest.approx(expected, abs=1e-5)


def test_velocity_calibration_fast_creature() -> None:
    """COM moving 0.2 cells/step (roughly the fastest real Lenia soliton)
    should normalize to 0.2 / 0.5 = 0.4. Well below the clipping ceiling,
    which is the whole point of the empirical normalizer."""
    coms = np.zeros((TRACE_LEN, 2), dtype=np.float32)
    coms[:, 1] = np.arange(TRACE_LEN) * 0.2
    expected = 0.2 / VELOCITY_NORMALIZER
    assert compute_velocity(_make_trace(com_path=coms)) == pytest.approx(expected, abs=1e-5)


def test_velocity_diagonal_motion_uses_l2_norm() -> None:
    """Motion along a diagonal at 0.03 cells/step per axis should give a
    velocity of sqrt(0.03^2 + 0.03^2) = 0.0424..., then divided by 0.5."""
    coms = np.zeros((TRACE_LEN, 2), dtype=np.float32)
    step_size = 0.03
    coms[:, 0] = np.arange(TRACE_LEN) * step_size
    coms[:, 1] = np.arange(TRACE_LEN) * step_size
    raw = np.sqrt(2) * step_size
    expected = raw / VELOCITY_NORMALIZER
    assert compute_velocity(_make_trace(com_path=coms)) == pytest.approx(expected, abs=1e-5)


def test_velocity_clips_at_one() -> None:
    """COM moving faster than VELOCITY_NORMALIZER cells/step should clip to 1.0."""
    coms = np.zeros((TRACE_LEN, 2), dtype=np.float32)
    coms[:, 0] = np.arange(TRACE_LEN) * 1.0  # 1.0 cells/step, well above 0.5
    assert compute_velocity(_make_trace(com_path=coms)) == 1.0


# === compute_gyradius ===


def test_gyradius_single_pixel_is_zero() -> None:
    """A single lit pixel has no spread from its own center -> gyradius 0."""
    state = np.zeros((GRID, GRID), dtype=np.float32)
    state[GRID // 2, GRID // 2] = 1.0
    assert compute_gyradius(_make_trace(final_state=state)) == 0.0


def test_gyradius_zero_mass_is_zero() -> None:
    """Empty state -> guarded early-out returns 0 instead of dividing by zero."""
    state = np.zeros((GRID, GRID), dtype=np.float32)
    assert compute_gyradius(_make_trace(final_state=state)) == 0.0


def test_gyradius_calibration_uniform_square() -> None:
    """A uniform 10x10 square centered in the grid. The mass-weighted RMS
    distance from the COM of a uniform square is computable: sum of squared
    distances in a discrete square of side L (1 unit between cells) is
    2 * L * sum(k^2 for k in [-(L-1)/2, (L-1)/2]) / L^2 per axis. For L=10
    with integer cell positions, this works out to variance ~= 8.25 per axis,
    total variance ~= 16.5, gyradius ~= sqrt(16.5) ~= 4.06. We compute the
    expected value numerically in the test to match the discrete arithmetic
    exactly."""
    state = np.zeros((GRID, GRID), dtype=np.float32)
    y0, x0 = GRID // 2 - 5, GRID // 2 - 5
    state[y0 : y0 + 10, x0 : x0 + 10] = 1.0

    # Compute the expected raw gyradius the same way the descriptor does,
    # independently of its implementation.
    y_idx, x_idx = np.indices(state.shape, dtype=np.float32)
    total_mass = state.sum()
    cy = (state * y_idx).sum() / total_mass
    cx = (state * x_idx).sum() / total_mass
    sq_dist = (y_idx - cy) ** 2 + (x_idx - cx) ** 2
    expected_raw = float(np.sqrt((state * sq_dist).sum() / total_mass))
    expected_normalized = expected_raw / (GRID / GYRADIUS_NORMALIZER_DIVISOR)

    got = compute_gyradius(_make_trace(final_state=state))
    assert got == pytest.approx(expected_normalized, abs=1e-4)
    # Also sanity-check that the value lands in a sensible range: a 10x10
    # square in a 96-grid with normalizer 24 should produce something in
    # roughly [0.15, 0.20].
    assert 0.1 < got < 0.25


def test_gyradius_calibration_large_diffuse_creature() -> None:
    """A 40x40 uniform square should produce a larger gyradius than a 10x10
    one. Specifically, gyradius scales linearly with L for uniform squares."""
    state = np.zeros((GRID, GRID), dtype=np.float32)
    y0, x0 = GRID // 2 - 20, GRID // 2 - 20
    state[y0 : y0 + 40, x0 : x0 + 40] = 1.0

    got = compute_gyradius(_make_trace(final_state=state))
    # A 40x40 creature should produce a gyradius roughly 4x a 10x10 creature
    # (linear in L), so something around 0.6-0.7 in normalized units.
    assert 0.5 < got < 0.8


def test_gyradius_clips_at_one() -> None:
    """A creature so diffuse it fills nearly the whole grid should clip to 1.
    A 96x96 uniform square has gyradius = 96/sqrt(6) ~= 39.2, normalizer 24,
    so raw 1.63, clipped to 1.0."""
    state = np.ones((GRID, GRID), dtype=np.float32)
    assert compute_gyradius(_make_trace(final_state=state)) == 1.0


# === compute_growth_centroid_distance ===


def test_dgm_coincident_centers_is_zero() -> None:
    """Mass and growth both centered at the same point -> dgm 0."""
    state = np.zeros((GRID, GRID), dtype=np.float32)
    growth = np.zeros((GRID, GRID), dtype=np.float32)
    state[GRID // 2, GRID // 2] = 1.0
    growth[GRID // 2, GRID // 2] = 1.0
    assert (
        compute_growth_centroid_distance(_make_trace(final_state=state, final_growth_field=growth))
        == 0.0
    )


def test_dgm_zero_mass_is_zero() -> None:
    """Empty state -> guarded early-out returns 0."""
    state = np.zeros((GRID, GRID), dtype=np.float32)
    growth = np.ones((GRID, GRID), dtype=np.float32)
    assert (
        compute_growth_centroid_distance(_make_trace(final_state=state, final_growth_field=growth))
        == 0.0
    )


def test_dgm_zero_growth_is_zero() -> None:
    """Non-zero mass but zero growth activity -> guarded early-out returns 0.
    This is the expected case for stable creatures where the growth function
    has settled to zero (mass neither growing nor shrinking anywhere)."""
    state = np.zeros((GRID, GRID), dtype=np.float32)
    state[GRID // 2, GRID // 2] = 1.0
    growth = np.zeros((GRID, GRID), dtype=np.float32)
    assert (
        compute_growth_centroid_distance(_make_trace(final_state=state, final_growth_field=growth))
        == 0.0
    )


def test_dgm_calibration_offset_centers() -> None:
    """Mass at (30, 30), growth at (60, 60). Distance = sqrt(900 + 900) ~= 42.43.
    Normalized by grid_size=96: 42.43/96 ~= 0.442.

    This is the canonical 'asymmetric creature' case the M1 Shannon-entropy
    descriptor could not detect. If the normalizer is wrong this fails
    immediately."""
    state = np.zeros((GRID, GRID), dtype=np.float32)
    growth = np.zeros((GRID, GRID), dtype=np.float32)
    state[30, 30] = 1.0
    growth[60, 60] = 1.0

    raw = float(np.sqrt((60 - 30) ** 2 + (60 - 30) ** 2))
    expected = raw / GRID
    got = compute_growth_centroid_distance(
        _make_trace(final_state=state, final_growth_field=growth)
    )
    assert got == pytest.approx(expected, abs=1e-4)
    # Sanity range check
    assert 0.4 < got < 0.5


def test_dgm_uses_absolute_value_of_growth() -> None:
    """Growth field can be negative (cells where mass should shrink). The
    descriptor should weight the centroid by |growth|, not by signed growth,
    because we want a center of 'growth activity' rather than a signed sum
    that cancels out."""
    state = np.zeros((GRID, GRID), dtype=np.float32)
    state[GRID // 2, GRID // 2] = 1.0
    # Growth is negative at (30, 30) and positive at (60, 60), equal magnitudes.
    # If the descriptor used signed growth, the centroid would be at the
    # midpoint of 30 and 60 (i.e. 45) because the two contributions would
    # partially cancel. Using |growth|, both contribute equally and the
    # centroid is at (45, 45), which happens to be the same result - not
    # a good discriminating test.
    # Better test: negative growth at (30, 30) only. Signed sum is negative
    # (interprets as center below origin), absolute sum treats it as a
    # positive contribution at (30, 30).
    growth = np.zeros((GRID, GRID), dtype=np.float32)
    growth[30, 30] = -1.0
    got = compute_growth_centroid_distance(
        _make_trace(final_state=state, final_growth_field=growth)
    )
    # The growth centroid should be at (30, 30) after |.|. Mass COM is at
    # (48, 48). Distance = sqrt(18^2 + 18^2) = 25.456. Normalized to 96:
    # ~0.265. If |.| weren't used, the test would fail because the
    # single negative value would break the positive-weight assumption
    # of the centroid formula entirely (division by a negative total).
    expected = float(np.sqrt(18 * 18 + 18 * 18)) / GRID
    assert got == pytest.approx(expected, abs=1e-3)


def test_dgm_clips_at_one() -> None:
    """Mass at one corner, growth at the opposite corner. Max distance on a
    96x96 grid is sqrt(2) * 95 / 96 ~= 1.4 in units of grid_size, which clips
    to 1.0."""
    state = np.zeros((GRID, GRID), dtype=np.float32)
    growth = np.zeros((GRID, GRID), dtype=np.float32)
    state[0, 0] = 1.0
    growth[95, 95] = 1.0
    got = compute_growth_centroid_distance(
        _make_trace(final_state=state, final_growth_field=growth)
    )
    # Not quite 1.0 due to normalization by GRID not GRID-1, but close.
    # sqrt(95^2 + 95^2) / 96 ~= 134.35/96 ~= 1.39, clipped to 1.0.
    assert got == 1.0


# === compute_descriptors ===


def test_compute_descriptors_returns_tuple_of_three() -> None:
    """Smoke test: with a non-dead trace, compute_descriptors returns a
    three-tuple of floats in [0, 1]."""
    state = np.zeros((GRID, GRID), dtype=np.float32)
    state[GRID // 2, GRID // 2] = 1.0
    result = compute_descriptors(_make_trace(final_state=state))
    assert result is not None
    assert len(result) == 3
    for v in result:
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0


def test_compute_descriptors_returns_none_on_dead_rollout() -> None:
    """Zero-mass final state -> returns None so the alive filter downstream
    can reject with reason 'dead'."""
    state = np.zeros((GRID, GRID), dtype=np.float32)
    assert compute_descriptors(_make_trace(final_state=state)) is None


def test_compute_descriptors_slow_stationary_blob() -> None:
    """A stationary creature of moderate size with coincident mass and
    growth centers should produce velocity=0, some nonzero gyradius, and
    dgm=0. This is the canonical 'symmetric stable blob' case."""
    coms = np.full((TRACE_LEN, 2), GRID / 2.0, dtype=np.float32)
    state = np.zeros((GRID, GRID), dtype=np.float32)
    y0, x0 = GRID // 2 - 5, GRID // 2 - 5
    state[y0 : y0 + 10, x0 : x0 + 10] = 1.0
    growth = np.zeros_like(state)
    growth[y0 : y0 + 10, x0 : x0 + 10] = 0.5

    result = compute_descriptors(
        _make_trace(com_path=coms, final_state=state, final_growth_field=growth)
    )
    assert result is not None
    velocity, gyradius, dgm = result
    assert velocity == 0.0
    assert 0.1 < gyradius < 0.25  # From the gyradius calibration test
    assert dgm == pytest.approx(0.0, abs=1e-4)
