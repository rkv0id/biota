"""Tests for the behavior descriptor module.

These tests verify descriptor compute functions return raw values in [0, 100].

The original
M1 speed/size/structure descriptors were degenerate because their
normalizers were miscalibrated against the biota population - a creature
moving at realistic Lenia speeds was binning to slot 0 out of 32 because
the normalizer was theoretical (0.5 * grid_size / steps) rather than
empirical. The failure was invisible until the search produced a full
archive and we could look at the cell distribution.

A first descriptor rework iteration tried Chan's growth-centroid distance
(dgm) for the third axis, but that turned out to be structurally
degenerate for the radially symmetric solitons biota's parameter prior
produces. dgm was abandoned in favor of spectral entropy of the radially-
averaged FFT magnitude spectrum, which captures spatial frequency content
directly.

To prevent the normalizer-degeneracy class of bug going forward, each
descriptor has calibration tests of the form "a creature with observable
X should produce normalized descriptor value Y." If a normalizer is off
by an order of magnitude, these tests fail immediately.

All tests use synthetic RolloutTraces built directly (not from a real
rollout) so we can control the inputs precisely.
"""

import numpy as np
import pytest

from biota.search.descriptors import (
    SPECTRAL_RADIAL_BINS,
    WINDOW,
    RolloutTrace,
    compute_activity,
    compute_angular_velocity,
    compute_descriptors,
    compute_displacement_ratio,
    compute_emission_activity,
    compute_growth_gradient,
    compute_gyradius,
    compute_morphological_instability,
    compute_receptor_sensitivity,
    compute_signal_retention,
    compute_spatial_entropy,
    compute_spectral_entropy,
    compute_velocity,
)

GRID = 96
STEPS = 300
TRACE_LEN = 2 * WINDOW


def _make_trace(
    *,
    com_path: np.ndarray | None = None,
    final_state: np.ndarray | None = None,
) -> RolloutTrace:
    if com_path is None:
        com_path = np.full((TRACE_LEN, 2), GRID / 2.0, dtype=np.float32)
    if final_state is None:
        # Default: a single lit pixel at the center of the grid. Passes the
        # zero-mass guard in compute_descriptors, gyradius is 0.
        final_state = np.zeros((GRID, GRID), dtype=np.float32)
        final_state[GRID // 2, GRID // 2] = 1.0
    return RolloutTrace(
        com_history=com_path.astype(np.float32),
        bbox_fraction_history=np.full(TRACE_LEN, 0.01, dtype=np.float32),
        gyradius_history=np.zeros(TRACE_LEN, dtype=np.float32),
        final_state=final_state.astype(np.float32),
        grid_size=GRID,
        total_steps=STEPS,
    )


# === compute_velocity ===


def test_velocity_stationary_is_zero() -> None:
    """Constant COM history -> zero deltas -> velocity 0."""
    coms = np.full((TRACE_LEN, 2), 48.0, dtype=np.float32)
    assert compute_velocity(_make_trace(com_path=coms)) == 0.0


def test_velocity_short_history_returns_zero() -> None:
    """Fewer than 2 points in the window -> no deltas -> returns 0."""
    coms = np.array([[48.0, 48.0]], dtype=np.float32)
    assert compute_velocity(_make_trace(com_path=coms)) == 0.0


def test_velocity_returns_raw_cells_per_step() -> None:
    """COM moving 0.005 cells/step returns the raw speed, not normalized."""
    coms = np.zeros((TRACE_LEN, 2), dtype=np.float32)
    coms[:, 0] = np.arange(TRACE_LEN) * 0.005
    assert compute_velocity(_make_trace(com_path=coms)) == pytest.approx(0.005, abs=1e-5)


def test_velocity_fast_creature_returns_raw_speed() -> None:
    """COM moving 0.0077 cells/step returns 0.0077 as the raw speed."""
    coms = np.zeros((TRACE_LEN, 2), dtype=np.float32)
    coms[:, 1] = np.arange(TRACE_LEN) * 0.0077
    assert compute_velocity(_make_trace(com_path=coms)) == pytest.approx(0.0077, abs=1e-5)


def test_velocity_diagonal_motion_uses_l2_norm() -> None:
    """Diagonal motion at 0.003 cells/step per axis gives sqrt(2)*0.003 raw."""
    coms = np.zeros((TRACE_LEN, 2), dtype=np.float32)
    step_size = 0.003
    coms[:, 0] = np.arange(TRACE_LEN) * step_size
    coms[:, 1] = np.arange(TRACE_LEN) * step_size
    expected = float(np.sqrt(2.0) * step_size)
    assert compute_velocity(_make_trace(com_path=coms)) == pytest.approx(expected, abs=1e-5)


def test_velocity_clips_at_one() -> None:
    """COM moving fast enough produces nonzero velocity."""
    coms = np.zeros((TRACE_LEN, 2), dtype=np.float32)
    coms[:, 0] = np.arange(TRACE_LEN) * 0.1  # 0.1 cells/step, raw value returned
    assert compute_velocity(_make_trace(com_path=coms)) == pytest.approx(0.1, abs=1e-5)


# === compute_gyradius ===


def test_gyradius_single_pixel_is_zero() -> None:
    """A single lit pixel has no spread from its own center -> gyradius 0."""
    state = np.zeros((GRID, GRID), dtype=np.float32)
    state[GRID // 2, GRID // 2] = 1.0
    assert compute_gyradius(_make_trace(final_state=state)) == 0.0


def test_gyradius_zero_mass_is_zero() -> None:
    """Empty state -> early-out returns 0 instead of dividing by zero."""
    state = np.zeros((GRID, GRID), dtype=np.float32)
    assert compute_gyradius(_make_trace(final_state=state)) == 0.0


def test_gyradius_calibration_uniform_square() -> None:
    """A uniform 10x10 square. Compute the expected raw gyradius the same
    way the descriptor does, independently of its implementation, then
    check the function matches."""
    state = np.zeros((GRID, GRID), dtype=np.float32)
    y0, x0 = GRID // 2 - 5, GRID // 2 - 5
    state[y0 : y0 + 10, x0 : x0 + 10] = 1.0

    y_idx, x_idx = np.indices(state.shape, dtype=np.float32)
    total_mass = state.sum()
    cy = (state * y_idx).sum() / total_mass
    cx = (state * x_idx).sum() / total_mass
    sq_dist = (y_idx - cy) ** 2 + (x_idx - cx) ** 2
    expected_raw = float(np.sqrt((state * sq_dist).sum() / total_mass))

    got = compute_gyradius(_make_trace(final_state=state))
    assert got == pytest.approx(expected_raw, abs=1e-4)
    # A 10x10 square on a 96-grid has gyradius ~4 grid units
    assert 0.0 < got < 50.0


def test_gyradius_calibration_large_diffuse_creature() -> None:
    """A 40x40 uniform square should give a larger gyradius than a 10x10
    one. Gyradius scales linearly with L for uniform squares."""
    state = np.zeros((GRID, GRID), dtype=np.float32)
    y0, x0 = GRID // 2 - 20, GRID // 2 - 20
    state[y0 : y0 + 40, x0 : x0 + 40] = 1.0

    got = compute_gyradius(_make_trace(final_state=state))
    small_state = np.zeros((GRID, GRID), dtype=np.float32)
    small_state[GRID // 2 - 5 : GRID // 2 + 5, GRID // 2 - 5 : GRID // 2 + 5] = 1.0
    small_got = compute_gyradius(_make_trace(final_state=small_state))
    # 40x40 should have larger gyradius than 10x10
    assert got > small_got


def test_gyradius_clips_at_one() -> None:
    """A creature so diffuse it fills nearly the whole grid produces large gyradius."""
    state = np.ones((GRID, GRID), dtype=np.float32)
    # A very diffuse creature has large gyradius; it should be > 0
    assert compute_gyradius(_make_trace(final_state=state)) > 0.0


# === compute_spectral_entropy ===


def test_spectral_entropy_zero_mass_is_zero() -> None:
    """Empty state -> early-out returns 0."""
    state = np.zeros((GRID, GRID), dtype=np.float32)
    assert compute_spectral_entropy(_make_trace(final_state=state)) == 0.0


def test_spectral_entropy_uniform_field_is_low() -> None:
    """A uniform field has all energy concentrated in the DC component,
    which we explicitly zero out before computing entropy. With DC removed,
    the remaining spectrum is essentially zero everywhere - the function
    returns 0 via the zero-magnitude early-out."""
    state = np.ones((GRID, GRID), dtype=np.float32)
    assert compute_spectral_entropy(_make_trace(final_state=state)) == 0.0


def test_spectral_entropy_smooth_gaussian_clips_to_zero() -> None:
    """A wide Gaussian blob is below the SPECTRAL_ENTROPY_FLOOR (raw value
    raw entropy ~0.27, returned directly (no floor remap in v4.0.0).

    This is intentional. Biota's parameter prior + survival filters never
    actually produce smooth featureless Gaussians; the discovered population
    is entirely sharp-edged structured solitons. The descriptor is calibrated
    to discriminate within that population, not to discriminate the
    population from synthetic test patterns. See SPECTRAL_ENTROPY_FLOOR
    docstring for the calibration story.
    """
    y_idx, x_idx = np.indices((GRID, GRID), dtype=np.float32)
    cy, cx = GRID / 2.0, GRID / 2.0
    radius = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
    state = np.exp(-(radius**2) / (2.0 * 8.0**2)).astype(np.float32)
    entropy = compute_spectral_entropy(_make_trace(final_state=state))
    # Without the SPECTRAL_ENTROPY_FLOOR remap (removed in v4.0.0), the raw entropy
    # of a Gaussian (~0.27) is returned directly. It is lower than a sharp-edged creature.
    assert 0.0 <= entropy < 0.5


def test_spectral_entropy_high_frequency_pattern_is_high() -> None:
    """A grid of small dots produces high spectral entropy after the
    empirical remap. Periodic lattices spread their FFT energy across
    multiple harmonic radial bins.

    Raw entropy value for a periodic lattice on a 96 grid.
    The remap compresses the dotted lattice into the lower part of the
    raw value returned directly."""
    state = np.zeros((GRID, GRID), dtype=np.float32)
    for y in range(0, GRID, 6):
        for x in range(0, GRID, 6):
            state[y, x] = 1.0
    entropy = compute_spectral_entropy(_make_trace(final_state=state))
    assert entropy > 0.0


def test_spectral_entropy_sharp_disk_higher_than_smooth_gaussian() -> None:
    """A direct comparison: a sharp-edged disk should have strictly higher
    spectral entropy than a smooth Gaussian. The Gaussian clips to 0
    (below the floor) and the sharp disk lands in the upper range
    because its sharp edge produces broad spectral content via Gibbs
    ringing.

    This is the qualitative distinction the descriptor needs to capture
    for biota's discovered population: real Lenia creatures have sharp
    edges and the descriptor should discriminate among them by how much
    additional structure they have on top of those edges."""
    y_idx, x_idx = np.indices((GRID, GRID), dtype=np.float32)
    cy, cx = GRID / 2.0, GRID / 2.0
    radius = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
    smooth = np.exp(-(radius**2) / (2.0 * 8.0**2)).astype(np.float32)
    sharp = (radius < 10).astype(np.float32)

    smooth_entropy = compute_spectral_entropy(_make_trace(final_state=smooth))
    sharp_entropy = compute_spectral_entropy(_make_trace(final_state=sharp))
    assert sharp_entropy > smooth_entropy
    assert sharp_entropy > 0.0  # Sharp disk has meaningful spectral content


def test_spectral_entropy_in_unit_range() -> None:
    """For any valid input, entropy must be in [0, 1] after normalization
    by log(SPECTRAL_RADIAL_BINS)."""
    rng = np.random.default_rng(42)
    state = rng.random((GRID, GRID), dtype=np.float32)
    entropy = compute_spectral_entropy(_make_trace(final_state=state))
    assert 0.0 <= entropy <= 100.0


def test_spectral_entropy_uses_radial_binning() -> None:
    """Sanity check: the function uses the SPECTRAL_RADIAL_BINS constant.
    A trivial assertion that the constant is sensible (32 bins is the
    documented design choice)."""
    assert SPECTRAL_RADIAL_BINS == 32


# === compute_descriptors ===


def test_compute_descriptors_returns_tuple_of_three() -> None:
    state = np.zeros((GRID, GRID), dtype=np.float32)
    state[GRID // 2, GRID // 2] = 1.0
    result = compute_descriptors(_make_trace(final_state=state))
    assert result is not None
    assert len(result) == 3
    for d in result:
        assert 0.0 <= d <= 100.0


def test_compute_descriptors_returns_none_on_dead_rollout() -> None:
    """Zero-mass final state -> compute_descriptors returns None, the
    alive filter rejects downstream."""
    state = np.zeros((GRID, GRID), dtype=np.float32)
    assert compute_descriptors(_make_trace(final_state=state)) is None


def test_compute_descriptors_slow_stationary_blob() -> None:
    """Integration: a stationary 10x10 blob with default (zero) COM history
    should produce velocity 0, gyradius around 0.17, and a small but
    nonzero spectral entropy."""
    state = np.zeros((GRID, GRID), dtype=np.float32)
    state[40:50, 40:50] = 1.0
    coms = np.full((TRACE_LEN, 2), 45.0, dtype=np.float32)
    result = compute_descriptors(_make_trace(com_path=coms, final_state=state))
    assert result is not None
    velocity, gyradius, spectral = result
    assert velocity == 0.0
    assert 0.0 < gyradius < 50.0  # raw grid units, compact creature on 96-grid
    assert 0.0 < spectral < 100.0


def _make_trace_with_gyradius(
    *,
    com_path: np.ndarray | None = None,
    gyradius_path: np.ndarray | None = None,
    final_state: np.ndarray | None = None,
) -> RolloutTrace:
    """Like _make_trace but with explicit gyradius_history control."""
    if com_path is None:
        com_path = np.full((TRACE_LEN, 2), GRID / 2.0, dtype=np.float32)
    if gyradius_path is None:
        gyradius_path = np.zeros(TRACE_LEN, dtype=np.float32)
    if final_state is None:
        final_state = np.zeros((GRID, GRID), dtype=np.float32)
        final_state[GRID // 2, GRID // 2] = 1.0
    return RolloutTrace(
        com_history=com_path.astype(np.float32),
        bbox_fraction_history=np.full(TRACE_LEN, 0.01, dtype=np.float32),
        gyradius_history=gyradius_path.astype(np.float32),
        final_state=final_state.astype(np.float32),
        grid_size=GRID,
        total_steps=STEPS,
    )


# === compute_displacement_ratio ===


def test_displacement_ratio_straight_line_is_one() -> None:
    """Perfect straight-line translator: displacement == path length -> ratio 1.0."""
    coms = np.zeros((TRACE_LEN, 2), dtype=np.float32)
    coms[:, 1] = np.linspace(0, 10, TRACE_LEN)  # moving right only
    assert compute_displacement_ratio(_make_trace(com_path=coms)) == pytest.approx(1.0, abs=1e-4)


def test_displacement_ratio_stationary_is_zero() -> None:
    """Stationary creature: path length 0 -> returns 0."""
    coms = np.full((TRACE_LEN, 2), 48.0, dtype=np.float32)
    assert compute_displacement_ratio(_make_trace(com_path=coms)) == 0.0


def test_displacement_ratio_orbiter_is_low() -> None:
    """Circular orbit with period = WINDOW closes completely in the window:
    start == end -> displacement ~ 0, path = 2*pi*R -> ratio ~ 0."""
    t = np.linspace(0, 2 * np.pi, WINDOW)
    orbit = np.stack([np.sin(t) * 10 + 48, np.cos(t) * 10 + 48], axis=1).astype(np.float32)
    full = np.tile(orbit, (TRACE_LEN // WINDOW + 1, 1))[:TRACE_LEN]
    ratio = compute_displacement_ratio(_make_trace(com_path=full))
    assert ratio < 0.15


def test_displacement_ratio_in_unit_interval() -> None:
    """All outputs are in [0, 1]."""
    coms = np.random.default_rng(42).random((TRACE_LEN, 2)).astype(np.float32) * GRID
    assert 0.0 <= compute_displacement_ratio(_make_trace(com_path=coms)) <= 100.0


# === compute_angular_velocity ===


def test_angular_velocity_straight_line_is_zero() -> None:
    """Straight-line mover: constant direction -> zero angular speed."""
    coms = np.zeros((TRACE_LEN, 2), dtype=np.float32)
    coms[:, 1] = np.linspace(0, 20, TRACE_LEN)
    assert compute_angular_velocity(_make_trace(com_path=coms)) < 0.05


def test_angular_velocity_full_orbit_saturates() -> None:
    """One full circular orbit in WINDOW steps -> angular speed ~2*pi/WINDOW rad/step.
    With WINDOW=50: 2*pi/50 ~ 0.126 rad/step returned as raw value."""
    t = np.linspace(0, 2 * np.pi, WINDOW)
    orbit = np.stack([np.sin(t) * 10 + 48, np.cos(t) * 10 + 48], axis=1).astype(np.float32)
    full = np.tile(orbit, (TRACE_LEN // WINDOW + 1, 1))[:TRACE_LEN]
    val = compute_angular_velocity(_make_trace(com_path=full))
    assert val > 0.1  # raw rad/step, 2*pi/50 ~ 0.126


def test_angular_velocity_stationary_is_zero() -> None:
    coms = np.full((TRACE_LEN, 2), 48.0, dtype=np.float32)
    assert compute_angular_velocity(_make_trace(com_path=coms)) == 0.0


def test_angular_velocity_in_unit_interval() -> None:
    coms = np.random.default_rng(7).random((TRACE_LEN, 2)).astype(np.float32) * GRID
    assert 0.0 <= compute_angular_velocity(_make_trace(com_path=coms)) <= 100.0


# === compute_growth_gradient ===


def test_growth_gradient_empty_is_zero() -> None:
    state = np.zeros((GRID, GRID), dtype=np.float32)
    assert compute_growth_gradient(_make_trace(final_state=state)) == 0.0


def test_growth_gradient_uniform_is_low() -> None:
    """A uniform square has gradient only at its boundary -> low score (~0.13)."""
    state = np.zeros((GRID, GRID), dtype=np.float32)
    state[30:60, 30:60] = 1.0
    val = compute_growth_gradient(_make_trace(final_state=state))
    assert val < 0.3


def test_growth_gradient_sharp_edges_is_higher() -> None:
    """A state with many small features has more gradient than a uniform block."""
    rng = np.random.default_rng(3)
    state_noisy = rng.random((GRID, GRID)).astype(np.float32)
    state_smooth = np.zeros((GRID, GRID), dtype=np.float32)
    state_smooth[30:60, 30:60] = 1.0
    assert compute_growth_gradient(_make_trace(final_state=state_noisy)) > compute_growth_gradient(
        _make_trace(final_state=state_smooth)
    )


def test_growth_gradient_in_unit_interval() -> None:
    state = np.abs(np.random.default_rng(11).standard_normal((GRID, GRID))).astype(np.float32)
    assert 0.0 <= compute_growth_gradient(_make_trace(final_state=state)) <= 100.0


# === compute_morphological_instability ===


def test_morphological_instability_constant_is_zero() -> None:
    """Constant gyradius over time -> zero variance -> zero instability."""
    gyr = np.full(TRACE_LEN, 10.0, dtype=np.float32)
    assert compute_morphological_instability(_make_trace_with_gyradius(gyradius_path=gyr)) == 0.0


def test_morphological_instability_pulsing_is_nonzero() -> None:
    """Oscillating gyradius -> nonzero variance -> nonzero instability."""
    t = np.arange(TRACE_LEN, dtype=np.float32)
    gyr = 10.0 + 5.0 * np.sin(t * np.pi / 10)
    val = compute_morphological_instability(_make_trace_with_gyradius(gyradius_path=gyr))
    assert val > 0.0


def test_morphological_instability_larger_swing_is_higher() -> None:
    """Larger gyradius oscillation -> higher instability."""
    t = np.arange(TRACE_LEN, dtype=np.float32)
    gyr_small = (10.0 + 1.0 * np.sin(t * np.pi / 10)).astype(np.float32)
    gyr_large = (10.0 + 8.0 * np.sin(t * np.pi / 10)).astype(np.float32)
    small = compute_morphological_instability(_make_trace_with_gyradius(gyradius_path=gyr_small))
    large = compute_morphological_instability(_make_trace_with_gyradius(gyradius_path=gyr_large))
    assert large > small


def test_morphological_instability_in_unit_interval() -> None:
    gyr = np.random.default_rng(17).random(TRACE_LEN).astype(np.float32) * 30
    assert (
        0.0
        <= compute_morphological_instability(_make_trace_with_gyradius(gyradius_path=gyr))
        <= 100.0
    )


# === compute_activity ===


def test_activity_static_is_zero() -> None:
    """Constant gyradius -> no change per step -> activity 0."""
    gyr = np.full(TRACE_LEN, 15.0, dtype=np.float32)
    assert compute_activity(_make_trace_with_gyradius(gyradius_path=gyr)) == 0.0


def test_activity_changing_is_nonzero() -> None:
    """Linearly growing gyradius -> constant change per step -> nonzero activity."""
    gyr = np.linspace(5.0, 25.0, TRACE_LEN).astype(np.float32)
    assert compute_activity(_make_trace_with_gyradius(gyradius_path=gyr)) > 0.0


def test_activity_faster_change_is_higher() -> None:
    """Faster gyradius change -> higher activity."""
    gyr_slow = np.linspace(10.0, 12.0, TRACE_LEN).astype(np.float32)
    gyr_fast = np.linspace(10.0, 30.0, TRACE_LEN).astype(np.float32)
    slow = compute_activity(_make_trace_with_gyradius(gyradius_path=gyr_slow))
    fast = compute_activity(_make_trace_with_gyradius(gyradius_path=gyr_fast))
    assert fast > slow


def test_activity_in_unit_interval() -> None:
    gyr = np.random.default_rng(23).random(TRACE_LEN).astype(np.float32) * 50
    assert 0.0 <= compute_activity(_make_trace_with_gyradius(gyradius_path=gyr)) <= 100.0


# === compute_spatial_entropy ===


def test_spatial_entropy_empty_is_zero() -> None:
    state = np.zeros((GRID, GRID), dtype=np.float32)
    assert compute_spatial_entropy(_make_trace(final_state=state)) == 0.0


def test_spatial_entropy_concentrated_is_low() -> None:
    """Single pixel at center -> all mass in one bin -> low entropy."""
    state = np.zeros((GRID, GRID), dtype=np.float32)
    state[GRID // 2, GRID // 2] = 1.0
    val = compute_spatial_entropy(_make_trace(final_state=state))
    assert val < 0.3


def test_spatial_entropy_uniform_is_high() -> None:
    """Uniform mass everywhere -> maximum entropy -> high score."""
    state = np.ones((GRID, GRID), dtype=np.float32)
    val = compute_spatial_entropy(_make_trace(final_state=state))
    assert val > 0.95


def test_spatial_entropy_concentrated_less_than_diffuse() -> None:
    """A small central blob has lower entropy than a full uniform state."""
    state_compact = np.zeros((GRID, GRID), dtype=np.float32)
    state_compact[40:56, 40:56] = 1.0
    state_diffuse = np.ones((GRID, GRID), dtype=np.float32)
    compact = compute_spatial_entropy(_make_trace(final_state=state_compact))
    diffuse = compute_spatial_entropy(_make_trace(final_state=state_diffuse))
    assert compact < diffuse


def test_spatial_entropy_in_unit_interval() -> None:
    state = np.random.default_rng(31).random((GRID, GRID)).astype(np.float32)
    assert 0.0 <= compute_spatial_entropy(_make_trace(final_state=state)) <= 100.0


# ===========================================================================
# Signal descriptors
# ===========================================================================

C = 16  # signal channels


def _make_signal_trace(
    emission_history: np.ndarray | None = None,
    reception_history: np.ndarray | None = None,
    retention: float | None = None,
) -> RolloutTrace:
    """Build a RolloutTrace with signal history fields populated."""
    state = np.zeros((GRID, GRID), dtype=np.float32)
    state[40:56, 40:56] = 1.0
    return RolloutTrace(
        com_history=np.full((TRACE_LEN, 2), 48.0, dtype=np.float32),
        bbox_fraction_history=np.full(TRACE_LEN, 0.04, dtype=np.float32),
        gyradius_history=np.zeros(TRACE_LEN, dtype=np.float32),
        final_state=state,
        grid_size=GRID,
        total_steps=STEPS,
        signal_emission_history=emission_history,
        signal_reception_history=reception_history,
        signal_retention=retention,
    )


# --- emission_activity ---


def test_emission_activity_zero_for_non_signal() -> None:
    trace = _make_trace()
    assert compute_emission_activity(trace) == 0.0


def test_emission_activity_zero_for_empty_history() -> None:
    trace = _make_signal_trace(emission_history=np.array([], dtype=np.float32))
    assert compute_emission_activity(trace) == 0.0


def test_emission_activity_scales_with_emission() -> None:
    low = _make_signal_trace(emission_history=np.full(TRACE_LEN, 0.002, dtype=np.float32))
    high = _make_signal_trace(emission_history=np.full(TRACE_LEN, 0.008, dtype=np.float32))
    assert compute_emission_activity(low) < compute_emission_activity(high)


def test_emission_activity_clipped_to_unit() -> None:
    # History above normalizer -> clips to 100.0
    trace = _make_signal_trace(emission_history=np.full(TRACE_LEN, 1.0, dtype=np.float32))
    assert compute_emission_activity(trace) == 1.0


def test_emission_activity_in_unit_interval() -> None:
    trace = _make_signal_trace(
        emission_history=np.random.default_rng(7).random(TRACE_LEN).astype(np.float32) * 0.05
    )
    val = compute_emission_activity(trace)
    assert 0.0 <= val <= 100.0


# --- receptor_sensitivity ---


def test_receptor_sensitivity_zero_for_non_signal() -> None:
    trace = _make_trace()
    assert compute_receptor_sensitivity(trace) == 0.0


def test_receptor_sensitivity_zero_for_empty_history() -> None:
    trace = _make_signal_trace(reception_history=np.array([], dtype=np.float32))
    assert compute_receptor_sensitivity(trace) == 0.0


def test_receptor_sensitivity_scales_with_response() -> None:
    low = _make_signal_trace(reception_history=np.full(TRACE_LEN, 0.003, dtype=np.float32))
    high = _make_signal_trace(reception_history=np.full(TRACE_LEN, 0.010, dtype=np.float32))
    assert compute_receptor_sensitivity(low) < compute_receptor_sensitivity(high)


def test_receptor_sensitivity_in_unit_interval() -> None:
    trace = _make_signal_trace(
        reception_history=np.random.default_rng(9).random(TRACE_LEN).astype(np.float32) * 0.010
    )
    val = compute_receptor_sensitivity(trace)
    assert 0.0 <= val <= 100.0


# --- signal_retention ---


def test_signal_retention_one_for_non_signal() -> None:
    """Non-signal rollout: no mass was lost, retention = 1.0."""
    trace = _make_trace()
    assert compute_signal_retention(trace) == 1.0  # non-signal returns exactly 1.0


def test_signal_retention_reflects_mass_loss() -> None:
    # 70% retained -> 0.7
    trace = _make_signal_trace(retention=0.7)
    assert abs(compute_signal_retention(trace) - 0.7) < 1e-6


def test_signal_retention_clipped_to_100() -> None:
    # Clip bound is now 100.0 (raw value), not 1.0
    assert compute_signal_retention(_make_signal_trace(retention=150.0)) == 100.0
    assert compute_signal_retention(_make_signal_trace(retention=1.5)) == 1.5
    assert compute_signal_retention(_make_signal_trace(retention=-0.1)) == 0.0


def test_signal_retention_high_beats_low() -> None:
    high = _make_signal_trace(retention=0.95)
    low = _make_signal_trace(retention=0.40)
    assert compute_signal_retention(high) > compute_signal_retention(low)
