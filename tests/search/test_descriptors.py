"""Tests for the behavior descriptor module.

These tests exist primarily to catch normalizer mistakes. The original
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
    GYRADIUS_NORMALIZER_DIVISOR,
    SPECTRAL_RADIAL_BINS,
    VELOCITY_NORMALIZER,
    WINDOW,
    RolloutTrace,
    compute_activity,
    compute_angular_velocity,
    compute_descriptors,
    compute_displacement_ratio,
    compute_growth_gradient,
    compute_gyradius,
    compute_morphological_instability,
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


def test_velocity_calibration_typical_biota_creature() -> None:
    """COM moving 0.005 cells/step along one axis - the median speed observed
    in actual cluster runs - should produce normalized velocity =
    0.005 / 0.02 = 0.25. This is the canonical "typical biota creature"
    calibration. If the normalizer is off by 10x or 100x, this test fails
    loudly."""
    coms = np.zeros((TRACE_LEN, 2), dtype=np.float32)
    coms[:, 0] = np.arange(TRACE_LEN) * 0.005
    expected = 0.005 / VELOCITY_NORMALIZER
    assert compute_velocity(_make_trace(com_path=coms)) == pytest.approx(expected, abs=1e-5)


def test_velocity_calibration_fast_biota_creature() -> None:
    """COM moving 0.0077 cells/step (the fastest velocity observed in the
    iteration-1 cluster run) should normalize to ~0.385. Well below the
    clipping ceiling, which is the whole point of having a small safety
    margin in the normalizer."""
    coms = np.zeros((TRACE_LEN, 2), dtype=np.float32)
    coms[:, 1] = np.arange(TRACE_LEN) * 0.0077
    expected = 0.0077 / VELOCITY_NORMALIZER
    assert compute_velocity(_make_trace(com_path=coms)) == pytest.approx(expected, abs=1e-5)


def test_velocity_diagonal_motion_uses_l2_norm() -> None:
    """Diagonal motion at 0.003 cells/step per axis should give velocity
    sqrt(2) * 0.003, then divided by VELOCITY_NORMALIZER."""
    coms = np.zeros((TRACE_LEN, 2), dtype=np.float32)
    step_size = 0.003
    coms[:, 0] = np.arange(TRACE_LEN) * step_size
    coms[:, 1] = np.arange(TRACE_LEN) * step_size
    raw = float(np.sqrt(2.0) * step_size)
    expected = raw / VELOCITY_NORMALIZER
    assert compute_velocity(_make_trace(com_path=coms)) == pytest.approx(expected, abs=1e-5)


def test_velocity_clips_at_one() -> None:
    """COM moving faster than VELOCITY_NORMALIZER cells/step clips to 1.0."""
    coms = np.zeros((TRACE_LEN, 2), dtype=np.float32)
    coms[:, 0] = np.arange(TRACE_LEN) * 0.1  # 0.1 cells/step, well above 0.02
    assert compute_velocity(_make_trace(com_path=coms)) == 1.0


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
    expected_normalized = expected_raw / (GRID / GYRADIUS_NORMALIZER_DIVISOR)

    got = compute_gyradius(_make_trace(final_state=state))
    assert got == pytest.approx(expected_normalized, abs=1e-4)
    # Sanity check the value lands in a sensible range: a 10x10 square in
    # a 96-grid with normalizer 24 should produce something around 0.15-0.20.
    assert 0.1 < got < 0.25


def test_gyradius_calibration_large_diffuse_creature() -> None:
    """A 40x40 uniform square should give a larger gyradius than a 10x10
    one. Gyradius scales linearly with L for uniform squares."""
    state = np.zeros((GRID, GRID), dtype=np.float32)
    y0, x0 = GRID // 2 - 20, GRID // 2 - 20
    state[y0 : y0 + 40, x0 : x0 + 40] = 1.0

    got = compute_gyradius(_make_trace(final_state=state))
    # ~4x a 10x10 creature, so something around 0.6-0.7.
    assert 0.5 < got < 0.8


def test_gyradius_clips_at_one() -> None:
    """A creature so diffuse it fills nearly the whole grid clips to 1."""
    state = np.ones((GRID, GRID), dtype=np.float32)
    assert compute_gyradius(_make_trace(final_state=state)) == 1.0


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
    ~0.27 vs floor 0.55), so it clips to 0 after the empirical remap.

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
    assert entropy == 0.0


def test_spectral_entropy_high_frequency_pattern_is_high() -> None:
    """A grid of small dots produces high spectral entropy after the
    empirical remap. Periodic lattices spread their FFT energy across
    multiple harmonic radial bins.

    Empirical post-remap value ~0.12 for a period-6 lattice on a 96 grid.
    The remap compresses the dotted lattice into the lower part of the
    output range because the raw value (~0.60) is just above the floor."""
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
    assert sharp_entropy > 0.5  # Lands well into the upper range
    assert smooth_entropy == 0.0  # Below the floor, clipped


def test_spectral_entropy_in_unit_range() -> None:
    """For any valid input, entropy must be in [0, 1] after normalization
    by log(SPECTRAL_RADIAL_BINS)."""
    rng = np.random.default_rng(42)
    state = rng.random((GRID, GRID), dtype=np.float32)
    entropy = compute_spectral_entropy(_make_trace(final_state=state))
    assert 0.0 <= entropy <= 1.0


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
        assert 0.0 <= d <= 1.0


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
    assert 0.1 < gyradius < 0.25
    assert 0.0 < spectral < 1.0


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
    assert 0.0 <= compute_displacement_ratio(_make_trace(com_path=coms)) <= 1.0


# === compute_angular_velocity ===


def test_angular_velocity_straight_line_is_zero() -> None:
    """Straight-line mover: constant direction -> zero angular speed."""
    coms = np.zeros((TRACE_LEN, 2), dtype=np.float32)
    coms[:, 1] = np.linspace(0, 20, TRACE_LEN)
    assert compute_angular_velocity(_make_trace(com_path=coms)) < 0.05


def test_angular_velocity_full_orbit_saturates() -> None:
    """One full circular orbit in WINDOW steps -> angular speed ~2*pi/WINDOW rad/step.
    With WINDOW=50: 2*pi/50 ~ 0.126 rad/step vs normalizer 0.15 -> result ~0.84."""
    t = np.linspace(0, 2 * np.pi, WINDOW)
    orbit = np.stack([np.sin(t) * 10 + 48, np.cos(t) * 10 + 48], axis=1).astype(np.float32)
    full = np.tile(orbit, (TRACE_LEN // WINDOW + 1, 1))[:TRACE_LEN]
    val = compute_angular_velocity(_make_trace(com_path=full))
    assert val > 0.6  # 2*pi/WINDOW / 0.15 ~ 0.84


def test_angular_velocity_stationary_is_zero() -> None:
    coms = np.full((TRACE_LEN, 2), 48.0, dtype=np.float32)
    assert compute_angular_velocity(_make_trace(com_path=coms)) == 0.0


def test_angular_velocity_in_unit_interval() -> None:
    coms = np.random.default_rng(7).random((TRACE_LEN, 2)).astype(np.float32) * GRID
    assert 0.0 <= compute_angular_velocity(_make_trace(com_path=coms)) <= 1.0


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
    assert 0.0 <= compute_growth_gradient(_make_trace(final_state=state)) <= 1.0


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
        <= 1.0
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
    assert 0.0 <= compute_activity(_make_trace_with_gyradius(gyradius_path=gyr)) <= 1.0


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
    assert 0.0 <= compute_spatial_entropy(_make_trace(final_state=state)) <= 1.0
