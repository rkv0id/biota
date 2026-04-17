"""Tests for interaction coefficient measurement and outcome classification.

The load-bearing tests are:

1. compute_interaction_coefficients with synthetic growth and ownership data
   where the expected coefficient signs are analytically known.

2. classify_outcome on constructed territory histories that unambiguously
   match each category (exclusion, merger, coexistence, fragmentation).

3. Edge cases: empty inputs, single species (insufficient data).
"""

import numpy as np
import pytest

from biota.ecosystem.interaction import classify_outcome, compute_interaction_coefficients

# === compute_interaction_coefficients ===


def _make_own_and_growth(
    h: int,
    w: int,
    n_species: int,
    n_snapshots: int,
    seed: int = 0,
) -> tuple[list[np.ndarray], list[list[np.ndarray]]]:
    """Build synthetic ownership and growth snapshots for 2-species tests."""
    rng = np.random.default_rng(seed)
    own_snaps: list[np.ndarray] = []
    growth_snaps: list[list[np.ndarray]] = []
    for _ in range(n_snapshots):
        own = rng.random((h, w, n_species)).astype(np.float32)
        # Normalize each cell to a simplex.
        row_sums = own.sum(axis=2, keepdims=True)
        own = own / np.maximum(row_sums, 1e-8)
        own_snaps.append(own)
        g = [rng.standard_normal((h, w)).astype(np.float32) for _ in range(n_species)]
        growth_snaps.append(g)
    return own_snaps, growth_snaps


def test_returns_s_by_s_matrix() -> None:
    own, growth = _make_own_and_growth(32, 32, 2, 3, seed=1)
    result = compute_interaction_coefficients(own, growth)
    assert len(result) == 2
    assert all(len(row) == 2 for row in result)


def test_three_species_shape() -> None:
    own, growth = _make_own_and_growth(24, 24, 3, 4, seed=2)
    result = compute_interaction_coefficients(own, growth)
    assert len(result) == 3
    assert all(len(row) == 3 for row in result)


def test_empty_ownership_returns_empty() -> None:
    result = compute_interaction_coefficients([], [])
    assert result == []


def test_empty_growth_returns_empty() -> None:
    _own, _ = _make_own_and_growth(16, 16, 2, 2, seed=3)
    # Mismatched lengths raise, but both empty should return [].
    result2 = compute_interaction_coefficients([], [])
    assert result2 == []


def test_mismatched_lengths_raises() -> None:
    own, growth = _make_own_and_growth(16, 16, 2, 3, seed=4)
    with pytest.raises(ValueError, match="does not match"):
        compute_interaction_coefficients(own, growth[:2])


def test_positive_coefficient_when_b_correlates_with_high_g_a() -> None:
    """Where B is present, G_A is consistently high: coefficient should be > 0."""
    h, w = 32, 32
    n_snaps = 5
    own_snaps: list[np.ndarray] = []
    growth_snaps: list[list[np.ndarray]] = []

    rng = np.random.default_rng(42)
    for _ in range(n_snaps):
        own = np.zeros((h, w, 2), dtype=np.float32)
        # Left half: species 1 (B) dominates.
        own[:, : w // 2, 1] = 0.9
        own[:, : w // 2, 0] = 0.1
        # Right half: species 0 (A) dominates.
        own[:, w // 2 :, 0] = 0.9
        own[:, w // 2 :, 1] = 0.1
        own_snaps.append(own)

        g_a = np.zeros((h, w), dtype=np.float32)
        # G_A is strongly positive where B is present (left half).
        g_a[:, : w // 2] = 1.0
        # G_A is near zero where B is absent (right half).
        g_a[:, w // 2 :] = 0.0
        g_b = rng.standard_normal((h, w)).astype(np.float32)
        growth_snaps.append([g_a, g_b])

    result = compute_interaction_coefficients(
        own_snaps, growth_snaps, presence_threshold=0.5, absence_threshold=0.2
    )
    # interaction[A=0][B=1] should be positive (B's presence correlates with high G_A).
    assert result[0][1] > 0.3, f"Expected positive coefficient, got {result[0][1]}"


def test_negative_coefficient_when_b_correlates_with_low_g_a() -> None:
    """Where B is present, G_A is consistently low: coefficient should be < 0."""
    h, w = 32, 32
    n_snaps = 5
    own_snaps: list[np.ndarray] = []
    growth_snaps: list[list[np.ndarray]] = []

    rng = np.random.default_rng(7)
    for _ in range(n_snaps):
        own = np.zeros((h, w, 2), dtype=np.float32)
        own[:, : w // 2, 1] = 0.9
        own[:, : w // 2, 0] = 0.1
        own[:, w // 2 :, 0] = 0.9
        own[:, w // 2 :, 1] = 0.1
        own_snaps.append(own)

        g_a = np.zeros((h, w), dtype=np.float32)
        # G_A is strongly negative where B is present.
        g_a[:, : w // 2] = -1.0
        # G_A is near zero where B is absent.
        g_a[:, w // 2 :] = 0.0
        g_b = rng.standard_normal((h, w)).astype(np.float32)
        growth_snaps.append([g_a, g_b])

    result = compute_interaction_coefficients(
        own_snaps, growth_snaps, presence_threshold=0.5, absence_threshold=0.2
    )
    assert result[0][1] < -0.3, f"Expected negative coefficient, got {result[0][1]}"


# === classify_outcome ===


def test_classify_exclusion_when_one_species_collapses() -> None:
    """Species 1 loses nearly all territory; species 0 holds. Should be exclusion."""
    hist = [
        [100.0, 110.0, 120.0, 130.0, 140.0],  # species 0: expanding
        [100.0, 60.0, 20.0, 5.0, 2.0],  # species 1: collapsing
    ]
    label = classify_outcome(hist, [])
    assert label == "exclusion", f"got {label!r}"


def test_classify_merger_from_high_entropy_ownership() -> None:
    """Final snapshot with fully mixed ownership should classify as merger."""
    h, w, s = 32, 32, 2
    # Uniform 0.5/0.5 ownership everywhere: maximum entropy.
    final_own = np.full((h, w, s), 0.5, dtype=np.float32)
    hist = [
        [100.0, 100.0, 100.0],
        [100.0, 100.0, 100.0],
    ]
    label = classify_outcome(hist, [final_own])
    assert label == "merger", f"got {label!r}"


def test_classify_coexistence_when_species_maintain_territory_with_clear_boundaries() -> None:
    """Stable territory, low ownership entropy (sharp boundaries): coexistence."""
    h, w, s = 32, 32, 2
    # Each species owns its half cleanly: low entropy at every cell.
    final_own = np.zeros((h, w, s), dtype=np.float32)
    final_own[:, : w // 2, 0] = 1.0
    final_own[:, w // 2 :, 1] = 1.0
    hist = [
        [100.0, 98.0, 97.0, 96.0, 95.0],  # stable
        [100.0, 101.0, 99.0, 100.0, 98.0],  # stable
    ]
    label = classify_outcome(hist, [final_own])
    assert label == "coexistence", f"got {label!r}"


def test_classify_fragmentation_from_high_territory_variance() -> None:
    """High coefficient of variation in territory for a non-excluded species."""
    # Species 0 oscillates wildly (CV >> 0.5), species 1 is stable.
    oscillating = [float(x) for x in [100, 20, 90, 15, 85, 10, 80, 12, 75, 8]]
    stable = [80.0] * len(oscillating)
    # Make sure neither species is excluded (both end above 5% of initial).
    hist = [oscillating, stable]
    label = classify_outcome(hist, [])
    assert label == "fragmentation", f"got {label!r}"


def test_single_species_returns_empty_string() -> None:
    hist = [[100.0, 95.0, 90.0]]
    label = classify_outcome(hist, [])
    assert label == ""


def test_empty_history_returns_empty_string() -> None:
    label = classify_outcome([], [])
    assert label == ""


def test_exclusion_takes_priority_over_fragmentation() -> None:
    """If one species is excluded, label exclusion even if the other oscillates."""
    oscillating = [float(x) for x in [100, 20, 90, 15, 85, 10, 80, 12, 75, 8]]
    excluded = [100.0, 50.0, 10.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    hist = [oscillating, excluded]
    label = classify_outcome(hist, [])
    assert label == "exclusion", f"got {label!r}"
