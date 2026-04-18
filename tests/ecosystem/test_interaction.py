"""Tests for interaction coefficient measurement and outcome classification.

Load-bearing tests:

1. compute_interaction_coefficients with synthetic data where coefficient
   signs are analytically known.

2. classify_outcome_hetero on constructed inputs that unambiguously trigger
   each label: exclusion, merger, fragmentation, coexistence. Also tests
   debounce behavior and per-species independence.

3. classify_outcome_homo on constructed patch count histories that trigger
   each label: full_merger, fragmentation, cannibalism, partial_clustering,
   stable_isolation.

4. build_windows debounce: transient labels below the debounce threshold
   must not produce spurious windows.

5. Edge cases: empty inputs, single species, insufficient data.
"""

import math

import numpy as np
import pytest

from biota.ecosystem.interaction import (
    OutcomeSequence,
    build_windows,
    classify_outcome_hetero,
    classify_outcome_homo,
    compute_interaction_coefficients,
)

# ---------------------------------------------------------------------------
# compute_interaction_coefficients
# ---------------------------------------------------------------------------


def _make_own_and_growth(
    h: int,
    w: int,
    n_species: int,
    n_snapshots: int,
    seed: int = 0,
) -> tuple[list[np.ndarray], list[list[np.ndarray]]]:
    rng = np.random.default_rng(seed)
    own_snaps: list[np.ndarray] = []
    growth_snaps: list[list[np.ndarray]] = []
    for _ in range(n_snapshots):
        own = rng.random((h, w, n_species)).astype(np.float32)
        own /= own.sum(axis=2, keepdims=True)
        own_snaps.append(own)
        growth_snaps.append(
            [rng.standard_normal((h, w)).astype(np.float32) for _ in range(n_species)]
        )
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
    assert compute_interaction_coefficients([], []) == []


def test_mismatched_lengths_raises() -> None:
    own, growth = _make_own_and_growth(16, 16, 2, 3, seed=4)
    with pytest.raises(ValueError, match="does not match"):
        compute_interaction_coefficients(own, growth[:2])


def test_positive_coefficient_when_b_correlates_with_high_g_a() -> None:
    h, w = 32, 32
    rng = np.random.default_rng(42)
    own_snaps, growth_snaps = [], []
    for _ in range(5):
        own = np.zeros((h, w, 2), dtype=np.float32)
        own[:, : w // 2, 1] = 0.9
        own[:, : w // 2, 0] = 0.1
        own[:, w // 2 :, 0] = 0.9
        own[:, w // 2 :, 1] = 0.1
        own_snaps.append(own)
        g_a = np.zeros((h, w), dtype=np.float32)
        g_a[:, : w // 2] = 1.0
        growth_snaps.append([g_a, rng.standard_normal((h, w)).astype(np.float32)])
    result = compute_interaction_coefficients(
        own_snaps, growth_snaps, presence_threshold=0.5, absence_threshold=0.2
    )
    assert result[0][1] > 0.3


def test_negative_coefficient_when_b_correlates_with_low_g_a() -> None:
    h, w = 32, 32
    rng = np.random.default_rng(7)
    own_snaps, growth_snaps = [], []
    for _ in range(5):
        own = np.zeros((h, w, 2), dtype=np.float32)
        own[:, : w // 2, 1] = 0.9
        own[:, : w // 2, 0] = 0.1
        own[:, w // 2 :, 0] = 0.9
        own[:, w // 2 :, 1] = 0.1
        own_snaps.append(own)
        g_a = np.zeros((h, w), dtype=np.float32)
        g_a[:, : w // 2] = -1.0
        growth_snaps.append([g_a, rng.standard_normal((h, w)).astype(np.float32)])
    result = compute_interaction_coefficients(
        own_snaps, growth_snaps, presence_threshold=0.5, absence_threshold=0.2
    )
    assert result[0][1] < -0.3


def test_interface_area_gating_restricts_to_contact_snapshots() -> None:
    # Two snapshots. Snap 0: species fully separated, no interface.
    # Snap 1: species overlap (interface > 0).
    # In snap 0: B present on right, G_A is -2 on right (B suppresses A).
    # In snap 1: B present on left, G_A is +2 everywhere.
    # Ungated result for [A][B]: averages both snapshots, coefficient pulled negative by snap 0.
    # Gated result for [A][B]: only snap 1 counts (interface[0]=0), coefficient positive.
    h, w = 32, 32
    # Snap 0: A left, B right, clean separation.
    own0 = np.zeros((h, w, 2), dtype=np.float32)
    own0[:, : w // 2, 0] = 0.95
    own0[:, : w // 2, 1] = 0.05
    own0[:, w // 2 :, 1] = 0.95
    own0[:, w // 2 :, 0] = 0.05
    # Snap 1: A right, B left -- overlap in the middle via blended region.
    own1 = np.zeros((h, w, 2), dtype=np.float32)
    own1[:, : w // 2, 1] = 0.95  # B dominates left
    own1[:, : w // 2, 0] = 0.05
    own1[:, w // 2 :, 0] = 0.95  # A dominates right
    own1[:, w // 2 :, 1] = 0.05

    # G_A snap 0: negative where B is present (right).
    g_a0 = np.zeros((h, w), dtype=np.float32)
    g_a0[:, w // 2 :] = -3.0
    # G_A snap 1: strongly positive everywhere.
    g_a1 = np.full((h, w), 3.0, dtype=np.float32)
    g_b = np.zeros((h, w), dtype=np.float32)

    own_snaps = [own0, own1]
    growth_snaps = [[g_a0, g_b], [g_a1, g_b]]
    # Interface area: 0 in snap 0 (separated), nonzero in snap 1.
    interface = [[[0, 0], [0, 0]], [[0, 10], [10, 0]]]

    result_gated = compute_interaction_coefficients(
        own_snaps,
        growth_snaps,
        presence_threshold=0.3,
        absence_threshold=0.1,
        interface_area=interface,
    )
    result_ungated = compute_interaction_coefficients(
        own_snaps,
        growth_snaps,
        presence_threshold=0.3,
        absence_threshold=0.1,
    )
    coeff_gated = result_gated[0][1]
    coeff_ungated = result_ungated[0][1]
    # Both should be finite: snap 1 for gated, both snaps for ungated.
    assert math.isfinite(coeff_gated), "gated must not be NaN"
    assert math.isfinite(coeff_ungated), "ungated must not be NaN"
    # Gated (only snap 1, positive G_A) must exceed ungated (snap 0 drags it down).
    assert coeff_gated > coeff_ungated


# ---------------------------------------------------------------------------
# build_windows
# ---------------------------------------------------------------------------


def test_build_windows_single_label() -> None:
    windows = build_windows(["coexistence"] * 5, [100, 200, 300, 400, 500], debounce=3)
    assert len(windows) == 1
    assert windows[0].label == "coexistence"
    assert windows[0].from_step == 100
    assert windows[0].to_step == 500


def test_build_windows_clean_transition() -> None:
    labels = ["coexistence"] * 4 + ["fragmentation"] * 4
    steps = list(range(100, 900, 100))
    windows = build_windows(labels, steps, debounce=3)
    assert len(windows) == 2
    assert windows[0].label == "coexistence"
    assert windows[1].label == "fragmentation"


def test_build_windows_debounce_suppresses_transient() -> None:
    labels = [
        "coexistence",
        "coexistence",
        "fragmentation",
        "fragmentation",
        "coexistence",
        "coexistence",
    ]
    steps = [100, 200, 300, 400, 500, 600]
    windows = build_windows(labels, steps, debounce=3)
    assert all(w.label == "coexistence" for w in windows)


def test_build_windows_debounce_commits_sustained_transition() -> None:
    labels = ["coexistence"] * 2 + ["fragmentation"] * 4
    steps = list(range(100, 700, 100))
    windows = build_windows(labels, steps, debounce=3)
    assert any(w.label == "fragmentation" for w in windows)


def test_build_windows_empty() -> None:
    assert build_windows([], [], debounce=3) == []


# ---------------------------------------------------------------------------
# classify_outcome_hetero
# ---------------------------------------------------------------------------


def _make_territory_history(
    n_species: int, n_steps: int, value: float = 100.0
) -> list[list[float]]:
    return [[value] * n_steps for _ in range(n_species)]


def _make_ownership_uniform(h: int, w: int, s: int) -> np.ndarray:
    own = np.zeros((h, w, s), dtype=np.float32)
    strip = w // s
    for i in range(s):
        own[:, i * strip : (i + 1) * strip, i] = 1.0
    return own


def _make_ownership_mixed(h: int, w: int, s: int) -> np.ndarray:
    return np.full((h, w, s), 1.0 / s, dtype=np.float32)


def test_hetero_empty_inputs_return_empty_sequence() -> None:
    result = classify_outcome_hetero([], [], [], [])
    assert isinstance(result, OutcomeSequence)
    assert result.series == []
    assert result.final_label == ""


def test_hetero_single_species_returns_empty() -> None:
    result = classify_outcome_hetero([[100.0] * 5], [], [100], [[1, 1]])
    assert result.series == []


def test_hetero_coexistence_stable_single_patch() -> None:
    h, w, s = 32, 32, 2
    steps = [100, 200, 300, 400, 500]
    own = _make_ownership_uniform(h, w, s)
    territory = _make_territory_history(s, 600, 100.0)
    patch_count = [[1] * 5, [1] * 5]
    result = classify_outcome_hetero(territory, [own] * 5, steps, patch_count)
    assert result.final_label == "coexistence"
    for series in result.series:
        assert all(w.label == "coexistence" for w in series)


def test_hetero_exclusion_when_territory_collapses() -> None:
    h, w, s = 32, 32, 2
    steps = [100, 200, 300, 400, 500]
    own = _make_ownership_uniform(h, w, s)
    territory = [
        [100.0] * 600,
        [100.0, 50.0, 10.0, 2.0, 1.0, 0.5] + [0.5] * 594,
    ]
    patch_count = [[1] * 5, [1] * 5]
    result = classify_outcome_hetero(territory, [own] * 5, steps, patch_count)
    assert result.final_label == "exclusion"
    assert result.series[1][-1].label == "exclusion"


def test_hetero_merger_from_high_entropy_ownership() -> None:
    h, w, s = 32, 32, 2
    steps = [100, 200, 300, 400, 500]
    own_mixed = _make_ownership_mixed(h, w, s)
    territory = _make_territory_history(s, 600, 100.0)
    patch_count = [[1] * 5, [1] * 5]
    result = classify_outcome_hetero(
        territory, [own_mixed] * 5, steps, patch_count, merger_entropy_threshold=0.5
    )
    assert result.final_label == "merger"


def test_hetero_fragmentation_when_patch_count_sustained() -> None:
    h, w, s = 32, 32, 2
    steps = [100, 200, 300, 400, 500]
    own = _make_ownership_uniform(h, w, s)
    territory = _make_territory_history(s, 600, 100.0)
    patch_count = [[3, 4, 5, 4, 3], [1] * 5]
    result = classify_outcome_hetero(
        territory, [own] * 5, steps, patch_count, fragmentation_debounce=3
    )
    assert result.final_label == "fragmentation"
    assert result.series[0][-1].label == "fragmentation"
    assert result.series[1][-1].label == "coexistence"


def test_hetero_per_species_independence() -> None:
    h, w, s = 32, 32, 3
    steps = list(range(100, 600, 100))
    own = _make_ownership_uniform(h, w, s)
    territory = [
        [100.0] * 600,
        [100.0, 60.0, 10.0, 2.0, 0.5] + [0.5] * 595,
        [100.0] * 600,
    ]
    patch_count = [[1] * 5, [1] * 5, [4, 5, 6, 5, 4]]
    result = classify_outcome_hetero(
        territory, [own] * 5, steps, patch_count, fragmentation_debounce=3
    )
    assert result.final_label == "exclusion"
    assert result.series[1][-1].label == "exclusion"
    assert result.series[2][-1].label == "fragmentation"


def test_hetero_temporal_transition_coexistence_to_fragmentation() -> None:
    h, w, s = 32, 32, 2
    steps = list(range(100, 1100, 100))
    own = _make_ownership_uniform(h, w, s)
    territory = _make_territory_history(s, 1100, 100.0)
    patch_count = [[1, 1, 1, 1, 3, 3, 3, 4, 4, 4], [1] * 10]
    result = classify_outcome_hetero(
        territory, [own] * 10, steps, patch_count, fragmentation_debounce=3
    )
    s0_labels = [w.label for w in result.series[0]]
    assert "coexistence" in s0_labels
    assert "fragmentation" in s0_labels
    coex_idx = next(i for i, w in enumerate(result.series[0]) if w.label == "coexistence")
    frag_idx = next(i for i, w in enumerate(result.series[0]) if w.label == "fragmentation")
    assert coex_idx < frag_idx


def test_hetero_series_length_matches_species_count() -> None:
    h, w, s = 32, 32, 3
    steps = [100, 200, 300]
    own = _make_ownership_uniform(h, w, s)
    territory = _make_territory_history(s, 400, 100.0)
    patch_count = [[1] * 3] * s
    result = classify_outcome_hetero(territory, [own] * 3, steps, patch_count)
    assert len(result.series) == s


def test_hetero_windows_cover_full_step_range() -> None:
    h, w, s = 32, 32, 2
    steps = [50, 100, 150, 200, 250]
    own = _make_ownership_uniform(h, w, s)
    territory = _make_territory_history(s, 300, 100.0)
    patch_count = [[1] * 5, [1] * 5]
    result = classify_outcome_hetero(territory, [own] * 5, steps, patch_count)
    for series in result.series:
        assert series[0].from_step == steps[0]
        assert series[-1].to_step == steps[-1]


# ---------------------------------------------------------------------------
# classify_outcome_homo
# ---------------------------------------------------------------------------


def test_homo_empty_returns_empty() -> None:
    result = classify_outcome_homo([], [], [], [])
    assert isinstance(result, OutcomeSequence)
    assert result.series == []


def test_homo_full_merger_when_count_reaches_one() -> None:
    steps = [100, 200, 300, 400, 500, 600, 700]
    counts = [6, 4, 2, 1, 1, 1, 1]
    sizes = [[100] * 6, [120] * 4, [150] * 2, [200], [210], [220], [230]]
    result = classify_outcome_homo(steps, counts, sizes, [100] * 6)
    assert result.final_label == "full_merger"
    assert result.series[0][-1].label == "full_merger"


def test_homo_stable_isolation_when_count_unchanged() -> None:
    steps = [100, 200, 300, 400, 500]
    counts = [4, 4, 4, 4, 4]
    sizes = [[100, 100, 100, 100]] * 5
    result = classify_outcome_homo(steps, counts, sizes, [100, 100, 100, 100])
    assert result.final_label == "stable_isolation"


def test_homo_fragmentation_when_count_exceeds_initial() -> None:
    steps = [100, 200, 300, 400, 500]
    counts = [3, 4, 5, 6, 6]
    sizes = [[80] * c for c in counts]
    result = classify_outcome_homo(steps, counts, sizes, [100, 100, 100])
    assert result.final_label == "fragmentation"


def test_homo_cannibalism_when_largest_patch_grows() -> None:
    steps = [100, 200, 300, 400, 500]
    initial = [100, 100, 100, 100]  # median = 100, threshold = 150
    counts = [4, 3, 3, 3, 3]
    sizes = [
        [100, 100, 100, 100],
        [200, 80, 80],
        [220, 70, 70],
        [240, 60, 60],
        [260, 50, 50],
    ]
    result = classify_outcome_homo(steps, counts, sizes, initial)
    assert result.final_label == "cannibalism"


def test_homo_partial_clustering_without_large_patches() -> None:
    steps = [100, 200, 300, 400, 500]
    initial = [100, 100, 100, 100]  # 1.5x median = 150
    counts = [4, 3, 3, 3, 3]
    sizes = [
        [100, 100, 100, 100],
        [120, 100, 80],
        [125, 95, 80],
        [130, 90, 80],
        [130, 90, 80],
    ]
    result = classify_outcome_homo(steps, counts, sizes, initial)
    assert result.final_label == "partial_clustering"


def test_homo_single_entry_in_series() -> None:
    steps = [100, 200, 300]
    counts = [3, 3, 3]
    sizes = [[100, 100, 100]] * 3
    result = classify_outcome_homo(steps, counts, sizes, [100, 100, 100])
    assert len(result.series) == 1


def test_homo_windows_cover_full_step_range() -> None:
    steps = [50, 100, 150, 200]
    counts = [3, 3, 3, 3]
    sizes = [[100, 100, 100]] * 4
    result = classify_outcome_homo(steps, counts, sizes, [100, 100, 100])
    assert result.series[0][0].from_step == 50
    assert result.series[0][-1].to_step == 200


def test_homo_temporal_isolation_to_merger() -> None:
    steps = list(range(100, 800, 100))
    counts = [4, 4, 3, 2, 2, 1, 1]
    initial = [100, 100, 100, 100]
    sizes = [
        [100, 100, 100, 100],
        [100, 100, 100, 100],
        [120, 110, 90],
        [150, 130],
        [160, 120],
        [300],
        [320],
    ]
    result = classify_outcome_homo(steps, counts, sizes, initial, debounce=2)
    all_labels = [w.label for w in result.series[0]]
    assert "full_merger" in all_labels
