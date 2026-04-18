"""Tests for spatial observable computation in analytics.py.

Covers the boundary conditions and edge cases where bugs are likely:
- Empty inputs returning empty/zero results
- Single-patch fields counting as 1
- Fully filled fields counting as 1 (not N)
- Two separated patches counting as 2
- Diagonal-only touch counting as 2 (4-connected, not 8)
- Spatial entropy bounds (0 for concentrated, near-1 for uniform)
- Interface area only fires when both species are above threshold simultaneously
- COM distance is inf when a species has no mass
- contact_occurred accumulates across snapshots
- Homo path: initial_patch_sizes matches first snapshot
- Patch sizes are sorted descending
"""

import math

import numpy as np

from biota.ecosystem.analytics import (
    HeteroSpatial,
    HomoSpatial,
    compute_spatial_observables_hetero,
    compute_spatial_observables_homo,
    patch_count,
    patch_sizes,
    spatial_entropy,
)

# ---------------------------------------------------------------------------
# _patch_count
# ---------------------------------------------------------------------------


def test_patch_count_empty_field() -> None:
    mask = np.zeros((16, 16), dtype=bool)
    assert patch_count(mask) == 0


def test_patch_count_single_patch() -> None:
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:8, 4:8] = True
    assert patch_count(mask) == 1


def test_patch_count_fully_filled() -> None:
    # All cells true = one giant patch, not many.
    mask = np.ones((16, 16), dtype=bool)
    assert patch_count(mask) == 1


def test_patch_count_two_separated_patches() -> None:
    mask = np.zeros((16, 16), dtype=bool)
    mask[2:5, 2:5] = True
    mask[10:13, 10:13] = True
    assert patch_count(mask) == 2


def test_patch_count_diagonal_touch_is_two_patches() -> None:
    # 4-connected: diagonal-only contact does not merge patches.
    mask = np.zeros((8, 8), dtype=bool)
    mask[2, 2] = True
    mask[3, 3] = True
    assert patch_count(mask) == 2


def test_patch_count_cardinal_touch_is_one_patch() -> None:
    # Sharing an edge = same patch under 4-connectivity.
    mask = np.zeros((8, 8), dtype=bool)
    mask[3, 3] = True
    mask[3, 4] = True
    assert patch_count(mask) == 1


# ---------------------------------------------------------------------------
# _patch_sizes
# ---------------------------------------------------------------------------


def test_patch_sizes_empty_field() -> None:
    mask = np.zeros((16, 16), dtype=bool)
    assert patch_sizes(mask) == []


def test_patch_sizes_single_patch_size() -> None:
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:8, 4:8] = True  # 4x4 = 16 pixels
    assert patch_sizes(mask) == [16]


def test_patch_sizes_two_patches_descending() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[0:2, 0:2] = True  # 4 pixels
    mask[10:16, 10:16] = True  # 36 pixels
    sizes = patch_sizes(mask)
    assert sizes == [36, 4]


# ---------------------------------------------------------------------------
# _spatial_entropy
# ---------------------------------------------------------------------------


def test_spatial_entropy_empty_field_is_zero() -> None:
    field = np.zeros((32, 32), dtype=np.float32)
    assert spatial_entropy(field) == 0.0


def test_spatial_entropy_single_spike_is_low() -> None:
    field = np.zeros((32, 32), dtype=np.float32)
    field[0, 0] = 1.0
    ent = spatial_entropy(field, n_bins=8)
    # All mass in one bin -> near-zero entropy.
    assert ent < 0.1


def test_spatial_entropy_uniform_is_near_one() -> None:
    field = np.ones((32, 32), dtype=np.float32)
    ent = spatial_entropy(field, n_bins=8)
    # Perfectly uniform -> entropy == 1.0.
    assert abs(ent - 1.0) < 1e-6


def test_spatial_entropy_range() -> None:
    rng = np.random.default_rng(0)
    field = rng.random((64, 64)).astype(np.float32)
    ent = spatial_entropy(field, n_bins=8)
    assert 0.0 <= ent <= 1.0


def test_spatial_entropy_grid_smaller_than_bins_returns_zero() -> None:
    # 4x4 grid with 8 bins: bin_h = 0, should return 0.
    field = np.ones((4, 4), dtype=np.float32)
    assert spatial_entropy(field, n_bins=8) == 0.0


# ---------------------------------------------------------------------------
# compute_spatial_observables_hetero
# ---------------------------------------------------------------------------


def _make_ownership(h: int, w: int, s: int, seed: int = 0) -> np.ndarray:
    """Random (H, W, S) ownership array normalized to simplex per cell."""
    rng = np.random.default_rng(seed)
    own = rng.random((h, w, s)).astype(np.float32)
    own /= own.sum(axis=2, keepdims=True)
    return own


def test_hetero_empty_returns_empty_dataclass() -> None:
    result = compute_spatial_observables_hetero([])
    assert isinstance(result, HeteroSpatial)
    assert result.species_patch_count == []
    assert result.species_interface_area == []
    assert result.contact_occurred == []


def test_hetero_output_shapes() -> None:
    h, w, s, n_snaps = 32, 32, 3, 4
    snaps = [_make_ownership(h, w, s, seed=i) for i in range(n_snaps)]
    r = compute_spatial_observables_hetero(snaps)

    assert len(r.species_patch_count) == s
    assert all(len(row) == n_snaps for row in r.species_patch_count)

    assert len(r.species_interface_area) == n_snaps
    assert all(len(snap) == s for snap in r.species_interface_area)
    assert all(len(row) == s for snap in r.species_interface_area for row in snap)

    assert len(r.species_com_distance) == n_snaps
    assert len(r.species_spatial_entropy) == s
    assert all(len(row) == n_snaps for row in r.species_spatial_entropy)

    assert len(r.contact_occurred) == s
    assert all(len(row) == s for row in r.contact_occurred)


def test_hetero_diagonal_interface_area_is_zero() -> None:
    snaps = [_make_ownership(32, 32, 2, seed=5)]
    r = compute_spatial_observables_hetero(snaps)
    for snap in r.species_interface_area:
        for s_idx in range(len(snap)):
            assert snap[s_idx][s_idx] == 0


def test_hetero_interface_area_fires_when_species_overlap() -> None:
    h, w = 32, 32
    own = np.zeros((h, w, 2), dtype=np.float32)
    # Both species present everywhere at threshold-exceeding weight.
    own[:, :, 0] = 0.5
    own[:, :, 1] = 0.5
    r = compute_spatial_observables_hetero([own], ownership_threshold=0.3)
    assert r.species_interface_area[0][0][1] == h * w
    assert r.contact_occurred[0][1] is True


def test_hetero_interface_area_zero_when_species_separated() -> None:
    h, w = 32, 32
    own = np.zeros((h, w, 2), dtype=np.float32)
    # Species 0 owns left half, species 1 owns right half -- no overlap.
    own[:, : w // 2, 0] = 1.0
    own[:, w // 2 :, 1] = 1.0
    r = compute_spatial_observables_hetero([own], ownership_threshold=0.3)
    assert r.species_interface_area[0][0][1] == 0
    assert r.contact_occurred[0][1] is False


def test_hetero_contact_occurred_accumulates_across_snapshots() -> None:
    h, w = 32, 32
    # Snapshot 0: no overlap.
    own0 = np.zeros((h, w, 2), dtype=np.float32)
    own0[:, : w // 2, 0] = 1.0
    own0[:, w // 2 :, 1] = 1.0
    # Snapshot 1: full overlap.
    own1 = np.zeros((h, w, 2), dtype=np.float32)
    own1[:, :, 0] = 0.5
    own1[:, :, 1] = 0.5
    r = compute_spatial_observables_hetero([own0, own1], ownership_threshold=0.3)
    # Contact in snapshot 1 should propagate to contact_occurred.
    assert r.contact_occurred[0][1] is True
    assert r.contact_occurred[1][0] is True


def test_hetero_com_distance_inf_when_species_absent() -> None:
    h, w = 32, 32
    own = np.zeros((h, w, 2), dtype=np.float32)
    # Only species 0 has any mass; species 1 is zero everywhere.
    own[:, :, 0] = 1.0
    r = compute_spatial_observables_hetero([own])
    # COM for species 1 is undefined -> distance should be inf.
    assert r.species_com_distance[0][0][1] == float("inf")


def test_hetero_com_distance_finite_when_both_present() -> None:
    h, w = 32, 32
    own = np.zeros((h, w, 2), dtype=np.float32)
    own[:, : w // 2, 0] = 1.0
    own[:, w // 2 :, 1] = 1.0
    r = compute_spatial_observables_hetero([own])
    dist = r.species_com_distance[0][0][1]
    assert math.isfinite(dist)
    assert dist > 0.0


def test_hetero_patch_count_one_when_species_fully_merged() -> None:
    h, w = 32, 32
    own = np.zeros((h, w, 2), dtype=np.float32)
    own[:, :, 0] = 0.9  # Species 0 present everywhere.
    r = compute_spatial_observables_hetero([own], ownership_threshold=0.5)
    assert r.species_patch_count[0][0] == 1


# ---------------------------------------------------------------------------
# compute_spatial_observables_homo
# ---------------------------------------------------------------------------


def test_homo_empty_returns_empty_dataclass() -> None:
    result = compute_spatial_observables_homo([])
    assert isinstance(result, HomoSpatial)
    assert result.patch_count_history == []
    assert result.initial_patch_sizes == []


def test_homo_output_lengths() -> None:
    rng = np.random.default_rng(1)
    snaps = [rng.random((32, 32)).astype(np.float32) for _ in range(5)]
    r = compute_spatial_observables_homo(snaps)
    assert len(r.patch_count_history) == 5
    assert len(r.mass_spatial_entropy_history) == 5
    assert len(r.patch_size_history) == 5


def test_homo_initial_patch_sizes_matches_first_snapshot() -> None:
    rng = np.random.default_rng(2)
    snaps = [rng.random((32, 32)).astype(np.float32) for _ in range(3)]
    r = compute_spatial_observables_homo(snaps)
    assert r.initial_patch_sizes == r.patch_size_history[0]


def test_homo_single_blob_counts_as_one_patch() -> None:
    mass = np.zeros((32, 32), dtype=np.float32)
    mass[10:20, 10:20] = 1.0
    r = compute_spatial_observables_homo([mass])
    assert r.patch_count_history[0] == 1


def test_homo_two_blobs_count_as_two_patches() -> None:
    mass = np.zeros((32, 32), dtype=np.float32)
    mass[2:6, 2:6] = 1.0
    mass[20:24, 20:24] = 1.0
    r = compute_spatial_observables_homo([mass])
    assert r.patch_count_history[0] == 2


def test_homo_full_merger_yields_one_patch() -> None:
    # All mass uniformly distributed = one connected component.
    mass = np.ones((32, 32), dtype=np.float32)
    r = compute_spatial_observables_homo([mass])
    assert r.patch_count_history[0] == 1


def test_homo_patch_sizes_descending() -> None:
    mass = np.zeros((32, 32), dtype=np.float32)
    mass[0:2, 0:2] = 1.0  # 4 pixels
    mass[10:16, 10:16] = 1.0  # 36 pixels
    r = compute_spatial_observables_homo([mass])
    sizes = r.patch_size_history[0]
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[0] >= sizes[-1]


def test_homo_noise_threshold_suppresses_tiny_values() -> None:
    # A large uniform field plus tiny noise: should still count as 1 patch,
    # not many. The 1% threshold should suppress sub-threshold cells.
    rng = np.random.default_rng(42)
    mass = np.ones((32, 32), dtype=np.float32)
    mass += rng.random((32, 32)).astype(np.float32) * 0.001
    r = compute_spatial_observables_homo([mass], mass_threshold_fraction=0.01)
    assert r.patch_count_history[0] == 1


def test_homo_entropy_increases_as_mass_spreads() -> None:
    concentrated = np.zeros((32, 32), dtype=np.float32)
    concentrated[15, 15] = 1.0

    spread = np.ones((32, 32), dtype=np.float32)

    r_conc = compute_spatial_observables_homo([concentrated])
    r_spread = compute_spatial_observables_homo([spread])

    assert r_spread.mass_spatial_entropy_history[0] > r_conc.mass_spatial_entropy_history[0]
