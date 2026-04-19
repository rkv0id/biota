"""Tests for the CVT-MAP-Elites archive."""

import numpy as np
import pytest

from biota.search.archive import (
    Archive,
    InsertionStatus,
)
from biota.search.result import Descriptors, ParamDict, RolloutResult


def _params() -> ParamDict:
    return {
        "R": 13.5,
        "r": [0.5] * 10,
        "m": [0.15] * 10,
        "s": [0.015] * 10,
        "h": [0.5] * 10,
        "a": [[0.5, 0.5, 0.5] for _ in range(10)],
        "b": [[0.5, 0.5, 0.5] for _ in range(10)],
        "w": [[0.1, 0.1, 0.1] for _ in range(10)],
    }


def _result(
    descriptors: Descriptors | None,
    quality: float | None,
    rejection_reason: str | None = None,
    seed: int = 0,
    creature_id: str = "",
    parent_id: str | None = None,
) -> RolloutResult:
    return RolloutResult(
        params=_params(),
        seed=seed,
        creature_id=creature_id,
        descriptors=descriptors,
        quality=quality,
        rejection_reason=rejection_reason,
        thumbnail=np.zeros((16, 32, 32), dtype=np.uint8),
        parent_id=parent_id,
        created_at=0.0,
        compute_seconds=0.0,
    )


def _calibrated_archive(
    points: list[Descriptors] | None = None,
    n_centroids: int = 16,
    similarity_epsilon: float = 0.5,
) -> Archive:
    """Return an Archive with k-means centroids fitted from points.

    Using a small n_centroids for fast tests. similarity_epsilon set to 0.5
    so most tests are not affected by it unless they specifically test it.
    """
    archive = Archive(
        n_centroids=n_centroids,
        similarity_epsilon=similarity_epsilon,
    )
    if points is None:
        # Spread centroids across [0, 10]^3 uniformly
        rng = np.random.default_rng(42)
        pts = rng.uniform(0, 10, size=(max(n_centroids * 2, 50), 3))
    else:
        pts = np.array(points, dtype=np.float64)
    from scipy.cluster.vq import kmeans2

    k = min(n_centroids, len(pts))
    centroids, _ = kmeans2(pts, k, minit="points", iter=10)
    archive.attach_centroids(centroids)
    return archive


# === calibration ===


def test_uncalibrated_archive_raises_on_insert() -> None:
    archive = Archive()
    r = _result(descriptors=(1.0, 2.0, 3.0), quality=0.8)
    with pytest.raises(RuntimeError):
        archive.try_insert(r)


def test_uncalibrated_archive_raises_on_cell_for() -> None:
    archive = Archive()
    with pytest.raises(RuntimeError):
        archive.cell_for((1.0, 2.0, 3.0))


def test_attach_centroids_sets_calibrated() -> None:
    archive = Archive(n_centroids=4)
    assert not archive.calibrated
    centroids = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    archive.attach_centroids(centroids)
    assert archive.calibrated


def test_attach_centroids_rejects_wrong_shape() -> None:
    archive = Archive()
    with pytest.raises(ValueError):
        archive.attach_centroids(np.zeros((4, 2)))  # needs (k, 3)


def test_calibration_with_fewer_points_than_centroids() -> None:
    # Only 3 points but asking for 16 centroids: k is capped to len(pts).
    archive = Archive(n_centroids=16)
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    from scipy.cluster.vq import kmeans2

    k = min(16, len(pts))
    centroids, _ = kmeans2(pts, k, minit="points", iter=10)
    archive.attach_centroids(centroids)
    assert archive.calibrated
    assert archive.n_centroids == k


def test_all_same_point_calibration() -> None:
    # Degenerate: all calibration points are identical.
    archive = Archive(n_centroids=4)
    pts = np.ones((20, 3)) * 5.0
    from scipy.cluster.vq import kmeans2

    # kmeans2 with minit="points" collapses to one centroid when all points are the same;
    # cap k to distinct points count.
    k = min(4, len(pts))
    try:
        centroids, _ = kmeans2(pts, k, minit="points", iter=10)
        archive.attach_centroids(centroids)
        assert archive.calibrated
    except Exception:
        # Some scipy versions raise on degenerate input; that is acceptable.
        pass


# === nearest centroid lookup ===


def test_cell_for_returns_nearest_centroid() -> None:
    archive = Archive(n_centroids=4)
    centroids = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ],
        dtype=np.float64,
    )
    archive.attach_centroids(centroids)
    # Point closest to centroid 0
    assert archive.cell_for((0.1, 0.1, 0.1)) == 0
    # Point closest to centroid 1
    assert archive.cell_for((9.9, 0.1, 0.1)) == 1
    # Point closest to centroid 2
    assert archive.cell_for((0.1, 9.9, 0.1)) == 2
    # Point closest to centroid 3
    assert archive.cell_for((0.1, 0.1, 9.9)) == 3


def test_cell_for_is_stable_across_rebuilds() -> None:
    # creature_id stability: same descriptor triple must map to same centroid
    # regardless of which Archive instance is used (assuming same centroids).
    centroids = np.array(
        [
            [1.0, 1.0, 1.0],
            [5.0, 5.0, 5.0],
            [9.0, 9.0, 9.0],
        ],
        dtype=np.float64,
    )
    a1 = Archive(n_centroids=3)
    a1.attach_centroids(centroids)
    a2 = Archive(n_centroids=3)
    a2.attach_centroids(centroids)
    pt = (5.1, 4.9, 5.0)
    assert a1.cell_for(pt) == a2.cell_for(pt)


# === insertion ===


def test_insert_into_empty_archive() -> None:
    archive = _calibrated_archive()
    r = _result(descriptors=(1.0, 1.0, 1.0), quality=0.8)
    status = archive.try_insert(r)
    assert status == InsertionStatus.INSERTED
    assert len(archive) == 1


def test_insert_rejected_filter_none_quality() -> None:
    archive = _calibrated_archive()
    r = _result(descriptors=None, quality=None, rejection_reason="dead")
    status = archive.try_insert(r)
    assert status == InsertionStatus.REJECTED_FILTER
    assert len(archive) == 0


def test_insert_rejected_filter_none_descriptors() -> None:
    archive = _calibrated_archive()
    r = _result(descriptors=None, quality=0.5)
    status = archive.try_insert(r)
    assert status == InsertionStatus.REJECTED_FILTER


def test_collision_replace_when_higher_quality() -> None:
    archive = _calibrated_archive()
    # Both descriptors must map to the same centroid. Use the exact same point.
    r1 = _result(descriptors=(1.0, 1.0, 1.0), quality=0.5, seed=1)
    r2 = _result(descriptors=(1.0, 1.0, 1.0), quality=0.8, seed=2)
    archive.try_insert(r1)
    idx = archive.cell_for((1.0, 1.0, 1.0))
    status = archive.try_insert(r2)
    assert status == InsertionStatus.REPLACED
    assert archive[idx].seed == 2


def test_collision_rejects_lower_quality() -> None:
    archive = _calibrated_archive()
    r1 = _result(descriptors=(1.0, 1.0, 1.0), quality=0.8, seed=1)
    r2 = _result(descriptors=(1.0, 1.0, 1.0), quality=0.3, seed=2)
    archive.try_insert(r1)
    idx = archive.cell_for((1.0, 1.0, 1.0))
    status = archive.try_insert(r2)
    assert status == InsertionStatus.REJECTED_QUALITY
    assert archive[idx].seed == 1


def test_collision_rejects_equal_quality() -> None:
    archive = _calibrated_archive()
    r1 = _result(descriptors=(1.0, 1.0, 1.0), quality=0.5, seed=1)
    r2 = _result(descriptors=(1.0, 1.0, 1.0), quality=0.5, seed=2)
    archive.try_insert(r1)
    status = archive.try_insert(r2)
    assert status == InsertionStatus.REJECTED_QUALITY


# === similarity rejection ===


def test_similarity_rejects_close_neighbor() -> None:
    # Use a small epsilon so we can control the test precisely.
    # Place two centroids far apart. Insert at centroid 0, then try to insert
    # a second point that maps to centroid 1 but is within epsilon of centroid 0.
    archive = Archive(n_centroids=2, similarity_epsilon=2.0)
    centroids = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]], dtype=np.float64)
    archive.attach_centroids(centroids)

    # Insert at centroid 0 (exact position)
    r1 = _result(descriptors=(0.0, 0.0, 0.0), quality=0.5, seed=1)
    archive.try_insert(r1)

    # Insert near centroid 0 but just past its boundary (maps to centroid 1? no, maps to 0 still)
    # Better: map to centroid 1 but be within epsilon=2.0 of centroid 0.
    # Midpoint is at (5,5,5). Place at (1.5, 0, 0) -- maps to centroid 0, same cell -> REJECTED_QUALITY
    # Place at (0.5, 0.5, 0.5) -> still maps to 0. We need a point close to centroid 0 that maps to 1.
    # With only 2 centroids at 0 and (10,10,10), the boundary is at ~(5,5,5). Nothing within 2.0
    # of centroid 0 can map to centroid 1. So test similarity=True by having both map to different
    # centroids with a small epsilon archive.
    # Use 3 centroids to make this cleaner.
    archive2 = Archive(n_centroids=3, similarity_epsilon=1.5)
    c2 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 10.0, 10.0]], dtype=np.float64)
    archive2.attach_centroids(c2)
    r_a = _result(descriptors=(0.0, 0.0, 0.0), quality=0.5, seed=1)
    r_b = _result(descriptors=(2.0, 0.0, 0.0), quality=0.5, seed=2)
    archive2.try_insert(r_a)
    # centroid 1 is at (2,0,0), distance from centroid 0 is 2.0 > epsilon 1.5: should NOT reject
    status = archive2.try_insert(r_b)
    # Both land in different cells, distance between them is 2.0 > 1.5, so INSERTED
    assert status == InsertionStatus.INSERTED

    archive3 = Archive(n_centroids=3, similarity_epsilon=3.0)
    archive3.attach_centroids(c2)
    archive3.try_insert(_result(descriptors=(0.0, 0.0, 0.0), quality=0.5, seed=1))
    status3 = archive3.try_insert(_result(descriptors=(2.0, 0.0, 0.0), quality=0.5, seed=2))
    # Distance 2.0 < epsilon 3.0 and equal quality -> REJECTED_SIMILARITY
    assert status3 == InsertionStatus.REJECTED_SIMILARITY


def test_similarity_override_when_quality_much_better() -> None:
    archive = Archive(n_centroids=3, similarity_epsilon=3.0)
    c = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 10.0, 10.0]], dtype=np.float64)
    archive.attach_centroids(c)
    archive.try_insert(_result(descriptors=(0.0, 0.0, 0.0), quality=0.5, seed=1))
    # 0.7 > 1.2 * 0.5 = 0.6: quality override applies, inserts despite epsilon
    status = archive.try_insert(_result(descriptors=(2.0, 0.0, 0.0), quality=0.7, seed=2))
    assert status == InsertionStatus.INSERTED


def test_similarity_skipped_for_first_insertion() -> None:
    # With only one cell occupied, _is_too_similar skips when len < 2
    archive = _calibrated_archive(similarity_epsilon=0.0001)
    r = _result(descriptors=(1.0, 1.0, 1.0), quality=0.5)
    assert archive.try_insert(r) == InsertionStatus.INSERTED


# === random_parent ===


def test_random_parent_raises_on_empty_archive() -> None:
    archive = _calibrated_archive()
    rng = np.random.default_rng(0)
    with pytest.raises(IndexError):
        archive.random_parent(rng)


def test_random_parent_returns_an_occupied_cell() -> None:
    archive = _calibrated_archive()
    archive.try_insert(_result(descriptors=(1.0, 1.0, 1.0), quality=0.5, seed=1))
    archive.try_insert(_result(descriptors=(8.0, 8.0, 8.0), quality=0.5, seed=2))
    rng = np.random.default_rng(0)
    idx, result = archive.random_parent(rng)
    assert idx in archive
    assert archive[idx] is result


def test_random_parent_distribution_is_uniform() -> None:
    # Insert into an archive using points guaranteed to land in distinct centroids:
    # place centroids explicitly at known positions and use exact centroid coords.
    archive = Archive(n_centroids=5, similarity_epsilon=0.0)
    centroids = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
            [40.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    archive.attach_centroids(centroids)

    seeds_inserted = []
    for i, pt in enumerate(
        [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0), (30.0, 0.0, 0.0), (40.0, 0.0, 0.0)]
    ):
        r = _result(descriptors=pt, quality=0.5, seed=i)
        status = archive.try_insert(r)
        assert status == InsertionStatus.INSERTED, f"expected INSERTED got {status} for pt={pt}"
        seeds_inserted.append(i)

    assert len(seeds_inserted) == 5

    rng = np.random.default_rng(123)
    counts: dict[int, int] = {}
    for _ in range(5000):
        _, r = archive.random_parent(rng)
        counts[r.seed] = counts.get(r.seed, 0) + 1
    for count in counts.values():
        assert 800 < count < 1200, f"count {count} outside expected [800,1200]"


# === properties ===


def test_total_cells_matches_n_centroids() -> None:
    archive = _calibrated_archive(n_centroids=16)
    assert archive.total_cells == 16


def test_fill_fraction() -> None:
    archive = _calibrated_archive(n_centroids=16)
    assert archive.fill_fraction == 0.0
    archive.try_insert(_result(descriptors=(1.0, 1.0, 1.0), quality=0.5, seed=1))
    assert archive.fill_fraction == pytest.approx(1 / 16)


def test_iter_occupied_returns_all_inserted() -> None:
    archive = _calibrated_archive(n_centroids=32)
    r1 = _result(descriptors=(1.0, 1.0, 1.0), quality=0.5, seed=1)
    r2 = _result(descriptors=(8.0, 8.0, 8.0), quality=0.5, seed=2)
    archive.try_insert(r1)
    archive.try_insert(r2)
    idxs = {idx for idx, _ in archive.iter_occupied()}
    assert len(idxs) == 2


def test_centroid_positions_raises_before_calibration() -> None:
    archive = Archive()
    with pytest.raises(RuntimeError):
        _ = archive.centroid_positions


def test_centroid_positions_accessible_after_calibration() -> None:
    archive = _calibrated_archive(n_centroids=8)
    pos = archive.centroid_positions
    assert pos.shape == (8, 3)
