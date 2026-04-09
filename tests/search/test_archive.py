"""Tests for the 3D MAP-Elites archive."""

import numpy as np
import pytest

from biota.search.archive import (
    Archive,
    InsertionStatus,
    bin_descriptors,
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
) -> RolloutResult:
    return RolloutResult(
        params=_params(),
        seed=seed,
        descriptors=descriptors,
        quality=quality,
        rejection_reason=rejection_reason,
        thumbnail=np.zeros((16, 32, 32), dtype=np.uint8),
        parent_cell=None,
        created_at=0.0,
        compute_seconds=0.0,
    )


# === bin_descriptors ===


def test_bin_lower_corner() -> None:
    assert bin_descriptors((0.0, 0.0, 0.0)) == (0, 0, 0)


def test_bin_upper_corner_clamps_to_last_bin() -> None:
    assert bin_descriptors((1.0, 1.0, 1.0)) == (31, 31, 15)


def test_bin_midpoint() -> None:
    coord = bin_descriptors((0.5, 0.5, 0.5))
    assert coord == (16, 16, 8)


def test_bin_custom_resolution() -> None:
    coord = bin_descriptors((0.5, 0.5, 0.5), bins_speed=10, bins_size=10, bins_structure=10)
    assert coord == (5, 5, 5)


# === insertion ===


def test_insert_into_empty_archive() -> None:
    archive = Archive()
    r = _result(descriptors=(0.5, 0.5, 0.5), quality=0.8)
    status = archive.try_insert(r)
    assert status == InsertionStatus.INSERTED
    assert len(archive) == 1
    assert (16, 16, 8) in archive


def test_insert_rejected_filter() -> None:
    archive = Archive()
    r = _result(descriptors=None, quality=None, rejection_reason="dead")
    status = archive.try_insert(r)
    assert status == InsertionStatus.REJECTED_FILTER
    assert len(archive) == 0


def test_collision_replace_when_higher_quality() -> None:
    archive = Archive()
    r1 = _result(descriptors=(0.5, 0.5, 0.5), quality=0.5, seed=1)
    r2 = _result(descriptors=(0.51, 0.51, 0.51), quality=0.8, seed=2)
    archive.try_insert(r1)
    status = archive.try_insert(r2)
    assert status == InsertionStatus.REPLACED
    assert archive[(16, 16, 8)].seed == 2


def test_collision_rejects_lower_quality() -> None:
    archive = Archive()
    r1 = _result(descriptors=(0.5, 0.5, 0.5), quality=0.8, seed=1)
    r2 = _result(descriptors=(0.51, 0.51, 0.51), quality=0.3, seed=2)
    archive.try_insert(r1)
    status = archive.try_insert(r2)
    assert status == InsertionStatus.REJECTED_QUALITY
    assert archive[(16, 16, 8)].seed == 1


def test_collision_rejects_equal_quality() -> None:
    # Strict greater-than: equal quality does not replace
    archive = Archive()
    r1 = _result(descriptors=(0.5, 0.5, 0.5), quality=0.5, seed=1)
    r2 = _result(descriptors=(0.51, 0.51, 0.51), quality=0.5, seed=2)
    archive.try_insert(r1)
    status = archive.try_insert(r2)
    assert status == InsertionStatus.REJECTED_QUALITY


# === similarity rejection ===


def test_similarity_rejects_close_neighbor_with_lower_quality() -> None:
    # Cell width on speed axis is 1/32 = 0.03125. Descriptors 0.46875 and 0.5
    # bin to cells 15 and 16 - different but adjacent cells. Their L2 distance
    # is 0.03125, well under epsilon=0.05.
    # Second quality equals first, so override (1.2x) does not apply.
    # Pin epsilon=0.05 explicitly: this test was designed around that distance
    # and the default was lowered to 0.025 in A3, so the default would no
    # longer reject this candidate.
    archive = Archive(similarity_epsilon=0.05)
    r1 = _result(descriptors=(0.5, 0.5, 0.5), quality=0.6, seed=1)
    r2 = _result(descriptors=(0.46875, 0.5, 0.5), quality=0.6, seed=2)
    archive.try_insert(r1)
    status = archive.try_insert(r2)
    assert status == InsertionStatus.REJECTED_SIMILARITY
    assert len(archive) == 1


def test_similarity_override_when_quality_much_better() -> None:
    # Same neighboring-cells setup. r2 has quality 0.7, r1 has 0.5.
    # 0.7 > 1.2 * 0.5 = 0.6, so the override applies and r2 inserts despite
    # being too descriptor-similar.
    # Pin epsilon=0.05 for the same reason as the test above.
    archive = Archive(similarity_epsilon=0.05)
    r1 = _result(descriptors=(0.5, 0.5, 0.5), quality=0.5, seed=1)
    r2 = _result(descriptors=(0.46875, 0.5, 0.5), quality=0.7, seed=2)
    archive.try_insert(r1)
    status = archive.try_insert(r2)
    assert status == InsertionStatus.INSERTED
    assert len(archive) == 2


def test_similarity_does_not_apply_to_distant_neighbors() -> None:
    # Two far apart cells, no similarity issue
    archive = Archive()
    r1 = _result(descriptors=(0.1, 0.1, 0.1), quality=0.6, seed=1)
    r2 = _result(descriptors=(0.9, 0.9, 0.9), quality=0.6, seed=2)
    archive.try_insert(r1)
    status = archive.try_insert(r2)
    assert status == InsertionStatus.INSERTED


def test_default_similarity_epsilon_rejects_very_close_pair() -> None:
    # Sanity check on the actual default epsilon (0.025 after A3).
    # speed=0.49 -> cell 15, speed=0.5 -> cell 16. Adjacent cells. Raw L2
    # distance is 0.01, well under the 0.025 default. Equal quality so the
    # 1.2x override does not apply.
    archive = Archive()  # use default epsilon
    r1 = _result(descriptors=(0.5, 0.5, 0.5), quality=0.6, seed=1)
    r2 = _result(descriptors=(0.49, 0.5, 0.5), quality=0.6, seed=2)
    archive.try_insert(r1)
    status = archive.try_insert(r2)
    assert status == InsertionStatus.REJECTED_SIMILARITY
    assert len(archive) == 1


def test_default_similarity_epsilon_admits_pair_at_old_threshold() -> None:
    # Regression test for the A3 epsilon lowering. Descriptors 0.46875 and 0.5
    # have L2 distance 0.03125, which used to trigger rejection at the old
    # epsilon=0.05 default. Under the new 0.025 default, they should pass.
    archive = Archive()  # use default epsilon
    r1 = _result(descriptors=(0.5, 0.5, 0.5), quality=0.6, seed=1)
    r2 = _result(descriptors=(0.46875, 0.5, 0.5), quality=0.6, seed=2)
    archive.try_insert(r1)
    status = archive.try_insert(r2)
    assert status == InsertionStatus.INSERTED
    assert len(archive) == 2


def test_distant_descriptors_dont_trigger_similarity() -> None:
    # Two descriptors that are far apart (one at each corner of the unit cube)
    # bin to non-adjacent cells AND are far apart in raw distance, so the
    # similarity check passes trivially. Sanity check that distant cells don't
    # interact.
    archive = Archive()
    r1 = _result(descriptors=(0.0, 0.0, 0.0), quality=0.6, seed=1)
    r2 = _result(descriptors=(0.5, 0.5, 0.5), quality=0.6, seed=2)
    archive.try_insert(r1)
    status = archive.try_insert(r2)
    assert status == InsertionStatus.INSERTED


# === random_parent ===


def test_random_parent_raises_on_empty_archive() -> None:
    archive = Archive()
    rng = np.random.default_rng(0)
    with pytest.raises(IndexError):
        archive.random_parent(rng)


def test_random_parent_returns_an_occupied_cell() -> None:
    archive = Archive()
    archive.try_insert(_result(descriptors=(0.1, 0.1, 0.1), quality=0.5, seed=1))
    archive.try_insert(_result(descriptors=(0.9, 0.9, 0.9), quality=0.5, seed=2))
    rng = np.random.default_rng(0)
    coord, result = archive.random_parent(rng)
    assert coord in archive
    assert archive[coord] is result


def test_random_parent_distribution_is_uniform() -> None:
    archive = Archive()
    for i in range(5):
        d = (i / 10.0, 0.5, 0.5)
        archive.try_insert(_result(descriptors=d, quality=0.5, seed=i))
    assert len(archive) == 5

    rng = np.random.default_rng(123)
    counts: dict[int, int] = {}
    for _ in range(5000):
        _, r = archive.random_parent(rng)
        counts[r.seed] = counts.get(r.seed, 0) + 1
    # 5 cells, 5000 samples -> expect ~1000 per cell. Allow generous slack.
    for count in counts.values():
        assert 800 < count < 1200


# === iter_occupied and metadata ===


def test_total_cells_matches_default_resolution() -> None:
    archive = Archive()
    assert archive.total_cells == 32 * 32 * 16


def test_fill_fraction_grows_with_insertions() -> None:
    archive = Archive()
    assert archive.fill_fraction == 0.0
    archive.try_insert(_result(descriptors=(0.1, 0.1, 0.1), quality=0.5, seed=1))
    assert archive.fill_fraction == pytest.approx(1 / archive.total_cells)


def test_iter_occupied_returns_all_cells() -> None:
    archive = Archive()
    archive.try_insert(_result(descriptors=(0.1, 0.1, 0.1), quality=0.5, seed=1))
    archive.try_insert(_result(descriptors=(0.9, 0.9, 0.9), quality=0.5, seed=2))
    coords = {coord for coord, _ in archive.iter_occupied()}
    assert len(coords) == 2
