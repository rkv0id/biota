"""Tests for RolloutResult."""

import pickle

import numpy as np
import pytest

from biota.search.result import ParamDict, RolloutResult


def _make_thumbnail() -> np.ndarray:
    return np.zeros((16, 32, 32), dtype=np.uint8)


def _make_params() -> ParamDict:
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


def test_accepted_result_construction() -> None:
    result = RolloutResult(
        params=_make_params(),
        seed=42,
        creature_id="20260419-143022-hazy-beacon-42",
        descriptors=(1.5, 0.3, 0.8),
        quality=0.85,
        rejection_reason=None,
        thumbnail=_make_thumbnail(),
        parent_id=None,
        created_at=1700000000.0,
        compute_seconds=1.5,
    )
    assert result.accepted is True
    assert result.quality == 0.85
    assert result.descriptors == (1.5, 0.3, 0.8)
    assert result.parent_id is None
    assert result.creature_id == "20260419-143022-hazy-beacon-42"


def test_rejected_result_construction() -> None:
    result = RolloutResult(
        params=_make_params(),
        seed=42,
        creature_id="",
        descriptors=(0.0, 0.95, 0.5),
        quality=None,
        rejection_reason="exploded",
        thumbnail=_make_thumbnail(),
        parent_id="20260419-143022-hazy-beacon-5",
        created_at=1700000000.0,
        compute_seconds=1.2,
    )
    assert result.accepted is False
    assert result.rejection_reason == "exploded"
    assert result.parent_id == "20260419-143022-hazy-beacon-5"


def test_frozen_disallows_mutation() -> None:
    result = RolloutResult(
        params=_make_params(),
        seed=0,
        creature_id="",
        descriptors=(0.5, 0.5, 0.5),
        quality=0.5,
        rejection_reason=None,
        thumbnail=_make_thumbnail(),
        parent_id=None,
        created_at=0.0,
        compute_seconds=0.0,
    )
    with pytest.raises(Exception):  # noqa: B017
        result.quality = 0.9  # type: ignore[misc]


def test_pickle_roundtrip_accepted() -> None:
    result = RolloutResult(
        params=_make_params(),
        seed=99,
        creature_id="20260419-143022-hazy-beacon-99",
        descriptors=(1.1, 4.2, 0.8),
        quality=0.72,
        rejection_reason=None,
        thumbnail=np.arange(16 * 32 * 32, dtype=np.uint8).reshape(16, 32, 32),
        parent_id="20260419-143022-hazy-beacon-1",
        created_at=1700000123.5,
        compute_seconds=2.1,
    )
    restored = pickle.loads(pickle.dumps(result))
    assert restored.seed == 99
    assert restored.descriptors == (1.1, 4.2, 0.8)
    assert restored.quality == 0.72
    assert restored.creature_id == "20260419-143022-hazy-beacon-99"
    assert restored.parent_id == "20260419-143022-hazy-beacon-1"
    np.testing.assert_array_equal(restored.thumbnail, result.thumbnail)


def test_pickle_roundtrip_rejected() -> None:
    result = RolloutResult(
        params=_make_params(),
        seed=7,
        creature_id="",
        descriptors=None,
        quality=None,
        rejection_reason="dead",
        thumbnail=_make_thumbnail(),
        parent_id=None,
        created_at=0.0,
        compute_seconds=0.5,
    )
    restored = pickle.loads(pickle.dumps(result))
    assert restored.accepted is False
    assert restored.descriptors is None
    assert restored.rejection_reason == "dead"


def test_creature_id_worker_default_is_empty_string() -> None:
    # Workers produce results with creature_id="" before the driver assigns it.
    result = RolloutResult(
        params=_make_params(),
        seed=0,
        creature_id="",
        descriptors=(1.0, 2.0, 3.0),
        quality=0.5,
        rejection_reason=None,
        thumbnail=_make_thumbnail(),
        parent_id=None,
        created_at=0.0,
        compute_seconds=0.0,
    )
    assert result.creature_id == ""


def test_descriptors_raw_values_not_normalized() -> None:
    # v4.0.0: descriptors are raw floats in [0, 100], not [0, 1].
    # Verify the dataclass accepts values > 1.
    result = RolloutResult(
        params=_make_params(),
        seed=0,
        creature_id="",
        descriptors=(15.3, 0.003, 0.82),
        quality=0.7,
        rejection_reason=None,
        thumbnail=_make_thumbnail(),
        parent_id=None,
        created_at=0.0,
        compute_seconds=0.0,
    )
    assert result.descriptors == (15.3, 0.003, 0.82)
