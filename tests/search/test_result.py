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
        descriptors=(0.3, 0.2, 0.7),
        quality=0.85,
        rejection_reason=None,
        thumbnail=_make_thumbnail(),
        parent_cell=None,
        created_at=1700000000.0,
        compute_seconds=1.5,
    )
    assert result.accepted is True
    assert result.quality == 0.85
    assert result.descriptors == (0.3, 0.2, 0.7)
    assert result.parent_cell is None


def test_rejected_result_construction() -> None:
    result = RolloutResult(
        params=_make_params(),
        seed=42,
        descriptors=(0.0, 0.95, 0.5),
        quality=None,
        rejection_reason="exploded",
        thumbnail=_make_thumbnail(),
        parent_cell=(5, 10, 3),
        created_at=1700000000.0,
        compute_seconds=1.2,
    )
    assert result.accepted is False
    assert result.rejection_reason == "exploded"
    assert result.parent_cell == (5, 10, 3)


def test_frozen_disallows_mutation() -> None:
    result = RolloutResult(
        params=_make_params(),
        seed=0,
        descriptors=(0.5, 0.5, 0.5),
        quality=0.5,
        rejection_reason=None,
        thumbnail=_make_thumbnail(),
        parent_cell=None,
        created_at=0.0,
        compute_seconds=0.0,
    )
    with pytest.raises(Exception):  # noqa: B017 - dataclasses raise FrozenInstanceError
        result.quality = 0.9  # type: ignore[misc]


def test_pickle_roundtrip_accepted() -> None:
    result = RolloutResult(
        params=_make_params(),
        seed=99,
        descriptors=(0.1, 0.4, 0.8),
        quality=0.72,
        rejection_reason=None,
        thumbnail=np.arange(16 * 32 * 32, dtype=np.uint8).reshape(16, 32, 32),
        parent_cell=(1, 2, 3),
        created_at=1700000123.5,
        compute_seconds=2.1,
    )
    blob = pickle.dumps(result)
    restored = pickle.loads(blob)
    assert restored.seed == 99
    assert restored.descriptors == (0.1, 0.4, 0.8)
    assert restored.quality == 0.72
    assert restored.parent_cell == (1, 2, 3)
    np.testing.assert_array_equal(restored.thumbnail, result.thumbnail)


def test_pickle_roundtrip_rejected() -> None:
    result = RolloutResult(
        params=_make_params(),
        seed=7,
        descriptors=None,
        quality=None,
        rejection_reason="dead",
        thumbnail=_make_thumbnail(),
        parent_cell=None,
        created_at=0.0,
        compute_seconds=0.5,
    )
    restored = pickle.loads(pickle.dumps(result))
    assert restored.accepted is False
    assert restored.descriptors is None
    assert restored.rejection_reason == "dead"
