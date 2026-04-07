"""Tests for the parameter sampler."""

import numpy as np

from biota.search.params import (
    PARAMETER_SPECS,
    in_range,
    mutate,
    sample_random,
)
from biota.search.result import ParamDict


def test_sample_random_is_deterministic() -> None:
    a = sample_random(kernels=10, seed=42)
    b = sample_random(kernels=10, seed=42)
    assert a == b


def test_different_seeds_give_different_samples() -> None:
    a = sample_random(kernels=10, seed=1)
    b = sample_random(kernels=10, seed=2)
    assert a != b


def test_sample_has_all_expected_fields() -> None:
    sample = sample_random(kernels=10, seed=0)
    expected = {spec.name for spec in PARAMETER_SPECS}
    assert set(sample.keys()) == expected


def test_sample_field_shapes_match_specs() -> None:
    kernels = 10
    sample = sample_random(kernels=kernels, seed=0)
    assert isinstance(sample["R"], float)
    for name in ("r", "m", "s", "h"):
        values: list[float] = sample[name]  # type: ignore[literal-required]
        assert isinstance(values, list)
        assert len(values) == kernels
        for v in values:
            assert isinstance(v, float)
    for name in ("a", "b", "w"):
        rows: list[list[float]] = sample[name]  # type: ignore[literal-required]
        assert isinstance(rows, list)
        assert len(rows) == kernels
        for row in rows:
            assert isinstance(row, list)
            assert len(row) == 3


def test_sample_within_declared_ranges() -> None:
    # 100 random samples, each must pass in_range
    for seed in range(100):
        sample = sample_random(kernels=10, seed=seed)
        assert in_range(sample), f"seed {seed} produced out-of-range params"


def test_sample_uses_correct_kernel_count() -> None:
    sample = sample_random(kernels=20, seed=0)
    assert len(sample["r"]) == 20
    assert len(sample["a"]) == 20
    assert len(sample["a"][0]) == 3


# === mutation ===


def test_mutate_is_deterministic() -> None:
    parent = sample_random(kernels=10, seed=0)
    a = mutate(parent, seed=42)
    b = mutate(parent, seed=42)
    assert a == b


def test_mutate_changes_parent() -> None:
    parent = sample_random(kernels=10, seed=0)
    child = mutate(parent, seed=42)
    assert child != parent


def test_mutate_stays_in_range() -> None:
    parent = sample_random(kernels=10, seed=0)
    for seed in range(100):
        child = mutate(parent, seed=seed)
        assert in_range(child), f"mutation seed {seed} produced out-of-range params"


def test_mutate_clips_at_boundary() -> None:
    # Construct a parent at the edge of every range, mutate hard, verify clipping
    parent = ParamDict(
        R=25.0,  # at upper bound
        r=[1.0] * 10,
        m=[0.5] * 10,
        s=[0.18] * 10,
        h=[1.0] * 10,
        a=[[1.0, 1.0, 1.0]] * 10,
        b=[[1.0, 1.0, 1.0]] * 10,
        w=[[0.5, 0.5, 0.5]] * 10,
    )
    # Many random mutations, all must stay in range thanks to clipping
    for seed in range(50):
        child = mutate(parent, seed=seed)
        assert in_range(child)


def test_mutate_perturbation_is_small_for_typical_seeds() -> None:
    # Mean absolute perturbation should be roughly the sigma. Pick R with sigma 2.3.
    parent = sample_random(kernels=10, seed=0)
    deltas = []
    for seed in range(200):
        child = mutate(parent, seed=seed)
        deltas.append(abs(child["R"] - parent["R"]))
    mean_delta = float(np.mean(deltas))
    # Expected mean of |Normal(0, 2.3)| is about 1.83 (sigma * sqrt(2/pi))
    # Add tolerance for clipping near edges
    assert 1.0 < mean_delta < 3.0


# === in_range helper ===


def test_in_range_accepts_valid() -> None:
    sample = sample_random(kernels=10, seed=0)
    assert in_range(sample) is True


def test_in_range_rejects_below_low() -> None:
    sample = sample_random(kernels=10, seed=0)
    sample["R"] = 1.0  # below low (2.0)
    assert in_range(sample) is False


def test_in_range_rejects_above_high() -> None:
    sample = sample_random(kernels=10, seed=0)
    sample["R"] = 30.0  # above high (25.0)
    assert in_range(sample) is False


def test_in_range_checks_nested_lists() -> None:
    sample = sample_random(kernels=10, seed=0)
    a_field = sample["a"]
    assert isinstance(a_field, list)
    a_field[0][0] = 1.5  # above [0, 1] range
    assert in_range(sample) is False
