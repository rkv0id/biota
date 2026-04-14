"""Tests for the rollout function.

Two layers:

1. Cheap unit tests at 32x32, 50 steps. Fast (~1s each). These verify:
   - The rollout function returns a RolloutResult with the right field shapes
   - Determinism: same params + same seed -> same result
   - Preset factories produce sane configs
   - Short rollouts get the expected 'unstable' rejection (the persistent
     filter requires >= 100 trace steps)

2. One slower integration test at 96x96, 200 steps using the M0 fixture
   params, which we know produce a stable creature. Verifies the full
   end-to-end pipeline succeeds and the descriptors land in plausible ranges.
"""

import json
from pathlib import Path

import numpy as np

from biota.search.params import sample_random
from biota.search.result import ParamDict
from biota.search.rollout import (
    THUMBNAIL_FRAMES,
    THUMBNAIL_SIZE,
    RolloutConfig,
    dev_preset,
    pretty_preset,
    rollout,
    standard_preset,
)
from biota.sim.flowlenia import Config as SimConfig

CHEAP_CONFIG = RolloutConfig(sim=SimConfig(grid_h=32, grid_w=32, kernels=10), steps=50)
"""Used for fast unit tests. 50 steps means the persistent filter always
rejects, so quality is always None and rejection_reason is always 'unstable'.
We test the structure and behavior of the rollout function, not whether the
quality function is happy."""

FIXTURE_DIR = Path(__file__).parent.parent / "sim" / "fixtures" / "jax_reference"


# === preset factories ===


def test_dev_preset_shape() -> None:
    cfg = dev_preset()
    assert cfg.sim.grid == 96
    assert cfg.steps == 200


def test_standard_preset_shape() -> None:
    cfg = standard_preset()
    assert cfg.sim.grid == 192
    assert cfg.steps == 300


def test_pretty_preset_shape() -> None:
    cfg = pretty_preset()
    assert cfg.sim.grid == 384
    assert cfg.steps == 500


# === cheap unit tests ===


def test_rollout_returns_result_for_random_params() -> None:
    params = sample_random(kernels=10, seed=0)
    result = rollout(params, seed=42, config=CHEAP_CONFIG)
    assert result.params == params
    assert result.seed == 42
    assert result.created_at > 0
    assert result.compute_seconds > 0
    # Thumbnail size is clamped to min(THUMBNAIL_SIZE, sim.grid) so a small
    # test preset doesn't get an upsampled thumbnail.
    expected_size = min(THUMBNAIL_SIZE, CHEAP_CONFIG.sim.grid)
    assert result.thumbnail.shape == (THUMBNAIL_FRAMES, expected_size, expected_size)
    assert result.thumbnail.dtype == np.uint8


def test_rollout_is_deterministic_for_fixed_seed() -> None:
    params = sample_random(kernels=10, seed=0)
    a = rollout(params, seed=99, config=CHEAP_CONFIG)
    b = rollout(params, seed=99, config=CHEAP_CONFIG)
    assert a.descriptors == b.descriptors
    assert a.quality == b.quality
    assert a.rejection_reason == b.rejection_reason
    np.testing.assert_array_equal(a.thumbnail, b.thumbnail)


def test_rollout_seed_changes_state() -> None:
    params = sample_random(kernels=10, seed=0)
    a = rollout(params, seed=1, config=CHEAP_CONFIG)
    b = rollout(params, seed=2, config=CHEAP_CONFIG)
    # Different seeds -> different initial states -> different thumbnails
    assert not np.array_equal(a.thumbnail, b.thumbnail)


def test_rollout_preserves_params_unchanged() -> None:
    params = sample_random(kernels=10, seed=0)
    original_R = params["R"]
    result = rollout(params, seed=0, config=CHEAP_CONFIG)
    assert result.params["R"] == original_R
    assert result.params == params


def test_rollout_with_parent_cell() -> None:
    params = sample_random(kernels=10, seed=0)
    result = rollout(params, seed=0, config=CHEAP_CONFIG, parent_cell=(5, 10, 3))
    assert result.parent_cell == (5, 10, 3)


def test_rollout_short_steps_yields_unstable() -> None:
    # Trace tail < 2*WINDOW means the persistent filter cannot evaluate two
    # adjacent windows, so it rejects. Confirm this is what we get.
    params = sample_random(kernels=10, seed=0)
    result = rollout(params, seed=0, config=CHEAP_CONFIG)
    assert result.quality is None
    assert result.rejection_reason == "unstable"


# === integration test using the M0 fixture ===


def _load_m0_params() -> ParamDict:
    with open(FIXTURE_DIR / "params.json") as f:
        raw = json.load(f)
    return ParamDict(
        R=float(raw["R"]),
        r=list(raw["r"]),
        m=list(raw["m"]),
        s=list(raw["s_growth"]),
        h=list(raw["h"]),
        a=list(raw["a"]),
        b=list(raw["b"]),
        w=list(raw["w"]),
    )


def test_rollout_m0_fixture_produces_accepted_creature() -> None:
    """The M0 fixture params at the dev preset produce a known-good creature.

    This is the full end-to-end smoke test. Slow (~5s on CPU) because dev
    preset is 96x96 / 200 steps with the 121-pass reintegration tracker.
    """
    params = _load_m0_params()
    result = rollout(params, seed=5678, config=dev_preset())

    assert result.descriptors is not None
    assert result.quality is not None
    assert result.rejection_reason is None
    assert 0.0 <= result.quality <= 1.0
    for d in result.descriptors:
        assert 0.0 <= d <= 1.0

    # M0 fixture is a stationary, well-localized creature with sharp internal
    # structure. Empirically observed values:
    # - velocity: small for a stationary creature, but the new normalizer
    #   is 0.02 (was 0.5), so the same raw velocity gives 25x larger
    #   normalized values. Loosen the threshold accordingly.
    # - gyradius: ~0.6 in practice; mass-weighted RMS ~15 cells against the
    #   grid/4 = 24 normalizer
    # - spectral_entropy: ~0.92 in practice. Real Lenia creatures have sharp
    #   edges and internal structure, which produces high spectral entropy
    #   even for "smooth-looking" stationary creatures. The descriptor's
    #   useful range across the discovered population is roughly [0.6, 0.95]
    #   rather than [0, 1] - it discriminates sharpness/structure rather
    #   than literal frequency content. See descriptors.py for the design
    #   discussion.
    velocity, gyradius, spectral_entropy = result.descriptors
    assert velocity < 0.3, f"M0 fixture should barely move, got velocity={velocity}"
    assert gyradius < 0.9, f"M0 fixture should not fill the grid, got gyradius={gyradius}"
    assert 0.5 < spectral_entropy < 1.0, (
        f"M0 fixture spectral_entropy should be in the structured range, "
        f"got spectral_entropy={spectral_entropy}"
    )

    # Compactness should be very high (almost all mass inside the bbox)
    assert result.quality > 0.9

    # Thumbnail should have signal, not be all zeros
    assert int(result.thumbnail.max()) > 0
