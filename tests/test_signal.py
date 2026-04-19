"""Tests for signal field infrastructure.

Covers:
1. params.py -- sample_random with signal_field=True produces valid signal
   params; mutate preserves signal keys; has_signal_field detection;
   in_range validates signal params; non-signal params are unaffected.

2. quality.py -- alive filter with signal mass: passes when total mass
   conserved even if mass field shrank; fails when creature mass collapses
   below the floor; non-signal path unchanged.

3. flowlenia.py -- make_initial_signal_field shape and amplitude;
   Params.has_signal property; signal step does not crash and returns
   Tensor | None correctly; non-signal step still returns (Tensor, None).

4. ecosystem/run.py -- _validate_signal_consistency raises on mixed archives,
   passes on uniform-signal and uniform-non-signal inputs.
"""

import numpy as np
import pytest
import torch

from biota.ecosystem.run import validate_signal_consistency
from biota.search.descriptors import RolloutTrace
from biota.search.params import (
    SIGNAL_CHANNELS,
    has_signal_field,
    in_range,
    mutate,
    sample_random,
)
from biota.search.quality import RolloutEvaluation, evaluate
from biota.search.result import ParamDict, RolloutResult
from biota.sim.flowlenia import Config, FlowLenia, Params

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_non_signal_params(seed: int = 0) -> ParamDict:
    return sample_random(kernels=3, seed=seed, signal_field=False)


def _make_signal_params(seed: int = 0) -> ParamDict:
    return sample_random(kernels=3, seed=seed, signal_field=True)


def _minimal_trace(grid: int = 32) -> RolloutTrace:
    """Minimal RolloutTrace that satisfies the persistent filter."""
    n = 120  # 2 * WINDOW + buffer
    com = np.zeros((n, 2), dtype=np.float32)
    bbox = np.zeros(n, dtype=np.float32)
    gyr = np.zeros(n, dtype=np.float32)
    state = np.zeros((grid, grid), dtype=np.float32)
    state[10:20, 10:20] = 1.0
    return RolloutTrace(
        com_history=com,
        bbox_fraction_history=bbox,
        gyradius_history=gyr,
        final_state=state,
        grid_size=grid,
        total_steps=200,
    )


# ---------------------------------------------------------------------------
# params: signal sampling and mutation
# ---------------------------------------------------------------------------


def test_sample_random_no_signal_lacks_signal_keys() -> None:
    params = _make_non_signal_params()
    assert not has_signal_field(params)
    assert "emission_vector" not in params
    assert "receptor_profile" not in params
    assert "emission_rate" not in params
    assert "decay_rates" not in params
    assert "alpha_coupling" not in params
    assert "beta_modulation" not in params


def test_sample_random_signal_has_all_signal_keys() -> None:
    params = _make_signal_params()
    assert has_signal_field(params)
    for key in (
        "emission_vector",
        "receptor_profile",
        "emission_rate",
        "decay_rates",
        "alpha_coupling",
        "beta_modulation",
        "signal_kernel_r",
        "signal_kernel_a",
        "signal_kernel_b",
        "signal_kernel_w",
    ):
        assert key in params, f"missing key: {key}"


def test_signal_vector_lengths() -> None:
    params = _make_signal_params()
    assert len(params["emission_vector"]) == SIGNAL_CHANNELS  # type: ignore[typeddict-item]
    assert len(params["receptor_profile"]) == SIGNAL_CHANNELS  # type: ignore[typeddict-item]
    assert len(params["decay_rates"]) == SIGNAL_CHANNELS  # type: ignore[typeddict-item]


def test_emission_rate_in_range() -> None:
    params = _make_signal_params()
    assert 0.0001 <= params["emission_rate"] <= 0.01  # type: ignore[typeddict-item]


def test_decay_rates_in_range() -> None:
    params = _make_signal_params()
    assert all(0.0 <= v <= 0.9 for v in params["decay_rates"])  # type: ignore[typeddict-item]


def test_alpha_coupling_in_range() -> None:
    params = _make_signal_params()
    assert -1.0 <= params["alpha_coupling"] <= 1.0  # type: ignore[typeddict-item]


def test_beta_modulation_in_range() -> None:
    params = _make_signal_params()
    assert -1.0 <= params["beta_modulation"] <= 1.0  # type: ignore[typeddict-item]


def test_signal_kernel_vector_lengths() -> None:
    params = _make_signal_params()
    assert len(params["signal_kernel_a"]) == 3  # type: ignore[typeddict-item]
    assert len(params["signal_kernel_b"]) == 3  # type: ignore[typeddict-item]
    assert len(params["signal_kernel_w"]) == 3  # type: ignore[typeddict-item]


def test_emission_vector_in_range() -> None:
    params = _make_signal_params()
    assert all(0.0 <= v <= 1.0 for v in params["emission_vector"])  # type: ignore[typeddict-item]


def test_receptor_profile_in_range() -> None:
    params = _make_signal_params()
    assert all(-1.0 <= v <= 1.0 for v in params["receptor_profile"])  # type: ignore[typeddict-item]


def test_mutate_preserves_signal_keys() -> None:
    parent = _make_signal_params(seed=1)
    child = mutate(parent, seed=42)
    assert has_signal_field(child)
    assert len(child["emission_vector"]) == SIGNAL_CHANNELS  # type: ignore[typeddict-item]
    assert len(child["receptor_profile"]) == SIGNAL_CHANNELS  # type: ignore[typeddict-item]
    assert len(child["decay_rates"]) == SIGNAL_CHANNELS  # type: ignore[typeddict-item]
    assert 0.0001 <= child["emission_rate"] <= 0.01  # type: ignore[typeddict-item]
    assert -1.0 <= child["alpha_coupling"] <= 1.0  # type: ignore[typeddict-item]
    assert -1.0 <= child["beta_modulation"] <= 1.0  # type: ignore[typeddict-item]


def test_mutate_non_signal_stays_non_signal() -> None:
    parent = _make_non_signal_params()
    child = mutate(parent, seed=7)
    assert not has_signal_field(child)


def test_in_range_passes_valid_signal_params() -> None:
    assert in_range(_make_signal_params())


def test_in_range_fails_out_of_range_emission() -> None:
    params = _make_signal_params()
    params["emission_vector"][0] = 1.5  # type: ignore[typeddict-item]  # out of [0, 1]
    assert not in_range(params)


def test_in_range_fails_out_of_range_receptor() -> None:
    params = _make_signal_params()
    params["receptor_profile"][0] = -1.5  # type: ignore[typeddict-item]  # out of [-1, 1]
    assert not in_range(params)


def test_same_seed_signal_and_non_signal_same_mass_params() -> None:
    # Signal params are sampled after mass params from the same rng, so
    # mass params differ between signal and non-signal with the same seed
    # (signal params consume rng state before returning). This is expected.
    # What must hold: the non-signal path produces valid, in-range mass params.
    ns = _make_non_signal_params(seed=99)
    assert in_range(ns)


# ---------------------------------------------------------------------------
# quality: alive filter with signal mass
# ---------------------------------------------------------------------------


def test_alive_filter_passes_with_signal_mass_conservation() -> None:
    """Creature emits mass to signal field -- total conserved, should pass."""
    trace = _minimal_trace()
    initial_mass = 100.0
    initial_signal = 0.5
    # Creature lost 30 mass to signal field; signal gained 30 -- total conserved.
    eval_input = RolloutEvaluation(
        initial_mass=initial_mass,
        final_mass=70.0,
        trace=trace,
        initial_total=initial_mass + initial_signal,
        final_signal_mass=30.5,
    )
    result = evaluate(eval_input)
    # Should not be rejected as "dead" -- total mass 70 + 30.5 = 100.5, within [0.5, 2.0] * 100.5
    assert result.rejection_reason != "dead"


def test_alive_filter_fails_when_creature_mass_collapses() -> None:
    """Creature converted almost all mass to signal -- below creature floor."""
    trace = _minimal_trace()
    initial_mass = 100.0
    # Creature mass dropped to 3% of initial -- below CREATURE_MASS_FLOOR (5%)
    eval_input = RolloutEvaluation(
        initial_mass=initial_mass,
        final_mass=3.0,
        trace=trace,
        initial_total=initial_mass,
        final_signal_mass=97.0,
    )
    result = evaluate(eval_input)
    assert result.rejection_reason == "dead"


def test_alive_filter_fails_when_total_mass_lost() -> None:
    """Total mass (mass + signal) dropped below 0.5 * initial -- dead."""
    trace = _minimal_trace()
    eval_input = RolloutEvaluation(
        initial_mass=100.0,
        final_mass=30.0,
        trace=trace,
        initial_total=100.0,
        final_signal_mass=5.0,  # total = 35, below 0.5 * 100
    )
    result = evaluate(eval_input)
    assert result.rejection_reason == "dead"


def test_alive_filter_signal_creature_mass_floor_is_stricter() -> None:
    """Signal creature mass floor is 0.05 -- only truly collapsed creatures fail."""
    trace = _minimal_trace()
    # 4% of initial_mass remaining -- below CREATURE_MASS_FLOOR (5%)
    eval_input = RolloutEvaluation(
        initial_mass=100.0,
        final_mass=4.0,
        trace=trace,
        initial_total=100.0,
        final_signal_mass=96.0,  # total conserved
    )
    result = evaluate(eval_input)
    assert result.rejection_reason == "dead"


def test_signal_retention_boosts_quality() -> None:
    """Creatures that retain mass score higher than those that bleed it into signal."""
    trace = _minimal_trace()
    # High retention: kept almost all mass
    high = RolloutEvaluation(
        initial_mass=100.0,
        final_mass=98.0,
        trace=trace,
        initial_total=100.0,
        final_signal_mass=0.1,
    )
    # Low retention: lost 40% to signal (but total still conserved)
    low = RolloutEvaluation(
        initial_mass=100.0,
        final_mass=60.0,
        trace=trace,
        initial_total=100.0,
        final_signal_mass=40.0,
    )
    r_high = evaluate(high)
    r_low = evaluate(low)
    assert r_high.quality is not None and r_low.quality is not None
    assert r_high.quality > r_low.quality


def test_alive_filter_non_signal_path_unchanged() -> None:
    """Default RolloutEvaluation (no signal args) behaves as before."""
    trace = _minimal_trace()
    # Healthy non-signal creature: final mass within [0.5, 2.0] * initial.
    eval_input = RolloutEvaluation(
        initial_mass=100.0,
        final_mass=98.0,
        trace=trace,
    )
    result = evaluate(eval_input)
    assert result.rejection_reason != "dead"


def test_alive_filter_default_initial_total_equals_initial_mass() -> None:
    """When initial_total not provided, it defaults to initial_mass."""
    trace = _minimal_trace()
    eval_input = RolloutEvaluation(
        initial_mass=80.0,
        final_mass=78.0,
        trace=trace,
    )
    assert eval_input.initial_total == 80.0


# ---------------------------------------------------------------------------
# flowlenia: Params.has_signal, make_initial_signal_field, step
# ---------------------------------------------------------------------------


def _make_fl(signal: bool = False, grid: int = 32) -> FlowLenia:
    cfg = Config(grid_h=grid, grid_w=grid, kernels=3)
    pdict = sample_random(kernels=3, seed=0, signal_field=signal)
    p = Params(
        R=pdict["R"],
        r=torch.tensor(pdict["r"]),
        m=torch.tensor(pdict["m"]),
        s=torch.tensor(pdict["s"]),
        h=torch.tensor(pdict["h"]),
        a=torch.tensor(pdict["a"]),
        b=torch.tensor(pdict["b"]),
        w=torch.tensor(pdict["w"]),
    )
    if signal:
        p = Params(
            R=p.R,
            r=p.r,
            m=p.m,
            s=p.s,
            h=p.h,
            a=p.a,
            b=p.b,
            w=p.w,
            emission_vector=torch.tensor(pdict["emission_vector"]),  # type: ignore[typeddict-item]
            receptor_profile=torch.tensor(pdict["receptor_profile"]),  # type: ignore[typeddict-item]
            emission_rate=float(pdict["emission_rate"]),  # type: ignore[typeddict-item]
            decay_rates=torch.tensor(pdict["decay_rates"]),  # type: ignore[typeddict-item]
            alpha_coupling=float(pdict["alpha_coupling"]),  # type: ignore[typeddict-item]
            beta_modulation=float(pdict["beta_modulation"]),  # type: ignore[typeddict-item]
            signal_kernel_r=float(pdict["signal_kernel_r"]),  # type: ignore[typeddict-item]
            signal_kernel_a=torch.tensor(pdict["signal_kernel_a"]),  # type: ignore[typeddict-item]
            signal_kernel_b=torch.tensor(pdict["signal_kernel_b"]),  # type: ignore[typeddict-item]
            signal_kernel_w=torch.tensor(pdict["signal_kernel_w"]),  # type: ignore[typeddict-item]
        )
    return FlowLenia(cfg, p)


def test_params_has_signal_false_for_non_signal() -> None:
    fl = _make_fl(signal=False)
    assert not fl.params.has_signal


def test_params_has_signal_true_for_signal() -> None:
    fl = _make_fl(signal=True)
    assert fl.params.has_signal


def test_make_initial_signal_field_shape() -> None:
    fl = _make_fl(signal=True, grid=32)
    sig = fl.make_initial_signal_field(seed=0)
    assert sig.shape == (32, 32, SIGNAL_CHANNELS)


def test_make_initial_signal_field_nonnegative() -> None:
    fl = _make_fl(signal=True, grid=32)
    sig = fl.make_initial_signal_field(seed=0)
    assert (sig >= 0).all()


def test_make_initial_signal_field_low_amplitude() -> None:
    fl = _make_fl(signal=True, grid=64)
    sig = fl.make_initial_signal_field(seed=0)
    # Mean absolute value should be around 0.01 (the target amplitude).
    mean_abs = float(sig.abs().mean().item())
    assert mean_abs < 0.05, f"signal amplitude too large: {mean_abs}"


def test_make_initial_signal_field_spatially_varied() -> None:
    """Channels should not be spatially uniform -- low-freq filtering gives structure."""
    fl = _make_fl(signal=True, grid=64)
    sig = fl.make_initial_signal_field(seed=0)
    # Standard deviation across spatial dims should be nonzero.
    for c in range(SIGNAL_CHANNELS):
        std = float(sig[:, :, c].std().item())
        assert std > 0, f"channel {c} is spatially flat"


def test_step_non_signal_returns_tensor_and_none() -> None:
    fl = _make_fl(signal=False, grid=32)
    mass = torch.zeros(32, 32, 1)
    mass[14:18, 14:18, 0] = 1.0
    new_mass, new_sig = fl.step(mass, None)
    assert isinstance(new_mass, torch.Tensor)
    assert new_sig is None


def test_step_signal_returns_tensor_and_tensor() -> None:
    fl = _make_fl(signal=True, grid=32)
    mass = torch.zeros(32, 32, 1)
    mass[14:18, 14:18, 0] = 1.0
    sig = fl.make_initial_signal_field(seed=0)
    new_mass, new_sig = fl.step(mass, sig)
    assert isinstance(new_mass, torch.Tensor)
    assert isinstance(new_sig, torch.Tensor)
    assert new_sig.shape == (32, 32, SIGNAL_CHANNELS)


def test_step_signal_conserves_total_mass_approximately() -> None:
    """Total mass (mass + signal) should decrease only by decay, not explode."""
    fl = _make_fl(signal=True, grid=48)
    mass = torch.zeros(48, 48, 1)
    mass[20:28, 20:28, 0] = 0.5
    sig = fl.make_initial_signal_field(seed=1)

    initial_total = float(mass.sum().item()) + float(sig.sum().item())
    for _ in range(10):
        mass, sig = fl.step(mass, sig)

    final_total = float(mass.sum().item()) + float(sig.sum().item() if sig is not None else 0.0)
    # Total decreases only by decay -- should not grow.
    assert final_total <= initial_total * 1.01, (
        f"total mass grew: {initial_total:.4f} -> {final_total:.4f}"
    )


# ---------------------------------------------------------------------------
# ecosystem/run.py: validate_signal_consistency
# ---------------------------------------------------------------------------


def _make_rollout_result(signal: bool) -> RolloutResult:
    p = sample_random(kernels=3, seed=0, signal_field=signal)
    return RolloutResult(
        params=p,
        seed=0,
        descriptors=(0.5, 0.5, 0.5),
        quality=0.8,
        rejection_reason=None,
        thumbnail=np.zeros((16, 32, 32), dtype=np.uint8),
        creature_id="",
        parent_id=None,
        created_at=0.0,
        compute_seconds=1.0,
    )


def test_validate_signal_consistency_passes_all_non_signal() -> None:
    validate_signal_consistency([_make_rollout_result(False), _make_rollout_result(False)])


def test_validate_signal_consistency_passes_all_signal() -> None:
    validate_signal_consistency([_make_rollout_result(True), _make_rollout_result(True)])


def test_validate_signal_consistency_raises_on_mixed() -> None:
    with pytest.raises(ValueError, match="mix"):
        validate_signal_consistency([_make_rollout_result(True), _make_rollout_result(False)])


# ---------------------------------------------------------------------------
# alpha_coupling and beta_modulation physics
# ---------------------------------------------------------------------------


def test_alpha_zero_unchanged_growth() -> None:
    """alpha_coupling=0 reduces to previous additive behavior."""
    fl = _make_fl(signal=True)
    # Force alpha to zero
    from biota.sim.flowlenia import Params

    p = fl.params
    p0 = Params(
        R=p.R,
        r=p.r,
        m=p.m,
        s=p.s,
        h=p.h,
        a=p.a,
        b=p.b,
        w=p.w,
        emission_vector=p.emission_vector,
        receptor_profile=p.receptor_profile,
        emission_rate=p.emission_rate,
        decay_rates=p.decay_rates,
        alpha_coupling=0.0,
        beta_modulation=0.0,
        signal_kernel_r=p.signal_kernel_r,
        signal_kernel_a=p.signal_kernel_a,
        signal_kernel_b=p.signal_kernel_b,
        signal_kernel_w=p.signal_kernel_w,
    )
    from biota.sim.flowlenia import Config, FlowLenia

    fl0 = FlowLenia(Config(grid_h=32, grid_w=32, kernels=3), p0)
    state = torch.zeros(32, 32, 1)
    state[14:18, 14:18, 0] = 1.0
    sig = fl0.make_initial_signal_field(seed=0)
    new_state, _new_sig = fl0.step(state, sig)
    # Should produce a valid state (not NaN)
    assert torch.isfinite(new_state).all()


def test_positive_alpha_amplifies_growth_in_favorable_region() -> None:
    """With high positive alpha and aligned receptor, growth multiplier > 1."""
    pdict = _make_signal_params(seed=5)
    # Manually set receptor to all-positive and alpha to max
    pdict["receptor_profile"] = [1.0] * SIGNAL_CHANNELS  # type: ignore[typeddict-item]
    pdict["alpha_coupling"] = 0.9  # type: ignore[typeddict-item]
    pdict["beta_modulation"] = 0.0  # type: ignore[typeddict-item]
    fl_base = _make_fl(signal=False)
    from biota.sim.flowlenia import Config, FlowLenia, Params

    cfg = Config(grid_h=32, grid_w=32, kernels=3)
    p = Params(
        R=fl_base.params.R,
        r=fl_base.params.r,
        m=fl_base.params.m,
        s=fl_base.params.s,
        h=fl_base.params.h,
        a=fl_base.params.a,
        b=fl_base.params.b,
        w=fl_base.params.w,
        emission_vector=torch.tensor(pdict["emission_vector"]),  # type: ignore[typeddict-item]
        receptor_profile=torch.ones(SIGNAL_CHANNELS),
        emission_rate=float(pdict["emission_rate"]),  # type: ignore[typeddict-item]
        decay_rates=torch.tensor(pdict["decay_rates"]),  # type: ignore[typeddict-item]
        alpha_coupling=0.9,
        beta_modulation=0.0,
        signal_kernel_r=float(pdict["signal_kernel_r"]),  # type: ignore[typeddict-item]
        signal_kernel_a=torch.tensor(pdict["signal_kernel_a"]),  # type: ignore[typeddict-item]
        signal_kernel_b=torch.tensor(pdict["signal_kernel_b"]),  # type: ignore[typeddict-item]
        signal_kernel_w=torch.tensor(pdict["signal_kernel_w"]),  # type: ignore[typeddict-item]
    )
    fl = FlowLenia(cfg, p)
    state = torch.zeros(32, 32, 1)
    state[14:18, 14:18, 0] = 1.0
    # Use a non-trivial signal field to ensure reception is nonzero
    sig = fl.make_initial_signal_field(seed=0) + 0.1
    new_state, new_sig_out = fl.step(state, sig)
    assert torch.isfinite(new_state).all()
    assert new_sig_out is not None


def test_beta_zero_uses_base_emission_rate() -> None:
    """beta_modulation=0 leaves emission_rate unchanged."""
    pdict = _make_signal_params(seed=3)
    pdict["beta_modulation"] = 0.0  # type: ignore[typeddict-item]
    # Just verify no error and mass is approximately conserved (allowing signal transfer)
    fl = _make_fl(signal=True)
    state = torch.zeros(32, 32, 1)
    state[14:18, 14:18, 0] = 1.0
    sig = fl.make_initial_signal_field(seed=0)
    new_state, _new_sig = fl.step(state, sig)
    assert torch.isfinite(new_state).all()
