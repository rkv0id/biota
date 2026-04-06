"""The PyTorch port should produce numerically similar states to the JAX reference.

Tolerances loosen as steps accumulate. Step 1 should be near bit-identical to
JAX (different libraries, but same algorithm operating on the same float32 data).
By step 1000 the two implementations have diverged from compounding rounding
differences in FFT, sobel, and the reintegration sum order.

These tolerances are initial guesses. They will be tightened or loosened in M0
once the actual port is running and we can see the real divergence curve. The
useful test is "step 1 nearly matches" - if step 1 fails, the math is wrong.
"""

import json
from pathlib import Path

import numpy as np
import torch

from biota.sim.flowlenia import Config, FlowLenia, Params

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "jax_reference"


def _load_params() -> Params:
    with open(FIXTURE_DIR / "params.json") as f:
        raw = json.load(f)
    return Params(
        R=float(raw["R"]),
        r=torch.tensor(raw["r"], dtype=torch.float32),
        m=torch.tensor(raw["m"], dtype=torch.float32),
        s=torch.tensor(raw["s_growth"], dtype=torch.float32),
        h=torch.tensor(raw["h"], dtype=torch.float32),
        a=torch.tensor(raw["a"], dtype=torch.float32),
        b=torch.tensor(raw["b"], dtype=torch.float32),
        w=torch.tensor(raw["w"], dtype=torch.float32),
    )


def _build() -> tuple[FlowLenia, torch.Tensor]:
    config = Config(grid=96, kernels=10, dd=5, dt=0.2, sigma=0.65, border="wall")
    params = _load_params()
    A0 = torch.tensor(
        np.load(FIXTURE_DIR / "initial_state.npy"),
        dtype=torch.float32,
    )
    return FlowLenia(config, params, device="cpu"), A0


def _diff(actual: torch.Tensor, expected_path: Path) -> tuple[float, float]:
    expected = torch.tensor(np.load(expected_path), dtype=torch.float32)
    diff = (actual.cpu() - expected).abs()
    return float(diff.max()), float(diff.mean())


def test_match_step_1() -> None:
    fl, A0 = _build()
    A = fl.rollout(A0, steps=1)
    max_err, mean_err = _diff(A, FIXTURE_DIR / "state_step_1.npy")
    assert max_err < 1e-5, (
        f"step 1 max abs error {max_err:.2e} exceeds 1e-5 (mean {mean_err:.2e}). "
        f"This is the strictest test - if step 1 fails, the math is wrong."
    )


def test_match_step_10() -> None:
    fl, A0 = _build()
    A = fl.rollout(A0, steps=10)
    max_err, mean_err = _diff(A, FIXTURE_DIR / "state_step_10.npy")
    assert max_err < 1e-4, (
        f"step 10 max abs error {max_err:.2e} exceeds 1e-4 (mean {mean_err:.2e})."
    )


def test_match_step_100() -> None:
    fl, A0 = _build()
    A = fl.rollout(A0, steps=100)
    max_err, mean_err = _diff(A, FIXTURE_DIR / "state_step_100.npy")
    assert max_err < 1e-3, (
        f"step 100 max abs error {max_err:.2e} exceeds 1e-3 (mean {mean_err:.2e})."
    )


def test_match_step_1000() -> None:
    fl, A0 = _build()
    A = fl.rollout(A0, steps=1000)
    max_err, mean_err = _diff(A, FIXTURE_DIR / "state_step_1000.npy")
    # By 1000 steps, accumulated divergence between JAX and PyTorch is
    # expected. We just check the result isn't grossly wrong.
    assert max_err < 0.1, (
        f"step 1000 max abs error {max_err:.2e} exceeds 0.1 (mean {mean_err:.2e}). "
        f"This is the loosest test - it just catches gross failure."
    )
