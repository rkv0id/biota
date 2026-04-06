"""Mass conservation must be preserved within tolerance over 1000 steps.

Tolerances are derived from measuring the JAX reference implementation under
identical config (96x96, dd=5, dt=0.2, sigma=0.65, k=10, border='wall'). The
JAX reference accumulates 1.26e-04 relative error over 1000 steps in float32;
see fixtures/jax_reference/metadata.json. Our bound is roughly 2x that on CPU
to leave headroom for the PyTorch port being slightly less precise. MPS gets a
much looser bound (10x) and will be measured on the laptop in M0.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from biota.sim.flowlenia import Config, FlowLenia, Params

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "jax_reference"

JAX_REFERENCE_REL_ERROR = 1.26e-4
CPU_TOLERANCE = 2.5e-4
MPS_TOLERANCE = 1.25e-3


def _load_params(device: str) -> Params:
    with open(FIXTURE_DIR / "params.json") as f:
        raw = json.load(f)
    return Params(
        R=float(raw["R"]),
        r=torch.tensor(raw["r"], dtype=torch.float32, device=device),
        m=torch.tensor(raw["m"], dtype=torch.float32, device=device),
        s=torch.tensor(raw["s_growth"], dtype=torch.float32, device=device),
        h=torch.tensor(raw["h"], dtype=torch.float32, device=device),
        a=torch.tensor(raw["a"], dtype=torch.float32, device=device),
        b=torch.tensor(raw["b"], dtype=torch.float32, device=device),
        w=torch.tensor(raw["w"], dtype=torch.float32, device=device),
    )


def _load_initial_state(device: str) -> torch.Tensor:
    arr = np.load(FIXTURE_DIR / "initial_state.npy")
    return torch.tensor(arr, dtype=torch.float32, device=device)


def _run_and_check(device: str, tolerance: float) -> None:
    config = Config(grid=96, kernels=10, dd=5, dt=0.2, sigma=0.65, border="wall")
    params = _load_params(device)
    A0 = _load_initial_state(device)

    fl = FlowLenia(config, params, device=device)
    _, masses = fl.rollout_with_mass(A0, steps=1000)

    masses_np = masses.cpu().numpy()
    initial = float(masses_np[0])
    rel_errors = np.abs(masses_np - initial) / initial
    max_rel_error = float(rel_errors.max())

    assert max_rel_error < tolerance, (
        f"max relative mass error {max_rel_error:.2e} exceeds tolerance "
        f"{tolerance:.2e} on device {device}. JAX reference floor is "
        f"{JAX_REFERENCE_REL_ERROR:.2e}."
    )


def test_mass_conservation_cpu() -> None:
    _run_and_check("cpu", CPU_TOLERANCE)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available on this machine",
)
def test_mass_conservation_mps() -> None:
    _run_and_check("mps", MPS_TOLERANCE)
