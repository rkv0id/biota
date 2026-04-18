"""Tests for the species-indexed FlowLenia step.

The load-bearing tests are:

1. With a single species, LocalizedFlowLenia produces the same trajectory
   as scalar FlowLenia (within float roundoff). This proves the localized
   math reduces to the scalar math when S=1, which is the only invariant
   that lets us trust the new code path on real heterogeneous configs.

2. Mass conservation holds at the same float-roundoff bound as scalar.

3. Species weights stay a simplex: non-negative everywhere, sum to ~1
   where mass exists, zero where mass is zero.

4. Reproducibility: same inputs produce same outputs.
"""

import numpy as np
import pytest
import torch

from biota.sim.flowlenia import Config, FlowLenia, Params
from biota.sim.localized import LocalizedFlowLenia, LocalizedState


def _make_params(seed: int, k: int = 5) -> Params:
    """Build a deterministic Params for a small test sim."""
    rng = np.random.default_rng(seed)
    return Params(
        R=float(rng.uniform(5.0, 20.0)),
        r=torch.tensor(rng.uniform(0.3, 1.0, size=k).astype(np.float32)),
        m=torch.tensor(rng.uniform(0.1, 0.4, size=k).astype(np.float32)),
        s=torch.tensor(rng.uniform(0.01, 0.1, size=k).astype(np.float32)),
        h=torch.tensor(rng.uniform(0.1, 0.9, size=k).astype(np.float32)),
        a=torch.tensor(rng.uniform(0.1, 0.9, size=(k, 3)).astype(np.float32)),
        b=torch.tensor(rng.uniform(0.1, 0.9, size=(k, 3)).astype(np.float32)),
        w=torch.tensor(rng.uniform(0.05, 0.3, size=(k, 3)).astype(np.float32)),
    )


def _random_init_state(grid: int = 64, patch: int = 16, seed: int = 0) -> torch.Tensor:
    """Random patch in the centre on an otherwise-empty grid."""
    rng = np.random.default_rng(seed)
    state = torch.zeros((grid, grid, 1), dtype=torch.float32)
    half = patch // 2
    cy, cx = grid // 2, grid // 2
    state[cy - half : cy + half, cx - half : cx + half, 0] = torch.from_numpy(
        rng.random((patch, patch), dtype=np.float32)
    )
    return state


# === single-species equivalence: localized must reduce to scalar ===


def test_single_species_matches_scalar_one_step() -> None:
    """LocalizedFlowLenia with S=1, all-ones weight, must equal FlowLenia.step."""
    cfg = Config(grid_h=48, grid_w=48, kernels=5, border="wall")
    params = _make_params(seed=42, k=5)

    fl = FlowLenia(cfg, params, device="cpu")
    lfl = LocalizedFlowLenia(cfg, [params], device="cpu")

    A0 = _random_init_state(grid=48, patch=12, seed=1)
    W0 = torch.ones((48, 48, 1), dtype=torch.float32)
    # Weights only meaningful where mass exists; zero them elsewhere to match
    # the simplex-where-mass invariant the localized path produces.
    W0 = torch.where(A0 > 0, W0, torch.zeros_like(W0))

    scalar_next, _ = fl.step(A0, None)
    loc_next = lfl.step(LocalizedState(mass=A0, weights=W0))

    diff = (scalar_next - loc_next.mass).abs().max().item()
    assert diff < 1e-6, f"single-species mass mismatch: max abs diff {diff:.2e}"


def test_single_species_matches_scalar_many_steps() -> None:
    """Equivalence must hold over multiple steps, not just one.

    Tighter tolerance than the JAX reference match because we are comparing
    two PyTorch paths that share kernel build code; the only divergence
    source is the final weight normalize op which is a no-op when S=1.
    """
    cfg = Config(grid_h=48, grid_w=48, kernels=5, border="torus")
    params = _make_params(seed=7, k=5)

    fl = FlowLenia(cfg, params, device="cpu")
    lfl = LocalizedFlowLenia(cfg, [params], device="cpu")

    A0 = _random_init_state(grid=48, patch=12, seed=2)
    W0 = (A0 > 0).to(torch.float32).expand(-1, -1, 1).clone()

    A_scalar = A0
    state_loc = LocalizedState(mass=A0, weights=W0)
    for _ in range(20):
        A_scalar, _ = fl.step(A_scalar, None)
        state_loc = lfl.step(state_loc)

    diff = (A_scalar - state_loc.mass).abs().max().item()
    assert diff < 1e-5, f"after 20 steps, max abs diff {diff:.2e}"


# === mass conservation ===


def test_mass_conservation_single_species() -> None:
    cfg = Config(grid_h=48, grid_w=48, kernels=5, border="wall")
    lfl = LocalizedFlowLenia(cfg, [_make_params(seed=9, k=5)], device="cpu")

    A0 = _random_init_state(grid=48, patch=12, seed=3)
    W0 = (A0 > 0).to(torch.float32).expand(-1, -1, 1).clone()

    initial = float(A0.sum())
    state = LocalizedState(mass=A0, weights=W0)
    for _ in range(50):
        state = lfl.step(state)
    final = float(state.mass.sum())

    rel_err = abs(final - initial) / initial
    assert rel_err < 1e-3, f"mass conservation failed: rel error {rel_err:.2e}"


def test_mass_conservation_two_species_torus() -> None:
    """With two distinct species and torus border, mass still conserves."""
    cfg = Config(grid_h=48, grid_w=48, kernels=5, border="torus")
    p1, p2 = _make_params(seed=10, k=5), _make_params(seed=11, k=5)
    lfl = LocalizedFlowLenia(cfg, [p1, p2], device="cpu")

    # Two patches: species 0 on the left, species 1 on the right.
    A = torch.zeros((48, 48, 1), dtype=torch.float32)
    rng = np.random.default_rng(4)
    A[16:32, 6:18, 0] = torch.from_numpy(rng.random((16, 12), dtype=np.float32))
    A[16:32, 30:42, 0] = torch.from_numpy(rng.random((16, 12), dtype=np.float32))

    W = torch.zeros((48, 48, 2), dtype=torch.float32)
    W[16:32, 6:18, 0] = 1.0
    W[16:32, 30:42, 1] = 1.0

    initial = float(A.sum())
    state = LocalizedState(mass=A, weights=W)
    for _ in range(30):
        state = lfl.step(state)
    final = float(state.mass.sum())

    rel_err = abs(final - initial) / initial
    assert rel_err < 1e-3, f"two-species mass conservation failed: rel error {rel_err:.2e}"


# === simplex invariant on weights ===


def test_weights_remain_simplex_two_species() -> None:
    """Weights must be non-negative and sum to 1 wherever mass > 0."""
    cfg = Config(grid_h=48, grid_w=48, kernels=5, border="torus")
    lfl = LocalizedFlowLenia(
        cfg, [_make_params(seed=20, k=5), _make_params(seed=21, k=5)], device="cpu"
    )

    A = torch.zeros((48, 48, 1), dtype=torch.float32)
    rng = np.random.default_rng(5)
    A[10:30, 8:24, 0] = torch.from_numpy(rng.random((20, 16), dtype=np.float32))
    A[15:35, 26:42, 0] = torch.from_numpy(rng.random((20, 16), dtype=np.float32))

    W = torch.zeros((48, 48, 2), dtype=torch.float32)
    W[10:30, 8:24, 0] = 1.0
    W[15:35, 26:42, 1] = 1.0

    state = LocalizedState(mass=A, weights=W)
    for _ in range(25):
        state = lfl.step(state)

    weights = state.weights
    # Non-negative everywhere
    assert (weights >= 0).all().item(), "weights went negative"
    # Where mass exists (above a stricter threshold so the cells we test
    # actually carry meaningful weight), weights must sum to ~1.
    mass_present = state.mass[:, :, 0] > 1e-3
    weight_sums = weights.sum(dim=-1)
    if mass_present.any():
        sums_where_mass = weight_sums[mass_present]
        max_dev = (sums_where_mass - 1.0).abs().max().item()
        assert max_dev < 1e-4, (
            f"weight simplex violated where mass > 0: max |sum - 1| = {max_dev:.2e}"
        )
    # Where mass is essentially zero, weights must also be essentially zero.
    truly_empty = state.mass[:, :, 0] < 1e-8
    if truly_empty.any():
        max_w_empty = weights[truly_empty].abs().max().item()
        assert max_w_empty < 1e-6, f"weights nonzero in empty cells: max abs = {max_w_empty:.2e}"


def test_weights_pure_species_persists_when_isolated() -> None:
    """A single species patch in isolation should keep weight=1 there."""
    cfg = Config(grid_h=48, grid_w=48, kernels=5, border="wall")
    lfl = LocalizedFlowLenia(
        cfg, [_make_params(seed=30, k=5), _make_params(seed=31, k=5)], device="cpu"
    )

    A = torch.zeros((48, 48, 1), dtype=torch.float32)
    rng = np.random.default_rng(6)
    A[18:30, 18:30, 0] = torch.from_numpy(rng.random((12, 12), dtype=np.float32))
    # Only species 0 present anywhere; species 1 is absent from the entire grid.
    W = torch.zeros((48, 48, 2), dtype=torch.float32)
    W[A[:, :, 0] > 0, 0] = 1.0

    state = LocalizedState(mass=A, weights=W)
    for _ in range(15):
        state = lfl.step(state)

    # No species 1 mass should have appeared from nothing.
    species1_total = state.weights[:, :, 1].sum().item()
    assert species1_total < 1e-5, (
        f"species 1 mass appeared from nothing: total weight = {species1_total:.3e}"
    )


# === reproducibility ===


def test_step_is_deterministic() -> None:
    cfg = Config(grid_h=32, grid_w=32, kernels=5, border="wall")
    p1 = _make_params(seed=40, k=5)
    p2 = _make_params(seed=41, k=5)

    lfl1 = LocalizedFlowLenia(cfg, [p1, p2], device="cpu")
    lfl2 = LocalizedFlowLenia(cfg, [p1, p2], device="cpu")

    A = _random_init_state(grid=32, patch=10, seed=7)
    W = torch.zeros((32, 32, 2), dtype=torch.float32)
    W[A[:, :, 0] > 0, 0] = 0.6
    W[A[:, :, 0] > 0, 1] = 0.4

    s1 = LocalizedState(mass=A.clone(), weights=W.clone())
    s2 = LocalizedState(mass=A.clone(), weights=W.clone())
    for _ in range(10):
        s1 = lfl1.step(s1)
        s2 = lfl2.step(s2)
    assert torch.equal(s1.mass, s2.mass)
    assert torch.equal(s1.weights, s2.weights)


# === construction errors ===


def test_empty_species_list_rejected() -> None:
    cfg = Config(grid_h=32, grid_w=32, kernels=5, border="wall")
    with pytest.raises(ValueError, match="at least one"):
        LocalizedFlowLenia(cfg, [], device="cpu")


def test_weights_shape_mismatch_rejected() -> None:
    cfg = Config(grid_h=32, grid_w=32, kernels=5, border="wall")
    lfl = LocalizedFlowLenia(
        cfg, [_make_params(seed=50, k=5), _make_params(seed=51, k=5)], device="cpu"
    )
    A = torch.zeros((32, 32, 1), dtype=torch.float32)
    # Wrong species dimension (3 instead of 2)
    W_wrong = torch.zeros((32, 32, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="species count"):
        lfl.step(LocalizedState(mass=A, weights=W_wrong))
