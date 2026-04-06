"""Generate JAX reference fixtures for biota's M0 mass conservation test.

Runs the reference Flow-Lenia implementation (erwanplantec/FlowLenia) on a fixed
seed and saves:

  - initial_state.npy   : A at step 0
  - state_step_100.npy  : A at step 100
  - state_step_500.npy  : A at step 500
  - state_step_1000.npy : A at step 1000
  - mass_per_step.npy   : total mass at every step (length 1001)
  - metadata.json       : config, parameters, seeds, mass conservation summary

Single channel, 96x96 grid, dd=5, dt=0.2, sigma=0.65, k=10. Matches biota's dev
preset. Output goes into ./fixtures/jax_reference/.

Run with:  python generate_jax_reference.py
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from flowlenia.flowlenia import Config, FlowLenia

# Force CPU so the fixtures are deterministic across machines
jax.config.update("jax_platform_name", "cpu")

OUT_DIR = Path(__file__).parent / "fixtures" / "jax_reference"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- config matching biota dev preset
GRID = 96
STEPS = 1000
PATCH = GRID // 3  # 32 cells
KERNELS = 10
PARAM_SEED = 1234
INIT_SEED = 5678

cfg = Config(
    X=GRID,
    Y=GRID,
    C=1,
    c0=[0] * KERNELS,
    c1=[list(range(KERNELS))],
    k=KERNELS,
    dd=5,
    dt=0.2,
    sigma=0.65,
    border="wall",
)

# --- build the model
fl = FlowLenia(cfg, key=jr.key(PARAM_SEED))

# --- initialize state with a centered random patch
state = fl.initialize(jr.key(0))
patch_start = (GRID - PATCH) // 2
patch_end = patch_start + PATCH
patch = jr.uniform(jr.key(INIT_SEED), (PATCH, PATCH, 1))
A0 = state.A.at[patch_start:patch_end, patch_start:patch_end, :].set(patch)
state = state._replace(A=A0)

initial_mass = float(jnp.sum(state.A))
print(f"initial mass: {initial_mass:.10f}")

# --- run, recording mass at every step
masses = [initial_mass]
checkpoints = {}
checkpoints[0] = np.array(state.A)

# Avoid jit/scan because we want per-step mass for the conservation curve.
# Speed isn't the point here; correctness is.
for step in range(1, STEPS + 1):
    state = fl(state)
    masses.append(float(jnp.sum(state.A)))
    if step in (1, 10, 100, 500, 1000):
        checkpoints[step] = np.array(state.A)
        print(
            f"step {step:>4d}: mass = {masses[-1]:.10f}, "
            f"abs_err = {abs(masses[-1] - initial_mass):.2e}, "
            f"rel_err = {abs(masses[-1] - initial_mass) / initial_mass:.2e}"
        )

masses = np.array(masses)
abs_errors = np.abs(masses - initial_mass)
rel_errors = abs_errors / initial_mass

# --- save fixtures
np.save(OUT_DIR / "initial_state.npy", checkpoints[0])
np.save(OUT_DIR / "state_step_1.npy", checkpoints[1])
np.save(OUT_DIR / "state_step_10.npy", checkpoints[10])
np.save(OUT_DIR / "state_step_100.npy", checkpoints[100])
np.save(OUT_DIR / "state_step_500.npy", checkpoints[500])
np.save(OUT_DIR / "state_step_1000.npy", checkpoints[1000])
np.save(OUT_DIR / "mass_per_step.npy", masses)

# --- save the parameters used (so the PyTorch port can use the exact same)
params = {
    "R": float(fl.R),
    "r": np.array(fl.r).tolist(),
    "m": np.array(fl.m).tolist(),
    "s_growth": np.array(fl.s).tolist(),  # growth function sigma
    "h": np.array(fl.h).tolist(),
    "a": np.array(fl.a).tolist(),
    "b": np.array(fl.b).tolist(),
    "w": np.array(fl.w).tolist(),
}
with open(OUT_DIR / "params.json", "w") as f:
    json.dump(params, f, indent=2)

# --- summary metadata
metadata = {
    "grid": GRID,
    "steps": STEPS,
    "patch": PATCH,
    "kernels": KERNELS,
    "param_seed": PARAM_SEED,
    "init_seed": INIT_SEED,
    "config": {
        "dd": cfg.dd,
        "dt": cfg.dt,
        "sigma": cfg.sigma,
        "border": cfg.border,
    },
    "initial_mass": initial_mass,
    "final_mass": float(masses[-1]),
    "max_abs_error_over_run": float(np.max(abs_errors)),
    "max_rel_error_over_run": float(np.max(rel_errors)),
    "abs_error_at_step_1000": float(abs_errors[-1]),
    "rel_error_at_step_1000": float(rel_errors[-1]),
    "jax_version": jax.__version__,
    "platform": jax.default_backend(),
}
with open(OUT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print()
print("=" * 60)
print("MASS CONSERVATION SUMMARY")
print("=" * 60)
print(f"initial mass:           {initial_mass:.10f}")
print(f"final mass (step 1000): {masses[-1]:.10f}")
print(f"max abs error:          {metadata['max_abs_error_over_run']:.2e}")
print(f"max rel error:          {metadata['max_rel_error_over_run']:.2e}")
print(f"final abs error:        {metadata['abs_error_at_step_1000']:.2e}")
print(f"final rel error:        {metadata['rel_error_at_step_1000']:.2e}")
print()
print(f"fixtures written to: {OUT_DIR}")
