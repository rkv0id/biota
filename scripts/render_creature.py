"""Render a Flow-Lenia creature for visual sanity check.

Uses the same parameters and initial state as the M0 test fixtures, runs the
PyTorch port for a few hundred steps, and saves a handful of PNGs plus an
animated GIF to ./creature_render/. Useful for confirming the port produces
visually plausible creatures, not just numerically correct ones.

Run from the repo root:

    uv run python scripts/render_creature.py

Then open ./creature_render/creature.gif in your image viewer.
"""

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from biota.sim.flowlenia import Config, FlowLenia, Params

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "tests" / "sim" / "fixtures" / "jax_reference"
OUTPUT_DIR = ROOT / "creature_render"

STEPS = 300
CAPTURE_EVERY = 5  # frame every N steps -> 60 frames for the GIF
GIF_FRAME_DURATION_MS = 80  # 12.5 fps


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(FIXTURE_DIR / "params.json") as f:
        raw = json.load(f)

    params = Params(
        R=float(raw["R"]),
        r=torch.tensor(raw["r"], dtype=torch.float32),
        m=torch.tensor(raw["m"], dtype=torch.float32),
        s=torch.tensor(raw["s_growth"], dtype=torch.float32),
        h=torch.tensor(raw["h"], dtype=torch.float32),
        a=torch.tensor(raw["a"], dtype=torch.float32),
        b=torch.tensor(raw["b"], dtype=torch.float32),
        w=torch.tensor(raw["w"], dtype=torch.float32),
    )

    config = Config(grid=96, kernels=10, dd=5, dt=0.2, sigma=0.65, border="wall")
    fl = FlowLenia(config, params, device="cpu")

    A = torch.tensor(
        np.load(FIXTURE_DIR / "initial_state.npy"),
        dtype=torch.float32,
    )

    # Find the global max across all frames for stable normalization. Run once
    # to scout, then again to capture - cheaper than buffering full-precision
    # frames if we wanted to scale to bigger grids later.
    print(f"scouting {STEPS} steps for normalization range...")
    max_val = float(A.max())
    A_scout = A.clone()
    for _ in range(STEPS):
        A_scout = fl.step(A_scout)
        max_val = max(max_val, float(A_scout.max()))

    print(f"  global max: {max_val:.4f}")
    print(f"capturing {STEPS} steps, frame every {CAPTURE_EVERY}...")

    frames: list[np.ndarray] = []
    A_run = torch.tensor(
        np.load(FIXTURE_DIR / "initial_state.npy"),
        dtype=torch.float32,
    )

    for step in range(STEPS + 1):
        if step % CAPTURE_EVERY == 0:
            arr = A_run[:, :, 0].cpu().numpy()
            arr_u8 = np.clip(arr / max_val * 255.0, 0.0, 255.0).astype(np.uint8)
            frames.append(arr_u8)
        if step < STEPS:
            A_run = fl.step(A_run)

    print(f"  captured {len(frames)} frames")

    # Four checkpoint PNGs
    checkpoints = {
        "initial": 0,
        "early": len(frames) // 4,
        "mid": len(frames) // 2,
        "final": len(frames) - 1,
    }
    for label, idx in checkpoints.items():
        path = OUTPUT_DIR / f"creature_{label}.png"
        iio.imwrite(path, frames[idx])
        print(f"  wrote {path}")

    # Animated GIF of the full trajectory
    gif_path = OUTPUT_DIR / "creature.gif"
    iio.imwrite(
        gif_path,
        np.stack(frames),
        duration=GIF_FRAME_DURATION_MS,
        loop=0,
    )
    print(f"  wrote {gif_path}")

    print(f"\nOpen {gif_path} in your image viewer to watch the creature evolve.")
    print(
        "The creature is the one from PARAM_SEED=1234, INIT_SEED=5678 in the M0 "
        "test fixtures - it's a slowly-spreading pattern, not the most dynamic "
        "creature in Flow-Lenia space, but it confirms the port produces real "
        "Flow-Lenia output."
    )


if __name__ == "__main__":
    main()
