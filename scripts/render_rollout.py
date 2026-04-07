"""Render a rollout via the full search-layer pipeline.

Loads the M0 fixture parameters, runs them through biota.search.rollout
(ParamDict -> Params -> sim -> trace -> evaluate -> RolloutResult), and
saves the thumbnail GIF plus individual frames to ./rollout_render/.

Different from scripts/render_creature.py: that script calls biota.sim
directly to verify the sim layer in isolation. This script verifies the
search-layer integration: ParamDict conversion, trace capture, quality
evaluation, and thumbnail downsampling. When step 6 of M1 has a bug, run
both scripts to see whether the breakage is in the sim or the search layer.

Run from the repo root:

    uv run python scripts/render_rollout.py

Then open ./rollout_render/thumbnail.gif in your image viewer.
"""

import json
from pathlib import Path

import imageio.v3 as iio

from biota.search.result import ParamDict
from biota.search.rollout import dev_preset, rollout

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "tests" / "sim" / "fixtures" / "jax_reference"
OUTPUT_DIR = ROOT / "rollout_render"

GIF_FRAME_DURATION_MS = 200  # 5 fps - 16 frames so the GIF lasts ~3 seconds


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(FIXTURE_DIR / "params.json") as f:
        raw = json.load(f)
    params = ParamDict(
        R=float(raw["R"]),
        r=list(raw["r"]),
        m=list(raw["m"]),
        s=list(raw["s_growth"]),
        h=list(raw["h"]),
        a=list(raw["a"]),
        b=list(raw["b"]),
        w=list(raw["w"]),
    )

    config = dev_preset()
    print(f"running rollout: grid={config.sim.grid}, steps={config.steps}")
    result = rollout(params, seed=5678, config=config, device="cpu")

    print()
    print("=== rollout result ===")
    print(f"accepted:         {result.accepted}")
    print(f"quality:          {result.quality}")
    print(f"rejection_reason: {result.rejection_reason}")
    print(f"compute_seconds:  {result.compute_seconds:.2f}")
    if result.descriptors is not None:
        speed, size, structure = result.descriptors
        print(f"descriptors:      ({speed:.4f}, {size:.4f}, {structure:.4f})")
        print(f"  speed:          {speed:.4f}")
        print(f"  size:           {size:.4f}")
        print(f"  structure:      {structure:.4f}")
    else:
        print("descriptors:      None (rollout was dead)")
    print(
        f"thumbnail:        shape={result.thumbnail.shape} "
        f"dtype={result.thumbnail.dtype} "
        f"range=[{result.thumbnail.min()}, {result.thumbnail.max()}]"
    )
    print()

    # Save the full thumbnail as a GIF
    gif_path = OUTPUT_DIR / "thumbnail.gif"
    iio.imwrite(gif_path, result.thumbnail, duration=GIF_FRAME_DURATION_MS, loop=0)
    print(f"wrote {gif_path}")

    # Save four checkpoint PNGs from the thumbnail frames
    n_frames = result.thumbnail.shape[0]
    checkpoints = {
        "first": 0,
        "early": n_frames // 4,
        "mid": n_frames // 2,
        "last": n_frames - 1,
    }
    for label, idx in checkpoints.items():
        path = OUTPUT_DIR / f"thumbnail_{label}.png"
        iio.imwrite(path, result.thumbnail[idx])
        print(f"wrote {path}")

    print()
    print(f"Open {gif_path} in your image viewer to watch the rollout.")
    print(
        "The thumbnail is 32x32 by design - that's what gets stored in each "
        "archive cell. The dashboard will display it at a larger size with CSS. "
        "Pixelation in the standalone GIF is expected and not a bug."
    )


if __name__ == "__main__":
    main()
