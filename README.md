# biota

<p align="center">
  <img src="docs/logo-512.svg" width="96" alt="biota logo" />
</p>

**biota** is a distributed quality-diversity research platform for Flow-Lenia cellular automata. It runs MAP-Elites searches across a Ray cluster — dispatching batches of Flow-Lenia simulations as vectorized PyTorch forward passes to stateless GPU workers — and produces a structured behavioral archive of distinct artificial life-forms. The platform is designed as a full experimental loop: configure behavioral descriptors, search the parameter space, explore the resulting archive, then seed ecosystem simulations from selected creatures.

<p align="center">
  <a href="https://youtu.be/ZFrRKZXiH2Q">
    <img src="docs/demo.gif" width="900">
  </a>
</p>

<p align="center">
  <small><i>Click the GIF to watch the full archive demo video &nbsp;·&nbsp; Live atlas at <a href="https://biota-atlas.pages.dev">biota-atlas.pages.dev</a></i></small>
</p>

## What it does

[Flow-Lenia](https://arxiv.org/abs/2212.07906) is a continuous cellular automaton where matter is conserved by construction. Mass conservation prevents the explode/collapse failure modes that dominate vanilla Lenia's parameter space, producing stable, self-maintaining creatures — solitons — across a much wider range of parameters.

[MAP-Elites](https://arxiv.org/abs/1504.04909) searches that parameter space for diversity rather than a single optimum. Instead of finding the best creature, it fills a behavioral grid where each cell holds the highest-quality creature with a particular phenotypic fingerprint. The result is an atlas: a structured catalog of qualitatively distinct life-forms covering the behavioral space as broadly as possible.

A 500-rollout standard-preset search on a 3-node RTX 5060 Ti cluster takes **97 seconds** at **91-92% GPU utilization** — a 3.5x speedup over the pre-batching architecture — and produces around 229 distinct archive cells at a 45% insertion rate.

## Architecture

The driver process owns the in-memory MAP-Elites archive and the search loop. It accumulates a batch of B candidate parameter sets, fires a single Ray task, and awaits B results. Each Ray task runs B Flow-Lenia simulations as one vectorized PyTorch forward pass over a `(B, H, W)` state tensor — one forward pass fills a GPU rather than one small kernel call per rollout. Workers are stateless: they receive parameters, simulate, score, and return. Nothing persistent lives on the cluster between tasks.

```
driver (archive + loop)
    └── submit batch of B params
            ├── Ray worker 0: (B, H, W) → B results   [GPU 0]
            ├── Ray worker 1: (B, H, W) → B results   [GPU 1]
            └── Ray worker 2: (B, H, W) → B results   [GPU 2]
    └── insert results → update archive → next batch
```

`--workers N` controls how many batches are in flight simultaneously. `--workers 1` is synchronous MAP-Elites (each batch sees a fully updated archive before the next is generated). Higher values trade archive freshness for throughput on multi-node setups. Effective concurrent rollouts = `--workers` × `--batch-size`.

![search loop](docs/search-loop.svg)

## Behavioral descriptors

The archive grid has three axes, each a scalar measured empirically from the rollout. Nine built-in descriptors:

| Descriptor | What it captures |
|---|---|
| `velocity` | Mean COM displacement per step over the trailing 50 steps |
| `gyradius` | Mass-weighted RMS distance from the center of mass |
| `spectral_entropy` | Shannon entropy of the radially-averaged FFT spectrum of the final state |
| `oscillation` | Variance of bounding-box fraction over the trace tail (pulsing vs rigid) |
| `compactness` | Mass inside bounding box / total mass at the final step |
| `mass_asymmetry` | Directional bias of motion — straight movers vs orbiters |
| `png_compressibility` | PNG compressed/uncompressed ratio of final state (smooth vs complex) |
| `rotational_symmetry` | Angular variance of radial mass profile (rings vs asymmetric shapes) |
| `persistence_score` | Max descriptor drift across the trace tail (stable vs changing) |

Choose any three with `--descriptors`. With 9 built-ins there are C(9,3) = 84 possible archive configurations. Supply your own via `--descriptor-module`. The archive viewer renders all three axes — two as the spatial grid, the third as an interactive slice slider.

## Quickstart

```bash
git clone https://github.com/rkv0id/biota
cd biota
uv sync
uv run biota search --preset dev --budget 50
```

This runs 50 rollouts synchronously on CPU. When it finishes, build the viewer:

```bash
uv run python scripts/build_index.py
open runs/index.html
```

The output is a self-contained HTML file per run — every creature rendered as an animated magma-colorized thumbnail, with hover tooltips, lineage highlighting, and a click-through modal with full parameters. No server required.

## Running on a cluster

```bash
# On every node
just cluster-install && source ~/.biota-runtime/bin/activate

# Head node
ray start --head --node-ip-address=<ip> --port=6379 --num-gpus=1

# Worker nodes
ray start --address=<ip>:6379 --num-gpus=1

# Run from the head node
biota search --ray-address <ip>:6379 --preset standard --budget 500 \
    --device cuda --batch-size 64 --workers 3
```

Three presets: `dev` (64×64, 200 steps), `standard` (192×192, 300 steps), `pretty` (384×384, 500 steps).

## CLI reference

`biota search` flags:

| Flag | Default | Description |
|---|---|---|
| `--preset` | `standard` | `dev`, `standard`, or `pretty` |
| `--budget` | `500` | Total rollouts |
| `--random-phase` | `200` | Uniform random rollouts before mutation |
| `--batch-size` | `1` | Rollouts per dispatch. 32-128 on cuda/mps |
| `--workers` | `1` | Concurrent batch dispatches. 1 = synchronous MAP-Elites |
| `--device` | `cpu` | `cpu`, `mps`, or `cuda` |
| `--local-ray` | off | Start a fresh local Ray instance |
| `--ray-address` | none | Attach to an existing Ray cluster |
| `--base-seed` | `0` | Reproducibility seed |
| `--checkpoint-every` | `100` | Checkpoint cadence in rollouts |
| `--descriptors` | `velocity,gyradius,spectral_entropy` | Three descriptor names for the archive axes, comma-separated |
| `--descriptor-module` | none | Path to a Python file defining custom `Descriptor` objects |

`biota doctor` checks Python, torch, device availability, Ray, and module health.

## Run output

```
runs/20260412-152312-lithe-willow/
├── manifest.json       # run metadata, biota version, preset, descriptors used
├── config.json         # exact SearchConfig serialized
├── archive.pkl         # MAP-Elites archive, rewritten on checkpoint
└── events.jsonl        # append-only log of every rollout outcome
```

## Development

```bash
just check       # ruff + pyright + pytest (136 tests)
just smoke-ray   # local-Ray integration smoke test
```

The test suite runs entirely in no-Ray mode. `just smoke-ray` exercises the Ray code path — `@ray.remote` decoration, ObjectRef serialization, `ray.wait`, batch dispatch round-trip — and should be run after any change to `ray_compat.py`.

## Roadmap

biota is designed as a full research loop over Flow-Lenia's behavioral space:

1. **Configure** behavioral descriptors (built-in library or custom)
2. **Search** the parameter space with distributed MAP-Elites
3. **Explore** the archive — filter by any descriptor axis, follow lineage, inspect parameters
4. **Seed** ecosystem simulations from selected archive creatures

- [x] v0.1.0 - Flow-Lenia PyTorch port, mass conservation verified against JAX reference
- [x] v0.2.0 - Driver, Ray runtime, search loop, multi-node GPU Ray verified
- [x] v0.3.0 - Descriptor rework, visual pipeline, static index, per-run metrics
- [x] v0.4.0 - Batched rollout engine (`--batch-size`, `--workers`), 3.5x cluster speedup
- [x] v1.0.0 - Lineage view, atlas site, public launch at [biota-atlas.pages.dev](https://biota-atlas.pages.dev)
- [x] v1.1.0 - Extended descriptor library (9 built-ins), descriptor selection CLI, per-axis archive filtering, custom descriptor API
- [ ] v2.0.0 - Ecosystem simulation - spawn selected archive creatures on a shared grid
- [ ] v2.1.0 - Heterogeneous ecosystems with parameter localization
- [ ] v3.0.0 - Learned descriptors (AURORA-style autoencoder)

## References

- Plantec et al. 2022/2025, [Flow-Lenia](https://arxiv.org/abs/2212.07906) (ALIFE 2023 best paper; [journal version](https://arxiv.org/abs/2506.08569))
- Mouret and Clune 2015, [MAP-Elites](https://arxiv.org/abs/1504.04909)
- Faldor and Cully 2024, [Leniabreeder](https://arxiv.org/abs/2406.04235)
- Michel et al. 2025, [Exploring Flow-Lenia Universes](https://arxiv.org/abs/2505.15998)
- [Reference JAX implementation](https://github.com/erwanplantec/FlowLenia)
