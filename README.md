# biota

Quality-diversity search and population dynamics for Flow-Lenia creatures.

## What this is

biota finds, catalogs, and studies Flow-Lenia creatures. It runs locally, on a single GPU, or across a Ray cluster, and produces an archive of structurally distinct creatures that can be explored individually or studied collectively.

The engine is MAP-Elites over Flow-Lenia parameter space. The artifact is a 3D archive organized by velocity (how fast creatures move), gyradius (how spread out their mass is), and spectral entropy (how much internal structure they have). The compute fabric is Ray, so the same code runs from a laptop up to a multi-node GPU cluster without changes to the search loop.

Flow-Lenia is a continuous cellular automaton where matter is conserved by construction. It produces life-like creatures across a much wider parameter range than vanilla Lenia because the mass-conservation invariant prevents the explode/collapse failure modes that dominate Lenia's parameter space. MAP-Elites fills a grid of behavior cells, keeping the best creature found so far in each cell, producing an atlas of qualitatively distinct solutions instead of a single winner.

## Status

**v0.2.0 (2026-04-08) — M1 complete.** Search loop works end to end in three modes: synchronous no-Ray, local Ray, and multi-node Ray cluster attach.

**M2 closing out.** Perf fixes (wheel install, GPU fractioning), descriptor rework (velocity, gyradius, spectral entropy, all calibrated against real cluster measurements), and a visual pipeline overhaul have all shipped. A 500-rollout standard-preset search on a 24-core cluster produces 228 distinct archive cells in ~340s with a 45.6% insertion rate and a visibly diverse archive. Remaining M2 work is a small static-index build step and a per-run metrics view.

See `SPEC.md` for the full design, `SUMMARY.md` for current state, `DECISIONS.md` for the history of how we got here.

## Quickstart

```bash
git clone https://github.com/rkv0id/biota
cd biota
uv sync
uv run biota doctor                                    # verify install
uv run biota search --preset dev --budget 50          # small local run, no Ray
```

This runs a 50-rollout search synchronously on CPU, producing a populated archive in `runs/<run_id>/`. Once it finishes, view it:

```bash
uv run python scripts/view_archive.py runs/<run_id>/archive.pkl
```

That generates a self-contained HTML file with every creature in the archive as an animated thumbnail and opens it in your browser.

## Running on a GPU

```bash
uv run biota search --preset dev --budget 50 --device cuda
```

The `--device` flag takes `cpu` (default), `mps` (Apple Silicon), or `cuda`. On CUDA, biota uses `torch.cuda` directly for single-rollout execution. For parallel rollouts through Ray, see below.

## Running on Ray

Three modes:

```bash
# Default: synchronous, no Ray
uv run biota search --preset dev --budget 50

# Local Ray: fresh Ray instance in the driver process
uv run biota search --preset dev --budget 50 --local-ray --num-workers 4

# Attach to an existing Ray cluster
uv run biota search --preset dev --budget 50 --ray-address 10.10.12.1:6379
```

`--local-ray` and `--ray-address` are mutually exclusive. `--ray-address` takes `host`, `host:port` (port defaults to 6379), or `ray://host:port` for Ray Client protocol. For GPU-backed Ray runs, add `--device cuda`: biota declares `@ray.remote(num_gpus=1)` on CUDA rollouts so Ray schedules onto GPU-bearing workers correctly.

### Multi-node cluster setup

For a multi-node Ray cluster with GPUs, start Ray explicitly with `--num-gpus` declared on each node before launching biota:

```bash
# On the head node
ray start --head --node-ip-address=<head-ip> --port=6379 \
  --dashboard-host=0.0.0.0 --num-gpus=<gpus-per-node>

# On each worker node
ray start --address=<head-ip>:6379 --num-gpus=<gpus-per-node>

# Verify from any node
ray status

# Run from the head (or any node with biota installed)
uv run biota search --ray-address <head-ip>:6379 --preset dev --budget 50 --device cuda
```
## CLI reference

**`biota search`** runs a MAP-Elites search and writes results to `runs/<run_id>/`. Flags:

| Flag | Default | Description |
|---|---|---|
| `--preset` | `standard` | `dev` (64x64, 200 steps), `standard` (192x192, 300), `pretty` (384x384, 500) |
| `--budget` | `500` | Total rollouts to run |
| `--random-phase` | `200` | Rollouts of uniform random sampling before mutation |
| `--max-concurrent` | `8` | Maximum in-flight rollouts |
| `--local-ray` | off | Start a fresh local Ray instance |
| `--ray-address` | none | Attach to an existing Ray cluster at `HOST[:PORT]` |
| `--num-workers` | auto | Worker count for `--local-ray` (ignored when attaching) |
| `--device` | `cpu` | `cpu`, `mps`, or `cuda` |
| `--gpus-per-rollout` | `1.0` | GPU fraction per rollout. Set to `0.25` to fit 4 rollouts per GPU |
| `--base-seed` | `0` | Seed for reproducibility |
| `--checkpoint-every` | `100` | Atomic archive checkpoint cadence |
| `--runs-root` | `runs` | Root directory for run output |
| `--grid` | preset | Override preset grid size |
| `--steps` | preset | Override preset step count |

**`biota doctor`** prints runtime info (Python, torch, device availability, Ray version, biota module health) and exits.

## What's in a run directory

```
runs/20260408-175341-quiet-junco/
├── manifest.json       # run metadata, versions, git commit
├── config.json         # the exact SearchConfig used
├── archive.pkl         # the MAP-Elites archive, rewritten periodically
└── events.jsonl        # append-only log of every insertion/rejection
```

The archive is a 3D MAP-Elites grid (32x32x16 over velocity, gyradius, spectral entropy). Each populated cell contains a creature's parameters, descriptors, quality score, parent cell, and a small set of preview frames for visualization.

## Development

```bash
just check           # ruff + format + pyright + pytest (143 tests)
just smoke-ray       # manual Ray-mode smoke test (uv run biota search --local-ray ...)
```

Run `just check` after any change to `ray_compat.py` or the search loop. `just smoke-ray` catches a class of Ray-mode bugs that pytest deliberately can't exercise.

## Architecture

Single Python process is the driver. Driver owns the in-memory archive, runs the search loop, writes checkpoints to local disk, and (in M2) will host the dashboard. Ray workers are stateless: they take parameters, run a Flow-Lenia rollout, compute descriptors and quality, and return a result. Nothing persistent lives on the workers.

The full design, including the driver-owns-state rationale, worker protocol, storage layout, and planned dashboard shape, is in `SPEC.md`.

## Roadmap

- **v0.1.0 (M0)** ✅ Flow-Lenia PyTorch port, mass conservation verified against JAX reference
- **v0.2.0 (M1)** ✅ Driver, Ray runtime, search loop, multi-node GPU Ray verified
- **v0.3.0 (M2)** Perf fixes, descriptor rework (velocity/gyradius/spectral entropy), visual pipeline overhaul, static index with per-run metrics — *closing out*
- **v1.0.0 (M3)** Lineage view + public atlas launch. Visualize archive cell ancestry trees ("do good creatures come from few ancestors or many?"). Launch `rkv0id.github.io/biota/` as the project landing page with example published runs under `rkv0id.github.io/biota/runs/<run_id>/`, plus written documentation for how to publish additional runs. **Declared v1.0.0 because this is when biota becomes a finished, polished, shareable artifact — search works well, viewer is polished, archives are explorable, the project has a public face and a clear way for others to use it.**
- **v2.0.0 (M4)** Ecosystem simulation. Standalone script that spawns selected archive creatures on a large Flow-Lenia grid and studies emergent population dynamics. Placement (grid/random/Poisson), spawning (instantaneous/sequential), parameter regimes (global/union). Turns biota from an atlas into a platform for studying Flow-Lenia populations at scale — the first thing biota does that nobody has done before. **Declared v2.0.0 because this is a major scientific jump, not incremental polish.**
- **v3.0.0 (M5)** Learned descriptors. Unsupervised autoencoder-based descriptors trained on archive phenotypes or ecosystem outcomes. Revisit only if ecosystem results suggest hand-picked descriptors are missing meaningful structure.

**Explicitly not planned:**

- Atlas view with three simultaneous projections (current 2D projection is sufficient)
- Live dashboard / websocket updates / metrics / cluster tabs (Ray already provides cluster and worker dashboards; no live-data use case for biota's own search)
- Random-vs-MAP-Elites comparison (validates what's already known from the literature)
- Biota-specific cluster benchmarks (Ray provides all relevant monitoring)
- Crash recovery (deferred indefinitely; revisit only when a biota search is long enough that a crash would cost real time)

## References

- Plantec et al. 2022, Flow-Lenia (best paper, ALIFE 2023): https://arxiv.org/abs/2212.07906
- Plantec et al. 2025, Flow-Lenia journal version: https://arxiv.org/abs/2506.08569
- Reference JAX implementation: https://github.com/erwanplantec/FlowLenia
- Mouret and Clune 2015, MAP-Elites: https://arxiv.org/abs/1504.04909
- Moroz, Reintegration tracking: https://michaelmoroz.github.io/Reintegration-Tracking/
