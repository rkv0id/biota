# Changelog

## v2.1.0 - 2026-04-15

### New features

**6 new behavioral descriptors** (library grows from 9 to 15, C(15,3) = 455 archive configurations):

- `displacement_ratio` - total COM displacement / total path length over the trace tail. 0 = pure orbiter, 1 = straight-line glider. Separates true translators from fast-moving orbiters that velocity alone cannot distinguish. Grounded in Chan 2019's linear speed concept, normalized by path length.
- `angular_velocity` - mean absolute angular speed of COM direction changes over the trace tail. Separates rotors and orbiters from translators. Used by Plantec et al. as an optimization target; here adapted as a MAP-Elites descriptor.
- `growth_gradient` - mass-weighted mean spatial gradient magnitude at the final step. Approximates Chan's growth-centroid distance (dgm). Low = smooth internally consistent creature, high = labyrinthine channels and sharp internal structure.
- `morphological_instability` - variance of gyradius over the trace tail. Low = rigid stable form, high = creature that constantly reshapes or pulses. Directly predicts ecosystem contact behavior.
- `activity` - mean absolute gyradius change per step. Per-creature adaptation of evolutionary activity (Michel et al. 2025). Measures internal work rate: static or rigidly translating creatures score near 0, pulsing or morphing creatures score high.
- `spatial_entropy` - Shannon entropy of mass over a coarse 8x8 spatial grid. Adapted from Michel et al. 2025's multi-scale matter distribution metric. Low = compact localized mass, high = diffuse or multi-body spread.

**`gyradius_history` added to `RolloutTrace`** - tracked per step cheaply alongside COM in the rollout loop (both single and batch). Required by `morphological_instability` and `activity`.

**`_step_stats` and `_step_stats_batch`** now return a 4-tuple including gyradius (was 3-tuple).

---

## v2.0.1 - 2026-04-14

### Bug fixes

**Sobel circular padding for torus border** (`flowlenia.py`): Sobel gradients were using zero-padding regardless of border mode. For torus border this caused incorrect flow forces at grid seams - the 3x3 kernel saw zeros on the "outside" instead of the opposite edge's values. This produced a bright strip artifact at wrap boundaries in ecosystem GIFs. Now uses `F.pad(..., mode='circular')` + `padding=0` for torus, retains zero-padding for wall to preserve JAX reference match.

**Global GIF colorization normalization** (`ecosystem/run.py`): Per-frame normalization let any locally bright pixel anchor the colormap for that entire frame, amplifying seam artifacts and causing flickering as the normalization scale shifted between frames. Now computes a single global 99.9th percentile across the full trajectory and applies it uniformly to all frames.

---

## v2.0.0 - 2026-04-14

Ecosystem simulation. Spawn archived creatures onto a shared grid and observe what happens.

### New features

**Ecosystem simulation**
- `biota ecosystem` command runs homogeneous ecosystem simulations from a single archive cell
- Poisson disk spawning places N creatures on the grid with guaranteed minimum separation
- Torus and wall border support inherited from the underlying Flow-Lenia sim
- GIF output (default) or individual PNG frames via `--output-format`
- GIF downsampled to 256px on longest side during index build, full resolution preserved on disk
- `trajectory.npy` saves raw float32 state snapshots for downstream analysis

**Rectangular grids**
- `--grid 512` for square, `--grid 192x512` for rectangular (HxW)
- Flow-Lenia kernel FFT, position grid, and reintegration all support arbitrary H x W
- Per-axis wall clamp and torus wrap in both single and batch step
- Landscape ecosystem GIFs (e.g. `192x512`) display correctly in the atlas viewer

**Ecosystem viewer**
- `ecosystem.html` per-run page with mass-over-time chart, animated GIF, run metadata
- Cell coordinates link directly to the source archive cell modal via `view.html#cell-y-x-z`
- Archive viewer auto-opens the linked cell on page load via hash routing
- Ecosystem runs tab on the atlas index shows run cards when runs exist

**CLI hardening**
- `--n`, `--grid`, `--steps`, `--snapshot-every`, `--patch`, `--min-dist` are now required (no silent defaults)
- `biota doctor` now checks `biota.ecosystem` module health (3 modules)

### Changes

- `FlowLenia.Config`: `grid: int` replaced by `grid_h: int` and `grid_w: int`. The `.grid` property is kept for square grids and raises for rectangular ones
- `build_index.py`: `--output-dir` replaces `--runs-root`. `--ecosystem-dir` flag (default `ecosystem-runs/`). Incremental build skips up-to-date views
- Archive and ecosystem runs now live in `archive-runs/` and `ecosystem-runs/` by default (was `runs/`)
- Mass chart renders a flat line for conserved-mass simulations rather than an empty SVG
- Snapshot count in ecosystem viewer reads from `summary.json` rather than counting frame files (fixes 0 snapshots display in GIF mode)
- Ecosystem page and archive page share the same centered max-width layout

### Bug fixes

- `DEFAULT_ECO_ROOT` and ecosystem functions moved before `main()` in `build_index.py` (NameError on import)
- Ecosystem run IDs include milliseconds to prevent collisions during rapid test execution

---

## v1.1.0 - 2026-04-13

- 9 built-in behavioral descriptors
- `--descriptors` CLI flag for per-run descriptor selection
- Per-axis filtering sliders in the archive viewer
- Custom descriptor API via `--descriptor-module`
- Responsive atlas layout, incremental index build, logos and social preview

## v1.0.0 - 2026-04-12

- Public launch at [biota-atlas.pages.dev](https://biota-atlas.pages.dev)
- Lineage view: parent/child highlighting in the archive modal
- Static atlas site with per-run archive viewers

## v0.4.0

- Batched rollout engine: vectorized `(B, H, W)` forward pass
- 3.5x cluster throughput improvement

## v0.3.0

- Descriptor rework and visual pipeline
- Static index with per-run metrics charts

## v0.2.0

- Driver, Ray runtime, search loop
- Multi-node GPU cluster verified

## v0.1.0

- Flow-Lenia PyTorch port
- Mass conservation verified against JAX reference implementation
