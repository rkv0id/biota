# Changelog

## v2.2.0 - 2026-04-15

Heterogeneous ecosystem runs and a config-file-driven CLI. Multiple creatures from different archive cells share a grid; each species keeps its own complete parameter set. The ecosystem command is fully rewritten around YAML configs.

### New features

**Heterogeneous ecosystem runs**
- New `biota.sim.localized` module: `LocalizedFlowLenia` runs S species on a shared grid, each with its own kernel tensor and growth-window vector. Per-cell species ownership is tracked as a simplex weight that advects with mass. Growth fields blend by ownership.
- `LocalizedState` dataclass holds (mass, weights). Mass is `(H, W, 1)` float32; weights are `(H, W, S)` float32, simplex where mass is present, zero elsewhere.
- `build_initial_state_multi_species` in `biota.ecosystem.spawn` places per-species patches with one-hot weight initialization.
- Single-species `LocalizedFlowLenia` reduces exactly to scalar `FlowLenia.step` within float roundoff (verified by tests at 1e-6 single-step, 1e-5 over 20 steps). Empty cells use uniform `1/S` fallback weights for the growth blend so the reduction holds.

**Config-file CLI**
- `biota ecosystem --config experiments.yaml` replaces all the inline flags. Experiment-level parameters (grid, steps, sources, spawn) live in the YAML; CLI carries only infrastructure (`--archive-dir`, `--output-dir`, `--device`, `--local-ray`, `--ray-address`).
- One YAML file can declare multiple experiments; they run sequentially in v2.2.0.
- Every required field must be set explicitly. No silent defaults. Errors reference the offending experiment by name.
- `--local-ray` and `--ray-address` are accepted but not yet wired; they print a notice and the loop runs sequentially. Parallel dispatch lands in v2.3.0.

**Viz updates**
- Mode badge on ecosystem run pages and atlas cards: gray `homogeneous`, purple `heterogeneous`.
- Heterogeneous run pages list every source with its own archive deep link, cell deep link, and creature count. The header's meta line shows "N species" alongside total creature count.
- Index card grid surfaces source count for heterogeneous runs and shows a "het" pill instead of repeating mode in text.
- Empty-state copy on the Ecosystem tab rewritten to describe both modes as available now and what each one does.

### Breaking changes

**`biota ecosystem` CLI completely replaced.** The flag set from v2.0.x and v2.1.0 is gone. Existing scripts that called `biota ecosystem --run ... --cell ... --n ...` will fail. Migrate by translating each invocation into a single-experiment YAML config; the field names map directly. Pre-public, no migration tooling provided.

**`EcosystemConfig` dataclass shape.** New fields: `name` (required, string), `sources` (tuple of `CreatureSource`), `is_heterogeneous` property. Removed: `source_run_id`, `source_coords` (now per-source on `CreatureSource`). `SpawnConfig` no longer carries `n` (now per-source on `CreatureSource`).

**`summary.json` shape.** Adds `mode` (`"homogeneous"` or `"heterogeneous"`), `name`, `sources` list (each entry has `archive_dir`, `run`, `cell`, `n`). Removes top-level `source_run_id`, `source_coords`, and `spawn.n`. `build_index.py` reads both shapes; new ecosystem runs use the new shape, legacy v2.0.x runs still render.

**`run_ecosystem` signature.** Now takes `(config, output_root)`; the creature is loaded from disk via `config.sources` rather than passed in as a `RolloutResult`. Test code that constructed a creature in memory and handed it to `run_ecosystem` must now write a real archive pickle to disk first.

**`build_initial_state` and `compute_spawn_positions` signatures.** Both now take `n` as an explicit positional argument, since `SpawnConfig.n` is gone.

### Implementation notes

- The scalar `FlowLenia.step` and `step_batch` paths are untouched. The JAX reference match tolerances stay the same. Search rollouts are unaffected.
- Heterogeneous runs use S FFT passes per step (one per species). At S=2 to S=4 with the kernel counts the search produces, this is fine on a single GPU.
- All sources in a heterogeneous experiment must have the same kernel count. The runner validates this and raises a clear `ValueError` if not.
- Run id format changed from `<timestamp>-eco-<source-suffix>-<coords>` to `<timestamp>-<sanitized-experiment-name>`. Output dir naming is more legible at the cost of being less self-describing about provenance; the full source list is in `summary.json`.
- New dependencies: `pyyaml>=6.0` (runtime), `types-pyyaml>=6.0` (dev).

### Test count

273 passing, 1 skipped (was 230 + 1 in v2.1.0). 43 new tests across YAML parser (21), multi-species spawn (4), localized step (9), and heterogeneous run integration (2), plus reshape of existing ecosystem and CLI tests for the new dataclass surface.

---

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
