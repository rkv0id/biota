# Changelog

## v3.5.0 - 2026-04-18

Chemical coupling and adaptive signal. alpha_coupling and beta_modulation added as searchable
signal parameters. Three signal-only behavioral descriptors. Descriptor library 15→18.

### alpha_coupling [-1, 1]

New Params field. Reception-to-growth coupling: G *= (1 + alpha * reception).clamp(min=0).
Applied globally (no ownership gating) -- species B grows into species A's territory when B's
receptor_profile aligns with A's emission_vector. Positive = chemotaxis (predation pathway).
Negative = chemorepulsion. Zero = no coupling (previous behavior).

### beta_modulation [-1, 1]

New Params field. Adaptive emission: rate_eff = clip(emission_rate * (1 + beta * mean(reception)), 0, 0.1).
Reception computed before emission each step. Positive = quorum sensing (amplify when stimulated).
Negative = feedback inhibition (suppress when stimulated). Zero = static emission rate.

### Signal-only descriptors

Three new Descriptor objects with signal_only=True registered in REGISTRY:
- emission_activity: mean G_pos * emission_rate over trace tail, normalized to [0,1]
- receptor_sensitivity: mean |dot(convolved_signal, receptor_profile)| over trace tail, [0,1]
- signal_retention: final_mass / initial_mass clipped to [0,1]; 1.0 for non-signal traces

Descriptor dataclass gains signal_only: bool = False field. loop.py validates at search startup:
raises ValueError if any signal_only descriptor is active without signal_field=True.
RolloutTrace gains signal_emission_history, signal_reception_history, signal_retention fields.
rollout.py captures these per-step during signal rollouts.

### Quality metric adjustment

Signal run weights: stability 0.3→0.2, retention 0.2→0.3. Non-signal unchanged (0.6/0.4).

---

## v3.4.0 - 2026-04-18

Signal physics corrections, multi-component quality metric, signal observables, signal GIF,
viewer improvements across archive and ecosystem pages.

### Signal physics

emission_rate and decay_rates made per-creature searchable (were hardcoded constants).
standard_preset 300→500 steps. signal_preset auto-selects 800 steps with --signal-field.
CREATURE_MASS_FLOOR raised 0.1→0.2. Reception moved before emission in localized.py.

### Multi-component quality metric

q = 0.6·min(compact(T/2), compact(T)) + 0.4·stability  (non-signal)
q = 0.5·min(compact(T/2), compact(T)) + 0.2·stability + 0.3·retention  (signal)
midpoint_state captured in rollout loop and stored on RolloutTrace.
stability = clip(1 - drift/0.2, 0, 1) -- continuous version of persistent filter.

### Signal observables

SimOutput captures signal_total_history, signal_channel_snapshots, species_signal_received,
signal_sum_snapshots (downsampled 4x spatial signal for GIF). EcosystemMeasures gains 6 new
fields: signal_total_history, signal_mass_fraction, signal_channel_snapshots,
dominant_channel_history, receptor_alignment, emission_reception_matrix. All in summary.json.

### Signal GIF

signal.gif generated alongside ecosystem.gif using species-colored teal colormap.
Mass/Signal tab toggle in ecosystem viewer. signal.gif checked in smoke_ecosystem.sh.

### Viewer improvements

Archive: SIGNAL badge, alpha/beta in creature modal, per-channel bar charts for signal params.
Ecosystem: SIGNAL badge, outcome tooltips, signal charts sidebar, corrected mass chart label.
Index: System tab anchor nav, quality metric section with formula, signal field section with
diagram, signal descriptors in descriptor grid, SIGNAL/TORUS badges on run cards.
docs/signal-field.svg: new diagram showing one-step signal mechanics with alpha/beta.

---

## v3.3.0 - 2026-04-18

Signal field: per-creature emission and sensing in a shared (H,W,16) chemical field.

Per-creature parameters: emission_vector (C,), receptor_profile (C,), signal_kernel (ring),
emission_rate scalar, decay_rates (C,). --signal-field flag in CLI. Archive tagged
signal_field: true/false in manifest.json. Quality filter updated for mass+signal conservation.
Both homogeneous and heterogeneous ecosystem paths signal-aware. validate_signal_consistency()
enforces archive compatibility (signal and non-signal archives cannot be mixed).

---

## v3.2.0 - 2026-04-18

Temporal outcome classifier with separate taxonomies per run mode.

Heterogeneous: merger, coexistence, exclusion, fragmentation -- now as temporal label sequence
(list of OutcomeWindow with from_step, to_step, label) per species. Fragmentation uses patch
count from v3.1.0 HeteroSpatial. Homogeneous taxonomy: full_merger, stable_isolation,
partial_clustering, cannibalism, fragmentation -- all measurable from patch count dynamics.
Outcome timeline visualization in ecosystem viewer. outcome_sequence stored in summary.json.

---

## v3.1.0 - 2026-04-18

Spatial observables for both run modes from existing snapshot data (no new simulation code).

Heterogeneous: patch count per species per snapshot, interface area per pair per snapshot,
Euclidean COM distance per pair per snapshot, spatial entropy per species per snapshot,
contact_occurred S×S bool matrix. Interaction coefficients gated to windows where
interface_area > 0. 4-connected component labeling via scipy.ndimage.label.
Homogeneous: patch count over time, mass spatial entropy, initial patch sizes, patch size
distribution per snapshot. scipy declared as explicit dependency. HeteroSpatial /
HomoSpatial typed dataclasses in analytics.py.

---

## v3.0.0 - 2026-04-17

Growth field capture, empirical interaction coefficients, ecosystem outcome classification, interaction heatmap in viewer.

### Per-species growth field capture

`LocalizedFlowLenia.step` refactored through `_step_inner(capture_growth)`. New `step_with_diagnostics()` returns `(LocalizedState, list[Tensor])` where each tensor is `G_s_total` (H, W) for that species before ownership blending. Called only at snapshot steps; `step()` used at all others. Growth tensors stored as `SimOutput.growth_snapshots[snap_idx][species_idx]`.

### Empirical interaction coefficients

`compute_interaction_coefficients()` in `src/biota/ecosystem/interaction.py`. For each ordered pair (A, B): `mean(G_A | W_B > 0.3) - mean(G_A | W_B < 0.05)`, accumulated across all snapshots. NaN when species never co-occurred. Result is an S x S nested list stored in `summary.json["measures"]["interaction_coefficients"]`.

### Ecosystem outcome classification

`classify_outcome()` in `src/biota/ecosystem/interaction.py`. Priority order: exclusion (species territory drops below 5% of initial) > merger (per-cell ownership entropy > 0.85 of maximum) > fragmentation (territory CV > 0.5) > coexistence. Stored in `summary.json["measures"]["outcome_label"]`.

### Interaction heatmap and outcome badge in viewer

`ecosystem.html` extended with an outcome badge in the header (four CSS classes: merger, coexistence, exclusion, fragmentation) and an S x S canvas heatmap in the sidebar. Color scale: teal at +maxAbs, red at -maxAbs, dark neutral at zero, gray for NaN. Axis labels colored by species palette. `build_index.py` reads both new fields from `summary.json` and passes them to the template.

### Mobile ecosystem layout overhaul

Breakpoint moved from 700px to 1100px. `body` gets `height: auto; overflow: auto` at mobile. A separate `.eco-gif-mobile` element is rendered above the sidebar and shown at mobile; the pan/zoom viewport is hidden with `display: none !important`. Pan/zoom JS gates on `!isMobile` check at 1100px. Eliminates touch-scroll hijacking on all mobile viewports.

---

## v2.5.0 - 2026-04-16

Species-colored ecosystem rendering, per-species territory and mass charts, atlas IA restructure, architecture SVG diagrams.

### Species-colored GIF rendering

8-color perceptually distinct palette (`SPECIES_PALETTE` in `run.py`): warm orange, sky blue, lime green, hot pink, purple, gold, teal, coral. `_colorize_frame_species` blends palette colors by ownership weight per pixel, scales by mass intensity. Homogeneous runs keep the existing magma colormap.

### Per-species territory and mass charts

Effective area (`sum(weights[:,:,s])`) tracked at every step as `species_territory_history`. Per-species mass also tracked. Both stored in `EcosystemMeasures` and `summary.json`. Rendered as multi-line canvas charts in the ecosystem viewer sidebar via shared `drawSpeciesChart()` (DPR-aware, subsamples to canvas width).

### Atlas IA restructure

System tab now lands (was About). Contains architecture diagrams, cluster stats, descriptor grid. About trimmed to project context and further reading. Three architecture SVGs (`search-loop.svg`, `archive-grid.svg`, `ecosystem-dispatch.svg`) inlined at build time via Jinja variables in `build_index.py`.

---

## v2.4.0 - 2026-04-15


Cluster-safe ecosystem dispatch. The v2.3.0 cluster ecosystem path made two wrong filesystem assumptions: workers could read the driver's archive directory (broken — `FileNotFoundError` on the first task), and outputs written by workers would be visible to the driver (broken — outputs landed on the worker filesystem). v2.4.0 fixes both with the same architectural pattern: dispatcher tasks are self-contained payloads, no shared filesystem assumed at any point.

### Driver-side creature resolution

The dispatcher now loads creatures on the driver from the local archive directory before submitting Ray tasks, then ships the loaded `RolloutResult` objects in the task payload. Workers never read the archive filesystem.

- `run_ecosystem` accepts an optional `creatures: list[RolloutResult]` parameter. When provided, the runner uses these directly instead of calling `load_creature`. When `None`, the disk-load path is unchanged (the standard sequential CLI path).
- `_load_creature` promoted to public `load_creature` since it's now a shared utility used by `run.py` internally and `dispatch.py` externally.

### Driver-side output materialization

The runner is split into pure compute and disk materialization. Workers run the compute and ship artifact bytes back to the driver, which writes them under its own `output_root`. Cluster runs land on the driver, not on whichever worker happened to execute the task.

- New `compute_ecosystem(config, output_root, creatures=None) -> tuple[EcosystemResult, dict[str, bytes]]`. Pure compute: no I/O. Returns the result alongside an artifacts dict mapping relative filenames (`config.json`, `summary.json`, `trajectory.npy`, `ecosystem.gif` or `frames/step_*.png`) to their bytes.
- New `materialize_outputs(run_dir, artifacts)` writes the bytes dict to disk, creating subdirectories as needed.
- `run_ecosystem` is now a thin wrapper around `compute_ecosystem` + `materialize_outputs`. Same signature, same behavior for the sequential CLI path.
- Per-frame PNG streaming during the simulation loop is gone — frames mode now buffers all snapshots and renders at the end. Memory cost is acceptable since long runs already buffered the snapshots in memory anyway for `trajectory.npy` and GIF rendering.
- Old single-step writers `_save_frame_png`, `_write_gif`, `_write_outputs` removed (nothing referenced them after the refactor).

### Failure isolation

Driver-side load failures (missing archive, missing cell) isolate per experiment the same way runtime failures do. Worker-side simulation or render failures isolate similarly. Pre-failure lines print inline with runtime progress for honest tallying.

### Tests

291 passing, 1 skipped, 2 deselected (was 287 + 1). 4 new functional tests:

- `test_run_ecosystem_accepts_preloaded_creatures`: disk and pre-loaded paths give bit-identical results.
- `test_run_ecosystem_rejects_creatures_length_mismatch`: length validation.
- `test_compute_ecosystem_returns_artifacts_without_io`: compute path writes nothing to disk and returns the expected artifact set.
- `test_compute_ecosystem_then_materialize_matches_run_ecosystem`: round-trip equivalence with the sequential path.

### Smoke tests

`scripts/smoke_ecosystem.sh` reverts the cluster-skip workaround added in v2.3.0; output verification now applies uniformly across noray, local, and cluster transports because outputs always land driver-local.

`just smoke-cluster-cpu HEAD_ADDR` now passes end-to-end. v2.3.0's same recipe failed twice: first with `FileNotFoundError` because workers couldn't reach the archive seeded on the driver, then again after a workaround with a missing output directory because workers had written results to their own filesystem.

---

## v2.3.0 - 2026-04-15

Three additions across the ecosystem stack: per-source patch sizing in YAML configs, parallel multi-experiment dispatch via Ray, and a sidebar layout with a borderless pan/zoom canvas for ecosystem run pages.

### New features

**Per-source patch override.** Each `CreatureSource` in an experiment YAML can declare its own `patch` size, overriding the experiment-level `spawn.patch`. Useful when species in a single run have substantially different natural scales (small fast glider mixed with a large dense colony, etc.). When omitted, sources fall back to `spawn.patch`.

```yaml
sources:
  - run: my-run
    cell: [5, 8, 3]
    n: 4
    patch: 80
```

**Parallel ecosystem dispatch via Ray.** `biota ecosystem --local-ray` (or `--ray-address HOST:PORT`) now runs experiments in parallel instead of sequentially. Two new flags:
- `--workers N`: maximum experiments running concurrently. Defaults to detected CUDA GPU count (or 1).
- `--gpu-fraction F`: fraction of a GPU each worker reserves. Default 1.0 = one per GPU. Set to 0.5 to pack two workers per GPU when individual experiments leave the GPU underutilized.

Failures isolate per experiment: one bad run does not abort the others. Failed experiments are listed at the end with their exception type and the CLI exits non-zero. Progress streams via `ray.wait` as each task completes.

**Sidebar layout for ecosystem run pages.** The `view.html` template now uses a 320px sticky sidebar (source list, mass stats, mass chart) plus a borderless dark canvas hosting the animation. The canvas supports drag-to-pan, wheel-to-zoom centered on the cursor, and double-click to reset. Mass chart gets an explicit "mass essentially flat across run" fallback when the line path is suppressed for low variance. Mobile collapses to single column at <900px.

### Implementation notes

- Per-source patch is purely additive; v2.2.0 configs without overrides go through identical code paths because `patch_override=None` falls back to `spawn.patch` everywhere.
- `build_initial_state_multi_species` uses `max(patches)` for the Poisson disk margin so the largest creature still fits inside the wall border.
- Ray dispatch lives in `biota.ecosystem.dispatch` (pyright basic mode like `ray_compat.py`), keeping Ray imports out of the strict-typed core.
- Auto-derivation of patch from creature parameters was considered and rejected: depends on the search preset's R distribution which the ecosystem layer shouldn't have to know about, and shipping a "smart default" creates user expectations biota cannot reliably meet across custom presets.
- Per-source `min_dist` override is not supported in v2.3.0 since it would break the deterministic single-pass spawn ordering. Save for v2.4.0 if anyone asks.
- Pan/zoom uses a `requestAnimationFrame`-debounced refit on resize and a smooth multiplicative wheel zoom. Initial scale is computed from natural image size to fit the viewport with 8% breathing margin.

### Smoke tests

Reorganized as a transport×device×command grid. Naming: `smoke-<transport>-<device>-<command>` for individual recipes, `smoke-<transport>-<device>` umbrellas that run both commands in a cell.

Transports: `noray` (sequential, no Ray; catches packaging regressions and device code paths CI cannot reach), `local` (fresh single-host Ray), `cluster` (attach to running cluster). Devices: `cpu`, `mps`, `cuda`. The `cluster + mps` cell is omitted (Linux nodes don't have MPS), giving 8 cells × 3 recipes = 24 recipes. A top-level `smoke-ray` umbrella runs the CPU-only sanity (`smoke-noray-cpu` + `smoke-local-cpu`).

Both commands now share extracted scaffolds: `scripts/smoke_search.sh` (new) and `scripts/smoke_ecosystem.sh` (extended to support the noray transport). Each justfile recipe is a 5-line env-var-setting wrapper.

Recipe rename map (old → new):

- `smoke-ray` → `smoke-local-cpu-search`
- `smoke-ray-mps` → `smoke-noray-mps-search`
- `smoke-ray-cuda` → `smoke-noray-cuda-search`
- `smoke-cluster HEAD` → `smoke-cluster-cpu-search HEAD`
- `smoke-cluster-cuda HEAD` → `smoke-cluster-cuda-search HEAD`
- `smoke-ray-ecosystem` → `smoke-local-cpu-ecosystem`
- `smoke-ray-mps-ecosystem` → `smoke-local-mps-ecosystem`
- `smoke-ray-cuda-ecosystem` → `smoke-local-cuda-ecosystem`
- `smoke-cluster-ecosystem HEAD` → `smoke-cluster-cpu-ecosystem HEAD`
- `smoke-cluster-cuda-ecosystem HEAD` → `smoke-cluster-cuda-ecosystem HEAD`

Heads-up: shell history and any local automation referencing the old names breaks. The new naming is the trade for a discoverable grid where adding a new transport or device means adding rows or columns rather than inventing more ad-hoc names.

The `SMOKE_RAY_MODE` env var on `scripts/smoke_ecosystem.sh` is renamed to `SMOKE_TRANSPORT` for consistency with the new search script. Anyone calling the script directly (rather than via just) needs to update.

### Test count

287 passing, 1 skipped, 2 deselected (was 273 + 1). 10 new functional tests (3 parser + 3 spawn/integration for patch override; 4 dispatch validation; plus 5 CLI tests for device/gpu-fraction interaction guards) and 2 new `smoke_ray`-marked integration tests for real-Ray dispatch.

---

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
