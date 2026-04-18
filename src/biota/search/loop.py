"""The search loop.

Wires the search-layer primitives into a complete MAP-Elites search:

1. Initial random phase: sample_random(seed) for the first random_phase_size
   rollouts to seed the archive with diversity
2. Mutation phase: pick a uniformly random occupied cell, mutate its params,
   submit. Repeat until budget is exhausted.
3. Drain phase: wait for all in-flight rollouts to complete

All three phases share the same submit-then-drain inner pattern. The only
thing that varies is how params are generated.

The loop writes a run directory under runs/<run_id>/ containing:

  manifest.json   metadata (start time, biota version, config hash, etc.)
  config.toml     resolved SearchConfig serialized
  archive.pkl     latest archive snapshot, rewritten every checkpoint_every
  events.jsonl    append-only summary of every rollout (no thumbnails)
  metrics.jsonl   append-only throughput samples

Events also flow to an optional on_event callback so the CLI can print
progress and (eventually) the dashboard can stream live updates.

The submit-and-drain pattern uses ray_compat for all three modes (default
no-Ray, local Ray, attach to existing cluster) behind the same five-function API.
"""

import json
import os
import pickle
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

import numpy as np

from biota.ray_compat import (
    RolloutHandle,
    init,
    shutdown,
    submit_batch,
    wait_for_completed,
)
from biota.search.archive import Archive, InsertionStatus
from biota.search.descriptors import DEFAULT_DESCRIPTORS, resolve_descriptors
from biota.search.params import mutate, sample_random
from biota.search.result import CellCoord, ParamDict, RolloutResult
from biota.search.rollout import RolloutConfig

# === config ===


@dataclass(frozen=True)
class SearchConfig:
    """Top-level search configuration: budget, parallelism, persistence."""

    rollout: RolloutConfig
    """Per-rollout simulation parameters (grid, steps, kernels, etc.)."""

    budget: int = 500
    """Total number of rollouts to run, including the random phase."""

    random_phase_size: int = 200
    """Number of rollouts sampled uniformly from the prior before mutation
    kicks in. Seeds the archive with diversity."""

    batch_size: int = 1
    """Rollouts evaluated simultaneously per dispatch. Default 1 matches the
    pre-v0.4.0 behaviour. On cuda/mps, values of 32-128 give meaningful
    speedup by filling the GPU with a single vectorized forward pass. On cpu,
    leave at 1 - PyTorch already uses all threads internally for tensor ops
    and adding Python-level batch dispatch competes for the same cores."""

    workers: int = 1
    """Concurrent batch dispatches in flight. Controls the archive-freshness
    vs throughput tradeoff, not just utilisation. Default 1 = synchronous
    MAP-Elites: each batch completes and updates the archive before the next
    batch's parents are selected, giving maximally fresh parent selection.
    Higher values trade freshness for throughput on multi-node cluster
    setups. Effective concurrent rollouts = workers * batch_size."""

    local_ray: bool = False
    """If True, start a fresh local Ray instance in the driver process and run
    rollouts as Ray tasks. Mutually exclusive with ray_address."""

    ray_address: str | None = None
    """If set, attach to an already-running Ray cluster at this address. Value
    is passed verbatim to ray.init(address=...). Use 'host:port' for GCS-level
    connection from a node on the same network as the cluster. Use
    'ray://host:port' for Ray Client from an external machine (requires
    pip install ray[client], which biota does not declare). Mutually exclusive
    with local_ray. When neither local_ray nor ray_address is set, rollouts run
    synchronously in the driver (the no-Ray default)."""

    device: str = "cpu"
    """Torch device for the rollouts: 'cpu', 'mps', or 'cuda'."""

    checkpoint_every: int = 100
    """Rewrite the archive checkpoint after this many completed rollouts."""

    base_seed: int = 0
    """Seeds for individual rollouts are derived from this. Same base_seed
    produces the same sequence of sampled params (the mutation phase is
    nondeterministic in parallel mode because parent picking depends on
    completion order)."""

    descriptor_names: tuple[str, str, str] = DEFAULT_DESCRIPTORS
    """Names of the three active behavioral descriptors. Must all exist in
    biota.search.descriptors.REGISTRY (or in a custom module loaded via
    --descriptor-module). Controls both the archive axes and the rollout
    descriptor computation."""

    signal_field: bool = False
    """When True, signal field parameters (emission_vector, receptor_profile,
    signal_kernel_*) are sampled and mutated alongside the mass-kernel params.
    Produces a signal-enabled archive tagged in manifest.json. Incompatible
    with non-signal archives in ecosystem runs."""

    def __post_init__(self) -> None:
        if self.local_ray and self.ray_address is not None:
            raise ValueError(
                "local_ray and ray_address are mutually exclusive; pass one or the other, not both"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.workers < 1:
            raise ValueError(f"workers must be >= 1, got {self.workers}")


# === events ===


@dataclass(frozen=True)
class SearchStarted:
    """Emitted once at the start of the loop."""

    run_id: str
    run_dir: Path
    config: SearchConfig
    started_at: float


@dataclass(frozen=True)
class RolloutCompleted:
    """Emitted once for every completed rollout, accepted or rejected."""

    result: RolloutResult
    insertion_status: InsertionStatus
    completed_count: int
    """1-indexed counter of how many rollouts have finished so far."""


@dataclass(frozen=True)
class CheckpointWritten:
    """Emitted after a periodic checkpoint is written to disk."""

    path: Path
    archive_size: int


@dataclass(frozen=True)
class SearchFinished:
    """Emitted once at the end of the loop."""

    run_id: str
    run_dir: Path
    completed: int
    archive_size: int
    elapsed_seconds: float


SearchEvent = SearchStarted | RolloutCompleted | CheckpointWritten | SearchFinished
EventCallback = Callable[[SearchEvent], None]


# === public API ===


def search(
    config: SearchConfig,
    runs_root: Path | str = "archive",
    on_event: EventCallback | None = None,
    archive: Archive | None = None,
    run_id: str | None = None,
) -> Archive:
    """Run a complete MAP-Elites search to budget and return the populated archive.

    Creates a run directory under runs_root/<run_id>/, initializes the runtime
    (Ray or no-ray), runs the random + mutation + drain phases, writes
    checkpoints, and tears down. Always shuts down the runtime even if the
    loop raises mid-search.

    Pass an existing archive to resume from one (M5 work; for now just creates
    a fresh one if None).
    """
    if archive is None:
        archive = Archive(descriptor_names=config.descriptor_names)
    if run_id is None:
        run_id = _make_run_id()

    run_dir = Path(runs_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # Inject active descriptors into the rollout config so workers compute
    # the right three axes.
    active = resolve_descriptors(config.descriptor_names)
    rollout_with_descriptors = dc_replace(config.rollout, active_descriptors=active)
    config = dc_replace(config, rollout=rollout_with_descriptors)
    state = _LoopState(
        config=config,
        archive=archive,
        run_id=run_id,
        run_dir=run_dir,
        on_event=on_event,
        completed=0,
        rng=random.Random(config.base_seed + 31337),
    )
    _write_manifest(state)
    _write_config(state)

    started_at = time.time()
    state.started_at = started_at
    _emit(
        state,
        SearchStarted(run_id=run_id, run_dir=run_dir, config=config, started_at=started_at),
    )

    init(
        local_ray=config.local_ray,
        ray_address=config.ray_address,
    )
    try:
        in_flight: list[RolloutHandle] = []
        next_seed = config.base_seed

        # Phase 1: random sampling
        random_phase_end = config.base_seed + min(config.random_phase_size, config.budget)
        next_seed, in_flight = _submit_phase(
            state=state,
            sample_fn=_random_sampler,
            in_flight=in_flight,
            next_seed=next_seed,
            stop_seed=random_phase_end,
        )

        # Phase 2: mutation from archive
        if config.budget > config.random_phase_size:
            budget_end = config.base_seed + config.budget
            next_seed, in_flight = _submit_phase(
                state=state,
                sample_fn=_make_mutation_sampler(state),
                in_flight=in_flight,
                next_seed=next_seed,
                stop_seed=budget_end,
            )

        # Phase 3: drain remaining
        while in_flight:
            in_flight = _drain_some(state, in_flight)

        # Final checkpoint
        _write_archive_checkpoint(state)
    finally:
        shutdown()

    elapsed = time.time() - started_at
    _emit(
        state,
        SearchFinished(
            run_id=run_id,
            run_dir=run_dir,
            completed=state.completed,
            archive_size=len(archive),
            elapsed_seconds=elapsed,
        ),
    )
    return archive


# === internals ===


@dataclass
class _LoopState:
    config: SearchConfig
    archive: Archive
    run_id: str
    run_dir: Path
    on_event: EventCallback | None
    completed: int
    rng: random.Random
    started_at: float = 0.0


def _emit(state: _LoopState, event: SearchEvent) -> None:
    if state.on_event is not None:
        state.on_event(event)


# Type alias for the per-phase sampler. Returns (params, parent_cell, generation_seed).
# generation_seed gets attached to the rollout for reproducibility.
SamplerFn = Callable[[_LoopState, int], tuple[ParamDict, CellCoord | None]]


def _random_sampler(state: _LoopState, seed: int) -> tuple[ParamDict, CellCoord | None]:
    """Phase 1 sampler: uniform from the prior, no parent."""
    params = sample_random(
        kernels=state.config.rollout.sim.kernels,
        seed=seed,
        signal_field=state.config.signal_field,
    )
    return params, None


def _make_mutation_sampler(state: _LoopState) -> SamplerFn:
    """Phase 2 sampler: pick a parent from the archive, mutate, return."""

    def sampler(state: _LoopState, seed: int) -> tuple[ParamDict, CellCoord | None]:
        if len(state.archive) == 0:
            params = sample_random(
                kernels=state.config.rollout.sim.kernels,
                seed=seed,
                signal_field=state.config.signal_field,
            )
            return params, None
        rng_np = np.random.default_rng(seed)
        parent_cell, parent_result = state.archive.random_parent(rng_np)
        child = mutate(parent_result.params, seed=seed)
        return child, parent_cell

    return sampler


def _submit_phase(
    state: _LoopState,
    sample_fn: SamplerFn,
    in_flight: list[RolloutHandle],
    next_seed: int,
    stop_seed: int,
) -> tuple[int, list[RolloutHandle]]:
    """Submit batches until next_seed reaches stop_seed, draining as needed.

    Accumulates batch_size candidates, dispatches one batch handle, then
    checks whether workers handles are already in flight and drains if so.
    Budget is tracked in rollout units; each dispatch covers batch_size
    rollouts (last batch may be smaller if stop_seed - next_seed < batch_size).

    Returns the new (next_seed, in_flight) state.
    """
    batch_size = state.config.batch_size

    while next_seed < stop_seed:
        # Drain if we have reached the workers cap
        while len(in_flight) >= state.config.workers:
            in_flight = _drain_some(state, in_flight)

        # Build a batch of up to batch_size candidates
        this_batch_size = min(batch_size, stop_seed - next_seed)
        params_list: list[ParamDict] = []
        seeds: list[int] = []
        parent_cells: list[CellCoord | None] = []
        for i in range(this_batch_size):
            params, parent_cell = sample_fn(state, next_seed + i)
            params_list.append(params)
            seeds.append(next_seed + i)
            parent_cells.append(parent_cell)

        handle = submit_batch(
            params_list=params_list,
            seeds=seeds,
            config=state.config.rollout,
            device=state.config.device,
            parent_cells=parent_cells,
        )
        in_flight.append(handle)
        next_seed += this_batch_size

    return next_seed, in_flight


def _drain_some(state: _LoopState, in_flight: list[RolloutHandle]) -> list[RolloutHandle]:
    """Wait for at least one in-flight batch to complete, process all results
    from completed batches, and return the remaining handles.

    Each completed handle resolves to a list[RolloutResult] (one per rollout
    in that batch). All results are processed in arrival order.
    """
    completed_batches, still_pending = wait_for_completed(in_flight, min_completed=1)
    for batch in completed_batches:
        for result in batch:
            state.completed += 1
            status = state.archive.try_insert(result)
            _append_event_log(state, result, status)
            _emit(
                state,
                RolloutCompleted(
                    result=result,
                    insertion_status=status,
                    completed_count=state.completed,
                ),
            )
            if state.completed % state.config.checkpoint_every == 0:
                _write_archive_checkpoint(state)
    return still_pending


# === run directory and persistence ===


def _make_run_id() -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    adjective = random.choice(_ADJECTIVES)
    noun = random.choice(_NOUNS)
    return f"{timestamp}-{adjective}-{noun}"


_ADJECTIVES = (
    "amber",
    "azure",
    "brisk",
    "calm",
    "crisp",
    "dense",
    "dusky",
    "eager",
    "fierce",
    "frost",
    "gentle",
    "glossy",
    "hazy",
    "indigo",
    "jade",
    "keen",
    "lithe",
    "mossy",
    "noble",
    "ochre",
    "plush",
    "quiet",
    "rusty",
    "silver",
    "tawny",
    "umber",
    "vivid",
    "warm",
    "wry",
    "zesty",
)

_NOUNS = (
    "anchor",
    "beacon",
    "bramble",
    "cinder",
    "creek",
    "delta",
    "ember",
    "fjord",
    "glade",
    "harbor",
    "ibis",
    "junco",
    "kestrel",
    "lark",
    "moor",
    "nest",
    "orchid",
    "petal",
    "quartz",
    "ridge",
    "spire",
    "thistle",
    "umbra",
    "vale",
    "willow",
    "xenon",
    "yew",
    "zephyr",
    "atoll",
    "boulder",
)


def _write_manifest(state: _LoopState) -> None:
    manifest = {
        "run_id": state.run_id,
        "started_at": time.time(),
        "biota_version": _biota_version(),
        "config": _config_to_jsonable(state.config),
        "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "ray_active": False,  # set after init in a follow-up; not load-bearing
        "signal_field": state.config.signal_field,
    }
    path = state.run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))


def _write_config(state: _LoopState) -> None:
    """Mirror manifest's config block as a standalone toml file. Stdlib has
    no toml writer, so we emit a flat JSON structure under config.toml.json
    for now and document that the .toml format will land with the M2 dashboard.
    """
    path = state.run_dir / "config.json"
    path.write_text(json.dumps(_config_to_jsonable(state.config), indent=2))


def _write_archive_checkpoint(state: _LoopState) -> None:
    target = state.run_dir / "archive.pkl"
    tmp = target.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(state.archive, f)
    os.replace(tmp, target)
    _emit(
        state,
        CheckpointWritten(path=target, archive_size=len(state.archive)),
    )


def _append_event_log(state: _LoopState, result: RolloutResult, status: InsertionStatus) -> None:
    """Append one line to events.jsonl summarizing this rollout.

    The full RolloutResult (including the thumbnail) is too large for a
    grep-friendly JSONL log. We strip it down to the metadata that matters
    for forensic debugging: seed, descriptors, quality, status, parent.
    The full result still lives in archive.pkl when accepted.
    """
    summary = {
        "seed": result.seed,
        "descriptors": list(result.descriptors) if result.descriptors else None,
        "quality": result.quality,
        "rejection_reason": result.rejection_reason,
        "insertion_status": status.value,
        "parent_cell": list(result.parent_cell) if result.parent_cell else None,
        "compute_seconds": result.compute_seconds,
        "created_at": result.created_at,
    }
    path = state.run_dir / "events.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(summary) + "\n")


def _config_to_jsonable(config: SearchConfig) -> dict[str, Any]:
    """Flatten SearchConfig (which contains nested RolloutConfig and SimConfig)
    into a JSON-serializable dict.

    active_descriptors is excluded from the rollout sub-dict because it contains
    non-serializable Callable fields. descriptor_names on SearchConfig is the
    canonical JSON representation; the Descriptor objects are always re-derivable
    from the names at load time.
    """
    r = config.rollout
    out: dict[str, Any] = {
        "rollout": {
            "sim": asdict(r.sim),
            "steps": r.steps,
            "patch_fraction": r.patch_fraction,
            "thumbnail_frames": r.thumbnail_frames,
            "thumbnail_size": r.thumbnail_size,
        },
    }
    for key in (
        "budget",
        "random_phase_size",
        "batch_size",
        "workers",
        "local_ray",
        "ray_address",
        "device",
        "checkpoint_every",
        "base_seed",
        "descriptor_names",
    ):
        out[key] = getattr(config, key)
    return out


def _biota_version() -> str:
    try:
        from importlib.metadata import version

        return version("biota")
    except Exception:
        return "unknown"
