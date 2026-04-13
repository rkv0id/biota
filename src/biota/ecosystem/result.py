"""Data structures for ecosystem simulation configuration and output.

EcosystemConfig is the complete, serializable specification of one ecosystem
run. EcosystemResult is what comes back after the simulation completes.
Both are plain-Python dataclasses with no torch tensors or archive objects,
so they serialize cleanly to JSON.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SpawnConfig:
    """How creatures are placed on the shared grid.

    Positions are generated via Poisson disk sampling: no two creatures
    start closer than min_dist pixels. This gives naturalistic initial
    separation while remaining fully reproducible from the seed.

    Attributes:
        n:        Number of creatures to spawn.
        min_dist: Minimum pixel distance between any two spawn centers.
                  Should be at least 2x the creature's typical radius.
        patch:    Side length of the initial random patch placed at each
                  spawn position, in ecosystem grid pixels.
        seed:     Base RNG seed. Position sampling uses this seed;
                  patch initialization uses seed + 1000 + i per creature.
    """

    n: int
    min_dist: int
    patch: int
    seed: int


@dataclass(frozen=True)
class EcosystemConfig:
    """Complete specification of one ecosystem run.

    Attributes:
        source_run_id:  The archive run this creature came from.
        source_coords:  The (y, x, z) archive cell coordinate.
        grid:           Side length of the shared ecosystem grid.
        steps:          Number of simulation steps to run.
        snapshot_every: Capture a full state snapshot every N steps.
                        Snapshots are stored in trajectory.npy.
        spawn:          Spawn layout configuration.
        device:         Torch device for the simulation.
    """

    source_run_id: str
    source_coords: tuple[int, int, int]
    grid: int
    steps: int
    snapshot_every: int
    spawn: SpawnConfig
    device: str = "cpu"


@dataclass
class EcosystemMeasures:
    """Per-step and summary measures from the ecosystem run.

    Attributes:
        initial_mass:   Total mass at step 0 (sum of all spawn patches).
        final_mass:     Total mass at the final step.
        mass_history:   Total mass at each step. Length = steps + 1.
        peak_mass:      Maximum total mass observed during the run.
        min_mass:       Minimum total mass observed during the run.
        mass_turnover:  Mean absolute per-step mass change, normalized
                        by initial_mass. High = dynamic, low = stable.
        snapshot_steps: Steps at which full state snapshots were captured.
    """

    initial_mass: float
    final_mass: float
    mass_history: list[float]
    peak_mass: float
    min_mass: float
    mass_turnover: float
    snapshot_steps: list[int]


@dataclass
class EcosystemResult:
    """Output of one ecosystem simulation run.

    Attributes:
        config:     The EcosystemConfig that produced this result.
        run_id:     Unique identifier for this ecosystem run.
        run_dir:    Path to the output directory as a string.
        measures:   Quantitative measures from the simulation.
        elapsed_seconds: Wall-clock time for the simulation.
    """

    config: EcosystemConfig
    run_id: str
    run_dir: str
    measures: EcosystemMeasures
    elapsed_seconds: float

    def to_summary_dict(self) -> dict[str, object]:
        """Produce a JSON-serializable summary dict for summary.json."""
        cfg = self.config
        sp = cfg.spawn
        m = self.measures
        return {
            "run_id": self.run_id,
            "source_run_id": cfg.source_run_id,
            "source_coords": list(cfg.source_coords),
            "grid": cfg.grid,
            "steps": cfg.steps,
            "snapshot_every": cfg.snapshot_every,
            "device": cfg.device,
            "spawn": {
                "n": sp.n,
                "min_dist": sp.min_dist,
                "patch": sp.patch,
                "seed": sp.seed,
            },
            "measures": {
                "initial_mass": m.initial_mass,
                "final_mass": m.final_mass,
                "peak_mass": m.peak_mass,
                "min_mass": m.min_mass,
                "mass_turnover": m.mass_turnover,
                "snapshot_steps": m.snapshot_steps,
            },
            "elapsed_seconds": self.elapsed_seconds,
        }
