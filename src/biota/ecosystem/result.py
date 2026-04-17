"""Data structures produced by a completed ecosystem run.

EcosystemMeasures holds the time-series measures captured during the run.
EcosystemResult wraps the config, run id, output directory, measures, and
wall-clock timing into one object that serializes cleanly to summary.json.

Config-side dataclasses (EcosystemConfig, CreatureSource, SpawnConfig) live
in biota/ecosystem/config.py since they are parsed from YAML and validated
before a run starts.
"""

from dataclasses import dataclass

from biota.ecosystem.config import EcosystemConfig


@dataclass
class EcosystemMeasures:
    """Per-step and summary measures from one ecosystem run.

    Attributes:
        initial_mass:         Total mass at step 0 (sum of all spawn patches).
        final_mass:           Total mass at the final step.
        mass_history:         Total mass at each step. Length = steps + 1.
        peak_mass:            Maximum total mass observed during the run.
        min_mass:             Minimum total mass observed during the run.
        mass_turnover:        Mean absolute per-step mass change, normalized
                              by initial_mass. High = dynamic, low = stable.
        snapshot_steps:       Steps at which full state snapshots were captured.
        species_mass_history: Per-species mass at each step.
                              Outer list indexed by species (0..S-1), inner
                              list indexed by step (length = steps + 1).
                              For homogeneous runs, has one entry equal to
                              mass_history. For heterogeneous runs, the S
                              entries sum to mass_history at each step
                              (mass conservation). These are the data behind
                              the per-species mass chart in the viewer.
    """

    initial_mass: float
    final_mass: float
    mass_history: list[float]
    peak_mass: float
    min_mass: float
    mass_turnover: float
    snapshot_steps: list[int]
    species_mass_history: list[list[float]]
    species_territory_history: list[list[float]]
    # Empirical interaction coefficients: S x S matrix where entry [A][B]
    # is the mean effect of species B's presence on species A's growth field.
    # Negative = B suppresses A, positive = B enhances A, near zero = neutral.
    # Empty list for homogeneous runs or when growth snapshots are unavailable.
    interaction_coefficients: list[list[float]]
    # Ecosystem outcome class: "merger", "coexistence", "exclusion", or
    # "fragmentation". Empty string for homogeneous runs.
    outcome_label: str


@dataclass
class EcosystemResult:
    """Output of one ecosystem simulation run.

    Attributes:
        config:          The EcosystemConfig that produced this result.
        run_id:          Unique identifier for this ecosystem run.
        run_dir:         Path to the output directory as a string.
        measures:        Quantitative measures from the simulation.
        elapsed_seconds: Wall-clock time for the simulation.
    """

    config: EcosystemConfig
    run_id: str
    run_dir: str
    measures: EcosystemMeasures
    elapsed_seconds: float

    def to_summary_dict(self) -> dict[str, object]:
        """JSON-serializable summary dict for summary.json.

        The sources list preserves insertion order; species index in downstream
        tooling (viz, species weight tensor) follows the same ordering.
        """
        cfg = self.config
        sp = cfg.spawn
        m = self.measures
        return {
            "run_id": self.run_id,
            "name": cfg.name,
            "grid_h": cfg.grid_h,
            "grid_w": cfg.grid_w,
            "steps": cfg.steps,
            "snapshot_every": cfg.snapshot_every,
            "device": cfg.device,
            "border": cfg.border,
            "output_format": cfg.output_format,
            "mode": "heterogeneous" if cfg.is_heterogeneous else "homogeneous",
            "sources": [
                {
                    "archive_dir": s.archive_dir,
                    "run": s.run_id,
                    "cell": list(s.coords),
                    "n": s.n,
                    "patch": s.patch if s.patch is not None else sp.patch,
                }
                for s in cfg.sources
            ],
            "spawn": {
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
                "species_mass_history": m.species_mass_history,
                "species_territory_history": m.species_territory_history,
                "interaction_coefficients": m.interaction_coefficients,
                "outcome_label": m.outcome_label,
            },
            "elapsed_seconds": self.elapsed_seconds,
        }
