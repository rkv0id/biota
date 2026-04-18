"""Data structures produced by a completed ecosystem run.

EcosystemMeasures holds the time-series measures captured during the run.
EcosystemResult wraps the config, run id, output directory, measures, and
wall-clock timing into one object that serializes cleanly to summary.json.

Config-side dataclasses (EcosystemConfig, CreatureSource, SpawnConfig) live
in biota/ecosystem/config.py since they are parsed from YAML and validated
before a run starts.
"""

from dataclasses import dataclass, field

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
    # Dominant outcome label derived from the final window of outcome_sequence.
    # One of: merger, coexistence, exclusion, fragmentation (hetero) or
    # full_merger, stable_isolation, partial_clustering, cannibalism,
    # fragmentation (homo). Shown in the viewer badge.
    outcome_label: str
    # Temporal outcome sequence. For hetero runs, outer list indexed by species;
    # each entry is a list of OutcomeWindow dicts. For homo runs, one entry.
    # Each window: {"label": str, "from": int, "to": int}.
    outcome_sequence: list[list[dict[str, str | int]]] = field(default_factory=list)

    # --- Spatial observables: heterogeneous runs ---
    # Patch count per species per snapshot. Outer index = species (0..S-1),
    # inner index = snapshot. Empty for homogeneous runs.
    species_patch_count: list[list[int]] = field(default_factory=list)
    # Interface area per species pair per snapshot: cells where both species
    # exceed the ownership threshold simultaneously.
    # Shape: (n_snapshots, S, S). Symmetric; diagonal is zero.
    # Empty for homogeneous runs.
    species_interface_area: list[list[list[int]]] = field(default_factory=list)
    # Euclidean distance between centers of mass per species pair per snapshot.
    # Same shape as species_interface_area. inf when either species has no mass.
    species_com_distance: list[list[list[float]]] = field(default_factory=list)
    # Spatial entropy of each species' ownership distribution over a coarse grid,
    # per snapshot. Shape: (S, n_snapshots). Empty for homogeneous runs.
    species_spatial_entropy: list[list[float]] = field(default_factory=list)
    # contact_occurred[A][B]: True if any snapshot had interface_area[snap][A][B] > 0.
    # Lets the viewer distinguish "no contact" (NaN coeff) from "neutral contact".
    # Shape: (S, S). Empty for homogeneous runs.
    contact_occurred: list[list[bool]] = field(default_factory=list)

    # --- Signal observables: both run modes (empty when no signal field) ---
    # Total signal field mass per step (parallel to mass_history).
    signal_total_history: list[float] = field(default_factory=list)
    # Fraction of total mass in signal field per step: signal / (mass + signal).
    signal_mass_fraction: list[float] = field(default_factory=list)
    # Mean signal per channel at each snapshot step. Shape: (n_snapshots, C).
    signal_channel_snapshots: list[list[float]] = field(default_factory=list)
    # Index of the dominant (highest mean) signal channel at each snapshot.
    dominant_channel_history: list[float] = field(default_factory=list)

    # --- Signal observables: heterogeneous runs only ---
    # Receptor alignment per species per snapshot: dot(receptor_profile, mean_signal_received).
    # Shape: (S, n_snapshots). Positive = receiving aligned signal, negative = inhibitory.
    receptor_alignment: list[list[float]] = field(default_factory=list)
    # Emission-reception compatibility matrix: dot(emission_vector[i], receptor_profile[j]).
    # Shape: (S, S). Positive = species i's signal benefits species j.
    emission_reception_matrix: list[list[float]] = field(default_factory=list)

    # --- Spatial observables: homogeneous runs ---
    # Patch count of the mass field at each snapshot. Empty for heterogeneous runs.
    patch_count_history: list[int] = field(default_factory=list)
    # Shannon entropy of the mass distribution over a coarse grid, per snapshot.
    mass_spatial_entropy_history: list[float] = field(default_factory=list)
    # Patch sizes (pixel counts, descending) at the first snapshot.
    # Baseline for cannibalism detection: surviving patches that exceed their
    # initial size indicate one copy growing at another's expense.
    initial_patch_sizes: list[int] = field(default_factory=list)
    # Patch sizes (descending) at each snapshot.
    # Shape: (n_snapshots, n_patches_at_that_snapshot).
    patch_size_history: list[list[int]] = field(default_factory=list)


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
                "outcome_sequence": m.outcome_sequence,
                "species_patch_count": m.species_patch_count,
                "species_interface_area": m.species_interface_area,
                "species_com_distance": [
                    [
                        [
                            v if not (isinstance(v, float) and v == float("inf")) else None
                            for v in row
                        ]
                        for row in snap
                    ]
                    for snap in m.species_com_distance
                ],
                "species_spatial_entropy": m.species_spatial_entropy,
                "contact_occurred": m.contact_occurred,
                "patch_count_history": m.patch_count_history,
                "mass_spatial_entropy_history": m.mass_spatial_entropy_history,
                "initial_patch_sizes": m.initial_patch_sizes,
                "patch_size_history": m.patch_size_history,
                "signal_total_history": m.signal_total_history,
                "signal_mass_fraction": m.signal_mass_fraction,
                "signal_channel_snapshots": m.signal_channel_snapshots,
                "dominant_channel_history": m.dominant_channel_history,
                "receptor_alignment": m.receptor_alignment,
                "emission_reception_matrix": m.emission_reception_matrix,
            },
            "elapsed_seconds": self.elapsed_seconds,
        }
