"""Post-run interaction measurement and ecosystem outcome classification.

All functions operate on data already captured during the simulation loop.
No simulation code is touched.

compute_interaction_coefficients
    Takes the per-species growth field snapshots and the ownership snapshots.
    For each ordered pair (A, B), measures how B's presence at a cell correlates
    with A's growth field value at that cell, compared to cells where B is
    absent. The result is an S x S float matrix stored in summary.json.

classify_outcome_hetero
    Produces a per-species temporal label sequence from ownership snapshots,
    territory history, and patch count data. Each species gets an independent
    sequence of labeled windows. The run-level label is derived from the final
    window states across all species.

classify_outcome_homo
    Produces a single run-level temporal label sequence from patch count history
    and patch size history. Labels: full_merger, stable_isolation,
    partial_clustering, cannibalism, fragmentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OutcomeWindow:
    """One labeled interval in an outcome sequence.

    from_step and to_step are simulation step numbers (not snapshot indices).
    to_step is the last step of the window, inclusive.
    """

    label: str
    from_step: int
    to_step: int


@dataclass
class OutcomeSequence:
    """Temporal outcome classification for one ecosystem run.

    For heterogeneous runs, series is indexed by species: series[s] is the
    sequence of labeled windows for species s. For homogeneous runs, series
    has a single entry representing the collective run dynamics.

    final_label is the dominant run-level label derived from the last window
    of each series entry. Priority: exclusion > fragmentation > cannibalism >
    partial_clustering > merger > stable_isolation > coexistence.
    """

    series: list[list[OutcomeWindow]] = field(default_factory=list)
    final_label: str = ""


def compute_interaction_coefficients(
    ownership_snapshots: list[np.ndarray],
    growth_snapshots: list[list[np.ndarray]],
    presence_threshold: float = 0.3,
    absence_threshold: float = 0.05,
    interface_area: list[list[list[int]]] | None = None,
) -> list[list[float]]:
    """Compute empirical S x S interaction coefficient matrix.

    For each ordered pair (A, B), the coefficient measures the average
    difference in species A's growth field at cells where species B is
    clearly present versus cells where B is clearly absent:

        interaction[A][B] = mean(G_A | W_B > presence_threshold)
                          - mean(G_A | W_B < absence_threshold)

    Positive: B enhances A (mutualism or spatial attraction).
    Negative: B suppresses A (competition or spatial repulsion).
    Near zero: neutral.

    The diagonal (A == B) measures self-growth: mean of G_A over all cells
    where A itself is present. This is a useful baseline but not an
    interaction coefficient in the ecological sense.

    When interface_area is provided, measurement for pair (A, B) is restricted
    to snapshot windows where interface_area[snap][A][B] > 0. This prevents
    the coefficient from being dominated by snapshots where the species are
    spatially separated: when they do not overlap, the presence mask barely
    fires and the coefficient measures absence of contact, not a biological
    signal. If no snapshot has interface area > 0 for a pair, NaN is reported
    (same as the no-data case).

    Returns an S x S nested list of floats. Returns an empty list if
    ownership_snapshots or growth_snapshots are empty (homogeneous runs).

    Args:
        ownership_snapshots: list of (H, W, S) float32 arrays, one per
            snapshot step.
        growth_snapshots: list of lists of (H, W) float32 arrays. Outer
            index is snapshot, inner index is species.
        presence_threshold: ownership weight above which a species is
            considered "present" at a cell.
        absence_threshold: ownership weight below which a species is
            considered "absent" at a cell.
        interface_area: optional (n_snapshots, S, S) nested list from
            compute_spatial_observables_hetero. When given, snapshot windows
            with interface_area[snap][A][B] == 0 are excluded from
            accumulation for pair (A, B).
    """
    if not ownership_snapshots or not growth_snapshots:
        return []
    if len(ownership_snapshots) != len(growth_snapshots):
        raise ValueError(
            f"ownership_snapshots length {len(ownership_snapshots)} "
            f"does not match growth_snapshots length {len(growth_snapshots)}"
        )

    s = ownership_snapshots[0].shape[2]

    # Accumulate per-pair sums across all snapshots and cells.
    # present_sum[A][B]: sum of G_A values where W_B > presence_threshold
    # present_count[A][B]: number of such cells
    # absent_sum[A][B], absent_count[A][B]: same for W_B < absence_threshold
    present_sum = [[0.0] * s for _ in range(s)]
    present_count = [[0] * s for _ in range(s)]
    absent_sum = [[0.0] * s for _ in range(s)]
    absent_count = [[0] * s for _ in range(s)]

    for snap_idx, (own, growths) in enumerate(
        zip(ownership_snapshots, growth_snapshots, strict=True)
    ):
        if own.shape[2] != s:
            raise ValueError(
                f"snapshot {snap_idx}: ownership has {own.shape[2]} species, expected {s}"
            )
        if len(growths) != s:
            raise ValueError(
                f"snapshot {snap_idx}: growth_snapshots has {len(growths)} entries, expected {s}"
            )
        for a in range(s):
            g_a = growths[a].astype(np.float64)  # (H, W)
            for b in range(s):
                # For off-diagonal pairs, skip snapshots with no interface when
                # interface_area data is available. The coefficient is only
                # meaningful where species actually co-occur.
                if a != b and interface_area is not None and interface_area[snap_idx][a][b] == 0:
                    continue
                w_b = own[:, :, b].astype(np.float64)  # (H, W)
                present_mask = w_b > presence_threshold
                absent_mask = w_b < absence_threshold
                if present_mask.any():
                    present_sum[a][b] += float(g_a[present_mask].sum())
                    present_count[a][b] += int(present_mask.sum())
                if absent_mask.any():
                    absent_sum[a][b] += float(g_a[absent_mask].sum())
                    absent_count[a][b] += int(absent_mask.sum())

    coefficients: list[list[float]] = []
    for a in range(s):
        row: list[float] = []
        for b in range(s):
            pc = present_count[a][b]
            ac = absent_count[a][b]
            if pc > 0 and ac > 0:
                mean_present = present_sum[a][b] / pc
                mean_absent = absent_sum[a][b] / ac
                row.append(float(mean_present - mean_absent))
            else:
                # Insufficient data: species never co-occurred or were always
                # co-absent. Report NaN so the viewer can show "no data" rather
                # than a misleading zero.
                row.append(float("nan"))
        coefficients.append(row)

    return coefficients


def _ownership_entropy_at(ownership: np.ndarray) -> float:
    """Mean per-cell Shannon entropy of an (H, W, S) ownership snapshot, in [0, 1]."""
    s = ownership.shape[2]
    total = ownership.sum(axis=2)
    occupied = total > 0.1
    if not occupied.any():
        return 0.0
    norm = ownership[occupied] / total[occupied, np.newaxis]
    clipped = np.clip(norm, 1e-12, 1.0)
    per_cell = -(clipped * np.log(clipped)).sum(axis=1)
    max_e = np.log(s)
    return float(per_cell.mean()) / max_e if max_e > 0 else 0.0


def _derive_final_label(per_species_final: list[str]) -> str:
    """Derive a single run-level label from the final label of each species."""
    priority = [
        "exclusion",
        "fragmentation",
        "cannibalism",
        "partial_clustering",
        "merger",
        "stable_isolation",
        "coexistence",
    ]
    for label in priority:
        if label in per_species_final:
            return label
    return per_species_final[0] if per_species_final else ""


def build_windows(
    labels_per_snapshot: list[str],
    snapshot_steps: list[int],
    debounce: int,
) -> list[OutcomeWindow]:
    """Convert a per-snapshot label list into debounced OutcomeWindow list.

    A label transition is only committed after `debounce` consecutive snapshots
    with the new label. Prevents transient single-snapshot splits from producing
    spurious windows.
    """
    if not labels_per_snapshot:
        return []

    n = len(labels_per_snapshot)
    smoothed: list[str] = [labels_per_snapshot[0]] * n
    current = labels_per_snapshot[0]
    candidate = current
    candidate_count = 0
    for i in range(n):
        raw = labels_per_snapshot[i]
        if raw == current:
            candidate = current
            candidate_count = 0
        else:
            if raw == candidate:
                candidate_count += 1
            else:
                candidate = raw
                candidate_count = 1
            if candidate_count >= debounce:
                current = candidate
                candidate_count = 0
        smoothed[i] = current

    windows: list[OutcomeWindow] = []
    run_label = smoothed[0]
    run_start = snapshot_steps[0]
    for i in range(1, n):
        if smoothed[i] != run_label:
            windows.append(OutcomeWindow(run_label, run_start, snapshot_steps[i - 1]))
            run_label = smoothed[i]
            run_start = snapshot_steps[i]
    windows.append(OutcomeWindow(run_label, run_start, snapshot_steps[-1]))
    return windows


def classify_outcome_hetero(
    species_territory_history: list[list[float]],
    ownership_snapshots: list[np.ndarray],
    snapshot_steps: list[int],
    species_patch_count: list[list[int]],
    exclusion_threshold: float = 0.05,
    merger_entropy_threshold: float = 0.85,
    fragmentation_debounce: int = 3,
) -> OutcomeSequence:
    """Temporal outcome classification for a heterogeneous ecosystem run.

    Each species gets an independent sequence of labeled windows. Labels per
    snapshot per species follow priority: exclusion > merger > fragmentation >
    coexistence. Transitions are debounced to suppress transient splits.

    Args:
        species_territory_history: (S, steps+1) territory per step.
        ownership_snapshots: list of (H, W, S) arrays, one per snapshot.
        snapshot_steps: step numbers corresponding to ownership_snapshots.
        species_patch_count: (S, n_snapshots) patch counts from analytics.
        exclusion_threshold: territory fraction below which a species is excluded.
        merger_entropy_threshold: ownership entropy above which species are merged.
        fragmentation_debounce: consecutive snapshots to commit a transition.
    """
    n_species = len(species_territory_history)
    if n_species < 2 or not ownership_snapshots or not snapshot_steps:
        return OutcomeSequence()

    n_snaps = len(snapshot_steps)
    initial_territories = [hist[0] for hist in species_territory_history]
    all_series: list[list[OutcomeWindow]] = []

    for sp in range(n_species):
        init_t = initial_territories[sp]
        labels: list[str] = []

        for snap_i in range(n_snaps):
            step = snapshot_steps[snap_i]
            own = ownership_snapshots[snap_i]
            t_hist = species_territory_history[sp]
            t_idx = min(step, len(t_hist) - 1)
            territory = t_hist[t_idx]

            if init_t > 0 and (territory / init_t) < exclusion_threshold:
                labels.append("exclusion")
                continue

            if _ownership_entropy_at(own) >= merger_entropy_threshold:
                labels.append("merger")
                continue

            counts = species_patch_count[sp] if sp < len(species_patch_count) else []
            if snap_i < len(counts) and counts[snap_i] > 1:
                labels.append("fragmentation")
                continue

            labels.append("coexistence")

        windows = build_windows(labels, snapshot_steps, debounce=fragmentation_debounce)
        all_series.append(windows)

    final_labels = [s[-1].label if s else "coexistence" for s in all_series]
    return OutcomeSequence(series=all_series, final_label=_derive_final_label(final_labels))


def classify_outcome_homo(
    snapshot_steps: list[int],
    patch_count_history: list[int],
    patch_size_history: list[list[int]],
    initial_patch_sizes: list[int],
    debounce: int = 3,
) -> OutcomeSequence:
    """Temporal outcome classification for a homogeneous ecosystem run.

    Returns a single-entry OutcomeSequence representing the collective run
    dynamics.

    Labels:
      full_merger       -- patch count reached 1.
      fragmentation     -- patch count rose above the initial count.
      cannibalism       -- patch count decreasing while surviving patches exceed
                           1.5x the median initial patch size.
      partial_clustering -- patch count decreased from initial but stable above 1.
      stable_isolation  -- patch count near initial throughout.

    Args:
        snapshot_steps: step numbers for each snapshot.
        patch_count_history: patch count at each snapshot.
        patch_size_history: patch sizes (descending) at each snapshot.
        initial_patch_sizes: patch sizes at the first snapshot (baseline).
        debounce: consecutive snapshots to commit a label transition.
    """
    if not snapshot_steps or not patch_count_history:
        return OutcomeSequence()

    n_initial = len(initial_patch_sizes)
    median_initial = sorted(initial_patch_sizes)[n_initial // 2] if initial_patch_sizes else 0

    labels: list[str] = []
    for i, count in enumerate(patch_count_history):
        sizes = patch_size_history[i] if i < len(patch_size_history) else []

        if count == 1:
            labels.append("full_merger")
        elif n_initial > 0 and count > n_initial:
            labels.append("fragmentation")
        elif (
            n_initial > 0
            and count < n_initial
            and median_initial > 0
            and sizes
            and sizes[0] > median_initial * 1.5
        ):
            labels.append("cannibalism")
        elif n_initial > 0 and count < n_initial:
            labels.append("partial_clustering")
        else:
            labels.append("stable_isolation")

    windows = build_windows(labels, snapshot_steps, debounce=debounce)
    final = windows[-1].label if windows else "stable_isolation"
    return OutcomeSequence(series=[windows], final_label=final)
