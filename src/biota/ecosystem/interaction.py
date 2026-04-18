"""Post-run interaction measurement and ecosystem outcome classification.

Both functions operate on data already captured during the simulation loop.
Neither touches the simulation itself. No new physics, no prescribed rules.

compute_interaction_coefficients
    Takes the per-species growth field snapshots and the ownership snapshots.
    For each ordered pair (A, B), measures how B's presence at a cell correlates
    with A's growth field value at that cell, compared to cells where B is
    absent. The result is an S x S float matrix stored in summary.json.

classify_outcome
    Takes species_territory_history (already in v2.5.0 output) and optionally
    the ownership snapshots for spatial analysis. Returns one of four labels:
    merger, coexistence, exclusion, fragmentation.
"""

from __future__ import annotations

import numpy as np


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


def classify_outcome(
    species_territory_history: list[list[float]],
    ownership_snapshots: list[np.ndarray],
    exclusion_threshold: float = 0.05,
    merger_entropy_threshold: float = 0.85,
    species_patch_count: list[list[int]] | None = None,
) -> str:
    """Classify an ecosystem run's outcome from territory history and ownership.

    Returns one of four labels:

    merger       - species converge into a shared spatial structure. Final
                   territory is distributed relatively evenly and ownership
                   entropy at occupied cells is high (species are mixed).

    coexistence  - all species maintain meaningful territory throughout the run
                   and ownership entropy stays low (clear boundaries persist).

    exclusion    - one or more species' territory drops below exclusion_threshold
                   (as a fraction of that species' initial territory) while at
                   least one other species retains territory.

    fragmentation - one or more non-excluded species has patch count > 1 at
                   any snapshot and remains fragmented at the final snapshot.
                   When species_patch_count is not provided, falls back to a
                   coarse CV heuristic on the territory time series.

    Returns an empty string if there is insufficient data (e.g., fewer than
    two species or empty history).

    Args:
        species_territory_history: outer list indexed by species, inner by
            step. Length S x (steps + 1). From EcosystemMeasures.
        ownership_snapshots: list of (H, W, S) float32 arrays. Used to
            compute spatial ownership entropy at the final snapshot.
        exclusion_threshold: a species is considered excluded if its final
            territory is below this fraction of its initial territory.
        merger_entropy_threshold: mean per-cell ownership entropy at the
            final snapshot above which species are considered merged.
        species_patch_count: optional (S, n_snapshots) patch count data from
            compute_spatial_observables_hetero. When provided, fragmentation
            is detected directly: a species is fragmented if its patch count
            exceeds 1 at any snapshot and is still > 1 at the final snapshot.
    """
    n_species = len(species_territory_history)
    if n_species < 2:
        return ""
    for hist in species_territory_history:
        if len(hist) < 2:
            return ""

    initial_territories = [hist[0] for hist in species_territory_history]
    final_territories = [hist[-1] for hist in species_territory_history]

    # Exclusion: any species ends with < threshold * initial territory,
    # while at least one other species has nonzero territory.
    excluded = [
        init > 0 and (final / init) < exclusion_threshold
        for init, final in zip(initial_territories, final_territories, strict=True)
    ]
    survivors = [
        not exc and final > 0 for exc, final in zip(excluded, final_territories, strict=True)
    ]
    if any(excluded) and any(survivors):
        return "exclusion"

    # Merger vs coexistence: use ownership entropy at the final snapshot if
    # available, otherwise fall back to territory ratio analysis.
    if ownership_snapshots:
        final_own = ownership_snapshots[-1]  # (H, W, S)
        _h, _w, s = final_own.shape
        # Per-cell Shannon entropy of ownership distribution.
        # Only computed at cells where total ownership is meaningful.
        total_own = final_own.sum(axis=2)  # (H, W)
        occupied = total_own > 0.1
        if occupied.any():
            # Normalize to a proper distribution per occupied cell.
            own_norm = final_own[occupied] / total_own[occupied, np.newaxis]
            # Clip to avoid log(0); zero-weight species contribute 0 to entropy.
            own_clipped = np.clip(own_norm, 1e-12, 1.0)
            per_cell_entropy = -(own_clipped * np.log(own_clipped)).sum(axis=1)
            # Normalize by log(S) so entropy is in [0, 1].
            max_entropy = np.log(s)
            normalized_entropy = (
                float(per_cell_entropy.mean()) / max_entropy if max_entropy > 0 else 0.0
            )
            if normalized_entropy >= merger_entropy_threshold:
                return "merger"
    else:
        # Fallback: if final territory ratio is near uniform (within 20% of
        # mean), treat as merger candidate.
        total_final = sum(final_territories)
        if total_final > 0:
            fractions = [t / total_final for t in final_territories]
            expected = 1.0 / n_species
            max_deviation = max(abs(f - expected) for f in fractions)
            if max_deviation < 0.2:
                return "merger"

    # Fragmentation: a non-excluded species has patch count > 1 at any snapshot
    # and is still fragmented (patch count > 1) at the final snapshot.
    # Falls back to the territory CV heuristic when patch count data is absent.
    if species_patch_count is not None:
        for s_idx in range(n_species):
            if excluded[s_idx]:
                continue
            counts = species_patch_count[s_idx]
            if counts and counts[-1] > 1 and any(c > 1 for c in counts):
                return "fragmentation"
    else:
        # CV heuristic: kept as fallback for callers without spatial data.
        for s_idx, hist in enumerate(species_territory_history):
            if excluded[s_idx]:
                continue
            arr = np.array(hist, dtype=np.float64)
            mean_t = float(arr.mean())
            if mean_t > 0:
                cv = float(arr.std()) / mean_t
                if cv > 0.5:
                    return "fragmentation"

    return "coexistence"
