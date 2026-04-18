"""Spatial observables computed from snapshot data captured during ecosystem runs.

All functions operate purely on data already saved during the simulation loop.
No simulation code is touched. The two entry points are:

    compute_spatial_observables_hetero
        For heterogeneous runs: takes ownership_snapshots (H, W, S) and
        returns patch count, interface area, COM distance, spatial entropy
        per species, and contact_occurred per species pair.

    compute_spatial_observables_homo
        For homogeneous runs: takes mass snapshots (H, W) and returns
        patch count, spatial entropy, initial and per-snapshot patch size
        distributions of the mass field.

Connectivity: scipy.ndimage.label uses 4-connected (cross) structure by
default in 2D (generate_binary_structure(2, 1)). Two patches touching only
at a diagonal corner are counted as separate patches. This matches the
biological intuition that diagonal contact is not meaningful territory contact.

Coarse grid for spatial entropy: 8x8 bins by default. Shannon entropy is
computed over the mass or ownership distribution within each bin. A uniform
distribution yields maximum entropy; a single concentrated patch yields low
entropy. Entropy is normalized to [0, 1] by dividing by log(n_bins).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from scipy.ndimage import label as _cc_label

# 4-connected structure for 2D connected-component labeling.
# Two cells that touch only diagonally are NOT considered connected.
_STRUCTURE_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


@dataclass
class HeteroSpatial:
    """Spatial observables for one heterogeneous ecosystem run."""

    # Patch count per species per snapshot. Outer = species, inner = snapshot.
    species_patch_count: list[list[int]] = field(default_factory=list)
    # Interface area (cells where both species exceed threshold) per snapshot.
    # Shape: (n_snapshots, S, S). Symmetric; diagonal zero.
    species_interface_area: list[list[list[int]]] = field(default_factory=list)
    # Euclidean COM distance per species pair per snapshot. Same shape.
    # inf when either species has no mass at that snapshot.
    species_com_distance: list[list[list[float]]] = field(default_factory=list)
    # Spatial entropy of each species' ownership distribution. Shape: (S, n_snapshots).
    species_spatial_entropy: list[list[float]] = field(default_factory=list)
    # contact_occurred[A][B]: True if any snapshot had interface_area > 0 for pair (A,B).
    contact_occurred: list[list[bool]] = field(default_factory=list)


@dataclass
class HomoSpatial:
    """Spatial observables for one homogeneous ecosystem run."""

    # Patch count of the mass field at each snapshot.
    patch_count_history: list[int] = field(default_factory=list)
    # Shannon entropy of the mass distribution over a coarse grid, per snapshot.
    mass_spatial_entropy_history: list[float] = field(default_factory=list)
    # Patch sizes (pixel counts, descending) at the first snapshot.
    # Baseline for cannibalism detection.
    initial_patch_sizes: list[int] = field(default_factory=list)
    # Patch sizes (descending) at each snapshot. Shape: (n_snapshots, n_patches).
    patch_size_history: list[list[int]] = field(default_factory=list)


def _label(binary_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Thin wrapper around scipy.ndimage.label with explicit return types.

    scipy's stubs are incomplete; pyright infers the count as Literal[0|1]
    rather than int, which breaks range(n). This wrapper asserts the correct
    types so the rest of the module stays clean.
    """
    result = cast(tuple[np.ndarray, int], _cc_label(binary_mask, structure=_STRUCTURE_4))
    return result[0], result[1]


def patch_count(binary_mask: np.ndarray) -> int:
    """Number of 4-connected components in a boolean or binary 2D array."""
    if not binary_mask.any():
        return 0
    _, n = _label(binary_mask)
    return n


def patch_sizes(binary_mask: np.ndarray) -> list[int]:
    """Pixel count of each 4-connected component in descending order."""
    if not binary_mask.any():
        return []
    labeled, n = _label(binary_mask)
    sizes = [int((labeled == i).sum()) for i in range(1, n + 1)]
    sizes.sort(reverse=True)
    return sizes


def spatial_entropy(field: np.ndarray, n_bins: int = 8) -> float:
    """Shannon entropy of a 2D field's spatial distribution over an n_bins x n_bins grid.

    The field is summed into coarse grid cells. The resulting distribution is
    normalized and entropy is computed, then divided by log(n_bins^2) to put
    it in [0, 1]. Returns 0.0 for fields with no mass.
    """
    h, w = field.shape
    total = float(field.sum())
    if total <= 0.0:
        return 0.0

    bin_h = h // n_bins
    bin_w = w // n_bins
    if bin_h == 0 or bin_w == 0:
        # Grid smaller than coarse bins; treat as single bin.
        return 0.0

    # Crop to a multiple of bin size to avoid partial bins.
    cropped = field[: bin_h * n_bins, : bin_w * n_bins]
    # Reshape to (n_bins, bin_h, n_bins, bin_w) then sum over the bin axes.
    reshaped = cropped.reshape(n_bins, bin_h, n_bins, bin_w)
    coarse = reshaped.sum(axis=(1, 3))  # (n_bins, n_bins)

    flat = coarse.ravel().astype(np.float64)
    flat = flat / flat.sum()
    # Shannon entropy with base e, clipping zeros to avoid log(0).
    nonzero = flat[flat > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    max_entropy = math.log(n_bins * n_bins)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _center_of_mass(field: np.ndarray) -> tuple[float, float] | None:
    """Weighted center of mass of a 2D field. Returns None if total weight is zero."""
    total = float(field.sum())
    if total <= 0.0:
        return None
    h, w = field.shape
    rows = np.arange(h, dtype=np.float64)
    cols = np.arange(w, dtype=np.float64)
    row_com = float((field.sum(axis=1) * rows).sum()) / total
    col_com = float((field.sum(axis=0) * cols).sum()) / total
    return row_com, col_com


def compute_spatial_observables_hetero(
    ownership_snapshots: list[np.ndarray],
    ownership_threshold: float = 0.2,
    entropy_bins: int = 8,
) -> HeteroSpatial:
    """Spatial observables for heterogeneous runs from ownership snapshots.

    Args:
        ownership_snapshots: list of (H, W, S) float32 arrays, one per snapshot.
        ownership_threshold: cells above this weight are considered occupied by
            a species. Used for patch counting and interface area.
        entropy_bins: coarse grid side length for spatial entropy computation.
    """
    if not ownership_snapshots:
        return HeteroSpatial()

    s = ownership_snapshots[0].shape[2]

    # Accumulators indexed by species (outer) then snapshot (inner).
    species_patch_counts: list[list[int]] = [[] for _ in range(s)]
    species_entropies: list[list[float]] = [[] for _ in range(s)]
    interface_area: list[list[list[int]]] = []
    com_distance: list[list[list[float]]] = []
    # contact_occurred[a][b] is True if any snapshot had interface_area > 0 for pair (a,b).
    contact_occurred: list[list[bool]] = [[False] * s for _ in range(s)]

    for own in ownership_snapshots:
        snap_interface: list[list[int]] = [[0] * s for _ in range(s)]
        snap_com_dist: list[list[float]] = [[0.0] * s for _ in range(s)]

        masks = [own[:, :, sp] > ownership_threshold for sp in range(s)]
        coms: list[tuple[float, float] | None] = [_center_of_mass(own[:, :, sp]) for sp in range(s)]

        for sp in range(s):
            species_patch_counts[sp].append(patch_count(masks[sp]))
            species_entropies[sp].append(spatial_entropy(own[:, :, sp], n_bins=entropy_bins))

        for a in range(s):
            for b in range(s):
                if a == b:
                    continue
                # Interface area: cells where both species exceed threshold simultaneously.
                area = int((masks[a] & masks[b]).sum())
                snap_interface[a][b] = area
                if area > 0:
                    contact_occurred[a][b] = True

                # COM distance: Euclidean distance between centers of mass.
                ca, cb = coms[a], coms[b]
                if ca is not None and cb is not None:
                    dist = math.sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2)
                    snap_com_dist[a][b] = dist
                else:
                    snap_com_dist[a][b] = float("inf")

        interface_area.append(snap_interface)
        com_distance.append(snap_com_dist)

    return HeteroSpatial(
        species_patch_count=species_patch_counts,
        species_interface_area=interface_area,
        species_com_distance=com_distance,
        species_spatial_entropy=species_entropies,
        contact_occurred=contact_occurred,
    )


def compute_spatial_observables_homo(
    mass_snapshots: list[np.ndarray],
    mass_threshold_fraction: float = 0.01,
    entropy_bins: int = 8,
) -> HomoSpatial:
    """Spatial observables for homogeneous runs from mass snapshots.

    Args:
        mass_snapshots: list of (H, W) float32 arrays, one per snapshot.
        mass_threshold_fraction: threshold for binarizing the mass field is
            this fraction of the peak mass across all snapshots. Keeps noise
            pixels from inflating patch counts.
        entropy_bins: coarse grid side length for spatial entropy computation.
    """
    if not mass_snapshots:
        return HomoSpatial()

    all_mass = np.stack(mass_snapshots, axis=0)
    peak = float(all_mass.max())
    threshold = peak * mass_threshold_fraction if peak > 0 else 0.0

    patch_count_history: list[int] = []
    entropy_history: list[float] = []
    patch_size_history: list[list[int]] = []

    for mass in mass_snapshots:
        binary = mass > threshold
        patch_count_history.append(patch_count(binary))
        entropy_history.append(spatial_entropy(mass, n_bins=entropy_bins))
        patch_size_history.append(patch_sizes(binary))

    initial_patch_sizes = patch_size_history[0] if patch_size_history else []

    return HomoSpatial(
        patch_count_history=patch_count_history,
        mass_spatial_entropy_history=entropy_history,
        initial_patch_sizes=initial_patch_sizes,
        patch_size_history=patch_size_history,
    )
