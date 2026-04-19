"""CVT-MAP-Elites archive over behavioral descriptor space.

Replaces the v3.x fixed 32x32x16 grid with a Voronoi tessellation fitted
from observed rollout distributions. Archive cells are Voronoi regions around
k-means centroids; nearest-centroid lookup via cKDTree replaces integer binning.

The core failure mode this fixes: the fixed grid assumed known descriptor bounds
set by hand. Any descriptor outside the default (velocity, gyradius,
spectral_entropy) produced degenerate archives because the hardcoded normalizers
didn't match actual rollout distributions. CVT needs no normalizers -- centroids
are placed where creatures actually live in descriptor space.

Calibration (fitting centroids) is a separate phase handled in loop.py. The
archive receives an already-fitted centroid array via attach_centroids() before
any rollouts are inserted. Inserting before attach_centroids() raises RuntimeError.

Similarity rejection uses distance to the nearest occupied centroid rather than
the 26-neighbor Moore neighborhood check from v3.x. Semantics are the same;
O(log k) cKDTree query replaces the fixed-neighbor scan.
"""

from collections.abc import Iterator
from enum import Enum
from typing import Any

import numpy as np
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from biota.search.result import CellCoord, Descriptors, RolloutResult

DEFAULT_N_CENTROIDS = 1024
DEFAULT_DESCRIPTOR_NAMES: tuple[str, str, str] = ("velocity", "gyradius", "spectral_entropy")

# Similarity rejection: a candidate whose nearest occupied centroid is within
# this Euclidean distance in raw descriptor space is rejected unless its quality
# clears the override factor. Value calibrated against observed descriptor spreads
# at standard preset; CVT centroids are roughly sqrt(fill * range^2 / N) apart,
# so this is intentionally loose.
DEFAULT_SIMILARITY_EPSILON = 1.0
DEFAULT_QUALITY_OVERRIDE_FACTOR = 1.2


class InsertionStatus(Enum):
    """Outcome of try_insert for one rollout result."""

    INSERTED = "inserted"
    """Cell was empty. Stored."""

    REPLACED = "replaced"
    """Cell was occupied but the new result has strictly higher quality. Replaced."""

    REJECTED_FILTER = "rejected_filter"
    """Quality is None -- the rollout was filtered out (dead, exploded, etc.)."""

    REJECTED_QUALITY = "rejected_quality"
    """Cell was occupied and the new result is no better than the resident."""

    REJECTED_SIMILARITY = "rejected_similarity"
    """Nearest occupied centroid holds a result too similar to this candidate,
    and the candidate's quality isn't high enough to override the rejection."""


class Archive:
    """CVT-MAP-Elites archive over three behavioral descriptor axes.

    Stores at most one RolloutResult per centroid (Voronoi cell). Centroids are
    fitted from observed rollout distributions during a calibration phase and
    attached via attach_centroids() before search begins.

    descriptor_names records which three descriptors were active when this
    archive was created. The renderer reads this to display correct axis labels.
    """

    def __init__(
        self,
        n_centroids: int = DEFAULT_N_CENTROIDS,
        similarity_epsilon: float = DEFAULT_SIMILARITY_EPSILON,
        quality_override_factor: float = DEFAULT_QUALITY_OVERRIDE_FACTOR,
        descriptor_names: tuple[str, str, str] = DEFAULT_DESCRIPTOR_NAMES,
    ) -> None:
        self.n_centroids = n_centroids
        self.similarity_epsilon = similarity_epsilon
        self.quality_override_factor = quality_override_factor
        self.descriptor_names = descriptor_names

        # Set by attach_centroids() after calibration.
        self._centroids: np.ndarray | None = None  # shape (n_centroids, 3)
        self._tree: Any = None  # cKDTree after calibration
        self._cells: dict[CellCoord, RolloutResult] = {}

    def __len__(self) -> int:
        return len(self._cells)

    def __contains__(self, idx: CellCoord) -> bool:
        return idx in self._cells

    def __getitem__(self, idx: CellCoord) -> RolloutResult:
        return self._cells[idx]

    @property
    def calibrated(self) -> bool:
        return self._centroids is not None

    @property
    def centroid_positions(self) -> np.ndarray:
        """Centroid array of shape (n_centroids, 3). Raises if not calibrated."""
        if self._centroids is None:
            raise RuntimeError("archive has not been calibrated yet")
        return self._centroids

    @property
    def total_cells(self) -> int:
        return self.n_centroids

    @property
    def fill_fraction(self) -> float:
        return len(self) / self.n_centroids

    def attach_centroids(self, centroids: np.ndarray) -> None:
        """Attach fitted centroid positions and build the lookup tree.

        centroids must be shape (k, 3) float32/float64. Called once by
        loop.py after the calibration phase completes. Rebuilding is allowed
        (e.g. after loading a checkpoint) but any existing cells remain.
        """
        if centroids.ndim != 2 or centroids.shape[1] != 3:
            raise ValueError(f"centroids must be shape (k, 3), got {centroids.shape}")
        self._centroids = centroids.astype(np.float64)
        self._tree = cKDTree(self._centroids)
        self.n_centroids = len(centroids)

    def cell_for(self, descriptors: Descriptors) -> CellCoord:
        """Return the centroid index nearest to this descriptor triple.

        Raises RuntimeError if the archive has not been calibrated.
        """
        if self._tree is None:
            raise RuntimeError("archive has not been calibrated yet")
        pt = np.array(descriptors, dtype=np.float64)
        _, idx = self._tree.query(pt)
        return int(idx)

    def iter_occupied(self) -> Iterator[tuple[CellCoord, RolloutResult]]:
        return iter(self._cells.items())

    def try_insert(self, result: RolloutResult) -> InsertionStatus:
        """Apply gating rules and either store the result or reject it."""
        if result.quality is None or result.descriptors is None:
            return InsertionStatus.REJECTED_FILTER

        idx = self.cell_for(result.descriptors)

        if idx in self._cells:
            existing = self._cells[idx]
            assert existing.quality is not None
            if result.quality > existing.quality:
                self._cells[idx] = result
                return InsertionStatus.REPLACED
            return InsertionStatus.REJECTED_QUALITY

        if self._is_too_similar(result):
            return InsertionStatus.REJECTED_SIMILARITY

        self._cells[idx] = result
        return InsertionStatus.INSERTED

    def random_parent(self, rng: np.random.Generator) -> tuple[CellCoord, RolloutResult]:
        """Pick a uniformly random occupied cell. Raises IndexError if empty."""
        if not self._cells:
            raise IndexError("cannot sample parent from empty archive")
        idxs = list(self._cells.keys())
        chosen = int(rng.integers(0, len(idxs)))
        idx = idxs[chosen]
        return idx, self._cells[idx]

    # === private ===

    def _is_too_similar(self, candidate: RolloutResult) -> bool:
        """True if any occupied centroid is within similarity_epsilon of candidate.

        Queries only the occupied subset. If fewer than 2 cells are occupied
        the similarity check is skipped (nothing meaningful to compare against).
        Override: if candidate quality exceeds quality_override_factor times the
        neighbor quality, the rejection is waived.
        """
        assert candidate.descriptors is not None
        assert candidate.quality is not None

        if len(self._cells) < 1:
            return False

        if self._tree is None:
            return False

        occupied_idxs = list(self._cells.keys())
        occupied_centroids = self._centroids[occupied_idxs]  # type: ignore[index]
        occupied_tree = cKDTree(occupied_centroids)

        pt = np.array(candidate.descriptors, dtype=np.float64)
        dist, local_idx = occupied_tree.query(pt)

        if dist >= self.similarity_epsilon:
            return False

        neighbor = self._cells[occupied_idxs[int(local_idx)]]
        assert neighbor.quality is not None
        return candidate.quality <= self.quality_override_factor * neighbor.quality
