"""3D MAP-Elites archive over (speed, size, structure).

Sparse dict of CellCoord -> RolloutResult, with try_insert handling the
filter/quality/similarity gating in one place.

The archive's job:
- Bin a descriptor triple into a discrete 3D cell coordinate.
- Decide whether to accept a new RolloutResult, returning an explicit status.
- Provide uniform-random parent selection for the mutation phase.
- Iterate occupied cells for serialization and visualization.

The cell-binning function is module-level so the rollout function (or anyone
else) can compute cell coordinates without holding an archive instance.
"""

from collections.abc import Iterator
from enum import Enum

import numpy as np

from biota.search.result import CellCoord, Descriptors, RolloutResult

DEFAULT_BINS_SPEED = 32
DEFAULT_BINS_SIZE = 32
DEFAULT_BINS_STRUCTURE = 16
DEFAULT_SIMILARITY_EPSILON = 0.05
DEFAULT_QUALITY_OVERRIDE_FACTOR = 1.2


class InsertionStatus(Enum):
    """Outcome of try_insert for one rollout result."""

    INSERTED = "inserted"
    """Cell was empty and no neighbors were too similar. Stored."""

    REPLACED = "replaced"
    """Cell was occupied but the new result has higher quality. Replaced."""

    REJECTED_FILTER = "rejected_filter"
    """Quality is None - the rollout was filtered out (dead, exploded, etc.)."""

    REJECTED_QUALITY = "rejected_quality"
    """Cell was occupied and the new result is no better than the resident."""

    REJECTED_SIMILARITY = "rejected_similarity"
    """Cell was empty but a neighbor cell holds a result too similar to this one,
    and the new candidate's quality isn't high enough to override the rejection."""


def bin_descriptors(
    descriptors: Descriptors,
    bins_speed: int = DEFAULT_BINS_SPEED,
    bins_size: int = DEFAULT_BINS_SIZE,
    bins_structure: int = DEFAULT_BINS_STRUCTURE,
) -> CellCoord:
    """Map a descriptor triple in [0, 1]^3 to a discrete cell coordinate.

    Each axis is binned independently with floor(d * bins), clamped to
    [0, bins-1] so that the d=1.0 edge case maps to the last bin instead of
    one past the end.
    """
    s, a, h = descriptors
    return (
        min(int(s * bins_speed), bins_speed - 1),
        min(int(a * bins_size), bins_size - 1),
        min(int(h * bins_structure), bins_structure - 1),
    )


class Archive:
    """3D MAP-Elites archive over (speed, size, structure).

    Stores at most one RolloutResult per cell. Insertion enforces three
    things: filter rejections never enter, occupied-cell collisions resolve
    by quality, and empty-cell insertions are gated by similarity to nearby
    occupied cells.
    """

    def __init__(
        self,
        bins_speed: int = DEFAULT_BINS_SPEED,
        bins_size: int = DEFAULT_BINS_SIZE,
        bins_structure: int = DEFAULT_BINS_STRUCTURE,
        similarity_epsilon: float = DEFAULT_SIMILARITY_EPSILON,
        quality_override_factor: float = DEFAULT_QUALITY_OVERRIDE_FACTOR,
    ) -> None:
        self.bins_speed = bins_speed
        self.bins_size = bins_size
        self.bins_structure = bins_structure
        self.similarity_epsilon = similarity_epsilon
        self.quality_override_factor = quality_override_factor
        self._cells: dict[CellCoord, RolloutResult] = {}

    def __len__(self) -> int:
        return len(self._cells)

    def __contains__(self, coord: CellCoord) -> bool:
        return coord in self._cells

    def __getitem__(self, coord: CellCoord) -> RolloutResult:
        return self._cells[coord]

    @property
    def total_cells(self) -> int:
        """Total number of cells in the archive (occupied or not)."""
        return self.bins_speed * self.bins_size * self.bins_structure

    @property
    def fill_fraction(self) -> float:
        """Occupied cells divided by total cells."""
        return len(self) / self.total_cells

    def cell_for(self, descriptors: Descriptors) -> CellCoord:
        """Bin a descriptor triple to a cell using this archive's resolution."""
        return bin_descriptors(descriptors, self.bins_speed, self.bins_size, self.bins_structure)

    def iter_occupied(self) -> Iterator[tuple[CellCoord, RolloutResult]]:
        return iter(self._cells.items())

    def try_insert(self, result: RolloutResult) -> InsertionStatus:
        """Apply the gating rules and either store the result or reject it."""
        if result.quality is None or result.descriptors is None:
            return InsertionStatus.REJECTED_FILTER

        coord = self.cell_for(result.descriptors)

        # Occupied-cell collision: replace iff strictly higher quality
        if coord in self._cells:
            existing = self._cells[coord]
            assert existing.quality is not None  # invariant: stored results are accepted
            if result.quality > existing.quality:
                self._cells[coord] = result
                return InsertionStatus.REPLACED
            return InsertionStatus.REJECTED_QUALITY

        # Empty cell: check 3D Moore-neighborhood occupied cells for similarity
        if self._is_too_similar(coord, result):
            return InsertionStatus.REJECTED_SIMILARITY

        self._cells[coord] = result
        return InsertionStatus.INSERTED

    def random_parent(self, rng: np.random.Generator) -> tuple[CellCoord, RolloutResult]:
        """Pick a uniformly random occupied cell. Raises IndexError if empty."""
        if not self._cells:
            raise IndexError("cannot sample parent from empty archive")
        coords = list(self._cells.keys())
        idx = int(rng.integers(0, len(coords)))
        coord = coords[idx]
        return coord, self._cells[coord]

    # === private ===

    def _is_too_similar(self, coord: CellCoord, candidate: RolloutResult) -> bool:
        assert candidate.descriptors is not None
        assert candidate.quality is not None
        epsilon = self.similarity_epsilon
        epsilon_sq = epsilon * epsilon

        for neighbor_coord in self._neighbors(coord):
            if neighbor_coord not in self._cells:
                continue
            neighbor = self._cells[neighbor_coord]
            assert neighbor.descriptors is not None
            assert neighbor.quality is not None

            dist_sq = sum(
                (a - b) ** 2
                for a, b in zip(candidate.descriptors, neighbor.descriptors, strict=True)
            )
            if dist_sq < epsilon_sq:
                # Override the similarity rejection if the candidate is much better
                if candidate.quality > self.quality_override_factor * neighbor.quality:
                    continue
                return True
        return False

    def _neighbors(self, coord: CellCoord) -> Iterator[CellCoord]:
        """26-neighborhood (3D Moore neighborhood) around coord, excluding coord itself.

        Yields only coordinates that are inside the archive grid.
        """
        cy, cs, ch = coord
        for dy in (-1, 0, 1):
            for ds in (-1, 0, 1):
                for dh in (-1, 0, 1):
                    if dy == 0 and ds == 0 and dh == 0:
                        continue
                    ny, ns, nh = cy + dy, cs + ds, ch + dh
                    if not (0 <= ny < self.bins_speed):
                        continue
                    if not (0 <= ns < self.bins_size):
                        continue
                    if not (0 <= nh < self.bins_structure):
                        continue
                    yield (ny, ns, nh)
