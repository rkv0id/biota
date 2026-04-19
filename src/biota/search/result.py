"""Rollout result: the contract between workers and the driver.

A worker takes parameters, runs a Flow-Lenia rollout, computes descriptors,
applies the quality function, and returns one of these. The result is
checkpoint-safe: no torch tensors, no device-specific state, just plain Python
floats, ints, lists, dicts, and a small numpy thumbnail.

Filter rejections still produce a RolloutResult, with `quality=None` and
`rejection_reason` set. This lets the driver count rejections, display them in
metrics, and surface them in the dashboard without losing the parameters that
were tried.
"""

from dataclasses import dataclass
from typing import TypedDict

import numpy as np

# === type aliases ===


class _SignalParams(TypedDict, total=False):
    """Optional signal field parameters. Present only in signal-enabled archives.

    emission_vector:  (C,) floats in [0, 1]. How the creature distributes
                      emitted signal across the C channels.
    receptor_profile: (C,) floats in [-1, 1]. Dot product with the convolved
                      signal field produces a scalar growth boost. Negative
                      values produce an inhibitory (aversive) response.
    emission_rate:    Scalar in [0.0001, 0.01]. Fraction of positive growth
                      activity converted to signal per step. Kept low to
                      prevent catastrophic mass bleed over 500+ steps.
    decay_rates:      (C,) floats in [0, 0.9]. Per-channel decay rate.
    alpha_coupling:   Scalar in [-1, 1]. Reception-to-growth coupling strength.
                      Positive = chemotaxis (grow into favorable signal gradients,
                      including other species' territory -- enables predation).
                      Negative = chemorepulsion (grow away from signal).
                      Zero = current additive behavior (no cross-species coupling).
    beta_modulation:  Scalar in [-1, 1]. Adaptive emission modulation.
                      Positive = quorum sensing (amplify emission when receiving
                      high signal -- positive chemical feedback).
                      Negative = feedback inhibition (suppress emission under
                      high signal -- self-limiting, stabilizes coexistence).
                      Zero = static emission rate.
    signal_kernel_r:  Kernel radius scale in [0.2, 1.0].
    signal_kernel_a:  (3,) ring centers in [0, 1].
    signal_kernel_b:  (3,) ring weights in [0, 1].
    signal_kernel_w:  (3,) ring widths in [0.01, 0.5].
    """

    emission_vector: list[float]
    receptor_profile: list[float]
    emission_rate: float
    decay_rates: list[float]
    alpha_coupling: float
    beta_modulation: float
    signal_kernel_r: float
    signal_kernel_a: list[float]
    signal_kernel_b: list[float]
    signal_kernel_w: list[float]


class ParamDict(_SignalParams):
    """Parameters as plain Python primitives, suitable for pickling and JSON.

    Mirrors the Params dataclass in biota.sim.flowlenia but without torch
    tensors. Keys map directly to FlowLenia's parameter names. Field shapes
    follow the JAX reference: scalar R, length-k vectors for r/m/s/h,
    (k, 3)-shaped lists for a/b/w.

    Signal field parameters (emission_vector, receptor_profile,
    signal_kernel_*) are optional and only present in archives produced by
    searches run with --signal-field. See _SignalParams for field semantics.
    """

    R: float
    r: list[float]
    m: list[float]
    s: list[float]
    h: list[float]
    a: list[list[float]]
    b: list[list[float]]
    w: list[list[float]]


Descriptors = tuple[float, float, float]
"""Three raw behavioral descriptors returned by compute() functions.

Values are in [0, 100] (loose sanity bound). CVT-MAP-Elites handles scale
implicitly via centroid fitting, so no per-descriptor normalization is applied.
Typical observed ranges are well below 10 for all built-in descriptors; the
100 cap guards against numerical outliers distorting centroid positions.
"""

# v4.0.0: CellCoord is now a centroid index (int). The old tuple form is kept
# as a deprecated alias so that old pickled archives and callers that have not
# been updated yet continue to parse without import errors.
CellCoord = int
"""Centroid index in the CVT archive. Opaque int in [0, N_CENTROIDS).

Replaces the old tuple[int, int, int] grid coordinate from v3.x.
"""

LegacyCellCoord = tuple[int, int, int]
"""Deprecated. The 3D grid coordinate used in v3.x archives.

Present only for backward-compatible loading of old archive.pkl files.
New code must not produce or consume this type.
"""


# === RolloutResult ===


@dataclass(frozen=True)
class RolloutResult:
    """Outcome of one Flow-Lenia rollout.

    Attributes:
        params: The parameter dict that produced this rollout.
        seed: The integer seed used for the initial state.
        creature_id: Stable identity string of the form "{run_id}-{seed}".
            Assigned at insertion time by the driver. Stable across archive
            rebuilds; used in ecosystem YAML, lineage links, and deep-link URLs.
            Empty string on worker-side results before the driver assigns it.
        descriptors: Raw behavioral descriptor triple, each value clipped to
            [0, 100]. None if the rollout failed before descriptors could be
            computed (e.g. mass collapsed to zero).
        quality: Quality score in [0, 1], or None if rejected by a filter.
        rejection_reason: Short human-readable reason if quality is None.
            Examples: "dead", "exploded", "unstable", "no_descriptors". None
            if the rollout passed all filters.
        thumbnail: Animation frames for the atlas. Shape (16, 32, 32), uint8.
        parent_id: creature_id of the parent this rollout was mutated from,
            or None if it came from the calibration or random phase.
        created_at: Unix timestamp when the result was constructed on the
            worker. Used for events log ordering.
        compute_seconds: Wall-clock time the rollout took on the worker.
    """

    params: ParamDict
    seed: int
    creature_id: str
    descriptors: Descriptors | None
    quality: float | None
    rejection_reason: str | None
    thumbnail: np.ndarray
    parent_id: str | None
    created_at: float
    compute_seconds: float

    @property
    def accepted(self) -> bool:
        """True if this rollout passed all filters and has a quality score."""
        return self.quality is not None
