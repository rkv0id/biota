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
    emission_rate:    Scalar in [0.001, 0.05]. Fraction of positive growth
                      activity converted to signal per step.
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
"""Three normalized behavior descriptors: (velocity, gyradius, dgm), each in
[0, 1]. The tuple positions are unchanged from the M1 (speed, size, structure)
design but the underlying observables and normalizers were replaced as part of
the descriptor rework. See descriptors.py for the
new design."""

CellCoord = tuple[int, int, int]
"""Discrete archive cell coordinate. Tuple positions correspond to the three
descriptors in Descriptors above. Field names elsewhere in the codebase still
use the historical (speed, size, structure) names; rename is deferred to a
follow-up cleanup."""


# === RolloutResult ===


@dataclass(frozen=True)
class RolloutResult:
    """Outcome of one Flow-Lenia rollout.

    Attributes:
        params: The parameter dict that produced this rollout.
        seed: The integer seed used for the initial state.
        descriptors: The (velocity, gyradius, dgm) tuple in [0, 1]^3, or None
            if the rollout failed early enough that descriptors couldn't be
            computed (e.g. mass collapsed to zero). Tuple positions are
            unchanged from M1 but the underlying observables were replaced;
            see descriptors.py.
        quality: The quality score in [0, 1], or None if the rollout was
            rejected by a filter (or descriptors are None).
        rejection_reason: Short human-readable reason if quality is None.
            Examples: "dead", "exploded", "unstable", "no_descriptors". None if
            the rollout passed all filters.
        thumbnail: 16-frame 32x32 grayscale animation for the dashboard atlas.
            Shape (16, 32, 32), dtype uint8. Captures the rollout's final
            stretch (steps T-15..T downsampled).
        parent_cell: The 3D archive coordinate this rollout was mutated from,
            or None if it came from the initial random phase.
        created_at: Unix timestamp (seconds since epoch) when the result was
            constructed on the worker. Used for events log ordering.
        compute_seconds: Wall-clock time the rollout took on the worker, from
            params-in to result-out. Used for the metrics tab and benchmarks.
    """

    params: ParamDict
    seed: int
    descriptors: Descriptors | None
    quality: float | None
    rejection_reason: str | None
    thumbnail: np.ndarray
    parent_cell: CellCoord | None
    created_at: float
    compute_seconds: float

    @property
    def accepted(self) -> bool:
        """True if this rollout passed all filters and has a quality score."""
        return self.quality is not None
