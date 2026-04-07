"""Parameter sampler for the search loop.

Two functions:

- sample_random(kernels, seed): draw a fresh ParamDict from the uniform prior
- mutate(parent, seed): perturb a parent ParamDict with Gaussian noise

Parameter ranges follow the JAX reference (erwanplantec/FlowLenia)'s __init__,
not paper Table 1: tighter on h, b, and s_growth to avoid degenerate kernels.
Mutation sigma is 10% of each parameter's range width per DECISIONS.md.

Returns ParamDict (TypedDict of plain Python primitives) rather than the
Params dataclass from biota.sim.flowlenia. The rollout function (step 6)
converts ParamDict -> Params on the worker's device.
"""

from dataclasses import dataclass

import numpy as np

from biota.search.result import ParamDict


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    """Field name in the ParamDict."""

    shape: str
    """One of 'scalar', 'k', 'k3'. Defines how the random sample is shaped."""

    low: float
    high: float
    sigma: float

    @property
    def width(self) -> float:
        return self.high - self.low


# Source of truth: ranges and sigmas. Used by in_range and as documentation
# for the per-field sampling/mutation logic below.
PARAMETER_SPECS: tuple[ParameterSpec, ...] = (
    ParameterSpec("R", "scalar", 2.0, 25.0, 2.3),
    ParameterSpec("r", "k", 0.2, 1.0, 0.08),
    ParameterSpec("m", "k", 0.05, 0.5, 0.045),
    ParameterSpec("s", "k", 0.001, 0.18, 0.0179),
    ParameterSpec("h", "k", 0.01, 1.0, 0.099),
    ParameterSpec("a", "k3", 0.0, 1.0, 0.1),
    ParameterSpec("b", "k3", 0.001, 1.0, 0.0999),
    ParameterSpec("w", "k3", 0.01, 0.5, 0.049),
)

_SPECS_BY_NAME: dict[str, ParameterSpec] = {spec.name: spec for spec in PARAMETER_SPECS}


def _uniform_scalar(rng: np.random.Generator, name: str) -> float:
    spec = _SPECS_BY_NAME[name]
    return float(rng.uniform(spec.low, spec.high))


def _uniform_k(rng: np.random.Generator, name: str, kernels: int) -> list[float]:
    spec = _SPECS_BY_NAME[name]
    return rng.uniform(spec.low, spec.high, size=kernels).tolist()


def _uniform_k3(rng: np.random.Generator, name: str, kernels: int) -> list[list[float]]:
    spec = _SPECS_BY_NAME[name]
    return rng.uniform(spec.low, spec.high, size=(kernels, 3)).tolist()


def sample_random(kernels: int = 10, seed: int = 0) -> ParamDict:
    """Draw a fresh ParamDict uniformly from the JAX-reference parameter prior.

    Same seed -> same output, regardless of host.
    """
    rng = np.random.default_rng(seed)
    return ParamDict(
        R=_uniform_scalar(rng, "R"),
        r=_uniform_k(rng, "r", kernels),
        m=_uniform_k(rng, "m", kernels),
        s=_uniform_k(rng, "s", kernels),
        h=_uniform_k(rng, "h", kernels),
        a=_uniform_k3(rng, "a", kernels),
        b=_uniform_k3(rng, "b", kernels),
        w=_uniform_k3(rng, "w", kernels),
    )


def _perturb_scalar(rng: np.random.Generator, name: str, value: float) -> float:
    spec = _SPECS_BY_NAME[name]
    noise = float(rng.normal(0.0, spec.sigma))
    return float(np.clip(value + noise, spec.low, spec.high))


def _perturb_k(rng: np.random.Generator, name: str, value: list[float]) -> list[float]:
    spec = _SPECS_BY_NAME[name]
    arr = np.asarray(value, dtype=np.float64)
    noise = rng.normal(0.0, spec.sigma, size=arr.shape)
    perturbed = np.clip(arr + noise, spec.low, spec.high)
    return perturbed.tolist()


def _perturb_k3(rng: np.random.Generator, name: str, value: list[list[float]]) -> list[list[float]]:
    spec = _SPECS_BY_NAME[name]
    arr = np.asarray(value, dtype=np.float64)
    noise = rng.normal(0.0, spec.sigma, size=arr.shape)
    perturbed = np.clip(arr + noise, spec.low, spec.high)
    return perturbed.tolist()


def mutate(parent: ParamDict, seed: int = 0) -> ParamDict:
    """Return a perturbed copy of parent: per-field Gaussian noise, clipped to range.

    Same parent + same seed -> same child.
    """
    rng = np.random.default_rng(seed)
    return ParamDict(
        R=_perturb_scalar(rng, "R", parent["R"]),
        r=_perturb_k(rng, "r", parent["r"]),
        m=_perturb_k(rng, "m", parent["m"]),
        s=_perturb_k(rng, "s", parent["s"]),
        h=_perturb_k(rng, "h", parent["h"]),
        a=_perturb_k3(rng, "a", parent["a"]),
        b=_perturb_k3(rng, "b", parent["b"]),
        w=_perturb_k3(rng, "w", parent["w"]),
    )


def in_range(params: ParamDict) -> bool:
    """True if every field of params is within its declared range. For tests
    and defensive checks at archive insertion time.
    """
    r_spec = _SPECS_BY_NAME["R"]
    if not (r_spec.low <= params["R"] <= r_spec.high):
        return False

    for name in ("r", "m", "s", "h"):
        spec = _SPECS_BY_NAME[name]
        values: list[float] = params[name]  # type: ignore[literal-required]
        for v in values:
            if v < spec.low or v > spec.high:
                return False

    for name in ("a", "b", "w"):
        spec = _SPECS_BY_NAME[name]
        rows: list[list[float]] = params[name]  # type: ignore[literal-required]
        for row in rows:
            for v in row:
                if v < spec.low or v > spec.high:
                    return False

    return True
