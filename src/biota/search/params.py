"""Parameter sampler for the search loop.

Two functions:

- sample_random(kernels, seed): draw a fresh ParamDict from the uniform prior
- mutate(parent, seed): perturb a parent ParamDict with Gaussian noise

Parameter ranges follow the JAX reference (erwanplantec/FlowLenia)'s __init__,
not paper Table 1: tighter on h, b, and s_growth to avoid degenerate kernels.
Mutation sigma is 10% of each parameter's range width.

Signal field parameters (emission_vector, receptor_profile, signal_kernel_*)
are sampled and mutated only when signal_field=True is passed to sample_random
or mutate. Standard searches omit these keys entirely; the ParamDict type
declares them as optional so the rest of the codebase can gate on their
presence without extra flags.

Returns ParamDict (TypedDict of plain Python primitives) rather than the
Params dataclass from biota.sim.flowlenia. The rollout function (step 6)
converts ParamDict -> Params on the worker's device.
"""

from dataclasses import dataclass

import numpy as np

from biota.search.result import ParamDict

SIGNAL_CHANNELS = 16
"""Number of signal field channels. Fixed at the platform level -- changing
this invalidates all existing signal-enabled archives."""


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    """Field name in the ParamDict."""

    shape: str
    """One of 'scalar', 'k', 'k3', 'c', 'c_sym', 'k3_fixed'.
    'c'       = length-C vector in [low, high].
    'c_sym'   = length-C vector in [low, high] (symmetric range, e.g. [-1, 1]).
    'k3_fixed'= (3,) vector (signal kernel has fixed 3-ring structure, not per-k)."""

    low: float
    high: float
    sigma: float

    @property
    def width(self) -> float:
        return self.high - self.low


# Source of truth: ranges and sigmas for mass-kernel parameters.
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

# Signal field parameter specs. Sampled/mutated only when signal_field=True.
# sigma is ~10% of range width, matching the mass-kernel convention.
SIGNAL_PARAMETER_SPECS: tuple[ParameterSpec, ...] = (
    ParameterSpec("emission_vector", "c", 0.0, 1.0, 0.1),
    ParameterSpec("receptor_profile", "c_sym", -1.0, 1.0, 0.2),
    ParameterSpec("emission_rate", "scalar", 0.001, 0.05, 0.005),
    ParameterSpec("decay_rates", "c", 0.0, 0.9, 0.05),
    ParameterSpec("alpha_coupling", "scalar", -1.0, 1.0, 0.15),
    ParameterSpec("beta_modulation", "scalar", -1.0, 1.0, 0.15),
    ParameterSpec("signal_kernel_r", "scalar", 0.2, 1.0, 0.08),
    ParameterSpec("signal_kernel_a", "k3_fixed", 0.0, 1.0, 0.1),
    ParameterSpec("signal_kernel_b", "k3_fixed", 0.001, 1.0, 0.0999),
    ParameterSpec("signal_kernel_w", "k3_fixed", 0.01, 0.5, 0.049),
)

_SPECS_BY_NAME: dict[str, ParameterSpec] = {
    spec.name: spec for spec in (*PARAMETER_SPECS, *SIGNAL_PARAMETER_SPECS)
}


def _uniform_scalar(rng: np.random.Generator, name: str) -> float:
    spec = _SPECS_BY_NAME[name]
    return float(rng.uniform(spec.low, spec.high))


def _uniform_k(rng: np.random.Generator, name: str, kernels: int) -> list[float]:
    spec = _SPECS_BY_NAME[name]
    return rng.uniform(spec.low, spec.high, size=kernels).tolist()


def _uniform_k3(rng: np.random.Generator, name: str, kernels: int) -> list[list[float]]:
    spec = _SPECS_BY_NAME[name]
    return rng.uniform(spec.low, spec.high, size=(kernels, 3)).tolist()


def _uniform_c(rng: np.random.Generator, name: str) -> list[float]:
    spec = _SPECS_BY_NAME[name]
    return rng.uniform(spec.low, spec.high, size=SIGNAL_CHANNELS).tolist()


def _uniform_k3_fixed(rng: np.random.Generator, name: str) -> list[float]:
    """Sample a (3,) vector for the signal kernel's ring structure."""
    spec = _SPECS_BY_NAME[name]
    return rng.uniform(spec.low, spec.high, size=3).tolist()


def sample_random(kernels: int = 10, seed: int = 0, signal_field: bool = False) -> ParamDict:
    """Draw a fresh ParamDict uniformly from the JAX-reference parameter prior.

    Same seed -> same output, regardless of host.

    When signal_field=True, also samples signal parameters (emission_vector,
    receptor_profile, signal_kernel_*). When False, these keys are absent.
    """
    rng = np.random.default_rng(seed)
    params = ParamDict(
        R=_uniform_scalar(rng, "R"),
        r=_uniform_k(rng, "r", kernels),
        m=_uniform_k(rng, "m", kernels),
        s=_uniform_k(rng, "s", kernels),
        h=_uniform_k(rng, "h", kernels),
        a=_uniform_k3(rng, "a", kernels),
        b=_uniform_k3(rng, "b", kernels),
        w=_uniform_k3(rng, "w", kernels),
    )
    if signal_field:
        params["emission_vector"] = _uniform_c(rng, "emission_vector")
        params["receptor_profile"] = _uniform_c(rng, "receptor_profile")
        params["emission_rate"] = _uniform_scalar(rng, "emission_rate")
        params["decay_rates"] = _uniform_c(rng, "decay_rates")
        params["alpha_coupling"] = _uniform_scalar(rng, "alpha_coupling")
        params["beta_modulation"] = _uniform_scalar(rng, "beta_modulation")
        params["signal_kernel_r"] = _uniform_scalar(rng, "signal_kernel_r")
        params["signal_kernel_a"] = _uniform_k3_fixed(rng, "signal_kernel_a")
        params["signal_kernel_b"] = _uniform_k3_fixed(rng, "signal_kernel_b")
        params["signal_kernel_w"] = _uniform_k3_fixed(rng, "signal_kernel_w")
    return params


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


def _perturb_c(rng: np.random.Generator, name: str, value: list[float]) -> list[float]:
    spec = _SPECS_BY_NAME[name]
    arr = np.asarray(value, dtype=np.float64)
    noise = rng.normal(0.0, spec.sigma, size=arr.shape)
    perturbed = np.clip(arr + noise, spec.low, spec.high)
    return perturbed.tolist()


def _perturb_k3_fixed(rng: np.random.Generator, name: str, value: list[float]) -> list[float]:
    spec = _SPECS_BY_NAME[name]
    arr = np.asarray(value, dtype=np.float64)
    noise = rng.normal(0.0, spec.sigma, size=arr.shape)
    perturbed = np.clip(arr + noise, spec.low, spec.high)
    return perturbed.tolist()


def mutate(parent: ParamDict, seed: int = 0) -> ParamDict:
    """Return a perturbed copy of parent: per-field Gaussian noise, clipped to range.

    Same parent + same seed -> same child.

    Signal parameters are mutated when present in the parent; their presence
    is detected from the ParamDict keys, so no extra flag is needed.
    """
    rng = np.random.default_rng(seed)
    child = ParamDict(
        R=_perturb_scalar(rng, "R", parent["R"]),
        r=_perturb_k(rng, "r", parent["r"]),
        m=_perturb_k(rng, "m", parent["m"]),
        s=_perturb_k(rng, "s", parent["s"]),
        h=_perturb_k(rng, "h", parent["h"]),
        a=_perturb_k3(rng, "a", parent["a"]),
        b=_perturb_k3(rng, "b", parent["b"]),
        w=_perturb_k3(rng, "w", parent["w"]),
    )
    if "emission_vector" in parent:
        child["emission_vector"] = _perturb_c(rng, "emission_vector", parent["emission_vector"])  # type: ignore[typeddict-item]
        child["receptor_profile"] = _perturb_c(rng, "receptor_profile", parent["receptor_profile"])  # type: ignore[typeddict-item]
        child["emission_rate"] = _perturb_scalar(rng, "emission_rate", parent["emission_rate"])  # type: ignore[typeddict-item]
        child["decay_rates"] = _perturb_c(rng, "decay_rates", parent["decay_rates"])  # type: ignore[typeddict-item]
        child["alpha_coupling"] = _perturb_scalar(rng, "alpha_coupling", parent["alpha_coupling"])  # type: ignore[typeddict-item]
        beta_parent: float = parent["beta_modulation"]  # type: ignore[typeddict-item]
        child["beta_modulation"] = _perturb_scalar(rng, "beta_modulation", beta_parent)
        child["signal_kernel_r"] = _perturb_scalar(
            rng,
            "signal_kernel_r",
            parent["signal_kernel_r"],  # type: ignore[typeddict-item]
        )
        child["signal_kernel_a"] = _perturb_k3_fixed(
            rng,
            "signal_kernel_a",
            parent["signal_kernel_a"],  # type: ignore[typeddict-item]
        )
        child["signal_kernel_b"] = _perturb_k3_fixed(
            rng,
            "signal_kernel_b",
            parent["signal_kernel_b"],  # type: ignore[typeddict-item]
        )
        child["signal_kernel_w"] = _perturb_k3_fixed(
            rng,
            "signal_kernel_w",
            parent["signal_kernel_w"],  # type: ignore[typeddict-item]
        )
    return child


def has_signal_field(params: ParamDict) -> bool:
    """True if params contains signal field parameters."""
    return "emission_vector" in params


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

    if has_signal_field(params):
        for name in ("emission_vector", "receptor_profile", "decay_rates"):
            spec = _SPECS_BY_NAME[name]
            for v in params[name]:  # type: ignore[literal-required]
                if v < spec.low or v > spec.high:
                    return False
        er_spec = _SPECS_BY_NAME["emission_rate"]
        if not (er_spec.low <= params["emission_rate"] <= er_spec.high):  # type: ignore[typeddict-item]
            return False
        for coupling_name in ("alpha_coupling", "beta_modulation"):
            c_spec = _SPECS_BY_NAME[coupling_name]
            if not (c_spec.low <= params[coupling_name] <= c_spec.high):  # type: ignore[literal-required]
                return False
        sk_r_spec = _SPECS_BY_NAME["signal_kernel_r"]
        if not (sk_r_spec.low <= params["signal_kernel_r"] <= sk_r_spec.high):  # type: ignore[typeddict-item]
            return False
        for name in ("signal_kernel_a", "signal_kernel_b", "signal_kernel_w"):
            spec = _SPECS_BY_NAME[name]
            for v in params[name]:  # type: ignore[literal-required]
                if v < spec.low or v > spec.high:
                    return False

    return True
