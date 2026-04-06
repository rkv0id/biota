from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Config:
    grid: int = 96
    kernels: int = 10
    dd: int = 5
    dt: float = 0.2
    sigma: float = 0.65
    border: str = "wall"


@dataclass(frozen=True)
class Params:
    R: float
    r: torch.Tensor
    m: torch.Tensor
    s: torch.Tensor
    h: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    w: torch.Tensor


class FlowLenia:
    def __init__(self, config: Config, params: Params, device: str = "cpu") -> None:
        self.config = config
        self.params = params
        self.device = device

    def step(self, A: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("FlowLenia.step lands in M0 implementation")

    def rollout(self, A0: torch.Tensor, steps: int) -> torch.Tensor:
        A = A0
        for _ in range(steps):
            A = self.step(A)
        return A

    def rollout_with_mass(self, A0: torch.Tensor, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        masses = torch.empty(steps + 1, dtype=A0.dtype)
        masses[0] = A0.sum()
        A = A0
        for i in range(steps):
            A = self.step(A)
            masses[i + 1] = A.sum()
        return A, masses
