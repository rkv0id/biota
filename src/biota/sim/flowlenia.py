"""Single-channel Flow-Lenia in PyTorch.

Ports the JAX reference implementation (erwanplantec/FlowLenia) with the
following deliberate scope reductions:

- single channel only (C = 1)
- one connectivity matrix (all kernels feed the single channel)
- wall border only

The reference uses a few parameterization choices that differ from Plantec
2022 Table 1. We follow the reference, not the paper:

- kernel scale is `(R + 15) * r[k]`, not `R * r[k]`
- ring function is `b * exp(-(D - a)^2 / w)`, not `b * exp(-(D - a)^2 / (2 w^2))`
- alpha is hardcoded as `clip(A^2, 0, 1)` (theta_A = 1, n = 2)
- parameter ranges are tightened (h >= 0.01, b >= 0.001, s_growth <= 0.18)

Mass conservation matches the JAX reference's float32 floor of ~1.26e-4 over
1000 steps at the dev preset (96x96, dd=5, dt=0.2, sigma=0.65).
"""

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F


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
    r: torch.Tensor  # (k,)
    m: torch.Tensor  # (k,) growth function center
    s: torch.Tensor  # (k,) growth function width
    h: torch.Tensor  # (k,) per-kernel weight
    a: torch.Tensor  # (k, 3) ring centers
    b: torch.Tensor  # (k, 3) ring weights
    w: torch.Tensor  # (k, 3) ring widths


class FlowLenia:
    def __init__(self, config: Config, params: Params, device: str = "cpu") -> None:
        if config.border not in ("wall", "torus"):
            raise ValueError(f"border must be 'wall' or 'torus', got {config.border!r}")
        self.config = config
        self.device = device
        self.params = Params(
            R=params.R,
            r=params.r.to(device),
            m=params.m.to(device),
            s=params.s.to(device),
            h=params.h.to(device),
            a=params.a.to(device),
            b=params.b.to(device),
            w=params.w.to(device),
        )
        self._fK = self._build_kernels_fft()
        self._sobel_kx, self._sobel_ky = self._build_sobel_kernels()
        self._pos = self._build_position_grid()

    def step(self, A: torch.Tensor) -> torch.Tensor:
        A2 = A[:, :, 0]

        # Lenia: convolve, growth function, sum over kernels
        fA = torch.fft.fft2(A2)
        U = torch.fft.ifft2(self._fK * fA.unsqueeze(0)).real  # (k, X, Y)
        m = self.params.m.view(-1, 1, 1)
        s = self.params.s.view(-1, 1, 1)
        h = self.params.h.view(-1, 1, 1)
        G = (torch.exp(-(((U - m) / s) ** 2) / 2.0) * 2.0 - 1.0) * h
        U_sum = G.sum(dim=0)

        # Sobel gradients of growth and density
        nabla_U = self._sobel(U_sum)
        nabla_A = self._sobel(A2)

        # Affinity (single channel: alpha = clip(A^2, 0, 1))
        alpha = torch.clamp(A2**2, 0.0, 1.0)

        # Flow
        F_flow = nabla_U * (1.0 - alpha) - nabla_A * alpha

        # Reintegration tracking
        new_A2 = self._reintegration(A2, F_flow)
        return new_A2.unsqueeze(-1)

    def step_batch(self, A: torch.Tensor) -> torch.Tensor:
        """Vectorized step over a batch of states.

        A: (B, H, W, 1) -> (B, H, W, 1)

        Runs B Flow-Lenia simulations in one forward pass. All operations
        broadcast over the leading batch dimension. Memory scales linearly
        with B; at standard preset (192x192) each element is ~150KB so B=128
        fits comfortably in 16GB VRAM.
        """
        B = A.shape[0]
        A2 = A[:, :, :, 0]  # (B, H, W)

        # Convolve each batch element with all K kernels.
        # fA: (B, H, W) -> unsqueeze to (B, 1, H, W)
        # _fK: (K, H, W) -> unsqueeze to (1, K, H, W)
        # product: (B, K, H, W); ifft2 over last two dims
        fA = torch.fft.fft2(A2)
        U = torch.fft.ifft2(
            self._fK.unsqueeze(0) * fA.unsqueeze(1), dim=(-2, -1)
        ).real  # (B, K, H, W)

        m = self.params.m.view(1, -1, 1, 1)
        s = self.params.s.view(1, -1, 1, 1)
        h = self.params.h.view(1, -1, 1, 1)
        G = (torch.exp(-(((U - m) / s) ** 2) / 2.0) * 2.0 - 1.0) * h
        U_sum = G.sum(dim=1)  # (B, H, W)

        # Sobel over the batch: reshape to (B, 1, H, W) for F.conv2d
        nabla_U = self._sobel_batch(U_sum)  # (B, 2, H, W)
        nabla_A = self._sobel_batch(A2)  # (B, 2, H, W)

        alpha = torch.clamp(A2**2, 0.0, 1.0)  # (B, H, W)

        # Flow: (B, 2, H, W)
        alpha4 = alpha.unsqueeze(1)
        F_flow = nabla_U * (1.0 - alpha4) - nabla_A * alpha4

        new_A2 = self._reintegration_batch(A2, F_flow, B)  # (B, H, W)
        return new_A2.unsqueeze(-1)

    def rollout(self, A0: torch.Tensor, steps: int) -> torch.Tensor:
        A = A0
        for _ in range(steps):
            A = self.step(A)
        return A

    def kernel_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the pre-built simulation tensors needed for batched execution.

        Returns (fK, sobel_kx, sobel_ky, pos) where:
          fK:       (K, H, W) complex - Fourier-space kernels for this param set
          sobel_kx: (1, 1, 3, 3) - x-Sobel kernel (param-independent)
          sobel_ky: (1, 1, 3, 3) - y-Sobel kernel (param-independent)
          pos:      (2, H, W)    - position grid (param-independent)

        Used by rollout_batch to extract per-element fK tensors and borrow the
        shared param-independent tensors without private attribute access.
        """
        return self._fK, self._sobel_kx, self._sobel_ky, self._pos

    def rollout_with_mass(self, A0: torch.Tensor, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        masses = torch.empty(steps + 1, dtype=A0.dtype)
        masses[0] = A0.sum()
        A = A0
        for i in range(steps):
            A = self.step(A)
            masses[i + 1] = A.sum()
        return A, masses

    # === private helpers ===

    def _build_kernels_fft(self) -> torch.Tensor:
        """Build the Fourier-space kernels following the reference parameterization.

        Returns a complex tensor of shape (k, X, Y).
        """
        cfg = self.config
        p = self.params
        mid = cfg.grid // 2

        coords = torch.arange(-mid, mid, device=self.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        radius = torch.sqrt(xx**2 + yy**2)  # (X, X)

        # Per-kernel scale; the +15 is from the reference, not the paper
        scale = (p.R + 15.0) * p.r  # (k,)
        D = radius.unsqueeze(0) / scale.view(-1, 1, 1)  # (k, X, X)

        # Smooth indicator that cuts the kernel off where D > 1
        mask = torch.sigmoid(-(D - 1.0) * 10.0)

        # Ring function: sum over 3 rings of b * exp(-(D - a)^2 / w)
        ring_arg = D.unsqueeze(-1) - p.a.view(-1, 1, 1, 3)
        ring = (p.b.view(-1, 1, 1, 3) * torch.exp(-(ring_arg**2) / p.w.view(-1, 1, 1, 3))).sum(
            dim=-1
        )

        K = mask * ring
        nK = K / K.sum(dim=(1, 2), keepdim=True)

        # FFT shift then FFT, matching the reference's get_kernels_fft
        nK_shifted = torch.fft.fftshift(nK, dim=(1, 2))
        return cast(torch.Tensor, torch.fft.fft2(nK_shifted, dim=(1, 2)))

    def _build_sobel_kernels(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sobel kernels, flipped along both spatial axes to match JAX's
        jax.scipy.signal.convolve2d (true convolution, not cross-correlation).
        Returned in (1, 1, 3, 3) shape ready for F.conv2d.
        """
        kx = torch.tensor(
            [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
            device=self.device,
        )
        ky = kx.t()
        kx_flipped = torch.flip(kx, dims=(0, 1)).unsqueeze(0).unsqueeze(0)
        ky_flipped = torch.flip(ky, dims=(0, 1)).unsqueeze(0).unsqueeze(0)
        return kx_flipped, ky_flipped

    def _build_position_grid(self) -> torch.Tensor:
        """Position grid for reintegration tracking. Each cell holds (y + 0.5,
        x + 0.5), matching the reference's `pos = jnp.dstack((Y, X)) + .5`.
        Shape (2, X, Y).
        """
        coords = torch.arange(self.config.grid, device=self.device, dtype=torch.float32) + 0.5
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        return torch.stack([yy, xx], dim=0)

    def _sobel(self, field: torch.Tensor) -> torch.Tensor:
        """Sobel gradients of a 2D field. Returns (gy, gx) stacked, shape (2, X, Y).
        Uses zero-padding to match jax.scipy.signal.convolve2d(mode='same').
        """
        field_4d = field.unsqueeze(0).unsqueeze(0)
        gx = F.conv2d(field_4d, self._sobel_kx, padding=1)[0, 0]
        gy = F.conv2d(field_4d, self._sobel_ky, padding=1)[0, 0]
        return torch.stack([gy, gx], dim=0)

    def _sobel_batch(self, field: torch.Tensor) -> torch.Tensor:
        """Sobel gradients over a batch of 2D fields.

        field: (B, H, W) -> (B, 2, H, W) stacked as [gy, gx].

        F.conv2d treats the first dimension as the batch, so we reshape
        to (B, 1, H, W), run both kernels, and stack the gradient channels.
        """
        B = field.shape[0]
        field_4d = field.unsqueeze(1)  # (B, 1, H, W)
        gx = F.conv2d(field_4d, self._sobel_kx, padding=1)[:, 0]  # (B, H, W)
        gy = F.conv2d(field_4d, self._sobel_ky, padding=1)[:, 0]  # (B, H, W)
        _ = B  # used implicitly via field_4d shape
        return torch.stack([gy, gx], dim=1)  # (B, 2, H, W)

    def _reintegration(self, A: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Pull-pattern reintegration tracking via (2*dd+1)^2 vectorized rolls.

        For each integer offset (dx, dy) in [-dd, dd]^2, roll the source state
        and target positions by (dx, dy), compute the closed-form box overlap
        between the cell at this position and the rolled source's diffused box,
        and accumulate. Mass conservation is exact modulo float roundoff
        because the per-source overlaps tile the destination plane.

        Wall border: clamp mu to [sigma, grid-sigma] so mass stays inside.
        Torus border: wrap mu via modulo so mass crossing one edge reappears
        on the opposite edge. The distance dpmu uses the shortest-path
        (minimum image) convention so the overlap area is computed correctly
        across the seam.

        A: (X, Y), flow: (2, X, Y) -> (X, Y)
        """
        cfg = self.config
        dd = cfg.dd
        sigma = cfg.sigma
        ma = dd - sigma

        clipped_flow = torch.clamp(cfg.dt * flow, -ma, ma)
        mu = self._pos + clipped_flow

        mu = mu % cfg.grid if cfg.border == "torus" else torch.clamp(mu, sigma, cfg.grid - sigma)

        new_A = torch.zeros_like(A)
        sigma_clip_max = min(1.0, 2.0 * sigma)
        denom = 4.0 * sigma * sigma

        for dx in range(-dd, dd + 1):
            for dy in range(-dd, dd + 1):
                Ar = torch.roll(A, shifts=(dx, dy), dims=(0, 1))
                mur = torch.roll(mu, shifts=(dx, dy), dims=(1, 2))
                dpmu = torch.abs(self._pos - mur)  # (2, X, Y)
                if cfg.border == "torus":
                    # Minimum image convention: shortest distance across seam
                    dpmu = torch.minimum(dpmu, cfg.grid - dpmu)
                sz = 0.5 - dpmu + sigma
                sz_clipped = torch.clamp(sz, 0.0, sigma_clip_max)
                area = (sz_clipped[0] * sz_clipped[1]) / denom
                new_A = new_A + Ar * area

        return new_A

    def _reintegration_batch(self, A: torch.Tensor, flow: torch.Tensor, B: int) -> torch.Tensor:
        """Batch reintegration tracking.

        A: (B, H, W), flow: (B, 2, H, W) -> (B, H, W)

        Same pull-pattern logic as _reintegration but vectorized over the
        leading batch dimension. _pos (2, H, W) is unsqueezed to (1, 2, H, W)
        to broadcast against (B, 2, H, W). Roll operations shift the two
        spatial dims (1, 2) for A and (2, 3) for mu.

        Torus border wraps mu via modulo and uses minimum image convention
        for dpmu; wall border clamps as before.
        """
        cfg = self.config
        dd = cfg.dd
        sigma = cfg.sigma
        ma = dd - sigma

        pos = self._pos.unsqueeze(0)  # (1, 2, H, W) broadcasts over B

        clipped_flow = torch.clamp(cfg.dt * flow, -ma, ma)  # (B, 2, H, W)
        mu = pos + clipped_flow

        mu = (
            mu % cfg.grid if cfg.border == "torus" else torch.clamp(mu, sigma, cfg.grid - sigma)
        )  # (B, 2, H, W)

        new_A = torch.zeros_like(A)
        sigma_clip_max = min(1.0, 2.0 * sigma)
        denom = 4.0 * sigma * sigma

        for dx in range(-dd, dd + 1):
            for dy in range(-dd, dd + 1):
                Ar = torch.roll(A, shifts=(dx, dy), dims=(1, 2))  # (B, H, W)
                mur = torch.roll(mu, shifts=(dx, dy), dims=(2, 3))  # (B, 2, H, W)
                dpmu = torch.abs(pos - mur)  # (B, 2, H, W)
                if cfg.border == "torus":
                    dpmu = torch.minimum(dpmu, cfg.grid - dpmu)
                sz = 0.5 - dpmu + sigma
                sz_clipped = torch.clamp(sz, 0.0, sigma_clip_max)
                area = (sz_clipped[:, 0] * sz_clipped[:, 1]) / denom  # (B, H, W)
                new_A = new_A + Ar * area

        _ = B  # shape enforced by A; explicit parameter aids callers' type clarity
        return new_A
