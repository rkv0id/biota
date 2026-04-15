"""Species-indexed Flow-Lenia for heterogeneous ecosystem runs.

This module implements parameter localization at the species level. Each
species keeps its full param set (R, r, m, s, h, a, b, w) and its own
precomputed FFT kernel tensor. At each step:

  1. Convolve mass with each species' kernels independently (S * K FFTs).
  2. Compute each species' growth field using its own m / s / h.
  3. Blend the growth fields by per-cell species ownership weight.
  4. Compute flow from the blended growth field and mass density.
  5. Reintegrate mass and species weights together: weights advect with
     mass, mixing proportionally where mass from different species
     converges. Where mass is zero, weight is zero.

The scalar-path FlowLenia (biota.sim.flowlenia.FlowLenia) is untouched.
Search rollouts and homogeneous ecosystem runs use it. Only heterogeneous
ecosystem runs reach this module.

Position grid and Sobel kernels are shared from the first species'
FlowLenia instance because they are param-independent.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from biota.sim.flowlenia import Config, FlowLenia, Params


@dataclass(frozen=True)
class LocalizedState:
    """Mass and species ownership for a heterogeneous ecosystem run.

    Attributes:
        mass:    (H, W, 1) float32 mass density.
        weights: (H, W, S) float32 species ownership. Where mass > 0, the
                 weight vector at each cell is a probability simplex
                 (non-negative, sums to 1). Where mass == 0, weights are
                 all zero.
    """

    mass: torch.Tensor
    weights: torch.Tensor


class LocalizedFlowLenia:
    """Multi-species Flow-Lenia with weighted growth blending.

    Constructed with one Config and a list of Params (one per species).
    Internally builds one scalar FlowLenia per species so each species'
    fK tensor is computed by the same code path as a search rollout.
    """

    def __init__(
        self,
        config: Config,
        species_params: list[Params],
        device: str = "cpu",
    ) -> None:
        if not species_params:
            raise ValueError("species_params must contain at least one Params")
        self.config = config
        self.device = device
        # One FlowLenia per species. Each owns its precomputed fK and the
        # device-side copy of its Params. Position grid and Sobel kernels are
        # the same across all of them (param-independent), so we just borrow
        # from species 0.
        self._species: list[FlowLenia] = [
            FlowLenia(config, p, device=device) for p in species_params
        ]
        # Stash references to the param-independent tensors built once.
        ref = self._species[0]
        self._sobel_kx, self._sobel_ky = ref._sobel_kx, ref._sobel_ky  # pyright: ignore[reportPrivateUsage]
        self._pos = ref._pos  # pyright: ignore[reportPrivateUsage]

    @property
    def num_species(self) -> int:
        return len(self._species)

    def step(self, state: LocalizedState) -> LocalizedState:
        """Advance one step. Returns a new LocalizedState."""
        A = state.mass[:, :, 0]  # (H, W)
        W = state.weights  # (H, W, S)
        S = self.num_species

        if W.shape[-1] != S:
            raise ValueError(f"weights last dim {W.shape[-1]} does not match species count {S}")
        if W.shape[:2] != A.shape:
            raise ValueError(
                f"weights spatial shape {tuple(W.shape[:2])} does not match mass {tuple(A.shape)}"
            )

        # Compute each species' growth field, blend by species ownership.
        # Where no species is present (weight sum = 0), fall back to uniform
        # weights across all species. Two reasons: (1) single-species runs
        # then reduce exactly to the scalar path (verified by tests), and
        # (2) flow at empty cells should still respond to neighbouring mass
        # via nabla_U, which requires a non-zero growth field outside the
        # currently-occupied region.
        weight_sum = W.sum(dim=-1, keepdim=True)  # (H, W, 1)
        empty_cells = weight_sum < 1e-8
        uniform = torch.full_like(W, 1.0 / S)
        W_eff = torch.where(empty_cells.expand_as(W), uniform, W)

        fA = torch.fft.fft2(A)
        G_blend = torch.zeros_like(A)
        for s, sp in enumerate(self._species):
            fK_s = sp._fK  # pyright: ignore[reportPrivateUsage] - intentional sharing
            U_s = torch.fft.ifft2(fK_s * fA.unsqueeze(0)).real  # (K, H, W)
            m = sp.params.m.view(-1, 1, 1)
            sigma = sp.params.s.view(-1, 1, 1)
            h = sp.params.h.view(-1, 1, 1)
            G_s = (torch.exp(-(((U_s - m) / sigma) ** 2) / 2.0) * 2.0 - 1.0) * h
            G_s_total = G_s.sum(dim=0)  # (H, W)
            G_blend = G_blend + W_eff[:, :, s] * G_s_total

        # Flow: same formulation as scalar path, applied to the blended growth.
        nabla_U = self._sobel(G_blend)
        nabla_A = self._sobel(A)
        alpha = torch.clamp(A**2, 0.0, 1.0)
        F_flow = nabla_U * (1.0 - alpha) - nabla_A * alpha

        new_A, new_W = self._reintegrate(A, W, F_flow)
        return LocalizedState(mass=new_A.unsqueeze(-1), weights=new_W)

    def _sobel(self, field: torch.Tensor) -> torch.Tensor:
        """Sobel gradients of a 2D field. Same code path as scalar FlowLenia."""
        field_4d = field.unsqueeze(0).unsqueeze(0)
        if self.config.border == "torus":
            field_4d = F.pad(field_4d, (1, 1, 1, 1), mode="circular")
            gx = F.conv2d(field_4d, self._sobel_kx, padding=0)[0, 0]
            gy = F.conv2d(field_4d, self._sobel_ky, padding=0)[0, 0]
        else:
            gx = F.conv2d(field_4d, self._sobel_kx, padding=1)[0, 0]
            gy = F.conv2d(field_4d, self._sobel_ky, padding=1)[0, 0]
        return torch.stack([gy, gx], dim=0)

    def _reintegrate(
        self,
        A: torch.Tensor,
        W: torch.Tensor,
        flow: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reintegrate mass and species weights together.

        The geometric area computation is identical to the scalar path's
        _reintegration. For each (dx, dy) source offset, the mass overlap
        Ar * area is accumulated as before. Simultaneously, the per-species
        weight contribution Wr * (Ar * area) is accumulated. After all
        offsets are summed, weights are normalized by the new mass at each
        cell to recover a simplex.

        Where new_A is zero (no mass arrived), weights are set to zero.
        """
        cfg = self.config
        dd = cfg.dd
        sigma = cfg.sigma
        ma = dd - sigma

        clipped_flow = torch.clamp(cfg.dt * flow, -ma, ma)
        mu = self._pos + clipped_flow

        if cfg.border == "torus":
            mu = torch.stack([mu[0] % cfg.grid_h, mu[1] % cfg.grid_w], dim=0)
        else:
            mu = torch.stack(
                [
                    torch.clamp(mu[0], sigma, cfg.grid_h - sigma),
                    torch.clamp(mu[1], sigma, cfg.grid_w - sigma),
                ],
                dim=0,
            )

        new_A = torch.zeros_like(A)
        new_AW = torch.zeros_like(W)  # accumulator for mass-weighted weights
        sigma_clip_max = min(1.0, 2.0 * sigma)
        denom = 4.0 * sigma * sigma
        grid_hw = torch.tensor([cfg.grid_h, cfg.grid_w], device=A.device, dtype=A.dtype).view(
            2, 1, 1
        )

        for dx in range(-dd, dd + 1):
            for dy in range(-dd, dd + 1):
                Ar = torch.roll(A, shifts=(dx, dy), dims=(0, 1))
                Wr = torch.roll(W, shifts=(dx, dy), dims=(0, 1))
                mur = torch.roll(mu, shifts=(dx, dy), dims=(1, 2))
                dpmu = torch.abs(self._pos - mur)
                if cfg.border == "torus":
                    dpmu = torch.minimum(dpmu, grid_hw - dpmu)
                sz = 0.5 - dpmu + sigma
                sz_clipped = torch.clamp(sz, 0.0, sigma_clip_max)
                area = (sz_clipped[0] * sz_clipped[1]) / denom  # (H, W)
                contribution = Ar * area  # (H, W)
                new_A = new_A + contribution
                # Weight contribution: each species' weight at the source
                # carries the same fraction of mass as the mass itself.
                new_AW = new_AW + Wr * contribution.unsqueeze(-1)

        # Normalize: where mass exists, divide by mass to get a simplex.
        # The 1e-8 threshold (rather than > 0) suppresses float roundoff:
        # cells with mass ~1e-12 are physically empty but would otherwise
        # carry a normalized weight of ~1, which is nonsense.
        eps = torch.finfo(new_A.dtype).tiny
        denom_safe = new_A.unsqueeze(-1).clamp(min=eps)
        new_W = new_AW / denom_safe
        mass_present = (new_A > 1e-8).unsqueeze(-1)
        new_W = torch.where(mass_present, new_W, torch.zeros_like(new_W))
        return new_A, new_W
