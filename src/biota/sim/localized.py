"""Species-indexed Flow-Lenia for heterogeneous ecosystem runs.

This module implements parameter localization at the species level. Each
species keeps its full param set (R, r, m, s, h, a, b, w) and its own
precomputed FFT kernel tensor. At each step:

  1. Convolve mass with each species' kernels independently (S * K FFTs).
  2. Compute each species' growth field using its own m / s / h.
  3. Apply signal reception per species: each species reads the convolved
     shared signal field through its own receptor_profile.
  4. Blend the growth fields (plus reception boosts) by species ownership.
  5. Apply signal emission per species: each species emits into the shared
     field proportional to its growth activity and emission_vector.
  6. Compute flow from the blended growth field and mass density.
  7. Reintegrate mass and species weights together.
  8. Decay the signal field.

The scalar-path FlowLenia (biota.sim.flowlenia.FlowLenia) is untouched.
Search rollouts and homogeneous ecosystem runs use it. Only heterogeneous
ecosystem runs reach this module.

Position grid and Sobel kernels are shared from the first species'
FlowLenia instance because they are param-independent.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from biota.sim.flowlenia import EMISSION_RATE, Config, FlowLenia, Params


@dataclass(frozen=True)
class LocalizedState:
    """Mass, species ownership, and optional signal field for a heterogeneous run.

    Attributes:
        mass:    (H, W, 1) float32 mass density.
        weights: (H, W, S) float32 species ownership. Where mass > 0, the
                 weight vector at each cell is a probability simplex
                 (non-negative, sums to 1). Where mass == 0, weights are
                 all zero.
        signal:  (H, W, C) float32 shared signal field. None for non-signal runs.
    """

    mass: torch.Tensor
    weights: torch.Tensor
    signal: torch.Tensor | None = None


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
        self._species: list[FlowLenia] = [
            FlowLenia(config, p, device=device) for p in species_params
        ]
        ref = self._species[0]
        self._sobel_kx, self._sobel_ky = ref._sobel_kx, ref._sobel_ky  # pyright: ignore[reportPrivateUsage]
        self._pos = ref._pos  # pyright: ignore[reportPrivateUsage]

    @property
    def num_species(self) -> int:
        return len(self._species)

    def step(self, state: LocalizedState) -> LocalizedState:
        """Advance one step. Returns a new LocalizedState."""
        new_state, _ = self._step_inner(state, capture_growth=False)
        return new_state

    def step_with_diagnostics(
        self, state: LocalizedState
    ) -> tuple[LocalizedState, list[torch.Tensor]]:
        """Advance one step and return per-species growth fields.

        Returns (new_state, growth_fields) where growth_fields[s] is the
        (H, W) float32 tensor G_s_total for species s before ownership blending.
        These are the raw material for empirical interaction coefficient measurement.

        More expensive than step() because it retains intermediate tensors.
        Call only at snapshot steps; use step() for all other steps.
        """
        return self._step_inner(state, capture_growth=True)

    def _step_inner(
        self, state: LocalizedState, capture_growth: bool
    ) -> tuple[LocalizedState, list[torch.Tensor]]:
        """Shared implementation for step and step_with_diagnostics."""
        A = state.mass[:, :, 0]  # (H, W)
        W = state.weights  # (H, W, S)
        signal = state.signal  # (H, W, C) or None
        S = self.num_species

        if W.shape[-1] != S:
            raise ValueError(f"weights last dim {W.shape[-1]} does not match species count {S}")
        if W.shape[:2] != A.shape:
            raise ValueError(
                f"weights spatial shape {tuple(W.shape[:2])} does not match mass {tuple(A.shape)}"
            )

        # Pre-convolve shared signal field once if signal is active.
        # All species use the same convolution kernel (signal_kernel from species 0
        # that has signal params, or None). The receptor dot product is per-species.
        convolved_signal: torch.Tensor | None = None
        if signal is not None:
            # Find the first species with a signal kernel to convolve.
            for sp in self._species:
                if sp._fK_signal is not None:  # pyright: ignore[reportPrivateUsage]
                    sig_t = signal.permute(2, 0, 1)  # (C, H, W)
                    fSig = torch.fft.fft2(sig_t)
                    convolved_signal = torch.fft.ifft2(
                        sp._fK_signal * fSig  # pyright: ignore[reportPrivateUsage]
                    ).real  # (C, H, W)
                    break

        # Effective weights: uniform at empty cells so flow responds to neighbours.
        weight_sum = W.sum(dim=-1, keepdim=True)
        empty_cells = weight_sum < 1e-8
        uniform = torch.full_like(W, 1.0 / S)
        W_eff = torch.where(empty_cells.expand_as(W), uniform, W)

        fA = torch.fft.fft2(A)
        G_blend = torch.zeros_like(A)
        growth_fields: list[torch.Tensor] = []

        # New signal accumulator: emission is added here, then merged at the end.
        new_signal = signal.clone() if signal is not None else None

        for s, sp in enumerate(self._species):
            fK_s = sp._fK  # pyright: ignore[reportPrivateUsage]
            U_s = torch.fft.ifft2(fK_s * fA.unsqueeze(0)).real  # (K, H, W)
            m = sp.params.m.view(-1, 1, 1)
            sigma = sp.params.s.view(-1, 1, 1)
            h = sp.params.h.view(-1, 1, 1)
            G_s = (torch.exp(-(((U_s - m) / sigma) ** 2) / 2.0) * 2.0 - 1.0) * h
            G_s_total = G_s.sum(dim=0)  # (H, W)

            # Reception: species-specific receptor profile dotted with convolved signal.
            if convolved_signal is not None and sp.params.receptor_profile is not None:
                receptor_response = (
                    convolved_signal * sp.params.receptor_profile.view(-1, 1, 1)
                ).sum(dim=0)  # (H, W)
                G_s_total = G_s_total + receptor_response

            G_blend = G_blend + W_eff[:, :, s] * G_s_total

            # Emission: species s emits into signal field proportional to positive growth.
            if new_signal is not None and sp.params.emission_vector is not None:
                G_pos = G_s_total.clamp(min=0.0)
                ownership_s = W_eff[:, :, s]  # (H, W) -- how much of this cell species s owns
                emitted_scalar = G_pos * ownership_s * EMISSION_RATE
                emitted_scalar = torch.minimum(emitted_scalar, A.clamp(min=0.0))
                emit_per_channel = emitted_scalar.unsqueeze(-1) * sp.params.emission_vector
                new_signal = new_signal + emit_per_channel

            if capture_growth:
                growth_fields.append(G_s_total.detach().cpu())

        # Flow and reintegration.
        nabla_U = self._sobel(G_blend)
        nabla_A = self._sobel(A)
        alpha = torch.clamp(A**2, 0.0, 1.0)
        F_flow = nabla_U * (1.0 - alpha) - nabla_A * alpha

        new_A, new_W = self._reintegrate(A, W, F_flow)

        # Decay signal field.
        if new_signal is not None:
            decay = self._species[0]._decay  # pyright: ignore[reportPrivateUsage]
            new_signal = new_signal * (1.0 - decay)

        return (
            LocalizedState(mass=new_A.unsqueeze(-1), weights=new_W, signal=new_signal),
            growth_fields,
        )

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
        new_AW = torch.zeros_like(W)
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
                contribution = Ar * area
                new_A = new_A + contribution
                new_AW = new_AW + Wr * contribution.unsqueeze(-1)

        eps = torch.finfo(new_A.dtype).tiny
        denom_safe = new_A.unsqueeze(-1).clamp(min=eps)
        new_W = new_AW / denom_safe
        mass_present = (new_A > 1e-8).unsqueeze(-1)
        new_W = torch.where(mass_present, new_W, torch.zeros_like(new_W))
        return new_A, new_W
