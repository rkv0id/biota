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

# Number of signal field channels. Matches SIGNAL_CHANNELS in params.py --
# kept as a local constant here so flowlenia.py has no search-layer dependency.
SIGNAL_CHANNELS = 16


@dataclass(frozen=True)
class Config:
    grid_h: int = 96
    grid_w: int = 96
    kernels: int = 10
    dd: int = 5
    dt: float = 0.2
    sigma: float = 0.65
    border: str = "wall"

    @property
    def grid(self) -> int:
        """Convenience accessor for square grids. Raises if not square."""
        if self.grid_h != self.grid_w:
            raise ValueError(
                f"Config.grid is only valid for square grids "
                f"(grid_h={self.grid_h}, grid_w={self.grid_w})"
            )
        return self.grid_h


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
    # Signal field parameters -- optional. None for non-signal creatures.
    emission_vector: torch.Tensor | None = None  # (C,) in [0, 1]
    receptor_profile: torch.Tensor | None = None  # (C,) in [-1, 1]
    emission_rate: float | None = None  # scalar in [0.001, 0.05]
    decay_rates: torch.Tensor | None = None  # (C,) in [0, 0.9] per-channel
    signal_kernel_r: float | None = None
    signal_kernel_a: torch.Tensor | None = None  # (3,) ring centers
    signal_kernel_b: torch.Tensor | None = None  # (3,) ring weights
    signal_kernel_w: torch.Tensor | None = None  # (3,) ring widths
    # Coupling parameters -- optional, default to zero (no coupling).
    alpha_coupling: float | None = None  # scalar in [-1, 1]: chemotaxis (+) vs chemorepulsion (-)
    beta_modulation: float | None = (
        None  # scalar in [-1, 1]: quorum sensing (+) vs feedback inhibition (-)
    )

    @property
    def has_signal(self) -> bool:
        return self.emission_vector is not None


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
            emission_vector=params.emission_vector.to(device)
            if params.emission_vector is not None
            else None,
            receptor_profile=params.receptor_profile.to(device)
            if params.receptor_profile is not None
            else None,
            emission_rate=params.emission_rate,
            decay_rates=params.decay_rates.to(device) if params.decay_rates is not None else None,
            signal_kernel_r=params.signal_kernel_r,
            signal_kernel_a=params.signal_kernel_a.to(device)
            if params.signal_kernel_a is not None
            else None,
            signal_kernel_b=params.signal_kernel_b.to(device)
            if params.signal_kernel_b is not None
            else None,
            signal_kernel_w=params.signal_kernel_w.to(device)
            if params.signal_kernel_w is not None
            else None,
        )
        self._fK = self._build_kernels_fft()
        self._fK_signal: torch.Tensor | None = (
            self._build_signal_kernel_fft() if self.params.has_signal else None
        )
        # _decay built from per-creature decay_rates; fallback zeros if somehow absent.
        self._decay: torch.Tensor = (
            self.params.decay_rates
            if self.params.decay_rates is not None
            else torch.zeros(SIGNAL_CHANNELS, dtype=torch.float32, device=device)
        )
        self._sobel_kx, self._sobel_ky = self._build_sobel_kernels()
        self._pos = self._build_position_grid()

    def make_initial_signal_field(self, seed: int = 0) -> torch.Tensor:
        """Build a spatially varied (H, W, C) float32 signal field background.

        Each channel is independently sampled from filtered Gaussian noise --
        low-frequency spatial structure so the receptor dot product varies
        meaningfully across the grid. Amplitude is kept low (~0.01 mean
        absolute value) to perturb without dominating mass dynamics.

        Used to initialize the signal field before a rollout, giving the
        receptor profile something to respond to in solo creature searches.
        The initial total signal mass is small and known, so the alive filter
        accounts for it correctly via initial_total.
        """
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        h, w = self.config.grid_h, self.config.grid_w

        # Sample independent Gaussian noise per channel.
        noise = torch.randn(SIGNAL_CHANNELS, h, w, generator=rng, device=self.device)

        # Lowpass filter: zero out the top 80% of spatial frequencies per channel.
        # This gives smooth, spatially correlated patterns per channel.
        noise_fft = torch.fft.fft2(noise)  # (C, H, W) complex
        freq_h = torch.fft.fftfreq(h, device=self.device).abs()
        freq_w = torch.fft.fftfreq(w, device=self.device).abs()
        fh, fw = torch.meshgrid(freq_h, freq_w, indexing="ij")
        freq_mag = torch.sqrt(fh**2 + fw**2)  # (H, W)
        lowpass = (freq_mag < 0.2).float()  # keep low 20% of frequencies
        filtered_fft = noise_fft * lowpass.unsqueeze(0)
        signal = torch.fft.ifft2(filtered_fft).real  # (C, H, W)

        # Normalize each channel to have mean absolute value ~0.01.
        mean_abs = signal.abs().mean(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        signal = signal / mean_abs * 0.01
        signal = signal.clamp(min=0.0)  # signal field is non-negative

        return signal.permute(1, 2, 0)  # (H, W, C)

    def step(
        self, A: torch.Tensor, signal: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Advance the simulation by one step.

        A:      (H, W, 1) mass field.
        signal: (H, W, C) signal field, or None for non-signal creatures.

        Returns (new_A, new_signal). new_signal is None when signal is None.

        Signal physics (only when signal is not None and params.has_signal):
          1. Compute growth field G as usual.
          2. Emission: add emission_vector * G_magnitude * emission_rate to
             the signal field at each cell, drain the same amount from mass.
          3. Reception: compute dot(receptor_profile, conv(signal)) per cell,
             add as a boost to the growth field before reintegration.
          4. Reintegrate mass with the boosted growth field.
          5. Decay signal field by per-channel decay rates.
          Mass + signal total is conserved modulo decay (decay is the only leak).
        """
        A2 = A[:, :, 0]

        fA = torch.fft.fft2(A2)
        U = torch.fft.ifft2(self._fK * fA.unsqueeze(0)).real  # (k, H, W)
        m = self.params.m.view(-1, 1, 1)
        s = self.params.s.view(-1, 1, 1)
        h = self.params.h.view(-1, 1, 1)
        G = (torch.exp(-(((U - m) / s) ** 2) / 2.0) * 2.0 - 1.0) * h
        U_sum = G.sum(dim=0)  # (H, W) blended growth field

        if signal is not None and self.params.has_signal and self._fK_signal is not None:
            assert self.params.emission_vector is not None
            assert self.params.receptor_profile is not None

            # G magnitude as emission driver: clamp to [0, inf) so only
            # positive growth activity emits. Negative growth (dying regions)
            # does not emit signal.
            G_pos = U_sum.clamp(min=0.0)  # (H, W)

            # Reception: convolve signal field, dot with receptor profile.
            # Done before emission so beta_modulation can read the pre-emission field.
            sig_t = signal.permute(2, 0, 1)  # (C, H, W)
            fSig = torch.fft.fft2(sig_t)  # (C, H, W) complex
            convolved = torch.fft.ifft2(self._fK_signal * fSig).real  # (C, H, W)
            # Dot product with receptor profile -> (H, W) reception field.
            receptor_response = (convolved * self.params.receptor_profile.view(-1, 1, 1)).sum(dim=0)

            # Alpha coupling: multiplicative reception-to-growth.
            # G *= (1 + alpha * reception) -- applies everywhere, not just own territory.
            # Positive alpha = chemotaxis (grow toward favorable signal).
            # Negative alpha = chemorepulsion (grow away from signal).
            # Clamped to [0, inf) to prevent growth reversal.
            alpha_c = self.params.alpha_coupling if self.params.alpha_coupling is not None else 0.0
            if alpha_c != 0.0:
                growth_multiplier = (1.0 + alpha_c * receptor_response).clamp(min=0.0)
                U_sum = U_sum * growth_multiplier
                G_pos = U_sum.clamp(min=0.0)  # recompute after coupling

            # Adaptive emission via beta_modulation.
            # Positive beta = quorum sensing: received signal amplifies emission.
            # Negative beta = feedback inhibition: received signal suppresses emission.
            base_rate = self.params.emission_rate if self.params.emission_rate is not None else 0.0
            beta_m = self.params.beta_modulation if self.params.beta_modulation is not None else 0.0
            if beta_m != 0.0:
                received_mean = float(receptor_response.mean().item())
                effective_rate = float(
                    max(0.0, min(0.1, base_rate * (1.0 + beta_m * received_mean)))
                )
            else:
                effective_rate = base_rate

            # Emission: drain from mass, add to signal field.
            emitted = G_pos * effective_rate  # (H, W) scalar per cell
            emitted = torch.minimum(emitted, A2.clamp(min=0.0))
            emit_per_channel = emitted.unsqueeze(-1) * self.params.emission_vector  # (H, W, C)
            signal = signal + emit_per_channel
            A2 = A2 - emitted  # drain from mass

        nabla_U = self._sobel(U_sum)
        nabla_A = self._sobel(A2)
        alpha = torch.clamp(A2**2, 0.0, 1.0)
        F_flow = nabla_U * (1.0 - alpha) - nabla_A * alpha
        new_A2 = self._reintegration(A2, F_flow)

        if signal is not None and self.params.has_signal:
            # Decay signal field by per-channel rates.
            new_signal = signal * (1.0 - self._decay)  # (H, W, C)
        else:
            new_signal = signal  # None passthrough for non-signal path

        return new_A2.unsqueeze(-1), new_signal

    def step_with_signal_diagnostics(
        self, A: torch.Tensor, signal: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None, float, float]:
        """Like step(), but also returns per-step signal diagnostic scalars.

        Used by the rollout loop to capture emission_activity and
        receptor_sensitivity descriptor histories without re-running the sim.

        Returns:
            (new_A, new_signal, emission_activity, receptor_sensitivity)
            emission_activity:    mean(G_pos * effective_rate) over the grid --
                                  the actual per-cell emission scalar, averaged.
                                  Proportional to how much signal was emitted this step.
            receptor_sensitivity: mean(|receptor_response|) over the grid --
                                  how strongly the creature responded to the convolved
                                  signal field this step.
            Both are 0.0 for non-signal creatures or when signal is None.
        """
        A2 = A[:, :, 0]

        fA = torch.fft.fft2(A2)
        U = torch.fft.ifft2(self._fK * fA.unsqueeze(0)).real
        m = self.params.m.view(-1, 1, 1)
        s = self.params.s.view(-1, 1, 1)
        h = self.params.h.view(-1, 1, 1)
        G = (torch.exp(-(((U - m) / s) ** 2) / 2.0) * 2.0 - 1.0) * h
        U_sum = G.sum(dim=0)

        emission_activity_scalar = 0.0
        receptor_sensitivity_scalar = 0.0

        if signal is not None and self.params.has_signal and self._fK_signal is not None:
            assert self.params.emission_vector is not None
            assert self.params.receptor_profile is not None

            G_pos = U_sum.clamp(min=0.0)

            sig_t = signal.permute(2, 0, 1)
            fSig = torch.fft.fft2(sig_t)
            convolved = torch.fft.ifft2(self._fK_signal * fSig).real
            receptor_response = (convolved * self.params.receptor_profile.view(-1, 1, 1)).sum(dim=0)

            # receptor_sensitivity: sum of absolute reception response over the grid.
            # sum() avoids dilution by empty cells.
            receptor_sensitivity_scalar = float(receptor_response.abs().sum().item())

            alpha_c = self.params.alpha_coupling if self.params.alpha_coupling is not None else 0.0
            if alpha_c != 0.0:
                growth_multiplier = (1.0 + alpha_c * receptor_response).clamp(min=0.0)
                U_sum = U_sum * growth_multiplier
                G_pos = U_sum.clamp(min=0.0)

            base_rate = self.params.emission_rate if self.params.emission_rate is not None else 0.0
            beta_m = self.params.beta_modulation if self.params.beta_modulation is not None else 0.0
            if beta_m != 0.0:
                received_mean = float(receptor_response.mean().item())
                effective_rate = float(
                    max(0.0, min(0.1, base_rate * (1.0 + beta_m * received_mean)))
                )
            else:
                effective_rate = base_rate

            # emission_activity: sum(G_pos * effective_rate) over the grid.
            # sum() rather than mean() avoids dilution by empty cells, which
            # makes the value proportional to actual creature output regardless
            # of grid size (dev vs standard vs pretty).
            emission_activity_scalar = float((G_pos * effective_rate).sum().item())

            emitted = G_pos * effective_rate
            emitted = torch.minimum(emitted, A2.clamp(min=0.0))
            emit_per_channel = emitted.unsqueeze(-1) * self.params.emission_vector
            signal = signal + emit_per_channel
            A2 = A2 - emitted

        nabla_U = self._sobel(U_sum)
        nabla_A = self._sobel(A2)
        alpha = torch.clamp(A2**2, 0.0, 1.0)
        F_flow = nabla_U * (1.0 - alpha) - nabla_A * alpha
        new_A2 = self._reintegration(A2, F_flow)

        if signal is not None and self.params.has_signal:
            new_signal = signal * (1.0 - self._decay)
        else:
            new_signal = signal

        return (
            new_A2.unsqueeze(-1),
            new_signal,
            emission_activity_scalar,
            receptor_sensitivity_scalar,
        )
        """Vectorized step over a batch of states. Signal field not supported
        in batch mode (search rollouts don't use it; ecosystem uses single-step).

        A: (B, H, W, 1) -> (B, H, W, 1)
        """
        B = A.shape[0]
        A2 = A[:, :, :, 0]  # (B, H, W)

        fA = torch.fft.fft2(A2)
        U = torch.fft.ifft2(
            self._fK.unsqueeze(0) * fA.unsqueeze(1), dim=(-2, -1)
        ).real  # (B, K, H, W)

        m = self.params.m.view(1, -1, 1, 1)
        s = self.params.s.view(1, -1, 1, 1)
        h = self.params.h.view(1, -1, 1, 1)
        G = (torch.exp(-(((U - m) / s) ** 2) / 2.0) * 2.0 - 1.0) * h
        U_sum = G.sum(dim=1)  # (B, H, W)

        nabla_U = self._sobel_batch(U_sum)  # (B, 2, H, W)
        nabla_A = self._sobel_batch(A2)  # (B, 2, H, W)

        alpha = torch.clamp(A2**2, 0.0, 1.0)  # (B, H, W)

        alpha4 = alpha.unsqueeze(1)
        F_flow = nabla_U * (1.0 - alpha4) - nabla_A * alpha4

        new_A2 = self._reintegration_batch(A2, F_flow, B)  # (B, H, W)
        return new_A2.unsqueeze(-1)

    def rollout(self, A0: torch.Tensor, steps: int) -> torch.Tensor:
        A: torch.Tensor = A0
        for _ in range(steps):
            A, _ = self.step(A, None)
        return A

    def kernel_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the pre-built simulation tensors needed for batched execution."""
        return self._fK, self._sobel_kx, self._sobel_ky, self._pos

    def rollout_with_mass(self, A0: torch.Tensor, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        masses = torch.empty(steps + 1, dtype=A0.dtype)
        masses[0] = A0.sum()
        A: torch.Tensor = A0
        for i in range(steps):
            A, _ = self.step(A, None)
            masses[i + 1] = A.sum()
        return A, masses

    # === private helpers ===

    def _build_signal_kernel_fft(self) -> torch.Tensor:
        """Build the Fourier-space signal convolution kernel (H, W) complex.

        Uses the same ring-function parameterization as the mass kernels but
        with the creature's signal_kernel_* params. Single kernel shared across
        all C channels; the receptor dot product is what makes channels distinct.
        """
        p = self.params
        assert p.signal_kernel_r is not None
        assert p.signal_kernel_a is not None
        assert p.signal_kernel_b is not None
        assert p.signal_kernel_w is not None

        cfg = self.config
        mid_h = cfg.grid_h // 2
        mid_w = cfg.grid_w // 2
        coords_y = torch.arange(-mid_h, mid_h, device=self.device, dtype=torch.float32)
        coords_x = torch.arange(-mid_w, mid_w, device=self.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords_y, coords_x, indexing="ij")
        radius = torch.sqrt(xx**2 + yy**2)

        scale = (p.R + 15.0) * p.signal_kernel_r
        D = radius / max(scale, 1e-6)
        mask = torch.sigmoid(-(D - 1.0) * 10.0)

        a = p.signal_kernel_a.view(1, 1, 3)
        b_ring = p.signal_kernel_b.view(1, 1, 3)
        w_ring = p.signal_kernel_w.view(1, 1, 3)
        D3 = D.unsqueeze(-1)
        ring = (b_ring * torch.exp(-((D3 - a) ** 2) / w_ring)).sum(dim=-1)

        K = mask * ring
        total = K.sum()
        nK = K / total if total > 0 else K
        nK_shifted = torch.fft.fftshift(nK)
        return cast(torch.Tensor, torch.fft.fft2(nK_shifted))

    def _build_kernels_fft(self) -> torch.Tensor:
        """Build the Fourier-space kernels following the reference parameterization.

        Returns a complex tensor of shape (k, grid_h, grid_w).

        For rectangular grids, separate y and x coordinate ranges are used.
        The kernel radius is computed in pixel space using an aspect-corrected
        coordinate system so the kernel shape is circular in physical space
        regardless of aspect ratio.
        """
        cfg = self.config
        p = self.params
        mid_h = cfg.grid_h // 2
        mid_w = cfg.grid_w // 2

        coords_y = torch.arange(-mid_h, mid_h, device=self.device, dtype=torch.float32)
        coords_x = torch.arange(-mid_w, mid_w, device=self.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords_y, coords_x, indexing="ij")
        radius = torch.sqrt(xx**2 + yy**2)  # (H, W)

        # Per-kernel scale; the +15 is from the reference, not the paper
        scale = (p.R + 15.0) * p.r  # (k,)
        D = radius.unsqueeze(0) / scale.view(-1, 1, 1)  # (k, H, W)

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
        Shape (2, grid_h, grid_w).
        """
        coords_y = torch.arange(self.config.grid_h, device=self.device, dtype=torch.float32) + 0.5
        coords_x = torch.arange(self.config.grid_w, device=self.device, dtype=torch.float32) + 0.5
        yy, xx = torch.meshgrid(coords_y, coords_x, indexing="ij")
        return torch.stack([yy, xx], dim=0)

    def _sobel(self, field: torch.Tensor) -> torch.Tensor:
        """Sobel gradients of a 2D field. Returns (gy, gx) stacked, shape (2, X, Y).

        Wall border: zero-padding matches jax.scipy.signal.convolve2d(mode='same').
        Torus border: circular padding so gradients at the seam see the opposite edge.
        """
        field_4d = field.unsqueeze(0).unsqueeze(0)
        if self.config.border == "torus":
            field_4d = F.pad(field_4d, (1, 1, 1, 1), mode="circular")
            gx = F.conv2d(field_4d, self._sobel_kx, padding=0)[0, 0]
            gy = F.conv2d(field_4d, self._sobel_ky, padding=0)[0, 0]
        else:
            gx = F.conv2d(field_4d, self._sobel_kx, padding=1)[0, 0]
            gy = F.conv2d(field_4d, self._sobel_ky, padding=1)[0, 0]
        return torch.stack([gy, gx], dim=0)

    def _sobel_batch(self, field: torch.Tensor) -> torch.Tensor:
        """Sobel gradients over a batch of 2D fields.

        field: (B, H, W) -> (B, 2, H, W) stacked as [gy, gx].

        Wall border: zero-padding; torus border: circular padding so seam
        gradients are computed correctly across the wrap edge.
        """
        B = field.shape[0]
        field_4d = field.unsqueeze(1)  # (B, 1, H, W)
        if self.config.border == "torus":
            field_4d = F.pad(field_4d, (1, 1, 1, 1), mode="circular")
            gx = F.conv2d(field_4d, self._sobel_kx, padding=0)[:, 0]
            gy = F.conv2d(field_4d, self._sobel_ky, padding=0)[:, 0]
        else:
            gx = F.conv2d(field_4d, self._sobel_kx, padding=1)[:, 0]
            gy = F.conv2d(field_4d, self._sobel_ky, padding=1)[:, 0]
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
        sigma_clip_max = min(1.0, 2.0 * sigma)
        denom = 4.0 * sigma * sigma
        grid_hw = torch.tensor([cfg.grid_h, cfg.grid_w], device=A.device, dtype=A.dtype).view(
            2, 1, 1
        )

        for dx in range(-dd, dd + 1):
            for dy in range(-dd, dd + 1):
                Ar = torch.roll(A, shifts=(dx, dy), dims=(0, 1))
                mur = torch.roll(mu, shifts=(dx, dy), dims=(1, 2))
                dpmu = torch.abs(self._pos - mur)  # (2, H, W)
                if cfg.border == "torus":
                    dpmu = torch.minimum(dpmu, grid_hw - dpmu)
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
