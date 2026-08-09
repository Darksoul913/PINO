"""
Frequency-Domain Spectral Differentiator & Anti-Aliasing Engine for PINO.
Calculates exact spatial derivatives (∂u/∂x, ∂u/∂y, ∇^2 u) via Fourier operational multipliers
and applies Orszag 3/2 dealiasing to non-linear advection terms.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class SpectralDiff2D(nn.Module):
    """
    Exact Frequency-Domain Spatial Differentiator for 2D periodic functions on [0, 2π]^2.
    Uses ik_x, ik_y, and -|k|^2 multipliers to compute derivatives without numerical dispersion.
    """

    def __init__(self, s_x: int = 64, s_y: int = 64, domain_len: float = 2.0 * math.pi, device: str = "cpu"):
        super().__init__()
        self.s_x = s_x
        self.s_y = s_y
        self.L = domain_len
        self.device = device

        # Wavenumber matrices
        kx = torch.fft.fftfreq(s_x, d=domain_len / (2 * math.pi * s_x)).to(device)
        ky = torch.fft.fftfreq(s_y, d=domain_len / (2 * math.pi * s_y)).to(device)
        Kx, Ky = torch.meshgrid(kx, ky, indexing="ij")

        # Register multiplier tensors as non-trainable buffers
        self.register_buffer("Kx", Kx)
        self.register_buffer("Ky", Ky)
        self.register_buffer("K2", Kx**2 + Ky**2)
        self.register_buffer("ikx", 1j * Kx)
        self.register_buffer("iky", 1j * Ky)
        self.register_buffer("neg_k2", -1.0 * (Kx**2 + Ky**2))

        # Orszag 2/3 dealiasing cutoff mask
        k_max_x = s_x // 3
        k_max_y = s_y // 3
        dealias_mask = (torch.abs(Kx) <= k_max_x) & (torch.abs(Ky) <= k_max_y)
        self.register_buffer("dealias_mask", dealias_mask)

    def diff_x(self, u: torch.Tensor) -> torch.Tensor:
        """Exact spatial derivative ∂u/∂x."""
        u_hat = torch.fft.fft2(u)
        du_dx_hat = u_hat * self.ikx
        return torch.fft.ifft2(du_dx_hat).real

    def diff_y(self, u: torch.Tensor) -> torch.Tensor:
        """Exact spatial derivative ∂u/∂y."""
        u_hat = torch.fft.fft2(u)
        du_dy_hat = u_hat * self.iky
        return torch.fft.ifft2(du_dy_hat).real

    def laplacian(self, u: torch.Tensor) -> torch.Tensor:
        """Exact spatial Laplacian ∇^2 u = ∂^2 u / ∂x^2 + ∂^2 u / ∂y^2."""
        u_hat = torch.fft.fft2(u)
        lap_hat = u_hat * self.neg_k2
        return torch.fft.ifft2(lap_hat).real

    def dealiased_advection(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Computes dealiased point-wise product of fields u and v (e.g. u · ∇u)
        using Orszag spectral filtering to prevent aliasing instability.
        """
        u_hat = torch.fft.fft2(u)
        v_hat = torch.fft.fft2(v)

        # Apply dealiasing mask in spectral domain
        u_hat_filt = u_hat * self.dealias_mask
        v_hat_filt = v_hat * self.dealias_mask

        # Transform back to physical space for multiplication
        u_filt = torch.fft.ifft2(u_hat_filt).real
        v_filt = torch.fft.ifft2(v_hat_filt).real

        # Multiply in physical space and re-apply dealiasing filter
        product_phys = u_filt * v_filt
        product_hat = torch.fft.fft2(product_phys) * self.dealias_mask
        return torch.fft.ifft2(product_hat).real
