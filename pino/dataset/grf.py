"""
Gaussian Random Field (GRF) Sampler for continuous initial conditions.
Generates smooth 2D spatial fields with specified covariance length scale and smoothness.
"""

import torch
import math
from typing import Tuple


class GaussianRandomField2D:
    """
    2D Gaussian Random Field (GRF) generator on periodic domain [0, 2π]^2.
    Covariance kernel in Fourier space: C(k) = (|k|^2 + τ^2)^(-α/2)
    where τ = 1/l (inverse length scale) and α governs field smoothness.
    """

    def __init__(
        self,
        s_x: int = 64,
        s_y: int = 64,
        length_scale: float = 0.5,
        alpha: float = 2.5,
        device: str = "cpu"
    ):
        self.s_x = s_x
        self.s_y = s_y
        self.length_scale = max(length_scale, 1e-5)
        self.alpha = alpha
        self.tau = 1.0 / self.length_scale
        self.device = device

        # Precompute 2D wavenumber grid
        kx = torch.fft.fftfreq(s_x, d=1.0 / s_x).to(device)
        ky = torch.fft.fftfreq(s_y, d=1.0 / s_y).to(device)
        Kx, Ky = torch.meshgrid(kx, ky, indexing="ij")

        # Squared wavenumber magnitude |k|^2
        K2 = Kx**2 + Ky**2

        # Covariance amplitude spectrum sqrt(C(k))
        # Add tau^2 to avoid division by zero at k=(0,0)
        self.sqrt_C = (K2 + (self.tau**2)) ** (-self.alpha / 2.0)
        # Set DC component (zero frequency) to 0 for zero-mean field
        self.sqrt_C[0, 0] = 0.0

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """
        Sample random 2D fields.
        Returns:
            torch.Tensor of shape (num_samples, s_x, s_y)
        """
        # Complex standard Gaussian noise in Fourier space
        noise_real = torch.randn(num_samples, self.s_x, self.s_y, device=self.device)
        noise_imag = torch.randn(num_samples, self.s_x, self.s_y, device=self.device)
        noise_fourier = torch.complex(noise_real, noise_imag)

        # Apply covariance spectrum filtering
        field_fourier = noise_fourier * self.sqrt_C.unsqueeze(0)

        # Inverse 2D FFT to physical space
        field = torch.fft.ifft2(field_fourier).real

        # Standardize output field
        std = field.std(dim=(-2, -1), keepdim=True) + 1e-8
        mean = field.mean(dim=(-2, -1), keepdim=True)
        field = (field - mean) / std

        return field
