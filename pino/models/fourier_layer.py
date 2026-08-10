"""
2D Spectral Convolution Layer (Fourier Convolution) for Neural Operators.
Computes frequency-domain mode filtering, complex weight tensor multiplication,
and inverse FFT with zero-shot resolution invariance.
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

# Suppress PyTorch internal ATen MPS backend deprecation warning for internal FFT buffer resizing
warnings.filterwarnings("ignore", category=UserWarning, message=".*resized since it had shape.*")


class SpectralConv2d(nn.Module):
    """
    2D Fourier Layer.
    Applies 2D FFT, mode truncation to k_max (modes1, modes2), complex weight multiplication,
    and inverse 2D FFT.
    Supports resolution-invariant evaluation (zero-shot super-resolution).
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # k_max in x-dimension
        self.modes2 = modes2  # k_max in y-dimension

        # Scale factor for weight initialization
        scale = 1.0 / (in_channels * out_channels)

        # Complex weights for top-left (positive x, positive y) Fourier modes
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        # Complex weights for bottom-left (negative x, positive y) Fourier modes
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

        # Local spatial convolution shortcut W (1x1 conv)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def _compl_mul2d(self, input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Complex matrix multiplication over Fourier modes.
        (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        """
        return torch.einsum("bixy,ioxy->boxy", input_tensor, weights)

    def forward(self, x: torch.Tensor, target_s: tuple = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, in_channels, s_x, s_y)
            target_s: Optional (s_x_out, s_y_out) for super-resolution evaluation
        Returns:
            Output tensor of shape (batch, out_channels, s_x_out, s_y_out)
        """
        batch_size = x.shape[0]
        s_x, s_y = x.shape[2], x.shape[3]
        out_s_x, out_s_y = target_s if target_s is not None else (s_x, s_y)

        # 1. 2D Real Fast Fourier Transform (rfft2)
        # x_ft shape: (batch, in_channels, s_x, s_y // 2 + 1)
        x_ft = torch.fft.rfft2(x.contiguous())

        # 2. Allocate output frequency tensor for target resolution
        out_ft = torch.zeros(
            batch_size, self.out_channels, out_s_x, out_s_y // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        # 3. Multiply low-frequency modes by complex weights
        # Positive x-frequencies
        out_ft[:, :, :self.modes1, :self.modes2] = self._compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        # Negative x-frequencies
        out_ft[:, :, -self.modes1:, :self.modes2] = self._compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        # 4. Inverse 2D Real Fast Fourier Transform (irfft2)
        x_spectral = torch.fft.irfft2(out_ft.contiguous(), s=(out_s_x, out_s_y))

        # 5. Spatial residual shortcut W(x)
        if (out_s_x, out_s_y) != (s_x, s_y):
            x_res = F.interpolate(x, size=(out_s_x, out_s_y), mode="bilinear", align_corners=False)
        else:
            x_res = x
        x_spatial = self.shortcut(x_res)

        return x_spectral + x_spatial
