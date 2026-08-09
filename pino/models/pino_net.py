"""
PINO2D Neural Operator Architecture.
Combines Lifting MLP, stacked 2D Spectral Convolutions, and Projection MLP.
Enables resolution-invariant zero-shot super-resolution evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from pino.models.fourier_layer import SpectralConv2d
from pino.models.mlp import LiftingBlock, ProjectionBlock
from pino.config import ModelConfig


class PINO2D(nn.Module):
    """
    Physics-Informed Fourier Neural Operator (PINO) 2D Network.
    
    Lifting Layer (d_in -> d_v)
        │
        ├── SpectralConv2d Layer 1 (modes1, modes2) + GELU
        ├── SpectralConv2d Layer 2 (modes1, modes2) + GELU
        ├── ...
        └── SpectralConv2d Layer N (modes1, modes2) + GELU
        │
    Projection Layer (d_v -> d_out)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        hidden_dim: int = 64,
        modes1: int = 16,
        modes2: int = 16,
        num_layers: int = 4
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.modes1 = modes1
        self.modes2 = modes2
        self.num_layers = num_layers

        # 1. Lifting block
        self.lifting = LiftingBlock(in_channels=in_channels, hidden_dim=hidden_dim)

        # 2. Stacked Spectral Convolution layers
        self.fourier_layers = nn.ModuleList([
            SpectralConv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                modes1=modes1,
                modes2=modes2
            )
            for _ in range(num_layers)
        ])

        # GELU non-linearities between layers
        self.activation = nn.GELU()

        # 3. Projection block
        self.projection = ProjectionBlock(hidden_dim=hidden_dim, out_channels=out_channels)

    @classmethod
    def from_config(cls, config: ModelConfig):
        """Instantiate model directly from ModelConfig object."""
        return cls(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            hidden_dim=config.hidden_dim,
            modes1=config.modes1,
            modes2=config.modes2,
            num_layers=config.num_layers
        )

    def forward(self, x: torch.Tensor, target_s: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch, in_channels, s_x, s_y)
            target_s: Optional (s_x_out, s_y_out) for super-resolution output
        Returns:
            Output solution field of shape (batch, out_channels, s_x_out, s_y_out)
        """
        # 1. Lifting: (batch, in_channels, s_x, s_y) -> (batch, hidden_dim, s_x, s_y)
        h = self.lifting(x)

        # 2. Fourier convolution stack
        for i, layer in enumerate(self.fourier_layers):
            # Target resolution applied at the final layer if target_s specified
            layer_target_s = target_s if (i == len(self.fourier_layers) - 1) else None
            h = layer(h, target_s=layer_target_s)
            if i < len(self.fourier_layers) - 1:
                h = self.activation(h)

        # 3. Projection: (batch, hidden_dim, s_x_out, s_y_out) -> (batch, out_channels, s_x_out, s_y_out)
        out = self.projection(h)
        return out
