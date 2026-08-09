"""
Lifting and Projection MLP Blocks for PINO.
Provides point-wise feature mapping between input space, latent channels, and output fields.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LiftingBlock(nn.Module):
    """
    Lifting layer: Projects input channels (e.g. x-grid, y-grid, a(x))
    to high-dimensional latent representation space d_v.
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, hidden_dim // 2, 1)
        self.fc2 = nn.Conv2d(hidden_dim // 2, hidden_dim, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, in_channels, s_x, s_y)
        Returns:
            Tensor of shape (batch, hidden_dim, s_x, s_y)
        """
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


class ProjectionBlock(nn.Module):
    """
    Projection layer: Projects latent representations d_v
    back to solution field output dimension d_out.
    """

    def __init__(self, hidden_dim: int = 64, out_channels: int = 1):
        super().__init__()
        self.fc1 = nn.Conv2d(hidden_dim, hidden_dim // 2, 1)
        self.fc2 = nn.Conv2d(hidden_dim // 2, out_channels, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, hidden_dim, s_x_out, s_y_out)
        Returns:
            Tensor of shape (batch, out_channels, s_x_out, s_y_out)
        """
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x
