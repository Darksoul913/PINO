"""
Multi-Objective PDE Physics Residual and Data Loss Engine for PINO.
Calculates continuous function-space L2 relative errors for Data, Initial Condition,
and Frequency-Domain PDE residual losses.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from pino.physics.spectral_diff import SpectralDiff2D
from pino.config import LossConfig, PDEConfig


class PINOLossEngine(nn.Module):
    """
    Multi-objective Loss Engine for PINO.
    Combines:
    1. L_data: Supervised relative L2 state error against reference data
    2. L_ic: Initial condition constraint error
    3. L_pde: Physics residual error evaluated exact in spectral domain
    """

    def __init__(self, s_x: int = 64, s_y: int = 64, viscosity: float = 1e-3, device: str = "cpu"):
        super().__init__()
        self.s_x = s_x
        self.s_y = s_y
        self.viscosity = viscosity
        self.device = device

        # Spectral differentiator engine
        self.diff = SpectralDiff2D(s_x=s_x, s_y=s_y, device=device)

    def relative_l2_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes relative L2 norm error ||pred - target||_2 / ||target||_2."""
        diff_norms = torch.norm(pred - target, p=2, dim=(-2, -1))
        target_norms = torch.norm(target, p=2, dim=(-2, -1)) + 1e-8
        return torch.mean(diff_norms / target_norms)

    def compute_ic_loss(self, u_pred_ic: torch.Tensor, a_init: torch.Tensor) -> torch.Tensor:
        """Computes initial condition loss L_ic."""
        return self.relative_l2_loss(u_pred_ic, a_init)

    def compute_pde_residual(
        self,
        u: torch.Tensor,
        dt: float = 0.01,
        forcing: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes 2D Navier-Stokes vorticity residual:
        P(ω) = ∂ω/∂t + u · ∇ω - ν ∇^2 ω - f(x, y)
        Args:
            u: Predicted vorticity trajectory of shape (batch, T_steps, s_x, s_y)
            dt: Time step size
        Returns:
            Residual tensor of shape (batch, T_steps - 1, s_x, s_y)
        """
        # 1. Temporal derivative ∂ω/∂t (finite difference across time steps)
        d_omega_dt = (u[:, 1:] - u[:, :-1]) / dt  # (batch, T-1, s_x, s_y)

        # Average vorticity state across adjacent time steps
        w_mid = 0.5 * (u[:, 1:] + u[:, :-1])      # (batch, T-1, s_x, s_y)

        batch_size, num_steps, s_x, s_y = w_mid.shape
        w_mid_flat = w_mid.reshape(-1, s_x, s_y)

        # 2. Compute streamfunction ψ_hat = w_hat / |k|^2 in spectral domain
        w_hat = torch.fft.fft2(w_mid_flat)
        inv_k2 = torch.zeros_like(self.diff.K2)
        nonzero = self.diff.K2 > 1e-10
        inv_k2[nonzero] = 1.0 / self.diff.K2[nonzero]

        psi_hat = w_hat * inv_k2
        ux_hat = 1j * self.diff.Ky * psi_hat
        uy_hat = -1j * self.diff.Kx * psi_hat

        # Velocities and vorticity gradients in physical space
        ux = torch.fft.ifft2(ux_hat).real
        uy = torch.fft.ifft2(uy_hat).real
        wx = self.diff.diff_x(w_mid_flat)
        wy = self.diff.diff_y(w_mid_flat)

        # 3. Dealiased non-linear advection: u · ∇ω = ux * wx + uy * wy
        advection = self.diff.dealiased_advection(ux, wx) + self.diff.dealiased_advection(uy, wy)
        advection = advection.reshape(batch_size, num_steps, s_x, s_y)

        # 4. Viscous diffusion term: ν ∇^2 ω
        laplacian = self.diff.laplacian(w_mid_flat).reshape(batch_size, num_steps, s_x, s_y)
        diffusion = self.viscosity * laplacian

        # 5. Total PDE residual
        residual = d_omega_dt + advection - diffusion
        if forcing is not None:
            residual = residual - forcing.unsqueeze(1)

        return residual

    def forward(
        self,
        pred_u: torch.Tensor,
        a_init: torch.Tensor,
        target_u: Optional[torch.Tensor] = None,
        forcing: Optional[torch.Tensor] = None,
        loss_config: Optional[LossConfig] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes multi-objective loss breakdown.
        Args:
            pred_u: Predicted trajectory (batch, T, s_x, s_y) or (batch, 1, s_x, s_y)
            a_init: Input initial condition (batch, 1, s_x, s_y)
            target_u: Optional reference ground-truth trajectory
        """
        config = loss_config if loss_config is not None else LossConfig()

        # Initial condition loss (t = 0)
        u_ic = pred_u[:, 0:1] if pred_u.ndim == 4 else pred_u
        l_ic = self.compute_ic_loss(u_ic, a_init)

        # Data loss
        l_data = torch.tensor(0.0, device=self.device)
        if target_u is not None:
            l_data = self.relative_l2_loss(pred_u, target_u)

        # PDE residual loss
        l_pde = torch.tensor(0.0, device=self.device)
        if pred_u.ndim == 4 and pred_u.shape[1] > 1:
            pde_res = self.compute_pde_residual(pred_u, forcing=forcing)
            l_pde = torch.mean(torch.norm(pde_res, p=2, dim=(-2, -1)))

        # Total combined multi-objective loss
        l_total = (
            config.weight_ic * l_ic +
            config.weight_data * l_data +
            config.weight_pde * l_pde
        )

        return {
            "loss_total": l_total,
            "loss_ic": l_ic,
            "loss_data": l_data,
            "loss_pde": l_pde
        }
