"""
High-Order Pseudo-Spectral RK4 Solver for 2D Incompressible Navier-Stokes / Viscous Burgers' Equations.
Generates baseline ground-truth spatiotemporal trajectories for data loss evaluation.
"""

import torch
import math
from typing import Tuple, Optional


class NavierStokes2DSolver:
    """
    2D Incompressible Navier-Stokes solver using Vorticity-Streamfunction formulation
    with 4th-order Runge-Kutta (RK4) pseudo-spectral time integration and Orszag 2/3 dealiasing.
    
    ∂ω/∂t + u · ∇ω = ν ∇^2 ω + f(x, y)
    -∇^2 ψ = ω
    u = (∂ψ/∂y, -∂ψ/∂x)
    """

    def __init__(
        self,
        s_x: int = 64,
        s_y: int = 64,
        viscosity: float = 1e-3,
        domain_len: float = 2.0 * math.pi,
        device: str = "cpu"
    ):
        self.s_x = s_x
        self.s_y = s_y
        self.nu = viscosity
        self.L = domain_len
        self.device = device

        # Wavenumbers
        kx = torch.fft.fftfreq(s_x, d=domain_len / (2 * math.pi * s_x)).to(device)
        ky = torch.fft.fftfreq(s_y, d=domain_len / (2 * math.pi * s_y)).to(device)
        self.Kx, self.Ky = torch.meshgrid(kx, ky, indexing="ij")
        self.K2 = self.Kx**2 + self.Ky**2

        # Inverse Laplacian mask for streamfunction (avoid division by 0 at k=0)
        self.inv_K2 = torch.zeros_like(self.K2)
        nonzero_mask = self.K2 > 1e-10
        self.inv_K2[nonzero_mask] = 1.0 / self.K2[nonzero_mask]

        # Orszag 2/3 dealiasing mask
        k_max_x = (s_x // 3)
        k_max_y = (s_y // 3)
        self.dealias_mask = (torch.abs(self.Kx) <= k_max_x) & (torch.abs(self.Ky) <= k_max_y)

    def _compute_rhs(self, w_hat: torch.Tensor, forcing_hat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Computes RHS = -FFT(u · ∇ω) - ν |k|^2 w_hat + forcing_hat in spectral domain."""
        # Streamfunction: -∇^2 ψ = ω => ψ_hat = w_hat / |k|^2
        psi_hat = w_hat * self.inv_K2

        # Velocity in spectral domain
        ux_hat = 1j * self.Ky * psi_hat
        uy_hat = -1j * self.Kx * psi_hat

        # Vorticity gradients in spectral domain
        wx_hat = 1j * self.Kx * w_hat
        wy_hat = 1j * self.Ky * w_hat

        # Transform velocities and gradients to physical space for advection
        ux = torch.fft.ifft2(ux_hat).real
        uy = torch.fft.ifft2(uy_hat).real
        wx = torch.fft.ifft2(wx_hat).real
        wy = torch.fft.ifft2(wy_hat).real

        # Non-linear advection: u · ∇ω = ux * wx + uy * wy
        advection_phys = ux * wx + uy * wy
        advection_hat = torch.fft.fft2(advection_phys)

        # Apply dealiasing mask to non-linear term
        advection_hat = advection_hat * self.dealias_mask

        # Viscous diffusion term: -ν |k|^2 w_hat
        viscous_hat = -self.nu * self.K2 * w_hat

        rhs = -advection_hat + viscous_hat
        if forcing_hat is not None:
            rhs = rhs + forcing_hat

        return rhs

    def step_rk4(self, w_hat: torch.Tensor, dt: float, forcing_hat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """4th Order Runge-Kutta step in spectral domain."""
        k1 = self._compute_rhs(w_hat, forcing_hat)
        k2 = self._compute_rhs(w_hat + 0.5 * dt * k1, forcing_hat)
        k3 = self._compute_rhs(w_hat + 0.5 * dt * k2, forcing_hat)
        k4 = self._compute_rhs(w_hat + dt * k3, forcing_hat)

        w_hat_next = w_hat + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return w_hat_next

    def solve(
        self,
        w_initial: torch.Tensor,
        t_horizon: float = 1.0,
        num_steps: int = 100,
        save_steps: int = 10,
        forcing: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Solves vorticity field over time horizon T.
        Args:
            w_initial: Initial vorticity field (batch_size, s_x, s_y)
            t_horizon: Total simulation time
            num_steps: Number of integration steps
            save_steps: Number of temporal snapshots to return
        Returns:
            torch.Tensor of shape (batch_size, save_steps, s_x, s_y)
        """
        dt = t_horizon / num_steps
        batch_size = w_initial.shape[0]

        # Convert initial condition to spectral domain
        w_hat = torch.fft.fft2(w_initial)
        forcing_hat = torch.fft.fft2(forcing) if forcing is not None else None

        save_every = max(1, num_steps // save_steps)
        history = [w_initial.cpu()]

        current_w_hat = w_hat
        for step in range(1, num_steps + 1):
            current_w_hat = self.step_rk4(current_w_hat, dt, forcing_hat)
            if step % save_every == 0 and len(history) < save_steps:
                w_phys = torch.fft.ifft2(current_w_hat).real.cpu()
                history.append(w_phys)

        # Always include the true final state at t=T as the last snapshot
        w_final = torch.fft.ifft2(current_w_hat).real.cpu()
        history[-1] = w_final

        # Stack temporal trajectory: (batch_size, save_steps, s_x, s_y)
        trajectory = torch.stack(history, dim=1)
        return trajectory
