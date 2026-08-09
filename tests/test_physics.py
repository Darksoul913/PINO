"""Unit tests for Spectral Differentiation and PDE Loss Engine."""

import torch
import math
from pino.physics.spectral_diff import SpectralDiff2D
from pino.physics.pde_loss import PINOLossEngine


def test_exact_spectral_derivatives():
    s_x, s_y = 64, 64
    diff = SpectralDiff2D(s_x=s_x, s_y=s_y)

    x = torch.linspace(0, 2 * math.pi * (s_x - 1) / s_x, s_x)
    y = torch.linspace(0, 2 * math.pi * (s_y - 1) / s_y, s_y)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

    # Analytical test function: u(x, y) = sin(x) * cos(y)
    u = torch.sin(grid_x) * torch.cos(grid_y)

    # Analytical exact derivatives
    du_dx_exact = torch.cos(grid_x) * torch.cos(grid_y)
    du_dy_exact = -torch.sin(grid_x) * torch.sin(grid_y)
    lap_exact = -2.0 * torch.sin(grid_x) * torch.cos(grid_y)

    # Spectral derivatives
    du_dx_num = diff.diff_x(u)
    du_dy_num = diff.diff_y(u)
    lap_num = diff.laplacian(u)

    # Errors
    err_dx = torch.max(torch.abs(du_dx_num - du_dx_exact)).item()
    err_dy = torch.max(torch.abs(du_dy_num - du_dy_exact)).item()
    err_lap = torch.max(torch.abs(lap_num - lap_exact)).item()

    assert err_dx < 1e-3, f"∂u/∂x error too high: {err_dx}"
    assert err_dy < 1e-3, f"∂u/∂y error too high: {err_dy}"
    assert err_lap < 1e-3, f"Laplacian error too high: {err_lap}"

    print(f"Exact Fourier Differentiation test passed! (Max Err dx: {err_dx:.2e}, lap: {err_lap:.2e})")


def test_pino_loss_engine():
    loss_engine = PINOLossEngine(s_x=32, s_y=32, viscosity=1e-3)
    
    pred_u = torch.randn(2, 5, 32, 32)
    a_init = pred_u[:, 0:1]
    target_u = pred_u + 0.01 * torch.randn_like(pred_u)

    losses = loss_engine(pred_u=pred_u, a_init=a_init, target_u=target_u)
    assert "loss_total" in losses
    assert "loss_ic" in losses
    assert "loss_data" in losses
    assert "loss_pde" in losses
    assert not torch.isnan(losses["loss_total"]).any()

    print("PINO Loss Engine test passed!")


if __name__ == "__main__":
    test_exact_spectral_derivatives()
    test_pino_loss_engine()
