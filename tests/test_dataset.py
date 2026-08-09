"""Unit tests for PINO dataset module (GRF, Reference Solver, DataLoader)."""

import torch
from pino.dataset.grf import GaussianRandomField2D
from pino.dataset.reference_solver import NavierStokes2DSolver
from pino.dataset.pde_dataset import PDEDataset2D, get_pde_dataloader


def test_grf_sampling():
    grf = GaussianRandomField2D(s_x=64, s_y=64, length_scale=0.5, alpha=2.5)
    samples = grf.sample(num_samples=5)
    assert samples.shape == (5, 64, 64)
    assert not torch.isnan(samples).any()
    print("GRF Sampling test passed!")


def test_reference_solver():
    grf = GaussianRandomField2D(s_x=32, s_y=32, length_scale=0.5, alpha=2.5)
    w0 = grf.sample(num_samples=2)
    solver = NavierStokes2DSolver(s_x=32, s_y=32, viscosity=1e-3)
    trajectory = solver.solve(w0, t_horizon=0.1, num_steps=10, save_steps=5)
    assert trajectory.shape == (2, 5, 32, 32)
    assert not torch.isnan(trajectory).any()
    print("Navier-Stokes Reference Solver test passed!")


def test_pde_dataset_and_dataloader():
    dataloader = get_pde_dataloader(num_samples=10, batch_size=4, s_x=64, s_y=64, shuffle=False)
    batch = next(iter(dataloader))
    assert batch["x_input"].shape == (4, 3, 64, 64)
    assert batch["a_init"].shape == (4, 1, 64, 64)
    assert batch["grid"].shape == (4, 2, 64, 64)
    print("PDE Dataset & DataLoader test passed!")


if __name__ == "__main__":
    test_grf_sampling()
    test_reference_solver()
    test_pde_dataset_and_dataloader()
