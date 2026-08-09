"""
PyTorch Dataset and DataLoader Pipeline for PINO.
Generates input spatial coordinate grids concatenated with GRF initial conditions.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional

from pino.dataset.grf import GaussianRandomField2D
from pino.dataset.reference_solver import NavierStokes2DSolver


class PDEDataset2D(Dataset):
    """
    2D PDE Dataset.
    Generates spatial grid coordinates (x, y) concatenated with GRF initial fields a(x, y).
    Optionally computes ground-truth reference trajectory using Navier-Stokes solver.
    """

    def __init__(
        self,
        num_samples: int = 100,
        s_x: int = 64,
        s_y: int = 64,
        length_scale: float = 0.5,
        alpha: float = 2.5,
        generate_reference_data: bool = False,
        viscosity: float = 1e-3,
        device: str = "cpu"
    ):
        self.num_samples = num_samples
        self.s_x = s_x
        self.s_y = s_y
        self.device = device

        # 1. Sample GRF Initial Conditions a(x, y) of shape (num_samples, s_x, s_y)
        grf = GaussianRandomField2D(s_x=s_x, s_y=s_y, length_scale=length_scale, alpha=alpha, device=device)
        self.initial_conditions = grf.sample(num_samples=num_samples)

        # 2. Build 2D Spatial Grid Coordinates [0, 2π]
        x = torch.linspace(0, 2 * 3.141592653589793, s_x, device=device)
        y = torch.linspace(0, 2 * 3.141592653589793, s_y, device=device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        # Shape: (2, s_x, s_y)
        self.grid = torch.stack([grid_x, grid_y], dim=0)

        # 3. Compute ground-truth trajectory if requested
        self.trajectories: Optional[torch.Tensor] = None
        if generate_reference_data:
            solver = NavierStokes2DSolver(s_x=s_x, s_y=s_y, viscosity=viscosity, device=device)
            # Solve trajectory for each batch sample
            self.trajectories = solver.solve(self.initial_conditions, t_horizon=1.0, num_steps=50, save_steps=10)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        a_init = self.initial_conditions[idx].unsqueeze(0)  # (1, s_x, s_y)
        # Input tensor concatenates (grid_x, grid_y, a_init) -> (3, s_x, s_y)
        x_input = torch.cat([self.grid, a_init], dim=0)

        sample = {
            "x_input": x_input,        # (3, s_x, s_y)
            "a_init": a_init,          # (1, s_x, s_y)
            "grid": self.grid          # (2, s_x, s_y)
        }

        if self.trajectories is not None:
            sample["target"] = self.trajectories[idx]  # (T, s_x, s_y)

        return sample


def get_pde_dataloader(
    num_samples: int = 100,
    batch_size: int = 8,
    s_x: int = 64,
    s_y: int = 64,
    shuffle: bool = True,
    generate_reference_data: bool = False,
    device: str = "cpu"
) -> DataLoader:
    """Helper factory function to instantiate DataLoader."""
    dataset = PDEDataset2D(
        num_samples=num_samples,
        s_x=s_x,
        s_y=s_y,
        generate_reference_data=generate_reference_data,
        device=device
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
