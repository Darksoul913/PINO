"""
Central Configuration System for PINO Framework.
Provides strongly-typed dataclasses for hyperparameter and PDE problem settings.
"""

import torch
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DomainConfig:
    """Spatial and temporal domain configuration."""
    s_x: int = 64                  # Input spatial grid x-resolution
    s_y: int = 64                  # Input spatial grid y-resolution
    eval_s_x: int = 256            # Super-resolution target x-resolution
    eval_s_y: int = 256            # Super-resolution target y-resolution
    t_horizon: float = 1.0         # Simulation time horizon T
    spatial_domain: Tuple[float, float] = (0.0, 2 * 3.141592653589793)  # [0, 2π]


@dataclass
class PDEConfig:
    """Governing PDE equation hyperparameters."""
    name: str = "navier_stokes_2d"  # PDE equation name ("navier_stokes_2d" or "burgers_2d")
    reynolds_number: float = 100.0   # Reynolds number Re
    viscosity: float = 1e-3         # Kinematic viscosity nu = 1 / Re
    forcing_amplitude: float = 0.1   # External forcing function scale f(x)
    grf_length_scale: float = 0.5    # Gaussian Random Field covariance length scale l
    grf_alpha: float = 2.5           # GRF smoothness exponent alpha


@dataclass
class ModelConfig:
    """PINO Neural Network Architecture hyperparameters."""
    in_channels: int = 3           # Input channels: (x, y, a(x))
    out_channels: int = 1          # Output state field (e.g. vorticity or velocity)
    hidden_dim: int = 64           # Lifting width dv
    modes1: int = 24               # Number of Fourier modes in x-dimension (k_max_x)
    modes2: int = 24               # Number of Fourier modes in y-dimension (k_max_y)
    num_layers: int = 4            # Number of stacked Spectral Conv layers
    padding: int = 8               # Non-periodic boundary padding (if needed)


@dataclass
class LossConfig:
    """Multi-Objective Physics and Data Loss Weights."""
    weight_data: float = 1.0       # Weight for data loss L_data
    weight_pde: float = 5.0        # Weight for physics residual loss L_pde (increased for high-gradient boundary physics)
    weight_ic: float = 10.0        # Weight for initial condition loss L_ic/bc
    use_dealiasing: bool = True    # Apply Orszag 3/2 rule for non-linear terms



def get_default_device() -> str:
    """Helper function to auto-detect available GPU accelerator (MPS for Apple Silicon, CUDA, or CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class PINOConfig:
    """Master configuration container for PINO."""
    domain: DomainConfig = field(default_factory=DomainConfig)
    pde: PDEConfig = field(default_factory=PDEConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    device: str = field(default_factory=get_default_device)  # Auto-detected target compute device
    batch_size: int = 8
    learning_rate: float = 1e-3
    epochs: int = 100
