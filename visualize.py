"""
Spatiotemporal Field Visualization Pipeline for PINO.
Generates publication-quality side-by-side contour plots comparing Initial Condition,
RK4 Reference Solver, PINO Predictions, Zero-Shot Super-Resolution, and Point-wise Error.
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plot generation
import matplotlib.pyplot as plt

from pino.config import PINOConfig
from pino.models.pino_net import PINO2D
from pino.dataset.grf import GaussianRandomField2D
from pino.dataset.reference_solver import NavierStokes2DSolver


def visualize_pino_fields(
    save_path: str = "plots/pino_spatiotemporal_comparison.png",
    s_x: int = 64,
    s_y: int = 64,
    eval_s: int = 256
):
    """
    Generates and saves a 5-panel diagnostic figure comparing fluid fields.
    """
    config = PINOConfig()
    device = config.device
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 1. Sample GRF initial condition a(x, y)
    grf = GaussianRandomField2D(s_x=s_x, s_y=s_y, length_scale=0.5, alpha=2.5, device=device)
    w0 = grf.sample(num_samples=1)  # (1, 64, 64)

    # 2. Run RK4 Reference Numerical Solver
    solver = NavierStokes2DSolver(s_x=s_x, s_y=s_y, viscosity=1e-3, device=device)
    trajectory_ref = solver.solve(w0, t_horizon=1.0, num_steps=50, save_steps=10)
    u_rk4 = trajectory_ref[0, -1].cpu().numpy()  # Final snapshot (64, 64)

    # 3. Prepare PINO input grid
    x = torch.linspace(0, 2 * 3.141592653589793 * (s_x - 1) / s_x, s_x, device=device)
    y = torch.linspace(0, 2 * 3.141592653589793 * (s_y - 1) / s_y, s_y, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    x_input = torch.cat([grid_x.unsqueeze(0), grid_y.unsqueeze(0), w0], dim=0).unsqueeze(0).to(device)

    # 4. Run PINO Model (Standard & Zero-Shot Super-Resolution)
    model = PINO2D.from_config(config.model).to(device)
    checkpoint_path = "checkpoints/pino_best.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[*] Loaded trained model checkpoint from '{checkpoint_path}' (Loss: {checkpoint.get('loss', 0.0):.4f})")
    else:
        print("[!] Warning: No trained checkpoint found in 'checkpoints/pino_best.pt'. Using untrained initial weights.")
    model.eval()

    with torch.no_grad():
        u_pino_tensor = model(x_input)                           # (1, 1, 64, 64)
        u_super_res_tensor = model(x_input, target_s=(eval_s, eval_s))  # (1, 1, 256, 256)

    u_pino = u_pino_tensor[0, 0].cpu().numpy()
    u_super_res = u_super_res_tensor[0, 0].cpu().numpy()
    a_init = w0[0].cpu().numpy()

    # 5. Compute Absolute Error Heatmap
    abs_error = np.abs(u_pino - u_rk4)

    # 6. Plotting 1x5 Figure Panel
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), dpi=300)
    plt.subplots_adjust(wspace=0.35)

    # Panel 1: Initial Condition a(x, y)
    im0 = axes[0].imshow(a_init, cmap="viridis", origin="lower")
    axes[0].set_title("Initial Field $a(x, y)$\n($64 \\times 64$)", fontsize=11, fontweight="bold")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 2: RK4 Reference Solver
    im1 = axes[1].imshow(u_rk4, cmap="twilight", origin="lower")
    axes[1].set_title("RK4 Solver $u_{rk4}(t)$\n($64 \\times 64$)", fontsize=11, fontweight="bold")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: PINO Output (Standard)
    im2 = axes[2].imshow(u_pino, cmap="twilight", origin="lower")
    axes[2].set_title("PINO Output $u_{pino}(t)$\n($64 \\times 64$)", fontsize=11, fontweight="bold")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4: PINO Zero-Shot Super-Resolution
    im3 = axes[3].imshow(u_super_res, cmap="twilight", origin="lower")
    axes[3].set_title("Zero-Shot Super-Res\n($256 \\times 256$)", fontsize=11, fontweight="bold")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    # Panel 5: Absolute Error Heatmap
    im4 = axes[4].imshow(abs_error, cmap="magma", origin="lower")
    axes[4].set_title("Absolute Error\n$|u_{pino} - u_{rk4}|$", fontsize=11, fontweight="bold")
    fig.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Physics-Informed Neural Operator (PINO) Spatiotemporal Diagnostic Suite", fontsize=14, fontweight="bold", y=1.03)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"[✓] Visualization figure saved successfully to: '{save_path}'")
    return save_path


if __name__ == "__main__":
    visualize_pino_fields()
