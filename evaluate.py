"""
Quantitative Mathematical Evaluation Suite for PINO Framework.
Computes Relative L2 Error, MAE, RMSE, L_infinity Error, Mass/Circulation Drift,
Energy Invariant Conservation, and Exact Spectral PDE Residual Norms.
"""

import os
import torch
import numpy as np
from typing import Dict

from pino.config import PINOConfig
from pino.models.pino_net import PINO2D
from pino.physics.pde_loss import PINOLossEngine
from pino.dataset.grf import GaussianRandomField2D
from pino.dataset.reference_solver import NavierStokes2DSolver


def evaluate_pino_metrics(
    checkpoint_path: str = "checkpoints/pino_best.pt",
    num_test_samples: int = 20,
    s_x: int = 64,
    s_y: int = 64
) -> Dict[str, float]:
    """
    Computes exact quantitative metrics across test dataset instances.
    """
    config = PINOConfig()
    device = config.device

    print("==========================================================================")
    print("      QUANTITATIVE MATHEMATICAL EVALUATION: PINO FRAMEWORK                ")
    print("==========================================================================\n")
    print(f"[*] Target Compute Device: {device.upper()}")
    print(f"[*] Test Dataset Size: {num_test_samples} independent GRF samples")
    print(f"[*] Spatial Resolution: {s_x} x {s_y}\n")

    # 1. Instantiate Model and Load Checkpoint
    model = PINO2D.from_config(config.model).to(device)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[*] Successfully loaded trained checkpoint: '{checkpoint_path}' (Training Loss: {checkpoint.get('loss', 0.0):.4f})\n")
    else:
        print("[!] Warning: No checkpoint found. Evaluating initialized model weights.\n")

    model.eval()

    # 2. Instantiate Solvers and Loss Engine
    grf = GaussianRandomField2D(s_x=s_x, s_y=s_y, length_scale=0.5, alpha=2.5, device=device)
    solver = NavierStokes2DSolver(s_x=s_x, s_y=s_y, viscosity=config.pde.viscosity, device=device)
    loss_engine = PINOLossEngine(s_x=s_x, s_y=s_y, viscosity=config.pde.viscosity, device=device)

    # 3. Generate Test Samples and Compute Predictions
    w0_all = grf.sample(num_samples=num_test_samples)
    u_ref_all = solver.solve(w0_all, t_horizon=1.0, num_steps=50, save_steps=10)[:, -1]  # (N, s_x, s_y)

    x = torch.linspace(0, 2 * 3.141592653589793 * (s_x - 1) / s_x, s_x, device=device)
    y = torch.linspace(0, 2 * 3.141592653589793 * (s_y - 1) / s_y, s_y, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(num_test_samples, 1, 1, 1)

    x_inputs = torch.cat([grid, w0_all.unsqueeze(1)], dim=1).to(device)

    with torch.no_grad():
        u_pred_all = model(x_inputs).squeeze(1)  # (N, s_x, s_y)

    # Convert to NumPy for metrics
    pred = u_pred_all.cpu().numpy()
    ref = u_ref_all.cpu().numpy()

    # 4. Compute Quantitative Mathematical Metrics
    # (a) Relative L2 Error
    diff_l2 = np.linalg.norm(pred - ref, axis=(-2, -1))
    ref_l2 = np.linalg.norm(ref, axis=(-2, -1)) + 1e-8
    rel_l2_errors = (diff_l2 / ref_l2) * 100.0
    mean_rel_l2 = float(np.mean(rel_l2_errors))

    # (b) Mean Absolute Error (MAE) & Root Mean Squared Error (RMSE)
    mae = float(np.mean(np.abs(pred - ref)))
    rmse = float(np.sqrt(np.mean((pred - ref)**2)))

    # (c) Max Point-wise Error (L_infinity)
    l_inf = float(np.max(np.abs(pred - ref)))

    # (d) Physical Conservation Law Metrics (Circulation & Enstrophy Energy)
    # Circulation = Sum(u) * dx dy
    circ_pred = np.sum(pred, axis=(-2, -1))
    circ_ref = np.sum(ref, axis=(-2, -1))
    circ_drift = float(np.mean(np.abs(circ_pred - circ_ref) / (np.abs(circ_ref) + 1e-8))) * 100.0

    # Enstrophy Energy = Sum(u^2)
    energy_pred = np.sum(pred**2, axis=(-2, -1))
    energy_ref = np.sum(ref**2, axis=(-2, -1))
    energy_drift = float(np.mean(np.abs(energy_pred - energy_ref) / (energy_ref + 1e-8))) * 100.0

    # (e) Physics PDE Residual Norm (Spectral Norm of P(u) = 0)
    with torch.no_grad():
        # Predict 2-step trajectory for PDE residual evaluation
        pred_u_2step = u_pred_all.unsqueeze(1).repeat(1, 2, 1, 1)
        pde_res = loss_engine.compute_pde_residual(pred_u_2step)
        pde_res_norm = float(torch.mean(torch.norm(pde_res, p=2, dim=(-2, -1))).item())

    # 5. Format & Print Mathematical Text Metrics Table
    metrics = {
        "rel_l2_error_pct": mean_rel_l2,
        "mae": mae,
        "rmse": rmse,
        "max_l_inf_error": l_inf,
        "circulation_drift_pct": circ_drift,
        "energy_drift_pct": energy_drift,
        "pde_residual_norm": pde_res_norm
    }

    print("--------------------------------------------------------------------------")
    print(" METRIC CATEGORY                     VALUE          TARGET BENCHMARK       ")
    print("--------------------------------------------------------------------------")
    print(f" Relative L2 Error (Rel-L2)       :  {mean_rel_l2:6.2f} %        < 2.50 %              ")
    print(f" Mean Absolute Error (MAE)        :  {mae:8.4f}        < 0.0500              ")
    print(f" Root Mean Squared Error (RMSE)   :  {rmse:8.4f}        < 0.0800              ")
    print(f" Max Point-wise Error (L_inf)     :  {l_inf:8.4f}        Local Gradient Peaks  ")
    print(f" Circulation Invariant Drift      :  {circ_drift:6.2f} %        < 1.00 %              ")
    print(f" Kinetic Energy / Enstrophy Drift :  {energy_drift:6.2f} %        < 1.00 %              ")
    print(f" Exact PDE Residual Norm ||P(u)|| :  {pde_res_norm:8.4f}        < 1e-2                ")
    print("--------------------------------------------------------------------------\n")

    return metrics


if __name__ == "__main__":
    evaluate_pino_metrics()
