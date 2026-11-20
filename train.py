"""
Physics-Informed & Multi-Horizon Fourier Neural Operator (PINO) Training Pipeline.
Streams fresh diverse initial conditions (GRF across length scales/regularity,
multi-vortex clusters, and trigonometric shear flows) with periodic shift augmentation,
multi-step autoregressive rollout loss, circulation conservation, and spectral PDE
residual regularization, checkpointing on a held-out validation suite.
"""

import math
import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.optim as optim

from pino.analytics import build_normalized_input, generate_custom_vortex_field
from pino.config import PINOConfig
from pino.dataset.grf import GaussianRandomField2D
from pino.dataset.reference_solver import NavierStokes2DSolver
from pino.models.pino_net import PINO2D
from pino.physics.pde_loss import PINOLossEngine


def clip_grad_norm_safe(parameters, max_norm: float = 1.0):
    """
    Global gradient norm clipping supporting complex parameters on PyTorch MPS backend.
    """
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return 0.0

    total_sq_norm = 0.0
    for p in parameters:
        g = p.grad.data
        if g.is_complex():
            total_sq_norm += (torch.view_as_real(g).norm() ** 2).item()
        else:
            total_sq_norm += (g.norm() ** 2).item()

    total_norm = torch.sqrt(torch.tensor(total_sq_norm))
    clip_coef = max_norm / (total_norm + 1e-6)

    if clip_coef < 1.0:
        for p in parameters:
            p.grad.data.mul_(clip_coef)

    return total_norm


def normalize_grid(x_input: torch.Tensor) -> torch.Tensor:
    """
    Normalize grid channels from [0, 2π] to [0, 1].
    Input: (batch, 3, s_x, s_y) where channels 0,1 are grid_x, grid_y.
    Initial condition channel 2 is untouched.
    """
    x_out = x_input.clone()
    x_out[:, 0] = x_input[:, 0] / (2 * math.pi)
    x_out[:, 1] = x_input[:, 1] / (2 * math.pi)
    return x_out


def compute_batch_metrics_gpu(pred: torch.Tensor, target: torch.Tensor):
    """
    Computes diagnostic metrics as raw PyTorch GPU tensors without CPU synchronization stalls.
    """
    diff = pred - target
    mse = torch.mean(diff ** 2)
    mae = torch.mean(torch.abs(diff))
    l_inf = torch.max(torch.abs(diff))

    pred_energy = torch.sum(pred ** 2, dim=(-2, -1))
    target_energy = torch.sum(target ** 2, dim=(-2, -1)) + 1e-8
    energy_drift = torch.mean(torch.abs(pred_energy - target_energy) / target_energy) * 100.0

    diff_norms = torch.norm(diff, p=2, dim=(-2, -1))
    target_norms = torch.norm(target, p=2, dim=(-2, -1)) + 1e-8
    rel_l2 = torch.mean(diff_norms / target_norms)

    return mse, mae, l_inf, energy_drift, rel_l2


def _relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff_norms = torch.norm(pred - target, p=2, dim=(-2, -1))
    target_norms = torch.norm(target, p=2, dim=(-2, -1)) + 1e-8
    return torch.mean(diff_norms / target_norms)


def _solve_integer_horizon(
    solver: NavierStokes2DSolver,
    w0: torch.Tensor,
    horizon: int = 3,
    steps_per_unit: int = 50,
) -> torch.Tensor:
    """
    Integrates 2D Navier-Stokes via RK4 and returns exact integer-time snapshots
    [t=1, t=2, ..., t=horizon] of shape (B, horizon, s_x, s_y).
    """
    dt = 1.0 / float(steps_per_unit)
    with torch.no_grad():
        w_hat = torch.fft.fft2(w0)
        frames = []
        for _ in range(horizon):
            for _ in range(steps_per_unit):
                w_hat = solver.step_rk4(w_hat, dt)
            frames.append(torch.fft.ifft2(w_hat).real)
    return torch.stack(frames, dim=1)


def _sample_diverse_initial_fields(
    num_samples: int,
    s_x: int,
    s_y: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Samples a rich multi-regime initial vorticity batch:
      - 45% standard GRF (l=0.50, alpha=2.50)
      - 35% parametric GRF across l in [0.22, 0.90], alpha in [1.80, 3.40]
      - 10% randomized periodic Gaussian vortex clusters
      - 10% randomized trigonometric shear/lattice fields
    """
    n_std = max(1, int(round(num_samples * 0.45)))
    n_param = max(1, int(round(num_samples * 0.35)))
    n_vortex = max(0, int(round(num_samples * 0.10)))
    n_trig = max(0, num_samples - n_std - n_param - n_vortex)

    chunks = [
        GaussianRandomField2D(s_x=s_x, s_y=s_y, length_scale=0.5, alpha=2.5, device=device).sample(n_std)
    ]

    # Parametric GRF buckets for vectorized diversity
    remaining_param = n_param
    bucket_specs = [
        (0.24, 2.3),
        (0.32, 2.5),
        (0.40, 2.2),
        (0.65, 2.8),
        (0.85, 3.2),
        (0.35, 1.9),
    ]
    per_bucket = max(1, n_param // len(bucket_specs))
    for l_val, a_val in bucket_specs:
        if remaining_param <= 0:
            break
        cnt = min(per_bucket, remaining_param)
        chunks.append(
            GaussianRandomField2D(s_x=s_x, s_y=s_y, length_scale=l_val, alpha=a_val, device=device).sample(cnt)
        )
        remaining_param -= cnt
    if remaining_param > 0:
        chunks.append(
            GaussianRandomField2D(s_x=s_x, s_y=s_y, length_scale=0.4, alpha=2.5, device=device).sample(remaining_param)
        )

    # Multi-vortex Gaussian clusters
    for _ in range(n_vortex):
        k_vort = int(torch.randint(2, 7, (1,)).item())
        vortices = []
        for idx in range(k_vort):
            sign = 1.0 if idx % 2 == 0 else -1.0
            vortices.append({
                "x": float(torch.rand(1).item()),
                "y": float(torch.rand(1).item()),
                "strength": sign * float(0.6 + 1.2 * torch.rand(1).item()),
                "sigma": float(0.05 + 0.08 * torch.rand(1).item()),
            })
        chunks.append(generate_custom_vortex_field(vortices, s_x=s_x, s_y=s_y, device=device))

    # Randomized trigonometric shear / lattice modes
    if n_trig > 0:
        x = torch.linspace(0.0, 2.0 * math.pi * (s_x - 1) / s_x, s_x, device=device)
        y = torch.linspace(0.0, 2.0 * math.pi * (s_y - 1) / s_y, s_y, device=device)
        Xc, Yc = torch.meshgrid(x, y, indexing="ij")
        for _ in range(n_trig):
            kx1 = int(torch.randint(1, 4, (1,)).item())
            ky1 = int(torch.randint(1, 4, (1,)).item())
            kx2 = int(torch.randint(1, 4, (1,)).item())
            ph1 = float(2.0 * math.pi * torch.rand(1).item())
            ph2 = float(2.0 * math.pi * torch.rand(1).item())
            w = torch.sin(kx1 * Xc + ph1) * torch.cos(ky1 * Yc) + 0.45 * torch.cos(kx2 * Xc - ky1 * Yc + ph2)
            w = (w - w.mean()) / (w.std() + 1e-8)
            chunks.append(w.unsqueeze(0))

    combined = torch.cat(chunks, dim=0)
    perm = torch.randperm(combined.shape[0], device=device)
    return combined[perm][:num_samples]


def _build_validation_suite(
    solver: NavierStokes2DSolver,
    s_x: int,
    s_y: int,
    device: torch.device,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Constructs fixed held-out validation sets across standard, fine, broad, and OOD regimes."""
    torch.manual_seed(2026)
    w_std = GaussianRandomField2D(s_x=s_x, s_y=s_y, length_scale=0.5, alpha=2.5, device=device).sample(96)
    y_std = _solve_integer_horizon(solver, w_std, horizon=3)

    w_fine = GaussianRandomField2D(s_x=s_x, s_y=s_y, length_scale=0.25, alpha=2.5, device=device).sample(64)
    y_fine = _solve_integer_horizon(solver, w_fine, horizon=1)

    w_broad = torch.cat([
        GaussianRandomField2D(s_x=s_x, s_y=s_y, length_scale=0.35, alpha=2.2, device=device).sample(32),
        GaussianRandomField2D(s_x=s_x, s_y=s_y, length_scale=0.75, alpha=3.0, device=device).sample(32),
    ], dim=0)
    y_broad = _solve_integer_horizon(solver, w_broad, horizon=1)

    return {
        "std": (w_std, y_std),
        "fine": (w_fine, y_fine),
        "broad": (w_broad, y_broad),
    }


def _evaluate_validation(
    model: PINO2D,
    val_suite: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates single-step and 3-step rollout validation metrics."""
    model.eval()
    with torch.no_grad():
        w_std, y_std = val_suite["std"]
        p1 = model(build_normalized_input(w_std, device))
        p2 = model(build_normalized_input(p1[:, 0], device))
        p3 = model(build_normalized_input(p2[:, 0], device))

        std_t1_rel = _relative_l2(p1, y_std[:, 0:1]).item()
        std_t1_mse = torch.mean((p1 - y_std[:, 0:1]) ** 2).item()
        std_t3_rel = _relative_l2(p3, y_std[:, 2:3]).item()

        w_fine, y_fine = val_suite["fine"]
        pf = model(build_normalized_input(w_fine, device))
        fine_rel = _relative_l2(pf, y_fine[:, 0:1]).item()

        w_broad, y_broad = val_suite["broad"]
        pb = model(build_normalized_input(w_broad, device))
        broad_rel = _relative_l2(pb, y_broad[:, 0:1]).item()

    composite = 0.45 * std_t1_rel + 0.25 * std_t3_rel + 0.15 * fine_rel + 0.15 * broad_rel
    return {
        "std_t1_rel": std_t1_rel,
        "std_t1_mse": std_t1_mse,
        "std_t3_rel": std_t3_rel,
        "fine_rel": fine_rel,
        "broad_rel": broad_rel,
        "composite": composite,
    }


def train_pino(
    config: Optional[PINOConfig] = None,
    num_samples: int = 1600,
    epochs: int = 80,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
):
    """
    Streaming multi-regime, multi-horizon Physics-Informed Neural Operator training pipeline.
    """
    if config is None:
        config = PINOConfig()

    device = torch.device(config.device)
    s_x, s_y = config.domain.s_x, config.domain.s_y
    batch_size = batch_size if batch_size is not None else 32
    max_lr = lr if lr is not None else 2.2e-3

    print(
        f"\n--- Multi-Horizon PINO Training on {str(device).upper()} "
        f"({epochs} Epochs, {num_samples} Fresh Trajectories/Epoch, Batch {batch_size}, MaxLR {max_lr:.1e}) ---",
        flush=True,
    )

    solver = NavierStokes2DSolver(s_x=s_x, s_y=s_y, viscosity=config.pde.viscosity, device=device)
    loss_engine = PINOLossEngine(s_x=s_x, s_y=s_y, viscosity=config.pde.viscosity, device=device)
    val_suite = _build_validation_suite(solver, s_x, s_y, device)

    model = PINO2D.from_config(config.model).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Model Parameters: {total_params:,}", flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=max_lr * 0.5, weight_decay=2e-4)
    steps_per_epoch = max(1, num_samples // batch_size)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.15,
        div_factor=10.0,
        final_div_factor=100.0,
    )

    os.makedirs("checkpoints", exist_ok=True)
    best_score = float("inf")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # 1. Stream fresh diverse trajectories every epoch to prevent memorization
        w_epoch = _sample_diverse_initial_fields(num_samples, s_x, s_y, device)
        y_epoch = _solve_integer_horizon(solver, w_epoch, horizon=3, steps_per_unit=50)

        model.train()
        running_mse = 0.0
        running_rel = 0.0
        running_pde = 0.0
        num_batches = 0

        # Ramp rollout horizon weight smoothly after warm-up
        rollout_ramp = min(1.0, max(0.0, (epoch - 6) / 14.0))
        w1 = 1.0 - 0.50 * rollout_ramp
        w2 = 0.30 * rollout_ramp
        w3 = 0.20 * rollout_ramp

        perm = torch.randperm(num_samples, device=device)
        for start in range(0, num_samples - batch_size + 1, batch_size):
            idx = perm[start : start + batch_size]
            wb = w_epoch[idx]
            yb = y_epoch[idx]

            # Random periodic torus shift augmentation (translation equivariance)
            sx = int(torch.randint(0, s_x, (1,)).item())
            sy = int(torch.randint(0, s_y, (1,)).item())
            if sx != 0 or sy != 0:
                wb = torch.roll(wb, shifts=(sx, sy), dims=(-2, -1))
                yb = torch.roll(yb, shifts=(sx, sy), dims=(-2, -1))

            optimizer.zero_grad()

            p1 = model(build_normalized_input(wb, device))
            loss_t1 = _relative_l2(p1, yb[:, 0:1]) + 0.05 * torch.mean((p1 - yb[:, 0:1]) ** 2)

            # Circulation conservation + exact spectral PDE residual regularization
            loss_circ = torch.mean((torch.mean(p1[:, 0], dim=(-2, -1)) - torch.mean(wb, dim=(-2, -1))) ** 2)
            traj_2step = torch.stack([wb, p1[:, 0]], dim=1)
            pde_res = loss_engine.compute_pde_residual(traj_2step, dt=1.0)
            loss_pde = torch.mean(pde_res ** 2)

            if rollout_ramp > 0.0:
                p2 = model(build_normalized_input(p1[:, 0], device))
                p3 = model(build_normalized_input(p2[:, 0], device))
                loss_t2 = _relative_l2(p2, yb[:, 1:2])
                loss_t3 = _relative_l2(p3, yb[:, 2:3])
                data_loss = w1 * loss_t1 + w2 * loss_t2 + w3 * loss_t3
            else:
                data_loss = loss_t1

            total_loss = data_loss + 0.005 * loss_pde + 0.05 * loss_circ
            total_loss.backward()
            clip_grad_norm_safe(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                mse_b, _, _, _, rel_b = compute_batch_metrics_gpu(p1, yb[:, 0:1])
                running_mse += mse_b
                running_rel += rel_b
                running_pde += torch.sqrt(loss_pde)
            num_batches += 1

        n = max(num_batches, 1)
        avg_mse = (running_mse / n).item()
        avg_rel = (running_rel / n).item()
        avg_pde = (running_pde / n).item()

        val_metrics = _evaluate_validation(model, val_suite, device)
        if val_metrics["composite"] < best_score:
            best_score = val_metrics["composite"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "loss": val_metrics["std_t1_mse"],
                    "val_rel_l2": val_metrics["std_t1_rel"],
                    "val_rollout3_rel_l2": val_metrics["std_t3_rel"],
                    "val_fine_rel_l2": val_metrics["fine_rel"],
                    "val_broad_rel_l2": val_metrics["broad_rel"],
                },
                "checkpoints/pino_best.pt",
            )

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            current_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time
            print(
                f"  Ep [{epoch:03d}/{epochs:03d}] ({elapsed:5.1f}s) | "
                f"TrRel: {avg_rel*100:5.2f}% | PDE: {avg_pde:.3f} | "
                f"Val std(t=1): {val_metrics['std_t1_rel']*100:5.2f}% | "
                f"Val std(t=3): {val_metrics['std_t3_rel']*100:5.2f}% | "
                f"Val fine(l=0.25): {val_metrics['fine_rel']*100:5.2f}% | "
                f"Val MSE: {val_metrics['std_t1_mse']:.4f} | LR: {current_lr:.2e}",
                flush=True,
            )

    total_time = time.time() - start_time
    print(
        f"\n--- Training Complete in {total_time:.1f}s! Best Checkpoint Saved to 'checkpoints/pino_best.pt' "
        f"(Best Composite Score: {best_score*100:.2f}%) ---",
        flush=True,
    )
    return model


if __name__ == "__main__":
    train_pino(num_samples=1600, epochs=80, batch_size=32, lr=2.2e-3)
