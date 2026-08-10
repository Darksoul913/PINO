"""
Corrected & Stable Physics-Gated Training Pipeline for PINO (Navier-Stokes 2D).
1. Removes conflicting IC loss penalty on final time snapshot t=1.0.
2. Implements true global gradient norm clipping across complex and real parameters on Metal MPS.
3. Warmup Phase (Epochs 1-40): Pure data learning (lambda_data=1.0, lambda_pde=0.0).
4. Physics Phase (Epochs 41-200): Physics residual ramp (lambda_pde=1e-4 -> 1e-3).
"""

import os
import torch
import torch.optim as optim
from tqdm import tqdm

from pino.config import PINOConfig, LossConfig
from pino.models.pino_net import PINO2D
from pino.physics.pde_loss import PINOLossEngine
from pino.dataset.pde_dataset import get_pde_dataloader


def clip_grad_norm_safe(parameters, max_norm: float = 1.0):
    """
    True Global Gradient Norm Clipping supporting complex parameters on PyTorch MPS backend.
    Calculates total global norm across all parameters combined (viewing complex grads as real)
    and scales all gradients uniformly if total_norm > max_norm.
    """
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return torch.tensor(0.0)

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


def train_pino(config: PINOConfig = None, num_samples: int = 1000, epochs: int = 200, base_lr: float = 4e-4):
    """
    Physics-Gated Training Loop for PINO.
    """
    if config is None:
        config = PINOConfig()

    device = torch.device(config.device)
    print(f"--- Starting Corrected PINO Training Pipeline on Device: {device} ({epochs} Epochs, {num_samples} Samples, lr={base_lr}) ---")

    # 1. DataLoader with ground-truth reference data
    dataloader = get_pde_dataloader(
        num_samples=num_samples,
        batch_size=config.batch_size,
        s_x=config.domain.s_x,
        s_y=config.domain.s_y,
        shuffle=True,
        generate_reference_data=True,
        device=str(device)
    )

    # 2. Model & Loss Engine
    model = PINO2D.from_config(config.model).to(device)
    loss_engine = PINOLossEngine(
        s_x=config.domain.s_x,
        s_y=config.domain.s_y,
        viscosity=config.pde.viscosity,
        device=str(device)
    )

    # 3. AdamW Optimizer & Cosine Annealing Scheduler (base_lr=4e-4 -> 1e-6)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    os.makedirs("checkpoints", exist_ok=True)
    best_data_loss = float("inf")

    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()

        # Warmup Schedule:
        # Epochs 1-40: Pure data learning (lambda_pde = 0.0)
        # Epochs 41-200: Gentle physics regularization (lambda_pde = 1e-4 -> 1e-3)
        if epoch <= 40:
            lambda_data = 1.0
            lambda_pde = 0.0
        else:
            ramp = (epoch - 40) / (epochs - 40)
            lambda_data = 1.0
            lambda_pde = 1e-4 + ramp * (1e-3 - 1e-4)

        running_data, running_pde, running_total = 0.0, 0.0, 0.0

        for batch in dataloader:
            x_input = batch["x_input"].to(device)   # (batch, 3, s_x, s_y)
            target = batch.get("target", None)
            if target is not None:
                target = target[:, -1:].to(device)   # Match final snapshot shape (batch, 1, s_x, s_y)

            optimizer.zero_grad()

            # Forward pass (predicts vorticity field u at t_final = 1.0)
            pred_u = model(x_input)  # (batch, 1, s_x, s_y)

            # 1. Trajectory Data Loss against target at t_final
            l_data = loss_engine.relative_l2_loss(pred_u, target) if target is not None else torch.tensor(0.0, device=device)

            # 2. Exact PDE Residual
            pred_u_eval = pred_u if pred_u.shape[1] > 1 else pred_u.repeat(1, 2, 1, 1)

            if lambda_pde > 0:
                raw_pde = torch.mean(torch.norm(loss_engine.compute_pde_residual(pred_u_eval), p=2, dim=(-2, -1)))
                l_pde = torch.clamp(raw_pde, max=10.0)
                total_loss = lambda_data * l_data + lambda_pde * l_pde
            else:
                with torch.no_grad():
                    raw_pde = torch.mean(torch.norm(loss_engine.compute_pde_residual(pred_u_eval), p=2, dim=(-2, -1)))
                l_pde = raw_pde
                total_loss = lambda_data * l_data

            total_loss.backward()

            # True Global Gradient Norm Clipping supporting complex parameters
            clip_grad_norm_safe(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_data += l_data.item()
            running_pde += l_pde.item()
            running_total += total_loss.item()

        scheduler.step()

        num_batches = len(dataloader)
        avg_data = running_data / num_batches
        avg_pde = running_pde / num_batches
        avg_total = running_total / num_batches

        # Save checkpoint on true data improvement
        if avg_data < best_data_loss and epoch > 5:
            best_data_loss = avg_data
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "loss": best_data_loss
            }, "checkpoints/pino_best.pt")

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            current_lr = scheduler.get_last_lr()[0]
            pde_str = " [Warmup]" if epoch <= 40 else f" (λ={lambda_pde:.1e})"
            print(f"Epoch [{epoch:03d}/{epochs:03d}] | Total: {avg_total:.4f} | Data Loss: {avg_data:.6f} | PDE Loss{pde_str}: {avg_pde:.4e} | LR: {current_lr:.2e}")

    print(f"\n--- Training Completed! Best Checkpoint Saved to 'checkpoints/pino_best.pt' (Data Loss: {best_data_loss:.6f}) ---")
    return model


if __name__ == "__main__":
    train_pino(num_samples=1000, epochs=200, base_lr=4e-4)
