"""
Stable Physics-Gated Training Loop for PINO (Navier-Stokes 2D).
1. Pure Data Warmup Phase (Epochs 1-40): lambda_pde = 0.0 to master spatial phase alignment.
2. Physics Loss Gating & Clamping (Epochs 41-200): lambda_pde = 1e-4 -> 1e-3, l_pde = torch.clamp(raw_pde, max=10.0).
3. Complex-Safe Gradient Clipping (max_norm = 0.5) on PyTorch MPS backend.
4. AdamW (lr=4e-4, weight_decay=1e-4) + CosineAnnealingLR (decay 4e-4 -> 1e-6).
"""

import os
import torch
import torch.optim as optim
from tqdm import tqdm

from pino.config import PINOConfig, LossConfig
from pino.models.pino_net import PINO2D
from pino.physics.pde_loss import PINOLossEngine
from pino.dataset.pde_dataset import get_pde_dataloader


def clip_grad_norm_safe(parameters, max_norm: float = 0.5):
    """
    Safe gradient norm clipping supporting complex parameters on PyTorch MPS backend.
    Views complex gradients as real tensors (torch.view_as_real) to prevent
    RuntimeError: norm ops are not supported for complex yet on MPS.
    """
    for p in parameters:
        if p.grad is not None:
            grad = p.grad.data
            if grad.is_complex():
                grad_norm = torch.view_as_real(grad).norm()
            else:
                grad_norm = grad.norm()
            if grad_norm > max_norm:
                p.grad.data = grad * (max_norm / (grad_norm + 1e-6))


def train_pino(config: PINOConfig = None, num_samples: int = 1000, epochs: int = 200, base_lr: float = 4e-4):
    """
    Physics-Gated Training Loop for PINO.
    """
    if config is None:
        config = PINOConfig()

    device = torch.device(config.device)
    print(f"--- Starting Physics-Gated PINO Training Pipeline on Device: {device} ({epochs} Epochs, {num_samples} Samples, lr={base_lr}) ---")

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

    # 4. Training Loop with Physics Loss Gating & Clamping
    for epoch in range(1, epochs + 1):
        model.train()

        # Physics Loss Gating (Warmup Schedule):
        # Epochs 1-40: Pure data learning (lambda_pde = 0.0)
        # Epochs 41-200: Gentle physics regularization (lambda_pde = 1e-4 -> 1e-3)
        if epoch <= 40:
            lambda_data = 1.0
            lambda_ic = 1.0
            lambda_pde = 0.0
        else:
            ramp = (epoch - 40) / (epochs - 40)
            lambda_data = 1.0
            lambda_ic = 1.0
            lambda_pde = 1e-4 + ramp * (1e-3 - 1e-4)

        running_data, running_ic, running_pde, running_total = 0.0, 0.0, 0.0, 0.0

        for batch in dataloader:
            x_input = batch["x_input"].to(device)   # (batch, 3, s_x, s_y)
            a_init = batch["a_init"].to(device)     # (batch, 1, s_x, s_y)
            target = batch.get("target", None)
            if target is not None:
                target = target[:, -1:].to(device)   # Match final snapshot shape (batch, 1, s_x, s_y)

            optimizer.zero_grad()

            # Forward pass
            pred_u = model(x_input)  # (batch, 1, s_x, s_y)

            # 1. Trajectory Data Loss
            l_data = loss_engine.relative_l2_loss(pred_u, target) if target is not None else torch.tensor(0.0, device=device)

            # 2. Initial Condition Loss at t=0
            l_ic = loss_engine.compute_ic_loss(pred_u, a_init)

            # 3. Exact PDE Residual (only evaluated when lambda_pde > 0)
            if lambda_pde > 0:
                pred_u_eval = pred_u if pred_u.shape[1] > 1 else pred_u.repeat(1, 2, 1, 1)
                raw_pde = torch.mean(torch.norm(loss_engine.compute_pde_residual(pred_u_eval), p=2, dim=(-2, -1)))
                # GUARD: Clamp PDE residual to prevent gradient explosion
                l_pde = torch.clamp(raw_pde, max=10.0)
            else:
                l_pde = torch.tensor(0.0, device=device)

            total_loss = lambda_data * l_data + lambda_ic * l_ic + lambda_pde * l_pde

            total_loss.backward()

            # GUARD: Strict complex-safe gradient norm clipping for Fourier layers
            clip_grad_norm_safe(model.parameters(), max_norm=0.5)

            optimizer.step()

            running_data += l_data.item()
            running_ic += l_ic.item()
            running_pde += l_pde.item()
            running_total += total_loss.item()

        scheduler.step()

        num_batches = len(dataloader)
        avg_data = running_data / num_batches
        avg_ic = running_ic / num_batches
        avg_pde = running_pde / num_batches
        avg_total = running_total / num_batches

        # Save checkpoint on true data improvement
        if avg_data < best_data_loss and epoch > 10:
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
            print(f"Epoch [{epoch:03d}/{epochs:03d}] | Total: {avg_total:.4f} | IC: {avg_ic:.4f} | Data Loss: {avg_data:.6f} | PDE Loss: {avg_pde:.4e} | LR: {current_lr:.2e}")

    print(f"\n--- Physics-Gated Training Completed! Best Checkpoint Saved to 'checkpoints/pino_best.pt' (Data Loss: {best_data_loss:.6f}) ---")
    return model


if __name__ == "__main__":
    train_pino(num_samples=1000, epochs=200, base_lr=4e-4)
