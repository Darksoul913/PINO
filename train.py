"""
Two-Stage Physics-Informed Neural Operator (PINO) Training Pipeline.
Phase 1 (Epochs 1-80): Heavy data loss & IC loss to master spatial phase alignment & vortex geometry.
Phase 2 (Epochs 81-200): Physics residual ramp (lambda_pde 0.001 -> 0.1, lambda_ic 1.0 -> 10.0).
Includes Gradient Clipping (max_norm=1.0) and Cosine Annealing LR Scheduler (1e-3 -> 1e-6).
"""

import os
import torch
import torch.optim as optim
from tqdm import tqdm

from pino.config import PINOConfig, LossConfig
from pino.models.pino_net import PINO2D
from pino.physics.pde_loss import PINOLossEngine
from pino.dataset.pde_dataset import get_pde_dataloader


def train_pino(config: PINOConfig = None, num_samples: int = 400, epochs: int = 200):
    """
    Two-stage curriculum training loop for PINO with gradient clipping.
    """
    if config is None:
        config = PINOConfig()

    device = torch.device(config.device)
    print(f"--- Starting Two-Stage Curriculum PINO Training Pipeline on Device: {device} ({epochs} Epochs, {num_samples} Samples) ---")

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

    # 3. AdamW Optimizer & Cosine Annealing Scheduler (decay 1e-3 -> 1e-5)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 4. Training Loop with Two-Phase Loss Curriculum
    os.makedirs("checkpoints", exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_total_loss = 0.0
        running_ic_loss = 0.0
        running_data_loss = 0.0
        running_pde_loss = 0.0

        # Two-Phase Curriculum Loss Schedule
        if epoch <= 80:
            # Phase 1: Heavy data loss to learn spatial phase alignment & vortex geometry
            lambda_data = 1.0
            lambda_pde = 0.001
            lambda_ic = 1.0
        else:
            # Phase 2: Gradually ramp up physics & IC loss to refine high-frequency boundaries
            progress = (epoch - 80) / (epochs - 80)
            lambda_data = 1.0
            lambda_pde = 0.001 + progress * (0.1 - 0.001)  # Ramp to 0.1
            lambda_ic = 1.0 + progress * 9.0               # Ramp to 10.0

        current_loss_config = LossConfig(
            weight_data=lambda_data,
            weight_ic=lambda_ic,
            weight_pde=lambda_pde
        )

        for batch in dataloader:
            x_input = batch["x_input"].to(device)   # (batch, 3, s_x, s_y)
            a_init = batch["a_init"].to(device)     # (batch, 1, s_x, s_y)
            target = batch.get("target", None)
            if target is not None:
                target = target[:, -1:].to(device)   # Match final snapshot shape (batch, 1, s_x, s_y)

            optimizer.zero_grad()

            # Forward pass
            pred_u = model(x_input)  # (batch, 1, s_x, s_y)

            # Compute losses
            loss_dict = loss_engine(
                pred_u=pred_u,
                a_init=a_init,
                target_u=target,
                loss_config=current_loss_config
            )

            total_loss = loss_dict["loss_total"]
            total_loss.backward()

            # Gradient clipping to prevent spectral explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_total_loss += total_loss.item()
            running_ic_loss += loss_dict["loss_ic"].item()
            running_data_loss += loss_dict["loss_data"].item()
            running_pde_loss += loss_dict["loss_pde"].item()

        scheduler.step()

        num_batches = len(dataloader)
        avg_total = running_total_loss / num_batches
        avg_ic = running_ic_loss / num_batches
        avg_data = running_data_loss / num_batches
        avg_pde = running_pde_loss / num_batches

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch:03d}/{epochs:03d}] | Total: {avg_total:.4f} | IC (λ={lambda_ic:.1f}): {avg_ic:.4f} | Data: {avg_data:.4f} | PDE (λ={lambda_pde:.3f}): {avg_pde:.4f} | LR: {current_lr:.2e}")

        if avg_total < best_loss:
            best_loss = avg_total
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "loss": best_loss
            }, "checkpoints/pino_best.pt")

    print(f"\n--- Two-Stage Curriculum Training Completed! Best Checkpoint Saved to 'checkpoints/pino_best.pt' (Loss: {best_loss:.4f}) ---")
    return model


if __name__ == "__main__":
    train_pino(num_samples=1000, epochs=200)
