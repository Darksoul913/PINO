"""
End-to-End Physics-Informed Neural Operator (PINO) Training Pipeline.
Uses Dynamic Physics Loss Weight Scheduling (Data-first -> Physics-ramp)
across extended epochs for low-error convergence (< 0.05).
"""

import os
import torch
import torch.optim as optim
from tqdm import tqdm

from pino.config import PINOConfig, LossConfig
from pino.models.pino_net import PINO2D
from pino.physics.pde_loss import PINOLossEngine
from pino.dataset.pde_dataset import get_pde_dataloader


def train_pino(config: PINOConfig = None, num_samples: int = 150, epochs: int = 100):
    """
    Main training loop for PINO with dynamic physics loss weighting.
    """
    if config is None:
        config = PINOConfig()

    device = torch.device(config.device)
    print(f"--- Starting PINO Training Pipeline on Device: {device} ({epochs} Epochs) ---")

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

    # 3. Optimizer & Cosine Annealing Scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 4. Training Loop with Dynamic Physics Loss Scheduling
    os.makedirs("checkpoints", exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_total_loss = 0.0
        running_ic_loss = 0.0
        running_data_loss = 0.0
        running_pde_loss = 0.0

        # Dynamic Loss Weight Schedule:
        # Epochs 1-30: lambda_pde = 0.01 (Establish global spatial features)
        # Epochs 31-100: Linearly ramp lambda_pde up to 1.0 (Enforce physical conservation)
        if epoch <= 30:
            lambda_pde = 0.01
        else:
            progress = (epoch - 30) / (epochs - 30)
            lambda_pde = 0.01 + progress * (1.0 - 0.01)

        current_loss_config = LossConfig(
            weight_data=1.0,
            weight_ic=10.0,
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

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:03d}/{epochs:03d}] | Total: {avg_total:.4f} | IC: {avg_ic:.4f} | Data: {avg_data:.4f} | PDE (λ={lambda_pde:.2f}): {avg_pde:.4f}")

        if avg_total < best_loss:
            best_loss = avg_total
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "loss": best_loss
            }, "checkpoints/pino_best.pt")

    print(f"\n--- Training Completed! Best Checkpoint Saved to 'checkpoints/pino_best.pt' (Loss: {best_loss:.4f}) ---")
    return model


if __name__ == "__main__":
    train_pino(num_samples=150, epochs=100)
