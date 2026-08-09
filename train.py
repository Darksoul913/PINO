"""
End-to-End Physics-Informed Neural Operator (PINO) Training Pipeline.
Trains PINO using combined Data Loss, Initial Condition Loss, and Exact Spectral PDE Residual Loss.
"""

import os
import torch
import torch.optim as optim
from tqdm import tqdm

from pino.config import PINOConfig
from pino.models.pino_net import PINO2D
from pino.physics.pde_loss import PINOLossEngine
from pino.dataset.pde_dataset import get_pde_dataloader


def train_pino(config: PINOConfig = None, num_samples: int = 50, epochs: int = 5):
    """
    Main training loop for PINO.
    """
    if config is None:
        config = PINOConfig()

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"--- Starting PINO Training Pipeline on Device: {device} ---")

    # 1. DataLoader
    dataloader = get_pde_dataloader(
        num_samples=num_samples,
        batch_size=config.batch_size,
        s_x=config.domain.s_x,
        s_y=config.domain.s_y,
        shuffle=True,
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

    # 3. Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 4. Training Loop
    os.makedirs("checkpoints", exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_total_loss = 0.0
        running_ic_loss = 0.0
        running_pde_loss = 0.0

        for batch in dataloader:
            x_input = batch["x_input"].to(device)   # (batch, 3, s_x, s_y)
            a_init = batch["a_init"].to(device)     # (batch, 1, s_x, s_y)
            target = batch.get("target", None)
            if target is not None:
                target = target.to(device)

            optimizer.zero_grad()

            # Forward pass
            pred_u = model(x_input)  # (batch, 1, s_x, s_y)

            # Compute losses
            loss_dict = loss_engine(
                pred_u=pred_u,
                a_init=a_init,
                target_u=target,
                loss_config=config.loss
            )

            total_loss = loss_dict["loss_total"]
            total_loss.backward()
            optimizer.step()

            running_total_loss += total_loss.item()
            running_ic_loss += loss_dict["loss_ic"].item()
            running_pde_loss += loss_dict["loss_pde"].item()

        scheduler.step()

        num_batches = len(dataloader)
        avg_total = running_total_loss / num_batches
        avg_ic = running_ic_loss / num_batches
        avg_pde = running_pde_loss / num_batches

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Total Loss: {avg_total:.4f} | IC Loss: {avg_ic:.4f} | PDE Loss: {avg_pde:.4f}")

        if avg_total < best_loss:
            best_loss = avg_total
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "loss": best_loss
            }, "checkpoints/pino_best.pt")

    print(f"--- Training Completed! Best Checkpoint Saved to 'checkpoints/pino_best.pt' (Loss: {best_loss:.4f}) ---")
    return model


if __name__ == "__main__":
    train_pino(epochs=3)
