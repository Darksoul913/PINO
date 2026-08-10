"""
Pure Data-Driven Fourier Neural Operator (FNO) Training Pipeline.
Standard approach from Li et al. (2021): learns the operator mapping a(x,y) → u(x,y,T)
using ONLY supervised data loss (relative L2 norm) against RK4 reference solutions.

No PDE residual loss during training — physics regularization is applied only
during Test-Time Adaptation (TTA) in evaluate.py.
"""

import os
import torch
import torch.optim as optim

from pino.config import PINOConfig
from pino.models.pino_net import PINO2D
from pino.physics.pde_loss import PINOLossEngine
from pino.dataset.pde_dataset import get_pde_dataloader


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

    total_norm = total_sq_norm ** 0.5
    clip_coef = max_norm / (total_norm + 1e-6)

    if clip_coef < 1.0:
        for p in parameters:
            p.grad.data.mul_(clip_coef)

    return total_norm


def train_pino(config: PINOConfig = None, num_samples: int = 1000, epochs: int = 200):
    """
    Pure data-driven FNO training loop.
    Loss = relative_l2(model(a), u_ref(T))
    """
    if config is None:
        config = PINOConfig()

    device = torch.device(config.device)
    lr = 1e-3
    print(f"--- Pure Data-Driven FNO Training on Device: {device} ({epochs} Epochs, {num_samples} Samples, lr={lr}) ---")

    # 1. DataLoader with ground-truth reference trajectories
    dataloader = get_pde_dataloader(
        num_samples=num_samples,
        batch_size=config.batch_size,
        s_x=config.domain.s_x,
        s_y=config.domain.s_y,
        shuffle=True,
        generate_reference_data=True,
        device=str(device)
    )

    # 2. Model
    model = PINO2D.from_config(config.model).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Model Parameters: {total_params:,}")

    # 3. Relative L2 loss function (same as loss_engine but standalone)
    def relative_l2_loss(pred, target):
        """||pred - target||_2 / ||target||_2, averaged over batch."""
        diff_norms = torch.norm(pred - target, p=2, dim=(-2, -1))
        target_norms = torch.norm(target, p=2, dim=(-2, -1)) + 1e-8
        return torch.mean(diff_norms / target_norms)

    # 4. AdamW Optimizer + Cosine Annealing Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    os.makedirs("checkpoints", exist_ok=True)
    best_data_loss = float("inf")

    # 5. Training Loop — Pure Data Loss Only
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            x_input = batch["x_input"].to(device)   # (batch, 3, s_x, s_y)
            target = batch.get("target", None)
            if target is None:
                continue
            # Take final time snapshot t=T as supervision target
            target = target[:, -1:].to(device)       # (batch, 1, s_x, s_y)

            optimizer.zero_grad()

            # Forward: a(x,y) → u_pred(x,y,T)
            pred_u = model(x_input)                  # (batch, 1, s_x, s_y)

            # Pure supervised relative L2 loss
            loss = relative_l2_loss(pred_u, target)

            loss.backward()
            clip_grad_norm_safe(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        scheduler.step()

        avg_loss = running_loss / max(num_batches, 1)

        # Save checkpoint on data loss improvement
        if avg_loss < best_data_loss and epoch > 3:
            best_data_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "loss": best_data_loss
            }, "checkpoints/pino_best.pt")

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            current_lr = scheduler.get_last_lr()[0]
            pct = avg_loss * 100
            print(f"Epoch [{epoch:03d}/{epochs:03d}] | Rel-L2 Loss: {avg_loss:.6f} ({pct:.2f}%) | LR: {current_lr:.2e}")

    print(f"\n--- Training Complete! Best: 'checkpoints/pino_best.pt' (Rel-L2: {best_data_loss:.6f} = {best_data_loss*100:.2f}%) ---")
    return model


if __name__ == "__main__":
    train_pino(num_samples=1000, epochs=200)
