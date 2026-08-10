"""
Targeted Layer-Specific Test-Time Adaptation (TTA) Engine for PINO.
Freezes global Fourier convolution layers and optimizes ONLY the local MLP projection head
with spatial anchoring (alpha_anchor) and initial condition enforcement (beta_ic).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple

from pino.models.pino_net import PINO2D
from pino.physics.pde_loss import PINOLossEngine


class TestTimeAdapter:
    """
    Targeted Layer-Specific Test-Time Adapter.
    Updates only local MLP projection parameters (fc1, fc2) while keeping global Fourier weights frozen.
    L_TTA = L_pde + alpha_anchor * ||u_pred - u_base||^2 + beta_ic * ||u_pred(0) - a_input||^2
    """

    def __init__(
        self,
        model: PINO2D,
        loss_engine: PINOLossEngine,
        learning_rate: float = 1e-5,
        steps: int = 10,
        alpha_anchor: float = 1.0,
        beta_ic: float = 10.0
    ):
        self.model = model
        self.loss_engine = loss_engine
        self.lr = learning_rate
        self.steps = steps
        self.alpha_anchor = alpha_anchor
        self.beta_ic = beta_ic

    def adapt_instance(
        self,
        x_input: torch.Tensor,
        a_init: torch.Tensor,
        forcing: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Adapts model on a single input instance by fine-tuning ONLY projection head parameters.
        """
        self.model.eval()

        # 1. Clone baseline prediction u_base as fixed anchor
        with torch.no_grad():
            u_base = self.model(x_input).detach().clone()

        # 2. Freeze Fourier convolution layers; keep ONLY projection head trainable
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.projection.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr
        )

        initial_loss = 0.0
        final_loss = 0.0

        for step in range(self.steps):
            optimizer.zero_grad()

            # Forward pass
            u_pred = self.model(x_input)  # (1, 1, s_x, s_y)

            # 1. Physics Residual Loss
            # Form 2-step trajectory for PDE residual evaluation
            pred_u_2step = u_pred.repeat(1, 2, 1, 1)
            loss_pde = torch.mean(self.loss_engine.compute_pde_residual(pred_u_2step))

            # 2. Anchor Loss: Prevent drift from neural operator baseline
            loss_anchor = torch.mean((u_pred - u_base) ** 2)

            # 3. Initial Condition Loss: Enforce u(x,y,0) == a(x,y)
            loss_ic = self.loss_engine.compute_ic_loss(u_pred, a_init)

            # Total regularized TTA loss
            total_tta_loss = loss_pde + self.alpha_anchor * loss_anchor + self.beta_ic * loss_ic

            total_tta_loss.backward()
            optimizer.step()

            if step == 0:
                initial_loss = total_tta_loss.item()
            final_loss = total_tta_loss.item()

        # Final adapted prediction
        with torch.no_grad():
            adapted_pred = self.model(x_input)

        # 3. Restore requires_grad = True for all parameters
        for param in self.model.parameters():
            param.requires_grad = True

        history = {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "adaptation_gain": (initial_loss - final_loss) / (initial_loss + 1e-8)
        }

        return adapted_pred, history
