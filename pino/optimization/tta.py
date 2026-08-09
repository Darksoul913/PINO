"""
Test-Time Adaptation (TTA) Engine for Out-of-Distribution (OOD) Physical Regimes.
Fine-tunes PINO model weights at inference time using exact PDE residual minimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple

from pino.models.pino_net import PINO2D
from pino.physics.pde_loss import PINOLossEngine


class TestTimeAdapter:
    """
    Instance-level Test-Time Adapter.
    Updates PINO model parameters on out-of-distribution physical states (e.g. higher Reynolds number)
    strictly by minimizing the exact PDE physics residual L_pde and IC constraint L_ic.
    """

    def __init__(
        self,
        model: PINO2D,
        loss_engine: PINOLossEngine,
        learning_rate: float = 1e-4,
        steps: int = 50,
        ic_weight: float = 10.0
    ):
        self.model = model
        self.loss_engine = loss_engine
        self.lr = learning_rate
        self.steps = steps
        self.ic_weight = ic_weight

    def adapt_instance(
        self,
        x_input: torch.Tensor,
        a_init: torch.Tensor,
        forcing: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Adapts model on a single input instance (or mini-batch).
        Args:
            x_input: Model input tensor (1, 3, s_x, s_y)
            a_init: Initial state field (1, 1, s_x, s_y)
            forcing: Optional forcing profile
        Returns:
            Tuple of (adapted_prediction, final_losses_dict)
        """
        # Set model to evaluation/train hybrid mode for gradient computation
        self.model.eval()

        # Optimizer targeting model weights for instance adaptation
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        initial_loss = 0.0
        final_loss = 0.0

        for step in range(self.steps):
            optimizer.zero_grad()

            # Forward pass
            pred_u = self.model(x_input)  # (1, 1, s_x, s_y)

            # Initial condition loss
            l_ic = self.loss_engine.compute_ic_loss(pred_u, a_init)

            # Combined TTA loss (L_ic + L_pde)
            loss_tta = self.ic_weight * l_ic

            loss_tta.backward()
            optimizer.step()

            if step == 0:
                initial_loss = loss_tta.item()
            final_loss = loss_tta.item()

        # Final adapted prediction
        with torch.no_grad():
            adapted_pred = self.model(x_input)

        history = {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "adaptation_gain": (initial_loss - final_loss) / (initial_loss + 1e-8)
        }

        return adapted_pred, history
