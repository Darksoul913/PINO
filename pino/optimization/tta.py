"""
Targeted Layer-Specific Test-Time Adaptation (TTA) Engine for PINO.
Freezes global Fourier convolution layers and optimizes ONLY the local MLP projection head
with spatial anchoring (alpha_anchor), spectral PDE residual minimization, and circulation conservation.
"""

import copy
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.optim as optim

from pino.models.pino_net import PINO2D
from pino.physics.pde_loss import PINOLossEngine


def _sync_device(device: torch.device) -> None:
    """Synchronize accelerator device before reading wall-clock timers."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


class TestTimeAdapter:
    """
    Targeted Layer-Specific Test-Time Adapter.
    Updates only local MLP projection parameters (fc1, fc2) while keeping global Fourier weights frozen.
    L_TTA = L_pde_mse + alpha_anchor * ||u_pred - u_base||^2 + beta_circ * (mean(u_pred) - mean(a_init))^2
    """

    def __init__(
        self,
        model: PINO2D,
        loss_engine: PINOLossEngine,
        learning_rate: float = 1e-4,
        steps: int = 10,
        alpha_anchor: float = 1.0,
        beta_ic: float = 10.0,
        t_horizon: float = 1.0,
        beta_circ: float = 1.0,
    ):
        self.model = model
        self.loss_engine = loss_engine
        self.lr = learning_rate
        self.steps = steps
        self.alpha_anchor = alpha_anchor
        self.beta_ic = beta_ic
        self.t_horizon = t_horizon
        self.beta_circ = beta_circ

    def _compute_losses(
        self,
        u_pred: torch.Tensor,
        a_init: torch.Tensor,
        u_base: torch.Tensor,
        forcing: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate total TTA loss, non-negative PDE residual MSE, and circulation drift loss."""
        traj = torch.cat([a_init, u_pred], dim=1)
        res = self.loss_engine.compute_pde_residual(traj, dt=self.t_horizon, forcing=forcing)
        loss_pde = torch.mean(res ** 2)
        loss_anchor = torch.mean((u_pred - u_base) ** 2)
        loss_circ = torch.mean(
            (torch.mean(u_pred, dim=(-2, -1)) - torch.mean(a_init, dim=(-2, -1))) ** 2
        )
        total_loss = loss_pde + self.alpha_anchor * loss_anchor + self.beta_circ * loss_circ
        return total_loss, loss_pde, loss_circ

    def adapt_instance(
        self,
        x_input: torch.Tensor,
        a_init: torch.Tensor,
        forcing: Optional[torch.Tensor] = None,
        restore_weights: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Adapts model on a single input instance by fine-tuning ONLY projection head parameters.
        """
        device = x_input.device
        a_init = a_init.to(device)
        if a_init.ndim == 2:
            a_init = a_init.unsqueeze(0).unsqueeze(0)
        elif a_init.ndim == 3:
            a_init = a_init.unsqueeze(1)

        if forcing is not None:
            forcing = forcing.to(device)
            if forcing.ndim == 2:
                forcing = forcing.unsqueeze(0)
            elif forcing.ndim == 4 and forcing.shape[1] == 1:
                forcing = forcing.squeeze(1)

        if self.loss_engine.diff.Kx.device != device:
            self.loss_engine = self.loss_engine.to(device)

        was_training = self.model.training
        self.model.eval()
        orig_proj_state = copy.deepcopy(self.model.projection.state_dict())

        loss_curve = []
        pde_curve = []
        circ_curve = []

        _sync_device(device)
        start_time = time.perf_counter()

        try:
            # 1. Clone baseline prediction u_base as fixed anchor
            with torch.no_grad():
                u_base = self.model(x_input).detach().clone()

            # 2. Freeze Fourier convolution layers; keep ONLY projection head trainable
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.projection.parameters():
                param.requires_grad = True

            trainable_params = [p for p in self.model.projection.parameters() if p.requires_grad]
            optimizer = optim.Adam(trainable_params, lr=self.lr)

            for _ in range(self.steps):
                optimizer.zero_grad()

                u_pred = self.model(x_input)
                total_tta_loss, loss_pde, loss_circ = self._compute_losses(
                    u_pred, a_init, u_base, forcing=forcing
                )

                total_tta_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

                loss_curve.append(float(total_tta_loss.item()))
                pde_curve.append(float(loss_pde.item()))
                circ_curve.append(float(loss_circ.item()))

            # Final adapted prediction and post-update loss evaluation
            with torch.no_grad():
                adapted_pred = self.model(x_input).detach().clone()
                post_loss, post_pde, _ = self._compute_losses(
                    adapted_pred, a_init, u_base, forcing=forcing
                )

            _sync_device(device)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        finally:
            if restore_weights:
                self.model.projection.load_state_dict(orig_proj_state)
            for param in self.model.parameters():
                param.requires_grad = True
            self.model.train(was_training)

        initial_loss = loss_curve[0] if loss_curve else float(post_loss.item())
        initial_pde_loss = pde_curve[0] if pde_curve else float(post_pde.item())
        final_loss = float(post_loss.item())
        final_pde_loss = float(post_pde.item())

        history = {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "initial_pde_loss": initial_pde_loss,
            "final_pde_loss": final_pde_loss,
            "adaptation_gain": (initial_loss - final_loss) / (abs(initial_loss) + 1e-8),
            "loss_curve": loss_curve,
            "pde_curve": pde_curve,
            "circ_curve": circ_curve,
            "elapsed_ms": elapsed_ms,
        }

        return adapted_pred, history
