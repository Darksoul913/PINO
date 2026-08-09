"""
Vectorized Metaheuristic Surrogate Harness & Particle Swarm Optimization (PSO) Interface.
Provides sub-15ms forward-pass batch queries for optimal active boundary and forcing control.
"""

import torch
import torch.nn as nn
import time
from typing import Callable, Tuple, Dict

from pino.models.pino_net import PINO2D


class PSOSurrogateHarness:
    """
    Vectorized Surrogate Harness wrapping PINO network for ultra-fast candidate evaluation
    in metaheuristic optimization algorithms (PSO, Evolutionary Strategies, NSGA-II).
    """

    def __init__(self, model: PINO2D, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def evaluate_candidates(
        self,
        candidate_forcings: torch.Tensor,
        initial_state: torch.Tensor,
        objective_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, float]:
        """
        Evaluates a large batch of candidate forcing profiles in a single vectorized pass.
        Args:
            candidate_forcings: Batch of candidate forcing fields (batch_size, 1, s_x, s_y)
            initial_state: Base initial state (1, 1, s_x, s_y) repeated to batch_size
            objective_fn: Function mapping predicted state field (batch, 1, s_x, s_y) -> fitness (batch,)
        Returns:
            Tuple of (fitness_scores, execution_latency_ms)
        """
        batch_size = candidate_forcings.shape[0]
        s_x, s_y = candidate_forcings.shape[2], candidate_forcings.shape[3]

        # Prepare grid coordinates (2, s_x, s_y)
        x = torch.linspace(0, 2 * 3.141592653589793 * (s_x - 1) / s_x, s_x, device=self.device)
        y = torch.linspace(0, 2 * 3.141592653589793 * (s_y - 1) / s_y, s_y, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # Concatenate grid + candidate forcing fields -> (batch_size, 3, s_x, s_y)
        x_inputs = torch.cat([grid, candidate_forcings], dim=1)

        # Benchmark latency
        start_time = time.perf_counter()
        pred_states = self.model(x_inputs)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Evaluate objective fitness across all candidates simultaneously
        fitness_scores = objective_fn(pred_states)

        return fitness_scores, latency_ms


class ParticleSwarmOptimizer:
    """
    Vectorized Particle Swarm Optimizer operating on surrogate PINO evaluations.
    """

    def __init__(
        self,
        harness: PSOSurrogateHarness,
        num_particles: int = 100,
        num_iterations: int = 20,
        s_x: int = 64,
        s_y: int = 64,
        c1: float = 2.0,
        c2: float = 2.0,
        w: float = 0.7
    ):
        self.harness = harness
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.s_x = s_x
        self.s_y = s_y
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.device = harness.device

    def optimize(
        self,
        initial_state: torch.Tensor,
        objective_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Runs PSO optimization to find optimal forcing profile f*(x).
        """
        # Initialize swarm positions (forcings) and velocities
        positions = torch.randn(self.num_particles, 1, self.s_x, self.s_y, device=self.device) * 0.1
        velocities = torch.randn_like(positions) * 0.01

        # Personal best positions and scores
        pbest_positions = positions.clone()
        pbest_scores, _ = self.harness.evaluate_candidates(positions, initial_state, objective_fn)

        # Global best position and score
        gbest_idx = torch.argmin(pbest_scores)
        gbest_position = pbest_positions[gbest_idx].clone()
        gbest_score = pbest_scores[gbest_idx].item()

        total_eval_time = 0.0

        for iter_idx in range(self.num_iterations):
            r1 = torch.rand_like(positions)
            r2 = torch.rand_like(positions)

            # Update velocities
            velocities = (
                self.w * velocities +
                self.c1 * r1 * (pbest_positions - positions) +
                self.c2 * r2 * (gbest_position.unsqueeze(0) - positions)
            )

            # Update positions
            positions = positions + velocities

            # Evaluate swarm
            scores, latency = self.harness.evaluate_candidates(positions, initial_state, objective_fn)
            total_eval_time += latency

            # Update personal bests
            improved_mask = scores < pbest_scores
            pbest_positions[improved_mask] = positions[improved_mask].clone()
            pbest_scores[improved_mask] = scores[improved_mask]

            # Update global best
            current_best_idx = torch.argmin(pbest_scores)
            if pbest_scores[current_best_idx] < gbest_score:
                gbest_score = pbest_scores[current_best_idx].item()
                gbest_position = pbest_positions[current_best_idx].clone()

        return {
            "optimal_forcing": gbest_position,
            "best_fitness": gbest_score,
            "total_evaluations": self.num_particles * self.num_iterations,
            "total_latency_ms": total_eval_time,
            "eval_per_sec": (self.num_particles * self.num_iterations) / (total_eval_time / 1000.0 + 1e-8)
        }
