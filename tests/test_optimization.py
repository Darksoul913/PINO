"""Unit tests for Test-Time Adaptation and PSO Metaheuristic Surrogate Harness."""

import unittest
import torch
from pino.models.pino_net import PINO2D
from pino.physics.pde_loss import PINOLossEngine
from pino.optimization.tta import TestTimeAdapter
from pino.optimization.pso_harness import PSOSurrogateHarness, ParticleSwarmOptimizer


class TestOptimization(unittest.TestCase):
    def test_test_time_adapter(self):
        model = PINO2D(in_channels=3, out_channels=1, hidden_dim=32, modes1=8, modes2=8, num_layers=2)
        loss_engine = PINOLossEngine(s_x=32, s_y=32)

        x_input = torch.randn(1, 3, 32, 32)
        a_init = torch.randn(1, 1, 32, 32)

        adapter = TestTimeAdapter(model, loss_engine, steps=5, learning_rate=1e-4)
        adapted_pred, history = adapter.adapt_instance(x_input, a_init)

        self.assertEqual(adapted_pred.shape, (1, 1, 32, 32))
        self.assertLessEqual(history["final_loss"], history["initial_loss"] + 1e-4)

    def test_pso_surrogate_harness(self):
        model = PINO2D(in_channels=3, out_channels=1, hidden_dim=32, modes1=8, modes2=8, num_layers=2)
        harness = PSOSurrogateHarness(model)

        def objective_fn(pred_states: torch.Tensor) -> torch.Tensor:
            return torch.mean(pred_states**2, dim=(-2, -1)).squeeze(-1)

        pso = ParticleSwarmOptimizer(harness, num_particles=20, num_iterations=5, s_x=32, s_y=32)
        init_state = torch.randn(1, 1, 32, 32)

        results = pso.optimize(init_state, objective_fn)
        self.assertEqual(results["optimal_forcing"].shape, (1, 32, 32))
        self.assertGreater(results["eval_per_sec"], 50.0)


if __name__ == "__main__":
    unittest.main()
