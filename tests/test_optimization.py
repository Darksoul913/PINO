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

    def test_tta_curves_and_restore_weights(self):
        torch.manual_seed(0)
        model = PINO2D(in_channels=3, out_channels=1, hidden_dim=32, modes1=8, modes2=8, num_layers=2)
        loss_engine = PINOLossEngine(s_x=32, s_y=32)

        x_input = torch.randn(1, 3, 32, 32)
        a_init = torch.randn(1, 32, 32)  # 3D input to verify normalization to (B, 1, s_x, s_y)
        forcing = torch.randn(1, 32, 32)

        orig_proj_params = [p.detach().clone() for p in model.projection.parameters()]

        steps = 6
        adapter = TestTimeAdapter(
            model, loss_engine, steps=steps, learning_rate=1e-4, t_horizon=1.0, beta_circ=1.0
        )
        adapted_pred, history = adapter.adapt_instance(
            x_input, a_init, forcing=forcing, restore_weights=True
        )

        self.assertEqual(adapted_pred.shape, (1, 1, 32, 32))
        self.assertEqual(len(history["loss_curve"]), steps)
        self.assertEqual(len(history["pde_curve"]), steps)
        self.assertEqual(len(history["circ_curve"]), steps)
        self.assertTrue(all(val >= 0.0 for val in history["pde_curve"]))
        self.assertGreaterEqual(history["elapsed_ms"], 0.0)
        self.assertGreaterEqual(history["initial_pde_loss"], 0.0)
        self.assertGreaterEqual(history["final_pde_loss"], 0.0)
        self.assertLessEqual(history["final_loss"], history["initial_loss"] + 1e-4)

        for p_before, p_after in zip(orig_proj_params, model.projection.parameters()):
            self.assertTrue(torch.equal(p_before, p_after))

        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_pso_history_and_forcing_bound(self):
        torch.manual_seed(0)
        model = PINO2D(in_channels=3, out_channels=1, hidden_dim=32, modes1=8, modes2=8, num_layers=2)
        harness = PSOSurrogateHarness(model)

        def objective_fn(pred_states: torch.Tensor) -> torch.Tensor:
            return torch.mean(pred_states**2, dim=(-2, -1)).squeeze(-1)

        num_iterations = 7
        bound = 0.25
        pso = ParticleSwarmOptimizer(
            harness,
            num_particles=20,
            num_iterations=num_iterations,
            s_x=32,
            s_y=32,
            forcing_bound=bound,
        )
        init_state = torch.randn(1, 1, 32, 32)
        results = pso.optimize(init_state, objective_fn)

        fitness_history = results["fitness_history"]
        mean_fitness_history = results["mean_fitness_history"]

        self.assertEqual(len(fitness_history), num_iterations + 1)
        self.assertEqual(len(mean_fitness_history), num_iterations + 1)
        for i in range(len(fitness_history) - 1):
            self.assertLessEqual(fitness_history[i + 1], fitness_history[i])

        max_forcing_abs = torch.max(torch.abs(results["optimal_forcing"])).item()
        self.assertLessEqual(max_forcing_abs, bound + 1e-7)


if __name__ == "__main__":
    unittest.main()
