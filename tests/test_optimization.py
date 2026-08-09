"""Unit tests for Test-Time Adaptation and PSO Metaheuristic Surrogate Harness."""

import torch
from pino.models.pino_net import PINO2D
from pino.physics.pde_loss import PINOLossEngine
from pino.optimization.tta import TestTimeAdapter
from pino.optimization.pso_harness import PSOSurrogateHarness, ParticleSwarmOptimizer


def test_test_time_adapter():
    model = PINO2D(in_channels=3, out_channels=1, hidden_dim=32, modes1=8, modes2=8, num_layers=2)
    loss_engine = PINOLossEngine(s_x=32, s_y=32)

    x_input = torch.randn(1, 3, 32, 32)
    a_init = torch.randn(1, 1, 32, 32)

    adapter = TestTimeAdapter(model, loss_engine, steps=10, learning_rate=1e-3)
    adapted_pred, history = adapter.adapt_instance(x_input, a_init)

    assert adapted_pred.shape == (1, 1, 32, 32)
    assert history["final_loss"] <= history["initial_loss"]
    print(f"Test-Time Adaptation test passed! (Initial Loss: {history['initial_loss']:.4f} -> Final: {history['final_loss']:.4f})")


def test_pso_surrogate_harness():
    model = PINO2D(in_channels=3, out_channels=1, hidden_dim=32, modes1=8, modes2=8, num_layers=2)
    harness = PSOSurrogateHarness(model)

    # Simple objective function: minimize total field variance / kinetic energy
    def objective_fn(pred_states: torch.Tensor) -> torch.Tensor:
        return torch.mean(pred_states**2, dim=(-2, -1)).squeeze(-1)

    pso = ParticleSwarmOptimizer(harness, num_particles=20, num_iterations=5, s_x=32, s_y=32)
    init_state = torch.randn(1, 1, 32, 32)

    results = pso.optimize(init_state, objective_fn)
    assert results["optimal_forcing"].shape == (1, 32, 32)
    assert results["eval_per_sec"] > 100.0

    print(f"PSO Surrogate Optimization test passed! ({results['total_evaluations']} evals in {results['total_latency_ms']:.1f}ms, {results['eval_per_sec']:.0f} evals/sec)")


if __name__ == "__main__":
    test_test_time_adapter()
    test_pso_surrogate_harness()
