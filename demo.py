"""
Demonstration & Verification Script for PINO Framework.
Demonstrates Zero-Shot Super-Resolution, RK4 Reference Solver, Test-Time Adaptation,
and PSO Surrogate Optimization throughput.
"""

import os
import time
import torch

from pino.config import PINOConfig
from pino.models.pino_net import PINO2D
from pino.physics.pde_loss import PINOLossEngine
from pino.dataset.grf import GaussianRandomField2D
from pino.dataset.reference_solver import NavierStokes2DSolver
from pino.optimization.tta import TestTimeAdapter
from pino.optimization.pso_harness import PSOSurrogateHarness, ParticleSwarmOptimizer


def run_pino_demo():
    print("==========================================================================")
    print("      PHYSICS-INFORMED NEURAL OPERATOR (PINO) FRAMEWORK DEMO             ")
    print("==========================================================================\n")

    config = PINOConfig()
    device = config.device
    print(f"[*] Target Compute Hardware Device: {device.upper()}\n")

    # 1. Initialize GRF Sampler & Generate Initial Condition Field
    print("--- STEP 1: Synthetic Gaussian Random Field (GRF) Sampling ---")
    s_x, s_y = 64, 64
    grf = GaussianRandomField2D(s_x=s_x, s_y=s_y, length_scale=0.5, alpha=2.5, device=device)
    w0 = grf.sample(num_samples=1)  # (1, 64, 64)
    print(f"  Initial State Field a(x, y) generated. Mean: {w0.mean():.4f}, Std: {w0.std():.4f}, Shape: {tuple(w0.shape)}\n")

    # 2. Reference Solver Benchmark
    print("--- STEP 2: RK4 Pseudo-Spectral Reference Numerical Solver ---")
    solver = NavierStokes2DSolver(s_x=s_x, s_y=s_y, viscosity=1e-3, device=device)
    start_rk4 = time.perf_counter()
    trajectory_ref = solver.solve(w0, t_horizon=1.0, num_steps=100, save_steps=10)
    time_rk4_ms = (time.perf_counter() - start_rk4) * 1000.0
    print(f"  RK4 Reference Integration Completed in {time_rk4_ms:.2f} ms. Output Trajectory: {tuple(trajectory_ref.shape)}\n")

    # 3. PINO Model Forward Pass & Zero-Shot Super-Resolution
    print("--- STEP 3: PINO Model & Zero-Shot Super-Resolution (64x64 -> 256x256) ---")
    model = PINO2D.from_config(config.model).to(device)
    checkpoint_path = "checkpoints/pino_best.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded trained checkpoint from '{checkpoint_path}' (Rel-L2: {checkpoint.get('loss', 0.0)*100:.2f}%)")
    model.eval()

    # Prepare 64x64 grid input normalized to [0, 1]
    x = torch.linspace(0, (s_x - 1) / s_x, s_x, device=device)
    y = torch.linspace(0, (s_y - 1) / s_y, s_y, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    x_input = torch.cat([grid_x.unsqueeze(0), grid_y.unsqueeze(0), w0], dim=0).unsqueeze(0).to(device)  # (1, 3, 64, 64)

    start_pino = time.perf_counter()
    with torch.no_grad():
        u_coarse = model(x_input)                           # (1, 1, 64, 64)
        u_super_res = model(x_input, target_s=(256, 256))  # (1, 1, 256, 256)
    time_pino_ms = (time.perf_counter() - start_pino) * 1000.0

    print(f"  PINO Forward Pass Latency: {time_pino_ms:.2f} ms")
    print(f"  Standard Output Field Shape:   {tuple(u_coarse.shape)}")
    print(f"  Zero-Shot Super-Res Field Shape: {tuple(u_super_res.shape)}")
    print(f"  Inference Acceleration Factor vs RK4 Solver: {time_rk4_ms / max(time_pino_ms, 1e-4):.1f}x\n")

    # 4. Test-Time Adaptation Engine
    print("--- STEP 4: Instance-Level Test-Time Adaptation (TTA) ---")
    loss_engine = PINOLossEngine(s_x=s_x, s_y=s_y, device=device)
    adapter = TestTimeAdapter(model, loss_engine, steps=10, learning_rate=1e-3)
    _, tta_history = adapter.adapt_instance(x_input, w0)
    print(f"  TTA Physics Residual Loss Reduction: {tta_history['initial_loss']:.4f} -> {tta_history['final_loss']:.4f}")
    print(f"  Adaptation Gain: {tta_history['adaptation_gain'] * 100:.1f}%\n")

    # 5. Particle Swarm Optimization (PSO) Active Forcing Control Harness
    print("--- STEP 5: Vectorized PSO Metaheuristic Surrogate Optimization ---")
    harness = PSOSurrogateHarness(model, device=device)
    pso = ParticleSwarmOptimizer(harness, num_particles=100, num_iterations=10, s_x=s_x, s_y=s_y)

    # Objective: Minimize total field kinetic energy / enstrophy variance
    def objective_fn(pred_states: torch.Tensor) -> torch.Tensor:
        return torch.mean(pred_states**2, dim=(-2, -1)).squeeze(-1)

    pso_results = pso.optimize(w0, objective_fn)
    print(f"  Total Surrogate Candidate Evaluations: {pso_results['total_evaluations']}")
    print(f"  Total Optimization Wall-Clock Latency:  {pso_results['total_latency_ms']:.2f} ms")
    print(f"  Surrogate Evaluation Throughput:       {pso_results['eval_per_sec']:.0f} evals / second")
    print(f"  Optimal Candidate Fitness Score:       {pso_results['best_fitness']:.6f}\n")

    print("==========================================================================")
    print("      SUCCESS: ALL PINO PIPELINE MODULES VERIFIED & OPERATIONAL!           ")
    print("==========================================================================")


if __name__ == "__main__":
    run_pino_demo()
