# Physics-Informed Neural Operator (PINO) for 2D Navier-Stokes & Active Flow Control

An end-to-end **Physics-Informed Neural Operator (`PINO2D`)** framework for 2D incompressible Navier-Stokes vorticity simulation, alias-free zero-shot super-resolution, projection-only Test-Time Adaptation (TTA), and vectorized amplitude-bounded Particle Swarm Optimization (PSO) active flow control.

---

## Key Capabilities

- **Spectral Operator Backbone (`PINO2D`):** 4-layer 2D Fourier Neural Operator (`k_max = 16`, hidden width `64`, `8,409,601` parameters) with `LiftingBlock` (`3 -> 32 -> 64`) and `ProjectionBlock` (`64 -> 32 -> 1`).
- **Exact Frequency-Domain Physics Engine:** Analytical Fourier differentiation ($\widehat{\nabla u} = i\mathbf{k}\hat{u}$, $\widehat{\nabla^2 u} = -\|\mathbf{k}\|^2\hat{u}$) with $2/3$ Orszag dealiasing against a 4th-order Runge-Kutta (RK4) pseudo-spectral reference solver on $\Omega = [0, 2\pi)^2$.
- **Zero-Shot Super-Resolution:** Bandlimited Fourier zero-padding lifts coarse $64 \times 64$ inputs to $128 \times 128$ and $256 \times 256$ meshes ($0.07\%$ subsampled $256^2 \leftrightarrow 64^2$ consistency).
- **Projection-Only Test-Time Adaptation (TTA):** Freezes all global spectral layers and adapts only the `2,113`-parameter `ProjectionBlock` against a 2-step spectral PDE residual, baseline anchor, and circulation invariance constraint, with atomic weight restoration.
- **Vectorized Bounded PSO Control:** Evaluates entire particle swarms in a single batched `PINO2D` forward pass with hard actuator bounds ($\|f\|_\infty \le B$) at $>25,000\text{ evals/sec}$.
- **Interactive Web Studio:** Thread-safe HTTP backend (`server.py`) + 6-tab Swiss Light Mode frontend (`static/`) featuring live Vortex Painter, time rollout player, super-resolution, CFL scaling benchmark, TTA, and PSO flow control.

---

## Repository Structure

```text
├── checkpoints/
│   └── pino_best.pt                # Trained checkpoint (tracked via Git LFS)
├── pino/
│   ├── config.py                   # Domain, PDE, model, and loss dataclasses
│   ├── analytics.py                # Streamlines, invariants, spectra, custom vortices
│   ├── dataset/
│   │   ├── grf.py                  # Matérn-spectrum 2D Gaussian Random Field sampler
│   │   ├── reference_solver.py     # Orszag 2/3-dealiased RK4 pseudo-spectral solver
│   │   └── pde_dataset.py          # PyTorch Dataset & DataLoader pipeline
│   ├── models/
│   │   ├── fourier_layer.py        # 2D SpectralConv2d layer
│   │   ├── mlp.py                  # LiftingBlock & ProjectionBlock (1x1 Convs)
│   │   └── pino_net.py             # PINO2D end-to-end neural operator
│   ├── physics/
│   │   ├── spectral_diff.py        # Exact Fourier derivatives & 2/3 dealiasing
│   │   └── pde_loss.py             # Spectral Navier-Stokes PDE residual loss engine
│   └── optimization/
│       ├── tta.py                  # Projection-only TTA with atomic weight restore
│       └── pso_harness.py          # Vectorized amplitude-bounded PSO surrogate harness
├── static/                         # 6-tab Swiss Light Mode web studio (HTML/CSS/JS)
├── tests/                          # Unit tests (analytics, config, dataset, model, physics, opt)
├── server.py                       # Multi-threaded scientific HTTP API server (port 8000)
├── train.py                        # Training pipeline with MPS-safe complex gradient clipping
├── evaluate.py                     # 20-sample quantitative benchmark (baseline & TTA)
├── visualize.py                    # 5-panel spatiotemporal diagnostic plot generator
└── demo.py                         # CLI verification script
```

---

## Quickstart

### 1. Clone (with Git LFS for the Checkpoint)
```bash
git lfs install
git clone https://github.com/Darksoul913/PINO.git
cd PINO
git lfs pull
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Interactive Web Studio
```bash
python server.py 8000
```
Open `http://localhost:8000` in your browser.

### 4. Run Evaluation, Visualization & Tests
```bash
# Quantitative 20-sample evaluation (Baseline & 10-step TTA)
python evaluate.py

# Generate 5-panel spatiotemporal diagnostic plot (plots/pino_spatiotemporal_comparison.png)
python visualize.py

# Run unit test suite
pytest
```

### 5. Retrain from Scratch (Optional)
```bash
python train.py
```

---

## Verified Benchmark Snapshot (`checkpoints/pino_best.pt`, Apple Silicon MPS)

| Metric | PINO Baseline | PINO + TTA (10 steps) |
| :--- | :---: | :---: |
| **Mean Rel-$L^2$ Error (20 GRF samples)** | $7.33\%$ | **$7.29\%$** |
| **MAE / RMSE** | $0.0564 / 0.0729$ | **$0.0558 / 0.0727$** |
| **Circulation Drift ($\Delta\Gamma$)** | $0.20\%$ | **$0.16\%$** |
| **Kinetic Energy Drift ($\Delta E$)** | **$0.75\%$** | $1.69\%$ |
| **2-Step RMS Spectral PDE Residual** | $0.7343$ | **$0.6946$** |
| **CFL Speedup vs. RK4 ($64^2 \to 128^2 \to 256^2$)** | $20.6\times \to 20.9\times \to 16.8\times$ | $1.20\text{ ms} \to 4.05\text{ ms}$ latency |
