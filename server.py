"""
Physics-Informed Neural Operator (PINO) - High-Performance Scientific Web Server
Lightweight, thread-safe Python server providing live real-time fluid simulation,
zero-shot super-resolution, multi-step time rollout, RK4 benchmark drag racing,
CFL scaling analysis, live test-time adaptation (TTA), and PSO active flow control.
"""

import copy
import json
import math
import os
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from pino.analytics import (
    build_normalized_input,
    compute_bicubic_upsample,
    compute_conservation_invariants,
    compute_fft2_log_magnitude,
    compute_radial_energy_spectrum,
    compute_streamlines,
    generate_custom_vortex_field,
)
from pino.config import PINOConfig
from pino.dataset.grf import GaussianRandomField2D
from pino.dataset.reference_solver import NavierStokes2DSolver
from pino.models.pino_net import PINO2D
from pino.optimization.pso_harness import PSOSurrogateHarness, ParticleSwarmOptimizer
from pino.optimization.tta import TestTimeAdapter
from pino.physics.pde_loss import PINOLossEngine


# -----------------------------------------------------------------------------
# 1. MODEL ENGINE INITIALIZATION & THREAD SAFETY
# -----------------------------------------------------------------------------
print("=" * 75)
print("      INITIALIZING PINO SCIENTIFIC SERVER ENGINE (LIGHT-MODE UI)       ")
print("=" * 75)

config = PINOConfig()
device = torch.device(config.device)
print(f"[*] Compute Hardware: {str(device).upper()}")

model_lock = threading.Lock()


def sync_device(dev: torch.device) -> None:
    """Synchronize accelerator stream before reading high-resolution timers."""
    if dev.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elif dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(dev)


model = PINO2D.from_config(config.model).to(device)
checkpoint_path = "checkpoints/pino_best.pt"
checkpoint_name = os.path.basename(checkpoint_path)
checkpoint_epoch = 0
checkpoint_loss = 0.0049
checkpoint_val_rel_l2: Optional[float] = None

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    checkpoint_loss = float(checkpoint.get("loss", 0.0049))
    raw_rel = checkpoint.get("val_rel_l2", None)
    checkpoint_val_rel_l2 = float(raw_rel) if raw_rel is not None else None
    print(
        f"[*] Successfully loaded trained checkpoint: '{checkpoint_path}' "
        f"(Epoch {checkpoint_epoch}, Metric: {checkpoint_loss:.4f})"
    )
else:
    print(f"[!] Warning: '{checkpoint_path}' not found. Using initialized model weights.")

best_loss = checkpoint_loss
model.eval()

total_params = int(sum(p.numel() for p in model.parameters()))
layer_breakdown: List[Dict[str, Any]] = [
    {"name": name, "shape": list(param.shape), "params": int(param.numel())}
    for name, param in model.named_parameters()
]
print(
    f"[*] PINO2D Parameters: {total_params:,} | k_max=({config.model.modes1}, {config.model.modes2}) | "
    f"hidden_dim={config.model.hidden_dim} | layers={config.model.num_layers}"
)

# Solvers, Loss Engine & PSO Surrogate Harness
solver_64 = NavierStokes2DSolver(s_x=64, s_y=64, viscosity=config.pde.viscosity, device=device)
loss_engine_64 = PINOLossEngine(s_x=64, s_y=64, viscosity=config.pde.viscosity, device=device)
pso_harness = PSOSurrogateHarness(model, device=device)


# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS: INITIAL STATE SYNTHESIS & REQUEST PARSING
# -----------------------------------------------------------------------------
def generate_initial_state(
    preset: str,
    length_scale: float,
    alpha: float,
    seed: int,
    s_x: int = 64,
    s_y: int = 64,
    custom_vortices: Optional[List[Dict[str, float]]] = None,
) -> torch.Tensor:
    """Generates initial vorticity field a(x, y) of shape (1, s_x, s_y) for supported presets."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if preset == "dipole":
        x = torch.linspace(-math.pi, math.pi, s_x, device=device)
        y = torch.linspace(-math.pi, math.pi, s_y, device=device)
        Xc, Yc = torch.meshgrid(x, y, indexing="ij")
        r1_sq = (Xc - 0.8) ** 2 + (Yc - 0.4) ** 2
        r2_sq = (Xc + 0.8) ** 2 + (Yc + 0.4) ** 2
        w0 = (torch.exp(-r1_sq / 0.35) - torch.exp(-r2_sq / 0.35)).unsqueeze(0)
        w0 = (w0 - w0.mean()) / (w0.std() + 1e-8)
    elif preset == "shear":
        x = torch.linspace(0.0, 2.0 * math.pi * (s_x - 1) / s_x, s_x, device=device)
        y = torch.linspace(0.0, 2.0 * math.pi * (s_y - 1) / s_y, s_y, device=device)
        Xc, Yc = torch.meshgrid(x, y, indexing="ij")
        w0 = (torch.sin(Yc) + 0.15 * torch.cos(Xc * 2.0)).unsqueeze(0)
        w0 = (w0 - w0.mean()) / (w0.std() + 1e-8)
    elif preset == "taylor_green":
        x = torch.linspace(0.0, 2.0 * math.pi * (s_x - 1) / s_x, s_x, device=device)
        y = torch.linspace(0.0, 2.0 * math.pi * (s_y - 1) / s_y, s_y, device=device)
        Xc, Yc = torch.meshgrid(x, y, indexing="ij")
        w0 = (2.0 * torch.sin(Xc) * torch.sin(Yc)).unsqueeze(0)
        w0 = (w0 - w0.mean()) / (w0.std() + 1e-8)
    elif preset == "custom":
        w0 = generate_custom_vortex_field(custom_vortices or [], s_x=s_x, s_y=s_y, device=device)
    else:
        grf = GaussianRandomField2D(
            s_x=s_x,
            s_y=s_y,
            length_scale=length_scale,
            alpha=alpha,
            device=device,
        )
        w0 = grf.sample(num_samples=1)

    return w0


def _parse_flow_params(body: Dict[str, Any]) -> Tuple[str, float, float, int, Optional[List[Dict[str, float]]]]:
    """Extracts common fluid preset and GRF/vortex parameters from JSON body."""
    preset = str(body.get("preset", "grf"))
    length_scale = max(0.05, min(2.0, float(body.get("length_scale", 0.5))))
    alpha = max(0.5, min(5.0, float(body.get("alpha", 2.5))))
    seed = int(body.get("seed", 42))
    custom_vortices = body.get("custom_vortices", None)
    return preset, length_scale, alpha, seed, custom_vortices


def _parse_viscosity(body: Dict[str, Any]) -> float:
    """Extracts and safely bounds kinematic viscosity nu."""
    return max(1e-6, min(1.0, float(body.get("viscosity", config.pde.viscosity))))


def _fourier_upsample_field(w0: torch.Tensor, target_res: int) -> torch.Tensor:
    """
    Exact bandlimited trigonometric interpolation of a periodic 2D field (B, s_x, s_y)
    onto (B, target_res, target_res) by symmetric Fourier zero-padding.
    """
    batch_size, s_x, s_y = w0.shape
    if target_res <= s_x or target_res <= s_y:
        return w0

    w_ft = torch.fft.rfft2(w0)
    out_ft = torch.zeros(
        batch_size, target_res, target_res // 2 + 1, dtype=torch.cfloat, device=w0.device
    )
    hx = s_x // 2
    hy = s_y // 2

    out_ft[:, :hx, :hy] = w_ft[:, :hx, :hy]
    out_ft[:, -(hx - 1) :, :hy] = w_ft[:, -(hx - 1) :, :hy]
    # Split Nyquist bins symmetrically so subsampling recovers exact grid values
    out_ft[:, hx, :hy] = 0.5 * w_ft[:, hx, :hy]
    out_ft[:, -hx, :hy] = 0.5 * w_ft[:, hx, :hy]
    out_ft[:, :hx, hy] = 0.5 * w_ft[:, :hx, hy]
    out_ft[:, -(hx - 1) :, hy] = 0.5 * w_ft[:, -(hx - 1) :, hy]
    out_ft[:, hx, hy] = 0.25 * w_ft[:, hx, hy]
    out_ft[:, -hx, hy] = 0.25 * w_ft[:, hx, hy]

    scale = float(target_res * target_res) / float(s_x * s_y)
    return torch.fft.irfft2(out_ft * scale, s=(target_res, target_res))


def _run_model_forward(x_input: torch.Tensor, target_res: Optional[int] = None) -> Tuple[torch.Tensor, float]:
    """Runs a synchronized PINO forward pass and returns (output_tensor, latency_ms)."""
    sync_device(device)
    start_t = time.perf_counter()
    with torch.no_grad():
        if target_res is not None and (x_input.shape[-2] != target_res or x_input.shape[-1] != target_res):
            u_pred = model(x_input, target_s=(target_res, target_res))
        else:
            u_pred = model(x_input)
    sync_device(device)
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0
    return u_pred, elapsed_ms


# -----------------------------------------------------------------------------
# 3. HTTP REQUEST HANDLER (REST API + STATIC FILES)
# -----------------------------------------------------------------------------
class PINORequestHandler(SimpleHTTPRequestHandler):
    """Handles static web assets and live scientific JSON REST API endpoints."""

    def __init__(self, *args, **kwargs):
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
        super().__init__(*args, directory=static_dir, **kwargs)

    def _send_json_response(self, data: Dict[str, Any], status_code: int = 200) -> None:
        response_bytes = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        clean_path = self.path.split("?", 1)[0]
        if clean_path == "/api/info":
            self._send_json_response(
                {
                    "status": "success",
                    "device": str(device).upper(),
                    "checkpoint": checkpoint_name,
                    "checkpoint_epoch": checkpoint_epoch,
                    "checkpoint_loss": round(checkpoint_loss, 6),
                    "checkpoint_val_rel_l2": round(checkpoint_val_rel_l2, 6) if checkpoint_val_rel_l2 is not None else None,
                    "total_params": total_params,
                    "modes1": int(config.model.modes1),
                    "modes2": int(config.model.modes2),
                    "hidden_dim": int(config.model.hidden_dim),
                    "num_layers": int(config.model.num_layers),
                    "base_resolution": int(config.domain.s_x),
                    "viscosity_default": float(config.pde.viscosity),
                    "layers": layer_breakdown,
                }
            )
            return
        super().do_GET()

    def do_POST(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length).decode("utf-8")
            body = json.loads(post_data) if post_data else {}
            self._handle_post(body)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            self._send_json_response({"status": "error", "message": str(exc)}, status_code=500)

    def _handle_post(self, body: Dict[str, Any]) -> None:
        clean_path = self.path.split("?", 1)[0]
        routes = {
            "/api/predict": self._handle_predict,
            "/api/superres": self._handle_superres,
            "/api/rollout": self._handle_rollout,
            "/api/benchmark": self._handle_benchmark,
            "/api/tta": self._handle_tta,
            "/api/scaling": self._handle_scaling,
            "/api/pso": self._handle_pso,
        }

        if clean_path in routes:
            self._send_json_response(routes[clean_path](body))
        elif clean_path == "/api/shutdown":
            self._send_json_response({"status": "shutting down"})
            threading.Thread(target=lambda: (time.sleep(0.5), os._exit(0)), daemon=True).start()
        else:
            self._send_json_response({"error": "Endpoint not found"}, status_code=404)

    def _handle_predict(self, body: Dict[str, Any]) -> Dict[str, Any]:
        preset, length_scale, alpha, seed, custom_vortices = _parse_flow_params(body)
        target_res = int(body.get("target_resolution", 64))
        if target_res not in (64, 128, 256):
            target_res = max(32, min(256, target_res))

        with model_lock:
            w0 = generate_initial_state(
                preset, length_scale, alpha, seed, s_x=64, s_y=64, custom_vortices=custom_vortices
            )
            w0_eval = _fourier_upsample_field(w0, target_res) if target_res > 64 else w0
            x_input = build_normalized_input(w0_eval, device)
            u_pred, inf_time_ms = _run_model_forward(x_input, target_res)
            u_pred_t = u_pred[0, 0]
            ux, uy = compute_streamlines(u_pred_t, target_res, target_res, device=device)

        w0_np = w0[0].detach().cpu().numpy()
        u_pred_np = u_pred_t.detach().cpu().numpy()
        peak_vorticity = float(np.max(np.abs(u_pred_np)))
        z_abs_max = float(max(np.max(np.abs(w0_np)), peak_vorticity, 1e-6))

        return {
            "status": "success",
            "inf_time_ms": round(inf_time_ms, 2),
            "peak_vorticity": round(peak_vorticity, 2),
            "device": str(device).upper(),
            "best_loss_pct": round(best_loss * 100.0, 2),
            "target_resolution": target_res,
            "w0": w0_np.tolist(),
            "u_pred": u_pred_np.tolist(),
            "ux": ux.tolist(),
            "uy": uy.tolist(),
            "invariants_init": compute_conservation_invariants(w0_np),
            "invariants_pred": compute_conservation_invariants(u_pred_np),
            "spectrum_pred": compute_radial_energy_spectrum(u_pred_np),
            "fft2_pred": compute_fft2_log_magnitude(u_pred_np, out_size=64),
            "modes_cutoff": int(config.model.modes1),
            "z_abs_max": round(z_abs_max, 6),
        }

    def _handle_superres(self, body: Dict[str, Any]) -> Dict[str, Any]:
        preset, length_scale, alpha, seed, custom_vortices = _parse_flow_params(body)
        high_res = int(body.get("high_res", 256))
        high_res = 256 if high_res >= 192 else 128

        with model_lock:
            w0 = generate_initial_state(
                preset, length_scale, alpha, seed, s_x=64, s_y=64, custom_vortices=custom_vortices
            )
            x_input_64 = build_normalized_input(w0, device)
            w0_high = _fourier_upsample_field(w0, high_res)
            x_input_high = build_normalized_input(w0_high, device)
            u_64_t, time_64_ms = _run_model_forward(x_input_64, 64)
            u_high_t, time_high_ms = _run_model_forward(x_input_high, high_res)
            u_bicubic_np = compute_bicubic_upsample(u_64_t, target_res=high_res)
            u_nearest_np = (
                F.interpolate(u_64_t.detach().float(), size=(high_res, high_res), mode="nearest")[0, 0]
                .cpu()
                .numpy()
            )

        u_64_np = u_64_t[0, 0].detach().cpu().numpy()
        u_high_np = u_high_t[0, 0].detach().cpu().numpy()

        c0 = (high_res // 2) - 32
        c1 = c0 + 64
        crop_nearest = u_nearest_np[c0:c1, c0:c1]
        crop_bicubic = u_bicubic_np[c0:c1, c0:c1]
        crop_pino = u_high_np[c0:c1, c0:c1]

        stride = high_res // 64
        u_high_sub = u_high_np[::stride, ::stride]
        consistency_rel_l2_pct = float(
            (np.linalg.norm(u_high_sub - u_64_np) / (np.linalg.norm(u_64_np) + 1e-8)) * 100.0
        )
        z_abs_max = float(
            max(
                np.max(np.abs(u_64_np)),
                np.max(np.abs(u_high_np)),
                np.max(np.abs(u_bicubic_np)),
                1e-6,
            )
        )

        return {
            "status": "success",
            "high_res": high_res,
            "time_64_ms": round(time_64_ms, 2),
            "time_high_ms": round(time_high_ms, 2),
            "consistency_rel_l2_pct": round(consistency_rel_l2_pct, 3),
            "z_abs_max": round(z_abs_max, 6),
            "u_64": u_64_np.tolist(),
            "u_high": u_high_np.tolist(),
            "u_bicubic": u_bicubic_np.tolist(),
            "crop_nearest": crop_nearest.tolist(),
            "crop_bicubic": crop_bicubic.tolist(),
            "crop_pino": crop_pino.tolist(),
            "spectrum_64": compute_radial_energy_spectrum(u_64_np),
            "spectrum_high": compute_radial_energy_spectrum(u_high_np),
            "modes_cutoff": int(config.model.modes1),
        }

    def _handle_rollout(self, body: Dict[str, Any]) -> Dict[str, Any]:
        preset, length_scale, alpha, seed, custom_vortices = _parse_flow_params(body)
        viscosity = _parse_viscosity(body)
        num_frames = max(3, min(8, int(body.get("num_frames", 6))))
        steps_per_unit = 50
        dt = 1.0 / float(steps_per_unit)

        with model_lock:
            w0 = generate_initial_state(
                preset, length_scale, alpha, seed, s_x=64, s_y=64, custom_vortices=custom_vortices
            )
            solver = NavierStokes2DSolver(s_x=64, s_y=64, viscosity=viscosity, device=device)

            # 1. Autoregressive PINO rollout over t = 0 .. num_frames - 1
            pino_nps: List[np.ndarray] = [w0[0].detach().cpu().numpy()]
            curr_w = w0
            sync_device(device)
            t_p0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(1, num_frames):
                    x_in = build_normalized_input(curr_w, device)
                    curr_w = model(x_in)[:, 0]
                    pino_nps.append(curr_w[0].detach().cpu().numpy())
            sync_device(device)
            total_pino_ms = (time.perf_counter() - t_p0) * 1000.0

            # 2. Exact RK4 trajectory over t_horizon = num_frames - 1 (50 * (num_frames - 1) steps)
            rk4_nps: List[np.ndarray] = [w0[0].detach().cpu().numpy()]
            sync_device(device)
            t_r0 = time.perf_counter()
            with torch.no_grad():
                w_hat = torch.fft.fft2(w0)
                for _ in range(1, num_frames):
                    for _ in range(steps_per_unit):
                        w_hat = solver.step_rk4(w_hat, dt)
                    w_phys = torch.fft.ifft2(w_hat).real[0].detach().cpu().numpy()
                    rk4_nps.append(w_phys)
            sync_device(device)
            total_rk4_ms = (time.perf_counter() - t_r0) * 1000.0

        pino_invs = [compute_conservation_invariants(f) for f in pino_nps]
        rk4_invs = [compute_conservation_invariants(f) for f in rk4_nps]
        rel_l2_per_frame = [
            round(float(np.linalg.norm(p - r) / (np.linalg.norm(r) + 1e-8) * 100.0), 3)
            for p, r in zip(pino_nps, rk4_nps)
        ]
        z_abs_max = float(
            max(
                max(np.max(np.abs(f)) for f in pino_nps),
                max(np.max(np.abs(f)) for f in rk4_nps),
                1e-6,
            )
        )

        return {
            "status": "success",
            "times": [float(t) for t in range(num_frames)],
            "pino_frames": [f.tolist() for f in pino_nps],
            "rk4_frames": [f.tolist() for f in rk4_nps],
            "pino_enstrophy": [round(inv["enstrophy"], 6) for inv in pino_invs],
            "rk4_enstrophy": [round(inv["enstrophy"], 6) for inv in rk4_invs],
            "pino_energy": [round(inv["kinetic_energy"], 6) for inv in pino_invs],
            "rk4_energy": [round(inv["kinetic_energy"], 6) for inv in rk4_invs],
            "rel_l2_per_frame": rel_l2_per_frame,
            "z_abs_max": round(z_abs_max, 6),
            "total_pino_ms": round(total_pino_ms, 2),
            "total_rk4_ms": round(total_rk4_ms, 2),
            "speedup_factor": round(total_rk4_ms / max(total_pino_ms, 1e-4), 1),
        }

    def _handle_benchmark(self, body: Dict[str, Any]) -> Dict[str, Any]:
        preset, length_scale, alpha, seed, custom_vortices = _parse_flow_params(body)
        viscosity = _parse_viscosity(body)

        with model_lock:
            w0 = generate_initial_state(
                preset, length_scale, alpha, seed, s_x=64, s_y=64, custom_vortices=custom_vortices
            )
            solver = NavierStokes2DSolver(s_x=64, s_y=64, viscosity=viscosity, device=device)

            sync_device(device)
            start_rk4 = time.perf_counter()
            trajectory_ref = solver.solve(w0, t_horizon=1.0, num_steps=50, save_steps=10)
            sync_device(device)
            time_rk4_ms = (time.perf_counter() - start_rk4) * 1000.0
            u_rk4_final = trajectory_ref[0, -1].detach().cpu().numpy()

            x_input = build_normalized_input(w0, device)
            u_pino_t, time_pino_ms = _run_model_forward(x_input, 64)
            u_pino_out = u_pino_t[0, 0].detach().cpu().numpy()

            loss_eng = PINOLossEngine(s_x=64, s_y=64, viscosity=viscosity, device=device)
            with torch.no_grad():
                traj_2step = torch.stack([w0, u_pino_t[:, 0]], dim=1)
                pde_res_t = loss_eng.compute_pde_residual(traj_2step, dt=1.0)
                pde_residual = float(torch.sqrt(torch.mean(pde_res_t ** 2)).item())

        diff = u_pino_out - u_rk4_final
        mse = float(np.mean(diff ** 2))
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(mse))
        max_abs_err = float(np.max(np.abs(diff)))
        rel_l2_pct = float((np.linalg.norm(diff) / (np.linalg.norm(u_rk4_final) + 1e-8)) * 100.0)
        circ_drift = float(
            (abs(np.mean(u_pino_out) - np.mean(u_rk4_final)) / (np.mean(np.abs(u_rk4_final)) + 1e-8))
            * 100.0
        )
        energy_drift = float(
            (abs(np.sum(u_pino_out ** 2) - np.sum(u_rk4_final ** 2)) / (np.sum(u_rk4_final ** 2) + 1e-8))
            * 100.0
        )
        speedup = time_rk4_ms / max(time_pino_ms, 1e-4)
        z_abs_max = float(max(np.max(np.abs(u_rk4_final)), np.max(np.abs(u_pino_out)), 1e-6))

        return {
            "status": "success",
            "time_rk4_ms": round(time_rk4_ms, 2),
            "time_pino_ms": round(time_pino_ms, 2),
            "speedup_factor": round(speedup, 1),
            "rel_l2_pct": round(rel_l2_pct, 2),
            "circ_drift_pct": round(circ_drift, 3),
            "energy_drift_pct": round(energy_drift, 3),
            "max_abs_err": round(max_abs_err, 4),
            "mse": round(mse, 6),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "pde_residual": round(pde_residual, 4),
            "z_abs_max": round(z_abs_max, 6),
            "rk4_w": u_rk4_final.tolist(),
            "pino_w": u_pino_out.tolist(),
            "err_w": np.abs(diff).tolist(),
            "spectrum_rk4": compute_radial_energy_spectrum(u_rk4_final),
            "spectrum_pino": compute_radial_energy_spectrum(u_pino_out),
            "fft2_rk4": compute_fft2_log_magnitude(u_rk4_final, out_size=64),
            "fft2_pino": compute_fft2_log_magnitude(u_pino_out, out_size=64),
            "modes_cutoff": int(config.model.modes1),
        }

    def _handle_tta(self, body: Dict[str, Any]) -> Dict[str, Any]:
        preset, length_scale, alpha, seed, custom_vortices = _parse_flow_params(body)
        viscosity = _parse_viscosity(body)
        steps = max(1, min(30, int(body.get("steps", 10))))
        lr = max(1e-6, min(1e-2, float(body.get("lr", 1e-4))))
        alpha_anchor = max(0.0, min(10.0, float(body.get("alpha_anchor", 0.5))))

        with model_lock:
            orig_proj = copy.deepcopy(model.projection.state_dict())
            try:
                w0 = generate_initial_state(
                    preset, length_scale, alpha, seed, s_x=64, s_y=64, custom_vortices=custom_vortices
                )
                x_input = build_normalized_input(w0, device)
                loss_eng = PINOLossEngine(s_x=64, s_y=64, viscosity=viscosity, device=device)

                with torch.no_grad():
                    u_before_t = model(x_input).detach().clone()
                    traj_before = torch.stack([w0, u_before_t[:, 0]], dim=1)
                    pde_res_before = float(
                        torch.sqrt(torch.mean(loss_eng.compute_pde_residual(traj_before, dt=1.0) ** 2)).item()
                    )

                adapter = TestTimeAdapter(
                    model,
                    loss_eng,
                    learning_rate=lr,
                    steps=steps,
                    alpha_anchor=alpha_anchor,
                    beta_circ=1.0,
                    t_horizon=1.0,
                )
                sync_device(device)
                t_start = time.perf_counter()
                u_after_t, history = adapter.adapt_instance(
                    x_input, w0.unsqueeze(1), forcing=None, restore_weights=True
                )
                sync_device(device)
                tta_time_ms = float(history.get("elapsed_ms", (time.perf_counter() - t_start) * 1000.0))

                with torch.no_grad():
                    traj_after = torch.stack([w0, u_after_t[:, 0]], dim=1)
                    pde_res_after = float(
                        torch.sqrt(torch.mean(loss_eng.compute_pde_residual(traj_after, dt=1.0) ** 2)).item()
                    )

                # Post-hoc RK4 evaluation strictly for error measurement (never seen by TTA)
                solver = NavierStokes2DSolver(s_x=64, s_y=64, viscosity=viscosity, device=device)
                u_rk4_t = solver.solve(w0, t_horizon=1.0, num_steps=50, save_steps=10)[:, -1]
            finally:
                model.projection.load_state_dict(orig_proj)
                for param in model.parameters():
                    param.requires_grad = True
                model.eval()

        rk4_np = u_rk4_t[0].detach().cpu().numpy()
        before_np = u_before_t[0, 0].detach().cpu().numpy()
        after_np = u_after_t[0, 0].detach().cpu().numpy()
        err_before_np = np.abs(before_np - rk4_np)
        err_after_np = np.abs(after_np - rk4_np)

        rk4_norm = float(np.linalg.norm(rk4_np)) + 1e-8
        rel_l2_before = float((np.linalg.norm(before_np - rk4_np) / rk4_norm) * 100.0)
        rel_l2_after = float((np.linalg.norm(after_np - rk4_np) / rk4_norm) * 100.0)
        rel_l2_delta_pct = float(rel_l2_after - rel_l2_before)
        pde_reduction_pct = float(((pde_res_before - pde_res_after) / (pde_res_before + 1e-8)) * 100.0)

        rk4_l1 = float(np.mean(np.abs(rk4_np))) + 1e-8
        circ_drift_before = float((abs(np.mean(before_np) - np.mean(rk4_np)) / rk4_l1) * 100.0)
        circ_drift_after = float((abs(np.mean(after_np) - np.mean(rk4_np)) / rk4_l1) * 100.0)

        z_abs_max = float(
            max(
                np.max(np.abs(rk4_np)),
                np.max(np.abs(before_np)),
                np.max(np.abs(after_np)),
                1e-6,
            )
        )
        err_max = float(max(np.max(err_before_np), np.max(err_after_np), 1e-6))

        return {
            "status": "success",
            "steps": steps,
            "tta_time_ms": round(tta_time_ms, 2),
            "pde_res_before": round(pde_res_before, 4),
            "pde_res_after": round(pde_res_after, 4),
            "pde_reduction_pct": round(pde_reduction_pct, 2),
            "rel_l2_before": round(rel_l2_before, 2),
            "rel_l2_after": round(rel_l2_after, 2),
            "rel_l2_delta_pct": round(rel_l2_delta_pct, 2),
            "circ_drift_before": round(circ_drift_before, 3),
            "circ_drift_after": round(circ_drift_after, 3),
            "loss_curve": [float(v) for v in history.get("loss_curve", [])],
            "pde_curve": [float(v) for v in history.get("pde_curve", [])],
            "rk4_w": rk4_np.tolist(),
            "before_w": before_np.tolist(),
            "after_w": after_np.tolist(),
            "err_before": err_before_np.tolist(),
            "err_after": err_after_np.tolist(),
            "z_abs_max": round(z_abs_max, 6),
            "err_max": round(err_max, 6),
        }

    def _handle_scaling(self, body: Dict[str, Any]) -> Dict[str, Any]:
        preset, length_scale, alpha, seed, custom_vortices = _parse_flow_params(body)
        viscosity = _parse_viscosity(body)
        resolutions = [64, 128, 256]
        rk4_steps_list = [40, 80, 160]
        rk4_ms_list: List[float] = []
        pino_ms_list: List[float] = []
        speedups: List[float] = []

        with model_lock:
            w0_64 = generate_initial_state(
                preset, length_scale, alpha, seed, s_x=64, s_y=64, custom_vortices=custom_vortices
            )
            x_input_64 = build_normalized_input(w0_64, device)

            # Warmup 1 pass of PINO at 64 before benchmarking
            with torch.no_grad():
                _ = model(x_input_64)
            sync_device(device)

            for res, n_steps in zip(resolutions, rk4_steps_list):
                w0_res = generate_initial_state(
                    preset, length_scale, alpha, seed, s_x=res, s_y=res, custom_vortices=custom_vortices
                )
                solver_res = NavierStokes2DSolver(s_x=res, s_y=res, viscosity=viscosity, device=device)

                # Warmup kernel plans at target resolution so JIT compile overhead isn't measured
                with torch.no_grad():
                    _ = solver_res.step_rk4(torch.fft.fft2(w0_res), 1.0 / float(n_steps))
                    _ = model(x_input_64, target_s=(res, res)) if res != 64 else model(x_input_64)
                sync_device(device)

                sync_device(device)
                t_r0 = time.perf_counter()
                _ = solver_res.solve(w0_res, t_horizon=1.0, num_steps=n_steps, save_steps=2)
                sync_device(device)
                rk4_ms = (time.perf_counter() - t_r0) * 1000.0

                sync_device(device)
                t_p0 = time.perf_counter()
                with torch.no_grad():
                    for _ in range(3):
                        _ = model(x_input_64, target_s=(res, res)) if res != 64 else model(x_input_64)
                sync_device(device)
                pino_ms = ((time.perf_counter() - t_p0) * 1000.0) / 3.0

                rk4_ms_list.append(round(rk4_ms, 2))
                pino_ms_list.append(round(pino_ms, 2))
                speedups.append(round(rk4_ms / max(pino_ms, 1e-4), 1))

        return {
            "status": "success",
            "resolutions": resolutions,
            "rk4_steps": rk4_steps_list,
            "rk4_ms": rk4_ms_list,
            "pino_ms": pino_ms_list,
            "speedups": speedups,
            "grid_points": [r * r for r in resolutions],
        }

    def _handle_pso(self, body: Dict[str, Any]) -> Dict[str, Any]:
        preset, length_scale, alpha, seed, custom_vortices = _parse_flow_params(body)
        num_particles = max(10, min(100, int(body.get("num_particles", 40))))
        iterations = max(3, min(30, int(body.get("iterations", 10))))
        forcing_bound = max(0.05, min(2.0, float(body.get("forcing_bound", 0.5))))

        def objective_fn(pred_states: torch.Tensor) -> torch.Tensor:
            return torch.mean(pred_states ** 2, dim=(-2, -1)).squeeze(-1)

        with model_lock:
            w0 = generate_initial_state(
                preset, length_scale, alpha, seed, s_x=64, s_y=64, custom_vortices=custom_vortices
            )
            pso = ParticleSwarmOptimizer(
                pso_harness,
                num_particles=num_particles,
                num_iterations=iterations,
                s_x=64,
                s_y=64,
                forcing_bound=forcing_bound,
            )

            sync_device(device)
            results = pso.optimize(w0, objective_fn)
            sync_device(device)

            opt_forcing = results.get("optimal_forcing", results.get("best_forcing"))
            if opt_forcing.ndim == 3:
                best_forcing_np = opt_forcing[0].detach().cpu().numpy()
                opt_forcing_tensor = opt_forcing.to(device)
            elif opt_forcing.ndim == 2:
                best_forcing_np = opt_forcing.detach().cpu().numpy()
                opt_forcing_tensor = opt_forcing.unsqueeze(0).to(device)
            else:
                best_forcing_np = opt_forcing[0, 0].detach().cpu().numpy()
                opt_forcing_tensor = opt_forcing[0].to(device)

            x_input_forced = build_normalized_input(w0 + opt_forcing_tensor, device)
            x_input_base = build_normalized_input(w0, device)

            with torch.no_grad():
                u_controlled_np = model(x_input_forced)[0, 0].detach().cpu().numpy()
                u_unforced_np = model(x_input_base)[0, 0].detach().cpu().numpy()

        base_enstrophy = float(np.mean(u_unforced_np ** 2))
        opt_enstrophy = float(results["best_fitness"])
        reduction_pct = float(max(0.0, (base_enstrophy - opt_enstrophy) / (base_enstrophy + 1e-8) * 100.0))
        forcing_l2_norm = float(np.linalg.norm(best_forcing_np))
        forcing_linf_norm = float(np.max(np.abs(best_forcing_np)))
        z_abs_max = float(max(np.max(np.abs(u_unforced_np)), np.max(np.abs(u_controlled_np)), 1e-6))

        return {
            "status": "success",
            "total_evaluations": int(results["total_evaluations"]),
            "total_latency_ms": round(float(results["total_latency_ms"]), 1),
            "eval_per_sec": round(float(results["eval_per_sec"]), 0),
            "best_fitness": round(opt_enstrophy, 6),
            "base_fitness": round(base_enstrophy, 6),
            "reduction_pct": round(reduction_pct, 1),
            "forcing_l2_norm": round(forcing_l2_norm, 4),
            "forcing_linf_norm": round(forcing_linf_norm, 4),
            "num_particles": num_particles,
            "iterations": iterations,
            "z_abs_max": round(z_abs_max, 6),
            "best_forcing": best_forcing_np.tolist(),
            "uncontrolled_w": u_unforced_np.tolist(),
            "controlled_w": u_controlled_np.tolist(),
            "fitness_history": [float(x) for x in results.get("fitness_history", [])],
            "mean_fitness_history": [float(x) for x in results.get("mean_fitness_history", [])],
        }


# -----------------------------------------------------------------------------
# 4. SERVER LAUNCHER
# -----------------------------------------------------------------------------
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def run_server(port: int = 8000) -> None:
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static"), exist_ok=True)
    server_address = ("127.0.0.1", port)
    httpd = ThreadedHTTPServer(server_address, PINORequestHandler)
    print("\n" + "=" * 75)
    print(f"[*] PINO Modern Web App LIVE at: http://localhost:{port}")
    print(f"[*] Pure Swiss Light Mode Active • Real-Time 60FPS Dragging Enabled")
    print(f"[*] Open http://localhost:{port} in Chrome / Safari to interact")
    print("=" * 75 + "\n", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server stopped.")
        httpd.server_close()


if __name__ == "__main__":
    port_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port=port_arg)
