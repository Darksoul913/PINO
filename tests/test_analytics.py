"""Unit tests for pure PyTorch/NumPy scientific analytics helpers in pino.analytics."""

import math
import unittest
import numpy as np
import torch

from pino.analytics import (
    compute_streamlines,
    compute_conservation_invariants,
    compute_radial_energy_spectrum,
    compute_fft2_log_magnitude,
    generate_custom_vortex_field,
    compute_bicubic_upsample,
    build_normalized_input,
)


class TestAnalytics(unittest.TestCase):
    def _make_taylor_green(self, s_x: int = 64, s_y: int = 64) -> np.ndarray:
        x = np.linspace(0.0, 2.0 * math.pi * (s_x - 1) / s_x, s_x)
        y = np.linspace(0.0, 2.0 * math.pi * (s_y - 1) / s_y, s_y)
        X, Y = np.meshgrid(x, y, indexing="ij")
        return np.sin(X) * np.cos(Y)

    def test_compute_streamlines_biot_savart(self):
        w_np = self._make_taylor_green(64, 64)
        w_t = torch.tensor(w_np, dtype=torch.float32)
        ux, uy = compute_streamlines(w_t, 64, 64, device="cpu")

        self.assertEqual(ux.shape, (64, 64))
        self.assertEqual(uy.shape, (64, 64))

        # Analytical solution for omega = sin(x)cos(y):
        # psi = 0.5*sin(x)cos(y), ux = -0.5*sin(x)sin(y), uy = -0.5*cos(x)cos(y)
        x = np.linspace(0.0, 2.0 * math.pi * 63 / 64, 64)
        y = np.linspace(0.0, 2.0 * math.pi * 63 / 64, 64)
        X, Y = np.meshgrid(x, y, indexing="ij")
        ux_exact = -0.5 * np.sin(X) * np.sin(Y)
        uy_exact = -0.5 * np.cos(X) * np.cos(Y)

        self.assertLess(np.max(np.abs(ux - ux_exact)), 1e-4)
        self.assertLess(np.max(np.abs(uy - uy_exact)), 1e-4)

        # Also test 3D singleton batch input
        ux_3d, uy_3d = compute_streamlines(w_t.unsqueeze(0), 64, 64, device="cpu")
        self.assertEqual(ux_3d.shape, (64, 64))
        self.assertEqual(uy_3d.shape, (64, 64))

    def test_compute_conservation_invariants(self):
        w_np = self._make_taylor_green(64, 64)
        inv = compute_conservation_invariants(w_np)

        self.assertEqual(
            set(inv.keys()),
            {"circulation", "enstrophy", "kinetic_energy", "peak_vorticity"},
        )
        # Analytical values for sin(x)cos(y):
        # mean = 0, enstrophy = 0.5 * 0.25 = 0.125, KE = 0.0625, peak = 1.0
        self.assertAlmostEqual(inv["circulation"], 0.0, places=6)
        self.assertAlmostEqual(inv["enstrophy"], 0.125, places=6)
        self.assertAlmostEqual(inv["kinetic_energy"], 0.0625, places=6)
        self.assertGreater(inv["peak_vorticity"], 0.99)
        self.assertLessEqual(inv["peak_vorticity"], 1.0 + 1e-6)

    def test_compute_radial_energy_spectrum(self):
        w_np = self._make_taylor_green(64, 64)
        spec = compute_radial_energy_spectrum(w_np)

        self.assertEqual(set(spec.keys()), {"k", "energy", "enstrophy"})
        self.assertEqual(spec["k"], list(range(1, 33)))
        self.assertEqual(len(spec["energy"]), 32)
        self.assertEqual(len(spec["enstrophy"]), 32)
        self.assertTrue(all(e >= 1e-12 for e in spec["energy"]))
        self.assertTrue(all(z >= 1e-12 for z in spec["enstrophy"]))

        # For sin(x)cos(y), |k| = sqrt(2) -> round(sqrt(2)) = 1, so k=1 holds all energy/enstrophy
        self.assertAlmostEqual(spec["energy"][0], 0.0625, places=5)
        self.assertAlmostEqual(spec["enstrophy"][0], 0.125, places=5)

        # Parseval consistency: summed radial spectrum matches physical invariants on bandlimited fields
        x = np.linspace(0.0, 2.0 * math.pi * 63 / 64, 64)
        y = np.linspace(0.0, 2.0 * math.pi * 63 / 64, 64)
        X, Y = np.meshgrid(x, y, indexing="ij")
        w_multi = np.sin(X) * np.cos(Y) + 0.5 * np.cos(3.0 * X + 2.0 * Y)
        inv_multi = compute_conservation_invariants(w_multi)
        spec_multi = compute_radial_energy_spectrum(w_multi)
        self.assertAlmostEqual(sum(spec_multi["energy"]), inv_multi["kinetic_energy"], places=5)
        self.assertAlmostEqual(sum(spec_multi["enstrophy"]), inv_multi["enstrophy"], places=5)

    def test_compute_fft2_log_magnitude(self):
        w_64 = np.random.RandomState(0).randn(64, 64)
        mag_64 = compute_fft2_log_magnitude(w_64, out_size=64)
        self.assertIsInstance(mag_64, list)
        self.assertEqual(len(mag_64), 64)
        self.assertEqual(len(mag_64[0]), 64)

        w_256 = self._make_taylor_green(256, 256)
        mag_cropped = compute_fft2_log_magnitude(w_256, out_size=64)
        self.assertEqual(len(mag_cropped), 64)
        self.assertEqual(len(mag_cropped[0]), 64)
        # Verify DC peak of sin(x)cos(y) modes at (+/-1, +/-1) sits right around center (32, 32)
        arr = np.array(mag_cropped)
        self.assertGreater(arr[33, 33], -2.0)

    def test_generate_custom_vortex_field(self):
        # Empty list falls back to clean counter-rotating dipole
        fallback = generate_custom_vortex_field([], s_x=64, s_y=64, device="cpu")
        self.assertEqual(fallback.shape, (1, 64, 64))
        self.assertAlmostEqual(float(fallback.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(fallback.std()), 1.0, places=4)

        # Periodic wrap test: vortex near x=0.98 and x=0.02 should sit close in periodic distance
        vortices = [
            {"x": 0.98, "y": 0.50, "strength": 1.0, "sigma": 0.08},
            {"x": 0.25, "y": 0.25, "strength": -1.0, "sigma": 0.08},
        ]
        custom = generate_custom_vortex_field(vortices, s_x=64, s_y=64, device="cpu")
        self.assertEqual(custom.shape, (1, 64, 64))
        self.assertAlmostEqual(float(custom.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(custom.std()), 1.0, places=4)
        # Check continuity across periodic boundary x=0 and x=63/64
        boundary_diff = abs(float(custom[0, 0, 32]) - float(custom[0, -1, 32]))
        self.assertLess(boundary_diff, 0.5)

        # Zero-strength edge case must not divide by zero / produce NaN
        zero_field = generate_custom_vortex_field(
            [{"x": 0.5, "y": 0.5, "strength": 0.0, "sigma": 0.08}],
            s_x=32,
            s_y=32,
            device="cpu",
        )
        self.assertFalse(torch.isnan(zero_field).any())
        self.assertAlmostEqual(float(zero_field.abs().max()), 0.0, places=6)

    def test_compute_bicubic_upsample(self):
        u_64 = torch.randn(1, 1, 64, 64)
        up = compute_bicubic_upsample(u_64, target_res=256)
        self.assertIsInstance(up, np.ndarray)
        self.assertEqual(up.shape, (256, 256))
        self.assertFalse(np.isnan(up).any())

        # Also accept 2D tensor
        up_128 = compute_bicubic_upsample(u_64[0, 0], target_res=128)
        self.assertEqual(up_128.shape, (128, 128))

    def test_build_normalized_input(self):
        for shape in [(64, 64), (2, 64, 64), (2, 1, 64, 64)]:
            w0 = torch.randn(*shape)
            x_in = build_normalized_input(w0, device="cpu")
            expected_b = 1 if len(shape) == 2 else shape[0]
            self.assertEqual(x_in.shape, (expected_b, 3, 64, 64))
            self.assertAlmostEqual(float(x_in[0, 0, 0, 0]), 0.0, places=6)
            self.assertAlmostEqual(float(x_in[0, 0, -1, 0]), 63.0 / 64.0, places=6)
            self.assertAlmostEqual(float(x_in[0, 1, 0, -1]), 63.0 / 64.0, places=6)


if __name__ == "__main__":
    unittest.main()
