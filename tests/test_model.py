"""Unit tests for PINO architecture & Fourier convolution layers."""

import unittest
import torch
from pino.models.fourier_layer import SpectralConv2d
from pino.models.mlp import LiftingBlock, ProjectionBlock
from pino.models.pino_net import PINO2D
from pino.config import ModelConfig


class TestModel(unittest.TestCase):
    def test_spectral_conv2d_forward(self):
        layer = SpectralConv2d(in_channels=32, out_channels=32, modes1=12, modes2=12)
        x = torch.randn(2, 32, 64, 64)
        out = layer(x)
        self.assertEqual(out.shape, (2, 32, 64, 64))
        self.assertFalse(torch.isnan(out).any())

    def test_pino2d_forward(self):
        model = PINO2D(in_channels=3, out_channels=1, hidden_dim=32, modes1=12, modes2=12, num_layers=3)
        x = torch.randn(4, 3, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (4, 1, 64, 64))
        self.assertFalse(torch.isnan(out).any())

    def test_pino2d_zero_shot_super_resolution(self):
        model = PINO2D(in_channels=3, out_channels=1, hidden_dim=32, modes1=12, modes2=12, num_layers=3)
        x = torch.randn(2, 3, 64, 64)  # Coarse input (64x64)
        out_super_res = model(x, target_s=(256, 256))  # Target high-res (256x256)
        self.assertEqual(out_super_res.shape, (2, 1, 256, 256))
        self.assertFalse(torch.isnan(out_super_res).any())


if __name__ == "__main__":
    unittest.main()
