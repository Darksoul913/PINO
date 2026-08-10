"""Unit test for PINO Configuration System."""

import unittest
from pino.config import PINOConfig, DomainConfig, PDEConfig, ModelConfig, LossConfig


class TestConfig(unittest.TestCase):
    def test_pino_config_defaults(self):
        config = PINOConfig()
        self.assertEqual(config.domain.s_x, 64)
        self.assertEqual(config.domain.eval_s_x, 256)
        self.assertEqual(config.pde.reynolds_number, 100.0)
        self.assertEqual(config.model.modes1, 16)
        self.assertEqual(config.loss.weight_ic, 1.0)


if __name__ == "__main__":
    unittest.main()
