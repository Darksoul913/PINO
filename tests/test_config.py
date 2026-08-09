"""Unit test for PINO Configuration System."""

from pino.config import PINOConfig, DomainConfig, PDEConfig, ModelConfig, LossConfig


def test_pino_config_defaults():
    config = PINOConfig()
    assert config.domain.s_x == 64
    assert config.domain.eval_s_x == 256
    assert config.pde.reynolds_number == 100.0
    assert config.model.modes1 == 16
    assert config.loss.weight_ic == 10.0
    print("PINO Configuration defaults test passed successfully!")


if __name__ == "__main__":
    test_pino_config_defaults()
