"""
Physics-Informed Neural Operator (PINO) Framework for Dynamic Spatiotemporal PDEs
"""

import warnings

# Suppress PyTorch 2.x internal ATen Metal/MPS backend buffer resize deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*resized since it had shape.*")

__version__ = "0.1.0"
