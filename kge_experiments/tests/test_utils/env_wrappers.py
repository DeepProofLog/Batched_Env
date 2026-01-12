"""
Environment wrappers for parity testing and AAAI26 paper experiments.

These wrappers configure EnvVec with specific defaults for different use cases:
- ParityEnvVec: For parity tests against reference implementations
- AAAI26EnvVec: For reproducing AAAI26 paper results
"""

from typing import Optional
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from env import EnvVec
from unification import UnificationEngineVectorized


class ParityEnvVec(EnvVec):
    """EnvVec with parity test defaults.

    Automatically sets:
    - use_exact_memory = True (for collision-free matching)
    - skip_unary_actions = False (required for parity with tensor reference)
    - memory_pruning = True
    - compile = False (parity tests may have data-dependent branches)
    """
    def __init__(self, vec_engine: UnificationEngineVectorized, batch_size: int = 100, **kwargs):
        kwargs.setdefault('use_exact_memory', True)
        kwargs.setdefault('skip_unary_actions', False)
        kwargs.setdefault('memory_pruning', True)
        kwargs.setdefault('compile', False)
        super().__init__(vec_engine, batch_size, **kwargs)


class AAAI26EnvVec(EnvVec):
    """EnvVec with AAAI26 paper defaults.

    Automatically sets:
    - use_exact_memory = True (for reproducibility)
    - skip_unary_actions = True (AAAI26 unary action skipping)
    - memory_pruning = True
    """
    def __init__(self, vec_engine: UnificationEngineVectorized, batch_size: int = 100, **kwargs):
        kwargs.setdefault('use_exact_memory', True)
        kwargs.setdefault('skip_unary_actions', True)
        kwargs.setdefault('memory_pruning', True)
        super().__init__(vec_engine, batch_size, **kwargs)
