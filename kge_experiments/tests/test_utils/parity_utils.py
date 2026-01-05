"""
Parity utilities - DEPRECATED.

This module previously contained TensorDictEnvWrapper, evaluate_parity, and
_setup_parity_eval_step which were used for exact reproducibility testing
between tensor and vectorized implementations.

After unifying parity and non-parity modes, these functions are no longer needed.
Use the production ppo.evaluate() method instead.

For reference, the functionality was:
- TensorDictEnvWrapper: Wrapped EnvVec for TensorDict API compatibility
- evaluate_parity: Evaluation with exact RNG parity for testing
- _setup_parity_eval_step: Compiled step setup for evaluate_parity

All evaluation should now use ppo.evaluate() directly.
"""

# Placeholder imports for backwards compatibility with old backup files
import torch
from env import EnvVec


class TensorDictEnvWrapper:
    """DEPRECATED: TensorDict wrapper for EnvVec.

    This class is kept only for backwards compatibility with backup files.
    Do not use in new code.
    """
    def __init__(self, env: EnvVec):
        raise DeprecationWarning(
            "TensorDictEnvWrapper is deprecated. "
            "Use EnvVec directly with ppo.evaluate() instead."
        )
