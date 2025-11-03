"""
TorchRL-compatible model for Neural-guided Grounding.

This module provides policy and value networks compatible with TorchRL's API,
migrated from the original Stable-Baselines3 implementation.

NOTE: This file now redirects to the modular PPO implementation in the ppo/ directory.
For backward compatibility, all classes and functions are re-exported here.
"""

# Import everything from the new modular implementation
from ppo.ppo_model import (
    PolicyNetwork,
    ValueNetwork,
    EmbeddingExtractor,
    ActorCriticModel,
    TorchRLActorModule,
    TorchRLValueModule,
    create_torchrl_modules,
)

# Re-export for backward compatibility
__all__ = [
    "PolicyNetwork",
    "ValueNetwork",
    "EmbeddingExtractor",
    "ActorCriticModel",
    "TorchRLActorModule",
    "TorchRLValueModule",
    "create_torchrl_modules",
]
