"""
PPO (Proximal Policy Optimization) Module

This module provides a modular implementation of PPO for Neural-guided Grounding,
using TorchRL as the underlying RL framework.
"""

from .ppo_model import (
    PolicyNetwork,
    ValueNetwork,
    EmbeddingExtractor,
    ActorCriticModel,
    create_torchrl_modules,
)

from .ppo_rollout import collect_rollouts

from .ppo_agent import PPOAgent

__all__ = [
    "PolicyNetwork",
    "ValueNetwork",
    "EmbeddingExtractor",
    "ActorCriticModel",
    "create_torchrl_modules",
    "RolloutCollector",
    "collect_rollouts",
    "PPOAgent",
]
