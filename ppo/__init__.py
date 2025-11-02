"""
PPO (Proximal Policy Optimization) Module

This module provides a modular implementation of PPO for Neural-guided Grounding,
using TorchRL as the underlying RL framework.
"""

from .model import (
    PolicyNetwork,
    ValueNetwork,
    EmbeddingExtractor,
    ActorCriticModel,
    create_torchrl_modules,
)

from .rollout import RolloutCollector, collect_rollouts

from .learner import PPOLearner

from .ppo_agent import PPOAgent

__all__ = [
    "PolicyNetwork",
    "ValueNetwork",
    "EmbeddingExtractor",
    "ActorCriticModel",
    "create_torchrl_modules",
    "RolloutCollector",
    "collect_rollouts",
    "PPOLearner",
    "PPOAgent",
]
