"""
MCTS (Monte Carlo Tree Search) Module for DeepProofLog.

This module implements MuZero-style MCTS for knowledge graph reasoning,
providing an alternative to PPO for learning optimal proof strategies.

Key Components:
    - MCTSConfig: Hyperparameters for MCTS algorithm
    - MCTSPolicy: Policy/Value network for MCTS evaluation
    - MCTS: Tree search algorithm with UCB selection
    - MCTSReplayBuffer: Experience storage for off-policy training
    - MuZeroTrainer: Main training loop combining MCTS with neural network updates

Usage:
    from kge_experiments.mcts import MCTSConfig, MuZeroTrainer

    config = MCTSConfig(num_simulations=50)
    trainer = MuZeroTrainer(config, env, policy)
    trainer.learn(total_timesteps=100000)
"""

from .config import MCTSConfig
from .tree import MCTS, Node, MinMaxStats
from .networks import MCTSPolicy
from .replay_buffer import MCTSReplayBuffer
from .trainer import MuZeroTrainer
from .optimized_trainer import OptimizedMuZeroTrainer

__all__ = [
    'MCTSConfig',
    'MCTS',
    'Node',
    'MinMaxStats',
    'MCTSPolicy',
    'MCTSReplayBuffer',
    'MuZeroTrainer',
    'OptimizedMuZeroTrainer',
]
