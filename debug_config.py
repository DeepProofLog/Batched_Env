"""
Debug Configuration System for RL Pipeline

This module provides a centralized debug configuration system that can be used
across all components of the RL pipeline (environment, prover, agent, model, rollouts).

Usage:
    from debug_config import DebugConfig
    
    # Create debug config
    debug_cfg = DebugConfig(
        debug_env=True,
        debug_model=True,
        debug_agent=True
    )
    
    # Pass to components
    env = BatchedEnv(..., debug_config=debug_cfg)
    ppo = PPO(..., debug_config=debug_cfg)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch


@dataclass
class DebugConfig:
    """
    Centralized debug configuration for the RL pipeline.
    
    Main Categories:
        debug_env: Environment-level debugging (state transitions, rewards, etc.)
        debug_prover: Prover-level debugging (unification, proof search, etc.)
        debug_agent: Agent-level debugging (rollout statistics, training metrics, etc.)
        debug_model: Model-level debugging (logits, actions, distributions, etc.)
        debug_rollouts: Detailed rollout debugging (step-by-step info)
    
    Each category can be:
        - False/0: Disabled
        - True/1: Basic info
        - 2+: More detailed info (higher = more verbose)
    """
    
    # Main debug flags
    debug_env: int = 0
    debug_prover: int = 0
    debug_agent: int = 0
    debug_model: int = 0
    debug_rollouts: int = 0
    
    # Specific debug options for fine-grained control
    # Agent debugging
    debug_agent_rollout_stats: bool = True  # Show rollout statistics (avg actions, steps, success)
    debug_agent_train_stats: bool = True    # Show training statistics (loss, entropy, etc.)
    debug_agent_episode_info: bool = True   # Show per-episode information
    
    # Model debugging
    debug_model_logits: bool = False        # Show logits computation
    debug_model_actions: bool = False       # Show action selection
    debug_model_distribution: bool = False  # Show distribution parameters
    debug_model_entropy: bool = False       # Show entropy computation details
    debug_model_values: bool = False        # Show value predictions
    debug_model_action_mask: bool = False   # Show action masking
    
    # Rollout debugging
    debug_rollouts_steps: bool = False      # Show each step during rollouts
    debug_rollouts_rewards: bool = False    # Show reward details
    debug_rollouts_dones: bool = False      # Show episode terminations
    
    # Environment debugging
    debug_env_transitions: bool = False     # Show state transitions
    debug_env_action_spaces: bool = False   # Show available actions
    
    # Prover debugging
    debug_prover_unification: bool = False  # Show unification steps
    debug_prover_search: bool = False       # Show proof search
    
    # Sampling and frequency control
    debug_sample_envs: Optional[int] = None      # Only debug first N environments (None = all)
    debug_sample_frequency: int = 1              # Debug every N steps/episodes
    debug_max_actions_display: int = 10          # Max actions to display in detail
    
    # Output control
    debug_use_helper: bool = True                # Use DebugHelper for formatted output
    debug_prefix: str = "[Debug]"                # Prefix for debug messages
    
    def __post_init__(self):
        """Convert boolean flags to integers for consistency."""
        # Convert main flags to integers
        self.debug_env = int(self.debug_env)
        self.debug_prover = int(self.debug_prover)
        self.debug_agent = int(self.debug_agent)
        self.debug_model = int(self.debug_model)
        self.debug_rollouts = int(self.debug_rollouts)
    
    def is_enabled(self, category: str, level: int = 1) -> bool:
        """
        Check if debugging is enabled for a category at a given level.
        
        Args:
            category: One of 'env', 'prover', 'agent', 'model', 'rollouts'
            level: Minimum debug level required (default: 1)
        
        Returns:
            True if debugging is enabled at the specified level
        """
        attr_name = f"debug_{category}"
        if not hasattr(self, attr_name):
            return False
        return getattr(self, attr_name) >= level
    
    def should_debug_env(self, env_idx: Optional[int] = None) -> bool:
        """Check if environment should be debugged based on sampling."""
        if not self.is_enabled('env'):
            return False
        if self.debug_sample_envs is None:
            return True
        if env_idx is None:
            return True
        return env_idx < self.debug_sample_envs
    
    def should_debug_step(self, step: int) -> bool:
        """Check if step should be debugged based on frequency."""
        return step % self.debug_sample_frequency == 0
    
    def get_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return {
            'debug_env': self.debug_env,
            'debug_prover': self.debug_prover,
            'debug_agent': self.debug_agent,
            'debug_model': self.debug_model,
            'debug_rollouts': self.debug_rollouts,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DebugConfig':
        """Create DebugConfig from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def create_minimal(cls) -> 'DebugConfig':
        """Create config with no debugging enabled."""
        return cls()
    
    @classmethod
    def create_agent_debug(cls) -> 'DebugConfig':
        """Create config for debugging agent (rollout stats, training metrics)."""
        return cls(
            debug_agent=2,
            debug_agent_rollout_stats=True,
            debug_agent_train_stats=True,
            debug_agent_episode_info=True,
        )
    
    @classmethod
    def create_model_debug(cls) -> 'DebugConfig':
        """Create config for debugging model (logits, actions, entropy)."""
        return cls(
            debug_model=2,
            debug_model_logits=True,
            debug_model_actions=True,
            debug_model_distribution=True,
            debug_model_entropy=True,
            debug_model_values=True,
            debug_model_action_mask=True,
            debug_sample_envs=3,  # Only show first 3 envs
        )
    
    @classmethod
    def create_full_debug(cls) -> 'DebugConfig':
        """Create config with full debugging enabled."""
        return cls(
            debug_env=2,
            debug_prover=1,
            debug_agent=2,
            debug_model=2,
            debug_rollouts=2,
            debug_agent_rollout_stats=True,
            debug_agent_train_stats=True,
            debug_agent_episode_info=True,
            debug_model_logits=True,
            debug_model_actions=True,
            debug_model_distribution=True,
            debug_model_entropy=True,
            debug_model_values=True,
            debug_model_action_mask=True,
            debug_rollouts_steps=True,
            debug_rollouts_rewards=True,
            debug_rollouts_dones=True,
            debug_sample_envs=3,
        )
    
    @classmethod
    def create_entropy_debug(cls) -> 'DebugConfig':
        """Create config for debugging entropy issues."""
        return cls(
            debug_env=1,                     # Enable environment debugging
            debug_agent=2,
            debug_model=2,
            debug_env_action_spaces=True,   # Show action space evolution
            debug_agent_rollout_stats=True,  # To see avg number of actions
            debug_model_logits=True,         # To see logit distribution
            debug_model_actions=True,        # To see action selection
            debug_model_distribution=True,   # To see distribution parameters
            debug_model_entropy=True,        # To see entropy computation
            debug_model_action_mask=True,    # To see which actions are masked
            debug_sample_envs=5,             # Show first 5 environments
            debug_max_actions_display=20,    # Show more actions
            debug_sample_frequency=4,        # Only debug every 4th step to reduce output
        )
    
    def __str__(self) -> str:
        """String representation of debug config."""
        enabled = []
        if self.debug_env:
            enabled.append(f"env({self.debug_env})")
        if self.debug_prover:
            enabled.append(f"prover({self.debug_prover})")
        if self.debug_agent:
            enabled.append(f"agent({self.debug_agent})")
        if self.debug_model:
            enabled.append(f"model({self.debug_model})")
        if self.debug_rollouts:
            enabled.append(f"rollouts({self.debug_rollouts})")
        
        if not enabled:
            return "DebugConfig(disabled)"
        return f"DebugConfig({', '.join(enabled)})"
