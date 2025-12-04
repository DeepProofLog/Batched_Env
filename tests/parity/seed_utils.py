"""
Centralized Seeding Utilities for Parity Tests.

This module provides a unified way to handle seeding across all parity tests,
ensuring reproducibility without the need to scatter seed calls throughout the code.

This module wraps the main seeding utilities from utils.seeding and adds
parity-test-specific helpers like ParityTestSeeder.

Usage:
    from seed_utils import seed_all, SeedContext, ParityTestSeeder
    
    # Simple global seeding
    seed_all(42)
    
    # Context manager for temporary seeding
    with SeedContext(42):
        # Code that needs deterministic behavior
        pass
    
    # Full test setup with ParityTestSeeder
    seeder = ParityTestSeeder(seed=42)
    seeder.seed_for_model_creation()
    # ... create models ...
    seeder.seed_for_rollout_collection()
    # ... collect rollouts ...
"""
import sys
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

# Add project root to path to import utils.seeding
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import from the main seeding module
from utils.seeding import (
    seed_all,
    derive_seed,
    derive_seeds,
    create_local_rng,
    create_numpy_rng,
    create_torch_generator,
    LocalRNG,
    get_env_seed,
    get_worker_seed,
)

# Re-export for convenience
__all__ = [
    'seed_all',
    'derive_seed',
    'derive_seeds',
    'create_local_rng',
    'create_numpy_rng',
    'create_torch_generator',
    'LocalRNG',
    'get_env_seed',
    'get_worker_seed',
    'SeedContext',
    'get_rng_states',
    'set_rng_states',
    'ParityTestSeeder',
    'ParityTestConfig',
]

import random
import numpy as np
import torch


def get_rng_states() -> dict:
    """
    Capture current RNG states for all generators.
    
    Returns:
        Dictionary containing RNG states for random, numpy, and torch
    """
    states = {
        'random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        states['cuda'] = torch.cuda.get_rng_state_all()
    
    return states


def set_rng_states(states: dict) -> None:
    """
    Restore RNG states for all generators.
    
    Args:
        states: Dictionary containing RNG states (from get_rng_states)
    """
    random.setstate(states['random'])
    np.random.set_state(states['numpy'])
    torch.set_rng_state(states['torch'])
    
    if 'cuda' in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states['cuda'])


@contextmanager
def SeedContext(seed: int, restore_after: bool = True):
    """
    Context manager for temporary seeding.
    
    Args:
        seed: The seed to set within the context
        restore_after: If True, restore RNG states after exiting context
        
    Usage:
        with SeedContext(42):
            # Deterministic code here
            pass
        # RNG states restored (if restore_after=True)
    """
    if restore_after:
        saved_states = get_rng_states()
    
    seed_all(seed)
    
    try:
        yield
    finally:
        if restore_after:
            set_rng_states(saved_states)


@dataclass
class ParityTestSeeder:
    """
    Manages seeding for parity tests with predefined phases.
    
    This class provides a structured way to seed different phases of a parity test,
    ensuring that both SB3 and Tensor implementations receive identical random states.
    
    The seeder uses absolute seed values for each phase (not offsets from base seed)
    to maintain backward compatibility with existing tests.
    
    Usage:
        seeder = ParityTestSeeder(seed=42)
        
        # Phase 1: Model creation
        seeder.seed_for_model_creation()
        sb3_model = create_sb3_model(...)
        
        seeder.seed_for_model_creation()  # Reset for tensor model
        tensor_model = create_tensor_model(...)
        
        # Phase 2: Rollout collection
        seeder.seed_for_rollout_collection()
        sb3_rollout = collect_sb3_rollout(...)
        
        seeder.seed_for_rollout_collection()  # Reset for tensor
        tensor_rollout = collect_tensor_rollout(...)
    """
    seed: int = 42
    
    # Absolute seed values for different phases (backward compatible with original tests)
    # These are the actual seed values that were used in the original test code
    _model_creation_seed: int = 42      # Original: seed=42 passed to create_*_ppo
    _rollout_collection_seed: int = 123  # Original: torch.manual_seed(123) before collect
    _training_seed: int = 123            # Original: same seed used before training
    _evaluation_seed: int = 12345        # Original: eval_seed = 12345
    _negative_sampling_seed: int = 99999 # Original: DEBUG_SEED = 99999
    
    def seed_for_model_creation(self) -> None:
        """Seed RNGs for model/embedder creation phase (torch only for backward compat)."""
        # Only seed torch - this matches the original behavior where only
        # torch.manual_seed(seed) was called inside create_*_ppo functions
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
    
    def seed_for_rollout_collection(self) -> None:
        """Seed RNGs for rollout collection phase (torch only for backward compat)."""
        # Only seed torch - this matches the original torch.manual_seed(123) calls
        torch.manual_seed(self._rollout_collection_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self._rollout_collection_seed)
            torch.cuda.manual_seed_all(self._rollout_collection_seed)
    
    def seed_for_training(self) -> None:
        """Seed RNGs for training phase (numpy + torch for shuffling)."""
        # Seed both numpy and torch - original code used both for training
        np.random.seed(self._training_seed)
        torch.manual_seed(self._training_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self._training_seed)
            torch.cuda.manual_seed_all(self._training_seed)
    
    def seed_for_evaluation(self) -> None:
        """Seed RNGs for evaluation phase."""
        seed_all(self._evaluation_seed, deterministic_cudnn=False)
    
    def seed_for_negative_sampling(self) -> None:
        """Seed RNGs for negative sampling phase."""
        seed_all(self._negative_sampling_seed, deterministic_cudnn=False)
    
    def seed_custom(self, seed_value: int) -> None:
        """Seed RNGs with a specific absolute seed value."""
        seed_all(seed_value, deterministic_cudnn=False)
    
    def get_seed(self, phase: str = 'base') -> int:
        """
        Get the seed value for a specific phase.
        
        Args:
            phase: One of 'base', 'model', 'rollout', 'training', 'evaluation', 'negative_sampling'
            
        Returns:
            The seed value for the specified phase
        """
        seeds = {
            'base': self.seed,
            'model': self.seed,  # Model uses base seed
            'rollout': self._rollout_collection_seed,
            'training': self._training_seed,
            'evaluation': self._evaluation_seed,
            'negative_sampling': self._negative_sampling_seed,
        }
        return seeds.get(phase, self.seed)


class ParityTestConfig:
    """
    Configuration class for parity tests with built-in seeding support.
    
    This class combines test parameters with seeding management, providing
    a single source of truth for test configuration.
    
    Usage:
        config = ParityTestConfig(
            seed=42,
            dataset="countries_s3",
            n_envs=4,
            n_steps=20,
        )
        
        # Access seeder
        config.seeder.seed_for_model_creation()
        
        # Access parameters
        n_envs = config.n_envs
    """
    
    def __init__(
        self,
        seed: int = 42,
        dataset: str = "countries_s3",
        n_envs: int = 4,
        n_steps: int = 20,
        n_epochs: int = 1,
        padding_atoms: int = 6,
        padding_states: int = 100,
        n_vars_for_embedder: int = 1000,
        device: Optional[str] = None,
    ):
        self.seed = seed
        self.dataset = dataset
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.padding_atoms = padding_atoms
        self.padding_states = padding_states
        self.n_vars_for_embedder = n_vars_for_embedder
        
        # Set device
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Create seeder
        self.seeder = ParityTestSeeder(seed=seed)
    
    def __repr__(self) -> str:
        return (
            f"ParityTestConfig(seed={self.seed}, dataset='{self.dataset}', "
            f"n_envs={self.n_envs}, n_steps={self.n_steps}, n_epochs={self.n_epochs})"
        )
