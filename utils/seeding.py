"""
Unified Seeding Utilities for Reproducibility.

This module provides a centralized approach to seeding that should be used
throughout the project. The key principles are:

1. **Global seeding**: Call `seed_all(seed)` ONCE at the start of your script.
   This sets all global RNG states (torch, numpy, random, CUDA).

2. **Local RNGs for diversity**: When you need different random sequences per
   environment/batch (e.g., query sampling), use `create_local_rng(seed)` to
   create independent RNG instances that don't affect global state.

3. **Derived seeds**: Use `derive_seed(base_seed, index)` to create reproducible
   but different seeds for multiple workers/envs.

Usage Examples:

    # At script start - set global seed once
    from utils.seeding import seed_all
    seed_all(42)
    
    # For per-environment diversity (e.g., in BatchedEnv or VecEnv)
    from utils.seeding import create_local_rng, derive_seed
    
    class MyEnv:
        def __init__(self, env_id: int, base_seed: int = 42):
            # Each env gets a different but reproducible seed
            self.rng = create_local_rng(derive_seed(base_seed, env_id))
        
        def sample_query(self):
            # Use local RNG - doesn't affect global state
            idx = self.rng.randint(0, len(self.queries) - 1)
            return self.queries[idx]
    
    # For torch operations that need local seeds
    from utils.seeding import create_torch_generator
    
    gen = create_torch_generator(seed=123, device='cuda')
    noise = torch.randn(10, generator=gen)  # Uses local generator
"""
import random
from typing import Optional, Tuple, Union

import numpy as np
import torch


def seed_all(
    seed: int,
    deterministic: bool = True,
    deterministic_cudnn: bool = True,
    warn: bool = False
) -> None:
    """
    Set seeds for ALL random number generators globally.
    
    This is the CENTRAL seeding function - call this ONCE at the start
    of your script/training run (typically from runner.py).
    
    Args:
        seed: The seed value to use
        deterministic: If True, enables strict deterministic operations.
            - Sets torch.use_deterministic_algorithms(True)
            - May impact performance but ensures exact reproducibility
            - Set to False for production (faster, but non-reproducible)
        deterministic_cudnn: If True AND deterministic=True, sets CUDNN
            to deterministic mode. Ignored if deterministic=False.
        warn: If True, print a warning about deterministic mode
    
    Example:
        # For reproducible parity testing:
        seed_all(42, deterministic=True)
        
        # For production (faster):
        seed_all(42, deterministic=False)
    """
    import os
    
    # Core seeding - always done
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Deterministic mode - optional for performance
    if deterministic:
        # Set CUBLAS_WORKSPACE_CONFIG for deterministic CUDA matmul operations
        if torch.cuda.is_available():
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        
        # Enable deterministic algorithms for exact reproducibility
        torch.use_deterministic_algorithms(True, warn_only=False)
        print('ensuring determinism in the torch algorithm')
        
        if deterministic_cudnn and torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if warn:
                print(
                    "Warning: This setting is not reproducible when creating "
                    "2 models from scratch, but it is when loading pretrained models."
                )
    else:
        # Non-deterministic mode - faster for production
        torch.use_deterministic_algorithms(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # Enable cuDNN autotuner



def derive_seed(base_seed: int, index: int, multiplier: int = 1000003) -> int:
    """
    Derive a new seed from a base seed and an index.
    
    Useful for creating reproducible but different seeds for multiple
    workers, environments, or batch elements.
    
    Args:
        base_seed: The base/master seed
        index: An index (e.g., worker ID, env ID, batch index)
        multiplier: Prime multiplier for better distribution (default: 1000003)
    
    Returns:
        A derived seed that is deterministic given the inputs
        
    Example:
        base_seed = 42
        env_seeds = [derive_seed(base_seed, i) for i in range(num_envs)]
    """
    # Use a simple but effective hash combining technique
    # The multiplier is a prime to reduce collisions
    return (base_seed + index * multiplier) % (2**31 - 1)


def derive_seeds(base_seed: int, count: int) -> Tuple[int, ...]:
    """
    Derive multiple seeds from a base seed.
    
    Args:
        base_seed: The base/master seed
        count: Number of seeds to derive
    
    Returns:
        Tuple of derived seeds
        
    Example:
        env_seed, eval_seed, sampler_seed = derive_seeds(42, 3)
    """
    return tuple(derive_seed(base_seed, i) for i in range(count))


def create_local_rng(seed: Optional[int] = None) -> random.Random:
    """
    Create a local Python random.Random instance.
    
    This is the PREFERRED way to handle randomness when you need:
    - Per-environment different random sequences
    - Random sampling that shouldn't affect global reproducibility
    
    The returned RNG is completely independent of the global random state.
    
    Args:
        seed: Seed for the local RNG. If None, uses a random seed.
    
    Returns:
        A random.Random instance with its own state
        
    Example:
        # In an environment class
        def __init__(self, env_id, base_seed=42):
            self.rng = create_local_rng(derive_seed(base_seed, env_id))
        
        def sample_query(self):
            return self.rng.choice(self.queries)
    """
    rng = random.Random()
    if seed is not None:
        rng.seed(seed)
    return rng


def create_numpy_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Create a local NumPy random Generator instance.
    
    Uses the modern numpy.random.Generator API (recommended over legacy RandomState).
    
    Args:
        seed: Seed for the local RNG. If None, uses a random seed.
    
    Returns:
        A numpy.random.Generator instance with its own state
        
    Example:
        rng = create_numpy_rng(42)
        samples = rng.choice(array, size=10, replace=False)
    """
    return np.random.default_rng(seed)


def create_torch_generator(
    seed: Optional[int] = None,
    device: Optional[Union[str, object]] = None
) -> torch.Generator:
    """
    Create a local PyTorch Generator instance.
    
    This allows torch operations to use a local RNG without affecting
    the global torch RNG state.
    
    Args:
        seed: Seed for the generator. If None, uses a random seed.
        device: Device for the generator ('cpu', 'cuda', etc.)
    
    Returns:
        A torch.Generator instance
        
    Example:
        gen = create_torch_generator(42, device='cuda')
        noise = torch.randn(10, 10, generator=gen, device='cuda')
        indices = torch.randint(0, 100, (10,), generator=gen)
    """
    if device is None:
        device = 'cpu'
    
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    return gen


class LocalRNG:
    """
    A container for local RNGs (Python, NumPy, and PyTorch).
    
    Provides a unified interface when you need all three types of RNGs
    with the same base seed but independent states.
    
    Example:
        rng = LocalRNG(seed=42, device='cuda')
        
        # Python random operations
        choice = rng.py.choice([1, 2, 3])
        
        # NumPy operations
        array = rng.np.random(10)
        
        # PyTorch operations
        tensor = torch.randn(10, generator=rng.torch)
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        """
        Initialize local RNGs.
        
        Args:
            seed: Base seed. Each RNG type gets a derived seed.
            device: Device for the PyTorch generator.
        """
        self.base_seed = seed
        
        if seed is not None:
            # Derive different seeds for each RNG type to avoid correlation
            py_seed, np_seed, torch_seed = derive_seeds(seed, 3)
        else:
            py_seed = np_seed = torch_seed = None
        
        self.py = create_local_rng(py_seed)
        self.np = create_numpy_rng(np_seed)
        self.torch = create_torch_generator(torch_seed, device)
    
    def reset(self, seed: Optional[int] = None):
        """Reset all RNGs to their initial or a new seed."""
        if seed is None:
            seed = self.base_seed
        
        if seed is not None:
            py_seed, np_seed, torch_seed = derive_seeds(seed, 3)
            self.py.seed(py_seed)
            self.np = create_numpy_rng(np_seed)
            self.torch.manual_seed(torch_seed)
        
        self.base_seed = seed


# ============================================================
# Convenience functions for common patterns
# ============================================================

def get_env_seed(base_seed: int, env_index: int) -> int:
    """
    Get a seed for a specific environment index.
    
    Convenience wrapper around derive_seed with a descriptive name.
    
    Args:
        base_seed: The master seed for the experiment
        env_index: The index of this environment (0, 1, 2, ...)
    
    Returns:
        A reproducible seed for this specific environment
    """
    return derive_seed(base_seed, env_index)


def get_worker_seed(base_seed: int, worker_id: int, offset: int = 10000) -> int:
    """
    Get a seed for a specific worker in multi-process training.
    
    Uses a larger offset to ensure worker seeds don't overlap with env seeds.
    
    Args:
        base_seed: The master seed for the experiment
        worker_id: The worker ID (0, 1, 2, ...)
        offset: Offset to separate worker seeds from other derived seeds
    
    Returns:
        A reproducible seed for this specific worker
    """
    return derive_seed(base_seed, worker_id + offset)


# ============================================================
# Legacy compatibility - use the functions above in new code
# ============================================================

def _set_seeds(seed: int) -> None:
    """
    Legacy function for backward compatibility.
    
    Prefer using seed_all() in new code.
    """
    seed_all(seed, deterministic_cudnn=False)
