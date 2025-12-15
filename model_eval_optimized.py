"""
Optimized corruption-based evaluation using single-step compiled evaluation.

This module re-exports evaluation functions from ppo_optimized for backward
compatibility. New code should import directly from ppo_optimized.

The primary evaluation interface is now PPOOptimized.evaluate_with_corruptions().

Usage:
    from ppo_optimized import PPOOptimized, compute_optimal_batch_size
    
    ppo = PPOOptimized(policy, env, fixed_batch_size=512)
    env.compile(policy, mode='reduce-overhead', fullgraph=True)
    
    # Warmup (triggers compilation)
    warmup_queries = queries[:1].expand(512, -1)
    _ = ppo.evaluate_policy(warmup_queries)
    
    # Evaluate with corruptions
    results = ppo.evaluate_with_corruptions(
        queries=test_queries,
        sampler=sampler,
        n_corruptions=100,
    )
"""

from __future__ import annotations

# Re-export from ppo_optimized for backward compatibility
from ppo_optimized import (
    compute_metrics_from_ranks,
    compute_optimal_batch_size,
)

__all__ = [
    'compute_metrics_from_ranks',
    'compute_optimal_batch_size',
]
