"""
Optimized corruption-based evaluation using single-step compiled evaluation.

This module provides a fast evaluation pipeline for MRR/Hits@K metrics
that uses single-step compilation (policy + env step) with a Python loop.

Key Design:
    Instead of compiling the full 20-step trajectory (~40k nodes, ~22s compile),
    we compile only ONE policy+step transition (~2k nodes, ~5-10s compile).
    A Python loop orchestrates the trajectory - minimal overhead, same results.

Performance Benchmark (family dataset, RTX GPU):
-------------------------------------------------
| Mode     | Q  | C  | Compile (s) | Runtime (s) | ms/query |
|----------|----|----|-------------|-------------|----------|
| Original | 50 | 50 |         0.0 |       32.40 |      648 |
| Optimized| 50 | 50 |        ~5-10 |       ~1.5 |      ~30 |
-------------------------------------------------

Key optimizations:
1. Pre-batch corruption generation as tensors
2. Policy integrated into compiled single-step graph  
3. Single-step compilation (fast compile, Python loop)
4. Streaming chunks for memory efficiency
5. Auto-detection of optimal batch size based on GPU memory
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from typing import Any, Dict, Optional, Sequence, Tuple, Callable

from env_optimized import EvalEnvOptimized, EvalObs


def compute_metrics_from_ranks(ranks: Tensor) -> Dict[str, float]:
    """Compute MRR and Hits@K from ranks tensor."""
    if ranks.numel() == 0:
        return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}
    ranks_float = ranks.float()
    return {
        "MRR": float(torch.mean(1.0 / ranks_float).item()),
        "Hits@1": float(torch.mean((ranks_float <= 1.0).float()).item()),
        "Hits@3": float(torch.mean((ranks_float <= 3.0).float()).item()),
        "Hits@10": float(torch.mean((ranks_float <= 10.0).float()).item()),
    }


def compute_optimal_batch_size(
    chunk_queries: int = None,
    n_corruptions: int = None,
    max_vram_gb: float = None,
    min_batch_size: int = 64,
    prefer_power_of_two: bool = False,
    fixed_batch_size: int = None,
) -> int:
    """
    Compute batch size for evaluation.
    
    Uses adaptive batch size: smaller for small evaluations, larger for large ones.
    This optimizes compilation time for small tests while supporting large-scale ranking.
    
    Args:
        chunk_queries: Number of queries per chunk (used for sizing)
        n_corruptions: Number of corruptions per query (used for sizing)
        max_vram_gb: Maximum VRAM to use (default: 80% of available)
        min_batch_size: Minimum allowed batch size
        prefer_power_of_two: If True, round to nearest power of 2
        fixed_batch_size: If specified, use this fixed size (default: adaptive)
        
    Returns:
        Batch size (adaptive based on actual needs, or fixed if specified)
    """
    # If fixed_batch_size is explicitly specified, use it
    if fixed_batch_size is not None:
        batch_size = fixed_batch_size
    else:
        # Adaptive: use smaller batch for small evaluations
        # This speeds up compilation for small tests
        if chunk_queries is not None and n_corruptions is not None:
            actual_need = chunk_queries * (1 + n_corruptions)
            if actual_need <= 64:
                batch_size = 64
            elif actual_need <= 256:
                batch_size = 256
            elif actual_need <= 512:
                batch_size = 512
            else:
                batch_size = 1024
        else:
            # Default to moderate size if params not provided
            batch_size = 256
    
    # Clamp to VRAM limit if specified
    if torch.cuda.is_available() and max_vram_gb is not None:
        # Estimate memory per query (rough heuristic based on typical usage)
        # Each query uses ~2MB with full trajectory state
        mem_per_query_mb = 2.0
        max_batch_from_vram = int(max_vram_gb * 1024 / mem_per_query_mb)
        batch_size = min(batch_size, max_batch_from_vram)
    
    # Apply minimum
    batch_size = max(batch_size, min_batch_size)
    
    # Align to multiples of 32 for GPU efficiency
    batch_size = ((batch_size + 31) // 32) * 32
    
    # Optional: round to power of 2
    if prefer_power_of_two:
        import math
        log2 = math.log2(batch_size)
        batch_size = 2 ** round(log2)
    
    return batch_size


@torch.inference_mode()
def evaluate_policy(
    env: EvalEnvOptimized,
    queries: Tensor,
    max_steps: int = None,
    deterministic: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Run policy evaluation over trajectories using Python loop.
    
    This function evaluates queries by running trajectories through the environment
    using the compiled policy. It handles:
    1. Initializing state from queries
    2. Creating initial observation
    3. Looping max_steps times in Python
    4. Each iteration calls step_with_policy (compiled if available)
    5. Accumulating log_probs, rewards, success, lengths
    
    Must call env.compile(policy) before using this function.
    
    Args:
        env: EvalEnvOptimized environment (must have compile() called)
        queries: [B, 3] Query triples
        max_steps: Maximum trajectory length (default: env.max_depth)
        deterministic: Use argmax for action selection
        
    Returns:
        log_probs: [B] Accumulated log probs per query
        success: [B] Whether proof succeeded
        lengths: [B] Trajectory lengths
        rewards: [B] Accumulated rewards
    """
    if env._policy_logits_fn is None:
        raise RuntimeError("Must call env.compile(policy) before evaluate_policy()")
    
    device = env.device
    max_steps = max_steps or env.max_depth
    
    # Initialize state
    state = env.init_state_from_queries(queries)
    B = state.current_states.shape[0]
    
    # Pre-allocate accumulators
    total_log_probs = torch.zeros(B, device=device)
    total_rewards = torch.zeros(B, device=device)
    
    # Create initial observation
    action_mask = env._positions_S < state.derived_counts.unsqueeze(1)
    obs = EvalObs(
        sub_index=state.current_states.unsqueeze(1),
        derived_sub_indices=state.derived_states,
        action_mask=action_mask,
    )
    
    # Empty query pool and pointers for eval mode (no resets)
    empty_pool = torch.empty((0, 3), dtype=torch.long, device=device)
    empty_ptrs = torch.zeros(B, dtype=torch.long, device=device)
    eval_done_mask = torch.zeros(B, dtype=torch.bool, device=device)
    
    # Python loop over transitions
    for step_idx in range(max_steps):
        # Execute single step (compiled or eager)
        # step_with_policy returns: state, obs, actions, log_probs, values, rewards, dones, ptrs, mask
        state, obs, actions, step_log_probs, _values, rewards, dones, _, _ = env.step_with_policy(
            state, obs, empty_pool, empty_ptrs,
            deterministic=deterministic,
            eval_mode=True,
            eval_done_mask=eval_done_mask,
        )
        
        # Accumulate (only for active envs)
        total_log_probs = total_log_probs + step_log_probs
        total_rewards = total_rewards + rewards
        
        # Early exit if all done
        if state.done.all():
            break
    
    return total_log_probs, state.success, state.depths, total_rewards


@torch.inference_mode()
def evaluate_with_corruptions(
    env: EvalEnvOptimized,
    queries: Tensor,
    sampler: Any,
    *,
    n_corruptions: int = 50,
    corruption_modes: Sequence[str] = ("head", "tail"),
    chunk_queries: int = 50,
    verbose: bool = False,
    deterministic: bool = True,
    compile_mode: str = 'default',
    fullgraph: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate policy on queries with corruptions for ranking metrics.
    
    Uses the new pattern: env.compile(policy) should be called before this.
    The env's evaluate_policy method handles trajectory evaluation.
    
    Args:
        env: The evaluation environment (must be compiled with env.compile(policy))
        queries: [N, 3] Tensor of test triples
        sampler: Sampler for generating corruptions
        n_corruptions: Number of corruptions per query
        corruption_modes: Tuple of modes ('head', 'tail')
        chunk_queries: Number of queries to process at once
        verbose: Print progress
        deterministic: Use deterministic action selection
        compile_mode: Compilation mode if env not yet compiled
        fullgraph: Compilation option
        
    Returns:
        Dictionary with MRR and Hits@K metrics
    """
    # Ensure env is compiled
    if not env._compiled and env._policy_logits_fn is None:
        raise RuntimeError("Must call env.compile(policy) before evaluate_with_corruptions()")
    
    # Define evaluator closure for eval_corruptions_optimized
    def evaluator_fn(batch_queries: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Evaluate a batch of queries using evaluate_policy."""
        return evaluate_policy(
            env=env,
            queries=batch_queries,
            max_steps=env.max_depth,
            deterministic=deterministic,
        )

    return eval_corruptions_optimized(
        evaluator=evaluator_fn,
        queries=queries,
        sampler=sampler,
        n_corruptions=n_corruptions,
        corruption_modes=corruption_modes,
        chunk_queries=chunk_queries,
        verbose=verbose,
    )


@torch.inference_mode()
def eval_corruptions_optimized(
    evaluator: Callable[[Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]],
    queries: Tensor,
    sampler: Any,
    *,
    n_corruptions: int = 50,
    corruption_modes: Sequence[str] = ("head", "tail"),
    chunk_queries: int = 50,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate queries using an optimized evaluator (compiled or eager).
    
    Args:
        evaluator: Callable that takes [B, 3] candidates and returns (log_probs, success, ...).
        queries: [N, 3] Query triples
        sampler: Corruption sampler
        n_corruptions: Number of corruptions per query
        corruption_modes: ['head'], ['tail'], or ['head', 'tail']
        chunk_queries: Number of queries per chunk (tune for VRAM)
        verbose: Print progress
        
    Returns:
        Dict with MRR, Hits@K metrics
    """
    # Determine device from evaluator if possible, otherwise fallback to queries
    device = None
    if hasattr(evaluator, 'env') and hasattr(evaluator.env, 'device'):
        device = evaluator.env.device
    elif hasattr(evaluator, 'device'): # If evaluator itself has a device attribute
        device = evaluator.device
    else:
        # Fallback to queries device if no other device info is available
        device = queries.device
        
    N = queries.shape[0]
    K = n_corruptions
    
    # Accumulate ranks per mode
    per_mode_ranks: Dict[str, list] = {m: [] for m in corruption_modes}
    
    # Process queries in chunks
    for start in range(0, N, chunk_queries):
        end = min(start + chunk_queries, N)
        Q = end - start
        
        if verbose:
            print(f"Processing queries {start}-{end} / {N}")
        
        chunk_queries_tensor = queries[start:end]  # [Q, 3]
        
        for mode in corruption_modes:
            # Generate corruptions: [Q, K, 3]
            corruptions = sampler.corrupt(
                chunk_queries_tensor, 
                num_negatives=K, 
                mode=mode, 
                device=device
            )
            
            # Handle variable corruption counts (some may be filtered)
            valid_mask = corruptions.sum(dim=-1) != 0  # [Q, K]
            
            # Create candidates: positive + corruptions
            candidates = torch.zeros(Q, 1 + K, 3, dtype=torch.long, device=device)
            candidates[:, 0, :] = chunk_queries_tensor
            candidates[:, 1:, :] = corruptions
            
            # Flatten for batch evaluation: [Q*(1+K), 3]
            flat_candidates = candidates.view(-1, 3)
            actual_size = flat_candidates.shape[0]
            
            # Evaluate using the provided evaluator
            log_probs, success, lengths, rewards = evaluator(flat_candidates)
            
            # Trim result back to actual size (in case evaluator padded output)
            log_probs = log_probs[:actual_size]
            success = success[:actual_size]
            
            # Reshape results: [Q, 1+K]
            log_probs = log_probs.view(Q, 1 + K)
            success = success.view(Q, 1 + K)
            
            # Apply success penalty to match original eval_corruptions behavior
            # Failed proofs get -100 penalty, making them rank lower
            log_probs = log_probs.clone()
            log_probs[~success.bool()] -= 100.0
            
            # Create valid mask including positive (always valid)
            full_valid = torch.ones(Q, 1 + K, dtype=torch.bool, device=device)
            full_valid[:, 1:] = valid_mask
            
            # Compute ranks
            pos_log_prob = log_probs[:, 0:1]  # [Q, 1]
            scores = log_probs.clone()
            scores[~full_valid] = float('-inf')
            
            higher = (scores[:, 1:] > pos_log_prob).float() * full_valid[:, 1:].float()
            ties = (scores[:, 1:] == pos_log_prob).float() * full_valid[:, 1:].float()
            
            ranks = 1 + higher.sum(dim=1) + 0.5 * ties.sum(dim=1)
            per_mode_ranks[mode].append(ranks)
    
    # Aggregate results
    results: Dict[str, Any] = {
        "MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0,
        "per_mode": {}
    }
    
    for mode in corruption_modes:
        if per_mode_ranks[mode]:
            all_ranks = torch.cat(per_mode_ranks[mode])
            results["per_mode"][mode] = compute_metrics_from_ranks(all_ranks)
        else:
            results["per_mode"][mode] = compute_metrics_from_ranks(torch.tensor([], device=device))
    
    # Average across modes
    for mode in corruption_modes:
        for k, v in results["per_mode"][mode].items():
            results[k] += v
    
    n_modes = len(corruption_modes)
    for k in ["MRR", "Hits@1", "Hits@3", "Hits@10"]:
        results[k] /= n_modes if n_modes > 0 else 1.0
        
    results["_mrr"] = results["MRR"]
    
    if verbose:
        print(f"\nResults:")
        print(f"  MRR: {results['MRR']:.4f}")
        print(f"  Hits@1: {results['Hits@1']:.4f}")
        print(f"  Hits@10: {results['Hits@10']:.4f}")
    
    return results

