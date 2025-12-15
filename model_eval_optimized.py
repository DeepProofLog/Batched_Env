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
    
    Memory model:
    - Each query in a batch requires ~3-4MB for state tensors:
      - derived_states: [K_max=120, M_max=26, 3] × 8 bytes ≈ 75KB per query
      - Multiple intermediate tensors during unification: ~3MB per query
    - For 8GB GPU with ~6GB usable: max ~2000 queries safely, ~1500 conservatively
    
    Args:
        chunk_queries: Number of queries per chunk (used for sizing)
        n_corruptions: Number of corruptions per query (used for sizing)
        max_vram_gb: Maximum VRAM to use (default: detected from GPU)
        min_batch_size: Minimum allowed batch size
        prefer_power_of_two: If True, round to nearest power of 2
        fixed_batch_size: If specified, use this fixed size (default: adaptive)
        
    Returns:
        Batch size (adaptive based on actual needs, or fixed if specified)
    """
    # Detect available VRAM if not specified
    if max_vram_gb is None and torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        # Use ~60% of total memory to leave room for PyTorch overhead
        max_vram_gb = total_mem * 0.6
    elif max_vram_gb is None:
        max_vram_gb = 4.0  # Conservative default for CPU
    
    # If fixed_batch_size is explicitly specified, use it
    if fixed_batch_size is not None:
        batch_size = fixed_batch_size
    else:
        # Adaptive: use smaller batch for small evaluations
        # This speeds up compilation for small tests
        # NOTE: Batch sizes > 512 tend to have much worse per-query performance
        # due to GPU memory bandwidth limits with large derived_states tensors
        if chunk_queries is not None and n_corruptions is not None:
            actual_need = chunk_queries * (1 + n_corruptions)
            if actual_need <= 64:
                batch_size = 64
            elif actual_need <= 256:
                batch_size = 256
            else:
                # Cap at 512 for best throughput (larger sizes have worse per-query time)
                batch_size = 512
        else:
            # Default to moderate size if params not provided
            batch_size = 256
    
    # Clamp to VRAM limit - use more accurate memory estimation
    # Each query uses approximately 3MB during unification
    mem_per_query_mb = 3.0
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
    fixed_batch_size: int = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Run policy evaluation over trajectories using Python loop.
    
    This function evaluates queries by running trajectories through the environment
    using the compiled policy. It handles:
    1. Padding queries to fixed_batch_size if specified (for CUDA graph compatibility)
    2. Chunking large batches to avoid OOM
    3. Running trajectories with compiled step_with_policy
    
    Must call env.compile(policy) before using this function.
    
    Args:
        env: EvalEnvOptimized environment (must have compile() called)
        queries: [B, 3] Query triples
        max_steps: Maximum trajectory length (default: env.max_depth)
        deterministic: Use argmax for action selection
        fixed_batch_size: If set, pad all batches to this size to avoid recompilation
                         with torch.compile mode='reduce-overhead'. This is critical
                         for performance as each unique batch size triggers recompilation.
        
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
    B = queries.shape[0]
    
    # Determine chunk size - use fixed_batch_size if provided
    if fixed_batch_size is not None:
        chunk_size = fixed_batch_size
    else:
        # Adaptive chunk size based on available GPU memory
        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            chunk_size = max(64, min(512, int((total_mem - 2.0) * 1024 / 3)))
        else:
            chunk_size = 256
    
    # If batch fits in one chunk, process directly
    if B <= chunk_size:
        # Pad to fixed_batch_size if specified
        if fixed_batch_size is not None and B < fixed_batch_size:
            padded_queries = torch.zeros(fixed_batch_size, 3, dtype=queries.dtype, device=device)
            padded_queries[:B] = queries
            padded_queries[B:] = queries[-1]  # Fill with last query (valid but results ignored)
            log_probs, success, depths, rewards = _evaluate_policy_batch(
                env, padded_queries, max_steps, deterministic
            )
            return log_probs[:B], success[:B], depths[:B], rewards[:B]
        else:
            return _evaluate_policy_batch(env, queries, max_steps, deterministic)
    
    # Process in chunks to avoid OOM
    all_log_probs = []
    all_success = []
    all_depths = []
    all_rewards = []
    
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        chunk_queries = queries[start:end]
        actual_chunk_size = chunk_queries.shape[0]
        
        # Pad chunk to fixed_batch_size if specified
        if fixed_batch_size is not None and actual_chunk_size < fixed_batch_size:
            padded_chunk = torch.zeros(fixed_batch_size, 3, dtype=queries.dtype, device=device)
            padded_chunk[:actual_chunk_size] = chunk_queries
            padded_chunk[actual_chunk_size:] = chunk_queries[-1]
            log_probs, success, depths, rewards = _evaluate_policy_batch(
                env, padded_chunk, max_steps, deterministic
            )
            # Trim results
            all_log_probs.append(log_probs[:actual_chunk_size])
            all_success.append(success[:actual_chunk_size])
            all_depths.append(depths[:actual_chunk_size])
            all_rewards.append(rewards[:actual_chunk_size])
        else:
            log_probs, success, depths, rewards = _evaluate_policy_batch(
                env, chunk_queries, max_steps, deterministic
            )
            all_log_probs.append(log_probs)
            all_success.append(success)
            all_depths.append(depths)
            all_rewards.append(rewards)
    
    return (
        torch.cat(all_log_probs, dim=0),
        torch.cat(all_success, dim=0),
        torch.cat(all_depths, dim=0),
        torch.cat(all_rewards, dim=0),
    )


def _evaluate_policy_batch(
    env: EvalEnvOptimized,
    queries: Tensor,
    max_steps: int,
    deterministic: bool,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Evaluate a batch of queries (internal, no padding/chunking).
    
    This is the core evaluation loop. Queries should already be the correct
    shape (padded if using fixed_batch_size for CUDA graphs).
    """
    device = env.device
    
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
        state, obs, actions, step_log_probs, _values, rewards, dones, _, _ = env.step_with_policy(
            state, obs, empty_pool, empty_ptrs,
            deterministic=deterministic,
            eval_mode=True,
            eval_done_mask=eval_done_mask,
        )
        
        # Accumulate
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
    fixed_batch_size: int = None,
) -> Dict[str, Any]:
    """
    Evaluate policy on queries with corruptions for ranking metrics (MRR, Hits@K).
    
    For each query, generates corruptions and evaluates all candidates (positive + corruptions)
    to compute ranking metrics.
    
    Args:
        env: The evaluation environment (must be compiled with env.compile(policy))
        queries: [N, 3] Tensor of test triples
        sampler: Sampler for generating corruptions
        n_corruptions: Number of corruptions per query
        corruption_modes: Tuple of modes ('head', 'tail')
        chunk_queries: Number of queries to process at once
        verbose: Print progress
        deterministic: Use deterministic action selection
        fixed_batch_size: If set, pad batches to this size to avoid recompilation
        
    Returns:
        Dictionary with MRR and Hits@K metrics
    """
    if not env._compiled and env._policy_logits_fn is None:
        raise RuntimeError("Must call env.compile(policy) before evaluate_with_corruptions()")
    
    device = env.device
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
            
            # Evaluate using evaluate_policy (handles chunking/padding internally)
            log_probs, success, lengths, rewards = evaluate_policy(
                env=env,
                queries=flat_candidates,
                max_steps=env.max_depth,
                deterministic=deterministic,
                fixed_batch_size=fixed_batch_size,
            )
            
            # Reshape results: [Q, 1+K]
            log_probs = log_probs.view(Q, 1 + K)
            success = success.view(Q, 1 + K)
            
            # Apply success penalty - failed proofs get -100 penalty
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

