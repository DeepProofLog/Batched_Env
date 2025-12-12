"""
Optimized corruption-based evaluation using compiled trajectory evaluation.

This module provides a fully compiled evaluation pipeline for MRR/Hits@K metrics
that processes queries and corruptions in streaming chunks to fit in VRAM.

Key optimizations:
1. Pre-batch corruption generation as tensors
2. Policy integrated into compiled graph  
3. Full trajectory compilation (reduce-overhead + fullgraph)
4. Streaming chunks for memory efficiency
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from typing import Any, Dict, Optional, Sequence, Tuple, Callable

from env_eval_compiled import EvalOnlyEnvCompiled, EvalObs


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


class CompiledEvaluator:
    """
    Cached compiled evaluator with proper CUDA graph caching.
    
    Key optimization: Uses a STATIC input buffer with copy_() to ensure
    tensor storage address never changes. CUDA graphs record tensor addresses,
    so using a new tensor for each call causes graph re-recording.
    
    Performance: ~55ms per 500 candidates (0.11ms per candidate)
    """
    
    def __init__(
        self,
        env: EvalOnlyEnvCompiled,
        policy_logits_fn: Callable[[EvalObs], Tensor],
        batch_size: int,
        max_steps: int = 20,
        deterministic: bool = True,
    ):
        self.env = env
        self.policy_logits_fn = policy_logits_fn
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.deterministic = deterministic
        self._compiled_fn = None
        self._warmed_up = False
        
        # Static input buffer - CUDA graphs record this address
        self._input_buffer = torch.zeros(
            batch_size, 3, dtype=torch.long, device=env.device
        )
    
    def _get_compiled(self) -> callable:
        if self._compiled_fn is None:
            self._compiled_fn = torch.compile(
                lambda q: self.env.evaluate_trajectory_compiled(
                    q, self.policy_logits_fn,
                    max_steps=self.max_steps,
                    deterministic=self.deterministic
                ),
                mode='reduce-overhead',
                fullgraph=True,
            )
        return self._compiled_fn
    
    def warmup(self, sample_queries: Tensor) -> None:
        """Warmup compilation using static buffer."""
        if self._warmed_up:
            return
        
        compiled = self._get_compiled()
        n = sample_queries.shape[0]
        
        # Fill static buffer with sample data
        repeats = (self.batch_size + n - 1) // n
        warmup_data = sample_queries.repeat(repeats, 1)[:self.batch_size]
        self._input_buffer.copy_(warmup_data)
        
        # Warmup runs (3 is enough with static buffer)
        for _ in range(3):
            _ = compiled(self._input_buffer)
        
        torch.cuda.synchronize()
        self._warmed_up = True
    
    def __call__(self, queries: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Evaluate queries using static buffer for CUDA graph stability."""
        if not self._warmed_up:
            self.warmup(queries[:min(10, queries.shape[0])])
        
        compiled = self._get_compiled()
        
        # Prepare input with padding
        actual_size = queries.shape[0]
        if actual_size > self.batch_size:
            raise ValueError(f"Batch size {actual_size} > compiled batch size {self.batch_size}")
        
        # Copy into static buffer (preserves tensor storage address for CUDA graph)
        if actual_size < self.batch_size:
            # Pad with first query
            self._input_buffer[:actual_size].copy_(queries)
            self._input_buffer[actual_size:].copy_(queries[0:1].expand(self.batch_size - actual_size, -1))
        else:
            self._input_buffer.copy_(queries)
        
        # Run compiled function on static buffer
        log_probs, success, lengths, rewards = compiled(self._input_buffer)
        
        # Trim back to actual size
        return log_probs[:actual_size], success[:actual_size], lengths[:actual_size], rewards[:actual_size]


@torch.inference_mode()
def eval_corruptions_compiled(
    env: EvalOnlyEnvCompiled,
    policy_logits_fn: Callable[[EvalObs], Tensor],
    queries: Tensor,
    sampler: Any,
    *,
    n_corruptions: int = 50,
    corruption_modes: Sequence[str] = ("head", "tail"),
    deterministic: bool = True,
    max_steps: int = 20,
    chunk_queries: int = 50,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate queries using compiled trajectory evaluation.
    
    This function evaluates MRR/Hits@K by:
    1. Processing queries in chunks (to fit VRAM)
    2. For each chunk: generate corruptions, evaluate all candidates
    3. Compute rank of positive vs corruptions
    4. Aggregate metrics across all queries
    
    Args:
        env: Compiled evaluation environment
        policy_logits_fn: Function(obs) -> logits [B, S], should be compiled
        queries: [N, 3] Query triples
        sampler: Corruption sampler
        n_corruptions: Number of corruptions per query
        corruption_modes: ['head'], ['tail'], or ['head', 'tail']
        deterministic: Use argmax for action selection
        max_steps: Maximum trajectory length
        chunk_queries: Number of queries per chunk (tune for VRAM)
        verbose: Print progress
        
    Returns:
        Dict with MRR, Hits@K metrics
    """
    device = env.device
    N = queries.shape[0]
    K = n_corruptions
    
    # Compile the trajectory evaluation
    compiled_eval = torch.compile(
        lambda q: env.evaluate_trajectory_compiled(
            q, policy_logits_fn, max_steps=max_steps, deterministic=deterministic
        ),
        mode='reduce-overhead',
        fullgraph=True,
    )
    
    # Fixed batch size for compilation
    fixed_batch = chunk_queries * (1 + K)
    
    # Warmup compilation with fixed batch size
    if N > 0:
        n_warmup = min(chunk_queries, N)
        warmup_base = queries[:n_warmup]
        # Repeat to fill fixed_batch
        repeats = (fixed_batch + n_warmup - 1) // n_warmup
        warmup_q = warmup_base.repeat(repeats, 1)[:fixed_batch]
        for _ in range(3):
            _ = compiled_eval(warmup_q)
        torch.cuda.synchronize()
    
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
            # corruptions: [Q, K, 3] with zeros for invalid
            valid_mask = corruptions.sum(dim=-1) != 0  # [Q, K]
            
            # Create candidates: positive + corruptions
            # candidates: [Q, 1+K, 3]
            candidates = torch.zeros(Q, 1 + K, 3, dtype=torch.long, device=device)
            candidates[:, 0, :] = chunk_queries_tensor
            candidates[:, 1:, :] = corruptions
            
            # Flatten for batch evaluation: [Q*(1+K), 3]
            flat_candidates = candidates.view(-1, 3)
            actual_size = flat_candidates.shape[0]
            
            # Pad to fixed batch size to avoid recompilation
            if actual_size < fixed_batch:
                padding = flat_candidates[0:1].expand(fixed_batch - actual_size, -1)
                flat_candidates = torch.cat([flat_candidates, padding], dim=0)
            
            # Evaluate all candidates
            log_probs, success, lengths, rewards = compiled_eval(flat_candidates)
            
            # Trim back to actual size
            log_probs = log_probs[:actual_size]
            success = success[:actual_size]
            
            # Reshape results: [Q, 1+K]
            log_probs = log_probs.view(Q, 1 + K)
            success = success.view(Q, 1 + K)
            
            # Create valid mask including positive (always valid)
            full_valid = torch.ones(Q, 1 + K, dtype=torch.bool, device=device)
            full_valid[:, 1:] = valid_mask
            
            # Compute ranks for positive (index 0)
            # Rank = 1 + number of corruptions with higher log_prob
            pos_log_prob = log_probs[:, 0:1]  # [Q, 1]
            
            # Score: use log_prob for ranking (higher is better)
            scores = log_probs.clone()
            scores[~full_valid] = float('-inf')  # Invalid get lowest score
            
            # Count how many have strictly higher score than positive
            # Also count ties and add 0.5 for each (expected rank)
            higher = (scores[:, 1:] > pos_log_prob).float() * full_valid[:, 1:].float()
            ties = (scores[:, 1:] == pos_log_prob).float() * full_valid[:, 1:].float()
            
            # Rank = 1 + higher_count + 0.5 * tie_count
            ranks = 1 + higher.sum(dim=1) + 0.5 * ties.sum(dim=1)
            
            per_mode_ranks[mode].append(ranks)
    
    # Aggregate results
    results: Dict[str, Any] = {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}
    per_mode_results: Dict[str, Dict[str, float]] = {}
    
    for mode in corruption_modes:
        if per_mode_ranks[mode]:
            all_ranks = torch.cat(per_mode_ranks[mode])
            per_mode_results[mode] = compute_metrics_from_ranks(all_ranks)
        else:
            per_mode_results[mode] = compute_metrics_from_ranks(torch.tensor([], device=device))
    
    # Average across modes
    for mode in corruption_modes:
        for k, v in per_mode_results[mode].items():
            results[k] += v
    
    n_modes = len(corruption_modes)
    for k in results:
        results[k] /= n_modes if n_modes > 0 else 1.0
    
    results["per_mode"] = per_mode_results
    results["_mrr"] = results["MRR"]
    
    if verbose:
        print(f"\nResults:")
        print(f"  MRR: {results['MRR']:.4f}")
        print(f"  Hits@1: {results['Hits@1']:.4f}")
        print(f"  Hits@10: {results['Hits@10']:.4f}")
    
    return results


@torch.inference_mode()
def eval_corruptions_fast(
    evaluator: CompiledEvaluator,
    queries: Tensor,
    sampler: Any,
    *,
    n_corruptions: int = 50,
    corruption_modes: Sequence[str] = ("head", "tail"),
    chunk_queries: int = 50,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Fast corruption evaluation using pre-warmed CompiledEvaluator.
    
    Usage:
        # One-time setup (26s)
        evaluator = CompiledEvaluator(env, policy_fn, batch_size=1020, max_steps=20)
        evaluator.warmup(queries[:10])
        
        # Fast evaluation (~2s for 100 queries)
        results = eval_corruptions_fast(evaluator, queries, sampler)
    """
    device = evaluator.env.device
    N = queries.shape[0]
    K = n_corruptions
    
    per_mode_ranks: Dict[str, list] = {m: [] for m in corruption_modes}
    
    for start in range(0, N, chunk_queries):
        end = min(start + chunk_queries, N)
        Q = end - start
        
        if verbose:
            print(f"Processing queries {start}-{end} / {N}")
        
        chunk_queries_tensor = queries[start:end]
        
        for mode in corruption_modes:
            corruptions = sampler.corrupt(
                chunk_queries_tensor, 
                num_negatives=K, 
                mode=mode, 
                device=device
            )
            
            valid_mask = corruptions.sum(dim=-1) != 0
            
            candidates = torch.zeros(Q, 1 + K, 3, dtype=torch.long, device=device)
            candidates[:, 0, :] = chunk_queries_tensor
            candidates[:, 1:, :] = corruptions
            
            flat_candidates = candidates.view(-1, 3)
            
            # Use pre-warmed evaluator
            log_probs, success, lengths, rewards = evaluator(flat_candidates)
            
            log_probs = log_probs.view(Q, 1 + K)
            success = success.view(Q, 1 + K)
            
            full_valid = torch.ones(Q, 1 + K, dtype=torch.bool, device=device)
            full_valid[:, 1:] = valid_mask
            
            pos_log_prob = log_probs[:, 0:1]
            scores = log_probs.clone()
            scores[~full_valid] = float('-inf')
            
            higher = (scores[:, 1:] > pos_log_prob).float() * full_valid[:, 1:].float()
            ties = (scores[:, 1:] == pos_log_prob).float() * full_valid[:, 1:].float()
            ranks = 1 + higher.sum(dim=1) + 0.5 * ties.sum(dim=1)
            
            per_mode_ranks[mode].append(ranks)
    
    results: Dict[str, Any] = {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}
    per_mode_results: Dict[str, Dict[str, float]] = {}
    
    for mode in corruption_modes:
        if per_mode_ranks[mode]:
            all_ranks = torch.cat(per_mode_ranks[mode])
            per_mode_results[mode] = compute_metrics_from_ranks(all_ranks)
        else:
            per_mode_results[mode] = compute_metrics_from_ranks(torch.tensor([], device=device))
    
    for mode in corruption_modes:
        for k, v in per_mode_results[mode].items():
            results[k] += v
    
    n_modes = len(corruption_modes)
    for k in results:
        results[k] /= n_modes if n_modes > 0 else 1.0
    
    results["per_mode"] = per_mode_results
    results["_mrr"] = results["MRR"]
    
    if verbose:
        print(f"\nResults: MRR={results['MRR']:.4f}, Hits@10={results['Hits@10']:.4f}")
    
    return results


def create_policy_logits_fn(
    actor: nn.Module,
    deterministic: bool = True,
) -> Callable[[EvalObs], Tensor]:
    """
    Create a policy function that extracts logits from the actor.
    
    This wraps the actor to provide a simple obs -> logits interface
    that can be compiled with the trajectory evaluation.
    
    Args:
        actor: Policy network
        deterministic: Whether evaluation is deterministic
        
    Returns:
        Function that takes EvalObs and returns logits [B, S]
    """
    @torch.no_grad()
    def policy_fn(obs: EvalObs) -> Tensor:
        # Call actor with observation components
        if hasattr(actor, 'forward_eval'):
            # Dedicated eval forward
            out = actor.forward_eval(
                obs.sub_index,
                obs.derived_sub_indices,
                obs.action_mask,
            )
        else:
            # Standard forward expecting TensorDict-like input
            # Create a simple dict for compatibility
            out = actor({
                'sub_index': obs.sub_index,
                'derived_sub_indices': obs.derived_sub_indices,
                'action_mask': obs.action_mask,
            })
        
        # Extract logits
        if isinstance(out, tuple):
            logits = out[0]
        elif hasattr(out, 'get'):
            logits = out.get('logits', out.get('action', None))
            if logits is None:
                logits = out
        else:
            logits = out
        
        # Ensure 2D [B, S]
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        
        return logits
    
    return policy_fn
