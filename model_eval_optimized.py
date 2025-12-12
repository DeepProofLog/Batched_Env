"""
Optimized corruption-based evaluation using compiled trajectory evaluation.

This module provides a fully compiled evaluation pipeline for MRR/Hits@K metrics
that processes queries and corruptions in streaming chunks to fit in VRAM.

Key optimizations:
1. Pre-batch corruption generation as tensors
2. Policy integrated into compiled graph  
3. Full trajectory compilation (reduce-overhead + fullgraph)
4. Streaming chunks for memory efficiency
5. Auto-detection of optimal batch size based on GPU memory
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


def compute_optimal_batch_size(
    chunk_queries: int,
    n_corruptions: int,
    max_vram_gb: float = None,
    min_batch_size: int = 64,
    prefer_power_of_two: bool = False,
) -> int:
    """
    Compute optimal batch size for evaluation.
    
    The batch size is determined by:
    1. chunk_queries × (1 + n_corruptions) formula
    2. Available GPU memory
    3. Alignment preferences (power of 2, multiples of 32)
    
    Args:
        chunk_queries: Number of queries per chunk
        n_corruptions: Number of corruptions per query
        max_vram_gb: Maximum VRAM to use (default: 80% of available)
        min_batch_size: Minimum allowed batch size
        prefer_power_of_two: If True, round to nearest power of 2
        
    Returns:
        Optimal batch size
    """
    # Basic calculation
    raw_batch_size = chunk_queries * (1 + n_corruptions)
    
    # Get available VRAM
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if max_vram_gb is None:
            max_vram_gb = total_vram * 0.8  # Use 80% of available
        
        # Estimate memory per query (rough heuristic based on typical usage)
        # Each query uses ~2MB with full trajectory state
        mem_per_query_mb = 2.0
        max_batch_from_vram = int(max_vram_gb * 1024 / mem_per_query_mb)
        
        # Clamp to VRAM limit
        batch_size = min(raw_batch_size, max_batch_from_vram)
    else:
        batch_size = raw_batch_size
    
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
        compile_mode: str = 'default',
        fullgraph: bool = False,
    ):
        """
        Args:
            env: EvalOnlyEnvCompiled environment
            policy_logits_fn: Function that returns logits [B, S]
            batch_size: Fixed batch size for compilation
            max_steps: Maximum trajectory length
            deterministic: Use argmax for action selection
            compile_mode: torch.compile mode - 'default' (fast compile, good runtime)
                         or 'reduce-overhead' (slow compile, better runtime)
                         or 'max-autotune' (very slow compile, best runtime)
            fullgraph: If True, use fullgraph=True (slow compile, best runtime).
                      If False, allow graph breaks (fast compile, good runtime).
                      Recommended: False for testing, True for production.
        """
        self.env = env
        self.policy_logits_fn = policy_logits_fn
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.deterministic = deterministic
        self.compile_mode = compile_mode
        self.fullgraph = fullgraph
        self._compiled_fn = None
        self._warmed_up = False
        self.warmup_time_s: Optional[float] = None
        
        # Static input buffer - CUDA graphs record this address
        self._input_buffer = torch.zeros(
            batch_size, 3, dtype=torch.long, device=env.device
        )
    
    @classmethod
    def create_with_optimal_batch_size(
        cls,
        env: EvalOnlyEnvCompiled,
        policy_logits_fn: Callable[[EvalObs], Tensor],
        chunk_queries: int = 10,
        n_corruptions: int = 50,
        max_steps: int = 20,
        deterministic: bool = True,
        max_vram_gb: float = None,
        compile_mode: str = 'default',
        fullgraph: bool = False,
    ) -> 'CompiledEvaluator':
        """
        Create CompiledEvaluator with automatically computed optimal batch size.
        
        Args:
            env: EvalOnlyEnvCompiled environment
            policy_logits_fn: Policy function that returns logits
            chunk_queries: Number of queries per evaluation chunk
            n_corruptions: Number of corruptions per query
            max_steps: Maximum trajectory length
            deterministic: Use argmax for action selection
            max_vram_gb: Maximum VRAM to use (default: auto-detect)
            
        Returns:
            CompiledEvaluator with optimal batch size
        """
        batch_size = compute_optimal_batch_size(
            chunk_queries=chunk_queries,
            n_corruptions=n_corruptions,
            max_vram_gb=max_vram_gb,
        )
        
        return cls(
            env=env,
            policy_logits_fn=policy_logits_fn,
            batch_size=batch_size,
            max_steps=max_steps,
            deterministic=deterministic,
            compile_mode=compile_mode,
            fullgraph=fullgraph,
        )
    
    def _get_compiled(self) -> callable:
        if self._compiled_fn is None:
            # Check if compilation is disabled
            if self.compile_mode is None or self.compile_mode is False:
                # Eager mode
                self._compiled_fn = lambda q: self.env.evaluate_trajectory_compiled(
                    q, self.policy_logits_fn,
                    max_steps=self.max_steps,
                    deterministic=self.deterministic
                )
                return self._compiled_fn

            import os
            import torch._inductor.config as inductor_config
            
            # Set TF32 precision for better performance
            torch.set_float32_matmul_precision('high')
            
            # Speed up compilation (Option 1: Disable expensive optimizations)
            os.environ['TORCHINDUCTOR_MAX_AUTOTUNE'] = '0'
            os.environ['TORCHINDUCTOR_COORDINATE_DESCENT_TUNING'] = '0'
            os.environ['TORCHINDUCTOR_FREEZING'] = '0'
            
            # Speed up compilation (Option 5: Parallel compilation)
            inductor_config.compile_threads = 4
            
            print(f"Compiling with mode='{self.compile_mode}', fullgraph={self.fullgraph}, max_steps={self.max_steps}...")
            
            self._compiled_fn = torch.compile(
                lambda q: self.env.evaluate_trajectory_compiled(
                    q, self.policy_logits_fn,
                    max_steps=self.max_steps,
                    deterministic=self.deterministic
                ),
                mode=self.compile_mode,
                fullgraph=self.fullgraph,
                dynamic=False,
            )
        return self._compiled_fn
    
    def warmup(
        self, 
        sample_queries: Tensor, 
        n_warmup_runs: int = 1,
    ) -> None:
        """
        Warmup compilation using static buffer.
        
        This method triggers JIT compilation and CUDA graph recording.
        One warmup call is sufficient since all tensors have fixed shapes
        and there is only one computation graph with fullgraph=True.
        
        Args:
            sample_queries: [N, 3] Sample queries for warmup
            n_warmup_runs: Number of warmup iterations (default: 1)
        """
        if self._warmed_up:
            return
        if torch.cuda.is_available():
            start_t = torch.cuda.Event(enable_timing=True)
            end_t = torch.cuda.Event(enable_timing=True)
            start_t.record()
            cpu_start = None
        else:
            start_t = None
            end_t = None
            import time as _time
            cpu_start = _time.perf_counter()
        
        compiled = self._get_compiled()
        n = sample_queries.shape[0]
        
        # Fill static buffer with sample data
        repeats = (self.batch_size + n - 1) // n
        warmup_data = sample_queries.repeat(repeats, 1)[:self.batch_size]
        self._input_buffer.copy_(warmup_data)
        
        # Warmup runs - triggers compilation / CUDA graph capture
        for _ in range(n_warmup_runs):
            _ = compiled(self._input_buffer)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if end_t is not None and start_t is not None:
            end_t.record()
            torch.cuda.synchronize()
            self.warmup_time_s = float(start_t.elapsed_time(end_t) / 1000.0)
        else:
            import time as _time
            self.warmup_time_s = float(_time.perf_counter() - cpu_start)
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


@torch.no_grad()
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
        evaluator: Calable that takes [B, 3] candidates and returns (log_probs, success, ...).
                  Usually an instance of CompiledEvaluator.
        queries: [N, 3] Query triples
        sampler: Corruption sampler
        n_corruptions: Number of corruptions per query
        corruption_modes: ['head'], ['tail'], or ['head', 'tail']
        chunk_queries: Number of queries per chunk (tune for VRAM)
        verbose: Print progress
        
    Returns:
        Dict with MRR, Hits@K metrics
    """
    if hasattr(evaluator, 'env'):
        device = evaluator.env.device
    elif hasattr(evaluator, 'device'):
        device = evaluator.device
    else:
        # Fallback
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
            # Note: Evaluator is responsible for any padding if it requires fixed shapes
            log_probs, success, lengths, rewards = evaluator(flat_candidates)
            
            # Trim result back to actual size (in case evaluator padded output)
            log_probs = log_probs[:actual_size]
            
            # Reshape results: [Q, 1+K]
            log_probs = log_probs.view(Q, 1 + K)
            
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



def create_policy_logits_fn(
    actor: nn.Module,
    deterministic: bool = True,
) -> Callable[[EvalObs], Tensor]:
    """
    Create a policy function that extracts logits from the actor.
    
    This wraps the actor to provide a simple obs -> logits interface
    that can be compiled with the trajectory evaluation.
    
    Args:
        actor: Policy network (ActorCriticPolicy)
        deterministic: Whether evaluation is deterministic
        
    Returns:
        Function that takes EvalObs and returns logits [B, S]
    """
    @torch.no_grad()
    def policy_fn(obs: EvalObs) -> Tensor:
        # Build observation dict that the actor expects
        obs_dict = {
            'sub_index': obs.sub_index,
            'derived_sub_indices': obs.derived_sub_indices,
            'action_mask': obs.action_mask,
        }
        
        # Extract features and get raw logits from mlp_extractor
        # This bypasses the action distribution sampling in forward()
        if hasattr(actor, 'extract_features') and hasattr(actor, 'mlp_extractor'):
            # ActorCriticPolicy path - get raw logits directly
            features = actor.extract_features(obs_dict)
            if actor.share_features_extractor:
                logits, _ = actor.mlp_extractor(features)  # (logits, values)
            else:
                obs_emb, act_emb, act_mask = features[0]
                logits = actor.mlp_extractor.forward_actor((obs_emb, act_emb, act_mask))
        elif hasattr(actor, 'forward_eval'):
            # Dedicated eval forward path
            logits = actor.forward_eval(
                obs.sub_index,
                obs.derived_sub_indices,
                obs.action_mask,
            )
        else:
            # Fallback - standard forward, extract logits from tuple
            out = actor(obs_dict)
            if isinstance(out, tuple):
                # forward() returns (actions, values, log_probs) - actions have shape [B]
                # We need logits which have shape [B, S]
                # This path is incorrect for ActorCriticPolicy, but kept for other models
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
