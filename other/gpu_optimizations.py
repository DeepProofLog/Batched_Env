"""
GPU Optimization Implementations for Unification

This module provides optimized GPU implementations of critical unification operations
based on profiling insights.

Key Optimizations:
1. GPU-native hashing without CPU transfers
2. Fused kernel operations
3. Memory-coalesced access patterns
4. Batched operations
"""

import torch
from typing import List, Tuple, Set, Optional
from unification_cpu import _tensor_hash
from unification import gpu_batch_unique, gpu_parallel_hash



def fused_canonicalize_and_hash(
    states: List[torch.Tensor],
    var_threshold: int,
    padding_idx: int,
    next_var_idx: int
) -> Tuple[List[torch.Tensor], List[int], int]:
    """
    Fused operation: canonicalize and compute hash in one pass.
    
    This avoids redundant variable scanning by doing both operations together.
    
    Args:
        states: List of state tensors [num_atoms, arity+1]
        var_threshold: minimum value for variables
        padding_idx: padding index
        next_var_idx: next available variable index
        
    Returns:
        canonical_states: List of canonicalized states
        hashes: List of hash values
        updated_next_var_idx: updated variable index
    """
    canonical_states = []
    hashes = []
    current_var_idx = next_var_idx
    
    for state in states:
        if state.numel() == 0:
            canonical_states.append(state)
            hashes.append(0)
            continue
        
        # Remove padding
        valid_mask = state[:, 0] != padding_idx
        if not valid_mask.any():
            canonical_states.append(state[:0])
            hashes.append(0)
            continue
        
        state_clean = state[valid_mask]
        
        # Find variables and create mapping in one pass
        var_map = {}
        canonical = state_clean.clone()
        
        # Scan through state, building mapping and canonical form simultaneously
        for i in range(state_clean.shape[0]):
            for j in range(state_clean.shape[1]):
                val = state_clean[i, j].item()
                
                if val >= var_threshold:
                    if val not in var_map:
                        var_map[val] = current_var_idx
                        current_var_idx += 1
                    canonical[i, j] = var_map[val]
        
        # Compute hash while canonical form is still in cache
        hash_val = _tensor_hash(canonical)
        
        canonical_states.append(canonical)
        hashes.append(hash_val)
    
    return canonical_states, hashes, current_var_idx


def coalesced_state_access(
    states: torch.Tensor,
    indices: torch.Tensor
) -> torch.Tensor:
    """
    Optimize memory access patterns for better GPU coalescing.
    
    Args:
        states: [B, max_atoms, arity+1] states in batch
        indices: [N] indices to extract
        
    Returns:
        extracted: [N, max_atoms, arity+1] extracted states
    """
    # Ensure tensors are contiguous for better memory access
    states = states.contiguous()
    
    # Use advanced indexing (PyTorch optimizes this for GPU)
    extracted = states[indices]
    
    return extracted.contiguous()


def batched_mask_operations(
    queries: torch.Tensor,
    facts: torch.Tensor,
    padding_idx: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused masking operations for predicate matching.
    
    Combines multiple mask generations into fewer kernels.
    
    Args:
        queries: [B, arity+1] query atoms
        facts: [F, arity+1] fact atoms
        padding_idx: padding index
        
    Returns:
        match_mask: [B, F] which facts match which queries
        valid_queries: [B] which queries are valid (non-padding)
    """
    device = queries.device
    B = queries.shape[0]
    F = facts.shape[0]
    
    # Fused validity check and predicate extraction
    valid_queries = queries[:, 0] != padding_idx
    query_preds = queries[:, 0]
    fact_preds = facts[:, 0]
    
    # Broadcast comparison (single kernel)
    match_mask = (query_preds.unsqueeze(1) == fact_preds.unsqueeze(0)) & valid_queries.unsqueeze(1)
    
    return match_mask, valid_queries


class GPUOptimizedUnification:
    """
    GPU-optimized unification using all optimization techniques.
    """
    
    def __init__(self, index_manager, use_torch_compile: bool = True):
        self.index_manager = index_manager
        self.use_torch_compile = use_torch_compile
        
        # Compile critical functions if enabled
        if use_torch_compile and hasattr(torch, 'compile'):
            self.gpu_parallel_hash = torch.compile(gpu_parallel_hash)
            self.batched_mask_operations = torch.compile(batched_mask_operations)
        else:
            self.gpu_parallel_hash = gpu_parallel_hash
            self.batched_mask_operations = batched_mask_operations
    
    def deduplicate_gpu(
        self,
        states: torch.Tensor,
        var_threshold: int,
        padding_idx: int
    ) -> torch.Tensor:
        """
        GPU-accelerated deduplication with minimal CPU transfers.
        
        Args:
            states: [B, max_atoms, arity+1] states to deduplicate
            var_threshold: variable threshold
            padding_idx: padding index
            
        Returns:
            unique_states: deduplicated states
        """
        if states.shape[0] == 0:
            return states
        
        # Compute hashes on GPU
        hashes = self.gpu_parallel_hash(states, var_threshold, padding_idx)
        
        # Deduplicate on GPU
        unique_states, _ = gpu_batch_unique(states, hashes, return_inverse=False)
        
        return unique_states
    
    def unify_batch_optimized(
        self,
        queries: torch.Tensor,
        facts: torch.Tensor,
        remaining_goals: torch.Tensor,
        padding_idx: int
    ) -> List[torch.Tensor]:
        """
        Optimized batch unification with fused operations.
        
        Args:
            queries: [B, arity+1] query atoms
            facts: [F, arity+1] available facts
            remaining_goals: [B, max_remaining, arity+1] remaining goals
            padding_idx: padding index
            
        Returns:
            List of derived states for each query
        """
        # Fused masking (single kernel launch)
        match_mask, valid_queries = self.batched_mask_operations(
            queries, facts, padding_idx
        )
        
        # Process matches (vectorized where possible)
        derived_states = []
        
        for b in range(queries.shape[0]):
            if not valid_queries[b]:
                derived_states.append([])
                continue
            
            # Find matching facts
            matching_facts = facts[match_mask[b]]
            
            if matching_facts.shape[0] == 0:
                derived_states.append([])
                continue
            
            # For each match, create derived state
            # (This part still needs per-query processing, but matching is optimized)
            query_states = []
            for fact in matching_facts:
                # Combine with remaining goals
                # (Simplified - full unification logic would go here)
                query_states.append(remaining_goals[b])
            
            derived_states.append(query_states)
        
        return derived_states


def benchmark_optimizations(
    states_tensor: torch.Tensor,
    var_threshold: int,
    padding_idx: int,
    num_iterations: int = 100
):
    """
    Benchmark different optimization techniques.
    
    Args:
        states_tensor: [B, max_atoms, arity+1] test states
        var_threshold: variable threshold
        padding_idx: padding index
        num_iterations: number of iterations for timing
    """
    device = states_tensor.device
    B = states_tensor.shape[0]
    
    print(f"\nBenchmarking GPU Optimizations")
    print(f"Batch Size: {B}")
    print(f"Device: {device}")
    print("="*60)
    
    # Warmup
    for _ in range(5):
        _ = gpu_parallel_hash(states_tensor, var_threshold, padding_idx)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Test 1: Parallel hashing
    start = time.time()
    for _ in range(num_iterations):
        hashes = gpu_parallel_hash(states_tensor, var_threshold, padding_idx)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_parallel_hash = (time.time() - start) / num_iterations
    
    # Test 2: Batch unique
    hashes = gpu_parallel_hash(states_tensor, var_threshold, padding_idx)
    start = time.time()
    for _ in range(num_iterations):
        unique, _ = gpu_batch_unique(states_tensor, hashes, return_inverse=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_batch_unique = (time.time() - start) / num_iterations
    
    # Test 3: Combined operation
    start = time.time()
    for _ in range(num_iterations):
        hashes = gpu_parallel_hash(states_tensor, var_threshold, padding_idx)
        unique, _ = gpu_batch_unique(states_tensor, hashes, return_inverse=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_combined = (time.time() - start) / num_iterations
    
    # Results
    print(f"\nResults (average over {num_iterations} iterations):")
    print(f"  GPU Parallel Hash:  {time_parallel_hash*1000:.4f}ms")
    print(f"  GPU Batch Unique:   {time_batch_unique*1000:.4f}ms")
    print(f"  Combined Operation: {time_combined*1000:.4f}ms")
    print(f"\nThroughput:")
    print(f"  States/sec (hash):    {B/time_parallel_hash:.2f}")
    print(f"  States/sec (unique):  {B/time_batch_unique:.2f}")
    print(f"  States/sec (combined): {B/time_combined:.2f}")


if __name__ == "__main__":
    import time
    
    # Test with synthetic data
    print("GPU Optimizations Module")
    print("Testing with synthetic data...\n")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        batch_sizes = [32, 64, 128, 256]
        
        for batch_size in batch_sizes:
            # Create test states
            max_atoms = 5
            arity = 2
            var_threshold = 1000
            padding_idx = 0
            
            states = torch.randint(
                0, 100,
                (batch_size, max_atoms, arity + 1),
                device=device,
                dtype=torch.long
            )
            
            # Add some variables
            states[:, :, 1:] = torch.where(
                torch.rand(batch_size, max_atoms, arity, device=device) > 0.7,
                torch.randint(var_threshold, var_threshold + 50, 
                            (batch_size, max_atoms, arity), device=device),
                states[:, :, 1:]
            )
            
            benchmark_optimizations(states, var_threshold, padding_idx)
            print()
    else:
        print("CUDA not available. GPU optimizations require a GPU.")
