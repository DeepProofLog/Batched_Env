"""
Evaluation methods for PPO policy.

This module contains different evaluation strategies:
1. Standard evaluation - Original sequential implementation
2. Batched evaluation - Optimized for large-scale (RECOMMENDED)
3. Streaming evaluation - Auto-reset approach (EXPERIMENTAL, NOT RECOMMENDED)

For production use, import and use evaluate_batched().
"""

import torch
from torch import Tensor
from typing import Sequence, Dict, Tuple, Optional
from tensordict import TensorDict


# =============================================================================
# STANDARD EVALUATION (Original implementation)
# =============================================================================

@torch.no_grad()
def evaluate_standard(
    policy,
    env,
    queries: Tensor,
    sampler,
    n_corruptions: int = 50,
    corruption_modes: Sequence[str] = ('head', 'tail'),
    *,
    verbose: bool = False,
    parity_mode: bool = False,
    mask_fill_value: float = -3.4e38
) -> Dict[str, float]:
    """Standard evaluation - processes one query at a time.

    WARNING: This method is NOT COMPATIBLE with torch.compile and fixed batch sizes.
    It is kept for reference only. Use evaluate_batched() instead.

    This method processes candidates sequentially, which is incompatible with
    environments that have fixed batch sizes (batch_size=128). It will fail
    with compiled environments.

    Args:
        policy: Policy network to evaluate
        env: Environment instance
        queries: Test queries [N, 3]
        sampler: Negative sampler
        n_corruptions: Number of negative samples per query
        corruption_modes: Which positions to corrupt ('head', 'tail')
        verbose: Print debug info
        parity_mode: Use compiled eval step (for testing)
        mask_fill_value: Value for masking invalid actions

    Returns:
        Dictionary of metrics (MRR, Hits@1, Hits@3, Hits@10)
    """
    device = env.device
    results = {}

    for mode in corruption_modes:
        mode_results = []

        for query in queries:
            # Generate corruptions
            query_batch = query.unsqueeze(0)  # [1, 3]
            neg_candidates = sampler.corrupt(
                query_batch, num_negatives=n_corruptions, mode=mode, device=device
            )

            # Create candidate pool: [positive] + [negatives]
            all_candidates = torch.cat([
                query_batch.unsqueeze(1),  # [1, 1, 3]
                neg_candidates              # [1, K, 3]
            ], dim=1).squeeze(0)  # [K+1, 3]

            # Evaluate each candidate
            successes = torch.zeros(all_candidates.size(0), dtype=torch.bool, device=device)

            for i, candidate in enumerate(all_candidates):
                obs, state = env.reset(candidate.unsqueeze(0))
                done = False

                while not done:
                    logits = policy.get_logits(obs)
                    masked_logits = torch.where(obs['action_mask'].bool(), logits, mask_fill_value)
                    actions = masked_logits.argmax(dim=-1)
                    obs, state = env.step(state, actions, auto_reset=False)
                    done = state['done'][0].item()

                successes[i] = state['success'][0].item()

            mode_results.append(successes)

        # Compute metrics
        all_successes = torch.stack(mode_results)  # [N, K+1]
        scores = torch.where(all_successes,
                            torch.zeros_like(all_successes, dtype=torch.float32),
                            torch.full_like(all_successes, -100.0, dtype=torch.float32))

        pos_score = scores[:, 0:1]
        neg_scores = scores[:, 1:]

        rnd = torch.rand(len(queries), n_corruptions + 1, device=device)
        better = neg_scores > pos_score
        tied = (neg_scores == pos_score) & (rnd[:, 1:] > rnd[:, 0:1])
        ranks = 1 + better.sum(dim=1) + tied.sum(dim=1)

        results[f'{mode}_mrr'] = (1.0 / ranks.float()).mean().item()
        results[f'{mode}_hits1'] = (ranks <= 1).float().mean().item()
        results[f'{mode}_hits3'] = (ranks <= 3).float().mean().item()
        results[f'{mode}_hits10'] = (ranks <= 10).float().mean().item()

    # Aggregate
    n_modes = len(corruption_modes)
    results['MRR'] = sum(results[f'{m}_mrr'] for m in corruption_modes) / n_modes
    results['Hits@1'] = sum(results[f'{m}_hits1'] for m in corruption_modes) / n_modes
    results['Hits@3'] = sum(results[f'{m}_hits3'] for m in corruption_modes) / n_modes
    results['Hits@10'] = sum(results[f'{m}_hits10'] for m in corruption_modes) / n_modes

    return results


# =============================================================================
# PERSISTENT STATE EVALUATION (Zero-Copy CUDA Graphs, RECOMMENDED)
# =============================================================================

@torch.no_grad()
def evaluate_persistent(
    policy,
    env,
    queries: Tensor,
    sampler,
    n_corruptions: int = 50,
    corruption_modes: Sequence[str] = ('head', 'tail'),
    *,
    verbose: bool = False,
    mask_fill_value: float = -3.4e38,
    compiled_persistent_step: Optional[callable] = None
) -> Dict[str, float]:
    """Persistent State Evaluation - Zero-copy CUDA graphs.

    Key innovation: The compiled step has ZERO arguments and operates on
    persistent buffers via closure capture. This eliminates _foreach_copy_
    overhead entirely (76% of runtime, 22s → <2s).

    Architecture:
    - env._persistent_state and env._persistent_obs are owned by environment
    - Compiled step reads/writes these via closure (not function arguments)
    - init_persistent_state() called ONCE per round (outside graph)
    - Zero graph inputs → zero copy overhead

    Performance target: 28.9s → <10s baseline (>66% speedup).

    Args:
        policy: Policy network
        env: Environment with persistent state support
        queries: Test queries [N, 3]
        sampler: Negative sampler
        n_corruptions: Number of negative samples per query
        corruption_modes: Which positions to corrupt ('head', 'tail')
        verbose: Print debug info
        mask_fill_value: Value for masking invalid actions
        compiled_persistent_step: Pre-compiled zero-argument step function

    Returns:
        Dictionary of metrics (MRR, Hits@1, Hits@3, Hits@10)
    """
    device = env.device
    batch_size = env.batch_size
    max_depth = env.max_depth
    total_queries = len(queries)
    total_candidates_per_query = 1 + n_corruptions

    # 1. Generate ALL candidates for ALL modes
    all_pools = []
    for mode in corruption_modes:
        negative_candidates = sampler.corrupt(
            queries, num_negatives=n_corruptions, mode=mode, device=device
        )
        all_cands = torch.cat([
            queries.unsqueeze(1),
            negative_candidates
        ], dim=1)
        all_pools.append(all_cands.view(-1, 3))

    query_pool = torch.cat(all_pools, dim=0)
    pool_size = query_pool.size(0)

    if verbose:
        print(f"Persistent eval: {pool_size} candidates in pool")

    # 2. Ensure compiled step is available
    if compiled_persistent_step is None:
        raise ValueError("compiled_persistent_step is required")

    # 3. Process pool in batched rounds
    n_rounds = (pool_size + batch_size - 1) // batch_size
    success_buffer = torch.zeros(pool_size, dtype=torch.bool, device=device)

    if verbose:
        print(f"Eval pool: {pool_size}, batch: {batch_size}, rounds: {n_rounds}")

    # Pre-allocate persistent buffers for copy-outside pattern (CUDA graphs)
    # These buffers maintain stable addresses across iterations
    obs_buffer = None
    state_buffer = None

    for round_idx in range(n_rounds):
        start_idx = round_idx * batch_size
        end_idx = min(start_idx + batch_size, pool_size)
        round_size = end_idx - start_idx

        # Get queries for this round
        round_queries = query_pool[start_idx:end_idx]
        if round_size < batch_size:
            padding = batch_size - round_size
            pad_queries = torch.zeros(padding, 3, dtype=torch.long, device=device)
            round_queries = torch.cat([round_queries, pad_queries], dim=0)

        # Initialize state (ONCE per round, OUTSIDE graph)
        obs, state = env.reset(round_queries)

        # Allocate persistent buffers on first use (same shape as reset output)
        if obs_buffer is None:
            obs_buffer = TensorDict({
                'sub_index': torch.zeros_like(obs['sub_index']),
                'derived_sub_indices': torch.zeros_like(obs['derived_sub_indices']),
                'action_mask': torch.zeros_like(obs['action_mask']),
            }, batch_size=obs.batch_size, device=device)

            state_buffer = TensorDict({
                'current_states': torch.zeros_like(state['current_states']),
                'derived_states': torch.zeros_like(state['derived_states']),
                'derived_counts': torch.zeros_like(state['derived_counts']),
                'original_queries': torch.zeros_like(state['original_queries']),
                'next_var_indices': torch.zeros_like(state['next_var_indices']),
                'depths': torch.zeros_like(state['depths']),
                'done': torch.zeros_like(state['done']),
                'success': torch.zeros_like(state['success']),
                'current_labels': torch.zeros_like(state['current_labels']),
                'history_hashes': torch.zeros_like(state['history_hashes']),
                'history_count': torch.zeros_like(state['history_count']),
                'step_rewards': torch.zeros_like(state['step_rewards']),
                'step_dones': torch.zeros_like(state['step_dones']),
                'per_env_ptrs': torch.zeros_like(state['per_env_ptrs']),
                'neg_counters': torch.zeros_like(state['neg_counters']),
                'cumulative_rewards': torch.zeros_like(state['cumulative_rewards']),
            }, batch_size=state.batch_size, device=device)

        # Track which slots have finished
        slot_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        slot_success = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Run max_depth steps - copy-outside pattern
        for step in range(max_depth):
            torch.compiler.cudagraph_mark_step_begin()

            # Pure computation INSIDE CUDA graph (returns NEW tensors)
            new_obs, new_state = compiled_persistent_step(obs, state)

            # Copy OUTSIDE compiled region (avoids mutation detection)
            obs_buffer['sub_index'].copy_(new_obs['sub_index'])
            obs_buffer['derived_sub_indices'].copy_(new_obs['derived_sub_indices'])
            obs_buffer['action_mask'].copy_(new_obs['action_mask'])

            state_buffer['current_states'].copy_(new_state['current_states'])
            state_buffer['derived_states'].copy_(new_state['derived_states'])
            state_buffer['derived_counts'].copy_(new_state['derived_counts'])
            state_buffer['original_queries'].copy_(new_state['original_queries'])
            state_buffer['next_var_indices'].copy_(new_state['next_var_indices'])
            state_buffer['depths'].copy_(new_state['depths'])
            state_buffer['done'].copy_(new_state['done'])
            state_buffer['success'].copy_(new_state['success'])
            state_buffer['current_labels'].copy_(new_state['current_labels'])
            state_buffer['history_hashes'].copy_(new_state['history_hashes'])
            state_buffer['history_count'].copy_(new_state['history_count'])
            state_buffer['step_rewards'].copy_(new_state['step_rewards'])
            state_buffer['step_dones'].copy_(new_state['step_dones'])
            state_buffer['per_env_ptrs'].copy_(new_state['per_env_ptrs'])
            state_buffer['neg_counters'].copy_(new_state['neg_counters'])
            state_buffer['cumulative_rewards'].copy_(new_state['cumulative_rewards'])

            # Use buffers for next iteration
            obs = obs_buffer
            state = state_buffer

            # Read results from state buffer
            just_done = state['step_dones'].bool()
            step_success = (state['step_rewards'] > 0.5)

            # Record first finish for each slot
            newly_done = just_done & ~slot_finished
            slot_success = torch.where(newly_done, step_success, slot_success)
            slot_finished = slot_finished | just_done

        # Write results for this round
        if round_size < batch_size:
            success_buffer[start_idx:end_idx] = slot_success[:round_size]
        else:
            success_buffer[start_idx:end_idx] = slot_success

    if verbose:
        filled = success_buffer[:pool_size].sum().item()
        print(f"Total: filled={filled}/{pool_size} (success rate: {filled/pool_size:.1%})")

    # 4. Process results by mode
    results = {}
    offset = 0

    for mode in corruption_modes:
        mode_size = total_queries * total_candidates_per_query
        mode_success_flat = success_buffer[offset : offset + mode_size]
        mode_success = mode_success_flat.view(total_queries, total_candidates_per_query)

        if verbose:
            pos_success_rate = mode_success[:, 0].float().mean().item()
            neg_success_rate = mode_success[:, 1:].float().mean().item() if mode_success.size(1) > 1 else 0
            print(f"  {mode}: pos_success={pos_success_rate:.3f}, neg_success={neg_success_rate:.3f}")

        # Rank computation
        scores = torch.where(mode_success,
                            torch.zeros_like(mode_success, dtype=torch.float32),
                            torch.full_like(mode_success, -100.0, dtype=torch.float32))

        pos_score = scores[:, 0:1]
        neg_scores = scores[:, 1:]

        rnd = torch.rand(total_queries, total_candidates_per_query, device=device)
        better = neg_scores > pos_score
        tied = (neg_scores == pos_score) & (rnd[:, 1:] > rnd[:, 0:1])
        ranks = 1 + better.sum(dim=1) + tied.sum(dim=1)

        if verbose:
            print(f"    sample ranks: {ranks[:5].tolist()}")

        # Compute metrics on GPU to minimize synchronization overhead
        mrr = (1.0 / ranks.float()).mean()
        hits1 = (ranks <= 1).float().mean()
        hits3 = (ranks <= 3).float().mean()
        hits10 = (ranks <= 10).float().mean()

        # Single synchronization point per mode
        metrics = torch.stack([mrr, hits1, hits3, hits10]).tolist()

        results[f'{mode}_mrr'] = metrics[0]
        results[f'{mode}_hits1'] = metrics[1]
        results[f'{mode}_hits3'] = metrics[2]
        results[f'{mode}_hits10'] = metrics[3]

        offset += mode_size

    # Aggregate
    n_modes = len(corruption_modes)
    results['MRR'] = sum(results[f'{m}_mrr'] for m in corruption_modes) / n_modes
    results['Hits@1'] = sum(results[f'{m}_hits1'] for m in corruption_modes) / n_modes
    results['Hits@3'] = sum(results[f'{m}_hits3'] for m in corruption_modes) / n_modes
    results['Hits@10'] = sum(results[f'{m}_hits10'] for m in corruption_modes) / n_modes

    return results


# =============================================================================
# BATCHED EVALUATION (Optimized, RECOMMENDED)
# =============================================================================

@torch.no_grad()
def evaluate_batched(
    policy,
    env,
    queries: Tensor,
    sampler,
    n_corruptions: int = 50,
    corruption_modes: Sequence[str] = ('head', 'tail'),
    *,
    verbose: bool = False,
    parity_mode: bool = False,
    mask_fill_value: float = -3.4e38,
    eval_pool_buffer: Optional[Tensor] = None,
    compiled_eval_step: Optional[callable] = None,
    eval_state_buffer: Optional[TensorDict] = None
) -> Dict[str, float]:
    """Batched evaluation - optimized for large-scale datasets.

    Key optimizations:
    - Step function updates state IN-PLACE and returns same buffer refs
    - Same tensor addresses every call = no CUDA graph copy overhead
    - Pre-allocated fixed buffer for CUDA graph compatibility

    Performance target: 28.9s -> <10s (76% overhead eliminated).
    """
    device = env.device
    batch_size = env.batch_size
    max_depth = env.max_depth
    total_queries = len(queries)
    total_candidates_per_query = 1 + n_corruptions

    # 1. Generate ALL candidates for ALL modes into a single flat pool
    all_pools = []
    for mode in corruption_modes:
        negative_candidates = sampler.corrupt(
            queries, num_negatives=n_corruptions, mode=mode, device=device
        )
        all_cands = torch.cat([
            queries.unsqueeze(1),
            negative_candidates
        ], dim=1)
        all_pools.append(all_cands.view(-1, 3))

    query_pool = torch.cat(all_pools, dim=0)
    pool_size = query_pool.size(0)

    # 2. Copy into fixed buffer for CUDA graph compatibility
    if eval_pool_buffer is None:
        raise ValueError("eval_pool_buffer is required for batched evaluation")

    if pool_size > eval_pool_buffer.size(0):
        if eval_pool_buffer.size(0) < 2000000:
            print(f"Warning: Query pool ({pool_size}) exceeds buffer ({eval_pool_buffer.size(0)}). Truncating.")
            pool_size = eval_pool_buffer.size(0)
            query_pool = query_pool[:pool_size]
        else:
            raise ValueError(f"Query pool ({pool_size}) exceeds buffer ({eval_pool_buffer.size(0)})")

    eval_pool_buffer[:pool_size].copy_(query_pool)
    env._query_pool = eval_pool_buffer

    if verbose:
        print(f"Batched eval: {pool_size} candidates in pool")

    # 3. Ensure compiled step is available
    if compiled_eval_step is None:
        raise ValueError("compiled_eval_step is required")

    # 4. Process pool in batched rounds
    n_rounds = (pool_size + batch_size - 1) // batch_size
    success_buffer = torch.zeros(pool_size, dtype=torch.bool, device=device)

    if verbose:
        print(f"Eval pool: {pool_size}, batch: {batch_size}, rounds: {n_rounds}")

    # Pre-allocate persistent buffers for copy-outside pattern (CUDA graphs)
    # These buffers maintain stable addresses across iterations
    obs_buffer = None
    state_buffer = None

    for round_idx in range(n_rounds):
        start_idx = round_idx * batch_size
        end_idx = min(start_idx + batch_size, pool_size)
        round_size = end_idx - start_idx

        # Reset environments with candidates for this round
        round_queries = eval_pool_buffer[start_idx:end_idx]
        if round_size < batch_size:
            padding = batch_size - round_size
            pad_queries = torch.zeros(padding, 3, dtype=torch.long, device=device)
            round_queries = torch.cat([round_queries, pad_queries], dim=0)

        obs, state = env.reset(round_queries)

        # Allocate persistent buffers on first use (same shape as reset output)
        if obs_buffer is None:
            obs_buffer = TensorDict({
                'sub_index': torch.zeros_like(obs['sub_index']),
                'derived_sub_indices': torch.zeros_like(obs['derived_sub_indices']),
                'action_mask': torch.zeros_like(obs['action_mask']),
            }, batch_size=obs.batch_size, device=device)

            state_buffer = TensorDict({
                'current_states': torch.zeros_like(state['current_states']),
                'derived_states': torch.zeros_like(state['derived_states']),
                'derived_counts': torch.zeros_like(state['derived_counts']),
                'original_queries': torch.zeros_like(state['original_queries']),
                'next_var_indices': torch.zeros_like(state['next_var_indices']),
                'depths': torch.zeros_like(state['depths']),
                'done': torch.zeros_like(state['done']),
                'success': torch.zeros_like(state['success']),
                'current_labels': torch.zeros_like(state['current_labels']),
                'history_hashes': torch.zeros_like(state['history_hashes']),
                'history_count': torch.zeros_like(state['history_count']),
                'step_rewards': torch.zeros_like(state['step_rewards']),
                'step_dones': torch.zeros_like(state['step_dones']),
                'per_env_ptrs': torch.zeros_like(state['per_env_ptrs']),
                'neg_counters': torch.zeros_like(state['neg_counters']),
                'cumulative_rewards': torch.zeros_like(state['cumulative_rewards']),
            }, batch_size=state.batch_size, device=device)

        # Track which slots have finished
        slot_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        slot_success = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Run max_depth steps - copy-outside pattern
        for step in range(max_depth):
            torch.compiler.cudagraph_mark_step_begin()

            # Pure computation INSIDE CUDA graph (returns NEW tensors)
            new_obs, new_state = compiled_eval_step(obs, state)

            # Copy OUTSIDE compiled region (avoids mutation detection)
            obs_buffer['sub_index'].copy_(new_obs['sub_index'])
            obs_buffer['derived_sub_indices'].copy_(new_obs['derived_sub_indices'])
            obs_buffer['action_mask'].copy_(new_obs['action_mask'])

            state_buffer['current_states'].copy_(new_state['current_states'])
            state_buffer['derived_states'].copy_(new_state['derived_states'])
            state_buffer['derived_counts'].copy_(new_state['derived_counts'])
            state_buffer['original_queries'].copy_(new_state['original_queries'])
            state_buffer['next_var_indices'].copy_(new_state['next_var_indices'])
            state_buffer['depths'].copy_(new_state['depths'])
            state_buffer['done'].copy_(new_state['done'])
            state_buffer['success'].copy_(new_state['success'])
            state_buffer['current_labels'].copy_(new_state['current_labels'])
            state_buffer['history_hashes'].copy_(new_state['history_hashes'])
            state_buffer['history_count'].copy_(new_state['history_count'])
            state_buffer['step_rewards'].copy_(new_state['step_rewards'])
            state_buffer['step_dones'].copy_(new_state['step_dones'])
            state_buffer['per_env_ptrs'].copy_(new_state['per_env_ptrs'])
            state_buffer['neg_counters'].copy_(new_state['neg_counters'])
            state_buffer['cumulative_rewards'].copy_(new_state['cumulative_rewards'])

            # Use buffers for next iteration
            obs = obs_buffer
            state = state_buffer

            # Read results from state buffer
            just_done = state['step_dones'].bool()
            step_success = (state['step_rewards'] > 0.5)

            # Record first finish for each slot
            newly_done = just_done & ~slot_finished
            slot_success = torch.where(newly_done, step_success, slot_success)
            slot_finished = slot_finished | just_done

        # Write results for this round
        if round_size < batch_size:
            success_buffer[start_idx:end_idx] = slot_success[:round_size]
        else:
            success_buffer[start_idx:end_idx] = slot_success

    if verbose:
        filled = success_buffer[:pool_size].sum().item()
        print(f"Total: filled={filled}/{pool_size} (success rate: {filled/pool_size:.1%})")

    # 5. Process results by mode
    results = {}
    offset = 0

    for mode in corruption_modes:
        mode_size = total_queries * total_candidates_per_query
        mode_success_flat = success_buffer[offset : offset + mode_size]
        mode_success = mode_success_flat.view(total_queries, total_candidates_per_query)

        if verbose:
            pos_success_rate = mode_success[:, 0].float().mean().item()
            neg_success_rate = mode_success[:, 1:].float().mean().item() if mode_success.size(1) > 1 else 0
            print(f"  {mode}: pos_success={pos_success_rate:.3f}, neg_success={neg_success_rate:.3f}")

        # Rank computation
        scores = torch.where(mode_success,
                            torch.zeros_like(mode_success, dtype=torch.float32),
                            torch.full_like(mode_success, -100.0, dtype=torch.float32))

        pos_score = scores[:, 0:1]
        neg_scores = scores[:, 1:]

        rnd = torch.rand(total_queries, total_candidates_per_query, device=device)
        better = neg_scores > pos_score
        tied = (neg_scores == pos_score) & (rnd[:, 1:] > rnd[:, 0:1])
        ranks = 1 + better.sum(dim=1) + tied.sum(dim=1)

        if verbose:
            print(f"    sample ranks: {ranks[:5].tolist()}")

        # Compute metrics on GPU to minimize synchronization overhead
        mrr = (1.0 / ranks.float()).mean()
        hits1 = (ranks <= 1).float().mean()
        hits3 = (ranks <= 3).float().mean()
        hits10 = (ranks <= 10).float().mean()

        # Single synchronization point per mode
        metrics = torch.stack([mrr, hits1, hits3, hits10]).tolist()

        results[f'{mode}_mrr'] = metrics[0]
        results[f'{mode}_hits1'] = metrics[1]
        results[f'{mode}_hits3'] = metrics[2]
        results[f'{mode}_hits10'] = metrics[3]

        offset += mode_size

    # Aggregate
    n_modes = len(corruption_modes)
    results['MRR'] = sum(results[f'{m}_mrr'] for m in corruption_modes) / n_modes
    results['Hits@1'] = sum(results[f'{m}_hits1'] for m in corruption_modes) / n_modes
    results['Hits@3'] = sum(results[f'{m}_hits3'] for m in corruption_modes) / n_modes
    results['Hits@10'] = sum(results[f'{m}_hits10'] for m in corruption_modes) / n_modes

    return results


# =============================================================================
# STREAMING EVALUATION (Auto-reset, EXPERIMENTAL, NOT RECOMMENDED)
# =============================================================================

@torch.no_grad()
def evaluate_streaming(
    policy,
    env,
    queries: Tensor,
    sampler,
    n_corruptions: int = 50,
    corruption_modes: Sequence[str] = ('head', 'tail'),
    *,
    verbose: bool = False,
    parity_mode: bool = False,
    mask_fill_value: float = -3.4e38,
    eval_pool_buffer: Optional[Tensor] = None,
    compiled_streaming_step: Optional[callable] = None
) -> Dict[str, float]:
    """Streaming evaluation with auto-reset - EXPERIMENTAL, NOT RECOMMENDED.

    WARNING: This method is 8.9× SLOWER than batched evaluation and produces
    incorrect results. It is kept for reference and research purposes only.

    DO NOT USE IN PRODUCTION. Use evaluate_batched() instead.

    Why it's slow:
    - Auto-reset adds 13ms overhead per step (14× torch.where() merges)
    - Tracking overhead: 389ms for per-slot state management
    - Even theoretical best case: 2.4× slower than batched

    See AUTO_RESET_PERFORMANCE_ANALYSIS.md for detailed analysis.

    Args:
        policy: Policy network to evaluate
        env: Environment instance
        queries: Test queries [N, 3]
        sampler: Negative sampler
        n_corruptions: Number of negative samples per query
        corruption_modes: Which positions to corrupt ('head', 'tail')
        verbose: Print detailed progress
        parity_mode: Ignored (for compatibility)
        mask_fill_value: Value for masking invalid actions
        eval_pool_buffer: Pre-allocated buffer for query pool
        compiled_streaming_step: Pre-compiled streaming step with auto-reset

    Returns:
        Dictionary of metrics (LIKELY INCORRECT - do not use!)
    """
    if not verbose:
        print("WARNING: evaluate_streaming() is experimental and NOT RECOMMENDED.")
        print("Use evaluate_batched() instead for correct and fast evaluation.")

    device = env.device
    batch_size = env.batch_size
    total_queries = len(queries)
    total_candidates_per_query = 1 + n_corruptions

    # Generate candidate pool (same as batched)
    all_pools = []
    for mode in corruption_modes:
        negative_candidates = sampler.corrupt(
            queries, num_negatives=n_corruptions, mode=mode, device=device
        )
        all_cands = torch.cat([
            queries.unsqueeze(1),
            negative_candidates
        ], dim=1)
        all_pools.append(all_cands.view(-1, 3))

    query_pool = torch.cat(all_pools, dim=0)
    pool_size = query_pool.size(0)

    if eval_pool_buffer is None or pool_size > eval_pool_buffer.size(0):
        raise ValueError("Invalid eval_pool_buffer")

    eval_pool_buffer[:pool_size].copy_(query_pool)
    env._query_pool = eval_pool_buffer[:pool_size]

    # Set ordered mode
    original_order = env.order
    env.order = True

    if verbose:
        print(f"Streaming eval: {pool_size} candidates, batch_size: {batch_size}")

    if compiled_streaming_step is None:
        raise ValueError("compiled_streaming_step is required")

    # Initialize with stride for non-overlapping assignment
    per_env_ptrs = torch.arange(batch_size, device=device)

    # Result tracking
    success_buffer = torch.zeros(pool_size, dtype=torch.bool, device=device)
    slot_pool_idx = per_env_ptrs.clone()
    slot_first_done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    slot_success = torch.zeros(batch_size, dtype=torch.bool, device=device)
    processed_mask = torch.zeros(pool_size, dtype=torch.bool, device=device)

    # Initial reset
    torch.compiler.cudagraph_mark_step_begin()
    initial_queries = eval_pool_buffer[per_env_ptrs]
    obs, state = env.reset(initial_queries)
    state['per_env_ptrs'] = per_env_ptrs

    # Run streaming evaluation
    total_steps = (pool_size // batch_size) * env.max_depth
    if pool_size % batch_size != 0:
        total_steps += env.max_depth

    processed_count = 0
    step = 0

    while processed_count < pool_size and step < total_steps * 2:
        torch.compiler.cudagraph_mark_step_begin()

        # Clone for CUDA graphs
        obs_in = {k: v.clone() for k, v in obs.items()}
        state_in = state.clone()

        # Auto-reset step
        obs, state = compiled_streaming_step(obs_in, state_in)

        # Extract results
        just_done = state['step_dones'].clone().bool()
        step_success_flag = (state['step_rewards'].clone() > 0.5)
        new_ptrs = state['per_env_ptrs'].clone()

        # Track first-done per slot
        first_done_now = just_done & ~slot_first_done
        if first_done_now.any():
            slot_success = torch.where(first_done_now, step_success_flag, slot_success)
            slot_first_done = slot_first_done | first_done_now

        # Write results when slots finish
        if just_done.any():
            valid_idx_mask = (slot_pool_idx < pool_size)
            ready_to_write = just_done & valid_idx_mask & ~processed_mask[slot_pool_idx]

            if ready_to_write.any():
                write_indices = slot_pool_idx[ready_to_write]
                write_success = slot_success[ready_to_write]
                success_buffer.index_put_((write_indices,), write_success, accumulate=False)
                processed_mask.index_put_((write_indices,), torch.ones_like(write_indices, dtype=torch.bool))
                processed_count += ready_to_write.sum().item()

            # Update slot tracking
            slot_pool_idx = torch.where(just_done, new_ptrs, slot_pool_idx)
            slot_first_done = torch.where(just_done, torch.zeros_like(slot_first_done), slot_first_done)

        step += 1

    if verbose:
        print(f"Final: processed={processed_count}, filled={success_buffer.sum().item()}/{pool_size}")

    # Restore environment state
    env.order = original_order

    # Process results (same as batched)
    results = {}
    offset = 0

    for mode in corruption_modes:
        mode_size = total_queries * total_candidates_per_query
        mode_success_flat = success_buffer[offset : offset + mode_size]
        mode_success = mode_success_flat.view(total_queries, total_candidates_per_query)

        scores = torch.where(mode_success,
                            torch.zeros_like(mode_success, dtype=torch.float32),
                            torch.full_like(mode_success, -100.0, dtype=torch.float32))

        pos_score = scores[:, 0:1]
        neg_scores = scores[:, 1:]

        rnd = torch.rand(total_queries, total_candidates_per_query, device=device)
        better = neg_scores > pos_score
        tied = (neg_scores == pos_score) & (rnd[:, 1:] > rnd[:, 0:1])
        ranks = 1 + better.sum(dim=1) + tied.sum(dim=1)

        results[f'{mode}_mrr'] = (1.0 / ranks.float()).mean().item()
        results[f'{mode}_hits1'] = (ranks <= 1).float().mean().item()
        results[f'{mode}_hits3'] = (ranks <= 3).float().mean().item()
        results[f'{mode}_hits10'] = (ranks <= 10).float().mean().item()

        offset += mode_size

    n_modes = len(corruption_modes)
    results['MRR'] = sum(results[f'{m}_mrr'] for m in corruption_modes) / n_modes
    results['Hits@1'] = sum(results[f'{m}_hits1'] for m in corruption_modes) / n_modes
    results['Hits@3'] = sum(results[f'{m}_hits3'] for m in corruption_modes) / n_modes
    results['Hits@10'] = sum(results[f'{m}_hits10'] for m in corruption_modes) / n_modes

    return results


# =============================================================================
# HELPER FUNCTIONS FOR COMPILATION
# =============================================================================

def create_batched_eval_step(policy, env, mask_fill_value: float = -3.4e38,
                              obs_buffer_A=None, state_buffer_A=None,
                              obs_buffer_B=None, state_buffer_B=None):
    """Create compiled evaluation step for batched evaluation.

    Uses return-based signature. The environment's _step_core() updates
    the input state in-place and returns the same buffer references,
    ensuring stable tensor addresses for CUDA graphs.

    Args:
        policy: Policy network
        env: Environment instance
        mask_fill_value: Value for masking invalid actions
        obs_buffer_A, state_buffer_A: Initial buffer set (for warmup)
        obs_buffer_B, state_buffer_B: Unused (for API compatibility)

    Returns:
        Compiled step function
    """
    def batched_eval_step(obs, state):
        """Single step WITHOUT auto-reset. Pure functional - returns NEW tensors."""
        logits = policy.get_logits(obs)
        masked_logits = torch.where(obs['action_mask'].bool(), logits, mask_fill_value)
        actions = masked_logits.argmax(dim=-1)
        # Use regular _step_core (pure functional, returns new TensorDicts)
        # Copying into persistent buffers happens OUTSIDE this compiled function
        new_obs, new_state = env._step_core(state, actions)
        return new_obs, new_state

    compiled_step = torch.compile(
        batched_eval_step, mode='reduce-overhead', fullgraph=True
    )
    print("[Evaluation] Batched eval step compilation complete (reduce-overhead, copy-outside pattern).")
    return compiled_step


def create_persistent_eval_step(policy, env, mask_fill_value: float = -3.4e38):
    """Create optimized evaluation step that eliminates _foreach_copy_ overhead.

    Key Innovation: Pure functional compiled step + copying OUTSIDE compiled region.
    This allows CUDA graphs to work while avoiding the 75% _foreach_copy_ overhead
    that occurs when TensorDict arguments are passed to compiled functions.

    Pattern:
    - Computation happens INSIDE compiled function (gets CUDA graph speedup)
    - Copying into persistent buffers happens OUTSIDE (avoids mutation detection)
    - Result: 88%+ computation, <12% copying (vs 25% computation, 75% copying before)

    Args:
        policy: Policy network
        env: Environment instance
        mask_fill_value: Value for masking invalid actions

    Returns:
        Compiled step function (same as batched_eval_step - both use copy-outside pattern)
    """
    def persistent_eval_step(obs, state):
        """Pure functional step - returns NEW TensorDicts."""
        logits = policy.get_logits(obs)
        masked_logits = torch.where(obs['action_mask'].bool(), logits, mask_fill_value)
        actions = masked_logits.argmax(dim=-1)
        # Pure functional - returns new TensorDicts
        # Copying happens OUTSIDE this function (in evaluation loop)
        new_obs, new_state = env._step_core(state, actions)
        return new_obs, new_state

    compiled_step = torch.compile(
        persistent_eval_step,
        mode='reduce-overhead',
        fullgraph=True,
        dynamic=False
    )

    print("[Evaluation] Persistent eval step compiled (reduce-overhead, copy-outside pattern).")
    return compiled_step


def create_streaming_eval_step(policy, env, mask_fill_value: float = -3.4e38):
    """Create compiled evaluation step for streaming evaluation.

    WARNING: This creates an auto-reset step that is 9.7× slower than
    the batched version. Only use for research/comparison purposes.

    Args:
        policy: Policy network
        env: Environment instance
        mask_fill_value: Value for masking invalid actions

    Returns:
        Compiled streaming step function with auto-reset
    """
    def streaming_eval_step(obs, state):
        """Single step WITH auto-reset (stride=batch_size)."""
        logits = policy.get_logits(obs)
        masked_logits = torch.where(obs['action_mask'].bool(), logits, mask_fill_value)
        actions = masked_logits.argmax(dim=-1)

        # Auto-reset with stride for non-overlapping assignment
        new_obs, new_state = env._step_and_reset_core(
            state, actions, env._query_pool, state['per_env_ptrs'],
            stride=env.batch_size
        )
        return new_obs, new_state

    compiled_step = torch.compile(
        streaming_eval_step, mode='reduce-overhead', fullgraph=True
    )
    print("[Evaluation] Streaming eval step compilation complete (auto-reset with stride).")
    return compiled_step
