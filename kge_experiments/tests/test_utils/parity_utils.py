import torch
import numpy as np
from typing import Any, Optional, Tuple, Sequence, Dict, List, Union
from tensordict import TensorDict
from env import EnvVec, EnvState, EnvObs

class TensorDictEnvWrapper:
    """TensorDict wrapper for EnvVec providing slot-based evaluation logic.
    
    This wrapper serves two purposes:
    
    1. **TensorDict API Compatibility**: Provides reset()/step() methods that accept
       and return TensorDict objects, matching BatchedEnv's interface for parity tests.
    
    2. **Slot-Based Evaluation Protocol**: Implements per-slot query scheduling for
       eval_corruptions/evaluate_parity. Each environment slot processes a different
       number of queries (positive + N corruptions), requiring:
       - per_slot_lengths: Number of queries per slot
       - per_slot_offsets: Starting index in query pool per slot  
       - slot_ep_counts: Episode counter per slot
       - Partial reset via _reset mask to reset only finished slots
    
    Why not use _step_and_reset_core()?
        The compiled _step_and_reset_core() handles training resets via round-robin
        cycling through a flat query_pool. It doesn't support the ragged per-slot
        scheduling needed for evaluation where slot 0 might have 11 queries and
        slot 1 might have 8 queries.
    
    Usage:
        - Training: Use EnvVec directly with _step_and_reset_core() (no wrapper needed)
        - Evaluation: Use TensorDictEnvWrapper for slot-based query scheduling
    """

    def __init__(self, env: EnvVec):
        self.env = env
        self.batch_size, self.device = env.batch_size, env.device
        self._state = None
        self._eval_queries = None
        self._per_slot_lengths = None
        self._per_slot_offsets = None
        self._slot_ep_counts = None

    def eval(self, q: Optional[torch.Tensor] = None): return self.env.eval(q)
    def train(self): return self.env.train()

    # Forward critical engine properties for PPOOptimal compatibility
    @property
    def train_queries(self): return self.env.train_queries
    @property
    def eval_queries(self): return self.env.eval_queries
    @property
    def _query_pool(self): return self.env._query_pool
    @property
    def _per_env_ptrs(self): return self.env._per_env_ptrs
    @_per_env_ptrs.setter
    def _per_env_ptrs(self, val): self.env._per_env_ptrs = val

    @property
    def padding_idx(self): return self.env.padding_idx
    @property
    def true_pred_idx(self): return self.env.true_pred_idx
    @property
    def false_pred_idx(self): return self.env.false_pred_idx
    @property
    def end_pred_idx(self): return self.env.end_pred_idx
    @property
    def max_depth(self): return self.env.max_depth
    
    def get_derived_simple(self, *args, **kwargs): return self.env.get_derived_simple(*args, **kwargs)
    def sample_negatives(self, *args, **kwargs): return self.env.sample_negatives(*args, **kwargs)
    def _compute_hash(self, *args, **kwargs): return self.env._compute_hash(*args, **kwargs)
    def _reset_from_queries(self, *args, **kwargs): return self.env._reset_from_queries(*args, **kwargs)
    def _step_and_reset_core(self, *args, **kwargs): return self.env._step_and_reset_core(*args, **kwargs)
    def _state_to_obs(self, *args, **kwargs): return self.env._state_to_obs(*args, **kwargs)

    def set_eval_dataset(self, queries: torch.Tensor, labels: torch.Tensor, query_depths: torch.Tensor,
                         per_slot_lengths: Optional[torch.Tensor] = None):
        """Setup per-query slot tracking for eval_corruptions."""
        B = self.batch_size
        self._eval_queries = queries.to(self.device)
        self._per_slot_lengths = per_slot_lengths[:B].clone().to(self.device) if per_slot_lengths is not None else torch.ones(B, dtype=torch.long, device=self.device)
        cumsum = torch.cumsum(self._per_slot_lengths, dim=0)
        self._per_slot_offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device), cumsum[:-1]])
        self._slot_ep_counts = torch.zeros(B, dtype=torch.long, device=self.device)
        self.env.eval(queries)

    def _get_slot_queries(self, reset_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Fetch initial queries for all slots (used for first reset only).
        
        Incremental query fetching is handled by _step_and_reset_core via slot_lengths/slot_offsets.
        """
        B, pad = self.batch_size, self.env.padding_idx
        if self._eval_queries is None:
            return torch.full((B, 3), pad, dtype=torch.long, device=self.device)
        
        # Initialize slot counters
        self._slot_ep_counts = torch.zeros(B, dtype=torch.long, device=self.device)
        
        # Fetch first query for each slot
        valid = self._per_slot_lengths > 0
        q = torch.full((B, 3), pad, dtype=torch.long, device=self.device)
        if valid.any():
            idx = self._per_slot_offsets[valid].clamp(max=self._eval_queries.shape[0] - 1)
            fetched = self._eval_queries[idx]
            q[valid] = fetched.squeeze(1) if fetched.ndim == 3 else fetched
        return q


    def _make_combined_td(self, obs: TensorDict, state: EnvState) -> TensorDict:
        B = self.batch_size
        combined = TensorDict({}, batch_size=[B], device=self.device)
        for k, v in obs.items(): combined[k] = v
        for k, v in state.items():
            if k not in combined.keys() and k not in {'done', 'success', 'step_dones', 'is_success'}:
                combined[k] = v
        combined['done'] = state.get('done', torch.zeros(B, dtype=torch.uint8, device=self.device)).bool()
        combined['is_success'] = state.get('success', torch.zeros(B, dtype=torch.uint8, device=self.device)).bool()
        return combined

    def reset(self, queries: Optional[Union[torch.Tensor, TensorDict]] = None) -> TensorDict:

        """Full reset. Partial resets are handled by _step_and_reset_core in step_and_maybe_reset."""
        # Extract queries from TensorDict if needed (for compatibility)
        if isinstance(queries, TensorDict):
            queries = None  # Ignore _reset mask, use slot queries
        
        if queries is None and self._eval_queries is not None:
            queries = self._get_slot_queries(None)
        
        obs, self._state = self.env.reset(queries)
        return self._make_combined_td(obs, self._state)

    def step(self, actions) -> TensorDict:
        act = actions['action'] if isinstance(actions, TensorDict) else actions
        obs, self._state = self.env.step(self._state, act, auto_reset=False)
        combined = self._make_combined_td(obs, self._state)
        combined['reward'], combined['done'] = self._state['step_rewards'], self._state['step_dones'].bool()
        return combined

    def step_and_maybe_reset(self, action_td: TensorDict) -> Tuple[TensorDict, TensorDict]:
        """Fused step + auto-reset using _step_and_reset_core for both training and eval.
        
        - Training: Uses flat query_pool with round-robin or random selection
        - Eval: Uses slot-based scheduling via slot_lengths/slot_offsets
        """
        action = action_td.get('action')
        
        # Determine query pool and pointer source
        if self._eval_queries is not None:
            # Eval mode: slot-based scheduling
            query_pool = self._eval_queries
            per_env_ptrs = self._slot_ep_counts if self._slot_ep_counts is not None else torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
            slot_lengths = self._per_slot_lengths
            slot_offsets = self._per_slot_offsets
        elif self.env._query_pool is not None:
            # Training mode: flat pool
            query_pool = self.env._query_pool
            per_env_ptrs = self.env._per_env_ptrs
            slot_lengths = None
            slot_offsets = None
        else:
            # Fallback: step without auto-reset
            res = self.step(action_td)
            return res, res
        
        obs, self._state = self.env._step_and_reset_core(
            self._state, action, query_pool, per_env_ptrs,
            slot_lengths=slot_lengths, slot_offsets=slot_offsets,
        )
        
        # Update pointers
        if self._slot_ep_counts is not None:
            self._slot_ep_counts = self._state['per_env_ptrs']
        
        res = self._make_combined_td(obs, self._state)
        res['reward'], res['done'] = self._state['step_rewards'], self._state['step_dones'].bool()
        
        nxt = self._make_combined_td(obs, self._state)
        nxt['done'] = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        nxt['is_success'] = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        
        return res, nxt


def _setup_parity_eval_step(ppo, compile_mode: str = 'default'):
    """Setup compiled step for evaluate_parity.
    
    Args:
        ppo: PPO instance
        compile_mode: 'default' for compiled without CUDA graphs,
                     'eager' for no compilation (parity tests),
                     'max-autotune' for aggressive optimization
    
    NOTE: 'reduce-overhead' is NOT supported for evaluate_parity because
    the algorithm has data-dependent Python branches (if finished.any():)
    that are incompatible with CUDA graph recording. Use 'default' instead.
    """
    cache_key = f'_parity_step_{compile_mode}'
    if hasattr(ppo, cache_key):
        return (getattr(ppo, cache_key), 
                getattr(ppo, f'_parity_reset_{compile_mode}'),
                getattr(ppo, f'_parity_loop_step_{compile_mode}'))
        
    policy = ppo._uncompiled_policy
    main_env = ppo.env.env if hasattr(ppo.env, 'env') and not isinstance(ppo.env, EnvVec) else ppo.env
    device = ppo.device
    B = ppo.fixed_batch_size
    env = main_env
    
    # Create labels tensor once
    reset_labels = torch.ones(B, dtype=torch.long, device=device)
    
    def parity_step_fn(obs, state):
        """Fused policy + env step."""
        logits = policy.get_logits(obs)
        masked = logits.masked_fill(obs['action_mask'] == 0, -3.4e38)
        actions = masked.argmax(dim=-1)
        log_probs = torch.log_softmax(masked, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
        new_obs, new_state = env._step_core(state, actions)
        return new_obs, new_state, actions, log_probs
    
    def parity_reset_fn(queries):
        """Reset from queries."""
        state = env._reset_from_queries(queries, reset_labels)
        return env._state_to_obs(state), state
    
    def parity_loop_step_fn(obs, state, ep_logprob, ep_count, targets, 
                            logps_buf, success_buf, mask_buf,
                            flat_queries, per_slot_offsets, slot_lengths_tensor, pad):
        """One iteration of the main eval loop - maximally fused."""
        # Policy + step
        logits = policy.get_logits(obs)
        masked = logits.masked_fill(obs['action_mask'] == 0, -3.4e38)
        actions = masked.argmax(dim=-1)
        log_probs = torch.log_softmax(masked, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
        new_obs, new_state = env._step_core(state, actions)
        
        # Read done/success from new state
        done_curr = new_state['step_dones'].view(-1).bool()
        success_curr = new_state['success'].view(-1).bool()
        
        # Accumulate log probs
        new_ep_logprob = ep_logprob + log_probs
        
        # Check which slots finished an episode
        unfinished = ep_count < targets
        finished_this_step = done_curr & unfinished
        
        return (new_obs, new_state, new_ep_logprob, done_curr, success_curr, 
                finished_this_step)
    
    if compile_mode == 'eager' or ppo.parity:
        step_fn = parity_step_fn
        reset_fn = parity_reset_fn
        loop_step_fn = parity_loop_step_fn
        print(f"[PPOOptimal] Parity eval step ready (eager)")
    else:
        # Use 'default', 'max-autotune', etc. (not 'reduce-overhead')
        step_fn = torch.compile(parity_step_fn, mode=compile_mode, fullgraph=True)
        reset_fn = torch.compile(parity_reset_fn, mode=compile_mode, fullgraph=True)
        loop_step_fn = torch.compile(parity_loop_step_fn, mode=compile_mode, fullgraph=True)
        print(f"[PPOOptimal] Parity eval step compiled (mode={compile_mode})")
    
    setattr(ppo, cache_key, step_fn)
    setattr(ppo, f'_parity_reset_{compile_mode}', reset_fn)
    setattr(ppo, f'_parity_loop_step_{compile_mode}', loop_step_fn)
    return step_fn, reset_fn, loop_step_fn

@torch.inference_mode()
def evaluate_parity(
    ppo,
    queries: torch.Tensor,
    sampler,
    n_corruptions: int = 10,
    corruption_modes: Sequence[str] = ('head', 'tail'),
    query_depths: Optional[torch.Tensor] = None,
    verbose: bool = False,
    deterministic: bool = True,
    compile_mode: str = 'default',
) -> Dict[str, Any]:
    """
    Parity evaluation with eval_corruptions-style slot management.
    """
    # Setup compiled functions
    parity_step_fn, parity_reset_fn, parity_loop_step_fn = _setup_parity_eval_step(ppo, compile_mode)
    
    device = ppo.device
    env = ppo.env.env if hasattr(ppo.env, 'env') and not isinstance(ppo.env, EnvVec) else ppo.env
    B = ppo.fixed_batch_size
    pad = env.padding_idx
    
    if queries.ndim == 2:
        queries = queries.unsqueeze(1)
    N = queries.shape[0]
    A, D = queries.shape[1], queries.shape[2]
    
    # RNG for tie-breaking parity
    rng = np.random.RandomState(0)
    
    per_mode_ranks: Dict[str, List[torch.Tensor]] = {m: [] for m in corruption_modes}
    
    ppo._uncompiled_policy.eval()
    
    for start in range(0, N, B):
        if verbose:
            print(f"Processing batch {start}/{N}")
        Q = min(B, N - start)
        pos = queries[start:start + Q].to(device)
        pos_triples = pos.squeeze(1) if (A == 1 and D == 3) else pos

        # Generate corruptions for this batch (head-then-tail order for RNG parity)
        sampler_mode = getattr(sampler, 'default_mode', 'both')
        
        def extract_valid(result_tensor):
            return [result_tensor[i][result_tensor[i].sum(-1) != 0] for i in range(result_tensor.shape[0])]
        
        head_corrs_list = [torch.empty((0, 3), dtype=pos_triples.dtype, device=device) for _ in range(Q)]
        tail_corrs_list = [torch.empty((0, 3), dtype=pos_triples.dtype, device=device) for _ in range(Q)]
        
        if n_corruptions > 0:
            K = int(n_corruptions)
            if sampler_mode in ('head', 'both'):
                head_result = sampler.corrupt(pos_triples, num_negatives=K, mode='head', device=device)
                head_corrs_list = extract_valid(head_result)
            if sampler_mode in ('tail', 'both'):
                tail_result = sampler.corrupt(pos_triples, num_negatives=K, mode='tail', device=device)
                tail_corrs_list = extract_valid(tail_result)

        for mode in corruption_modes:
            if verbose:
                print(f"Processing mode {mode}")
            corrs_list = head_corrs_list if mode == "head" else tail_corrs_list
            
            # Build ragged lists and per-slot lengths
            ragged_lists = []
            lengths_i = []
            for i in range(Q):
                neg_tensor = corrs_list[i]
                if neg_tensor.numel() == 0:
                    ragged_lists.append(torch.empty((0, 3), dtype=pos.dtype, device=device))
                    lengths_i.append(0)
                else:
                    if A == 1 and D == 3 and neg_tensor.ndim == 2:
                        neg_tensor = neg_tensor.unsqueeze(1)
                    ragged_lists.append(neg_tensor.to(device))
                    flat_neg = neg_tensor.view(neg_tensor.shape[0], -1)
                    is_valid = (flat_neg != 0).all(dim=1)
                    lengths_i.append(int(is_valid.sum().item()))
            
            per_slot_lengths = [1 + li for li in lengths_i]  # +1 for positive
            
            # Construct flat queries: [pos_0, neg_0_0, ..., neg_0_k, pos_1, neg_1_0, ...]
            flat_queries_parts = []
            for i in range(Q):
                flat_queries_parts.append(pos[i].unsqueeze(0))
                if lengths_i[i] > 0:
                    flat_queries_parts.append(ragged_lists[i][:lengths_i[i]])
            flat_queries = torch.cat(flat_queries_parts, dim=0)
            total_queries = flat_queries.shape[0]
            
            # Build per_slot_lengths tensor padded to B
            slot_lengths_tensor = torch.tensor(per_slot_lengths + [0] * (B - Q), dtype=torch.long, device=device)
            targets = slot_lengths_tensor.clone()
            
            # Compute offsets for slot-based query indexing
            cumsum = torch.cumsum(slot_lengths_tensor, dim=0)
            per_slot_offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device), cumsum[:-1]])
            
            # Allocate result buffers
            T = int(targets.max().item()) if targets.numel() > 0 else 0
            logps_buf = torch.zeros((B, T), dtype=torch.float32, device=device)
            success_buf = torch.zeros((B, T), dtype=torch.float32, device=device)
            mask_buf = torch.zeros((B, T), dtype=torch.bool, device=device)
            
            # Episode tracking
            ep_count = torch.zeros(B, dtype=torch.long, device=device)
            ep_logprob = torch.zeros(B, dtype=torch.float32, device=device)
            
            # Get initial queries for each slot
            def get_slot_queries(ep_counts: torch.Tensor) -> torch.Tensor:
                """Get query for each slot based on current episode count."""
                safe_counts = ep_counts.clamp(max=slot_lengths_tensor.clamp(min=1) - 1).clamp(min=0)
                query_idx = per_slot_offsets + safe_counts
                q = torch.full((B, 3), pad, dtype=torch.long, device=device)
                valid = ep_counts < slot_lengths_tensor
                if valid.any():
                    idx = query_idx[valid].clamp(max=total_queries - 1)
                    fetched = flat_queries[idx]
                    q[valid] = fetched.squeeze(1) if fetched.ndim == 3 else fetched
                return q
            
            # Initial reset
            init_queries = get_slot_queries(ep_count)
            obs, state = parity_reset_fn(init_queries)
            
            # Main evaluation loop - vectorized episode tracking
            max_steps = T * ppo.max_depth * 2  # Conservative bound
            
            for step_idx in range(max_steps):
                # Check termination condition
                unfinished = ep_count < targets
                if not unfinished.any():
                    break
                
                # Fused policy + step + metrics (compiled or eager based on compile_mode)
                (obs, state, ep_logprob, done_curr, success_curr, 
                 finished_this_step) = parity_loop_step_fn(
                    obs, state, ep_logprob, ep_count, targets,
                    logps_buf, success_buf, mask_buf,
                    flat_queries, per_slot_offsets, slot_lengths_tensor, pad
                )
                
                # Handle done episodes - record results and optionally reset
                if finished_this_step.any():
                    rows = finished_this_step.nonzero(as_tuple=False).view(-1)
                    curr_ep_idx = ep_count[rows]
                    
                    logps_buf[rows, curr_ep_idx] = ep_logprob[rows]
                    success_buf[rows, curr_ep_idx] = success_curr[rows].float()
                    mask_buf[rows, curr_ep_idx] = True
                    
                    # Update counts and reset accumulators
                    ep_count[rows] = ep_count[rows] + 1
                    ep_logprob[rows] = 0.0
                    
                    # Reset logic - only reset slots that are done AND still need more episodes
                    still_unfinished = ep_count < targets
                    reset_mask = done_curr & still_unfinished
                    
                    if reset_mask.any():
                        # Get next queries for slots that need reset
                        next_queries = get_slot_queries(ep_count)
                        reset_obs, reset_state = parity_reset_fn(next_queries)
                        
                        # Merge reset state into current state for reset slots
                        for k in reset_state.keys():
                            if k in state.keys() and state[k].shape == reset_state[k].shape:
                                state[k] = torch.where(reset_mask.view(-1, *([1]*(state[k].ndim-1))), 
                                                      reset_state[k], state[k])
                        for k in reset_obs.keys():
                            if k in obs.keys() and obs[k].shape == reset_obs[k].shape:
                                obs[k] = torch.where(reset_mask.view(-1, *([1]*(obs[k].ndim-1))),
                                                    reset_obs[k], obs[k])
            
            # Compute ranks using log probabilities (same logic as eval_corruptions)
            logps_out = logps_buf
            msk = mask_buf
            success_out = success_buf
            
            # Apply success penalty
            success_mask = success_out.bool()
            logps_p = logps_out.clone()
            logps_p[~success_mask] -= 100.0
            
            # Vectorized ranking with RNG tie-breaking
            Tmax = logps_out.shape[1]
            
            # Generate random keys for tie-breaking (same shape as eval_corruptions)
            full_rnd = rng.rand(Q, 1 + int(n_corruptions))
            used_rnd = full_rnd[:, :Tmax]
            batch_random_keys = torch.as_tensor(used_rnd, device=device, dtype=torch.float32)
            
            logps_Q = logps_p[:Q]
            msk_Q = msk[:Q]
            rnd_Q = batch_random_keys
            
            pos_logp = logps_Q[:, 0].unsqueeze(1)  # [Q, 1]
            pos_rnd = rnd_Q[:, 0].unsqueeze(1)     # [Q, 1]
            
            is_better = (logps_Q > pos_logp)
            is_equal = (logps_Q == pos_logp)
            is_tie = (rnd_Q > pos_rnd)
            
            better_count = (is_better & msk_Q).sum(dim=1)
            tie_count = (is_equal & is_tie & msk_Q).sum(dim=1)
            
            ranks = 1 + better_count + tie_count
            per_mode_ranks[mode].append(ranks)
            
            if verbose:
                print(f"[batch {start // B:03d} | mode={mode}] Q={Q} mean_rank={ranks.float().mean().item():.2f} MRR={torch.mean(1.0 / ranks.float()).item():.3f}")
    
    ppo._uncompiled_policy.train()
    
    # Aggregate metrics
    def compute_metrics(ranks: torch.Tensor) -> Dict[str, float]:
        if ranks.numel() == 0:
            return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}
        r = ranks.float()
        return {
            "MRR": float((1.0 / r).mean().item()),
            "Hits@1": float((r <= 1.0).float().mean().item()),
            "Hits@3": float((r <= 3.0).float().mean().item()),
            "Hits@10": float((r <= 10.0).float().mean().item()),
        }
    
    agg = {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}
    per_mode_results = {}
    
    for m, rank_list in per_mode_ranks.items():
        if rank_list:
            all_ranks_tensor = torch.cat(rank_list)
            per_mode_results[m] = compute_metrics(all_ranks_tensor)
        else:
            per_mode_results[m] = compute_metrics(torch.tensor([], device=device))
    
    for m in corruption_modes:
        for k, v in per_mode_results[m].items():
            agg[k] += v
    for k in agg:
        agg[k] /= float(len(corruption_modes)) if corruption_modes else 1.0
    
    agg["per_mode"] = per_mode_results
    agg["_mrr"] = agg["MRR"]
    
    return agg
