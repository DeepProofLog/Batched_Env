"""
Model Evaluation for Neural-Guided Logical Reasoning.

This module provides evaluation utilities for tensor-based PPO agents:
- evaluate_policy(): Single-pass policy evaluation with batched envs
- eval_corruptions(): Ranking-based MRR evaluation with head/tail corruptions

Key tensor shapes:
    rewards:  (B, T) - per-env, per-episode rewards
    lengths:  (B, T) - per-env, per-episode lengths
    success:  (B, T) - per-env, per-episode success flags
    mask:     (B, T) - valid episode mask
"""
from __future__ import annotations
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.envs import EnvBase

from callbacks import _format_stat_string, _format_depth_key

DEBUG_EVAL = os.environ.get("NGG_DEBUG_EVAL", "").lower() in {"1", "true", "yes"}


# ------------------------------------------------------------
# Trace Types for detailed step-by-step comparison
# ------------------------------------------------------------

class EvalStepTrace(TypedDict, total=False):
    """Trace of a single step in evaluate_policy."""
    step: int                    # Global step index
    env_idx: int                 # Environment index in the batch
    episode_idx: int             # Episode index for this env
    action: int                  # Action taken
    reward: float                # Reward received
    done: bool                   # Whether episode ended
    success: bool                # Whether proof succeeded (if applicable)
    log_prob: float              # Log probability of the action
    cumulative_reward: float     # Cumulative reward so far in episode
    episode_length: int          # Steps so far in episode
    state_obs: Dict[str, Any]    # State observation (sub_index, derived_states, etc.)
    next_state_obs: Dict[str, Any]  # Next state observation
    value: float                 # Value estimate (if available)
    query: List[int]             # Query tensor [rel, head, tail]


class EvalCorruptionsTrace(TypedDict, total=False):
    """Trace of eval_corruptions evaluation."""
    batch_idx: int               # Batch index
    mode: str                    # Corruption mode (head/tail)
    query_idx: int               # Query index within batch
    query: List[int]             # Query tensor [rel, head, tail]
    negatives: List[List[int]]   # Generated negative samples
    num_negatives: int           # Number of negatives generated
    pos_logp: float              # Log probability for positive
    neg_logps: List[float]       # Log probabilities for negatives
    pos_success: bool            # Whether positive proof succeeded
    neg_successes: List[bool]    # Whether negative proofs succeeded
    rank: int                    # Rank of the positive query
    episode_traces: List[EvalStepTrace]  # Detailed traces per episode


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------

def infer_batch_size(env: EnvBase) -> int:
    bs = getattr(env, "batch_size", None)
    if bs is None:
        td0 = env.reset()
        return int(td0.batch_size[0])
    if isinstance(bs, torch.Size):
        return int(bs[0]) if len(bs) else 1
    return int(bs)


# ------------------------------------------------------------
# Policy evaluation — single batched env
# ------------------------------------------------------------

@torch.inference_mode()
def evaluate_policy(
    actor: nn.Module,
    env:  EnvBase,
    *,
    n_eval_episodes: Optional[int] = None,
    target_episodes: Optional[Sequence[int]] = None,
    deterministic: bool = True,
    track_logprobs: bool = False,
    collect_action_stats: bool = False,
    info_callback: Optional[Callable[[TensorDict], None]] = None,
    verbose: int = 0,
    return_traces: bool = False,
    return_on_device: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a policy on a single TorchRL env with internal batch dimension.
    
    This function runs the policy in the environment until the target number of episodes
    is reached for each batch slot. It handles partial resets and collects metrics
    per episode.
    
    Args:
        actor (nn.Module): Policy network.
        env (EnvBase): Batched TorchRL environment.
        n_eval_episodes (Optional[int]): Total episodes to run per slot (if target_episodes is None).
        target_episodes (Optional[Sequence[int]]): [B] Exact target episode counts per slot.
        deterministic (bool): If True, use mode of action distribution.
        track_logprobs (bool): Whether to return log probabilities (always tracked internally).
        collect_action_stats (bool): Unused, kept for API compatibility.
        info_callback (Optional[Callable]): Function called with step_td at each step.
        verbose (int): Verbosity level.
        return_traces (bool): If True, return detailed step-by-step execution traces.
        return_on_device (bool): If True, return tensors on the device (avoids CPU transfer).
        
    Returns:
        Dict[str, Any]: Dictionary containing:
            - "rewards" (Tensor): [B, T] Total reward per episode.
            - "lengths" (Tensor): [B, T] Length of each episode.
            - "logps" (Tensor):   [B, T] Sum of log probabilities per episode.
            - "success" (Tensor): [B, T] Success flag (1.0 or 0.0) per episode.
            - "mask" (Tensor):    [B, T] Boolean mask indicating valid episodes.
            - "traces" (List):    Optional list of execution traces.
    """
    device = getattr(env, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    B = infer_batch_size(env)
    
    if target_episodes is not None:
        targets = torch.as_tensor(target_episodes, dtype=torch.long, device=device)
        if targets.numel() != B:
             # Resize to match batch
             targets = torch.cat([targets, torch.zeros(max(0, B - targets.numel()), dtype=torch.long, device=device)])[:B]
    else:
        targets = torch.full((B,), n_eval_episodes if n_eval_episodes is not None else 1, dtype=torch.long, device=device)

    actor_was_training = actor.training
    actor.eval()
    
    verbose_level = max(int(verbose), 1 if DEBUG_EVAL else 0)
    T = int(targets.max().item()) if targets.numel() > 0 else 0
    # Pre-allocate result buffers
    rewards  = torch.zeros((B, T), dtype=torch.float32, device=device)
    lengths  = torch.zeros((B, T), dtype=torch.long,   device=device)
    logps    = torch.zeros((B, T), dtype=torch.float32, device=device)
    success  = torch.zeros((B, T), dtype=torch.float32, device=device)
    mask     = torch.zeros((B, T), dtype=torch.bool,    device=device)

    # Episode tracking
    ep_count   = torch.zeros(B, dtype=torch.long, device=device)
    ep_return  = torch.zeros(B, dtype=torch.float32, device=device)
    ep_length  = torch.zeros(B, dtype=torch.long,   device=device)
    ep_logprob = torch.zeros(B, dtype=torch.float32, device=device)

    # Trace collection
    traces: List[EvalStepTrace] = []
    global_step = 0

    td = env.reset().to(device, non_blocking=True)
    
    # Helper for trace extraction
    def _extract_state_obs(td_in: TensorDict, idx: int) -> Dict[str, Any]:
        obs = {}
        for key in ["sub_index", "derived_states", "derived_sub_indices", "action_mask", "query", "n_derived"]:
            val = td_in.get(key, None)
            if val is not None:
                # Handle scalar vs tensor
                item = val[idx]
                obs[key] = item.cpu().tolist() if item.dim() > 0 else item.item()
        return obs

    if verbose_level > 0:
        init_done = td.get("done", torch.zeros(B, 1, dtype=torch.bool, device=device)).view(-1).tolist()
        print("[evaluate_policy] initial done flags:", init_done)

    # Main evaluation loop
    while True:
        unfinished = ep_count < targets
        if not unfinished.any():
            break

        if verbose_level > 0:
             print(f"\r[eval] ep_counts: {ep_count.tolist()}", end="", flush=True)

        pre_step_obs = {}
        if return_traces:
            for env_idx in range(B):
                if unfinished[env_idx]:
                    pre_step_obs[env_idx] = _extract_state_obs(td, env_idx)

        # Forward pass
        policy_td = td.clone() 
        out = actor(policy_td, deterministic=deterministic)

        # Extract outputs
        if isinstance(out, tuple):
            action = out[0].view(-1).long()
            log_probs = out[2].view(-1) if len(out) > 2 else torch.zeros(B, device=device)
            value_estimates = out[1].view(-1) if len(out) > 1 and out[1] is not None else torch.zeros(B, device=device)
        elif isinstance(out, TensorDict):
            action = out.get("action").view(-1).long()
            log_probs = out.get("sample_log_prob", torch.zeros(B, device=device)).view(-1)
            value_estimates = out.get("state_value", torch.zeros(B, device=device)).view(-1)
        else:
            action = policy_td.get("action").view(-1).long()
            log_probs = policy_td.get("sample_log_prob", torch.zeros(B, device=device)).view(-1)
            value_estimates = torch.zeros(B, device=device)
        
        # Env Step
        action_td = TensorDict({"action": action}, batch_size=env.batch_size, device=device)
        step_td = env.step(action_td)
        
        # TorchRL wraps step output in a "next" key
        next_td = step_td.get("next", step_td)

        # Gather metrics from the 'next' tensordict
        rew = next_td.get("reward", torch.zeros(B, device=device)).view(-1)
        done_curr = next_td.get("done", torch.zeros(B, 1, dtype=torch.bool, device=device)).view(-1)
        success_curr = next_td.get("is_success", torch.zeros(B, 1, dtype=torch.bool, device=device)).view(-1)
        
        ep_return += rew
        ep_length += 1
        ep_logprob += log_probs
        
        if return_traces:
            for env_idx in range(B):
                if unfinished[env_idx]:
                    traces.append({
                        "step": global_step,
                        "env_idx": env_idx,
                        "episode_idx": int(ep_count[env_idx].item()),
                        "action": int(action[env_idx].item()),
                        "reward": float(rew[env_idx].item()),
                        "done": bool(done_curr[env_idx].item()),
                        "success": bool(success_curr[env_idx].item()),
                        "log_prob": float(log_probs[env_idx].item()),
                        "cumulative_reward": float(ep_return[env_idx].item()),
                        "episode_length": int(ep_length[env_idx].item()),
                        "state_obs": pre_step_obs.get(env_idx, {}),
                        "next_state_obs": _extract_state_obs(next_td, env_idx),
                        "value": float(value_estimates[env_idx].item()),
                        "query": pre_step_obs.get(env_idx, {}).get("query")
                    })
            global_step += 1

        # Handle done episodes
        # finished_this_step: done AND was properly running (unfinished)
        finished_this_step = done_curr & unfinished
        
        if finished_this_step.any():
            rows = finished_this_step.nonzero(as_tuple=False).view(-1)
            curr_ep_idx = ep_count[rows]
            
            rewards[rows, curr_ep_idx] = ep_return[rows]
            lengths[rows, curr_ep_idx] = ep_length[rows]
            logps[rows, curr_ep_idx] = ep_logprob[rows]
            success[rows, curr_ep_idx] = success_curr[rows].float()
            mask[rows, curr_ep_idx] = True
            
            ep_count[rows] += 1
            # Reset counters
            ep_return[rows] = 0
            ep_length[rows] = 0
            ep_logprob[rows] = 0

        # Reset logic:
        # We need to reset if done AND we still need more episodes for that slot.
        # Recalculate unfinished with new counts
        still_unfinished = ep_count < targets
        reset_mask = done_curr & still_unfinished
        
        # next_td already holds the 'next' observation from step
        if reset_mask.any():
            reset_td = env.reset(TensorDict({"_reset": reset_mask.view(-1, 1)}, batch_size=env.batch_size, device=device))
            # Merge reset_td into next_td where reset_mask is True
            next_obs = next_td.clone()
            for k in reset_td.keys():
                 val_reset = reset_td.get(k)
                 val_next  = next_obs.get(k, None)
                 if val_next is not None:
                     if val_reset.shape == val_next.shape:
                         val_copy = val_next.clone()
                         val_copy[reset_mask] = val_reset[reset_mask]
                         next_obs.set(k, val_copy)
                     else:
                         next_obs.set(k, val_reset)
                 else:
                     next_obs.set(k, val_reset)
            td = next_obs
        else:
            td = next_td

        if info_callback is not None:
            info_callback(step_td)

    if actor_was_training:
        actor.train()

    # Return on CPU for safety unless requested otherwise
    if return_on_device:
        return {
            "rewards": rewards,
            "lengths": lengths,
            "logps":   logps,
            "success": success,
            "mask":    mask,
            "traces": traces
        }
    else:
        return {
            "rewards": rewards.cpu(),
            "lengths": lengths.cpu(),
            "logps":   logps.cpu(),
            "success": success.cpu(),
            "mask":    mask.cpu(),
            "traces": traces
        }

def step_and_maybe_reset(env: EnvBase, action_td: TensorDict, *, reset_rows: torch.Tensor) -> Tuple[TensorDict, TensorDict]:
    """Helper unused in new loop but kept for compatibility logic if needed."""
    step_td = env.step(action_td)
    next_obs = step_td.get("next", step_td)
    if reset_rows.any():
        reset_td = env.reset(TensorDict({"_reset": reset_rows.view(-1, 1)}, batch_size=env.batch_size))
        # Merge logic
        for k in reset_td.keys():
            if k in next_obs.keys():
                v_next = next_obs.get(k)
                v_reset = reset_td.get(k)
                if v_next.shape == v_reset.shape:
                     v_next[reset_rows] = v_reset[reset_rows]
                else:
                     next_obs.set(k, v_reset)
    return step_td, next_obs

def _merge_rows(dst: TensorDict, src: TensorDict, mask: torch.Tensor) -> TensorDict:
    # Deprecated/Inlined
    return dst


# ------------------------------------------------------------
# Ranking metrics — SB3-style aggregation
# ------------------------------------------------------------

@torch.inference_mode()
def eval_corruptions(
    actor: nn.Module,
    env:   EnvBase,
    *,
    queries: torch.Tensor,
    sampler: Any,
    query_depths: Optional[torch.Tensor] = None,
    n_corruptions: Optional[int] = 10,
    corruption_modes: Sequence[str] = ("head", "tail"),
    deterministic: bool = True,
    verbose: bool = False,
    info_callback: Optional[Callable] = None,
    return_traces: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a model by ranking a positive query vs. its corruptions.
    
    This function implements the standard Mean Reciprocal Rank (MRR) and Hits@K
    evaluation protocol. For each test query (h, r, t), it generates K negatives,
    evaluates the policy on all of them, and computes the rank of the positive query.
    
    Optimized with vectorized ranking and metric aggregation.
    """
    env_device = getattr(env, "_device", None)
    actor_device = None
    try:
        actor_device = next(actor.parameters()).device
    except StopIteration:
        actor_device = None
    device = env_device or actor_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = infer_batch_size(env)
    if queries.ndim == 2:
        queries = queries.unsqueeze(1)
    elif queries.ndim != 3:
        raise ValueError(f"Expected queries with 2 or 3 dims, got shape {tuple(queries.shape)}")
    N = int(queries.shape[0])
    A, D = int(queries.shape[1]), int(queries.shape[2])
    
    # Trace collection
    all_traces: List[EvalCorruptionsTrace] = [] if return_traces else []

    def compute_metrics(ranks: torch.Tensor) -> Dict[str, float]:
        if ranks.numel() == 0:
            return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}
        ranks_float = ranks.float()
        return {
            "MRR": float(torch.mean(1.0 / ranks_float).item()),
            "Hits@1": float(torch.mean((ranks_float <= 1.0).float()).item()),
            "Hits@3": float(torch.mean((ranks_float <= 3.0).float()).item()),
            "Hits@10": float(torch.mean((ranks_float <= 10.0).float()).item()),
        }

    per_mode_ranks: Dict[str, List[torch.Tensor]] = {m: [] for m in corruption_modes}
    
    # Accumulators for aggregate stats
    acc_lengths: List[torch.Tensor] = []
    acc_rewards: List[torch.Tensor] = []
    acc_success: List[torch.Tensor] = []
    acc_is_pos: List[torch.Tensor] = []
    acc_depths: List[torch.Tensor] = [] 

    actor_was_training = actor.training
    actor.eval()
    
    # Pre-generate random state for tie-breaking parity
    rng = np.random.RandomState(0)

    for start in range(0, N, B):
        if verbose:
            print(f"Processing batch {start}/{N}")
        Q = min(B, N - start)
        pos = queries[start:start + Q].to(device)
        pos_triples = pos.squeeze(1) if (A == 1 and D == 3) else pos

        # Generate corruptions
        if n_corruptions == 0:
            head_corrs_list = [torch.empty((0, 3), dtype=pos_triples.dtype, device=device) for _ in range(Q)]
            tail_corrs_list = [torch.empty((0, 3), dtype=pos_triples.dtype, device=device) for _ in range(Q)]
        elif hasattr(sampler, 'get_negatives_from_states_separate'):
            head_corrs_list, tail_corrs_list = sampler.get_negatives_from_states_separate(
                 pos_triples, num_negatives=n_corruptions, device=device
            )
        else:
            if n_corruptions is None:
                head_corrs_list, _ = sampler.corrupt_all(pos_triples, mode='head')
                _, tail_corrs_list = sampler.corrupt_all(pos_triples, mode='tail')
            else:
                K = int(n_corruptions)
                head_neg = sampler.corrupt(pos_triples, num_negatives=K, mode='head').to(device)
                tail_neg = sampler.corrupt(pos_triples, num_negatives=K, mode='tail').to(device)
                head_corrs_list = [head_neg[i] for i in range(Q)]
                tail_corrs_list = [tail_neg[i] for i in range(Q)]

        for mode in corruption_modes:
            if verbose:
                print(f"Processing mode {mode}")
            corrs_list = head_corrs_list if mode == "head" else tail_corrs_list
            
            # Prepare ragged lists
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
                    # Valid count logic
                    flat_neg = neg_tensor.view(neg_tensor.shape[0], -1)
                    is_valid = (flat_neg != 0).all(dim=1)
                    lengths_i.append(int(is_valid.sum().item()))

            per_slot_lengths = [1 + li for li in lengths_i]
            
            # Construct flat queries
            flat_queries_parts = []
            for i in range(Q):
                flat_queries_parts.append(pos[i].unsqueeze(0))
                if lengths_i[i] > 0:
                    flat_queries_parts.append(ragged_lists[i][:lengths_i[i]])
            flat_queries = torch.cat(flat_queries_parts, dim=0)

            # Labels and depths construction
            labels_list = []
            for l in per_slot_lengths:
                labels_list.append(torch.tensor([1] + [0]*(l-1), dtype=torch.long, device=device))
            flat_labels = torch.cat(labels_list)
            
            if query_depths is not None:
                pos_depths = query_depths[start:start + Q].to(device)
                flat_depths_parts = []
                for i in range(Q):
                    d = pos_depths[i]
                    part = torch.cat([
                        d.view(1), 
                        torch.full((per_slot_lengths[i]-1,), -1, dtype=torch.long, device=device)
                    ])
                    flat_depths_parts.append(part)
                flat_depths = torch.cat(flat_depths_parts)
            else:
                flat_depths = torch.full((flat_queries.shape[0],), -1, dtype=torch.long, device=device)

            slot_lengths_tensor = torch.tensor(per_slot_lengths + [0]*(B-Q), dtype=torch.long, device=device)
            
            env.set_eval_dataset(
                queries=flat_queries,
                labels=flat_labels,
                query_depths=flat_depths,
                per_slot_lengths=slot_lengths_tensor,
            )

            # Evaluate
            out = evaluate_policy(
                actor, env, target_episodes=slot_lengths_tensor.tolist(),
                deterministic=deterministic, track_logprobs=True,
                info_callback=info_callback, return_traces=return_traces, verbose=verbose,
                return_on_device=True 
            )
            
            # Post-processing on Device
            logps_out = out["logps"]   # [B, Tmax] (already on device)
            msk = out["mask"]          # [B, Tmax] (already on device)
            success = out.get("success") if out.get("success") is not None else None
            
            if success is not None:
                success_mask = success.bool()
                logps_p = logps_out.clone()
                logps_p[~success_mask] -= 100.0
            else:
                success_mask = None
                logps_p = logps_out

            # --- Vectorized Ranking ---
            Tmax = logps_out.shape[1]
            batch_random_keys = torch.as_tensor(rng.rand(Q, Tmax), device=device, dtype=torch.float32)

            logps_Q = logps_p[:Q]
            msk_Q = msk[:Q]
            rnd_Q = batch_random_keys
            
            pos_logp = logps_Q[:, 0].unsqueeze(1) # [Q, 1]
            pos_rnd  = rnd_Q[:, 0].unsqueeze(1)   # [Q, 1]
            
            is_better = (logps_Q > pos_logp)
            is_equal  = (logps_Q == pos_logp)
            is_tie    = (rnd_Q > pos_rnd)
            
            better_count = (is_better & msk_Q).sum(dim=1)
            tie_count    = (is_equal & is_tie & msk_Q).sum(dim=1)
            
            ranks = 1 + better_count + tie_count
            per_mode_ranks[mode].append(ranks)

            # --- Vectorized Stats Accumulation ---
            rewards_out = out["rewards"][:Q] # [Q, Tmax] (on device)
            lengths_out = out["lengths"][:Q] # [Q, Tmax] (on device)
            
            pos_mask = torch.zeros_like(msk_Q, dtype=torch.bool)
            pos_mask[:, 0] = True
            pos_mask = pos_mask & msk_Q
            
            acc_lengths.append(lengths_out[msk_Q])
            acc_rewards.append(rewards_out[msk_Q])
            if success is not None:
                acc_success.append(success[:Q][msk_Q])
            
            # Flatten positive mask relative to flattened valid items
            acc_is_pos.append(pos_mask[msk_Q])
            
            if query_depths is not None:
                 # Expand depth
                 d_vals = pos_depths.view(Q, 1).expand(Q, Tmax).clone()
                 d_vals[~pos_mask] = -1
                 acc_depths.append(d_vals[msk_Q])
            
            if verbose:
                 print(f"[batch {start//B:03d} | mode={mode}] Q={Q} mean_rank={ranks.float().mean().item():.2f} MRR={torch.mean(1.0/ranks.float()).item():.3f}")

            if return_traces:
                # Add dummy trace handling or re-implement if needed. 
                # Keeping compatibility by adding empty logic or limited traces.
                pass

    if actor_was_training:
        actor.train()
    
    # Finalize Metrics
    agg = {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}
    
    per_mode_results = {}
    for m, rank_list in per_mode_ranks.items():
        if rank_list:
             all_ranks = torch.cat(rank_list)
             per_mode_results[m] = compute_metrics(all_ranks)
        else:
             per_mode_results[m] = compute_metrics(torch.tensor([], device=device))
    
    for m in corruption_modes:
        for k, v in per_mode_results[m].items():
            agg[k] += v
    for k in agg:
        agg[k] /= float(len(corruption_modes)) if corruption_modes else 1.0
    agg["per_mode"] = per_mode_results
    agg["_mrr"] = agg["MRR"]

    if acc_lengths:
        all_lens = torch.cat(acc_lengths).float()
        all_rews = torch.cat(acc_rewards).float()
        all_succ = torch.cat(acc_success).float() if acc_success else None
        all_pos  = torch.cat(acc_is_pos)
        all_depths = torch.cat(acc_depths) if acc_depths else None
        
        def fmt(t):
            return _format_stat_string(t.mean().item(), t.std().item(), t.numel())
            
        agg["length mean +/- std"] = fmt(all_lens)
        agg["ep_len_mean"] = getattr(all_lens.mean(), 'item', lambda: 0.0)()
        
        pos_idxs = torch.nonzero(all_pos).view(-1)
        neg_idxs = torch.nonzero(~all_pos).view(-1)
        
        if pos_idxs.numel() > 0:
            agg["reward_overall"] = fmt(all_rews[pos_idxs])
            agg["ep_rew_mean"] = getattr(all_rews.mean(), 'item', lambda: 0.0)() 
            if all_succ is not None:
                agg["success_rate"] = getattr(all_succ[pos_idxs].mean(), 'item', lambda: 0.0)()
        
        for lbl_bool, lbl_key, idxs in [(True, "pos", pos_idxs), (False, "neg", neg_idxs)]:
            if idxs.numel() > 0:
                agg[f"len_{lbl_key}"] = fmt(all_lens[idxs])
                agg[f"reward_label_{lbl_key}"] = fmt(all_rews[idxs])
                if all_succ is not None:
                    agg[f"proven_{lbl_key}"] = fmt(all_succ[idxs])

        if all_depths is not None:
            # We iterate unique depths present
            unique_d = torch.unique(all_depths)
            for d in unique_d:
                d_val = int(d.item())
                # If d_val is -1, it's negative (unless we had depth=-1 in queries)
                
                mask_d = (all_depths == d)
                for is_p, lbl in [(True, "pos"), (False, "neg")]:
                    mask_dp = mask_d & (all_pos if is_p else ~all_pos)
                    if mask_dp.any():
                        depth_key = _format_depth_key(d_val if is_p else -1)
                        agg[f"len_d_{depth_key}_{lbl}"] = fmt(all_lens[mask_dp])
                        agg[f"reward_d_{depth_key}_{lbl}"] = fmt(all_rews[mask_dp])
                        if all_succ is not None:
                            agg[f"proven_d_{depth_key}_{lbl}"] = fmt(all_succ[mask_dp])

    if return_traces:
        agg["traces"] = all_traces

    return agg
