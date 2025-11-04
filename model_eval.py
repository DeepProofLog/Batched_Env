"""
model_eval.py — Evaluation utilities for TorchRL-based policies

This module provides evaluation utilities for TorchRL policies trained on
logical reasoning environments, adapted from the SB3-based sb3_model_eval.py.

Key differences from SB3 version:
- Uses TorchRL policy modules instead of SB3 policies
- Works with TorchRL EnvBase instead of VecEnv
- Uses TensorDict for observations instead of dict/numpy arrays
- Direct tensor operations without SB3 wrappers

Main functions:
1) `evaluate_policy(...)` — Fast tensor-based evaluator that rolls out a TorchRL
   policy, collecting per-episode rewards, lengths and log-probabilities.

2) `eval_corruptions(...)` — Higher-level evaluator for link-prediction tasks.
   Generates head/tail corruptions and scores candidates using RL, KGE, or hybrid.

Assumptions:
• The environment is a TorchRL EnvBase with these attributes:
  - `env._episode_target` (np.ndarray[int]): episodes to run
  - `env._episode_count` (np.ndarray[int]): completed episodes
  - `env.active_envs` (np.ndarray[bool]): which envs are active
  - `env.index_manager` (optional, for state string conversion)

• The policy is a TorchRL TensorDictModule that accepts TensorDict and returns
  actions and log probabilities.

• Observations are TensorDict objects.

"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Union, Optional, Callable
import os
import time
import random

import numpy as np
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from tensordict import TensorDict


@torch.inference_mode()
def evaluate_policy(
    actor_module: torch.nn.Module,
    value_module: Optional[torch.nn.Module],
    env: Any,  # TorchRL EnvBase
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    target_episodes: np.ndarray | None = None,
    verbose: int = 0,
    track_logprobs: bool = False,
    info_callback: Optional[Callable[[TensorDict], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Roll out a TorchRL policy and aggregate per-episode stats.
    

    Parameters
    ----------
    actor_module : torch.nn.Module
        TorchRL actor module that accepts TensorDict and outputs actions/log_probs.
    value_module : torch.nn.Module or None
        TorchRL value module (optional, not used in evaluation but kept for compatibility).
    env : TorchRL EnvBase
        Evaluation environment.
    n_eval_episodes : int, default 10
        Total target episodes (if `target_episodes` is not provided).
    deterministic : bool, default True
        Whether to use deterministic policy actions.
    target_episodes : np.ndarray | None, default None
        Optional per-env episode targets (shape `(n_envs,)`).
    verbose : int, default 0
        Print running counter when > 0.
    track_logprobs : bool, default False
        If True, collects per-step log-prob trajectories and state strings.
    info_callback : callable, optional
        Called after each step with the TensorDict containing info.

    Returns
    -------
    rewards : np.ndarray, shape (L, T_max)
        Sum of rewards per finished episode.
    lengths : np.ndarray, shape (L, T_max)
        Episode lengths per slot.
    logps : np.ndarray, shape (L, T_max)
        Accumulated action log-probabilities per episode.
    mask : np.ndarray, shape (L, T_max), dtype=bool
        Mask marking valid episode slots.
    proof_successful : np.ndarray, shape (L, T_max), dtype=bool
        Whether the env reported success for each finished episode.
    """
    # Reference implementation (kept for comparison and debugging)
    device = next(actor_module.parameters()).device
    n_envs = 1  # TorchRL EnvBase is typically single-env, but can be batched
    
    # Check if env has batch_size attribute and handle batched envs
    if hasattr(env, 'batch_size') and len(env.batch_size) > 0:

        n_envs = env.batch_size[0].item() if torch.is_tensor(env.batch_size[0]) else env.batch_size[0]

    # --- targets ---
    if target_episodes is None:
        targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype=int)
    else:
        targets = np.asarray(target_episodes, dtype=int)

    padded_targets = np.zeros(n_envs, dtype=int)
    padded_targets[: len(targets)] = targets
    max_t = int(padded_targets.max())
    total = int(padded_targets.sum())

    if verbose:
        print(f"\nEvaluating {total} episodes on {n_envs} envs (avg target: {targets.mean():.2f})")

    # --- buffers on device ---
    rewards          = torch.zeros((n_envs, max_t), device=device)
    lengths          = torch.zeros_like(rewards, dtype=torch.int32)
    logps            = torch.zeros_like(rewards)
    proof_successful = torch.zeros_like(rewards, dtype=torch.bool)

    counts      = torch.zeros(n_envs, dtype=torch.int32, device=device)
    current_rew = torch.zeros(n_envs, device=device)
    current_len = torch.zeros(n_envs, dtype=torch.int32, device=device)
    current_lp  = torch.zeros(n_envs, device=device)

    # Optional histories for plotting/debug
    episode_logprob_histories    : list[np.ndarray] = []
    episode_choices_histories    : list[np.ndarray] = []
    episode_steplogprob_histories: list[np.ndarray] = []
    episode_state_histories      : list[np.ndarray] = []
    if track_logprobs:
        current_steplogprob_histories: list[list[float]] = [[] for _ in range(n_envs)]
        current_choices_histories: list[list[int]] = [[] for _ in range(n_envs)]
        current_state_histories: list[list[str]] = [[] for _ in range(n_envs)]
        index_manager = env.index_manager

    # Reset environment and get initial observation
    td_reset = env.reset()
    observations = td_reset
    padded_targets_t = torch.as_tensor(padded_targets, device=device)

    # Add a safety limit to prevent infinite loops
    max_total_steps = max_t * total * 100  # Allow up to 100 steps per episode on average
    step_count = 0
    
    while torch.any(counts < padded_targets_t).item():
        step_count += 1
        if step_count > max_total_steps:
            if verbose:
                print(f"\nWarning: Reached maximum step limit ({max_total_steps}). Stopping evaluation.")
                print(f"Completed episodes: {counts.cpu().tolist()} / {padded_targets.tolist()}")
            break
            
        active_mask_t = counts < padded_targets_t
        active_idx    = torch.where(active_mask_t)[0]

        # For single env, we don't need to slice
        if n_envs == 1:
            obs_active = observations
        else:
            # For batched envs, slice to active only
            obs_active = observations[active_idx]

        # Forward pass through actor
        with torch.no_grad():
            # Actor module expects TensorDict and returns TensorDict with action and sample_log_prob
            actor_output = actor_module(obs_active)
            
            # Extract action and log prob
            if "action" in actor_output.keys():
                acts_tensor = actor_output["action"]
            else:
                acts_tensor = actor_output.get("action", None)
                
            if "sample_log_prob" in actor_output.keys():
                lp_tensor = actor_output["sample_log_prob"].to(device)
            else:
                lp_tensor = torch.zeros(acts_tensor.shape[0], device=device)

        # Track histories before step
        if track_logprobs and n_envs == 1:
            # Count number of valid actions
            if "derived_sub_indices" in obs_active.keys():
                num_choices = (obs_active["derived_sub_indices"].sum(dim=(-1, -2)) != 0).sum(dim=-1).cpu().numpy()
            else:
                num_choices = np.array([1])
            
            if "sub_index" in obs_active.keys():
                subidx = obs_active["sub_index"].squeeze(0).cpu().numpy()
                state_str = index_manager.state_subindex_to_str(subidx, truncate=True)
                current_state_histories[0].append(state_str)
                current_choices_histories[0].append(int(num_choices[0]) if num_choices.size > 0 else 0)

        current_lp[active_idx] += lp_tensor

        # Create TensorDict for step
        # For single environment, batch_size should be empty torch.Size([])
        # For ParallelEnv, it should be torch.Size([n_envs])
        if hasattr(env, 'num_workers'):
            # ParallelEnv case
            action_td = TensorDict({"action": acts_tensor}, batch_size=acts_tensor.shape[:1])
        else:
            # Single environment case - need scalar action without batch dimension
            action_td = TensorDict({"action": acts_tensor.squeeze(0)}, batch_size=torch.Size([]))
        
        # Step environment
        td_next = env.step(action_td)
        
        # Extract info from TensorDict and ensure they're on the correct device
        rews_t = td_next.get("reward", torch.zeros(n_envs, device=device)).to(device)
        dones_t = td_next.get("done", torch.zeros(n_envs, dtype=torch.bool, device=device)).to(device)
        
        if rews_t.dim() == 0:
            rews_t = rews_t.unsqueeze(0)
        if dones_t.dim() == 0:
            dones_t = dones_t.unsqueeze(0)

        current_rew[active_idx] += rews_t[active_idx]
        current_len[active_idx] += 1

        # Append step lps BEFORE checking for done episodes
        if track_logprobs and n_envs == 1:
            step_lp_np = lp_tensor.detach().cpu().numpy()
            current_steplogprob_histories[0].append(float(step_lp_np[0]) if step_lp_np.size > 0 else 0.0)

        done_and_active = dones_t & active_mask_t
        if done_and_active.any():
            done_idx = torch.where(done_and_active)[0]
            slots    = counts[done_idx]

            rewards[done_idx, slots] = current_rew[done_idx]
            lengths[done_idx, slots] = current_len[done_idx]
            logps[done_idx, slots]   = current_lp[done_idx]

            # Extract success info from TensorDict
            # Look for is_success in the info part of tensordict
            is_success_vals = []
            for i in done_idx.cpu().tolist():
                # Try to get is_success from various possible locations in TensorDict
                if "is_success" in td_next.keys():
                    is_success_vals.append(bool(td_next["is_success"][i].item()))
                else:
                    # Default to False if not found
                    is_success_vals.append(False)
            
            proof_successful[done_idx, slots] = torch.as_tensor(is_success_vals, device=device)

            if track_logprobs and n_envs == 1 and 0 in done_idx.cpu().tolist():
                # Finalize histories for done env
                if current_steplogprob_histories[0]:
                    # Append final zero-action step for terminal state
                    current_choices_histories[0].append(0)
                    current_steplogprob_histories[0].append(0.0)

                    # Stash scalar histories
                    episode_logprob_histories.append(np.cumsum(current_steplogprob_histories[0]))
                    episode_steplogprob_histories.append(np.array(current_steplogprob_histories[0]))
                    episode_choices_histories.append(np.array(current_choices_histories[0]))

                    # Build final state string from terminal observation
                    if "next" in td_next.keys() and "sub_index" in td_next["next"].keys():
                        term_sub = td_next["next"]["sub_index"].squeeze(0).cpu().numpy()
                    elif "sub_index" in td_next.keys():
                        term_sub = td_next["sub_index"].squeeze(0).cpu().numpy()
                    else:
                        term_sub = None
                    
                    if term_sub is not None:
                        final_state_str = index_manager.state_subindex_to_str(term_sub, truncate=True)
                        current_state_histories[0].append(final_state_str)

                    # Freeze the whole list
                    episode_state_histories.append(np.array(current_state_histories[0]))

                    # Clear scratch buffers
                    current_steplogprob_histories[0].clear()
                    current_choices_histories[0].clear()
                    current_state_histories[0].clear()

            counts[done_idx] += 1
            current_rew[done_idx] = 0
            current_len[done_idx] = 0
            current_lp[done_idx] = 0

        # Call info_callback (disabled for TorchRL - TensorDict not compatible)
        # if info_callback is not None:
        #     info_callback(td_next)

        # Update observations for next iteration
        if "next" in td_next.keys():
            observations = td_next["next"]
            # Check if there's a nested "next" key (happens with single envs)
            if "next" in observations.keys():
                observations = observations["next"]
        else:
            observations = td_next

        if verbose:
            print(f"\rEpisodes done: {int(counts.sum())}/{total}", end="", flush=True)

    if verbose:
        print("\r" + " " * 80 + "\r", end="")

    # Mask marking valid episode slots
    mask = (torch.arange(max_t, device=device)[None, :]
            < torch.as_tensor(padded_targets, device=device)[:, None])

    # Trim to L = len(target_episodes) if provided
    L = len(target_episodes) if target_episodes is not None else n_envs
    rewards, lengths, logps, mask, proof_successful = [
        x[:L].cpu().numpy() for x in (rewards, lengths, logps, mask, proof_successful)
    ]

    if track_logprobs:
        return (
            rewards, lengths, logps, mask, proof_successful,
            episode_logprob_histories, episode_choices_histories,
            episode_steplogprob_histories, episode_state_histories
        )
    else:
        return (rewards, lengths, logps, mask, proof_successful)


# ---------------------------------------------------------------------------
# Shared helpers for high-level evaluation
# ---------------------------------------------------------------------------

_BATCH_METRIC_KEYS = [
    "pos_rw", "neg_rw", "pos_len", "neg_len", "pos_lp", "neg_lp",
    "pos_len_true", "pos_len_false", "neg_len_true", "neg_len_false",
    "y_true", "y_pred", "y_score", "mrr", "h1", "h3", "h10", "ap",
]

_GLOBAL_METRICS_TEMPLATE = {
    "pos_rw": [], "neg_rw": [], "pos_len": [], "neg_len": [],
    "pos_lp": [], "neg_lp": [], "pos_len_true": [], "pos_len_false": [],
    "neg_len_true": [], "neg_len_false": [], "y_true": [], "y_pred": [],
    "y_score": [], "head_mrr": [], "head_h1": [], "head_h3": [], "head_h10": [],
    "tail_mrr": [], "tail_h1": [], "tail_h3": [], "tail_h10": [],
    "head_ap": [], "tail_ap": [],
}


def _init_batch_metrics() -> Dict[str, List[float]]:
    return {key: [] for key in _BATCH_METRIC_KEYS}


def _init_global_metrics() -> Dict[str, List[float]]:
    return {key: [] for key in _GLOBAL_METRICS_TEMPLATE}


def _compute_targets_and_mask(
    corrs: List[List[Any]],
    num_envs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    targets = np.array([1 + len(c) for c in corrs], dtype=int)
    max_t = int(targets.max()) if targets.size > 0 else 0
    padded_targets = np.zeros(num_envs, dtype=int)
    padded_targets[: len(corrs)] = targets
    if max_t == 0:
        mask = np.zeros((num_envs, 0), dtype=bool)
    else:
        mask = (np.arange(max_t)[None, :] < padded_targets[:, None])
    return targets, mask


def _create_batch_env(
    batch: List[Any],
    corrs: List[List[Any]],
    data_depths: Optional[List[int]],
    batch_start_idx: int,
    index_manager: Any,
    data_handler: Any,
    args_dict: Dict[str, Any],
) -> Any:
    """Create a new environment configured for a specific batch of queries with corruptions.
    
    This creates a fresh single environment (not ParallelEnv) with all queries and their
    corruptions combined into one episode sequence.
    
    Args:
        batch: List of positive queries
        corrs: List of corruption lists (one list per query)
        data_depths: Optional depth information
        batch_start_idx: Starting index in the full dataset
        index_manager: Index manager for vocab/embeddings
        data_handler: Data handler with facts and configuration
        args_dict: Dictionary of environment arguments
        
    Returns:
        A new environment configured with the batch queries + corruptions
    """
    from env import LogicEnv_gym as LogicEnv
    
    # Combine all queries and corruptions into one sequence
    all_queries = []
    all_labels = []
    all_depths = []
    
    for i, (query, negatives) in enumerate(zip(batch, corrs)):
        # Add positive query
        all_queries.append(query)
        all_labels.append(1)
        batch_idx = batch_start_idx + i
        pos_depth = data_depths[batch_idx] if data_depths is not None else None
        all_depths.append(pos_depth)
        
        # Add corruptions
        for neg in negatives:
            all_queries.append(neg)
            all_labels.append(0)
            all_depths.append(-1)
    
    # Get facts from data handler
    facts_set = getattr(data_handler, "facts_set", None)
    if facts_set is None:
        facts_set = frozenset(data_handler.facts) if hasattr(data_handler, 'facts') else frozenset()
    
    # Create new single environment with combined queries
    new_env = LogicEnv(
        index_manager=index_manager,
        data_handler=data_handler,
        queries=all_queries,
        labels=all_labels,
        query_depths=all_depths,
        facts=facts_set,
        mode='eval',
        corruption_mode=None,  # No corruption in eval
        corruption_scheme=None,
        train_neg_ratio=1,
        seed=None,
        max_depth=args_dict.get('max_depth', 20),
        memory_pruning=args_dict.get('memory_pruning', True),
        endt_action=args_dict.get('endt_action', False),
        endf_action=args_dict.get('endf_action', False),
        skip_unary_actions=args_dict.get('skip_unary_actions', True),
        padding_atoms=args_dict.get('padding_atoms', 6),
        padding_states=args_dict.get('padding_states', 20),
        device=torch.device('cpu'),
        engine=args_dict.get('engine', 'python'),
        kge_action=False,
        reward_type=args_dict.get('reward_type', 4),
        shaping_beta=0.0,
        shaping_gamma=None,
        kge_inference_engine=None,
        verbose=0,
        prover_verbose=0,
    )
    
    return new_env


def _evaluate_with_rl(
    actor_module: torch.nn.Module,
    value_module: Optional[torch.nn.Module],
    template_env: Any,
    batch: List[Any],
    corrs: List[List[Any]],
    targets: np.ndarray,
    deterministic: bool,
    verbose: bool,
    info_callback: Optional[Callable[[TensorDict], None]],
    data_depths: Optional[List[int]],
    batch_start_idx: int,
    plot: bool,
    index_manager: Any,
    data_handler: Any,
    args_dict: Dict[str, Any],
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    Optional[Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]],
]:
    """Evaluate a batch using RL policy.
    
    Creates a new temporary environment configured with the batch queries and corruptions.
    """
    # Ensure args_dict is a dict
    if args_dict is None:
        args_dict = {}
    
    # Create a new environment for this specific batch
    batch_env = _create_batch_env(
        batch, corrs, data_depths, batch_start_idx, 
        index_manager, data_handler, args_dict
    )

    eval_kwargs = dict(
        actor_module=actor_module,
        value_module=value_module,
        env=batch_env,
        deterministic=deterministic,
        target_episodes=targets,
        verbose=verbose,
        info_callback=info_callback,
    )

    if plot:
        (
            rewards, lengths, log_probs, mask, proof_successful,
            logprob_histories, choices_histories,
            steplogprob_histories, state_histories,
        ) = evaluate_policy(track_logprobs=True, **eval_kwargs)
        plot_payload = (
            logprob_histories, choices_histories,
            steplogprob_histories, state_histories,
        )
    else:
        rewards, lengths, log_probs, mask, proof_successful = evaluate_policy(
            track_logprobs=False, **eval_kwargs
        )
        plot_payload = None
    
    # Clean up the temporary environment
    del batch_env
    
    return rewards, lengths, log_probs, mask, proof_successful, plot_payload


def _combine_hybrid_scores(
    rl_log_probs: np.ndarray,
    kge_log_scores: np.ndarray,
    mask: np.ndarray,
    proof_successful: np.ndarray,
    kge_weight: float,
    rl_weight: float,
    success_only: bool,
) -> np.ndarray:
    """Combine RL and KGE scores for hybrid evaluation."""
    combined = np.copy(rl_log_probs)
    combined[mask] = kge_weight * kge_log_scores[mask]
    if success_only:
        combined[mask] += np.where(
            proof_successful[mask], rl_weight * rl_log_probs[mask], 0.0
        )
    else:
        combined[mask] += rl_weight * rl_log_probs[mask]
    return combined


def kge_eval(
    batch: List[Any],
    corrs: List[List[Any]],
    mask: np.ndarray,
    kge_inference_engine: Any,
) -> np.ndarray:
    """Score candidates using KGE inference engine."""
    all_atoms_in_batch: List[str] = []
    atom_map: Dict[str, Tuple[int, int]] = {}
    
    for i, (q, negs) in enumerate(zip(batch, corrs)):
        if mask[i, 0]:
            atom_str = f"{q.predicate}({','.join(map(str, q.args))})"
            all_atoms_in_batch.append(atom_str)
            atom_map[atom_str] = (i, 0)
        for j, neg_tuple in enumerate(negs):
            if mask[i, j + 1]:
                atom_str = f"{neg_tuple.predicate}({','.join(map(str, neg_tuple.args))})"
                all_atoms_in_batch.append(atom_str)
                atom_map[atom_str] = (i, j + 1)
    
    kge_scores_flat = np.array(kge_inference_engine.predict_batch(all_atoms_in_batch))
    kge_log_scores_flat = np.log(kge_scores_flat + 1e-9)
    
    log_probs = np.full(mask.shape, -np.inf, dtype=float)
    for atom_str, log_score in zip(all_atoms_in_batch, kge_log_scores_flat):
        env_idx, ep_idx = atom_map[atom_str]
        log_probs[env_idx, ep_idx] = log_score
    return log_probs


def _extract_and_accumulate_metrics(
    batch_metrics: Dict[str, List[float]],
    global_metrics: Dict[str, List[float]],
    corruption_type: str,
    mask: np.ndarray,
    proof_successful: np.ndarray,
    log_probs: np.ndarray,
    rewards: Optional[np.ndarray] = None,
    lengths: Optional[np.ndarray] = None,
) -> None:
    """Extract and accumulate evaluation metrics."""
    # Classification targets/predictions
    true_labels = np.zeros_like(mask, dtype=int)
    true_labels[:, 0] = 1
    y_true_part = true_labels[mask]
    y_pred_part = proof_successful[mask].astype(int)
    y_score_part = log_probs[mask]

    batch_metrics['y_true'].extend(y_true_part.tolist())
    global_metrics['y_true'].extend(y_true_part.tolist())
    batch_metrics['y_pred'].extend(y_pred_part.tolist())
    global_metrics['y_pred'].extend(y_pred_part.tolist())
    batch_metrics['y_score'].extend(y_score_part.tolist())
    global_metrics['y_score'].extend(y_score_part.tolist())

    # Log probabilities
    pos_lp = log_probs[:, 0][mask[:, 0]]
    neg_lp = log_probs[:, 1:][mask[:, 1:]]
    batch_metrics['pos_lp'].extend(pos_lp.tolist())
    global_metrics['pos_lp'].extend(pos_lp.tolist())
    batch_metrics['neg_lp'].extend(neg_lp.tolist())
    global_metrics['neg_lp'].extend(neg_lp.tolist())

    # Rewards and lengths (only for RL modes)
    if rewards is not None and lengths is not None:
        pos_rw = rewards[:, 0][mask[:, 0]]
        neg_rw = rewards[:, 1:][mask[:, 1:]]
        pos_len = lengths[:, 0][mask[:, 0]]
        neg_len = lengths[:, 1:][mask[:, 1:]]
        
        batch_metrics['pos_rw'].extend(pos_rw.tolist())
        global_metrics['pos_rw'].extend(pos_rw.tolist())
        batch_metrics['neg_rw'].extend(neg_rw.tolist())
        global_metrics['neg_rw'].extend(neg_rw.tolist())
        batch_metrics['pos_len'].extend(pos_len.tolist())
        global_metrics['pos_len'].extend(pos_len.tolist())
        batch_metrics['neg_len'].extend(neg_len.tolist())
        global_metrics['neg_len'].extend(neg_len.tolist())

        # Detailed episode lengths
        pos_mask = mask[:, 0]
        if np.any(pos_mask):
            pos_lengths = lengths[:, 0][pos_mask]
            pos_success = proof_successful[:, 0][pos_mask]
            pos_len_true = pos_lengths[pos_success]
            pos_len_false = pos_lengths[~pos_success]
            batch_metrics['pos_len_true'].extend(pos_len_true.tolist())
            global_metrics['pos_len_true'].extend(pos_len_true.tolist())
            batch_metrics['pos_len_false'].extend(pos_len_false.tolist())
            global_metrics['pos_len_false'].extend(pos_len_false.tolist())

        neg_mask = mask[:, 1:]
        if np.any(neg_mask):
            neg_lengths = lengths[:, 1:][neg_mask]
            neg_success = proof_successful[:, 1:][neg_mask]
            neg_len_true = neg_lengths[neg_success]
            neg_len_false = neg_lengths[~neg_success]
            batch_metrics['neg_len_true'].extend(neg_len_true.tolist())
            global_metrics['neg_len_true'].extend(neg_len_true.tolist())
            batch_metrics['neg_len_false'].extend(neg_len_false.tolist())
            global_metrics['neg_len_false'].extend(neg_len_false.tolist())

    # Ranking metrics
    if mask.shape[1] > 1:
        lp_batch = np.where(mask, log_probs, -np.inf)
        random_keys = np.random.rand(*lp_batch.shape)
        sorted_indices = np.lexsort((-random_keys, -lp_batch), axis=1)
        ranks = np.where(sorted_indices == 0)[1] + 1
        mrr = 1.0 / ranks
        h1 = (ranks == 1).astype(float)
        h3 = (ranks <= 3).astype(float)
        h10 = (ranks <= 10).astype(float)
        ap = average_precision_score(y_true_part, y_score_part)
        
        batch_metrics['mrr'].extend(mrr.tolist())
        batch_metrics['h1'].extend(h1.tolist())
        batch_metrics['h3'].extend(h3.tolist())
        batch_metrics['h10'].extend(h10.tolist())
        batch_metrics['ap'].append(ap)
        
        global_metrics[f'{corruption_type}_mrr'].extend(mrr.tolist())
        global_metrics[f'{corruption_type}_h1'].extend(h1.tolist())
        global_metrics[f'{corruption_type}_h3'].extend(h3.tolist())
        global_metrics[f'{corruption_type}_h10'].extend(h10.tolist())
        global_metrics[f'{corruption_type}_ap'].append(ap)


def _report_batch_metrics(
    batch_metrics: Dict[str, List[float]],
    global_metrics: Dict[str, List[float]]
) -> None:
    """Print batch and rolling metrics."""
    def safe_mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    def safe_str(val: float) -> str:
        return f"{val:.3f}" if isinstance(val, float) else val

    # Batch metrics
    b_pos_rw = safe_mean(batch_metrics["pos_rw"])
    b_neg_rw = safe_mean(batch_metrics["neg_rw"])
    b_pos_len = safe_mean(batch_metrics["pos_len"])
    b_neg_len = safe_mean(batch_metrics["neg_len"])
    b_pos_lp = safe_mean(batch_metrics["pos_lp"])
    b_neg_lp = safe_mean(batch_metrics["neg_lp"])
    b_mrr = safe_mean(batch_metrics["mrr"])
    b_h1 = safe_mean(batch_metrics["h1"])
    b_h10 = safe_mean(batch_metrics["h10"])

    # Rolling metrics
    r_pos_rw = safe_mean(global_metrics["pos_rw"])
    r_neg_rw = safe_mean(global_metrics["neg_rw"])
    r_pos_len = safe_mean(global_metrics["pos_len"])
    r_neg_len = safe_mean(global_metrics["neg_len"])
    r_pos_lp = safe_mean(global_metrics["pos_lp"])
    r_neg_lp = safe_mean(global_metrics["neg_lp"])
    r_mrr = safe_mean(global_metrics["head_mrr"] + global_metrics["tail_mrr"])
    r_h1 = safe_mean(global_metrics["head_h1"] + global_metrics["tail_h1"])
    r_h10 = safe_mean(global_metrics["head_h10"] + global_metrics["tail_h10"])

    print(f"\nBatch: rwd_pos={safe_str(b_pos_rw)}, rwd_neg={safe_str(b_neg_rw)}, "
          f"len_pos={safe_str(b_pos_len)}, len_neg={safe_str(b_neg_len)}")
    print(f"       lp_pos={b_pos_lp:.3f}, lp_neg={safe_str(b_neg_lp)}, "
          f"mrr={b_mrr:.3f}, h1={b_h1:.3f}, h10={b_h10:.3f}")
    print(f"Rolling: rwd_pos={safe_str(r_pos_rw)}, rwd_neg={safe_str(r_neg_rw)}, "
          f"len_pos={safe_str(r_pos_len)}, len_neg={safe_str(r_neg_len)}")
    print(f"         lp_pos={r_pos_lp:.3f}, lp_neg={safe_str(r_neg_lp)}, "
          f"mrr={r_mrr:.3f}, h1={r_h1:.3f}, h10={r_h10:.3f}")


def _finalize_and_get_results(metrics: Dict[str, List[float]]) -> Dict[str, float]:
    """Summarize aggregated metrics into result dictionary."""
    def get_stats(data: List[float]) -> Tuple[float, float]:
        arr = np.array(data)
        return (arr.mean(), arr.std()) if arr.size > 0 else (0.0, 0.0)

    # Combined rank metrics
    all_mrr = np.array(metrics['head_mrr'] + metrics['tail_mrr'])
    all_h1 = np.array(metrics['head_h1'] + metrics['tail_h1'])
    all_h3 = np.array(metrics['head_h3'] + metrics['tail_h3'])
    all_h10 = np.array(metrics['head_h10'] + metrics['tail_h10'])
    all_ap = np.array(metrics['head_ap'] + metrics['tail_ap'])
    
    final_results = {
        'mrr_mean': all_mrr.mean() if all_mrr.size > 0 else 0.0,
        'hits1_mean': all_h1.mean() if all_h1.size > 0 else 0.0,
        'hits3_mean': all_h3.mean() if all_h3.size > 0 else 0.0,
        'hits10_mean': all_h10.mean() if all_h10.size > 0 else 0.0,
        'average_precision': all_ap.mean() if all_ap.size > 0 else 0.0,
    }
    
    for key in ['pos_rw', 'neg_rw', 'pos_len', 'neg_len', 'pos_lp', 'neg_lp']:
        mean, std = get_stats(metrics[key])
        name_map = {
            'pos_rw': 'rewards_pos', 'neg_rw': 'rewards_neg',
            'pos_len': 'episode_len_pos', 'neg_len': 'episode_len_neg',
            'pos_lp': 'log_probs_pos', 'neg_lp': 'log_probs_neg',
        }
        final_results[f'{name_map[key]}_mean'] = mean
        final_results[f'{name_map[key]}_std'] = std

    return {k: float(v) for k, v in final_results.items()}


# ---------------------------------------------------------------------------
# High-level evaluator for head/tail corruptions
# ---------------------------------------------------------------------------

def eval_corruptions(
    actor_module: torch.nn.Module,
    value_module: Optional[torch.nn.Module],
    env: Any,
    data: List[Any],
    sampler: Any,
    n_corruptions: Optional[int] = None,
    deterministic: bool = True,
    verbose: int = 1,
    plot: bool = False,
    kge_inference_engine = None,
    evaluation_mode: str = 'rl_only',
    corruption_scheme: List[str] | None = None,
    info_callback: Optional[Callable[[TensorDict], None]] = None,
    data_depths: Optional[List[int]] = None,
    hybrid_kge_weight: float = 2.0,
    hybrid_rl_weight: float = 1.0,
    hybrid_success_only: bool = True,
    index_manager: Any = None,
    data_handler: Any = None,
    **kwargs  # Accept additional args like group_size
) -> Dict[str, Any]:
    """Evaluate a TorchRL model by ranking positive queries vs corruptions.

    Parameters
    ----------
    actor_module : torch.nn.Module
        TorchRL actor module.
    value_module : torch.nn.Module or None
        TorchRL value module (optional).
    env : TorchRL EnvBase
        Evaluation environment.
    data : list
        List of positive queries to evaluate.
    sampler : Any
        Negative sampler for generating corruptions.
    n_corruptions : Optional[int]
        Number of negatives per type (None for all).
    deterministic : bool, default True
        Use deterministic actions.
    verbose : int, default 1
        Verbosity level.
    plot : bool, default False
        Collect trajectories for plotting.
    kge_inference_engine : optional
        KGE model for hybrid/kge_only modes.
    evaluation_mode : {'rl_only','kge_only','hybrid'}
        Scoring strategy.
    corruption_scheme : list[str]
        Subset of {"head","tail"} to evaluate.
    info_callback : callable, optional
        Callback for custom logging.
    data_depths : list[int], optional
        Depth values for queries.
    hybrid_kge_weight : float, default 2.0
        KGE weight in hybrid mode.
    hybrid_rl_weight : float, default 1.0
        RL weight in hybrid mode.
    hybrid_success_only : bool, default True
        Add RL bonus only for successful proofs in hybrid mode.

    Returns
    -------
    Dict[str, Any]
        Aggregated evaluation metrics.
    """
    if evaluation_mode not in ("rl_only", "kge_only", "hybrid"):
        raise ValueError(f"Invalid evaluation_mode: {evaluation_mode}")
    if evaluation_mode != "rl_only" and kge_inference_engine is None:
        raise ValueError(f"`kge_inference_engine` required for mode: '{evaluation_mode}'")

    num_envs = 1
    if n_corruptions == -1:
        n_corruptions = None

    if verbose:
        print(f"Evaluating {len(data)} queries in '{evaluation_mode}' mode.")
        print(f"N corruptions per query: {'All' if n_corruptions is None else n_corruptions}")

    global_metrics = _init_global_metrics()
    device = next(actor_module.parameters()).device

    total_batches = (len(data) + num_envs - 1) // num_envs
    
    for b, start in enumerate(range(0, len(data), num_envs)):
        time_start = time.time()
        if verbose:
            end_idx = min(start + num_envs - 1, len(data) - 1)
            print(f"\n--- Batch {b+1}/{total_batches} (Queries {start}-{end_idx}) ---")

        batch = data[start : start + num_envs]
        B = len(batch)
        batch_metrics = _init_batch_metrics()

        if n_corruptions == 0:
            head_corrs = [[] for _ in range(B)]
            tail_corrs = [[] for _ in range(B)]
        else:
            head_corrs, tail_corrs = sampler.get_negatives_from_states_separate(
                [[q] for q in batch], device, num_negs=n_corruptions
            )
            if B == 1:
                if head_corrs and not isinstance(head_corrs[0], list):
                    head_corrs = [head_corrs]
                if tail_corrs and not isinstance(tail_corrs[0], list):
                    tail_corrs = [tail_corrs]

        for corruption_type, corrs in (("head", head_corrs), ("tail", tail_corrs)):
            if corruption_scheme and corruption_type not in corruption_scheme:
                continue

            targets, mask = _compute_targets_and_mask(corrs, num_envs)
            if not np.any(targets > 1):
                continue

            rewards: Optional[np.ndarray] = None
            lengths: Optional[np.ndarray] = None

            if evaluation_mode == "kge_only":
                log_probs = kge_eval(batch, corrs, mask, kge_inference_engine)
                proof_successful = np.zeros_like(mask, dtype=bool)
            else:
                # Build args dict for environment creation
                env_args = kwargs.copy()  # Get any extra args passed to eval_corruptions
                
                (
                    rewards, lengths, log_probs, eval_mask, proof_successful, plot_payload,
                ) = _evaluate_with_rl(
                    actor_module=actor_module,
                    value_module=value_module,
                    template_env=env,  # Pass as template for creating batch envs
                    batch=batch,
                    corrs=corrs,
                    targets=targets,
                    deterministic=deterministic,
                    verbose=verbose > 1,
                    info_callback=info_callback,
                    data_depths=data_depths,
                    batch_start_idx=start,
                    plot=plot,
                    index_manager=index_manager,
                    data_handler=data_handler,
                    args_dict=env_args,
                )
                mask = eval_mask

                if evaluation_mode == "hybrid":
                    kge_log_scores = kge_eval(batch, corrs, mask, kge_inference_engine)
                    log_probs = _combine_hybrid_scores(
                        rl_log_probs=log_probs,
                        kge_log_scores=kge_log_scores,
                        mask=mask,
                        proof_successful=proof_successful,
                        kge_weight=hybrid_kge_weight,
                        rl_weight=hybrid_rl_weight,
                        success_only=hybrid_success_only,
                    )

                # Penalize failed proofs
                log_probs = log_probs.copy()
                log_probs[~proof_successful] -= 100.0

            _extract_and_accumulate_metrics(
                batch_metrics, global_metrics, corruption_type,
                mask, proof_successful, log_probs, rewards, lengths,
            )

        if verbose:
            _report_batch_metrics(batch_metrics, global_metrics)
            print(f"Batch {b+1} took {time.time() - time_start:.2f} seconds")

    return _finalize_and_get_results(global_metrics)


# ---------------------------------------------------------------------------
# Compatibility wrappers and aliases
# ---------------------------------------------------------------------------

def eval_corruptions_torchrl(
    actor: torch.nn.Module,
    env: Any,
    data: List[Any],
    sampler: Any,
    n_corruptions: Optional[int] = None,
    deterministic: bool = True,
    verbose: int = 1,
    plot: bool = False,
    kge_inference_engine = None,
    evaluation_mode: str = 'rl_only',
    corruption_scheme: List[str] | None = None,
    info_callback: Optional[Callable[[TensorDict], None]] = None,
    data_depths: Optional[List[int]] = None,
    hybrid_kge_weight: float = 2.0,
    hybrid_rl_weight: float = 1.0,
    hybrid_success_only: bool = True,
    group_size: int = 1,
    index_manager: Any = None,
    data_handler: Any = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compatibility wrapper for eval_corruptions that matches the old API.
    
    This function wraps eval_corruptions() to provide backwards compatibility
    with existing code that calls eval_corruptions_torchrl().
    
    Additional parameters like group_size, index_manager, and data_handler
    are accepted but may be ignored if not needed by the underlying implementation.
    """
    return eval_corruptions(
        actor_module=actor,
        value_module=None,  # Not needed for evaluation
        env=env,
        data=data,
        sampler=sampler,
        n_corruptions=n_corruptions,
        deterministic=deterministic,
        verbose=verbose,
        plot=plot,
        kge_inference_engine=kge_inference_engine,
        evaluation_mode=evaluation_mode,
        corruption_scheme=corruption_scheme,
        info_callback=info_callback,
        data_depths=data_depths,
        hybrid_kge_weight=hybrid_kge_weight,
        hybrid_rl_weight=hybrid_rl_weight,
        hybrid_success_only=hybrid_success_only,
        index_manager=index_manager,
        data_handler=data_handler,
        **kwargs  # Pass through remaining args for env creation
    )


class TorchRLPolicyWrapper:
    """
    Compatibility wrapper for TorchRL policies.
    
    This class is provided for backwards compatibility but is not needed
    with the current implementation since we work directly with actor modules.
    """
    def __init__(self, actor_module: torch.nn.Module, value_module: Optional[torch.nn.Module] = None):
        self.actor = actor_module
        self.value = value_module
    
    def __call__(self, obs: TensorDict) -> TensorDict:
        """Forward pass through the actor."""
        return self.actor(obs)
