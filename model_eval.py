"""
model_eval_torchrl.py — Pure TorchRL evaluation utilities (no SB3 dependencies)

This module provides TorchRL-native evaluation functions for logical reasoning
environments using ParallelEnv or single EnvBase instances directly.

Key components:
1. evaluate_policy_torchrl - Fast vectorized policy evaluation on TorchRL envs
2. eval_corruptions_torchrl - Link-prediction style evaluation with corruptions
3. kge_eval - KGE-based scoring for hybrid evaluation modes

All evaluation is done directly with TorchRL environments (ParallelEnv or EnvBase)
without any SB3 wrappers or dependencies.
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
import torch.nn as nn
from tensordict import TensorDict
from torchrl.envs import EnvBase


# ---------------------------------------------------------------------------
# Low-level, fast vectorized evaluator for TorchRL
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_policy_torchrl(
    actor: nn.Module,
    env: EnvBase,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    target_episodes: np.ndarray | None = None,
    verbose: int = 0,
    track_logprobs: bool = False,
    info_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Roll out a TorchRL actor on a TorchRL env and aggregate per-episode stats.

    Parameters
    ----------
    actor : nn.Module
        TorchRL actor module (e.g., ProbabilisticActor). Must accept TensorDict
        and return TensorDict with 'action' and optionally 'sample_log_prob'.
    env : EnvBase
        TorchRL environment (can be ParallelEnv or single EnvBase).
    n_eval_episodes : int, default 10
        Total target episodes across all sub-envs (if `target_episodes` is not
        provided).
    deterministic : bool, default True
        Whether to use deterministic policy actions (always True for eval).
    target_episodes : np.ndarray | None, default None
        Optional per-env episode targets (shape `(n_envs,)`). If `None`, targets
        are distributed as evenly as possible.
    verbose : int, default 0
        Print a running counter when > 0.
    track_logprobs : bool, default False
        If True, collects *per-step* log-prob trajectories and state strings.
    info_callback : callable, optional
        If provided, called after each environment step with the list of info
        dicts (useful for custom logging).

    Returns
    -------
    rewards : np.ndarray, shape (n_envs, max_episodes)
        Sum of rewards per finished episode slot for each env.
    lengths : np.ndarray, shape (n_envs, max_episodes)
        Episode lengths per slot.
    logps : np.ndarray, shape (n_envs, max_episodes)
        Accumulated action log-probabilities per episode.
    mask : np.ndarray, shape (n_envs, max_episodes), dtype=bool
        Mask marking valid episode slots per env.
    proof_successful : np.ndarray, shape (n_envs, max_episodes), dtype=bool
        Whether the env reported success for each finished episode.
    """
    # Set actor to eval mode
    actor.eval()
    
    # Determine number of environments from env's batch_size
    if hasattr(env, 'batch_size') and len(env.batch_size) > 0:
        n_envs = int(env.batch_size[0])
    elif hasattr(env, 'num_workers'):
        n_envs = int(env.num_workers)
    else:
        n_envs = 1
    
    device = next(actor.parameters()).device

    # --- targets ---
    if target_episodes is None:
        targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype=int)
    else:
        targets = np.asarray(target_episodes, dtype=int)

    padded_targets = np.zeros(n_envs, dtype=int)
    padded_targets[:len(targets)] = targets
    max_t = int(padded_targets.max())
    total = int(padded_targets.sum())

    if verbose:
        print(f"\nEvaluating {total} episodes on {n_envs} envs (avg target: {targets.mean():.2f})")

    # Set episode targets in environment if supported
    # ParallelEnv and BatchedVecEnv support direct attribute assignment
    from torchrl.envs import ParallelEnv
    from env import BatchedVecEnv
    
    # Both ParallelEnv and BatchedVecEnv can have these attributes set directly
    if isinstance(env, (ParallelEnv, BatchedVecEnv)):
        if hasattr(env, '_episode_target'):
            env._episode_target[:] = padded_targets
        if hasattr(env, '_episode_count'):
            env._episode_count[:] = 0
        if hasattr(env, 'active_envs'):
            env.active_envs[:] = True
    elif hasattr(env, '_episode_target'):
        # Single env case
        env._episode_target[:] = padded_targets
        if hasattr(env, '_episode_count'):
            env._episode_count[:] = 0
        if hasattr(env, 'active_envs'):
            env.active_envs[:] = True

    # --- buffers on device ---
    rewards = torch.zeros((n_envs, max_t), device=device)
    lengths = torch.zeros_like(rewards, dtype=torch.int32)
    logps = torch.zeros_like(rewards)
    proof_successful = torch.zeros_like(rewards, dtype=torch.bool)

    counts = torch.zeros(n_envs, dtype=torch.int32, device=device)
    current_rew = torch.zeros(n_envs, device=device)
    current_len = torch.zeros(n_envs, dtype=torch.int32, device=device)
    current_lp = torch.zeros(n_envs, device=device)

    # Optional histories for plotting/debug
    episode_logprob_histories: list[np.ndarray] = []
    episode_choices_histories: list[np.ndarray] = []
    episode_steplogprob_histories: list[np.ndarray] = []
    episode_state_histories: list[np.ndarray] = []
    if track_logprobs:
        current_steplogprob_histories: list[list[float]] = [[] for _ in range(n_envs)]
        current_choices_histories: list[list[int]] = [[] for _ in range(n_envs)]
        current_state_histories: list[list[str]] = [[] for _ in range(n_envs)]
        # Try to get index_manager for state string conversion
        index_manager = None
        if hasattr(env, 'index_manager'):
            index_manager = env.index_manager
        elif hasattr(env, 'envs') and len(env.envs) > 0:
            if hasattr(env.envs[0], 'index_manager'):
                index_manager = env.envs[0].index_manager

    # Reset environment
    td = env.reset()
    padded_targets_t = torch.as_tensor(padded_targets, device=device)

    # Main evaluation loop
    while torch.any(counts < padded_targets_t).item():
        active_mask_t = counts < padded_targets_t
        active_idx = torch.where(active_mask_t)[0]

        # Get observations from TensorDict
        obs_td = td.get("next") if "next" in td.keys() else td
        
        # Track histories before step
        if track_logprobs and index_manager is not None:
            num_choices = (obs_td["derived_sub_indices"].sum(dim=(-1, -2)) != 0).sum(dim=-1).cpu().numpy()
            all_sub_indices = obs_td["sub_index"].cpu().numpy()
            for i in active_idx.cpu().tolist():
                if n_envs == 1:
                    subidx = all_sub_indices.squeeze(0)
                else:
                    subidx = all_sub_indices[i].squeeze(0)
                state_str = index_manager.state_subindex_to_str(subidx, truncate=True)
                current_state_histories[i].append(state_str)
                if n_envs == 1:
                    current_choices_histories[i].append(num_choices.item() if num_choices.ndim == 0 else num_choices[0])
                else:
                    current_choices_histories[i].append(num_choices[i])

        # Get actions from actor
        td_with_action = actor(obs_td)
        
        # Extract actions and log probabilities
        action_output = td_with_action.get("action")
        if action_output is None:
            raise ValueError("Actor did not produce 'action' in output TensorDict")
        
        # Handle action format: convert to indices if needed
        if action_output.dim() >= 2:
            # One-hot or batched format
            if action_output.dim() == 3:
                action_output = action_output[:, -1, :]  # Take last timestep
            actions = torch.argmax(action_output, dim=-1)
        else:
            actions = action_output.long()
        
        # Get log probabilities
        # Extract log-probabilities and ensure they are on the same device
        # as the actor/accumulators to avoid device-mismatch when indexing.
        lp_tensor = td_with_action.get("sample_log_prob", None)
        if lp_tensor is None:
            lp_tensor = torch.zeros(n_envs, device=device)
        else:
            # Move incoming TensorDict tensors to the actor/device
            lp_tensor = lp_tensor.to(device)
            if lp_tensor.dim() == 2:
                lp_tensor = lp_tensor[:, -1]  # Take last timestep
        
        # Enforce action mask (safety check)
        action_mask = obs_td.get("action_mask", None)
        if action_mask is not None:
            action_mask = action_mask.to(torch.bool)
            for i in range(actions.shape[0]):
                a = int(actions[i].item())
                if action_mask.dim() == 2:
                    mask_i = action_mask[i]
                else:
                    mask_i = action_mask
                if a < 0 or a >= mask_i.shape[0] or not bool(mask_i[a].item()):
                    valid = torch.where(mask_i)[0]
                    if valid.numel() > 0:
                        actions[i] = valid[0]
                        lp_tensor[i] = torch.log(torch.tensor(1.0 / float(valid.numel()), device=device))
                    else:
                        actions[i] = 0
                        lp_tensor[i] = float('-inf')
        
        # Accumulate log probs for active environments
        current_lp[active_idx] += lp_tensor[active_idx]
        
        # Create action TensorDict
        if n_envs == 1:
            action_td = TensorDict({
                "action": actions[0] if actions.dim() > 0 else actions,
            }, batch_size=env.batch_size)
        else:
            action_td = TensorDict({
                "action": actions,
            }, batch_size=env.batch_size)
        
        # Step environment
        td = env.step(action_td)
        
        # Extract step results
        if "next" in td.keys():
            next_td = td["next"]
            reward_tensor = next_td.get("reward")
            done_tensor = next_td.get("done")
        else:
            reward_tensor = td.get("reward")
            done_tensor = td.get("done")
        
        if reward_tensor is None or done_tensor is None:
            raise ValueError(f"TensorDict missing 'reward' or 'done'. Keys: {list(td.keys())}")
        
        # Convert to CPU tensors
        rews_t = reward_tensor.to(device).flatten()[:n_envs]
        dones_t = done_tensor.to(device).flatten()[:n_envs].to(torch.bool)
        
        # Accumulate episode statistics
        current_rew[active_idx] += rews_t[active_idx]
        current_len[active_idx] += 1
        
        # Track step log probs
        if track_logprobs:
            step_lp_np = lp_tensor.detach().cpu().numpy()
            for i in active_idx.cpu().tolist():
                if n_envs == 1:
                    current_steplogprob_histories[i].append(step_lp_np.item() if step_lp_np.ndim == 0 else step_lp_np[0])
                else:
                    current_steplogprob_histories[i].append(step_lp_np[i])
        
        # Handle episode completions
        done_and_active = dones_t & active_mask_t
        if done_and_active.any():
            done_idx = torch.where(done_and_active)[0]
            slots = counts[done_idx]
            
            rewards[done_idx, slots] = current_rew[done_idx]
            lengths[done_idx, slots] = current_len[done_idx]
            logps[done_idx, slots] = current_lp[done_idx]
            
            # Extract success flag
            def safe_extract(tensor_val, index):
                if tensor_val.dim() == 0:
                    return tensor_val.item()
                else:
                    return tensor_val[index].item()
            
            succ_list = []
            for env_i in done_idx.cpu().tolist():
                if "is_success" in td.keys():
                    succ = bool(safe_extract(td["is_success"], env_i))
                elif "next" in td.keys() and "is_success" in td["next"].keys():
                    succ = bool(safe_extract(td["next"]["is_success"], env_i))
                else:
                    succ = False
                succ_list.append(succ)
            
            proof_successful[done_idx, slots] = torch.as_tensor(succ_list, device=device)
            
            # Build info dicts for callback
            if info_callback is not None:
                infos = []
                for i in range(n_envs):
                    info = {}
                    if i in done_idx.cpu().tolist():
                        info["episode"] = {
                            "r": float(current_rew[i].item()),
                            "l": int(current_len[i].item()),
                        }
                        info["is_success"] = succ_list[done_idx.cpu().tolist().index(i)]
                        
                        # Extract additional fields
                        if "label" in td.keys():
                            info["label"] = int(safe_extract(td["label"], i))
                        elif "next" in td.keys() and "label" in td["next"].keys():
                            info["label"] = int(safe_extract(td["next"]["label"], i))
                        
                        if "query_depth" in td.keys():
                            info["query_depth"] = int(safe_extract(td["query_depth"], i))
                        elif "next" in td.keys() and "query_depth" in td["next"].keys():
                            info["query_depth"] = int(safe_extract(td["next"]["query_depth"], i))
                        
                        if "episode_idx" in td.keys():
                            info["episode_idx"] = int(safe_extract(td["episode_idx"], i))
                        elif "next" in td.keys() and "episode_idx" in td["next"].keys():
                            info["episode_idx"] = int(safe_extract(td["next"]["episode_idx"], i))
                    
                    infos.append(info)
                
                info_callback(infos)
            
            # Track histories for completed episodes
            if track_logprobs and index_manager is not None:
                for env_i in done_idx.cpu().tolist():
                    if not current_steplogprob_histories[env_i]:
                        continue
                    
                    # Append final terminal state
                    current_choices_histories[env_i].append(0)
                    current_steplogprob_histories[env_i].append(0.0)
                    
                    episode_logprob_histories.append(np.cumsum(current_steplogprob_histories[env_i]))
                    episode_steplogprob_histories.append(np.array(current_steplogprob_histories[env_i]))
                    episode_choices_histories.append(np.array(current_choices_histories[env_i]))
                    
                    # Get terminal state
                    if "next" in td.keys():
                        if n_envs == 1:
                            term_sub = td["next"]["sub_index"].cpu().numpy().squeeze(0)
                        else:
                            term_sub = td["next"]["sub_index"][env_i].cpu().numpy().squeeze(0)
                    else:
                        if n_envs == 1:
                            term_sub = td["sub_index"].cpu().numpy().squeeze(0)
                        else:
                            term_sub = td["sub_index"][env_i].cpu().numpy().squeeze(0)
                    
                    final_state_str = index_manager.state_subindex_to_str(term_sub, truncate=True)
                    current_state_histories[env_i].append(final_state_str)
                    episode_state_histories.append(np.array(current_state_histories[env_i]))
                    
                    # Clear buffers
                    current_steplogprob_histories[env_i].clear()
                    current_choices_histories[env_i].clear()
                    current_state_histories[env_i].clear()
            
            # Update counters
            counts[done_idx] += 1
            current_rew[done_idx] = 0
            current_len[done_idx] = 0
            current_lp[done_idx] = 0
        
        if verbose:
            print(f"\rEpisodes done: {int(counts.sum())}/{total}", end="", flush=True)
    
    if verbose:
        print("\r" + " " * 80 + "\r", end="")
    
    # Create mask for valid episodes
    mask = (torch.arange(max_t, device=device)[None, :]
            < torch.as_tensor(padded_targets, device=device)[:, None])
    
    # Trim to requested number of envs
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
    "pos_lp": [], "neg_lp": [],
    "pos_len_true": [], "pos_len_false": [],
    "neg_len_true": [], "neg_len_false": [],
    "y_true": [], "y_pred": [], "y_score": [],
    "head_mrr": [], "head_h1": [], "head_h3": [], "head_h10": [],
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
    padded_targets[:len(corrs)] = targets
    if max_t == 0:
        mask = np.zeros((num_envs, 0), dtype=bool)
    else:
        mask = (np.arange(max_t)[None, :] < padded_targets[:, None])
    return targets, mask


def _configure_env_batch(
    env: EnvBase,
    batch: List[Any],
    corrs: List[List[Any]],
    data_depths: Optional[List[int]],
    batch_start_idx: int,
) -> None:
    """Configure environment batch for evaluation with queries and corruptions."""
    from torchrl.envs import ParallelEnv
    from env import BatchedVecEnv
    
    for i, (query, negatives) in enumerate(zip(batch, corrs)):
        sequence = [query] + negatives
        batch_idx = batch_start_idx + i
        pos_depth = data_depths[batch_idx] if data_depths is not None else None
        depths = [pos_depth] + [-1] * len(negatives)
        
        config_kwargs = {
            'mode': 'eval',
            'queries': sequence,
            'labels': [1] + [0] * len(negatives),
            'query_depths': depths,
            'n_episodes': len(sequence),
            'eval_idx': 0,
        }
        
        # Handle different environment types
        if isinstance(env, BatchedVecEnv):
            # BatchedVecEnv: configure directly on the batched environment
            if hasattr(env, 'configure_batch_slot'):
                env.configure_batch_slot(i, **config_kwargs)
            elif hasattr(env, 'configure'):
                env.configure(**config_kwargs)
        elif isinstance(env, ParallelEnv):
            # For ParallelEnv, we need to send configuration to worker processes
            # This is done through the worker's configure method
            try:
                # Access the worker environment and call configure
                # ParallelEnv doesn't expose envs directly, so we store config
                # and apply it on the next reset
                if not hasattr(env, '_pending_configs'):
                    env._pending_configs = [{} for _ in range(env.num_workers)]
                env._pending_configs[i] = config_kwargs
            except Exception:
                pass
        elif hasattr(env, 'envs') and len(env.envs) > i:
            # Direct access to environment list
            inner_env = env.envs[i]
            if hasattr(inner_env, 'configure'):
                inner_env.configure(**config_kwargs)
            else:
                for key, value in config_kwargs.items():
                    setattr(inner_env, key, value)
        else:
            # Single environment
            if hasattr(env, 'configure'):
                env.configure(**config_kwargs)
            else:
                for key, value in config_kwargs.items():
                    setattr(env, key, value)


def _evaluate_with_rl(
    actor: nn.Module,
    env: EnvBase,
    batch: List[Any],
    corrs: List[List[Any]],
    targets: np.ndarray,
    deterministic: bool,
    verbose: bool,
    info_callback: Optional[Callable[[List[Dict[str, Any]]], None]],
    data_depths: Optional[List[int]],
    batch_start_idx: int,
    plot: bool,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    Optional[Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]],
]:
    _configure_env_batch(env, batch, corrs, data_depths, batch_start_idx)
    
    eval_kwargs = dict(
        actor=actor,
        env=env,
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
        ) = evaluate_policy_torchrl(track_logprobs=True, **eval_kwargs)
        plot_payload = (
            logprob_histories, choices_histories,
            steplogprob_histories, state_histories,
        )
    else:
        rewards, lengths, log_probs, mask, proof_successful = evaluate_policy_torchrl(
            track_logprobs=False, **eval_kwargs
        )
        plot_payload = None
    
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
    """Score a (pos + negatives) set using a KGE inference engine.

    Returns log-scores aligned to the (env, episode-slot) layout defined by mask.
    """
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
    """Extract metrics from an evaluation pass and update accumulators."""
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
        
        # Compute average precision with error handling
        try:
            # Check for inf/nan in y_score_part before calling AP
            if np.any(np.isinf(y_score_part)) or np.any(np.isnan(y_score_part)):
                print(f"Warning: Found inf/nan in y_score_part after clipping. "
                      f"inf count: {np.isinf(y_score_part).sum()}, "
                      f"nan count: {np.isnan(y_score_part).sum()}")
                # Replace any remaining inf/nan
                y_score_part = np.nan_to_num(y_score_part, nan=0.0, posinf=1e10, neginf=-1e10)
            
            ap = average_precision_score(y_true_part, y_score_part)
        except ValueError as e:
            print(f"Warning: average_precision_score failed: {e}")
            print(f"  y_true_part shape: {y_true_part.shape}, unique values: {np.unique(y_true_part)}")
            print(f"  y_score_part shape: {y_score_part.shape}, min: {y_score_part.min()}, max: {y_score_part.max()}")
            ap = 0.0  # Default value on error
        
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
    """Pretty-print current batch metrics and rolling totals."""
    def safe_mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else 0.0
    
    def safe_str(val: float) -> str:
        return f"{val:.3f}" if isinstance(val, float) else val
    
    # Batch metrics
    b_pos_rw = safe_mean(batch_metrics["pos_rw"])
    b_neg_rw = safe_mean(batch_metrics["neg_rw"])
    b_pos_len = safe_mean(batch_metrics["pos_len"])
    b_neg_len = safe_mean(batch_metrics["neg_len"])
    b_mrr = safe_mean(batch_metrics["mrr"])
    b_h1 = safe_mean(batch_metrics["h1"])
    b_h3 = safe_mean(batch_metrics["h3"])
    b_h10 = safe_mean(batch_metrics["h10"])
    
    # Global metrics
    g_pos_rw = safe_mean(global_metrics["pos_rw"])
    g_neg_rw = safe_mean(global_metrics["neg_rw"])
    g_pos_len = safe_mean(global_metrics["pos_len"])
    g_neg_len = safe_mean(global_metrics["neg_len"])
    
    head_mrr = safe_mean(global_metrics["head_mrr"])
    head_h1 = safe_mean(global_metrics["head_h1"])
    tail_mrr = safe_mean(global_metrics["tail_mrr"])
    tail_h1 = safe_mean(global_metrics["tail_h1"])
    
    print(f"  Batch:  Pos Rw: {safe_str(b_pos_rw)} | Neg Rw: {safe_str(b_neg_rw)} | "
          f"Pos Len: {safe_str(b_pos_len)} | Neg Len: {safe_str(b_neg_len)}")
    print(f"          MRR: {safe_str(b_mrr)} | H@1: {safe_str(b_h1)} | "
          f"H@3: {safe_str(b_h3)} | H@10: {safe_str(b_h10)}")
    print(f"  Global: Pos Rw: {safe_str(g_pos_rw)} | Neg Rw: {safe_str(g_neg_rw)} | "
          f"Pos Len: {safe_str(g_pos_len)} | Neg Len: {safe_str(g_neg_len)}")
    print(f"          Head MRR: {safe_str(head_mrr)} | Head H@1: {safe_str(head_h1)} | "
          f"Tail MRR: {safe_str(tail_mrr)} | Tail H@1: {safe_str(tail_h1)}")


def _finalize_and_get_results(global_metrics: Dict[str, List[float]]) -> Dict[str, Any]:
    """Compute final statistics from accumulated metrics."""
    def safe_mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else 0.0
    
    def safe_std(values: List[float]) -> float:
        return float(np.std(values)) if values else 0.0
    
    results = {}
    
    # Aggregate metrics
    for key in ["pos_rw", "neg_rw", "pos_len", "neg_len", "pos_lp", "neg_lp"]:
        results[f"{key}_mean"] = safe_mean(global_metrics[key])
        results[f"{key}_std"] = safe_std(global_metrics[key])
    
    # Ranking metrics
    for prefix in ["head", "tail"]:
        for metric in ["mrr", "h1", "h3", "h10"]:
            key = f"{prefix}_{metric}"
            results[f"{key}_mean"] = safe_mean(global_metrics[key])
            results[f"{key}_std"] = safe_std(global_metrics[key])
        ap_key = f"{prefix}_ap"
        results[f"{ap_key}_mean"] = safe_mean(global_metrics[ap_key])
    
    # Overall metrics (average of head and tail)
    for metric in ["mrr", "h1", "h3", "h10"]:
        head_val = results.get(f"head_{metric}_mean", 0.0)
        tail_val = results.get(f"tail_{metric}_mean", 0.0)
        results[f"{metric}_mean"] = (head_val + tail_val) / 2.0
    
    # Classification metrics
    y_true = np.array(global_metrics["y_true"])
    y_pred = np.array(global_metrics["y_pred"])
    
    if len(y_true) > 0:
        results["accuracy"] = float(np.mean(y_true == y_pred))
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        results["precision"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        results["recall"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        
        if results["precision"] + results["recall"] > 0:
            results["f1"] = 2 * results["precision"] * results["recall"] / (results["precision"] + results["recall"])
        else:
            results["f1"] = 0.0
    
    return results


# ---------------------------------------------------------------------------
# High-level evaluator for head/tail corruptions and ranking metrics
# ---------------------------------------------------------------------------

def eval_corruptions_torchrl(
    actor: nn.Module,
    env: EnvBase,
    data: List[Any],
    sampler: Any,
    n_corruptions: Optional[int] = None,
    deterministic: bool = True,
    verbose: int = 1,
    plot: bool = False,
    kge_inference_engine = None,
    evaluation_mode: str = 'rl_only',
    corruption_scheme: List[str] | None = None,
    info_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    data_depths: Optional[List[int]] = None,
    hybrid_kge_weight: float = 2.0,
    hybrid_rl_weight: float = 1.0,
    hybrid_success_only: bool = True,
) -> Dict[str, Any]:
    """Evaluate a TorchRL actor by ranking positive queries vs. corruptions.

    For each positive query in `data`, generates head/tail corruptions using
    `sampler` and assigns scores according to `evaluation_mode`:

    - "rl_only": score = RL log-probability (failed proofs heavily penalized)
    - "kge_only": score = log(KGE score)
    - "hybrid": score = α·log(KGE) + (optionally) β·RL for successful proofs

    Parameters
    ----------
    actor : nn.Module
        TorchRL actor module (ProbabilisticActor).
    env : EnvBase
        TorchRL environment (ParallelEnv or single EnvBase).
    data : List[Any]
        List of positive queries to evaluate.
    sampler : Any
        Negative sampler for generating corruptions.
    n_corruptions : Optional[int]
        Number of negatives per type (None = all).
    deterministic : bool
        Whether to use deterministic actions.
    verbose : int
        Verbosity level.
    plot : bool
        Whether to collect trajectory data for plotting.
    kge_inference_engine : optional
        KGE model wrapper with `.predict_batch()` method.
    evaluation_mode : str
        One of 'rl_only', 'kge_only', 'hybrid'.
    corruption_scheme : List[str]
        Subset of ["head", "tail"] to evaluate.
    info_callback : callable
        Callback for info dicts after each step.
    data_depths : List[int]
        Depth values for queries.
    hybrid_kge_weight : float
        Weight for KGE scores in hybrid mode.
    hybrid_rl_weight : float
        Weight for RL scores in hybrid mode.
    hybrid_success_only : bool
        Add RL bonus only for successful proofs.

    Returns
    -------
    Dict[str, Any]
        Aggregated metrics (MRR, Hits@K, accuracy, etc.)
    """
    if evaluation_mode not in ("rl_only", "kge_only", "hybrid"):
        raise ValueError(f"Invalid evaluation_mode: {evaluation_mode}")
    if evaluation_mode != "rl_only" and kge_inference_engine is None:
        raise ValueError(f"kge_inference_engine required for mode: '{evaluation_mode}'")
    
    # Determine number of environments
    if hasattr(env, 'batch_size') and len(env.batch_size) > 0:
        num_envs = int(env.batch_size[0])
    elif hasattr(env, 'num_workers'):
        num_envs = int(env.num_workers)
    else:
        num_envs = 1
    
    if n_corruptions == -1:
        n_corruptions = None
    
    device = next(actor.parameters()).device
    
    if verbose:
        print(f"Evaluating {len(data)} queries in '{evaluation_mode}' mode.")
        print(f"N corruptions per query (per type): "
              f"{'All' if n_corruptions is None else n_corruptions} | Envs: {num_envs}")
    
    global_metrics = _init_global_metrics()
    
    total_batches = (len(data) + num_envs - 1) // num_envs
    for b, start in enumerate(range(0, len(data), num_envs)):
        time_start = time.time()
        if verbose:
            end_idx = min(start + num_envs - 1, len(data) - 1)
            print(f"\n--- Batch {b+1}/{total_batches} (Queries {start}-{end_idx}) ---")
        
        batch = data[start:start + num_envs]
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
                (
                    rewards, lengths, log_probs, eval_mask,
                    proof_successful, plot_payload,
                ) = _evaluate_with_rl(
                    actor=actor,
                    env=env,
                    batch=batch,
                    corrs=corrs,
                    targets=targets,
                    deterministic=deterministic,
                    verbose=verbose > 1,
                    info_callback=info_callback,
                    data_depths=data_depths,
                    batch_start_idx=start,
                    plot=plot,
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


# Re-export commonly used utilities
__all__ = [
    'evaluate_policy_torchrl',
    'eval_corruptions_torchrl',
    'kge_eval',
]
