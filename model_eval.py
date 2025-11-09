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
# Internal helpers
# ---------------------------------------------------------------------------

def _infer_num_envs(env: EnvBase) -> int:
    if hasattr(env, "batch_size") and len(env.batch_size) > 0:
        return int(env.batch_size[0])
    if hasattr(env, "num_workers"):
        return int(env.num_workers)
    return 1


def _compute_episode_targets(
    n_envs: int,
    n_eval_episodes: int,
    target_episodes: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    if target_episodes is None:
        offsets = np.arange(max(n_envs, 1), dtype=int)
        targets = ((n_eval_episodes + offsets) // max(n_envs, 1)).astype(int)
    else:
        targets = np.asarray(target_episodes, dtype=int)
    padded = np.zeros(max(n_envs, len(targets)), dtype=int)
    padded[:len(targets)] = targets
    max_t = int(padded.max()) if padded.size else 0
    total = int(padded.sum())
    return targets, padded, max_t, total


def _sync_env_episode_targets(env: EnvBase, padded_targets: np.ndarray) -> None:
    from torchrl.envs import ParallelEnv
    from env import BatchedEnv

    if isinstance(env, (ParallelEnv, BatchedEnv)):
        if hasattr(env, "_episode_target"):
            env._episode_target[:] = padded_targets
        if hasattr(env, "_episode_count"):
            env._episode_count[:] = 0
        if hasattr(env, "active_envs"):
            env.active_envs[:] = True
        return

    if hasattr(env, "_episode_target"):
        env._episode_target[:] = padded_targets
    if hasattr(env, "_episode_count"):
        env._episode_count[:] = 0
    if hasattr(env, "active_envs"):
        env.active_envs[:] = True


def _flatten_env_value(
    value: Optional[torch.Tensor],
    device: torch.device,
    n_envs: int,
) -> torch.Tensor:
    if value is None:
        return torch.zeros(n_envs, device=device)
    tensor = value.to(device)
    if tensor.ndim == 0:
        return tensor.repeat(n_envs)
    if tensor.shape[0] != n_envs:
        tensor = tensor.reshape(n_envs, -1)
    else:
        tensor = tensor.view(n_envs, -1)
    return tensor[:, 0]


def _extract_tensor_from_td(
    td: TensorDict,
    key: str,
    device: torch.device,
    n_envs: int,
) -> Optional[torch.Tensor]:
    source = None
    if key in td.keys():
        source = td.get(key)
    elif "next" in td.keys() and key in td["next"].keys():
        source = td["next"].get(key)
    if source is None:
        return None
    return _flatten_env_value(source, device, n_envs)


def _normalize_action_tensor(action_output: torch.Tensor, n_envs: int) -> torch.Tensor:
    if action_output is None:
        raise ValueError("Actor did not produce 'action' in output TensorDict")
    actions = action_output
    if actions.ndim >= 3:
        actions = actions[:, -1, :]
    if actions.ndim >= 2:
        actions = torch.argmax(actions, dim=-1)
    actions = actions.to(torch.long).view(-1)
    if actions.numel() == 1 and n_envs > 1:
        actions = actions.repeat(n_envs)
    return actions[:n_envs]


def _extract_log_prob(
    td_with_action: TensorDict,
    device: torch.device,
    n_envs: int,
) -> torch.Tensor:
    lp_tensor = td_with_action.get("sample_log_prob", None)
    if lp_tensor is None:
        return torch.zeros(n_envs, device=device)
    tensor = lp_tensor.to(device)
    if tensor.ndim == 2:
        tensor = tensor[:, -1]
    tensor = tensor.view(-1)
    if tensor.numel() == 1 and n_envs > 1:
        tensor = tensor.repeat(n_envs)
    return tensor[:n_envs]


def _apply_action_mask_vectorized(
    actions: torch.Tensor,
    action_mask: torch.Tensor,
    log_probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if action_mask.ndim == 1:
        flat_mask = action_mask.unsqueeze(0).expand(actions.shape[0], -1)
    elif action_mask.shape[0] != actions.shape[0]:
        flat_mask = action_mask.reshape(actions.shape[0], -1)
    else:
        flat_mask = action_mask.view(actions.shape[0], -1)

    flat_mask = flat_mask.to(torch.bool)
    action_dim = flat_mask.shape[-1]
    clipped = actions.clamp(min=0, max=max(action_dim - 1, 0))
    gather_idx = clipped.unsqueeze(-1)
    valid_taken = torch.gather(flat_mask, 1, gather_idx).squeeze(-1)
    within_bounds = (actions >= 0) & (actions < action_dim)
    valid = valid_taken & within_bounds
    invalid = ~valid

    any_valid = flat_mask.any(dim=-1)
    fallback = torch.argmax(flat_mask.int(), dim=-1)
    fallback = torch.where(any_valid, fallback, torch.zeros_like(fallback))
    adjusted_actions = torch.where(invalid, fallback, clipped)

    valid_counts = flat_mask.sum(dim=-1).clamp(min=1)
    replacement_lp = torch.where(
        any_valid,
        -torch.log(valid_counts.float()),
        torch.full_like(valid_counts.float(), float("-inf")),
    )
    adjusted_log_probs = torch.where(invalid, replacement_lp, log_probs)
    return adjusted_actions, adjusted_log_probs


def _build_info_dicts(
    done_mask: torch.Tensor,
    rewards: torch.Tensor,
    lengths: torch.Tensor,
    success: torch.Tensor,
    td: TensorDict,
    device: torch.device,
    n_envs: int,
) -> List[Dict[str, Any]]:
    if not torch.any(done_mask):
        return [{} for _ in range(n_envs)]

    labels = _extract_tensor_from_td(td, "label", device, n_envs)
    depths = _extract_tensor_from_td(td, "query_depth", device, n_envs)
    episode_idx = _extract_tensor_from_td(td, "episode_idx", device, n_envs)

    done_np = done_mask.detach().cpu().numpy().astype(bool)
    reward_np = rewards.detach().cpu().numpy()
    length_np = lengths.detach().cpu().numpy()
    success_np = success.detach().cpu().numpy().astype(bool)
    label_np = labels.detach().cpu().numpy() if labels is not None else None
    depth_np = depths.detach().cpu().numpy() if depths is not None else None
    episode_np = episode_idx.detach().cpu().numpy() if episode_idx is not None else None

    infos = [{} for _ in range(n_envs)]
    indices = np.nonzero(done_np)[0].tolist()
    for idx in indices:
        payload: Dict[str, Any] = {
            "episode": {
                "r": float(reward_np[idx]),
                "l": int(length_np[idx]),
            },
            "is_success": bool(success_np[idx]),
        }
        if label_np is not None:
            payload["label"] = int(label_np[idx])
        if depth_np is not None:
            payload["query_depth"] = int(depth_np[idx])
        if episode_np is not None:
            payload["episode_idx"] = int(episode_np[idx])
        infos[idx] = payload
    return infos


class _TrajectoryRecorder:
    """Utility to keep log-prob and state histories when plotting."""

    def __init__(self, env: EnvBase, n_envs: int):
        index_manager = getattr(env, "index_manager", None)
        if index_manager is None and hasattr(env, "envs") and len(getattr(env, "envs", [])) > 0:
            inner_env = env.envs[0]
            index_manager = getattr(inner_env, "index_manager", None)

        self.enabled = index_manager is not None
        self.index_manager = index_manager
        self.n_envs = n_envs

        self.current_state_histories = [[] for _ in range(n_envs)]
        self.current_choices_histories = [[] for _ in range(n_envs)]
        self.current_steplogprob_histories = [[] for _ in range(n_envs)]

        self.episode_logprob_histories: List[np.ndarray] = []
        self.episode_choices_histories: List[np.ndarray] = []
        self.episode_steplogprob_histories: List[np.ndarray] = []
        self.episode_state_histories: List[np.ndarray] = []

    def _compute_num_choices(self, obs_td: TensorDict) -> torch.Tensor:
        derived = obs_td.get("derived_sub_indices", None)
        if derived is None:
            return torch.zeros(self.n_envs, dtype=torch.int64)
        mask = derived.sum(dim=(-1, -2)) != 0
        return mask.sum(dim=-1).to(torch.int64)

    def record_state(self, obs_td: TensorDict, active_mask: torch.Tensor) -> None:
        if not self.enabled or "sub_index" not in obs_td.keys():
            return
        num_choices = self._compute_num_choices(obs_td)
        sub_index = obs_td["sub_index"].detach().cpu().numpy()
        active_indices = active_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
        for idx in active_indices:
            state_np = sub_index[idx]
            state_str = self.index_manager.state_subindex_to_str(state_np, truncate=True)
            self.current_state_histories[idx].append(state_str)
            self.current_choices_histories[idx].append(int(num_choices[idx]))

    def record_step_log_probs(self, log_probs: torch.Tensor, active_mask: torch.Tensor) -> None:
        if not self.enabled:
            return
        log_np = log_probs.detach().cpu().numpy()
        active_indices = active_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
        for idx in active_indices:
            value = log_np[idx] if np.ndim(log_np) > 0 else float(log_np)
            self.current_steplogprob_histories[idx].append(float(value))

    def finalize(self, done_indices: torch.Tensor, td: TensorDict) -> None:
        if not self.enabled or done_indices.numel() == 0:
            return
        next_td = td.get("next") if "next" in td.keys() else td
        sub_index = None
        if "sub_index" in next_td.keys():
            sub_index = next_td["sub_index"].detach().cpu().numpy()

        for env_idx in done_indices.detach().cpu().tolist():
            if not self.current_steplogprob_histories[env_idx]:
                continue

            self.current_choices_histories[env_idx].append(0)
            self.current_steplogprob_histories[env_idx].append(0.0)

            self.episode_logprob_histories.append(
                np.cumsum(self.current_steplogprob_histories[env_idx])
            )
            self.episode_steplogprob_histories.append(
                np.array(self.current_steplogprob_histories[env_idx])
            )
            self.episode_choices_histories.append(
                np.array(self.current_choices_histories[env_idx])
            )

            if sub_index is not None:
                final_state = sub_index[env_idx]
                final_state_str = self.index_manager.state_subindex_to_str(final_state, truncate=True)
                self.current_state_histories[env_idx].append(final_state_str)
                self.episode_state_histories.append(
                    np.array(self.current_state_histories[env_idx])
                )

            self.current_state_histories[env_idx].clear()
            self.current_choices_histories[env_idx].clear()
            self.current_steplogprob_histories[env_idx].clear()

    def payload(self) -> Tuple[
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
    ]:
        return (
            self.episode_logprob_histories,
            self.episode_choices_histories,
            self.episode_steplogprob_histories,
            self.episode_state_histories,
        )


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
    actor.eval()
    n_envs = _infer_num_envs(env)
    device = next(actor.parameters()).device

    targets, padded_targets, max_t, total = _compute_episode_targets(
        n_envs, n_eval_episodes, target_episodes
    )
    if verbose:
        avg_target = targets.mean() if targets.size > 0 else 0.0
        print(f"\nEvaluating {total} episodes on {n_envs} envs (avg target: {avg_target:.2f})")

    _sync_env_episode_targets(env, padded_targets)

    rewards = torch.zeros((n_envs, max_t), device=device)
    lengths = torch.zeros((n_envs, max_t), dtype=torch.int32, device=device)
    logps = torch.zeros((n_envs, max_t), device=device)
    proof_successful = torch.zeros((n_envs, max_t), dtype=torch.bool, device=device)

    counts = torch.zeros(n_envs, dtype=torch.int32, device=device)
    current_rew = torch.zeros(n_envs, device=device)
    current_len = torch.zeros(n_envs, dtype=torch.int32, device=device)
    current_lp = torch.zeros(n_envs, device=device)

    recorder = _TrajectoryRecorder(env, n_envs) if track_logprobs else None

    td = env.reset()
    padded_targets_t = torch.as_tensor(padded_targets, device=device)
    active_mask = counts < padded_targets_t

    while bool(torch.any(active_mask)):
        obs_td = td.get("next") if "next" in td.keys() else td
        if recorder is not None:
            recorder.record_state(obs_td, active_mask)

        policy_td = actor(obs_td)
        actions = _normalize_action_tensor(policy_td.get("action"), n_envs).to(device)
        log_probs = _extract_log_prob(policy_td, device, n_envs)

        action_mask = obs_td.get("action_mask", None)
        if action_mask is not None:
            actions, log_probs = _apply_action_mask_vectorized(
                actions, action_mask.to(device), log_probs
            )

        batch_size = getattr(env, "batch_size", torch.Size([n_envs]))
        try:
            batched_actions = actions.view(batch_size)
        except Exception:
            batched_actions = actions.view(-1)
        action_td = TensorDict({"action": batched_actions}, batch_size=batch_size)
        
        # Use step_and_maybe_reset to automatically handle episode resets
        step_td_complete, next_obs_td = env.step_and_maybe_reset(action_td)
        td = step_td_complete  # Contains the step result with reward/done
        # Update td for next iteration to use the reset observation if needed
        if "next" in next_obs_td.keys():
            td = next_obs_td
        else:
            td = TensorDict({"next": next_obs_td}, batch_size=batch_size)

        next_td = step_td_complete.get("next") if "next" in step_td_complete.keys() else step_td_complete
        reward_tensor = next_td.get("reward")
        done_tensor = next_td.get("done")
        if reward_tensor is None or done_tensor is None:
            raise ValueError(f"TensorDict missing 'reward' or 'done'. Keys: {list(next_td.keys())}")

        rewards_step = _flatten_env_value(reward_tensor, device, n_envs)
        dones = _flatten_env_value(done_tensor, device, n_envs).to(torch.bool)

        mask_float = active_mask.to(rewards_step.dtype)
        current_rew += rewards_step * mask_float
        current_len += active_mask.to(current_len.dtype)
        current_lp += log_probs * mask_float

        if recorder is not None:
            recorder.record_step_log_probs(log_probs, active_mask)

        done_and_active = dones & active_mask
        if torch.any(done_and_active):
            done_idx = torch.nonzero(done_and_active, as_tuple=False).squeeze(-1)
            slot_idx = counts[done_idx].long()

            rewards[done_idx, slot_idx] = current_rew[done_idx]
            lengths[done_idx, slot_idx] = current_len[done_idx]
            logps[done_idx, slot_idx] = current_lp[done_idx]

            success_tensor = _extract_tensor_from_td(td, "is_success", device, n_envs)
            if success_tensor is None:
                success_tensor = torch.zeros(n_envs, dtype=torch.bool, device=device)
            proof_successful[done_idx, slot_idx] = success_tensor.to(torch.bool)[done_idx]

            if info_callback is not None:
                infos = _build_info_dicts(
                    done_and_active,
                    current_rew.clone(),
                    current_len.clone(),
                    success_tensor.to(torch.bool),
                    td,
                    device,
                    n_envs,
                )
                info_callback(infos)

            if recorder is not None:
                recorder.finalize(done_idx, td)

            counts[done_idx] += 1
            current_rew[done_idx] = 0
            current_len[done_idx] = 0
            current_lp[done_idx] = 0

        if verbose:
            print(f"\rEpisodes done: {int(counts.sum())}/{total}", end="", flush=True)

        active_mask = counts < padded_targets_t

    if verbose:
        print("\r" + " " * 80 + "\r", end="")

    mask = (
        torch.arange(max_t, device=device)[None, :]
        < torch.as_tensor(padded_targets, device=device)[:, None]
    )

    trim = len(target_episodes) if target_episodes is not None else n_envs
    outputs = [rewards, lengths, logps, mask, proof_successful]
    rewards_np, lengths_np, logps_np, mask_np, success_np = [
        tensor[:trim].cpu().numpy() for tensor in outputs
    ]

    if recorder is not None:
        (
            episode_logprob_histories,
            episode_choices_histories,
            episode_steplogprob_histories,
            episode_state_histories,
        ) = recorder.payload()
        return (
            rewards_np,
            lengths_np,
            logps_np,
            mask_np,
            success_np,
            episode_logprob_histories,
            episode_choices_histories,
            episode_steplogprob_histories,
            episode_state_histories,
        )
    
    return rewards_np, lengths_np, logps_np, mask_np, success_np

   


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


def _evaluate_ranking_metrics(
    actor: nn.Module,
    env: EnvBase,
    queries: torch.Tensor,
    sampler: Any,
    n_corruptions: int = 10,
    corruption_modes: List[str] = None,
    deterministic: bool = True,
    verbose: bool = False,
    info_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
) -> Dict[str, Any]:
    """Evaluate ranking metrics by generating negatives and computing logprobs.
    
    This function:
    1. Takes a batch of positive queries (triples)
    2. Generates negatives using the sampler (head and/or tail corruptions)
    3. Distributes positive + negatives across the batched environment
    4. Evaluates each triple to get log probabilities
    5. Computes ranking metrics (MRR, Hits@K, etc.)
    
    Parameters
    ----------
    actor : nn.Module
        TorchRL actor module (e.g., ProbabilisticActor).
    env : EnvBase
        TorchRL batched environment (BatchedEnv or similar).
    queries : torch.Tensor
        Positive query triples of shape [N, 3] where each row is [relation, head, tail].
    sampler : Any
        Sampler instance with corrupt() method for generating negatives.
    n_corruptions : int, default 10
        Number of negative samples to generate per query.
    corruption_modes : List[str], optional
        List of corruption types to evaluate. Default is ['head', 'tail'].
    deterministic : bool, default True
        Whether to use deterministic policy actions.
    verbose : bool, default False
        Print progress information.
    info_callback : callable, optional
        Callback for info dicts after each step.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - head_metrics: Dict with head corruption metrics (MRR, Hits@K, etc.)
        - tail_metrics: Dict with tail corruption metrics
        - overall_metrics: Dict with averaged metrics across both corruption types
        - raw_results: Dict with raw evaluation results per corruption type
    """
    if corruption_modes is None:
        corruption_modes = ['head', 'tail']
    
    device = queries.device
    n_queries = queries.shape[0]
    batch_size = _infer_num_envs(env)
    
    if verbose:
        print(f"Evaluating {n_queries} queries with {n_corruptions} corruptions each")
        print(f"Corruption modes: {corruption_modes}")
        print(f"Batch size: {batch_size}")
    
    # Store results per corruption type
    all_results = {}
    global_metrics = _init_global_metrics()
    
    for corruption_type in corruption_modes:
        if verbose:
            print(f"\n=== Evaluating {corruption_type} corruptions ===")
        
        # Generate negatives for all queries
        negatives = sampler.corrupt(
            queries,
            num_negatives=n_corruptions,
            mode=corruption_type,
            device=device,
            filter=True,
            unique=True,
        )  # [N, K, 3]
        
        if verbose:
            print(f"Generated {negatives.shape[1]} negatives per query")
        
        # Process queries in batches
        batch_results = []
        batch_metrics = _init_batch_metrics()
        
        for batch_start in range(0, n_queries, batch_size):
            batch_end = min(batch_start + batch_size, n_queries)
            batch_queries = queries[batch_start:batch_end]  # [B, 3]
            batch_negs = negatives[batch_start:batch_end]  # [B, K, 3]
            actual_batch_size = batch_queries.shape[0]
            
            # Combine positive and negatives: [B, 1+K, 3]
            all_triples = torch.cat([
                batch_queries.unsqueeze(1),  # [B, 1, 3]
                batch_negs,  # [B, K, 3]
            ], dim=1)  # [B, 1+K, 3]
            
            n_episodes = all_triples.shape[1]  # 1 + K
            
            # Convert triples to query format expected by environment
            # Get original query structure from environment
            if hasattr(env, 'queries'):
                query_shape = env.queries.shape[1:]  # [L, max_arity+1]
            else:
                # Default shape for simple queries
                query_shape = (1, 4)  # Single atom, max_arity+1
            
            # Create query tensor
            expanded_queries = torch.zeros(
                actual_batch_size, 
                n_episodes,
                query_shape[0],  # L (sequence length)
                query_shape[1],  # max_arity+1
                dtype=torch.long,
                device=device
            )
            
            # Fill in the queries (assume single-atom queries with format [r, h, t])
            expanded_queries[:, :, 0, :3] = all_triples
            
            # Create labels: positive (1) at index 0, negatives (0) at indices 1..K
            labels = torch.zeros(actual_batch_size, n_episodes, dtype=torch.long, device=device)
            labels[:, 0] = 1
            
            # Create depths (use -1 for unknown depth for negatives)
            depths = torch.full((actual_batch_size, n_episodes), -1, dtype=torch.long, device=device)
            
            # Flatten batch for environment
            flat_queries = expanded_queries.reshape(-1, query_shape[0], query_shape[1])
            flat_labels = labels.reshape(-1)
            flat_depths = depths.reshape(-1)
            
            # Pad or truncate to match batch_size
            if flat_queries.shape[0] < batch_size:
                # Pad with zeros
                pad_size = batch_size - flat_queries.shape[0]
                flat_queries = torch.cat([
                    flat_queries,
                    torch.zeros(pad_size, *query_shape, dtype=torch.long, device=device)
                ], dim=0)
                flat_labels = torch.cat([
                    flat_labels,
                    torch.zeros(pad_size, dtype=torch.long, device=device)
                ], dim=0)
                flat_depths = torch.cat([
                    flat_depths,
                    torch.full((pad_size,), -1, dtype=torch.long, device=device)
                ], dim=0)
            elif flat_queries.shape[0] > batch_size:
                flat_queries = flat_queries[:batch_size]
                flat_labels = flat_labels[:batch_size]
                flat_depths = flat_depths[:batch_size]
            
            # Reconfigure environment
            env.queries = flat_queries
            env.labels = flat_labels
            env.query_depths = flat_depths
            env.n_queries = flat_queries.shape[0]
            
            # Set episode targets
            episode_targets = torch.full(
                (batch_size,), 
                n_episodes, 
                dtype=torch.long,
                device=device
            )
            # Only first actual_batch_size environments should run
            episode_targets[actual_batch_size:] = 0
            
            if hasattr(env, '_episode_target'):
                env._episode_target = episode_targets
            if hasattr(env, '_episode_count'):
                env._episode_count = torch.zeros(batch_size, dtype=torch.long, device=device)
            if hasattr(env, 'active_envs'):
                env.active_envs = torch.ones(batch_size, dtype=torch.bool, device=device)
                env.active_envs[actual_batch_size:] = False
            
            # Evaluate
            total_episodes = actual_batch_size * n_episodes
            results = evaluate_policy_torchrl(
                actor=actor,
                env=env,
                n_eval_episodes=total_episodes,
                deterministic=deterministic,
                target_episodes=episode_targets.cpu().numpy(),
                verbose=0,
                track_logprobs=False,
                info_callback=info_callback,
            )
            
            rewards, lengths, log_probs, mask, proof_successful = results
            
            # Reshape results back to [B, n_episodes]
            rewards = rewards[:actual_batch_size, :n_episodes]
            lengths = lengths[:actual_batch_size, :n_episodes]
            log_probs = log_probs[:actual_batch_size, :n_episodes]
            mask = mask[:actual_batch_size, :n_episodes]
            proof_successful = proof_successful[:actual_batch_size, :n_episodes]
            
            # Extract and accumulate metrics for this batch
            _extract_and_accumulate_metrics(
                batch_metrics=batch_metrics,
                global_metrics=global_metrics,
                corruption_type=corruption_type,
                mask=mask,
                proof_successful=proof_successful,
                log_probs=log_probs,
                rewards=rewards,
                lengths=lengths,
            )
            
            batch_results.append({
                'rewards': rewards,
                'lengths': lengths,
                'log_probs': log_probs,
                'mask': mask,
                'proof_successful': proof_successful,
            })
        
        # Store results for this corruption type
        all_results[corruption_type] = {
            'batch_metrics': batch_metrics,
            'batch_results': batch_results,
        }
        
        if verbose:
            _report_batch_metrics(batch_metrics, corruption_type)
    
    # Compute final aggregated metrics
    final_results = _finalize_and_get_results(global_metrics)
    final_results['raw_results'] = all_results
    
    # Add per-corruption-type metrics
    for corruption_type in corruption_modes:
        if corruption_type in all_results:
            metrics = all_results[corruption_type]['batch_metrics']
            final_results[f'{corruption_type}_metrics'] = {
                'mrr': float(np.mean(metrics['mrr'])) if metrics['mrr'] else 0.0,
                'h1': float(np.mean(metrics['h1'])) if metrics['h1'] else 0.0,
                'h3': float(np.mean(metrics['h3'])) if metrics['h3'] else 0.0,
                'h10': float(np.mean(metrics['h10'])) if metrics['h10'] else 0.0,
            }
    
    if verbose:
        print(f"\n=== Final Results ===")
        for key in ['mrr_mean', 'h1_mean', 'h3_mean', 'h10_mean']:
            if key in final_results:
                print(f"  {key}: {final_results[key]:.4f}")
    
    return final_results


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


def _report_batch_metrics(
    batch_metrics: Dict[str, List[float]],
    corruption_type: str,
) -> None:
    """Print summary statistics for batch metrics."""
    if not batch_metrics['mrr']:
        return
    
    def safe_mean(values):
        return float(np.mean(values)) if values else 0.0
    
    b_pos_rw = safe_mean(batch_metrics["pos_rw"])
    b_neg_rw = safe_mean(batch_metrics["neg_rw"])
    b_pos_len = safe_mean(batch_metrics["pos_len"])
    b_neg_len = safe_mean(batch_metrics["neg_len"])
    b_mrr = safe_mean(batch_metrics["mrr"])
    b_h1 = safe_mean(batch_metrics["h1"])
    b_h3 = safe_mean(batch_metrics["h3"])
    b_h10 = safe_mean(batch_metrics["h10"])
    
    print(f"  Completed {corruption_type} corruptions")
    print(f"    Pos Rw: {b_pos_rw:.3f} | Neg Rw: {b_neg_rw:.3f} | "
          f"Pos Len: {b_pos_len:.3f} | Neg Len: {b_neg_len:.3f}")
    print(f"    MRR: {b_mrr:.4f} | H@1: {b_h1:.4f} | "
          f"H@3: {b_h3:.4f} | H@10: {b_h10:.4f}")


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


# Re-export commonly used utilities
__all__ = [
    'evaluate_policy_torchrl',
    '_evaluate_ranking_metrics',
]
