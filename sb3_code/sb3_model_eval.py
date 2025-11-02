"""
model_eval.py — Evaluation utilities for RL + NeSy policies


This module contains two layers of evaluation utilities for a policy trained on
logical reasoning environments that expose a *vectorized* Gymnasium `VecEnv`:


1) `evaluate_policy(...)` — a fast, tensor-first evaluator that rolls out a
(possibly batched) policy on a `VecEnv`, collecting per-episode rewards,
lengths and *log-probabilities*.


2) `eval_corruptions(...)` — a higher-level evaluator for link-prediction style
tasks. For each positive query, it generates *head/tail corruptions* using a
negative sampler and scores all candidates using:
- RL only (agent policy),
- KGE only (knowledge-graph embedding model), or
- a Hybrid fusion (KGE everywhere + RL bonus on successful proofs).


It also computes a rich set of classification and ranking metrics (Accuracy,
Precision/Recall/F1, Average Precision; MRR and Hits@K) and can optionally
assemble plotting-ready trajectories for analysis.


Assumptions / Requirements
--------------------------
• The environment is a *custom* vectorized wrapper (see `custom_dummy_env`) with
these attributes used during evaluation:
- `env._episode_target` (np.ndarray[int]): episodes to run per sub-env
- `env._episode_count` (np.ndarray[int]): completed episodes per sub-env
- `env.active_envs` (np.ndarray[bool]): which sub-envs are active
- `env.get_attr("index_manager")[0]` (optional, for pretty state logs)
`evaluate_policy()` asserts `getattr(env, "type_", None) == "custom_dummy"`.


• The policy is a Stable-Baselines3 policy-like object: it is called via
`model.policy(obs_tensor, deterministic)` and returns `(actions, values, logp)`.


• Observations follow SB3's multi-input convention and are converted via
`stable_baselines3.common.on_policy_algorithm.obs_as_tensor`.


• For corruption evaluation, a negative sampler exposes:
`get_negatives_from_states_separate([[q] for q in batch], device, num_negs)`
(and the env supports sequential evaluation of a list of queries per sub-env).


Tip: This file deliberately stays *framework-agnostic* outside of SB3 to allow
plugging different policies and samplers.
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Union, Optional, Callable
import os
import time
import random

import gymnasium as gym
import numpy as np
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

from stable_baselines3.common import type_aliases
from stable_baselines3.common.on_policy_algorithm import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv


# ---------------------------------------------------------------------------
# Low-level, fast vectorized evaluator
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    target_episodes: np.ndarray | None = None,
    verbose: int = 0,
    track_logprobs: bool = False,
    info_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Roll out a policy on a *vectorized* env and aggregate per-episode stats.

    Parameters
    ----------
    model : type_aliases.PolicyPredictor
        SB3 policy-like model. Must expose `policy(obs, deterministic)` returning
        `(actions, values, log_probs)`.
    env : Union[gym.Env, VecEnv]
        Evaluation environment. If not already a `VecEnv`, it will be wrapped in
        `DummyVecEnv`. **Required** attribute: `type_ == "custom_dummy"` and
        internal buffers (`_episode_target`, `_episode_count`, `active_envs`).
    n_eval_episodes : int, default 10
        Total target episodes across all sub-envs (if `target_episodes` is not
        provided).
    deterministic : bool, default True
        Whether to use deterministic policy actions.
    target_episodes : np.ndarray | None, default None
        Optional per-env episode targets (shape `(n_envs,)`). If `None`, targets
        are distributed as evenly as possible.
    verbose : int, default 0
        Print a running counter when > 0.
    track_logprobs : bool, default False
        If True, collects *per-step* log-prob trajectories and state strings.
        (Kept for debugging/plotting; not returned in this simplified API.)
    info_callback : callable, optional
        If provided, called after each environment step with the list of info
        dicts returned by the vectorized environment (useful for custom logging).

    Returns
    -------
    rewards : np.ndarray, shape (L, T_max)
        Sum of rewards per finished episode slot for each env.
    lengths : np.ndarray, shape (L, T_max)
        Episode lengths per slot.
    logps : np.ndarray, shape (L, T_max)
        Accumulated action log-probabilities per episode.
    mask : np.ndarray, shape (L, T_max), dtype=bool
        Mask marking valid episode slots per env.
    proof_successful : np.ndarray, shape (L, T_max), dtype=bool
        Whether the env reported success for each finished episode.

    Notes
    -----
    • This is *tensor-first*: it computes per-env tensors on the GPU when
      possible and finalizes in vectorized fashion.
    • The function expects the env to expose numpy action spaces and to return
      SB3-compatible `(obs, reward, done, info)` per step (via `VecEnv.step`).
    """
    # --- ensure VecEnv ---
    if not isinstance(env, VecEnv):
        if verbose:
            print("Warning: wrapping single env in DummyVecEnv")
        env = DummyVecEnv([lambda: env])
    assert getattr(env, "type_", None) == "custom_dummy", "Requires custom_dummy VecEnv"

    device = model.device
    n_envs = env.num_envs

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

    # tell env
    env._episode_target[:] = padded_targets
    env._episode_count[:]  = 0
    env.active_envs[:]     = True

    # --- buffers on device ---
    rewards          = torch.zeros((n_envs, max_t), device=device)
    lengths          = torch.zeros_like(rewards, dtype=torch.int32)
    logps            = torch.zeros_like(rewards)
    proof_successful = torch.zeros_like(rewards, dtype=torch.bool)

    counts      = torch.zeros(n_envs, dtype=torch.int32, device=device)
    current_rew = torch.zeros(n_envs, device=device)
    current_len = torch.zeros(n_envs, dtype=torch.int32, device=device)
    current_lp  = torch.zeros(n_envs, device=device)

    # Optional histories for plotting/debug (not returned)
    episode_logprob_histories    : list[np.ndarray] = []
    episode_choices_histories    : list[np.ndarray] = []
    episode_steplogprob_histories: list[np.ndarray] = []
    episode_state_histories      : list[np.ndarray] = []
    if track_logprobs:
        current_steplogprob_histories: list[list[float]] = [[] for _ in range(n_envs)]
        current_choices_histories: list[list[int]] = [[] for _ in range(n_envs)]
        current_state_histories: list[list[str]] = [[] for _ in range(n_envs)]
        index_manager = env.get_attr("index_manager")[0]

    # env needs numpy actions
    action_shape = env.action_space.shape
    full_actions = np.zeros((n_envs, *action_shape), dtype=env.action_space.dtype)

    observations = env.reset()
    padded_targets_t = torch.as_tensor(padded_targets, device=device)

    while torch.any(counts < padded_targets_t).item():
        active_mask_t = counts < padded_targets_t
        active_idx    = torch.where(active_mask_t)[0]
        active_np     = active_idx.cpu().numpy()

        # slice obs to active envs only
        if isinstance(observations, dict):
            obs_active = {k: v[active_np] for k, v in observations.items()}
        else:
            obs_active = observations[active_np]

        obs_tensor = obs_as_tensor(obs_active, device)
        acts_tensor, _, lp_tensor = model.policy(obs_tensor, deterministic=deterministic)

        # track histories (before step)
        if track_logprobs:
            num_choices = (obs_tensor["derived_sub_indices"].sum(dim=(-1, -2)) != 0).sum(dim=-1).cpu().numpy()
            all_sub_indices = observations["sub_index"]
            for i, env_i in enumerate(active_np):
                subidx = all_sub_indices[env_i].squeeze(0)                # (P, A+1)
                state_str = index_manager.state_subindex_to_str(subidx,   # <- make the string now
                                                                truncate=True)
                current_state_histories[env_i].append(state_str)          # store the *string*
                current_choices_histories[env_i].append(num_choices[i])

        current_lp[active_idx] += lp_tensor

        full_actions[active_np] = acts_tensor.detach().cpu().numpy()
        new_obs, rews_np, dones_np, infos = env.step(full_actions)

        rews_t  = torch.as_tensor(rews_np, device=device, dtype=torch.float32)
        dones_t = torch.as_tensor(dones_np, device=device, dtype=torch.bool)

        current_rew[active_idx] += rews_t[active_idx]
        current_len[active_idx] += 1

        # append step lps BEFORE checking for done episodes
        if track_logprobs:
            step_lp_np = lp_tensor.detach().cpu().numpy()
            for i, env_i in enumerate(active_np):
                current_steplogprob_histories[env_i].append(step_lp_np[i])

        done_and_active = dones_t & active_mask_t
        if done_and_active.any():
            done_idx = torch.where(done_and_active)[0]
            slots    = counts[done_idx]

            rewards[done_idx, slots] = current_rew[done_idx]
            lengths[done_idx, slots] = current_len[done_idx]
            logps[done_idx, slots]   = current_lp[done_idx]

            succ_list = [infos[int(i)].get("is_success", False) for i in done_idx.cpu().tolist()]
            proof_successful[done_idx, slots] = torch.as_tensor(succ_list, device=device)
            
            # Add episode statistics to info dicts for callback tracking
            for env_i in done_idx.cpu().tolist():
                if env_i < len(infos):
                    infos[env_i]["episode"] = {
                        "r": float(current_rew[env_i].item()),  # Total episode reward
                        "l": int(current_len[env_i].item()),    # Total episode length
                    }

            if track_logprobs:
                # finalize per done env
                for env_i in done_idx.cpu().tolist():
                    # Check if there's anything to store to avoid empty histories
                    if not current_steplogprob_histories[env_i]:
                        continue
                        
                    # Append final zero-action step for terminal state alignment
                    current_choices_histories[env_i].append(0)
                    current_steplogprob_histories[env_i].append(0.0)

                    # 1. stash scalar histories
                    episode_logprob_histories.append(np.cumsum(current_steplogprob_histories[env_i]))
                    episode_steplogprob_histories.append(np.array(current_steplogprob_histories[env_i]))
                    episode_choices_histories.append(np.array(current_choices_histories[env_i]))

                    # 2. build the *final* state string
                    term_obs = infos[env_i].get("terminal_observation", None)
                    if term_obs is not None:
                        term_sub = term_obs["sub_index"].squeeze(0)
                    else:
                        term_sub = new_obs["sub_index"][env_i].squeeze(0)
                    final_state_str = index_manager.state_subindex_to_str(term_sub, truncate=True)
                    current_state_histories[env_i].append(final_state_str)

                    # 3. freeze the whole list
                    episode_state_histories.append(
                        np.array(current_state_histories[env_i])
                    )

                    # 4. clear scratch buffers
                    current_steplogprob_histories[env_i].clear()
                    current_choices_histories[env_i].clear()
                    current_state_histories[env_i].clear()

            counts[done_idx]     += 1
            current_rew[done_idx], current_len[done_idx], current_lp[done_idx] = 0, 0, 0
        
        # Call info_callback AFTER episode stats are added to infos
        # This ensures the callback receives complete episode information
        if info_callback is not None:
            info_callback(infos)

        observations = new_obs
        if verbose: print(f"\rEpisodes done: {int(counts.sum())}/{total}", end="", flush=True)

    if verbose: print("\r" + " " * 80 + "\r", end="")

    # mask marking valid episode slots per env
    mask = (torch.arange(max_t, device=device)[None, :]
            < torch.as_tensor(padded_targets, device=device)[:, None])

    # trim to L = len(target_episodes) if provided (multi-run convenience)
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
    "pos_rw",
    "neg_rw",
    "pos_len",
    "neg_len",
    "pos_lp",
    "neg_lp",
    "pos_len_true",
    "pos_len_false",
    "neg_len_true",
    "neg_len_false",
    "y_true",
    "y_pred",
    "y_score",
    "mrr",
    "h1",
    "h3",
    "h10",
    "ap",
]

_GLOBAL_METRICS_TEMPLATE = {
    "pos_rw": [],
    "neg_rw": [],
    "pos_len": [],
    "neg_len": [],
    "pos_lp": [],
    "neg_lp": [],
    "pos_len_true": [],
    "pos_len_false": [],
    "neg_len_true": [],
    "neg_len_false": [],
    "y_true": [],
    "y_pred": [],
    "y_score": [],
    "head_mrr": [],
    "head_h1": [],
    "head_h3": [],
    "head_h10": [],
    "tail_mrr": [],
    "tail_h1": [],
    "tail_h3": [],
    "tail_h10": [],
    "head_ap": [],
    "tail_ap": [],
}


def _ensure_vec_env(env: Union[gym.Env, VecEnv]) -> VecEnv:
    return env if isinstance(env, VecEnv) else DummyVecEnv([lambda: env])


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


def _configure_env_batch(
    env: VecEnv,
    batch: List[Any],
    corrs: List[List[Any]],
    data_depths: Optional[List[int]],
    batch_start_idx: int,
) -> None:
    for i, (query, negatives) in enumerate(zip(batch, corrs)):
        sequence = [query] + negatives
        inner_env = env.envs[i].env
        batch_idx = batch_start_idx + i
        pos_depth = (
            data_depths[batch_idx]
            if data_depths is not None and batch_idx < len(data_depths)
            else None
        )
        depths = [pos_depth] + [None] * len(negatives)
        inner_env.mode = "eval"
        inner_env.queries = sequence
        inner_env.labels = [1] + [0] * len(negatives)
        inner_env.query_depths = depths
        inner_env.n_episodes = len(sequence)
        inner_env.eval_idx = 0


def _evaluate_with_rl(
    model: "type_aliases.PolicyPredictor",
    env: VecEnv,
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
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]],
]:
    _configure_env_batch(env, batch, corrs, data_depths, batch_start_idx)

    eval_kwargs = dict(
        model=model,
        env=env,
        deterministic=deterministic,
        target_episodes=targets,
        verbose=verbose,
        info_callback=info_callback,
    )

    if plot:
        (
            rewards,
            lengths,
            log_probs,
            mask,
            proof_successful,
            logprob_histories,
            choices_histories,
            steplogprob_histories,
            state_histories,
        ) = evaluate_policy(track_logprobs=True, **eval_kwargs)
        plot_payload = (
            logprob_histories,
            choices_histories,
            steplogprob_histories,
            state_histories,
        )
    else:
        rewards, lengths, log_probs, mask, proof_successful = evaluate_policy(
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
# ---------------------------------------------------------------------------
# High-level evaluator for head/tail corruptions and ranking metrics
# ---------------------------------------------------------------------------

def eval_corruptions(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
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
    """Evaluate a model by ranking a positive query vs. its corruptions.

    For each positive query in `data`, this function generates *head* and/or
    *tail* corruptions using `sampler`. Then it assigns a score to each
    candidate according to the selected `evaluation_mode`:

    • "rl_only":
        score = RL log-probability (from `evaluate_policy`) — failed proofs are
        heavily penalized.
    • "kge_only":
        score = log(KGE score) — no proof notion here (success mask is False).
    • "hybrid":
        score = α·log(KGE score) for all candidates; optionally add β·RL
        log-probability either only to successful proofs (if
        `hybrid_success_only` is True) or to every candidate. Defaults use
        α=2.0 and β=1.0.

    Metrics accumulated (per-batch + rolling):
      - Classification: Accuracy, Precision, Recall, F1, Average Precision
      - Ranking: MRR, Hits@1/3/10 (tracked separately for head/tail and merged)
      - (RL-only) Episode rewards/lengths and log-probs for pos/neg.

    Parameters
    ----------
    model, env : see `evaluate_policy`.
    data : list
        The list of positive queries (domain-specific objects) to evaluate.
    sampler : Any
        Negative sampler able to produce head/tail corruptions for each query.
    n_corruptions : Optional[int]
        Number of *negatives per type* to generate. Use `None` for *all*.
        Use `-1` synonymously with `None`.
    deterministic : bool, default True
        Whether to use deterministic policy actions.
    verbose : int, default 1
        Print per-batch and rolling summaries.
    plot : bool, default False
        If True, collects per-step trajectories for later heatmap plots.
    kge_inference_engine : optional
        KGE model wrapper with a `.predict_batch([...]) -> List[float]` API.
    evaluation_mode : {'rl_only','kge_only','hybrid'}
        Scoring strategy.
    corruption_scheme : list[str]
        Subset of {"head","tail"} to evaluate.
    info_callback : callable, optional
        Callback receiving the list of info dicts after each env step during
        RL-based evaluation (allows custom logging such as depth statistics).
    data_depths : list[int], optional
        The list of depth values corresponding to each query in `data`.
        Used to properly set query_depths in the environment during evaluation.
        If None, depth information will not be updated in the environment.
    hybrid_kge_weight : float, default 2.0
        Multiplicative weight applied to log KGE scores in hybrid mode.
    hybrid_rl_weight : float, default 1.0
        Multiplicative weight applied to RL log-probabilities in hybrid mode.
    hybrid_success_only : bool, default True
        When True, adds the RL contribution only for successful proofs; when
        False, adds it to every candidate regardless of success.

    Returns
    -------
    Dict[str, Any]
        Aggregated metrics (means/stds) and classification scores.
    """
    if evaluation_mode not in ("rl_only", "kge_only", "hybrid"):
        raise ValueError(f"Invalid evaluation_mode: {evaluation_mode}")
    if evaluation_mode != "rl_only" and kge_inference_engine is None:
        raise ValueError(
            f"`kge_inference_engine` must be provided for mode: '{evaluation_mode}'"
        )

    env = _ensure_vec_env(env)
    num_envs = env.num_envs
    if n_corruptions == -1:
        n_corruptions = None

    if verbose:
        print(f"Evaluating {len(data)} queries in '{evaluation_mode}' mode.")
        print(
            f"N corruptions per query (per type): "
            f"{'All' if n_corruptions is None else n_corruptions} | Envs: {num_envs}"
        )

    global_metrics = _init_global_metrics()

    aggregated_plot_data: Optional[
        Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]
    ] = None
    date_str = ""
    if plot:
        os.makedirs("plots", exist_ok=True)
        date_str = time.strftime("%Y%m%d-%H%M%S")
        aggregated_plot_data = {
            "pos_success": [],
            "pos_fail": [],
            "neg_success": [],
            "neg_fail": [],
        }

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
                [[q] for q in batch], model.device, num_negs=n_corruptions
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
                    rewards,
                    lengths,
                    log_probs,
                    eval_mask,
                    proof_successful,
                    plot_payload,
                ) = _evaluate_with_rl(
                    model=model,
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

                if (
                    plot
                    and plot_payload is not None
                    and aggregated_plot_data is not None
                ):
                    (
                        logprob_histories,
                        choices_histories,
                        steplogprob_histories,
                        state_histories,
                    ) = plot_payload
                    prepare_batch_data(
                        logprob_histories=logprob_histories,
                        choices_histories=choices_histories,
                        steplogprob_histories=steplogprob_histories,
                        state_histories=state_histories,
                        proof_successful=proof_successful,
                        mask=mask,
                        targets=targets,
                        date_str=date_str,
                        aggregated_plot_data=aggregated_plot_data,
                        b=b,
                        B=B,
                    )

            _extract_and_accumulate_metrics(
                batch_metrics,
                global_metrics,
                corruption_type,
                mask,
                proof_successful,
                log_probs,
                rewards,
                lengths,
            )

        if verbose:
            _report_batch_metrics(batch_metrics, global_metrics)
            print(f"Batch {b+1} took {time.time() - time_start:.2f} seconds")

    if plot and aggregated_plot_data is not None:
        aggregated_filename = f"plots/{date_str}_logprob_heatmap_aggregated.png"
        plot_logprob_heatmap(
            aggregated_plot_data,
            filename=aggregated_filename,
            multicolor_lines=True,
            show_state_labels=False,
            show_action_counts=True,
            show_avg_trajectory=True,
            dot_alpha=0.7,
            line_alpha=0.7,
        )

    return _finalize_and_get_results(global_metrics)


# ---------------------------------------------------------------------------
# Helper utilities (KGE scoring, metrics, reporting, plotting hooks)
# ---------------------------------------------------------------------------

def kge_eval(
    batch: List[Any],
    corrs: List[List[Any]],
    mask: np.ndarray,
    kge_inference_engine: Any,
) -> np.ndarray:
    """Score a (pos + negatives) set using a KGE inference engine.

    The KGE API is expected to expose `predict_batch([...]) -> List[float]` and
    return *scores* in [0, 1]. This function returns **log-scores** aligned to
    the `(env, episode-slot)` layout defined by `mask`.
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
    """Extract metrics from an evaluation pass and update accumulators.

    The *positive* candidate is assumed at episode slot 0 for every env.
    """
    # Classification targets/predictions
    true_labels = np.zeros_like(mask, dtype=int); true_labels[:, 0] = 1
    y_true_part, y_pred_part = true_labels[mask], proof_successful[mask].astype(int)
    y_score_part = log_probs[mask]

    batch_metrics['y_true'].extend(y_true_part.tolist()); global_metrics['y_true'].extend(y_true_part.tolist())
    batch_metrics['y_pred'].extend(y_pred_part.tolist()); global_metrics['y_pred'].extend(y_pred_part.tolist())
    batch_metrics['y_score'].extend(y_score_part.tolist()); global_metrics['y_score'].extend(y_score_part.tolist())

    # Log probabilities
    pos_lp, neg_lp = log_probs[:, 0][mask[:, 0]], log_probs[:, 1:][mask[:, 1:]]
    batch_metrics['pos_lp'].extend(pos_lp.tolist()); global_metrics['pos_lp'].extend(pos_lp.tolist())
    batch_metrics['neg_lp'].extend(neg_lp.tolist()); global_metrics['neg_lp'].extend(neg_lp.tolist())

    # Rewards and lengths (only for RL modes)
    if rewards is not None and lengths is not None:
        pos_rw, neg_rw = rewards[:, 0][mask[:, 0]], rewards[:, 1:][mask[:, 1:]]
        pos_len, neg_len = lengths[:, 0][mask[:, 0]], lengths[:, 1:][mask[:, 1:]]
        batch_metrics['pos_rw'].extend(pos_rw.tolist()); global_metrics['pos_rw'].extend(pos_rw.tolist())
        batch_metrics['neg_rw'].extend(neg_rw.tolist()); global_metrics['neg_rw'].extend(neg_rw.tolist())
        batch_metrics['pos_len'].extend(pos_len.tolist()); global_metrics['pos_len'].extend(pos_len.tolist())
        batch_metrics['neg_len'].extend(neg_len.tolist()); global_metrics['neg_len'].extend(neg_len.tolist())

        # Detailed episode lengths
        # Positives
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

        # Negatives
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

    # Ranking metrics (avoid ties by random keys for stable tie‑breakers)
    if mask.shape[1] > 1:
        lp_batch = np.where(mask, log_probs, -np.inf)
        random_keys = np.random.rand(*lp_batch.shape)
        sorted_indices = np.lexsort((-random_keys, -lp_batch), axis=1)
        ranks = np.where(sorted_indices == 0)[1] + 1
        mrr, h1, h3, h10 = 1.0/ranks, (ranks == 1).astype(float), (ranks <= 3).astype(float), (ranks <= 10).astype(float)
        ap = average_precision_score(y_true_part, y_score_part)
        
        batch_metrics['mrr'].extend(mrr.tolist()); batch_metrics['h1'].extend(h1.tolist())
        batch_metrics['h3'].extend(h3.tolist()); batch_metrics['h10'].extend(h10.tolist())
        batch_metrics['ap'].append(ap)
        global_metrics[f'{corruption_type}_mrr'].extend(mrr.tolist())
        global_metrics[f'{corruption_type}_h1'].extend(h1.tolist())
        global_metrics[f'{corruption_type}_h3'].extend(h3.tolist())
        global_metrics[f'{corruption_type}_h10'].extend(h10.tolist())
        global_metrics[f'{corruption_type}_ap'].append(ap)

def _report_batch_metrics(batch_metrics: Dict[str, List[float]], global_metrics: Dict[str, List[float]]) -> None:
    """Pretty-print current batch metrics and rolling totals."""
    def safe_mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    def safe_str(val: float) -> str:
        return f"{val:.3f}" if isinstance(val, float) else val

    # --- Batch Metrics ---
    b_pos_rw = safe_mean(batch_metrics["pos_rw"])
    b_neg_rw = safe_mean(batch_metrics["neg_rw"])
    b_pos_len = safe_mean(batch_metrics["pos_len"])
    b_neg_len = safe_mean(batch_metrics["neg_len"])
    b_pos_len_true = safe_mean(batch_metrics["pos_len_true"])
    b_pos_len_false = safe_mean(batch_metrics["pos_len_false"])
    b_neg_len_true = safe_mean(batch_metrics["neg_len_true"])
    b_neg_len_false = safe_mean(batch_metrics["neg_len_false"])
    b_pos_lp = safe_mean(batch_metrics["pos_lp"])
    b_neg_lp = safe_mean(batch_metrics["neg_lp"])
    b_ap = safe_mean(batch_metrics["ap"])
    b_mrr = safe_mean(batch_metrics["mrr"])
    b_h1 = safe_mean(batch_metrics["h1"])
    b_h3 = safe_mean(batch_metrics["h3"])
    b_h10 = safe_mean(batch_metrics["h10"])

    # --- Rolling Metrics ---
    r_pos_rw = safe_mean(global_metrics["pos_rw"])
    r_neg_rw = safe_mean(global_metrics["neg_rw"])
    r_pos_len = safe_mean(global_metrics["pos_len"])
    r_neg_len = safe_mean(global_metrics["neg_len"])
    r_pos_len_true = safe_mean(global_metrics["pos_len_true"])
    r_pos_len_false = safe_mean(global_metrics["pos_len_false"])
    r_neg_len_true = safe_mean(global_metrics["neg_len_true"])
    r_neg_len_false = safe_mean(global_metrics["neg_len_false"])
    r_pos_lp = safe_mean(global_metrics["pos_lp"])
    r_neg_lp = safe_mean(global_metrics["neg_lp"])
    r_ap = safe_mean(global_metrics["head_ap"] + global_metrics["tail_ap"])
    r_mrr = safe_mean(global_metrics["head_mrr"] + global_metrics["tail_mrr"])
    r_h1 = safe_mean(global_metrics["head_h1"] + global_metrics["tail_h1"])
    r_h3 = safe_mean(global_metrics["head_h3"] + global_metrics["tail_h3"])
    r_h10 = safe_mean(global_metrics["head_h10"] + global_metrics["tail_h10"])

    # --- Print ---
    print(
        f"\nrwds pos    : {safe_str(b_pos_rw) if b_pos_rw else 'N/A'}   \trwds neg       : {safe_str(b_neg_rw) if b_neg_rw else 'N/A'}"
    )
    print(
        f"ep len pos  : {safe_str(b_pos_len) if b_pos_len else 'N/A'} \tep len neg    : {safe_str(b_neg_len) if b_neg_len else 'N/A'}"
    )
    print(
        f"  pos->T len: {safe_str(b_pos_len_true) if b_pos_len_true else 'N/A'} \t  neg->T len    : {safe_str(b_neg_len_true) if b_neg_len_true else 'N/A'}"
    )
    print(
        f"  pos->F len: {safe_str(b_pos_len_false) if b_pos_len_false else 'N/A'} \t  neg->F len    : {safe_str(b_neg_len_false) if b_neg_len_false else 'N/A'}"
    )
    print(
        f"logprobs pos: {b_pos_lp:.3f} \tlog probs neg  : {safe_str(b_neg_lp) if b_neg_lp else 'N/A'}"
    )
    print(
        f"\nrolling rwds pos    : {safe_str(r_pos_rw) if r_pos_rw else 'N/A'} \trolling rwds neg       : {safe_str(r_neg_rw) if r_neg_rw else 'N/A'}"
    )
    print(
        f"rolling ep len pos  : {safe_str(r_pos_len) if r_pos_len else 'N/A'} \trolling episode len neg: {safe_str(r_neg_len) if r_neg_len else 'N/A'}"
    )
    print(
        f"  rolling pos->T len: {safe_str(r_pos_len_true) if r_pos_len_true else 'N/A'} \t  rolling neg->T len: {safe_str(r_neg_len_true) if r_neg_len_true else 'N/A'}"
    )
    print(
        f"  rolling pos->F len: {safe_str(r_pos_len_false) if r_pos_len_false else 'N/A'} \t  rolling neg->F len: {safe_str(r_neg_len_false) if r_neg_len_false else 'N/A'}"
    )
    print(
        f"rolling logprobs pos: {r_pos_lp:.3f} \trolling log probs neg  : {safe_str(r_neg_lp) if r_neg_lp else 'N/A'}"
    )
    print(f"\nmrr   : {b_mrr:.3f} \trolling mrr   : {r_mrr:.3f}")
    print(f"AP   : {b_ap:.3f} \trolling AP   : {r_ap:.3f}")
    print(f"hits1 : {b_h1:.3f} \trolling hits1 : {r_h1:.3f}")
    print(f"hits3 : {b_h3:.3f} \trolling hits3 : {r_h3:.3f}")
    print(f"hits10: {b_h10:.3f} \trolling hits10: {r_h10:.3f}")


def _finalize_and_get_results(metrics: Dict[str, List[float]]) -> Dict[str, float]:
    """Summarize aggregated metrics into a flat result dictionary.

    Returns a `Dict[str, float]` with means/stds for episode stats and classic
    classification/ranking metrics (merged head/tail where applicable).
    """
    def get_stats(data: List[float]) -> Tuple[float, float]:
        arr = np.array(data)
        return (arr.mean(), arr.std()) if arr.size > 0 else (0.0, 0.0)

    # Combined rank metrics
    all_mrr, all_h1, all_h3, all_h10 = (np.array(metrics['head_mrr'] + metrics['tail_mrr']), np.array(metrics['head_h1'] + metrics['tail_h1']),
                                        np.array(metrics['head_h3'] + metrics['tail_h3']), np.array(metrics['head_h10'] + metrics['tail_h10']))
    all_ap = np.array(metrics['head_ap'] + metrics['tail_ap'])
    
    final_results = {
        'mrr_mean': all_mrr.mean() if all_mrr.size > 0 else 0.0, 'hits1_mean': all_h1.mean() if all_h1.size > 0 else 0.0,
        'hits3_mean': all_h3.mean() if all_h3.size > 0 else 0.0, 'hits10_mean': all_h10.mean() if all_h10.size > 0 else 0.0,
        'head_mrr_mean': get_stats(metrics['head_mrr'])[0], 'tail_mrr_mean': get_stats(metrics['tail_mrr'])[0],
        'head_hits1_mean': get_stats(metrics['head_h1'])[0], 'tail_hits1_mean': get_stats(metrics['tail_h1'])[0],
        'head_hits3_mean': get_stats(metrics['head_h3'])[0], 'tail_hits3_mean': get_stats(metrics['tail_h3'])[0],
        'head_hits10_mean': get_stats(metrics['head_h10'])[0], 'tail_hits10_mean': get_stats(metrics['tail_h10'])[0],
        'average_precision': all_ap.mean() if all_ap.size > 0 else 0.0,
    }
    for key in ['pos_rw', 'neg_rw', 'pos_len', 'neg_len', 'pos_lp', 'neg_lp',
                'pos_len_true', 'pos_len_false', 'neg_len_true', 'neg_len_false']:
        mean, std = get_stats(metrics[key])
        name_map = {
            'pos_rw': 'rewards_pos', 'neg_rw': 'rewards_neg', 
            'pos_len': 'episode_len_pos', 'neg_len': 'episode_len_neg', 
            'pos_lp': 'log_probs_pos', 'neg_lp': 'log_probs_neg',
            'pos_len_true': 'episode_len_pos_true', 'pos_len_false': 'episode_len_pos_false',
            'neg_len_true': 'episode_len_neg_true', 'neg_len_false': 'episode_len_neg_false'
            }
        final_results[f'{name_map[key]}_mean'], final_results[f'{name_map[key]}_std'] = mean, std

    return {k: float(v) for k, v in final_results.items()}


# ---------------------------------------------------------------------------
# Optional plotting helpers
# ---------------------------------------------------------------------------

def prepare_batch_data(
    logprob_histories: List[np.ndarray],
    choices_histories: List[np.ndarray],
    steplogprob_histories: List[np.ndarray],
    state_histories: List[np.ndarray],
    proof_successful: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    date_str: str,
    aggregated_plot_data: Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]],
    b: int = 0,
    B: int = 0,
) -> None:
    """Repackage per-episode histories into category buckets for heatmaps.

    Splits trajectories into four buckets: {pos,neg} × {success,fail} and saves
    a figure for the current batch. Extends the provided `aggregated_plot_data`.
    """
    hist_idx = 0
    batch_plot_data = {"pos_success": [], "pos_fail": [], "neg_success": [], "neg_fail": []}
    for i in range(B):
        num_episodes_in_env = targets[i]
        labels_for_env = [1] + [0] * (num_episodes_in_env - 1)
        for j in range(num_episodes_in_env):
            if not mask[i, j] or hist_idx >= len(logprob_histories): continue
            label = labels_for_env[j]
            success = proof_successful[i, j]
            log_hist, choice_hist, step_log_hist, state_hist = logprob_histories[hist_idx], choices_histories[hist_idx], steplogprob_histories[hist_idx], state_histories[hist_idx]
            hist_idx += 1
            data_tuple = (log_hist, choice_hist, step_log_hist, state_hist)
            if label == 1 and success: batch_plot_data["pos_success"].append(data_tuple)
            elif label == 1 and not success: batch_plot_data["pos_fail"].append(data_tuple)
            elif label == 0 and success: batch_plot_data["neg_success"].append(data_tuple)
            elif label == 0 and not success: batch_plot_data["neg_fail"].append(data_tuple)
    
    batch_filename = f"plots/{date_str}_logprob_heatmap_batch_{b}.png"
    plot_logprob_heatmap(batch_plot_data, filename=batch_filename, multicolor_lines=True, show_state_labels=True)
    
    for key in aggregated_plot_data:
        aggregated_plot_data[key].extend(batch_plot_data[key])


def plot_logprob_heatmap(
    data_dict: Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]],
    filename: str = "logprob_heatmap.png",
    multicolor_lines: bool = True,
    show_state_labels: bool = True,
    show_action_counts: bool = True,
    show_avg_trajectory: bool = False,
    max_trajectories_to_plot: Optional[int] = None,
    dot_alpha: float = 0.9,
    line_alpha: float = 0.7,
    dot_size: int = 150,
) -> None:
    """Plot per-step accumulated log-prob trajectories as a 2×2 grid of axes.

    Each panel corresponds to one of the four buckets (pos/neg × success/fail).
    Points are colored by *step* log-probabilities and optionally annotated with
    the number of valid actions at that step and the (truncated) state string.
    The function writes a PNG to `filename` and closes the figure.
    """
    fig, axs = plt.subplots(2, 2, figsize=(22, 18), sharex=True, sharey=True)
    axs = axs.flatten()
    titles = {"pos_success": "Positive Query -> Proved True", "pos_fail": "Positive Query -> Failed", "neg_success": "Negative Query -> Proved True (Model Error)", "neg_fail": "Negative Query -> Proved False (Correct)"}
    
    all_steplogprobs = [step_lp for category in data_dict.values() for _, _, step_logprob_hist, _ in category for step_lp in step_logprob_hist]
    vmin = min(all_steplogprobs) if all_steplogprobs else -5.0
    vmax = max(all_steplogprobs) if all_steplogprobs else 0.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.coolwarm_r
    scatter_mappable = None

    for i, key in enumerate(titles.keys()):
        ax = axs[i]
        all_episodes_in_category = data_dict.get(key, [])
        if max_trajectories_to_plot and len(all_episodes_in_category) > max_trajectories_to_plot:
            episodes_to_plot = random.sample(all_episodes_in_category, max_trajectories_to_plot)
            ax.set_title(f"{titles[key]} (Sampled {max_trajectories_to_plot}/{len(all_episodes_in_category)})", fontsize=14)
        else:
            episodes_to_plot = all_episodes_in_category
            ax.set_title(f"{titles[key]} ({len(episodes_to_plot)} episodes)", fontsize=14)

        if not episodes_to_plot:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.grid(True, linestyle='--', alpha=0.6)
            continue

        num_episodes = len(episodes_to_plot)
        line_colors = plt.cm.nipy_spectral(np.linspace(0, 1, num_episodes)) if multicolor_lines else ['grey'] * num_episodes
        for idx, (logprob_hist, choice_hist, step_logprob_hist, state_hist) in enumerate(episodes_to_plot):
            if len(logprob_hist) == 0: continue
            steps = np.arange(len(logprob_hist))
            line_color = line_colors[idx]
            ax.plot(steps, logprob_hist, linestyle='-', color=line_color, alpha=line_alpha, linewidth=2.5, zorder=1)
            scatter_mappable = ax.scatter(steps, logprob_hist, c=step_logprob_hist, cmap=cmap, norm=norm, alpha=dot_alpha, s=dot_size, edgecolors='black', linewidth=0.5, zorder=2)
            if show_action_counts:
                for j, step_lp in enumerate(logprob_hist):
                    ax.text(x=j, y=step_lp, s=str(choice_hist[j]), ha='center', va='center', fontsize=8, color='white', fontweight='bold')
            if show_state_labels:
                y_min, y_max = ax.get_ylim()
                offset = (y_max - y_min) * 0.03 if y_max > y_min else 0.1
                bbox = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75)
                for j, step_lp in enumerate(logprob_hist):
                    if j < len(state_hist):
                        ax.text(x=j, y=step_lp + offset, s=state_hist[j], ha='center', va='bottom', fontsize=8, fontweight='light', color='darkslategray', rotation=15,zorder=5,clip_on=False, bbox=bbox)
        
        if show_avg_trajectory and all_episodes_in_category:
            log_hists = [h for h, _, _, _ in all_episodes_in_category]
            max_len = max(len(h) for h in log_hists if len(h) > 0)
            if max_len > 0:
                padded_hists = np.full((len(log_hists), max_len), np.nan)
                for i_hist, hist in enumerate(log_hists):
                    padded_hists[i_hist, :len(hist)] = hist
                mean_trajectory = np.nanmean(padded_hists, axis=0)
                ax.plot(np.arange(max_len), mean_trajectory, color='black', linestyle='--', linewidth=3, label='Average Trajectory', zorder=3)
                ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    if scatter_mappable:
        fig.subplots_adjust(right=0.90)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(scatter_mappable, cax=cbar_ax, orientation='vertical', label='Step Log Probability')

    fig.supxlabel('Step Number in Episode', fontsize=16)
    fig.supylabel('Accumulated Log Probability', fontsize=16)
    fig.suptitle('Log Probability Heatmap vs. Step, with Action Counts', fontsize=20)
    print(f"Saving logprob heatmap plot to {filename}")
    plt.savefig(filename)
    plt.close()
