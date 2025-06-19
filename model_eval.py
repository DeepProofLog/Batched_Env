from typing import Any, Dict, List, Tuple, Union
import time
import sys

import gymnasium as gym
import numpy as np
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

from stable_baselines3.common import type_aliases
from stable_baselines3.common.on_policy_algorithm import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
import torch


def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    target_episodes: np.ndarray = None,
    verbose: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run policy for a specified number of episodes per env, returning
    rewards, lengths, log_probs, and a validity mask of shape (targets_len, max_target).
    """
    if not isinstance(env, VecEnv):
        print("Warning: wrapping single env in DummyVecEnv")
        env = DummyVecEnv([lambda: env])

    assert env.type_ == "custom_dummy", "Requires custom_dummy VecEnv"

    n_envs = env.num_envs
    if target_episodes is None:
        targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype=int)
    else:
        targets = np.array(target_episodes, dtype=int)

    targets_len = len(targets)
    if targets_len!= n_envs:
        padded_targets = np.zeros(n_envs, dtype=int)
        padded_targets[: len(targets)] = targets
    else:
        padded_targets = targets 

    total = padded_targets.sum()
    if verbose:
        print(f"\nEvaluating {total} episodes on {n_envs} envs (avg target: {padded_targets.mean():.2f})")
    
    env._episode_target[:] = padded_targets
    env._episode_count[:] = 0
    env.active_envs[:] = True

    rewards = np.zeros((n_envs, padded_targets.max()), dtype=float)
    lengths = np.zeros_like(rewards, dtype=int)
    logps   = np.zeros_like(rewards, dtype=float)
    counts  = np.zeros(n_envs, dtype=int)

    observations = env.reset()
    current_rew = np.zeros(n_envs, dtype=float)
    current_len = np.zeros(n_envs, dtype=int)
    current_lp  = np.zeros(n_envs, dtype=float)

    while (counts < padded_targets).any():
        active = counts < padded_targets

        # slice observations for active envs
        if isinstance(observations, dict):
            obs_active = {k: v[active] for k, v in observations.items()}
        else:
            obs_active = observations[active]

        # forward pass only for active
        obs_tensor = obs_as_tensor(obs_active, model.device)
        acts_tensor, _, lp_tensor = model.policy(obs_tensor, deterministic=deterministic)

        # to NumPy
        actions_active = acts_tensor.detach().cpu().numpy()    # shape: (n_active, *action_shape)
        lp_active      = lp_tensor.detach().cpu().numpy()

        # scatter log‐probs back
        current_lp[active] += lp_active

        # create a full‐size array with the correct NumPy dtype
        full_actions = np.zeros((n_envs, *actions_active.shape[1:]), dtype=actions_active.dtype)
        full_actions[active] = actions_active

        # step all envs (inactive ones will be skipped internally)
        new_obs, rew, dones, infos = env.step(full_actions)
        
        # print('\nRew active:', rew[active])
        current_rew[active] += rew[active]
        current_len[active] += 1

        # finalize any envs that just finished
        done_and_active = dones & active
        for idx in np.nonzero(done_and_active)[0]:
            slot = counts[idx]
            rewards[idx, slot] = current_rew[idx]
            lengths[idx, slot] = current_len[idx]
            logps[idx, slot]   = current_lp[idx]
            counts[idx]       += 1
            current_rew[idx]   = 0
            current_len[idx]   = 0
            current_lp[idx]    = 0

        observations = new_obs

        if verbose:
            done_total = counts.sum()
            print(f"\rEpisodes done: {done_total}/{total}", end="", flush=True)

    if verbose:
        print("\r" + " " * 80 + "\r", end="")

    # Build mask of valid entries
    mask = np.zeros_like(rewards, dtype=bool)
    for i, t in enumerate(padded_targets):
        mask[i, :t] = True

    # Filter up to target_len to skip unused envs
    if target_episodes is not None:
        rewards = rewards[:targets_len]
        lengths = lengths[:targets_len]
        logps   = logps[:targets_len]
        mask    = mask[:targets_len]
    return rewards, lengths, logps, mask


def eval_corruptions(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    data: torch.Tensor,
    sampler: Any,
    n_corruptions: int = None,
    deterministic: bool = True,
    verbose: int = 1,
    consult_janus: bool = False,
    plot: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate model on each query plus its corruptions, returning the same
    summary dict as before but built concisely with a validity mask.
    """
    # Ensure vector environment
    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])
    num_envs = env.num_envs

    if n_corruptions == -1:
        n_corruptions = None
    if verbose:
        print(f"Evaluating {len(data)} queries.")
        print(f"Max N corruptions per query: {'All' if n_corruptions is None else n_corruptions}")
        print(f"Using {num_envs} envs")
    # Accumulators for positives/negatives
    all_pos_lp, all_pos_rw, all_pos_len = [], [], []
    all_neg_lp, all_neg_rw, all_neg_len = [], [], []
    mrr_list, hits1_list, hits3_list, hits10_list = [], [], [], []

    total_batches = (len(data) + num_envs - 1) // num_envs # Calculate total number of batches
    # Batch through data
    for b, start in enumerate(range(0, len(data), num_envs)):
        batch = data[start : start + num_envs]
        B = len(batch)
        print(f"\n--- Batch {b+1}/{total_batches} (Queries {start+0}-{min(start+num_envs, len(data)-1)}) ---")

        # get corruptions
        print(f"Getting corruptions")
        start_time = time.time()
        data_for_negatives = batch.unsqueeze(-2) # (B=n_queries=1, padding_atoms, max_arity+1)
        corrs = sampler.get_negatives(data_for_negatives,
                                            padding_atoms=data_for_negatives.size(1),
                                            max_arity=data_for_negatives.size(2)-1,
                                            all_negatives=(n_corruptions is None),
                                            device='cuda' if torch.cuda.is_available() else 'cpu')
        # (B=n_queries, num_negs_per_pos, n_atoms or padding atoms, max_arity+1) --> (num_negs, max_arity+1)
        assert corrs.ndim == 4, f"Expected 4D tensor, got {corrs.ndim}D"
        corrs = corrs.squeeze(-2)  # Remove the batch dimension and the n_atoms dimension (it's just one query) 
        print(f"Corruption time: {time.time() - start_time:.2f}s")

        targets = np.array([1 + len(c) for c in corrs], dtype=int)

        print(f"Total episodes: {B} (envs) x {1+np.mean(targets):.1f} (avg targets) = {1+np.sum(targets)}")
        # configure each sub‐env
        start_time = time.time()
        for i, (q, negs) in enumerate(zip(batch, corrs)):
            seq = torch.cat([q.unsqueeze(0), negs], dim=0)  # (1 + num_negs, padding_atoms, max_arity+1)
            e = env.envs[i].env
            e.mode = "eval"
            e.queries, e.labels = seq, [1] + [0]*len(negs)
            e.n_episodes = len(seq)
            e.consult_janus_eval = consult_janus
            e.eval_idx = 0
        
        # run eval; gives shape (B, max_target)
        rewards, lengths, log_probs, mask = evaluate_policy(
            model,
            env,
            deterministic=deterministic,
            target_episodes=targets,
            verbose=verbose,
        )
        print(f"Eval time: {time.time() - start_time:.2f}s")

        log_probs[rewards == 0] -= 100

        # Collect metrics using mask
        # Positives: column 0
        all_pos_rw.extend(rewards[:, 0][mask[:, 0]])
        all_pos_len.extend(lengths[:, 0][mask[:, 0]])
        all_pos_lp.extend(log_probs[:, 0][mask[:, 0]])

        # Collect negatives across all valid slots
        neg_rw = rewards[:, 1:][mask[:, 1:]]
        neg_len = lengths[:, 1:][mask[:, 1:]]
        neg_lp = log_probs[:, 1:][mask[:, 1:]]
        all_neg_rw.extend(neg_rw)
        all_neg_len.extend(neg_len)
        all_neg_lp.extend(neg_lp)

        # Rank-based metrics
        if mask.shape[1] > 1:
            lp_batch = log_probs.copy()
            lp_batch[~mask] = -np.inf
            random_keys = np.random.rand(*lp_batch.shape)
            sorted_indices_per_query = np.lexsort((-random_keys, -lp_batch), axis=1)
            ranks = np.where(sorted_indices_per_query == 0)[1] + 1
            mrr = 1.0 / ranks
            hits1 = (ranks == 1).astype(int)
            hits3 = (ranks <= 3).astype(int)
            hits10 = (ranks <= 10).astype(int)

            mrr_list.extend(mrr.tolist())
            hits1_list.extend(hits1.tolist())
            hits3_list.extend(hits3.tolist())
            hits10_list.extend(hits10.tolist())

        print('\nrolling rwds pos    :',np.round(np.mean(all_pos_rw),3)    , '\trolling rwds neg       :',np.round(np.mean(all_neg_rw),3))
        print('rolling ep len pos  :',np.round(np.mean(all_pos_len),3), '\trolling episode len neg:',np.round(np.mean(all_neg_len),3))
        print('rolling logprobs pos:',np.round(np.mean(all_pos_lp),3)  , '\trolling log probs neg  :',np.round(np.mean(all_neg_lp),3))
        if mask.shape[1] > 1:
            print('\nmrr   :',np.round(np.mean(mrr),3)   ,'\trolling mrr   :',np.round(np.mean(mrr_list),3)) 
            print('hits1 :',np.round(np.mean(hits1),3) ,'\trolling hits1 :',np.round(np.mean(hits1_list),3))
            print('hits3 :',np.round(np.mean(hits3),3) ,'\trolling hits3 :',np.round(np.mean(hits3_list),3))
            print('hits10:',np.round(np.mean(hits10),3),'\trolling hits10:',np.round(np.mean(hits10_list),3))

    # to NumPy arrays
    pos_rw = np.array(all_pos_rw)
    pos_len = np.array(all_pos_len)
    pos_lp  = np.array(all_pos_lp)
    neg_rw  = np.array(all_neg_rw)
    neg_len = np.array(all_neg_len)
    neg_lp  = np.array(all_neg_lp)
    mrr_arr = np.array(mrr_list)
    h1 = np.array(hits1_list)
    h3 = np.array(hits3_list)
    h10 = np.array(hits10_list)

    return {
        'pos_queries': len(pos_rw),
        'neg_queries': len(neg_rw),
        'ratio_pos_queries': round(len(pos_rw) / (len(pos_rw) + len(neg_rw)), 2),
        'mrr_mean': float(mrr_arr.mean()) if mrr_arr.size else 0.0,
        'mrr_std': float(mrr_arr.std()) if mrr_arr.size else 0.0,

        'rewards_pos_mean': float(pos_rw.mean()),
        'rewards_pos_std': float(pos_rw.std()),
        'rewards_neg_mean': float(neg_rw.mean()) if neg_rw.size else 0.0,
        'rewards_neg_std': float(neg_rw.std()) if neg_rw.size else 0.0,
        'rewards_mean': float(np.concatenate([pos_rw, neg_rw]).mean()),
        'rewards_std': float(np.concatenate([pos_rw, neg_rw]).std()),

        'episode_len_pos_mean': float(pos_len.mean()),
        'episode_len_pos_std': float(pos_len.std()),
        'episode_len_neg_mean': float(neg_len.mean()) if neg_len.size else 0.0,
        'episode_len_neg_std': float(neg_len.std()) if neg_len.size else 0.0,
        'episode_len_mean': float(np.concatenate([pos_len, neg_len]).mean()),
        'episode_len_std': float(np.concatenate([pos_len, neg_len]).std()),

        'log_probs_pos_mean': float(pos_lp.mean()),
        'log_probs_pos_std': float(pos_lp.std()),
        'log_probs_neg_mean': float(neg_lp.mean()) if neg_lp.size else 0.0,
        'log_probs_neg_std': float(neg_lp.std()) if neg_lp.size else 0.0,
        'log_probs_mean': float(np.concatenate([pos_lp, neg_lp]).mean()),
        'log_probs_std': float(np.concatenate([pos_lp, neg_lp]).std()),

        'auc_pr': float(average_precision_score(
            np.concatenate([np.ones(len(pos_lp)), np.zeros(len(neg_lp))]),
            np.concatenate([pos_lp, neg_lp])
        )),
        'hits1_mean': float(h1.mean()) if h1.size else 0.0,
        'hits1_std': float(h1.std()) if h1.size else 0.0,
        'hits3_mean': float(h3.mean()) if h3.size else 0.0,
        'hits3_std': float(h3.std()) if h3.size else 0.0,
        'hits10_mean': float(h10.mean()) if h10.size else 0.0,
        'hits10_std': float(h10.std()) if h10.size else 0.0,
    }