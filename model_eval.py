from typing import Any, Dict, List, Tuple, Union, Optional
import time
import sys
import os
import random

import gymnasium as gym
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

from stable_baselines3.common import type_aliases
from stable_baselines3.common.on_policy_algorithm import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from kge_inference import KGEInference
# The KGCDataHandler is not needed for this hybrid approach
# from kge_loader import KGCDataHandler

@torch.inference_mode()
def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    target_episodes: np.ndarray | None = None,
    verbose: int = 0,
    track_logprobs: bool = False,
):
    """
    Fast eval: tensor-first, vectorized finalization.
    Returns:
        rewards, lengths, logps, mask, proof_successful,
        episode_logprob_histories, episode_choices_histories,
        episode_steplogprob_histories, episode_state_histories
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

    # histories (only if needed)
    episode_logprob_histories    : list[np.ndarray] = []
    episode_choices_histories    : list[np.ndarray] = []
    episode_steplogprob_histories: list[np.ndarray] = []
    episode_state_histories      : list[np.ndarray] = []
    if track_logprobs:
        current_steplogprob_histories = [[] for _ in range(n_envs)]
        current_choices_histories     = [[] for _ in range(n_envs)]
        current_state_histories       = [[] for _ in range(n_envs)]
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

        # slice obs
        if isinstance(observations, dict):
            obs_active = {k: torch.as_tensor(v[active_np], device=device) for k, v in observations.items()}
        else:
            obs_active = torch.as_tensor(observations[active_np], device=device)

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


        observations = new_obs
        if verbose: print(f"\rEpisodes done: {int(counts.sum())}/{total}", end="", flush=True)

    if verbose: print("\r" + " " * 80 + "\r", end="")

    # mask
    mask = (torch.arange(max_t, device=device)[None, :]
            < torch.as_tensor(padded_targets, device=device)[:, None])

    # trim
    L = len(target_episodes) if target_episodes is not None else n_envs
    return tuple(x[:L].cpu().numpy() for x in (rewards, lengths, logps, mask, proof_successful))




def eval_corruptions(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    data: List[Any],
    sampler: Any,
    n_corruptions: Optional[int] = None,
    deterministic: bool = True,
    verbose: int = 1,
    plot: bool = False,
    kge_inference_engine: Optional[KGEInference] = None,
    evaluation_mode: str = 'rl_only',
) -> Dict[str, Any]:
    """
    Evaluates a model on a dataset by generating head/tail corruptions for each query.

    Args:
        model: The policy model to evaluate.
        env: The vectorized environment.
        data: The list of positive queries for evaluation.
        sampler: The negative sampler for generating corruptions.
        n_corruptions: Number of corruptions to generate per query. `None` for all.
        deterministic: Whether the policy should be deterministic.
        verbose: Verbosity level.
        kge_inference_engine: An optional KGE model for hybrid scoring.
        evaluation_mode: One of 'rl_only', 'kge_only', or 'hybrid'.

    Returns:
        A dictionary containing detailed evaluation metrics.
    """
    # --- Setup and Initialization ---
    if evaluation_mode not in ['rl_only', 'kge_only', 'hybrid']: raise ValueError(f"Invalid evaluation_mode: {evaluation_mode}")
    if evaluation_mode != 'rl_only' and kge_inference_engine is None: raise ValueError(f"`kge_inference_engine` must be provided for mode: '{evaluation_mode}'")
    
    env = DummyVecEnv([lambda: env]) if not isinstance(env, VecEnv) else env
    num_envs = env.num_envs
    if n_corruptions == -1: n_corruptions = None
    
    if verbose:
        print(f"Evaluating {len(data)} queries in '{evaluation_mode}' mode.")
        print(f"N corruptions per query (per type): {'All' if n_corruptions is None else n_corruptions} | Envs: {num_envs}")

    # Global accumulator for all metrics across all batches
    global_metrics = {
        'pos_rw': [], 'neg_rw': [], 'pos_len': [], 'neg_len': [], 'pos_lp': [], 'neg_lp': [],
        'y_true': [], 'y_pred': [],
        'head_mrr': [], 'head_h1': [], 'head_h3': [], 'head_h10': [],
        'tail_mrr': [], 'tail_h1': [], 'tail_h3': [], 'tail_h10': []
    }

    if plot:
        os.makedirs("plots", exist_ok=True)
        aggregated_plot_data = {"pos_success": [], "pos_fail": [], "neg_success": [], "neg_fail": []}

    # --- Batch Processing Loop ---
    total_batches = (len(data) + num_envs - 1) // num_envs
    for b, start in enumerate(range(0, len(data), num_envs)):
        time_start = time.time()
        if verbose: print(f"\n--- Batch {b+1}/{total_batches} (Queries {start}-{min(start+num_envs-1, len(data)-1)}) ---")
        batch = data[start : start + num_envs]
        B = len(batch)

        # Per-batch accumulator for reporting
        batch_metrics = {key: [] for key in ['pos_rw', 'neg_rw', 'pos_len', 'neg_len', 'pos_lp', 'neg_lp', 'y_true', 'y_pred', 'mrr', 'h1', 'h3', 'h10']}

        # --- Corruption Generation ---
        head_corrs, tail_corrs = sampler.get_negatives_from_states_separate([[q] for q in batch], model.device, num_negs=n_corruptions) if n_corruptions != 0 else ([[] for _ in range(B)], [[] for _ in range(B)])
        if B == 1:
            if not isinstance(head_corrs[0], list): head_corrs = [head_corrs]
            if not isinstance(tail_corrs[0], list): tail_corrs = [tail_corrs]

        # --- Evaluate Head and Tail Corruptions ---
        for corruption_type, corrs in [("head", head_corrs), ("tail", tail_corrs)]:
            targets = np.array([1 + len(c) for c in corrs], dtype=int)
            if not np.any(targets): continue
            
            max_t = targets.max()
            padded_targets = np.zeros(num_envs, dtype=int); padded_targets[:B] = targets
            mask = (np.arange(max_t)[None, :] < padded_targets[:, None])
            
            rewards, lengths, log_probs, proof_successful = None, None, None, None

            # --- Score Calculation (KGE, RL, or Hybrid) ---
            if evaluation_mode == 'kge_only':
                log_probs = kge_eval(batch, corrs, mask, kge_inference_engine)
                proof_successful = np.zeros_like(mask, dtype=bool) # No proofs in KGE-only mode
            else: # 'rl_only' or 'hybrid'
                for i, (q, negs) in enumerate(zip(batch, corrs)):
                    seq = [q] + negs
                    e = env.envs[i].env
                    e.mode, e.queries, e.labels, e.n_episodes, e.eval_idx = "eval", seq, [1] + [0]*len(negs), len(seq), 0
                
                rewards, lengths, log_probs, _, proof_successful = evaluate_policy(model, env, 
                                                    deterministic=deterministic, target_episodes=targets, 
                                                    verbose=verbose>1,track_logprobs=plot)
                
                if evaluation_mode == 'hybrid':
                    kge_log_scores = kge_eval(batch, corrs, mask, kge_inference_engine)
                    log_probs += kge_log_scores # Add KGE score to RL log-prob

                log_probs[~proof_successful] -= 100 # Penalize failed proofs

            # --- Accumulate Metrics ---
            _extract_and_accumulate_metrics(batch_metrics, global_metrics, corruption_type, mask, 
                                            proof_successful, log_probs, rewards, lengths)

        # --- Report Batch Metrics ---
        if verbose:
            _report_batch_metrics(batch_metrics, global_metrics)
        print(f"Batch {b+1} took {time.time() - time_start:.2f} seconds") if verbose else None

    # --- Finalize and Return All Metrics ---
    return _finalize_and_get_results(global_metrics)






def kge_eval(batch, corrs, mask, kge_inference_engine):
    """Gets log-probability scores for a batch of atoms using the KGE model."""
    all_atoms_in_batch, atom_map = [], {}
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


def _extract_and_accumulate_metrics(batch_metrics, global_metrics, corruption_type, mask, proof_successful, log_probs, rewards=None, lengths=None):
    """Extracts metrics from evaluation results and updates batch/global accumulators."""
    # Classification metrics
    true_labels = np.zeros_like(mask, dtype=int); true_labels[:, 0] = 1
    y_true_part, y_pred_part = true_labels[mask], proof_successful[mask].astype(int)
    batch_metrics['y_true'].extend(y_true_part.tolist()); global_metrics['y_true'].extend(y_true_part.tolist())
    batch_metrics['y_pred'].extend(y_pred_part.tolist()); global_metrics['y_pred'].extend(y_pred_part.tolist())
    
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
        
    # Rank-based metrics
    if mask.shape[1] > 1:
        lp_batch = np.where(mask, log_probs, -np.inf)
        random_keys = np.random.rand(*lp_batch.shape)
        sorted_indices = np.lexsort((-random_keys, -lp_batch), axis=1)
        ranks = np.where(sorted_indices == 0)[1] + 1
        mrr, h1, h3, h10 = 1.0/ranks, (ranks == 1).astype(float), (ranks <= 3).astype(float), (ranks <= 10).astype(float)
        
        batch_metrics['mrr'].extend(mrr.tolist()); batch_metrics['h1'].extend(h1.tolist()); batch_metrics['h3'].extend(h3.tolist()); batch_metrics['h10'].extend(h10.tolist())
        global_metrics[f'{corruption_type}_mrr'].extend(mrr.tolist()); global_metrics[f'{corruption_type}_h1'].extend(h1.tolist()); global_metrics[f'{corruption_type}_h3'].extend(h3.tolist()); global_metrics[f'{corruption_type}_h10'].extend(h10.tolist())

def _report_batch_metrics(batch_metrics, global_metrics):
    """Prints a formatted summary of metrics for the current batch and rolling totals."""
    def safe_mean(arr): return np.mean(arr) if arr else 0.0
    def safe_str(val): return f"{val:.3f}" if isinstance(val, float) else val

    # --- Batch Metrics ---
    b_pos_rw = safe_mean(batch_metrics['pos_rw']); b_neg_rw = safe_mean(batch_metrics['neg_rw'])
    b_pos_len = safe_mean(batch_metrics['pos_len']); b_neg_len = safe_mean(batch_metrics['neg_len'])
    b_pos_lp = safe_mean(batch_metrics['pos_lp']); b_neg_lp = safe_mean(batch_metrics['neg_lp'])
    b_acc = accuracy_score(batch_metrics['y_true'], batch_metrics['y_pred']); b_prec = precision_score(batch_metrics['y_true'], batch_metrics['y_pred'], zero_division=0)
    b_rec = recall_score(batch_metrics['y_true'], batch_metrics['y_pred'], zero_division=0); b_f1 = f1_score(batch_metrics['y_true'], batch_metrics['y_pred'], zero_division=0)
    b_mrr = safe_mean(batch_metrics['mrr']); b_h1 = safe_mean(batch_metrics['h1']); b_h3 = safe_mean(batch_metrics['h3']); b_h10 = safe_mean(batch_metrics['h10'])
    
    # --- Rolling Metrics ---
    r_pos_rw = safe_mean(global_metrics['pos_rw']); r_neg_rw = safe_mean(global_metrics['neg_rw'])
    r_pos_len = safe_mean(global_metrics['pos_len']); r_neg_len = safe_mean(global_metrics['neg_len'])
    r_pos_lp = safe_mean(global_metrics['pos_lp']); r_neg_lp = safe_mean(global_metrics['neg_lp'])
    r_acc = accuracy_score(global_metrics['y_true'], global_metrics['y_pred']); r_prec = precision_score(global_metrics['y_true'], global_metrics['y_pred'], zero_division=0)
    r_rec = recall_score(global_metrics['y_true'], global_metrics['y_pred'], zero_division=0); r_f1 = f1_score(global_metrics['y_true'], global_metrics['y_pred'], zero_division=0)
    r_mrr = safe_mean(global_metrics['head_mrr'] + global_metrics['tail_mrr']); r_h1 = safe_mean(global_metrics['head_h1'] + global_metrics['tail_h1'])
    r_h3 = safe_mean(global_metrics['head_h3'] + global_metrics['tail_h3']); r_h10 = safe_mean(global_metrics['head_h10'] + global_metrics['tail_h10'])

    # --- Print ---
    print(f"\nrwds pos    : {safe_str(b_pos_rw) if b_pos_rw else 'N/A'}   \trwds neg       : {safe_str(b_neg_rw) if b_neg_rw else 'N/A'}")
    print(f"ep len pos  : {safe_str(b_pos_len) if b_pos_len else 'N/A'} \tep len neg    : {safe_str(b_neg_len) if b_neg_len else 'N/A'}")
    print(f"logprobs pos: {b_pos_lp:.3f} \tlog probs neg  : {safe_str(b_neg_lp) if b_neg_lp else 'N/A'}")
    print(f"\nAccuracy: {b_acc:.3f} \tPrecision: {b_prec:.3f} \tRecall: {b_rec:.3f} \tF1 Score: {b_f1:.3f}")
    print(f"\nrolling Accuracy: {r_acc:.3f} \trolling Precision: {r_prec:.3f} \trolling Recall: {r_rec:.3f} \trolling F1 Score: {r_f1:.3f}")
    print(f"\nrolling rwds pos    : {safe_str(r_pos_rw) if r_pos_rw else 'N/A'} \trolling rwds neg       : {safe_str(r_neg_rw) if r_neg_rw else 'N/A'}")
    print(f"rolling ep len pos  : {safe_str(r_pos_len) if r_pos_len else 'N/A'} \trolling episode len neg: {safe_str(r_neg_len) if r_neg_len else 'N/A'}")
    print(f"rolling logprobs pos: {r_pos_lp:.3f} \trolling log probs neg  : {safe_str(r_neg_lp) if r_neg_lp else 'N/A'}")
    print(f"\nmrr   : {b_mrr:.3f} \trolling mrr   : {r_mrr:.3f}")
    print(f"hits1 : {b_h1:.3f} \trolling hits1 : {r_h1:.3f}")
    print(f"hits3 : {b_h3:.3f} \trolling hits3 : {r_h3:.3f}")
    print(f"hits10: {b_h10:.3f} \trolling hits10: {r_h10:.3f}")


def _finalize_and_get_results(metrics):
    """Calculates final summary statistics from the global metrics accumulator and returns the results dict."""
    def get_stats(data):
        arr = np.array(data)
        return (arr.mean(), arr.std()) if arr.size > 0 else (0.0, 0.0)

    # Combined rank metrics
    all_mrr, all_h1, all_h3, all_h10 = (np.array(metrics['head_mrr'] + metrics['tail_mrr']), np.array(metrics['head_h1'] + metrics['tail_h1']),
                                        np.array(metrics['head_h3'] + metrics['tail_h3']), np.array(metrics['head_h10'] + metrics['tail_h10']))
    
    final_results = {
        'mrr_mean': all_mrr.mean() if all_mrr.size > 0 else 0.0, 'hits1_mean': all_h1.mean() if all_h1.size > 0 else 0.0,
        'hits3_mean': all_h3.mean() if all_h3.size > 0 else 0.0, 'hits10_mean': all_h10.mean() if all_h10.size > 0 else 0.0,
        'head_mrr_mean': get_stats(metrics['head_mrr'])[0], 'tail_mrr_mean': get_stats(metrics['tail_mrr'])[0],
        'head_hits1_mean': get_stats(metrics['head_h1'])[0], 'tail_hits1_mean': get_stats(metrics['tail_h1'])[0],
        'head_hits3_mean': get_stats(metrics['head_h3'])[0], 'tail_hits3_mean': get_stats(metrics['tail_h3'])[0],
        'head_hits10_mean': get_stats(metrics['head_h10'])[0], 'tail_hits10_mean': get_stats(metrics['tail_h10'])[0],
        'accuracy': accuracy_score(metrics['y_true'], metrics['y_pred']), 'precision': precision_score(metrics['y_true'], metrics['y_pred'], zero_division=0),
        'recall': recall_score(metrics['y_true'], metrics['y_pred'], zero_division=0), 'f1_score': f1_score(metrics['y_true'], metrics['y_pred'], zero_division=0)
    }
    for key in ['pos_rw', 'neg_rw', 'pos_len', 'neg_len', 'pos_lp', 'neg_lp']:
        mean, std = get_stats(metrics[key])
        name_map = {'pos_rw': 'rewards_pos', 'neg_rw': 'rewards_neg', 'pos_len': 'episode_len_pos', 'neg_len': 'episode_len_neg', 'pos_lp': 'log_probs_pos', 'neg_lp': 'log_probs_neg'}
        final_results[f'{name_map[key]}_mean'], final_results[f'{name_map[key]}_std'] = mean, std
        
    return {k: float(v) for k, v in final_results.items()}




def prepare_batch_data(
    logprob_histories: List[np.ndarray], choices_histories: List[np.ndarray], steplogprob_histories: List[np.ndarray], state_histories: List[np.ndarray],
    proof_successful: np.ndarray, mask: np.ndarray, targets: np.ndarray, date_str: str,
    aggregated_plot_data: Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]], b: int = 0, B: int = 0
):
    # This function remains unchanged
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
    filename: str = "logprob_heatmap.png", multicolor_lines: bool = True, show_state_labels: bool = True,
    show_action_counts: bool = True, show_avg_trajectory: bool = False, max_trajectories_to_plot: Optional[int] = None,
    dot_alpha: float = 0.9, line_alpha: float = 0.7, dot_size: int = 150
):
    # This function remains unchanged
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