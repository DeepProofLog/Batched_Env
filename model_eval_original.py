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
            current_rew[done_idx] = 0
            current_len[done_idx] = 0
            current_lp[done_idx]  = 0

        observations = new_obs
        if verbose:
            print(f"\rEpisodes done: {int(counts.sum())}/{total}", end="", flush=True)

    if verbose:
        print("\r" + " " * 80 + "\r", end="")

    # mask
    mask = (torch.arange(max_t, device=device)[None, :]
            < torch.as_tensor(padded_targets, device=device)[:, None])

    # trim
    if target_episodes is not None:
        L = len(target_episodes)
        rewards, lengths, logps, mask, proof_successful = [
            x[:L].cpu().numpy() for x in (rewards, lengths, logps, mask, proof_successful)
        ]
    else:
        rewards, lengths, logps, mask, proof_successful = [
            x.cpu().numpy() for x in (rewards, lengths, logps, mask, proof_successful)
        ]

    return (rewards, lengths, logps, mask, proof_successful,
            episode_logprob_histories, episode_choices_histories,
            episode_steplogprob_histories, episode_state_histories)


def eval_corruptions(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    data: List[Any],
    sampler: Any,
    n_corruptions: Optional[int] = None,
    deterministic: bool = True,
    verbose: int = 1,
    consult_janus: bool = False,
    plot: bool = False,
    kge_inference_engine: Optional[Any] = None,
    use_kge_hybrid_score: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate model on each query plus its corruptions, returning the same
    summary dict as before but built concisely with a validity mask.
    """
    # Histories are only needed if we are plotting
    track_logprobs = plot
    date_str = time.strftime("%Y%m%d-%H%M%S")
    # track_logprobs = True
    # Ensure vector environment
    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])
    num_envs = env.num_envs

    if n_corruptions == -1: n_corruptions = None

    if verbose:
        print(f"Evaluating {len(data)} queries.")
        print(f"N corruptions per query: {'All' if n_corruptions is None else n_corruptions}")
        print(f"Using {num_envs} envs")
    # Accumulators for positives/negatives
    all_pos_lp, all_pos_rw, all_pos_len = [], [], []
    all_neg_lp, all_neg_rw, all_neg_len = [], [], []
    mrr_list, hits1_list, hits3_list, hits10_list = [], [], [], []

    # Accumulators for classification metrics
    all_y_true, all_y_pred = [], []

    # AGGREGATION SETUP: Initialize a dict to hold data from all batches for the final plot
    if track_logprobs:
        os.makedirs("plots", exist_ok=True)
        aggregated_plot_data = {"pos_success": [], "pos_fail": [], "neg_success": [], "neg_fail": []}

    total_batches = (len(data) + num_envs - 1) // num_envs # Calculate total number of batches

    # Batch through data
    for b, start in enumerate(range(0, len(data), num_envs)):
        batch = data[start : start + num_envs]
        B = len(batch)
        if verbose: print(f"\n--- Batch {b+1}/{total_batches} (Queries {start}-{min(start+num_envs-1, len(data)-1)}) ---")

        # get corruptions
        if n_corruptions==0:
            corrs = [[] for _ in range(B)] if B > 1 else []
        else:
            if verbose: print(f"Getting corruptions")
            start_time = time.time()
            corrs = sampler.get_negatives_from_states(
                [[q] for q in batch],
                model.device,
                num_negs=n_corruptions,
            )
            if verbose: print(f"Corruption time: {time.time() - start_time:.2f}s")
        if B == 1:
            corrs = [corrs]

        targets = np.array([1 + len(c) for c in corrs], dtype=int)

        if verbose: print(f"Total episodes: {B} (envs) x {np.mean(targets):.1f} (avg targets) = {np.sum(targets)}")
        # configure each sub‐env
        start_time = time.time()
        for i, (q, negs) in enumerate(zip(batch, corrs)):
            print(f"Queries in batch: {q} + {[n for n in negs]}") if verbose > 1 else None
            seq = [q] + negs
            e = env.envs[i].env
            e.mode = "eval"
            e.queries, e.labels = seq, [1] + [0]*len(negs)
            e.n_episodes = len(seq)
            e.consult_janus_eval = consult_janus
            e.eval_idx = 0

        # run eval; gives shape (B, max_target)
        rewards, lengths, log_probs, mask, proof_successful, logprob_histories, choices_histories, steplogprob_histories, state_histories = evaluate_policy(
            model,
            env,
            deterministic=deterministic,
            target_episodes=targets,
            verbose=verbose>1,
            track_logprobs=track_logprobs,
        )
        if verbose: print(f"Eval time: {time.time() - start_time:.2f}s")

        # Penalize log_probs for episodes that did NOT find a proof.
        log_probs[~proof_successful] -= 100
        
        # # do the same for logprobs_histories
        # if track_logprobs:
        #     hist_idx = 0
        #     for i in range(proof_successful.shape[0]):
        #         for j in range(proof_successful.shape[1]):
        #             if not mask[i, j]: continue
        #             if not proof_successful[i, j]:
        #                 if hist_idx < len(logprob_histories) and len(logprob_histories[hist_idx]) > 0:
        #                     logprob_histories[hist_idx] -= 100
        #             hist_idx += 1

        # --- Determine the final scores for ranking based on the selected mode ---
        if use_kge_hybrid_score:
            if kge_inference_engine is None:
                raise ValueError("`kge_inference_engine` must be provided when `use_kge_hybrid_score` is True.")
            
            # Collect all atoms in the batch to score with KGE
            all_atoms_in_batch, atom_map = [], {}
            for i, (q, negs) in enumerate(zip(batch, corrs)):
                q_str = f"{q.predicate}({','.join(map(str, q.args))})"
                all_atoms_in_batch.append(q_str)
                atom_map[q_str] = (i, 0)
                for j, neg_tuple in enumerate(negs):
                    neg_str = f"{neg_tuple.predicate}({','.join(map(str, neg_tuple.args))})"
                    all_atoms_in_batch.append(neg_str)
                    atom_map[neg_str] = (i, j + 1)

            print(f"Atom names in batch: {all_atoms_in_batch}")
            # print(f"scores for each atom:")
            # print(np.round(log_probs, 3))
            # Get KGE scores and reshape
            kge_scores = kge_inference_engine.predict_batch(all_atoms_in_batch)
            print(f"KGE scores for each atom:")
            print(np.round(kge_scores, 10))
            kge_log_scores_flat = np.log(np.array(kge_scores) + 1e-9)
            max_target_in_batch = targets.max()
            kge_log_scores = np.full((B, max_target_in_batch), -np.inf, dtype=float)
            print(f"KGE logscores for each atom:")
            print(np.round(kge_log_scores, 3))
            for atom_str, log_score in zip(all_atoms_in_batch, kge_log_scores_flat):
                env_idx, ep_idx = atom_map[atom_str]
                kge_log_scores[env_idx, ep_idx] = log_score
            
            # Combine Scores: if proof succeeded, score = rl_log_prob + kge_log_score, else score = kge_log_score
            # log_probs = np.where(proof_successful, log_probs + kge_log_scores, kge_log_scores)
            log_probs = kge_log_scores  # Use KGE scores only if proof was not successful

        # --- Classification Metrics Logic ---
        # Create true labels based on the query type (positive or negative)
        # Positive query is always at index 0 for each environment in the batch
        true_labels = np.zeros_like(mask, dtype=int)
        true_labels[:, 0] = 1 # Mark positive queries

        # Use the mask to get only valid (non-padded) episodes
        y_true_batch = true_labels[mask]
        y_pred_batch = proof_successful[mask].astype(int)

        all_y_true.extend(y_true_batch.tolist())
        all_y_pred.extend(y_pred_batch.tolist())

        # Collect metrics using mask
        # Positives: column 0
        rwds_pos = rewards[:, 0][mask[:, 0]]
        lengths_pos = lengths[:, 0][mask[:, 0]]
        log_probs_pos = log_probs[:, 0][mask[:, 0]]
        rwds_neg = rewards[:, 1:][mask[:, 1:]]
        lengths_neg = lengths[:, 1:][mask[:, 1:]]
        log_probs_neg = log_probs[:, 1:][mask[:, 1:]]

        all_pos_rw.extend(rwds_pos)
        all_pos_len.extend(lengths_pos)
        all_pos_lp.extend(log_probs_pos)

        # Collect negatives across all valid slots
        all_neg_rw.extend(rwds_neg)
        all_neg_len.extend(lengths_neg)
        all_neg_lp.extend(log_probs_neg)

        # Rank-based metrics
        if mask.shape[1] > 1:
            lp_batch = np.where(mask, log_probs, -np.inf)
            # True positive is at column 0
            random_keys = np.random.rand(*lp_batch.shape)
            sorted_indices = np.lexsort((-random_keys, -lp_batch), axis=1)
            ranks = np.where(sorted_indices == 0)[1] + 1
            mrr = 1.0 / ranks
            hits1, hits3, hits10 = (ranks == 1).astype(int), (ranks <= 3).astype(int), (ranks <= 10).astype(int)

            mrr_list.extend(mrr.tolist())
            hits1_list.extend(hits1.tolist())
            hits3_list.extend(hits3.tolist())
            hits10_list.extend(hits10.tolist())

        if verbose:
            # print(f"\nQueries in batch: ")
            # for (q, negs) in zip(batch, corrs):
            #     print(f"  {q} + {[n for n in negs]}")

            # set numpy option to print full arrays
            # np.set_printoptions(threshold=np.inf) # Set threshold to infinity to show all elements
            # np.set_printoptions(linewidth=1000) # Adjust linewidth if your array has many columns to prevent wrapping
            # print(f"Logprobs: {np.round(log_probs, 3)}")
            # print(f" Number of queries in batch with 0 logprob: {np.sum(log_probs == 0)}")
            # print(f"ranks: {ranks}") if mask.shape[1] > 1 else None

            print(f"\nrwds pos    : {np.round(np.mean(rwds_pos), 3)}   \trwds neg       : {np.round(np.mean(rwds_neg), 3)}")
            print(f"ep len pos  : {np.round(np.mean(lengths_pos), 3)} \tep len neg    : {np.round(np.mean(lengths_neg), 3)}")
            print(f"logprobs pos: {np.round(np.mean(log_probs_pos), 3)} \tlog probs neg  : {np.round(np.mean(log_probs_neg), 3)}")
            
            print(f"\nAccuracy: {accuracy_score(y_true_batch, y_pred_batch):.3f} \tPrecision: {precision_score(y_true_batch, y_pred_batch, zero_division=0):.3f}")
            print(f"Recall: {recall_score(y_true_batch, y_pred_batch, zero_division=0):.3f} \tF1 Score: {f1_score(y_true_batch, y_pred_batch, zero_division=0):.3f}")
            print(f"\nrolling Accuracy: {accuracy_score(all_y_true, all_y_pred):.3f} \trolling Precision: {precision_score(all_y_true, all_y_pred, zero_division=0):.3f}")
            print(f"rolling Recall: {recall_score(all_y_true, all_y_pred, zero_division=0):.3f} \trolling F1 Score: {f1_score(all_y_true, all_y_pred, zero_division=0):.3f}")

            print('\nrolling rwds pos    :',np.round(np.mean(all_pos_rw),3)    , '\trolling rwds neg       :',np.round(np.mean(all_neg_rw),3))
            print('rolling ep len pos  :',np.round(np.mean(all_pos_len),3), '\trolling episode len neg:',np.round(np.mean(all_neg_len),3))
            print('rolling logprobs pos:',np.round(np.mean(all_pos_lp),3)  , '\trolling log probs neg  :',np.round(np.mean(all_neg_lp),3))

            if mask.shape[1] > 1:
                print('\nmrr   :',np.round(np.mean(mrr),3)   ,'\trolling mrr   :',np.round(np.mean(mrr_list),3))
                print('hits1 :',np.round(np.mean(hits1),3) ,'\trolling hits1 :',np.round(np.mean(hits1_list),3))
                print('hits3 :',np.round(np.mean(hits3),3) ,'\trolling hits3 :',np.round(np.mean(hits3_list),3))
                print('hits10:',np.round(np.mean(hits10),3),'\trolling hits10:',np.round(np.mean(hits10_list),3))


        # --- PLOTTING LOGIC ---
        if track_logprobs:
            hist_idx = 0
            # This dict will store tuples of (logprob_history, choices_history, step_logprob_hist, state_history)
            # for the current batch only.
            batch_plot_data = {"pos_success": [], "pos_fail": [], "neg_success": [], "neg_fail": []}

            for i in range(B): # Iterate through envs in the batch
                num_episodes_in_env = targets[i]
                labels_for_env = [1] + [0] * (num_episodes_in_env - 1)
                
                for j in range(num_episodes_in_env):
                    if not mask[i, j] or hist_idx >= len(logprob_histories): continue
                    
                    # Use the proof_successful mask for categorization instead of reward
                    label = labels_for_env[j]
                    success = proof_successful[i, j]
                    log_hist, choice_hist, step_log_hist, state_hist = logprob_histories[hist_idx], choices_histories[hist_idx], steplogprob_histories[hist_idx], state_histories[hist_idx]
                    hist_idx += 1
                    # Categorize the episode data based on label and success status
                    data_tuple = (log_hist, choice_hist, step_log_hist, state_hist)
                    # print('Batch i, data tuple', data_tuple)
                    if label == 1 and success:
                        batch_plot_data["pos_success"].append(data_tuple)
                    elif label == 1 and not success:
                        batch_plot_data["pos_fail"].append(data_tuple)
                    elif label == 0 and success:
                        batch_plot_data["neg_success"].append(data_tuple)
                    elif label == 0 and not success:
                        batch_plot_data["neg_fail"].append(data_tuple)
            
            # 1. PLOT AND SAVE THE CURRENT BATCH
            batch_filename = f"plots/{date_str}_logprob_heatmap_batch_{b}.png"
            plot_logprob_heatmap(
                batch_plot_data,
                filename=batch_filename,
                multicolor_lines=True,
                show_state_labels=True
            )
            
            # 2. AGGREGATE DATA FOR THE FINAL PLOT
            for key in aggregated_plot_data:
                aggregated_plot_data[key].extend(batch_plot_data[key])

    # --- FINAL AGGREGATED PLOT GENERATION ---
    if track_logprobs:
        aggregated_filename = f"plots/{date_str}_logprob_heatmap_aggregated.png"
        plot_logprob_heatmap(
            aggregated_plot_data,
            filename=aggregated_filename,
            multicolor_lines=True,
            show_state_labels=False,        # Hide state labels to avoid clutter
            show_action_counts=True,      # Hide action counts on dots
            show_avg_trajectory=True,      # Show the average trend line
            # max_trajectories_to_plot=100,  # Limit trajectories to prevent mess
            dot_alpha=0.7,                 # Make dots more transparent
            line_alpha=0.7                 # Make lines more transparent
        )
    # to NumPy arrays
    pos_rw, pos_len, pos_lp = np.array(all_pos_rw), np.array(all_pos_len), np.array(all_pos_lp)
    neg_rw, neg_len, neg_lp = np.array(all_neg_rw), np.array(all_neg_len), np.array(all_neg_lp)
    mrr_arr, h1, h3, h10 = np.array(mrr_list), np.array(hits1_list), np.array(hits3_list), np.array(hits10_list)

    # Calculate final classification metrics
    try:
        accuracy = accuracy_score(all_y_true, all_y_pred)
        precision = precision_score(all_y_true, all_y_pred, zero_division=0)
        recall = recall_score(all_y_true, all_y_pred, zero_division=0)
        f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
    except ValueError:
        accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
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
        # Add classification metrics to the returned dictionary
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
    }


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
    dot_size: int = 150
):
    """
    Plots logprob vs. step with enhanced visualization options.
    """
    fig, axs = plt.subplots(2, 2, figsize=(22, 18), sharex=True, sharey=True)
    axs = axs.flatten()

    titles = {
        "pos_success": "Positive Query -> Proved True",
        "pos_fail": "Positive Query -> Failed",
        "neg_success": "Negative Query -> Proved True (Model Error)",
        "neg_fail": "Negative Query -> Proved False (Correct)",
    }

    all_steplogprobs = [
        step_lp
        for category in data_dict.values()
        for _, _, step_logprob_hist, _ in category
        for step_lp in step_logprob_hist
    ]
    vmin = min(all_steplogprobs) if all_steplogprobs else -5.0
    vmax = max(all_steplogprobs) if all_steplogprobs else 0.0

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.coolwarm_r

    scatter_mappable = None

    for i, key in enumerate(titles.keys()):
        ax = axs[i]
        
        all_episodes_in_category = data_dict.get(key, [])
        
        # <<< NEW: Sample trajectories if there are too many
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
            
            ax.plot(
                steps,
                logprob_hist,
                linestyle='-',
                color=line_color,
                alpha=line_alpha, # Use parameter
                linewidth=2.5,
                zorder=1
            )

            scatter_mappable = ax.scatter(
                steps,
                logprob_hist,
                c=step_logprob_hist,
                cmap=cmap,
                norm=norm,
                alpha=dot_alpha, # Use parameter
                s=dot_size,      # Use parameter
                edgecolors='black',
                linewidth=0.5,
                zorder=2
            )
            
            # <<< NEW: Toggle for action counts
            if show_action_counts:
                for j, step_lp in enumerate(logprob_hist):
                    num_choices = choice_hist[j]
                    ax.text(
                        x=j, y=step_lp, s=str(num_choices),
                        ha='center', va='center', fontsize=8,
                        color='white', fontweight='bold'
                    )
            bbox = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75)
            if show_state_labels:
                y_min, y_max = ax.get_ylim()
                offset = (y_max - y_min) * 0.03 if y_max > y_min else 0.1
                for j, step_lp in enumerate(logprob_hist):
                    if j < len(state_hist):
                        state_label = state_hist[j]
                        ax.text(
                            x=j, y=step_lp + offset, s=state_label,
                            ha='center', va='bottom', fontsize=8,
                            fontweight='light', color='darkslategray',
                            rotation=15,zorder=5,clip_on=False,
                            bbox=bbox,
                        )
        
        # Plot average trajectory
        if show_avg_trajectory and all_episodes_in_category:
            log_hists = [h for h, _, _, _ in all_episodes_in_category]
            max_len = max(len(h) for h in log_hists if len(h) > 0)
            if max_len > 0:
                # Pad histories to the same length with NaN for averaging
                padded_hists = np.full((len(log_hists), max_len), np.nan)
                for i_hist, hist in enumerate(log_hists):
                    padded_hists[i_hist, :len(hist)] = hist
                
                # Calculate mean, ignoring NaNs
                mean_trajectory = np.nanmean(padded_hists, axis=0)
                ax.plot(
                    np.arange(max_len), mean_trajectory,
                    color='black', linestyle='--', linewidth=3,
                    label='Average Trajectory', zorder=3
                )
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