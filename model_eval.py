from typing import Any, Dict, List, Tuple, Union, Optional
import time
import sys
import os
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


def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    target_episodes: np.ndarray = None,
    verbose: int = 0,
    track_logprobs: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Run policy for a specified number of episodes per env, returning
    rewards, lengths, log_probs, a validity mask, a proof success mask,
    and histories for plotting.
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
        print(f"\nEvaluating {total} episodes on {n_envs} envs (avg target: {targets.mean()})")

    env._episode_target[:] = padded_targets
    env._episode_count[:] = 0
    env.active_envs[:] = True

    rewards = np.zeros((n_envs, padded_targets.max()), dtype=float)
    lengths = np.zeros_like(rewards, dtype=int)
    logps   = np.zeros_like(rewards, dtype=float)
    # Array to store whether a proof was successfully found
    proof_successful = np.zeros_like(rewards, dtype=bool)
    counts  = np.zeros(n_envs, dtype=int)

    # History tracking
    episode_logprob_histories = []
    episode_choices_histories = []
    episode_steplogprob_histories = []
    episode_state_histories = []
    if track_logprobs:
        current_logprob_histories = [[] for _ in range(n_envs)]
        current_choices_histories = [[] for _ in range(n_envs)]
        current_steplogprob_histories = [[] for _ in range(n_envs)]
        current_state_histories = [[] for _ in range(n_envs)]

    index_manager = env.get_attr("index_manager")[0]
    true_pred_idx = index_manager.predicate_str2idx['True']

    observations = env.reset()
    current_rew = np.zeros(n_envs, dtype=float)
    current_len = np.zeros(n_envs, dtype=int)
    current_lp  = np.zeros(n_envs, dtype=float)

    while (counts < padded_targets).any():
        # print("counts per env:", counts)
        active = counts < padded_targets

        # --- Data Collection for Plotting ---
        if track_logprobs:
            # Get number of choices from the *current* observation, before taking the step
            with torch.no_grad():
                obs_tensor_for_choices = obs_as_tensor(observations, model.device)
                # A choice is valid if the sum of its atom indices is not zero
                num_choices = (obs_tensor_for_choices["derived_sub_indices"].sum(dim=(-1, -2)) != 0).sum(dim=-1)
                num_choices_np = num_choices.cpu().numpy()

            all_sub_indices = observations["sub_index"]
            # Convert state observations to strings for active envs
            for env_idx in np.where(active)[0]:
                # Squeeze out the middle dimension of size 1, shape becomes (padding_atoms, max_arity+1)
                state_subidx_tensor = all_sub_indices[env_idx].squeeze(0)
                state_str = index_manager.state_subindex_to_str(state_subidx_tensor,truncate=True)
                current_state_histories[env_idx].append(state_str)


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

        # --- Store Histories ---
        if track_logprobs:
            for i, active_env in enumerate(np.where(active)[0]):
                current_logprob_histories[active_env].append(current_lp[active_env])
                current_choices_histories[active_env].append(num_choices_np[active_env])
                current_steplogprob_histories[active_env].append(lp_active[i])

        # create a full‐size array with the correct NumPy dtype
        full_actions = np.zeros((n_envs, *actions_active.shape[1:]), dtype=actions_active.dtype)
        full_actions[active] = actions_active

        # step all envs (inactive ones will be skipped internally)
        new_obs, rew, dones, infos = env.step(full_actions)
        # print('INFOS is success:', infos)
        # print("New observations in string format:")
        # for idx in np.where(active)[0]:
        #     # Squeeze out the middle dimension of size 1, shape becomes (padding_atoms, max_arity+1)
        #     state_subidx_tensor = new_obs["sub_index"][idx].squeeze(0)
        #     state_str = index_manager.state_subindex_to_str(state_subidx_tensor, truncate=True)
        #     print(f"Env {idx}: {state_str}")

        current_rew[active] += rew[active]
        current_len[active] += 1

        # finalize any envs that just finished
        done_and_active = dones & active
        for idx in np.nonzero(done_and_active)[0]:
            # print(f"\nEnv {idx} done after {current_len[idx]} steps with reward {current_rew[idx]:.3f}")
            slot = counts[idx]
            rewards[idx, slot] = current_rew[idx]
            lengths[idx, slot] = current_len[idx]
            logps[idx, slot]   = current_lp[idx]

            # final_sub_index = new_obs["sub_index"][idx]
            # print(f"Is the final state True? {final_sub_index[0, 0, 0] == true_pred_idx}, expected {true_pred_idx}")
            # print(f"String of final state sub_index: {index_manager.state_subindex_to_str(final_sub_index[0])}")

            # Check the `info` dictionary for the success flag provided by the Monitor wrapper.
            # This is more robust than inspecting the final observation as it handles truncations correctly.
            # The 'is_success' key is added in the custom environment's step function.
            is_success = infos[idx].get("is_success", False)
            # print(f"Env {idx} success: {is_success}")
            if is_success:
                proof_successful[idx, slot] = True

            counts[idx]       += 1
            current_rew[idx]   = 0
            current_len[idx]   = 0
            current_lp[idx]    = 0

            if track_logprobs:
                # An episode is done, so 'new_obs' holds the terminal state (e.g., True() or False()).
                # Append this final state's string representation to the history before saving.
                terminal_state_subidx = new_obs["sub_index"][idx].squeeze(0)
                # If the episode was truncated (e.g. max_depth), label it as a failure for the plot.
                if infos[idx].get("TimeLimit.truncated", False):
                    terminal_state_str = "False()"
                else:
                    terminal_state_str = index_manager.state_subindex_to_str(terminal_state_subidx)
                current_state_histories[idx].append(terminal_state_str)
                episode_logprob_histories.append(np.array(current_logprob_histories[idx]))
                episode_choices_histories.append(np.array(current_choices_histories[idx]))
                episode_steplogprob_histories.append(np.array(current_steplogprob_histories[idx]))
                episode_state_histories.append(np.array(current_state_histories[idx]))
                current_logprob_histories[idx] = []
                current_choices_histories[idx] = []
                current_steplogprob_histories[idx] = []
                current_state_histories[idx] = []

        observations = new_obs

        if verbose: print(f"\rEpisodes done: {counts.sum()}/{total}", end="", flush=True)

    if verbose: print("\r" + " " * 80 + "\r", end="")

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
        proof_successful = proof_successful[:targets_len]

    return rewards, lengths, logps, mask, proof_successful, episode_logprob_histories, episode_choices_histories, episode_steplogprob_histories, episode_state_histories

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
            verbose=1,
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
            lp_batch = log_probs.copy()
            lp_batch[~mask] = -np.inf
            # True positive is at column 0
            random_keys = np.random.rand(*lp_batch.shape)
            sorted_indices_per_query = np.lexsort((-random_keys, -lp_batch), axis=1)
            ranks = np.where(sorted_indices_per_query == 0)[1] + 1
            mrr = 1.0 / ranks
            hits1, hits3, hits10 = (ranks == 1).astype(int), (ranks <= 3).astype(int), (ranks <= 10).astype(int)

            mrr_list.extend(mrr.tolist())
            hits1_list.extend(hits1.tolist())
            hits3_list.extend(hits3.tolist())
            hits10_list.extend(hits10.tolist())

        if verbose:
            print(f"\nQueries in batch: ")
            for (q, negs) in zip(batch, corrs):
                print(f"  {q} + {[n for n in negs]}")

            # set numpy option to print full arrays
            # np.set_printoptions(threshold=np.inf) # Set threshold to infinity to show all elements
            # np.set_printoptions(linewidth=1000) # Adjust linewidth if your array has many columns to prevent wrapping
            # print(f"Logprobs: {np.round(log_probs, 3)}")
            # print(f" Number of queries in batch with 0 logprob: {np.sum(log_probs == 0)}")
            # print(f"ranks: {ranks}") if mask.shape[1] > 1 else None

            print(f"\nrwds pos    : {np.round(np.mean(rwds_pos), 3)}   \trwds neg       : {np.round(np.mean(rwds_neg), 3)}")
            print(f"ep len pos  : {np.round(np.mean(lengths_pos), 3)} \tep len neg    : {np.round(np.mean(lengths_neg), 3)}")
            print(f"logprobs pos: {np.round(np.mean(log_probs_pos), 3)} \tlog probs neg  : {np.round(np.mean(log_probs_neg), 3)}")

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
                            rotation=15
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
