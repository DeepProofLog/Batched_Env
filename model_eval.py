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


def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    verbose: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns arrays of rewards,
    episode lengths and log-probs per episode. See docstring for details.
    """
    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  

    n_envs = env.num_envs
    episode_rewards = np.zeros((n_envs, n_eval_episodes//n_envs))
    episode_lengths = np.zeros((n_envs, n_eval_episodes//n_envs))
    episode_log_probs = np.zeros((n_envs, n_eval_episodes//n_envs)) 
    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")
    total_episodes_to_run = np.sum(episode_count_targets)


    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    current_log_probs = np.zeros(n_envs)
    observations = env.reset()
    total_completed_episodes = 0 # Track total completed episodes
    progress_bar_width = 50 # Width of the progress message area
    print(f"Evaluating {total_episodes_to_run} episodes.")
    while (episode_counts < episode_count_targets).any():

        if verbose >= 1:
            # Calculate total completed episodes from the counts array
            total_completed_episodes = np.sum(episode_counts)
            completed_envs_count = np.sum(episode_counts >= episode_count_targets)
            # Use \r to return to the beginning of the line, end='' to prevent newline
            progress_str = (f"Evaluating episodes: {total_completed_episodes}/{total_episodes_to_run} | "
                            f"Envs done: {completed_envs_count}/{n_envs}")

            # Pad with spaces to clear previous longer messages and flush stdout
            print(f"\r{progress_str:<{progress_bar_width}}", end="")
            sys.stdout.flush() # Ensure it gets displayed immediately

        obs_tensor = obs_as_tensor(observations, model.device)
        actions, values, log_probs = model.policy(obs_tensor, deterministic=deterministic)
        log_probs = log_probs.detach().cpu().numpy()

        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        current_log_probs += log_probs

        # reset current_rewards, current_lengths, current_log_probs where dones is True
        if dones.any():
            # Only update for environments where dones is True and episode_counts < episode_count_targets
            update_mask = dones & (episode_counts < episode_count_targets) # Shape: (n_envs,)
            if update_mask.any():
                # In the second dim, just update the next index
                episode_rewards[update_mask, episode_counts[update_mask]] = current_rewards[update_mask]
                episode_lengths[update_mask, episode_counts[update_mask]] = current_lengths[update_mask]
                episode_log_probs[update_mask, episode_counts[update_mask]] = current_log_probs[update_mask]

        current_rewards[dones] = 0
        current_lengths[dones] = 0
        current_log_probs[dones] = 0
        episode_counts[dones] += 1

        observations = new_observations

    # --- Clear progress bar after loop ---
    if verbose >= 1:
        print("\r" + " " * progress_bar_width + "\r", end="") # Clear the line
        sys.stdout.flush()

    return episode_rewards, episode_lengths, episode_log_probs



def eval_corruptions(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    data: List[Any],
    sampler: Any = None,
    n_corruptions: int = None,
    deterministic: bool = True,
    verbose: int = 1,
    consult_janus: bool = False,
    plot: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates model performance by comparing original queries against their corrupted versions.
    Optimized for vectorized environments, processing queries in parallel where possible.
    
    :param model: The RL model to evaluate
    :param env: The environment or vectorized environment
    :param data: List of original queries to evaluate
    :param corruptions: Dictionary mapping original queries to their corrupted versions
    :param corruption_mode: 'static' (use provided corruptions) or 'dynamic' (generate on the fly)
    :param n_corruptions: Number of corruptions to use per query
    :param deterministic: Whether to use deterministic policy
    :param verbose: Verbosity level
    :param consult_janus: Whether to consult Janus for query processing
    :return: Dictionary with evaluation metrics
    """
    mrr_list, hits1_list, hits3_list, hits10_list = [], [], [], []
    rewards_list_pos, episode_len_list_pos, log_probs_list_pos = [], [], []
    rewards_list_neg, episode_len_list_neg, log_probs_list_neg = [], [], []
    
    is_vec_env = isinstance(env, VecEnv)
    num_envs = env.num_envs if is_vec_env else 1

    if n_corruptions == -1: n_corruptions = None
    
    print(f"Evaluating {len(data)} queries.")
    print(f"Max N corruptions per query: {'All' if n_corruptions is None else n_corruptions}")
    print(f"Using {'vectorized' if is_vec_env else 'single'} environment with {num_envs} envs")

    total_batches = (len(data) + num_envs - 1) // num_envs # Calculate total number of batches

    for b, batch_start in enumerate(range(0, len(data), num_envs)):
        batch_end = min(batch_start + num_envs, len(data))
        batch_queries = data[batch_start:batch_end]

        print(f"\n--- Batch {b+1}/{total_batches} (Queries {batch_start+1}-{batch_end}) ---") if verbose >= 1 else None

        print("Creating corruptions...") if verbose >= 1 else None
        neg_time = time.time()
        batch_corruptions = sampler.get_negatives_from_states([[query] for query in batch_queries],
                                                              model.device,
                                                              all_negatives=True if n_corruptions is None else False,
                                                              )
        assert all(len(corruption) == len(batch_corruptions[0]) for corruption in batch_corruptions), "All batch corruptions must have the same length"
        print(f"Batch corruptions took {time.time()-neg_time:.3f}s") if verbose >= 1 else None

        # For each query in the batch, reset its env with the query and the corruptions
        corruptions_list = []
        for env_idx, query in enumerate(batch_queries):

            query_corruptions = batch_corruptions[env_idx]
            corruptions_list.append(len(query_corruptions))

            eval_data = [query] + query_corruptions
            eval_labels = [1] + [0] * len(query_corruptions)

            env.envs[env_idx].env.mode = 'eval'
            env.envs[env_idx].env.queries = eval_data
            env.envs[env_idx].env.labels = eval_labels
            env.envs[env_idx].env.eval_idx = 0
            env.envs[env_idx].env.n_episodes = len(eval_data)
            env.envs[env_idx].env.consult_janus_eval = consult_janus

        assert all(corruptions_list[c] == corruptions_list[0] for c in range(len(corruptions_list))), "All environments must have the same number of corruptions"
        n_corruptions = corruptions_list[0]
        time_batch = time.time()
        # evaluate the model for n_eval_episodes=n_envs*(1+n_corruptions)
        rewards, lengths, log_probs = evaluate_policy(model, 
                                                    env, 
                                                    n_eval_episodes=num_envs*(1+n_corruptions), 
                                                    deterministic=deterministic,
                                                    verbose=verbose)
                                                    
        print(f'Batch {b+1} evaluation took {time.time()-time_batch:.3f}s')

        filter_mask = None
        if len(batch_queries) < num_envs:
            filter_mask = len(batch_queries)
        rewards = rewards[:filter_mask,:1+n_corruptions]
        lengths = lengths[:filter_mask,:1+n_corruptions]
        log_probs = log_probs[:filter_mask,:1+n_corruptions]

        # where the rewards are 0, substract 100 to the  (heuristic to differentiate between proof and non-proof)
        log_probs[rewards == 0] -= 100

        rewards_list_pos.extend(rewards[:,0])
        episode_len_list_pos.extend(lengths[:,0])
        log_probs_list_pos.extend(log_probs[:,0])

        rewards_list_neg.extend(rewards[:,1:])
        episode_len_list_neg.extend(lengths[:,1:])
        log_probs_list_neg.extend(log_probs[:,1:])

        # Batch calculation of ranks and MRR
        if rewards.shape[1] > 1:  # If there're corruptions
            # Sort log_probs in descending order and get indices
            sorted_indices = np.argsort(-log_probs, axis=-1)
            # Find position of positive examples (index 0) for each environment
            ranks = np.argmax(sorted_indices == 0, axis=-1) + 1
            # Calculate MRR for each environment
            batch_mrr = 1.0 / ranks
            mrr_list.extend(batch_mrr.tolist())
            # Calculate Hits@k for the batch
            batch_hits1 = (ranks == 1).astype(int)
            batch_hits3 = (ranks <= 3).astype(int)
            batch_hits10 = (ranks <= 10).astype(int)
            hits1_list.extend(batch_hits1.tolist())
            hits3_list.extend(batch_hits3.tolist())
            hits10_list.extend(batch_hits10.tolist())
        if verbose >= 1:
            print('\nrolling rwds pos    :',np.round(np.mean(rewards_list_pos),3)    , '\trolling rwds neg       :',np.round(np.mean(rewards_list_neg),3))
            print('rolling ep len pos  :',np.round(np.mean(episode_len_list_pos),3), '\trolling episode len neg:',np.round(np.mean(episode_len_list_neg),3))
            print('rolling logprobs pos:',np.round(np.mean(log_probs_list_pos),3)  , '\trolling log probs neg  :',np.round(np.mean(log_probs_list_neg),3))
            if rewards.shape[1] > 1:
                print('\nmrr   :',np.round(np.mean(batch_mrr),3)   ,'\trolling mrr   :',np.round(np.mean(mrr_list),3)) 
                print('hits1 :',np.round(np.mean(batch_hits1),3) ,'\trolling hits1 :',np.round(np.mean(hits1_list),3))
                print('hits3 :',np.round(np.mean(batch_hits3),3) ,'\trolling hits3 :',np.round(np.mean(hits3_list),3))
                print('hits10:',np.round(np.mean(batch_hits10),3),'\trolling hits10:',np.round(np.mean(hits10_list),3))
    
    # Prepare scores and labels for AUC-PR calculation
    log_probs_list_pos = np.array(log_probs_list_pos)
    log_probs_list_neg = np.array(log_probs_list_neg).reshape(-1,1).squeeze()
    scores = np.concatenate([log_probs_list_pos, log_probs_list_neg])
    labels = np.concatenate([np.ones(len(log_probs_list_pos)), np.zeros(len(log_probs_list_neg))])
    auc_pr = average_precision_score(labels, scores)

    rewards_pos = np.array(rewards_list_pos)
    rewards_neg = np.array(rewards_list_neg).reshape(-1,1).squeeze()
    episode_len_pos = np.array(episode_len_list_pos)
    episode_len_neg = np.array(episode_len_list_neg).reshape(-1,1).squeeze()
    log_probs_pos = np.array(log_probs_list_pos)
    log_probs_neg = np.array(log_probs_list_neg).reshape(-1,1).squeeze()
    mrr_array = np.array(mrr_list)
    hits1_array = np.array(hits1_list)
    hits3_array = np.array(hits3_list)
    hits10_array = np.array(hits10_list)

    if plot:
        # plot the positive and negative logit distributions as points
        # Create indices for x-axis positioning
        x_pos = np.arange(len(log_probs_pos))
        x_neg = np.arange(len(log_probs_neg))
        
        # Plot points instead of histograms
        plt.scatter(x_pos, log_probs_pos, alpha=0.8, label='positive', color='blue')
        plt.scatter(x_neg, log_probs_neg, alpha=0.5, label='negative', color='red')
        
        plt.legend(loc='upper right')
        plt.xlabel('Sample index')
        plt.ylabel('Log probability')
        plt.show()

    # Return evaluation metrics
    return {
        'pos_queries': len(rewards_pos),
        'neg_queries': len(rewards_neg),
        'ratio_pos_queries': round(len(rewards_pos)/(len(rewards_pos)+len(rewards_neg)), 2),
        'mrr_mean': float(np.mean(mrr_array)) if len(mrr_array) > 0 else 0,
        'mrr_std': float(np.std(mrr_array)) if len(mrr_array) > 0 else 0,
        
        'rewards_pos_mean': float(np.mean(rewards_pos)),
        'rewards_pos_std': float(np.std(rewards_pos)),
        'rewards_neg_mean': float(np.mean(rewards_neg)) if len(rewards_neg) > 0 else 0,
        'rewards_neg_std': float(np.std(rewards_neg)) if len(rewards_neg) > 0 else 0,
        'rewards_mean': float(np.mean(np.concatenate([rewards_pos, rewards_neg]))),
        'rewards_std': float(np.std(np.concatenate([rewards_pos, rewards_neg]))),
        
        'episode_len_pos_mean': float(np.mean(episode_len_pos)),
        'episode_len_pos_std': float(np.std(episode_len_pos)),
        'episode_len_neg_mean': float(np.mean(episode_len_neg)) if len(episode_len_neg) > 0 else 0,
        'episode_len_neg_std': float(np.std(episode_len_neg)) if len(episode_len_neg) > 0 else 0,
        'episode_len_mean': float(np.mean(np.concatenate([episode_len_pos, episode_len_neg]))),
        'episode_len_std': float(np.std(np.concatenate([episode_len_pos, episode_len_neg]))),
        
        'log_probs_pos_mean': float(np.mean(log_probs_pos)),
        'log_probs_pos_std': float(np.std(log_probs_pos)),
        'log_probs_neg_mean': float(np.mean(log_probs_neg)) if len(log_probs_neg) > 0 else 0,
        'log_probs_neg_std': float(np.std(log_probs_neg)) if len(log_probs_neg) > 0 else 0,
        'log_probs_mean': float(np.mean(np.concatenate([log_probs_pos, log_probs_neg]))),
        'log_probs_std': float(np.std(np.concatenate([log_probs_pos, log_probs_neg]))),
        
        'auc_pr': float(auc_pr),
        'hits1_mean': float(np.mean(hits1_array)),
        'hits1_std': float(np.std(hits1_array)),
        'hits3_mean': float(np.mean(hits3_array)),
        'hits3_std': float(np.std(hits3_array)),
        'hits10_mean': float(np.mean(hits10_array)),
        'hits10_std': float(np.std(hits10_array)),
    }