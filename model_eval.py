import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from sklearn.metrics import average_precision_score

from stable_baselines3.common import type_aliases
from stable_baselines3.common.on_policy_algorithm import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from utils import Term




def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    warn: bool = True,
) -> Union[tuple[float, float], tuple[list[float], list[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = np.zeros((n_envs, n_eval_episodes//n_envs))
    episode_lengths = np.zeros((n_envs, n_eval_episodes//n_envs))
    episode_log_probs = np.zeros((n_envs, n_eval_episodes//n_envs)) 
    # print('n eval episodes:',n_eval_episodes)
    # print('episode_rewards:',episode_rewards.shape, episode_rewards)
    # print('episode_lengths:',episode_lengths.shape, episode_lengths)
    # print('episode_log_probs:',episode_log_probs.shape, episode_log_probs)
    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    current_log_probs = np.zeros(n_envs)
    observations = env.reset()
    # states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        # print()
        # print('episode_counts:',episode_counts, '/', episode_count_targets)
        # for idx in range(n_envs):
        #     print('mask of env',idx, 'is', len(env.envs[idx].env.mask_eval), env.envs[idx].env.mask_eval)
        # actions, states = model.predict(
        #     observations,  # type: ignore[arg-type]
        #     state=states,
        #     episode_start=episode_starts,
        #     deterministic=deterministic,
        # )
        obs_tensor = obs_as_tensor(observations, model.device)
        actions, values, log_probs = model.policy(obs_tensor, deterministic=deterministic)
        log_probs = log_probs.detach().cpu().numpy()

        new_observations, rewards, dones, infos = env.step(actions)
        # print('new_observations:',[(k,v.shape) for k,v in new_observations.items()])
        current_rewards += rewards
        current_lengths += 1
        current_log_probs += log_probs
        # increase episode_counts by 1 where dones is True
        # print('actions:',actions)
        # print('rewards:',rewards, 'current_rewards:',current_rewards)
        # print('log_probs:',log_probs, 'current_log_probs:',current_log_probs)
        # print('current_lengths:',current_lengths)
        # print('dones:',dones)
        # reset current_rewards, current_lengths, current_log_probs where dones is True
        if dones.any():
            # Only update for environments where dones is True and episode_counts < episode_count_targets
            update_mask = dones & (episode_counts < episode_count_targets)
            if update_mask.any():
                # Update episode_rewards, episode_lengths, episode_log_probs for each valid done env
                episode_rewards[update_mask, episode_counts[update_mask]] = current_rewards[update_mask]
                episode_lengths[update_mask, episode_counts[update_mask]] = current_lengths[update_mask]
                episode_log_probs[update_mask, episode_counts[update_mask]] = current_log_probs[update_mask]
                # print('episode_rewards:',episode_rewards, '\nepisode_lengths:',episode_lengths, '\nepisode_log_probs:',episode_log_probs)

        current_rewards[dones] = 0
        current_lengths[dones] = 0
        current_log_probs[dones] = 0
        episode_counts[dones] += 1
        # print('current_rewards updated:',current_rewards, 'current_lengths updated:',current_lengths, 'current_log_probs updated:',current_log_probs)
        # print('episode_counts updated:',episode_counts)
        # for idx in range(n_envs):
        #     print('updated mask of env',idx, 'is', len(env.envs[idx].env.mask_eval), env.envs[idx].env.mask_eval)
        # if not all are done, update observations
        # if not np.all(dones):
        observations = new_observations
        
        # for i in range(n_envs):
        #     if episode_counts[i] < episode_count_targets[i]:
        #         # unpack values so that the callback can access the local variables
        #         reward = rewards[i]
        #         done = dones[i]
        #         info = infos[i]
        #         episode_starts[i] = done

        #         if dones[i]:
        #             if is_monitor_wrapped:
        #                 # Atari wrapper can send a "done" signal when
        #                 # the agent loses a life, but it does not correspond
        #                 # to the true end of episode
        #                 if "episode" in info.keys():
        #                     # Do not trust "done" with episode endings.
        #                     # Monitor wrapper includes "episode" key in info if environment
        #                     # has been wrapped with it. Use those rewards instead.
        #                     episode_rewards.append(info["episode"]["r"])
        #                     episode_lengths.append(info["episode"]["l"])
        #                     episode_log_probs.append(current_log_probs[i])
        #                     # Only increment at the real end of an episode
        #                     episode_counts[i] += 1
        #             else:
        #                 episode_rewards.append(current_rewards[i])
        #                 episode_lengths.append(current_lengths[i])
        #                 episode_log_probs.append(current_log_probs[i])
        #                 episode_counts[i] += 1
        #             current_rewards[i] = 0
        #             current_lengths[i] = 0
        #             current_log_probs[i] = 0
        # observations = new_observations

    return episode_rewards, episode_lengths, episode_log_probs



def eval_corruptions(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    data: List[Any],
    corruption_mode: str = 'static',
    corruptions: Optional[Dict[Any, List[Any]]] = None,
    n_corruptions: int = -1,
    deterministic: bool = True,
    verbose: int = 0,
    consult_janus: bool = False,
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
    mrr_list = []
    rewards_list_pos, episode_len_list_pos, log_probs_list_pos = [], [], []
    rewards_list_neg, episode_len_list_neg, log_probs_list_neg = [], [], []
    
    # Determine if we're using a vectorized environment
    is_vec_env = isinstance(env, VecEnv)
    num_envs = env.num_envs if is_vec_env else 1
    
    if verbose >= 1:
        print(f"Evaluating {len(data)} queries with {n_corruptions} corruptions each")
        print(f"Using {'vectorized' if is_vec_env else 'single'} environment with {num_envs} envs")
    
    # Process queries in batches for efficient parallel processing
    for batch_start in range(0, len(data), num_envs):
        batch_end = min(batch_start + num_envs, len(data))
        batch_size = batch_end - batch_start
        batch_queries = data[batch_start:batch_end]

        # print('batch start:',batch_start, 'batch end:',batch_end, 'batch size:',batch_size)
        
        if verbose >= 1:
            print(f"\nProcessing batch {batch_start//num_envs + 1}/{(len(data) + num_envs - 1)//num_envs}")
        else:
            print(f"\rProcessing {batch_end}/{len(data)}", end='', flush=True)
        
        # For each query in the batch, reset its env with the query and the corruptions
        corruptions_list = [] # to make sure that all environments have the same corruptions
        for env_idx, query in enumerate(batch_queries):
            # Select the appropriate environment
            query_env = env.envs[env_idx].env if is_vec_env else env
            
            # Get corruptions based on the mode
            if corruption_mode == 'static':
                if corruptions is None:
                    raise ValueError("Static corruption mode requires a corruptions dictionary")
                query_corruptions = corruptions[query][:n_corruptions]
            elif corruption_mode == 'dynamic':
                if hasattr(query_env, 'get_negatives'):
                    query_corruptions = query_env.get_negatives(query, all_negatives=True)[:n_corruptions]
                else:
                    raise ValueError("Dynamic corruption mode requires environment with get_negatives method")
            else:
                raise ValueError(f"Unknown corruption mode: {corruption_mode}")
            corruptions_list.append(len(query_corruptions))
            # Create evaluation data for this query and its corruptions
            eval_data = [query] + query_corruptions
            eval_labels = [1] + [0] * len(query_corruptions)
            # print('\n\n','-'*100)
            # print('env_idx:',env_idx)
            # print('eval_data:',eval_data)
            # print('eval_labels:',eval_labels)
            # print('-'*100,'\n')
            # Assign queries and labels to each environment and reset (if monitored, need to add extra env.)
            env.envs[env_idx].env.mode = 'eval_parallel'
            env.envs[env_idx].env.queries = eval_data
            env.envs[env_idx].env.labels = eval_labels
            env.envs[env_idx].env.eval_idx = 0
            env.envs[env_idx].env.n_episodes = len(eval_data)
            env.envs[env_idx].env.mask_eval = []
            env.envs[env_idx].env.consult_janus_eval = consult_janus

        assert all(corruptions_list[c] == corruptions_list[0] for c in range(len(corruptions_list))), "All environments must have the same number of corruptions"
        n_corruptions = corruptions_list[0]
        # evaluate the model for n_eval_episodes=n_envs*(1+n_corruptions)
        rewards, lengths, log_probs = evaluate_policy(model, env, 
                                                n_eval_episodes=num_envs*(1+n_corruptions), 
                                                deterministic=deterministic)
        # convert rewards, lengths, log_probs, mask_eval to np
        rewards = np.array(rewards)
        # print('\n\nrewards:',rewards.shape, rewards)
        lengths = np.array(lengths)
        log_probs = np.array(log_probs)
        # print('\n\nrewards:',rewards.shape, rewards, '\nlengths:',lengths.shape, lengths, '\nlog_probs:',log_probs.shape, log_probs)
        mask_eval = np.zeros((num_envs,(1+n_corruptions)), dtype=bool) # initialise it to boolean array of shape (num_envs,1+n_corruptions)
        for env_idx in range(num_envs):
            # print('mask',env_idx, len(env.envs[env_idx].env.mask_eval), env.envs[env_idx].env.mask_eval)
            # I need to filter because some envs have more than 1+n_corruptions. 
            mask_eval[env_idx] = env.envs[env_idx].env.mask_eval[:1+n_corruptions]
        # print('mask_eval:',mask_eval.shape, mask_eval)
        rewards = rewards[mask_eval].reshape(rewards.shape)
        lengths = lengths[mask_eval].reshape(lengths.shape)
        log_probs = log_probs[mask_eval].reshape(log_probs.shape)
        rewards_list_pos.extend(rewards[:,0])
        episode_len_list_pos.extend(lengths[:,0])
        log_probs_list_pos.extend(log_probs[:,0])

        #
        rewards_list_neg.extend(rewards[:,1:])
        episode_len_list_neg.extend(lengths[:,1:])
        log_probs_list_neg.extend(log_probs[:,1:])

        # Batch calculation of ranks and MRR
        if rewards.shape[1] > 1:  # If we have corruptions
            # Calculate ranks for each environment in the batch
            # Sort log_probs in descending order and get indices
            sorted_indices = np.argsort(log_probs, axis=1)[:, ::-1]
            # Find position of positive examples (index 0) for each environment
            ranks = np.where(sorted_indices == 0)[1] + 1
            # Calculate MRR for each environment
            batch_mrr = 1.0 / ranks
            mrr_list.extend(batch_mrr.tolist())

    # Prepare scores and labels for AUC-PR calculation
    log_probs_list_pos = np.array(log_probs_list_pos)
    # reshape and squeeze log_probs_list_neg
    log_probs_list_neg = np.array(log_probs_list_neg).reshape(-1,1).squeeze()
    scores = np.concatenate([log_probs_list_pos, log_probs_list_neg])
    labels = np.concatenate([np.ones(len(log_probs_list_pos)), np.zeros(len(log_probs_list_neg))])
    auc_pr = average_precision_score(labels, scores)

    # Convert lists to numpy arrays for efficient calculations
    rewards_pos = np.array(rewards_list_pos)
    rewards_neg = np.array(rewards_list_neg).reshape(-1,1).squeeze()
    episode_len_pos = np.array(episode_len_list_pos)
    episode_len_neg = np.array(episode_len_list_neg).reshape(-1,1).squeeze()
    log_probs_pos = np.array(log_probs_list_pos)
    log_probs_neg = np.array(log_probs_list_neg).reshape(-1,1).squeeze()
    mrr_array = np.array(mrr_list)


    # Return comprehensive evaluation metrics
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
        
        'auc_pr': float(auc_pr)
    }








# def evaluate_policy_nqueries(
#     model: "type_aliases.PolicyPredictor",
#     env: Union[gym.Env, VecEnv],
#     data: List[Term],
#     labels: List[int],
#     deterministic: bool = True,
#     render: bool = False,
#     callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
#     reward_threshold: Optional[float] = None,
#     return_episode_rewards: bool = False,
#     warn: bool = True,
#     verbose: int = 0,
#     return_dict:bool=True,
#     consult_janus:bool=False,
# ) -> Union[tuple[float, float], tuple[list[float], list[int]], tuple[list[float], list[int], list[float]]]:
#     """
#     Runs policy for each query in the data list and returns evaluation metrics.
    
#     :param model: The RL agent you want to evaluate.
#     :param env: The gym environment or ``VecEnv`` environment.
#     :param data: List of query Terms to evaluate
#     :param labels: List of labels corresponding to each query
#     :param deterministic: Whether to use deterministic or stochastic actions
#     :param render: Whether to render the environment or not
#     :param callback: callback function to do additional checks
#     :param reward_threshold: Minimum expected reward per episode
#     :param return_episode_rewards: If True, a list of rewards and episode lengths
#         per episode will be returned instead of the mean.
#     :param warn: If True (default), warns user about lack of a Monitor wrapper
#     :param return_log_probs: If True, returns log probabilities
#     :param verbose: Verbosity level (0: no output, 1: info, 2: debug)
#     :param return_dict: Whether to return results as a dictionary
#     :param consult_janus: Whether to consult Janus during query processing
#     :return: Mean reward per episode, std of reward per episode.
#         Returns ([float], [int]) when ``return_episode_rewards`` is True, first
#         list containing per-episode rewards and second containing per-episode lengths
#         (in number of steps). If return_log_probs is True, returns ([float], [int], [float])
#         with the third list containing log probabilities.
#     """
#     is_monitor_wrapped = False
#     # Avoid circular import
#     from stable_baselines3.common.monitor import Monitor

#     if not isinstance(env, VecEnv):
#         env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

#     is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

#     if not is_monitor_wrapped and warn:
#         warnings.warn(
#             "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
#             "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
#             "Consider wrapping environment first with ``Monitor`` wrapper.",
#             UserWarning,
#         )

#     n_envs = env.num_envs
#     episode_rewards = []
#     episode_lengths = []
#     episode_log_probs = []

#     current_rewards = np.zeros(n_envs)
#     current_lengths = np.zeros(n_envs, dtype="int")
#     current_log_probs = np.zeros(n_envs)
    
#     # Initialize with the first query
#     current_query = np.zeros(n_envs, dtype="int")
#     observations, _ = env.reset_from_query(data[0], labels[0], consult_janus=consult_janus)
#     episode_starts = np.ones((n_envs,), dtype=bool)

#     while current_query < len(data):
#         if current_query % 10 == 0 and verbose > 0: 
#             print(f'\rProcessing query {current_query[0]}/{len(data)}', end='', flush=True)
        
#         # Use policy to get action, values, and log probabilities
#         if hasattr(model, "policy") and hasattr(model.policy, "__call__"):
#             obs_tensor = obs_as_tensor(observations, model.device)
#             actions, values, log_probs = model.policy(obs_tensor, deterministic=deterministic)

#             log_probs = log_probs.detach().cpu().numpy().item()
#             current_log_probs += log_probs
#         else:
#             # Fallback to regular predict method (doesn't return log probs)
#             actions, states = model.predict(
#                 observations,
#                 deterministic=deterministic,
#             )
        
#         new_observations, rewards, dones, infos = env.step(actions)
#         current_rewards += rewards
#         current_lengths += 1
        
#         for i in range(n_envs):
#             # unpack values so that the callback can access the local variables
#             reward = rewards[i]
#             done = dones[i]
#             info = infos[i]
#             episode_starts[i] = done

#             if callback is not None:
#                 callback(locals(), globals())
                
#             if dones[i]:
#                 if is_monitor_wrapped:
#                     # Atari wrapper can send a "done" signal when
#                     # the agent loses a life, but it does not correspond
#                     # to the true end of episode
#                     if "episode" in info.keys():
#                         # Use rewards from Monitor wrapper
#                         episode_rewards.append(info["episode"]["r"])
#                         episode_lengths.append(info["episode"]["l"])
#                         episode_log_probs.append(current_log_probs[i])
#                 else:
#                     episode_rewards.append(current_rewards[i])
#                     episode_lengths.append(current_lengths[i])
#                     episode_log_probs.append(current_log_probs[i])
                
#                 current_rewards[i] = 0
#                 current_lengths[i] = 0
#                 current_log_probs[i] = 0
#                 current_query[i] += 1
                
#                 # Move to the next query if available
#                 if current_query[i] < len(data):
#                     observations, _ = env.reset_from_query(data[current_query[i]], labels[current_query[i]], consult_janus=consult_janus)
        
#         # If not done, continue with the updated observations
#         if not np.all(dones):
#             observations = new_observations

#     mean_reward = np.mean(episode_rewards)
#     std_reward = np.std(episode_rewards)
    
#     if reward_threshold is not None:
#         assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    
#     if return_dict:
#         results = {
#             'mean_reward': float(mean_reward),
#             'std_reward': float(std_reward),
#             'mean_length': float(np.mean(episode_lengths)),
#             'std_length': float(np.std(episode_lengths)),
#             'mean_log_prob' : float(np.mean(episode_log_probs)),
#             'std_log_prob' : float(np.std(episode_log_probs)),
#         }
#         return results
    
#     return episode_rewards, episode_lengths, episode_log_probs










# def evaluate_corruptions(
#     model: "type_aliases.PolicyPredictor",
#     env: Union[gym.Env, VecEnv],
#     data: List[Any],
#     corruptions: Optional[Dict[Any, List[Any]]] = None,
#     corruption_mode: str = 'static',
#     n_corruptions: int = -1,
#     deterministic: bool = True,
#     verbose: int = 0,
#     consult_janus: bool = False,
# ) -> Dict[str, Any]:
#     """
#     Evaluates model performance by comparing original queries against their corrupted versions.
#     Optimized for vectorized environments, processing queries in parallel where possible.
    
#     :param model: The RL model to evaluate
#     :param env: The environment or vectorized environment
#     :param data: List of original queries to evaluate
#     :param corruptions: Dictionary mapping original queries to their corrupted versions
#     :param corruption_mode: 'static' (use provided corruptions) or 'dynamic' (generate on the fly)
#     :param n_corruptions: Number of corruptions to use per query
#     :param deterministic: Whether to use deterministic policy
#     :param verbose: Verbosity level
#     :param consult_janus: Whether to consult Janus for query processing
#     :return: Dictionary with evaluation metrics
#     """
#     mrr_list = []
#     rewards_list_pos, episode_len_list_pos, log_probs_list_pos = [], [], []
#     rewards_list_neg, episode_len_list_neg, log_probs_list_neg = [], [], []
    
#     # Determine if we're using a vectorized environment
#     is_vec_env = isinstance(env, VecEnv)
#     num_envs = env.num_envs if is_vec_env else 1
    
#     if verbose >= 1:
#         print(f"Evaluating {len(data)} queries with {n_corruptions} corruptions each")
#         print(f"Using {'vectorized' if is_vec_env else 'single'} environment with {num_envs} envs")
    
#     # Process queries in batches for efficient parallel processing
#     for batch_start in range(0, len(data), num_envs):
#         batch_end = min(batch_start + num_envs, len(data))
#         batch_size = batch_end - batch_start
#         batch_queries = data[batch_start:batch_end]
        
#         if verbose >= 1:
#             print(f"\nProcessing batch {batch_start//num_envs + 1}/{(len(data) + num_envs - 1)//num_envs}")
#         else:
#             print(f"\rProcessing {batch_end}/{len(data)}", end='', flush=True)
        
#         # For each query in the batch
#         for env_idx, query in enumerate(batch_queries):
#             # Select the appropriate environment
#             query_env = env.envs[env_idx] if is_vec_env else env
            
#             # Get corruptions based on the mode
#             if corruption_mode == 'static':
#                 if corruptions is None:
#                     raise ValueError("Static corruption mode requires a corruptions dictionary")
#                 query_corruptions = corruptions[query][:n_corruptions]
#             elif corruption_mode == 'dynamic':
#                 if hasattr(query_env, 'get_negatives'):
#                     query_corruptions = query_env.get_negatives(query, all_negatives=True)[:n_corruptions]
#                 else:
#                     raise ValueError("Dynamic corruption mode requires environment with get_negatives method")
#             else:
#                 raise ValueError(f"Unknown corruption mode: {corruption_mode}")
            
#             # Create evaluation data for this query and its corruptions
#             eval_data = [query] + query_corruptions
#             eval_labels = [1] + [0] * len(query_corruptions)
            
#             # Process each item (query + corruptions) sequentially
#             all_rewards, all_episodes, all_log_probs = [], [], []
            
#             for i, (item, label) in enumerate(zip(eval_data, eval_labels)):
#                 # Reset environment with current item
#                 if hasattr(query_env, 'reset_from_query'):
#                     obs, _ = query_env.reset_from_query(item, label, consult_janus=consult_janus)
#                 else:
#                     obs = query_env.reset()
                
#                 # Track metrics for current item
#                 done = False
#                 item_reward = 0
#                 item_length = 0
#                 item_log_prob = 0
                
#                 # Run episode
#                 while not done:
#                     # Get action and log probability
#                     obs_tensor = obs_as_tensor(obs, model.device)
#                     action, values, log_prob = model.policy(obs_tensor, deterministic=deterministic)
                    
#                     if verbose >= 2:
#                         prob = np.exp(log_prob.detach().cpu().numpy())
#                         print(f'action: {action}, values: {values}, log_prob: {log_prob}, prob: {prob}')
                    
#                     log_prob = log_prob.detach().cpu().numpy().item()
#                     item_log_prob += log_prob
                    
#                     # Take step in environment
#                     obs, reward, done, info = query_env.step(action[0])[:4]  # Handles both old and new gym API
                    
#                     item_reward += reward
#                     item_length += 1
                
#                 # Record results
#                 all_rewards.append(item_reward)
#                 all_episodes.append(item_length)
#                 all_log_probs.append(item_log_prob)
            
#             # Process results for this query and its corruptions
#             rewards_list_pos.append(all_rewards[0])
#             episode_len_list_pos.append(all_episodes[0])
#             log_probs_list_pos.append(all_log_probs[0])
            
#             if len(all_rewards) > 1:  # We have corruptions
#                 # Calculate rank based on log probabilities (higher is better)
#                 rank_by_logprob = np.argsort(all_log_probs)[::-1].tolist().index(0) + 1
#                 mrr = 1.0 / rank_by_logprob
#                 mrr_list.append(mrr)
                
#                 # Store negative examples metrics
#                 rewards_list_neg.extend(all_rewards[1:])
#                 episode_len_list_neg.extend(all_episodes[1:])
#                 log_probs_list_neg.extend(all_log_probs[1:])
    
#     # Calculate aggregate metrics
#     scores = log_probs_list_pos + log_probs_list_neg
#     labels = [1] * len(log_probs_list_pos) + [0] * len(log_probs_list_neg)
#     auc_pr = average_precision_score(labels, scores) if len(labels) > 1 else 0
    
#     # Return comprehensive evaluation metrics
#     return {
#         'pos_queries': len(rewards_list_pos),
#         'neg_queries': len(rewards_list_neg),
#         'ratio_pos_queries': round(len(rewards_list_pos)/(len(rewards_list_pos)+len(rewards_list_neg)), 2),
#         'mrr_mean': float(np.mean(mrr_list)) if mrr_list else 0,
#         'mrr_std': float(np.std(mrr_list)) if mrr_list else 0,
        
#         'rewards_pos_mean': float(np.mean(rewards_list_pos)),
#         'rewards_pos_std': float(np.std(rewards_list_pos)),
#         'rewards_neg_mean': float(np.mean(rewards_list_neg)) if rewards_list_neg else 0,
#         'rewards_neg_std': float(np.std(rewards_list_neg)) if rewards_list_neg else 0,
#         'rewards_mean': float(np.mean(rewards_list_pos + rewards_list_neg)),
#         'rewards_std': float(np.std(rewards_list_pos + rewards_list_neg)),
        
#         'episode_len_pos_mean': float(np.mean(episode_len_list_pos)),
#         'episode_len_pos_std': float(np.std(episode_len_list_pos)),
#         'episode_len_neg_mean': float(np.mean(episode_len_list_neg)) if episode_len_list_neg else 0,
#         'episode_len_neg_std': float(np.std(episode_len_list_neg)) if episode_len_list_neg else 0,
#         'episode_len_mean': float(np.mean(episode_len_list_pos + episode_len_list_neg)),
#         'episode_len_std': float(np.std(episode_len_list_pos + episode_len_list_neg)),
        
#         'log_probs_pos_mean': float(np.mean(log_probs_list_pos)),
#         'log_probs_pos_std': float(np.std(log_probs_list_pos)),
#         'log_probs_neg_mean': float(np.mean(log_probs_list_neg)) if log_probs_list_neg else 0,
#         'log_probs_neg_std': float(np.std(log_probs_list_neg)) if log_probs_list_neg else 0,
#         'log_probs_mean': float(np.mean(log_probs_list_pos + log_probs_list_neg)),
#         'log_probs_std': float(np.std(log_probs_list_pos + log_probs_list_neg)),
        
#         'auc_pr': float(auc_pr)
#     }