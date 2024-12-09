
from typing import Tuple
import numpy as np
import gymnasium as gym
from utils import Term, print_state_transition
from stable_baselines3.common.on_policy_algorithm import obs_as_tensor
from stable_baselines3 import PPO


def eval_test(  data: list[Term],
                labels: list[int],
                env: gym.Env,
                model: PPO,
                deterministic: bool = True,
                verbose:int=0) -> Tuple[list[float], list[int], list[float]]:

    rewards_list,episode_len_list, log_probs= [], [], []
    next_query, trajectory_reward, episode_len, cum_log_prob = 0, 0, 0, 0

    obs, _ = env.reset_from_query(data[next_query],labels[next_query])
    print_state_transition(env.tensordict['state'], env.tensordict['derived_states'],env.tensordict['reward'], env.tensordict['done']) if verbose >=1 else None
    while next_query < len(data)-1:
        # action, _states = model.predict(obs, deterministic=deterministic)
        obs_tensor = obs_as_tensor(obs, model.device)
        action, values, log_prob = model.policy(obs_tensor, deterministic=deterministic)
        cum_log_prob += log_prob.detach().numpy()
        
        obs, rewards, dones, truncated, info = env.step(action)
        print_state_transition(env.tensordict['state'], env.tensordict['derived_states'],env.tensordict['reward'], env.tensordict['done'], action=env.tensordict['action'],truncated=truncated) if verbose >=1 else None
        trajectory_reward, episode_len, log_prob = trajectory_reward + rewards, episode_len + 1, log_prob + log_prob

        if dones:
            rewards_list.append(trajectory_reward)
            episode_len_list.append(episode_len)
            log_probs.append(cum_log_prob)

            # print(' done,truncated,rewards',dones,truncated,rewards)
            next_query += 1
            obs, _ = env.reset_from_query(data[next_query],labels[next_query])
            # print('\nquery',next_query, 'with label',labels[next_query])
            trajectory_reward, episode_len, cum_log_prob = 0, 0, 0
            print_state_transition(env.tensordict['state'], env.tensordict['derived_states'],env.tensordict['reward'], env.tensordict['done']) if verbose >=1 else None
    return rewards_list, episode_len_list, log_probs
    

def eval_test_corruptions(  data: list[Term],
                            labels: list[int],
                            env: gym.Env,
                            model: PPO,
                            deterministic: bool = True,
                            verbose:int=0) -> Tuple[list[float], list[int]]:
    '''
    For every query in data, evaluate the model on that query and all its corruptions (based on the logprobs) and rank the query and its corruptions (MRR)
    By now we are returning, for postive queries, the reward (how many are proven, with their ratio), and their prob distribution. Same for negatives
    '''
    # 1. split the data in positives and negatives
    (data_pos, labels_pos) = zip(*[(data[i],labels[i]) for i in range(len(data)) if labels[i] == 1])
    (data_neg, labels_neg) = zip(*[(data[i],labels[i]) for i in range(len(data)) if labels[i] == 0])
    
    # 2. evaluate the model on the positive/negative queries
    rewards_list_pos, episode_len_list_pos, log_probs_pos = eval_test(data_pos, labels_pos, env, model, deterministic, verbose)
    rewards_list_neg, episode_len_list_neg, log_probs_neg = eval_test(data_neg, labels_neg, env, model, deterministic, verbose)
    
    mean_rwd_pos, std_rwd_pos = np.round(np.mean(rewards_list_pos),3), np.round(np.std(rewards_list_pos),3)
    mean_rwd_neg, std_rwd_neg = np.round(np.mean(rewards_list_neg),3), np.round(np.std(rewards_list_neg),3)
    mean_len_pos, std_len_pos = np.round(np.mean(episode_len_list_pos),3), np.round(np.std(episode_len_list_pos),3)
    mean_len_neg, std_len_neg = np.round(np.mean(episode_len_list_neg),3), np.round(np.std(episode_len_list_neg),3)
    mean_log_probs_pos, std_log_probs_pos = np.round(np.mean(log_probs_pos),3), np.round(np.std(log_probs_pos),3) 
    mean_log_probs_neg, std_log_probs_neg = np.round(np.mean(log_probs_neg),3), np.round(np.std(log_probs_neg),3)

    print('\nPositive queries:',len(rewards_list_pos), 'Negative queries:',len(rewards_list_neg))
    print('Positive queries rewards:',mean_rwd_pos,'+/-', std_rwd_pos)
    print('Negative queries rewards:',mean_rwd_neg,'+/-', std_rwd_neg)
    print('Positive queries episode len:',mean_len_pos, '+/-', std_len_pos)
    print('Negative queries episode len:',mean_len_neg, '+/-', std_len_neg)
    print('Positive queries log probs:',mean_log_probs_pos, '+/-', std_log_probs_pos)
    print('Negative queries log probs:',mean_log_probs_neg, '+/-', std_log_probs_neg)
    
    #plot the log probs points. In the title, round the numbers
    import matplotlib.pyplot as plt
    plt.title("Log probs: \npositive {:.2f} +/- {:.2f} \nnegative {:.2f} +/- {:.2f}".format(mean_log_probs_pos, std_log_probs_pos, mean_log_probs_neg, std_log_probs_neg))
    plt.scatter(range(len(log_probs_pos)), log_probs_pos, color='blue')
    plt.scatter(range(len(log_probs_neg)), log_probs_neg, color='red')
    plt.xlabel('query')
    plt.ylabel('log probs')
    # positive in blue, negative in red
    plt.legend(['positive','negative'])
    plt.show()

    return rewards_list_pos + rewards_list_neg, episode_len_list_pos + episode_len_list_neg,
