from typing import Tuple, Optional
import numpy as np
import gymnasium as gym
from utils import Term, print_state_transition
from stable_baselines3.common.on_policy_algorithm import obs_as_tensor
from stable_baselines3 import PPO
from sklearn.metrics import average_precision_score



def eval(  data: list[Term],
                labels: list[int],
                env: gym.Env,
                model: PPO,
                deterministic: bool = True,
                verbose:int=0,
                return_dict:bool=True,
                consult_janus:bool=False) -> Tuple[list[float], list[int], list[float]]:
    """
    Evaluate the model on a set of queries.
    Works with both single and vectorized environments.
    """
    rewards_list, episode_len_list, log_probs = [], [], []
    next_query, trajectory_reward, episode_len, cum_log_prob = 0, 0, 0, 0

    obs, _ = env.reset_from_query(data[next_query],labels[next_query],consult_janus=consult_janus)
    print_state_transition(env.tensordict['state'], env.tensordict['derived_states'],env.tensordict['reward'], env.tensordict['done']) if verbose >=1 else None
    while next_query < len(data):
        if next_query % 10 == 0: print(f'\rCorruption {next_query}/{len(data)}', end='', flush=True)
        # action, _states = model.predict(obs, deterministic=deterministic) # it is the same as model.policy, but we get more info
        obs_tensor = obs_as_tensor(obs, model.device)
        action, values, log_prob = model.policy(obs_tensor, deterministic=deterministic)
        print(f'action:{action}, values:{values}, log_prob:{log_prob}, prob:{np.exp(log_prob.detach().cpu().numpy().item())}') if verbose >=1 else None
        log_prob = log_prob.detach().cpu().numpy().item()
        cum_log_prob += log_prob
        
        # Take step in environment
        obs, rewards, dones, truncated, info = env.step(action[0])
        print_state_transition(env.tensordict['state'], env.tensordict['derived_states'],env.tensordict['reward'], env.tensordict['done'], action=env.tensordict['action'],truncated=truncated) if verbose >=1 else None
        trajectory_reward, episode_len = trajectory_reward + rewards, episode_len + 1

        if dones:
            if rewards == 0:
                cum_log_prob -= 100
                
            rewards_list.append(trajectory_reward)
            episode_len_list.append(episode_len)
            log_probs.append(cum_log_prob)
            print(f'reward {trajectory_reward}, episode len {episode_len}, cum log prob {cum_log_prob}') if verbose >=1 else None
            print(' done,truncated,rewards', dones, truncated, rewards) if verbose >=1 else None
            next_query += 1
            if next_query < len(data):
                obs, _ = env.reset_from_query(data[next_query], labels[next_query], consult_janus=consult_janus)
                print('\nquery', next_query, 'with label', labels[next_query]) if verbose >=1 else None
                trajectory_reward, episode_len, cum_log_prob = 0, 0, 0
                print_state_transition(env.tensordict['state'], env.tensordict['derived_states'],env.tensordict['reward'], env.tensordict['done']) if verbose >=1 else None
    
    if return_dict:
        return {
            'rewards_mean': np.mean(rewards_list), 
            'rewards_std': np.std(rewards_list), 
            'episode_len_mean': np.mean(episode_len_list), 
            'episode_len_std': np.std(episode_len_list), 
            'log_probs_mean': np.mean(log_probs), 
            'log_probs_std': np.std(log_probs)
        }

    return rewards_list, episode_len_list, log_probs



def eval_corruptions(  
                        data: list[Term],
                        env: gym.Env,
                        model: PPO,
                        corruptions: Optional[dict[Term, list[Term]]] = None,
                        deterministic: bool = True,
                        verbose:int=0,
                        consult_janus:bool=False,
                        corruption_mode: str = 'static',
                        n_corruptions: int = 10,
                        n_eval_envs: int = 1) -> Tuple[list[float], list[int], list[float]]:
    '''
    For every positive query, get its corruptions, evaluate the model on the query and all its corruptions (based on the logprobs) and rank the query and its corruptions (MRR)
    Now supports multiple environments for parallel evaluation.
    '''
    mrr_list, rewards_list_pos, episode_len_list_pos, log_probs_list_pos, rewards_list_neg, episode_len_list_neg, log_probs_list_neg = [], [], [], [], [], [], []
    
    # If using a vectorized environment, we need to process queries in batches
    is_vec_env = hasattr(env, 'num_envs')
    num_envs = env.num_envs if is_vec_env else 1
    
    # Process queries in batches if using vectorized environment
    for i in range(0, len(data), num_envs):
        batch_queries = data[i:i+num_envs]
        print(f'\nBatch {i+1}-{min(i+num_envs, len(data))}/{len(data)}')
        # print(f'\rQueries {i+1}-{min(i+num_envs, len(data))}/{len(data)}', end='', flush=True)
        
        # Process each query in the batch
        batch_results = []
        for j, query in enumerate(batch_queries):
            # If using single env, evaluate directly
            if not is_vec_env:
                if corruption_mode == 'static':
                    corruptions_query = corruptions[query][:n_corruptions]
                elif corruption_mode == 'dynamic':
                    corruptions_query = env.get_negatives(query, all_negatives=True)[:n_corruptions]
                    
                data_query = [query] + corruptions_query
                labels_query = [1] + [0 for _ in range(len(corruptions_query))]
                
                rewards, episode_len, log_probs = eval(data_query, labels_query, env, model, 
                                                     deterministic, verbose, return_dict=False, 
                                                     consult_janus=consult_janus)
                batch_results.append((rewards, episode_len, log_probs))
            else:
                # If using multiple envs, use env_idx to set which environment to use
                env_idx = j % num_envs
                # For vectorized environments, we need to select the appropriate environment
                # We extract the specific environment from the VecEnv
                sub_env = env.envs[env_idx]
                
                if corruption_mode == 'static':
                    corruptions_query = corruptions[query][:n_corruptions]
                elif corruption_mode == 'dynamic':
                    corruptions_query = sub_env.get_negatives(query, all_negatives=True)[:n_corruptions]
                    
                data_query = [query] + corruptions_query
                labels_query = [1] + [0 for _ in range(len(corruptions_query))]

                rewards, episode_len, log_probs = eval(data_query, labels_query, sub_env, model, 
                                                     deterministic, verbose, return_dict=False, 
                                                     consult_janus=consult_janus)
                batch_results.append((rewards, episode_len, log_probs))
        
        # Process results from the batch
        for rewards, episode_len, log_probs in batch_results:
            rewards_list_pos.append(rewards[0])
            episode_len_list_pos.append(episode_len[0])
            log_probs_list_pos.append(log_probs[0])

            if len(rewards) > 1:  # If we have corruptions
                rank = np.argsort(log_probs)[::-1].tolist().index(0)
                mrr = 1/(rank+1)
                mrr_list.append(mrr)

                rewards_list_neg.extend(rewards[1:])
                episode_len_list_neg.extend(episode_len[1:])
                log_probs_list_neg.extend(log_probs[1:])

    scores = log_probs_list_pos + log_probs_list_neg
    labels = [1] * len(log_probs_list_pos) + [0] * len(log_probs_list_neg)
    auc_pr = average_precision_score(labels, scores)

    info = {'pos_queries':len(rewards_list_pos), 'neg_queries':len(rewards_list_neg), 'ratio_pos_queries':round(len(rewards_list_pos)/(len(rewards_list_pos)+len(rewards_list_neg)),2),
            'mrr_mean':np.mean(mrr_list), 'mrr_std':np.std(mrr_list),
            'rewards_pos_mean':np.mean(rewards_list_pos), 'rewards_pos_std':np.std(rewards_list_pos),
            'rewards_neg_mean':np.mean(rewards_list_neg), 'rewards_neg_std':np.std(rewards_list_neg),
            'rewards_mean':np.mean(rewards_list_pos+rewards_list_neg), 'rewards_std':np.std(rewards_list_pos+rewards_list_neg),

            'episode_len_pos_mean':np.mean(episode_len_list_pos), 'episode_len_pos_std':np.std(episode_len_list_pos),
            'episode_len_neg_mean':np.mean(episode_len_list_neg), 'episode_len_neg_std':np.std(episode_len_list_neg),
            'episode_len_mean':np.mean(episode_len_list_pos+episode_len_list_neg), 'episode_len_std':np.std(episode_len_list_pos+episode_len_list_neg),

            'log_probs_pos_mean':np.mean(log_probs_list_pos), 'log_probs_pos_std':np.std(log_probs_list_pos),
            'log_probs_neg_mean':np.mean(log_probs_list_neg), 'log_probs_neg_std':np.std(log_probs_list_neg),
            'log_probs_mean':np.mean(log_probs_list_pos+log_probs_list_neg), 'log_probs_std':np.std(log_probs_list_pos+log_probs_list_neg),

            'auc_pr':auc_pr
            }
    return info




def eval_test_pos_neg(  data: list[Term],
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

    return rewards_list_pos + rewards_list_neg, episode_len_list_pos + episode_len_list_neg
