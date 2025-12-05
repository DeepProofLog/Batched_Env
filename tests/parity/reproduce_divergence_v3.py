
import sys
import torch
import numpy as np
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from collections import deque

# Add root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.parity.test_learn_parity import (
    create_default_config, create_aligned_environments, create_sb3_ppo, create_tensor_ppo,
    ParityTestSeeder, seed_all, compare_buffers, compare_train_metrics, compare_train_traces,
    BufferSnapshot
)
from tests.parity.test_train_parity import compare_weights

def run_full_parity_check():
    cfg = create_default_config()
    cfg.n_envs = 4
    cfg.n_steps = 32
    cfg.n_epochs = 10
    cfg.batch_size = 4096  # Match failure case
    cfg.learning_rate = 5e-5
    cfg.ent_coef = 0.2
    cfg.dataset = "countries_s3"
    cfg.verbose = True
    
    print(f"Running Full Parity Check with n_epochs={cfg.n_epochs}")
    
    env_data = create_aligned_environments(cfg)
    
    # helper to make fresh components
    def get_sb3():
        return create_sb3_ppo(cfg, env_data['sb3'], env_data['queries'])
    
    def get_tensor():
        return create_tensor_ppo(cfg, env_data['tensor'], env_data['tensor_queries'])

    # === run SB3 Sequence ===
    print("\nRunning SB3 Sequence...")
    sb3_ppo, sb3_env, sb3_im = get_sb3()
    sb3_ppo._logger = configure(None, ["stdout"]) 
    sb3_ppo._last_obs = sb3_ppo.env.reset()
    sb3_ppo._last_episode_starts = np.ones((sb3_ppo.env.num_envs,), dtype=bool)
    sb3_ppo.ep_info_buffer = deque(maxlen=100)
    sb3_ppo.ep_success_buffer = deque(maxlen=100)
    
    class DummyCallback(BaseCallback):
        def _on_step(self): return True
    callback = DummyCallback()
    callback.init_callback(sb3_ppo)
    
    # Store results
    sb3_results = []
    
    seed_all(cfg.seed)
    
    # Update 1
    print(f"SB3 Pre-Update 1 RNG: Torch={torch.get_rng_state().float().sum().item()}, Numpy={np.random.get_state()[1][0]}")
    sb3_ppo.collect_rollouts(sb3_ppo.env, callback, sb3_ppo.rollout_buffer, cfg.n_steps)
    res1 = sb3_ppo.train(return_traces=True)
    print(f"SB3 Post-Update 1 RNG: Torch={torch.get_rng_state().float().sum().item()}, Numpy={np.random.get_state()[1][0]}")
    sb3_results.append({
        'actions': sb3_ppo.rollout_buffer.actions.copy(),
        'metrics': res1
    })
    
    # Update 2
    sb3_ppo.collect_rollouts(sb3_ppo.env, callback, sb3_ppo.rollout_buffer, cfg.n_steps)
    res2 = sb3_ppo.train(return_traces=True)
    sb3_results.append({
        'actions': sb3_ppo.rollout_buffer.actions.copy(),
        'metrics': res2
    })
    
    # === run Tensor Sequence ===
    print("\nRunning Tensor Sequence...")
    tensor_ppo, tensor_env, tensor_im, engine = get_tensor()
    
    curr_obs_tensor = tensor_ppo.env.reset()
    episode_starts_tensor = torch.ones(tensor_ppo.n_envs, dtype=torch.float32, device=tensor_ppo.device)
    current_episode_reward = torch.zeros(tensor_ppo.n_envs, dtype=torch.float32, device=tensor_ppo.device)
    current_episode_length = torch.zeros(tensor_ppo.n_envs, dtype=torch.long, device=tensor_ppo.device)
    episode_rewards = []
    episode_lengths = []
    
    tensor_results = []
    
    seed_all(cfg.seed)
    
    # Update 1
    print(f"Tensor Pre-Update 1 RNG: Torch={torch.get_rng_state().float().sum().item()}, Numpy={np.random.get_state()[1][0]}")
    res = tensor_ppo.collect_rollouts(
        curr_obs_tensor, episode_starts_tensor, current_episode_reward, current_episode_length, 
        episode_rewards, episode_lengths, iteration=1, return_traces=False
    )
    curr_obs_tensor, episode_starts_tensor, current_episode_reward, current_episode_length = res[:4]
    
    res1 = tensor_ppo.train(return_traces=True)
    print(f"Tensor Post-Update 1 RNG: Torch={torch.get_rng_state().float().sum().item()}, Numpy={np.random.get_state()[1][0]}")
    tensor_results.append({
        'actions': tensor_ppo.rollout_buffer.actions.cpu().numpy(),
        'metrics': res1
    })
    
    # Check Weights 1
    print("Comparing Weights after Update 1...")
    match, diffs = compare_weights(sb3_ppo, tensor_ppo.policy)
    if not match:
        print("WEIGHT MISMATCH after Update 1!")
        if diffs['mismatched_params']:
            print(f"  Max Diff: {diffs['mismatched_params'][0]}")
    else:
        print("Weights Match after Update 1.")
        
    # Update 2
    print("Tensor Update 2")
    res = tensor_ppo.collect_rollouts(
        curr_obs_tensor, episode_starts_tensor, current_episode_reward, current_episode_length, 
        episode_rewards, episode_lengths, iteration=2, return_traces=False
    )
    curr_obs_tensor, episode_starts_tensor, current_episode_reward, current_episode_length = res[:4]
    
    res2 = tensor_ppo.train(return_traces=True)
    tensor_results.append({
        'actions': tensor_ppo.rollout_buffer.actions.cpu().numpy(),
        'metrics': res2
    })
    
    # Check Rollout 2
    sb3_act_2 = sb3_results[1]['actions'].flatten()
    tensor_act_2 = tensor_results[1]['actions'].flatten()
    if np.array_equal(sb3_act_2, tensor_act_2):
        print("Rollout 2 Actions Match.")
    else:
        print("Rollout 2 Actions MISMATCH!")
    
    # Check Training Traces 2
    print("Comparing Training Traces for Update 2...")
    sb3_traces = sb3_results[1]['metrics']['traces']
    tensor_traces = tensor_results[1]['metrics']['traces']
    match_traces, n_mis, _ = compare_train_traces(sb3_traces, tensor_traces, verbose=True)
    if not match_traces:
        print(f"Training Traces Mismatch in Update 2 ({n_mis} batches)!")
    
    # Check Weights 2
    print("Comparing Weights after Update 2...")
    match, diffs = compare_weights(sb3_ppo, tensor_ppo.policy)
    if not match:
        print("WEIGHT MISMATCH after Update 2!")
        if diffs['mismatched_params']:
             print(f"  Max Diff: {diffs['mismatched_params'][0][1]}")
             print(f"  Param: {diffs['mismatched_params'][0][0]}")
    else:
        print("Weights Match after Update 2.")

if __name__ == "__main__":
    run_full_parity_check()
