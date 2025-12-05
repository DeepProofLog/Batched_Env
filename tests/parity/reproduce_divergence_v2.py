
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

def run_rng_test():
    cfg = create_default_config()
    cfg.n_envs = 4
    cfg.n_steps = 32
    cfg.n_epochs = 10 
    cfg.dataset = "countries_s3"
    cfg.verbose = True
    
    print(f"Running RNG Drift Test with n_epochs={cfg.n_epochs}")
    
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
    
    seed_all(cfg.seed)
    # Iter 1 Rollout
    sb3_ppo.collect_rollouts(sb3_ppo.env, callback, sb3_ppo.rollout_buffer, cfg.n_steps)
    # Iter 1 Train
    sb3_ppo.train()
    # Iter 2 Rollout (should differ if RNG drifted)
    sb3_ppo.collect_rollouts(sb3_ppo.env, callback, sb3_ppo.rollout_buffer, cfg.n_steps)
    sb3_actions_2 = sb3_ppo.rollout_buffer.actions.copy()
    
    # === run Tensor Sequence ===
    print("\nRunning Tensor Sequence...")
    tensor_ppo, tensor_env, tensor_im, engine = get_tensor()
    
    # Tensor State
    curr_obs_tensor = tensor_ppo.env.reset()
    episode_starts_tensor = torch.ones(tensor_ppo.n_envs, dtype=torch.float32, device=tensor_ppo.device)
    current_episode_reward = torch.zeros(tensor_ppo.n_envs, dtype=torch.float32, device=tensor_ppo.device)
    current_episode_length = torch.zeros(tensor_ppo.n_envs, dtype=torch.long, device=tensor_ppo.device)
    episode_rewards = []
    episode_lengths = []
    
    seed_all(cfg.seed)
    # Iter 1 Rollout
    res = tensor_ppo.collect_rollouts(
        curr_obs_tensor, episode_starts_tensor, current_episode_reward, current_episode_length, 
        episode_rewards, episode_lengths, iteration=1, return_traces=False
    )
    curr_obs_tensor, episode_starts_tensor, current_episode_reward, current_episode_length = res[:4]
    
    # Iter 1 Train
    tensor_ppo.train()
    
    # Iter 2 Rollout
    res = tensor_ppo.collect_rollouts(
        curr_obs_tensor, episode_starts_tensor, current_episode_reward, current_episode_length, 
        episode_rewards, episode_lengths, iteration=2, return_traces=False
    )
    tensor_actions_2 = tensor_ppo.rollout_buffer.actions.cpu().numpy()
    
    # === Compare ===
    # SB3 actions shape: (n_steps, n_envs, ...) -> (32, 4, 1) usually
    # Tensor actions shape: (n_steps, n_envs) -> (32, 4)
    
    sb3_flat = sb3_actions_2.flatten()
    tensor_flat = tensor_actions_2.flatten()
    
    if np.array_equal(sb3_flat, tensor_flat):
        print("\nRNG Drift Test: PASSED (Rollout 2 Actions Match)")
    else:
        print("\nRNG Drift Test: FAILED (Rollout 2 Actions Mismatch)")
        print(f"SB3 first 10: {sb3_flat[:10]}")
        print(f"Tensor first 10: {tensor_flat[:10]}")
        
    # Also compare weights just in case
    match, _ = compare_weights(sb3_ppo, tensor_ppo.policy, verbose=True)
    if not match:
        print("Weights also mismatch at end of Iter 1 + Rollout 2.")

if __name__ == "__main__":
    run_rng_test()
