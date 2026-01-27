"""
Profile MCTS training using batched collect_episodes and train_step.

Usage:
    python kge_experiments/mcts/tests/profile_train.py
"""

import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'kge_experiments'))

import cProfile
import pstats
import io
from time import time

import torch

from kge_experiments.mcts.config import MCTSConfig
from kge_experiments.mcts.trainer import MuZeroTrainer
from kge_experiments.builder import create_env, create_policy, KGEConfig


def main():
    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # Configuration
    B = 100  # Batch size (environments in parallel)
    n_sims = 25
    total_timesteps = 500  # Total training steps
    n_iterations = 5
    dataset = 'family'
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

    print(f"\nConfiguration:")
    print(f"  Batch size: {B}")
    print(f"  Num simulations: {n_sims}")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  N iterations: {n_iterations}")

    # Build environment and policy using KGEConfig
    print("\nBuilding environment and policy...")
    kge_config = KGEConfig(
        dataset=dataset,
        data_path=data_path,
        device=str(device),
        n_envs=B,
        learning_rate=3e-4,
        gamma=0.99,
    )
    env = create_env(kge_config)
    policy = create_policy(kge_config, env)

    # MCTS-specific config - get max_actions from env
    max_actions = env.padding_states  # Actual action space size
    mcts_config = MCTSConfig(
        num_simulations=n_sims,
        max_episode_steps=20,
        mcts_batch_size=B,
        max_actions=max_actions,
        use_batched_mcts=True,
        device=str(device),
        learning_rate=3e-4,
        discount=0.99,
        compile=False,  # Disable compilation for now (TensorDict key access issue)
    )

    # Create trainer
    trainer = MuZeroTrainer(
        config=mcts_config,
        env=env,
        policy=policy,
        device=device,
    )

    # Set env to train mode
    env.train()

    # Warmup
    print("\nRunning warmup...")
    trainer.collect_episodes_batched(num_steps=B * 2, add_noise=True)
    torch.cuda.synchronize()

    # Profile training loop
    print(f"\nProfiling training: {n_iterations} iterations, {total_timesteps} total steps...")

    profiler = cProfile.Profile()
    profiler.enable()

    torch.cuda.synchronize()
    start = time()

    # Manual training loop using batched methods
    steps_per_iter = total_timesteps // n_iterations
    actual_steps = 0

    for iteration in range(n_iterations):
        # Collect episodes (batched)
        collect_stats = trainer.collect_episodes_batched(
            num_steps=steps_per_iter,
            add_noise=True,
        )
        actual_steps += collect_stats.get("steps_collected", steps_per_iter)

        # Train step
        if len(trainer.replay_buffer) >= trainer.config.min_buffer_size:
            train_metrics = trainer.train_step()

    torch.cuda.synchronize()
    elapsed = time() - start

    profiler.disable()

    # Calculate metrics
    steps_per_sec = actual_steps / elapsed

    print(f"\n{'='*60}")
    print("TIMING SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {elapsed:.3f}s for {actual_steps} steps ({n_iterations} iterations)")
    print(f"Time per step: {elapsed/actual_steps*1000:.2f} ms")
    print(f"Steps/sec: {steps_per_sec:.1f}")
    print(f"Env-steps/sec (B={B}): {steps_per_sec * B:.0f}")

    print(f"\n{'='*60}")
    print("CPU BOTTLENECKS (top 20 by cumulative time)")
    print(f"{'='*60}")
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())


if __name__ == "__main__":
    main()
