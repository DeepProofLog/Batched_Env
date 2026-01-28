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

import argparse
import cProfile
import pstats
import io
from datetime import datetime
from time import time

import torch

from kge_experiments.mcts.config import MCTSConfig
from kge_experiments.mcts.trainer import MuZeroTrainer
from kge_experiments.builder import create_env, create_policy, KGEConfig


def main():
    parser = argparse.ArgumentParser(description='Profile MCTS training')
    parser.add_argument('--dataset', type=str, default='family')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--n-steps', type=int, default=128)
    parser.add_argument('--n-iterations', type=int, default=5)
    parser.add_argument('--num-simulations', type=int, default=25)
    parser.add_argument('--compile', default=True, type=lambda x: x.lower() != 'false')
    parser.add_argument('--cpu-profile', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')

    print(f"MCTS Training Profile")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    B = args.batch_size
    n_sims = args.num_simulations
    n_steps = args.n_steps
    n_iterations = args.n_iterations
    total_timesteps = n_steps * n_iterations
    dataset = args.dataset
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

    print(f"\nConfiguration:")
    print(f"  Batch size (n_envs): {B}")
    print(f"  Num simulations: {n_sims}")
    print(f"  Steps per iteration: {n_steps}")
    print(f"  N iterations: {n_iterations}")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Compile: {args.compile}")

    # Build environment and policy
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

    max_actions = env.padding_states
    mcts_config = MCTSConfig(
        num_simulations=n_sims,
        max_episode_steps=20,
        mcts_batch_size=B,
        max_actions=max_actions,
        use_batched_mcts=True,
        device=str(device),
        learning_rate=3e-4,
        discount=0.99,
        compile=args.compile,
    )

    trainer = MuZeroTrainer(
        config=mcts_config,
        env=env,
        policy=policy,
    )

    # Set env to train mode
    env.train()

    # Warmup
    print("\nRunning warmup...")
    trainer.collect_episodes_batched(num_steps=B * 2, add_noise=True)
    torch.cuda.synchronize()

    # Profile training loop
    print(f"\nProfiling training: {n_iterations} iterations, {total_timesteps} total steps...")

    if args.cpu_profile:
        profiler = cProfile.Profile()
        profiler.enable()

    torch.cuda.synchronize()
    start = time()

    actual_steps = 0
    for iteration in range(n_iterations):
        collect_stats = trainer.collect_episodes_batched(
            num_steps=n_steps,
            add_noise=True,
        )
        actual_steps += collect_stats.get("steps_collected", n_steps)

        if len(trainer.replay_buffer) >= trainer.config.min_buffer_size:
            trainer.train_step()

    torch.cuda.synchronize()
    elapsed = time() - start

    if args.cpu_profile:
        profiler.disable()

    # Calculate metrics
    steps_per_sec = actual_steps / elapsed
    env_steps_per_sec = steps_per_sec * B

    print(f"\n{'='*60}")
    print("TIMING SUMMARY")
    print(f"{'='*60}")
    print(f"Runtime:       {elapsed:.3f}s")
    print(f"Steps:         {actual_steps}")
    print(f"Steps/sec:     {steps_per_sec:.1f}")
    print(f"Env-steps/sec: {env_steps_per_sec:.0f}")
    print(f"ms/step:       {elapsed/actual_steps*1000:.2f}")

    if args.cpu_profile:
        print(f"\n{'='*60}")
        print("CPU BOTTLENECKS (top 20 by cumulative time)")
        print(f"{'='*60}")
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.strip_dirs().sort_stats('cumulative').print_stats(20)
        print(s.getvalue())


if __name__ == "__main__":
    main()
