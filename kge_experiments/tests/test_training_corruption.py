"""
Debug test: Check if training corruption produces provable facts.

The mystery: Training shows 37% proven_neg but evaluation shows 0.5%.
This test investigates whether the training corruption is somehow different.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import TrainConfig
from builder import build_env


def test_training_corruption_provability():
    """Replicate training corruption and check provability."""
    print("=" * 60)
    print("Training Corruption Provability Test")
    print("=" * 60)

    config = TrainConfig(
        dataset='family',
        n_envs=64,
        corruption_scheme=('head', 'tail'),
        negative_ratio=1.0,
    )

    env = build_env(config)
    env.train()

    # Run several reset+step cycles to collect statistics
    n_episodes = 0
    proven_pos = 0
    proven_neg = 0
    total_pos = 0
    total_neg = 0

    obs, state = env.reset()

    for step in range(200):
        # Random actions
        actions = torch.randint(0, 10, (64,), device='cuda').clamp(max=state['num_valid_actions'] - 1)

        obs, state = env.step_and_reset(state, actions, env.query_pool, env.per_env_ptrs)

        # Check completed episodes
        done_mask = state['step_done']
        if done_mask.any():
            labels = state['step_labels'][done_mask]
            successes = state['step_successes'][done_mask].bool()

            for label, success in zip(labels.tolist(), successes.tolist()):
                if label == 1:
                    total_pos += 1
                    if success:
                        proven_pos += 1
                else:
                    total_neg += 1
                    if success:
                        proven_neg += 1
                n_episodes += 1

    print(f"\nAfter {n_episodes} episodes:")
    print(f"  Positives: {proven_pos}/{total_pos} proven ({100*proven_pos/max(total_pos,1):.1f}%)")
    print(f"  Negatives: {proven_neg}/{total_neg} proven ({100*proven_neg/max(total_neg,1):.1f}%)")

    print(f"\nThis matches training metrics if proven_neg is ~37%")


def compare_corruptions():
    """Compare corruption sources: training sampler vs evaluation sampler."""
    print("\n" + "=" * 60)
    print("Corruption Source Comparison")
    print("=" * 60)

    from nn.sampler import Sampler, SamplerConfig

    config = TrainConfig(
        dataset='family',
        n_envs=64,
    )

    # Build environment to get sampler
    env = build_env(config)

    # Get some training queries
    env.train()
    obs, state = env.reset()

    # Sample queries from training
    train_queries = state['original_queries'][:10]  # [10, 3]

    print(f"\nOriginal training queries:")
    for i, q in enumerate(train_queries):
        print(f"  [{i}] pred={q[0].item()}, head={q[1].item()}, tail={q[2].item()}")

    # Generate corruptions using the sampler
    print(f"\nGenerating corruptions with filter=True (default):")
    for mode in ['head', 'tail']:
        neg = env.sampler.corrupt(train_queries, num_negatives=5, mode=mode, filter=True)
        print(f"\n  Mode: {mode}")
        for i in range(min(3, train_queries.shape[0])):
            orig = train_queries[i]
            negs = neg[i]
            print(f"    Query {i}: pred={orig[0].item()}, head={orig[1].item()}, tail={orig[2].item()}")
            for j in range(min(3, negs.shape[0])):
                n = negs[j]
                if n.sum() > 0:  # Not padding
                    print(f"      -> neg {j}: pred={n[0].item()}, head={n[1].item()}, tail={n[2].item()}")


def test_env_corruption_labels():
    """Check if env correctly labels corrupted queries."""
    print("\n" + "=" * 60)
    print("Environment Label Verification")
    print("=" * 60)

    config = TrainConfig(
        dataset='family',
        n_envs=16,
        corruption_scheme=('head', 'tail'),
        negative_ratio=1.0,
    )

    env = build_env(config)
    env.train()

    obs, state = env.reset()

    # Check labels after reset (should alternate)
    labels = state['current_labels']
    queries = state['original_queries']

    print(f"\nAfter reset - labels and queries:")
    label_counts = {0: 0, 1: 0}
    for i in range(16):
        label = labels[i].item()
        label_counts[label] += 1
        q = queries[i]
        print(f"  [{i}] label={label}, pred={q[0].item()}, head={q[1].item()}, tail={q[2].item()}")

    print(f"\nLabel distribution: {label_counts}")
    print(f"With negative_ratio=1.0, expect ~50% positive, ~50% negative")


if __name__ == '__main__':
    test_env_corruption_labels()
    compare_corruptions()
    test_training_corruption_provability()
