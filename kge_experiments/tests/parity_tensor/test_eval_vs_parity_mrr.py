"""Test MRR parity between evaluate() and evaluate_parity()."""
import torch
import pytest
import numpy as np
from types import SimpleNamespace
from test_compiled_eval import (
    setup_shared_components,
    create_default_config,
    create_optimized_env,
    compute_optimal_batch_size
)
from ppo import PPO as PPOOptimized
from tests.test_utils.parity_utils import evaluate_parity


def test_evaluate_vs_evaluate_parity_mrr():
    """Verify evaluate() and evaluate_parity() produce similar MRR values (within 10%)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create config
    config = create_default_config()
    config.dataset = 'family'
    config.n_queries = 30
    config.n_corruptions = 10
    config.corruption_modes = ['head', 'tail']
    config.compile = False
    config.device = device

    # Compute batch size
    effective_chunk_queries = min(int(config.chunk_queries), int(config.n_queries))
    batch_size = compute_optimal_batch_size(
        chunk_queries=effective_chunk_queries,
        n_corruptions=config.n_corruptions,
    )
    config.batch_size_env = batch_size
    config.n_envs = batch_size
    print(f"Using batch_size_env: {batch_size}")

    # Setup components
    components = setup_shared_components(config, device)

    # Create environment (without compile - we'll compile separately)
    env = create_optimized_env(components, config, device)

    sampler = components['sampler']
    policy = components['policy_opt']
    queries = components['test_queries_unpadded'][:config.n_queries].to(device)

    # Create PPO instance
    ppo = PPOOptimized(policy, env, config)

    # Compile for evaluate()
    env.compile(mode='reduce-overhead', fullgraph=True)

    # Warmup both
    print("\nWarming up evaluate()...")
    _ = ppo.evaluate(queries, sampler, n_corruptions=config.n_corruptions, corruption_modes=tuple(config.corruption_modes))
    print("Warming up evaluate_parity()...")
    _ = evaluate_parity(ppo, queries, sampler, n_corruptions=config.n_corruptions, corruption_modes=tuple(config.corruption_modes), compile_mode='default')

    # Reset RNG for both
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    sampler.rng = np.random.RandomState(seed)

    # Run evaluate()
    print("\nRunning evaluate()...")
    res_evaluate = ppo.evaluate(
        queries, sampler,
        n_corruptions=config.n_corruptions,
        corruption_modes=tuple(config.corruption_modes),
    )

    # Reset RNG for parity
    torch.manual_seed(seed)
    np.random.seed(seed)
    sampler.rng = np.random.RandomState(seed)

    # Run evaluate_parity()
    print("Running evaluate_parity()...")
    res_parity = evaluate_parity(
        ppo,
        queries, sampler,
        n_corruptions=config.n_corruptions,
        corruption_modes=tuple(config.corruption_modes),
        compile_mode='default'
    )

    print(f"\nevaluate()        MRR: {res_evaluate['MRR']:.6f}")
    print(f"evaluate_parity() MRR: {res_parity['MRR']:.6f}")
    diff = abs(res_evaluate['MRR'] - res_parity['MRR'])
    print(f"Difference:           {diff:.6f}")
    print(f"Within 10% tolerance: {diff <= 0.10}")

    assert diff <= 0.10, f"MRR difference {diff:.4f} exceeds 10% tolerance"
    print("\n✓ MRR VALUES ARE SIMILAR!")


if __name__ == "__main__":
    test_evaluate_vs_evaluate_parity_mrr()
