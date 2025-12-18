"""
Runner - Generic interface for multiple experiment types.

Example:
    from runner import load_config, build_env, build_policy, get_algorithm
    
    config = load_config('kge')
    env = build_env(config)
    policy = build_policy(config)
    algorithm = get_algorithm(policy, env, config)
    algorithm.learn()
    results = algorithm.evaluate()
"""
import sys
from pathlib import Path

# Add paths
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "kge_experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "kge_experiments"))

# Re-export from registry
from registry import (
    load_config,
    build_env,
    build_policy,
    get_sampler,
    get_test_queries,
    get_algorithm,
)


__all__ = [
    'load_config',
    'build_env', 
    'build_policy',
    'get_sampler',
    'get_test_queries',
    'get_algorithm',
]


if __name__ == "__main__":
    import torch
    
    print("=" * 70)
    print("KGE Experiment Runner")
    print("=" * 70)
    
    config = load_config('kge', 
        dataset='countries_s3',
        n_envs=128,
        n_steps=128,
        batch_size=512,  # Must divide n_envs * n_steps evenly
        total_timesteps=700000,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    print(f"\n[1] Config: {config.dataset}, device={config.device}")
    
    env = build_env(config)
    print(f"[2] Environment: batch_size={env.batch_size}")
    
    policy = build_policy(config)
    print(f"[3] Policy: {sum(p.numel() for p in policy.parameters())} params")
    
    # Step 4: Create algorithm
    algorithm = get_algorithm(policy, env, config)
    print(f"[4] Algorithm created")
    
    # Step 5: Train
    print(f"\n[5] Training...")
    algorithm.learn(total_timesteps=config.total_timesteps)
    
    # Step 6: Evaluate
    print(f"\n[6] Evaluating...")
    results = algorithm.evaluate()
    
    print(f"\nResults: {[(k, v) for k, v in results.items()]}")
