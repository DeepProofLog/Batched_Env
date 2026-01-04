"""
Profile the SB3 PPO learn() function.

This script profiles the training loop to identify bottlenecks in the
SB3-based implementation. Uses both cProfile and torch.profiler for
comprehensive CPU and GPU profiling.

Usage:
    python tests/sb3_profile/profile_sb3_learn.py
    python tests/sb3_profile/profile_sb3_learn.py --use-gpu-profiler
"""

import os
import sys
# Add Batched_env to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import cProfile
import pstats
import io
from types import SimpleNamespace
from time import time
import torch

from utils.seeding import seed_all
from sb3.sb3_train import _build_data_and_index
from sb3.sb3_custom_dummy_env import create_environments
from sb3.sb3_model import CustomActorCriticPolicy, CustomCombinedExtractor, PPO_custom as PPO

def setup_components(device: torch.device, config: SimpleNamespace):
    """
    Initialize all components needed for training.
    
    Args:
        device: Target device (cuda or cpu)
        config: Configuration namespace
        
    Returns:
        Tuple of (ppo, policy, env, eval_env, sampler, dh, im)
    """
    # Enable TF32 for maximum performance on Ampere+ GPUs
    torch.set_float32_matmul_precision('high')
    
    # Set seeds for reproducibility
    seed_all(config.seed, deterministic=False, warn=False)
    
    # Set some defaults required by _build_data_and_index if not in config
    # These match defaults in sb3_runner.py
    defaults = {
        'janus_file': None,
        'train_file': 'train.txt',
        'valid_file': 'valid.txt',
        'test_file': 'test.txt',
        'rules_file': 'rules.txt',
        'facts_file': 'train.txt',
        'n_train_queries': None,
        'n_eval_queries': None,
        'n_test_queries': None,
        'corruption_mode': 'dynamic',
        'train_depth': None,
        'valid_depth': None,
        'test_depth': None,
        'prob_facts': False,
        'topk_facts': None,
        'topk_facts_threshold': 0.33,
        'max_total_vars': 100,
        'padding_atoms': 6,
        'kge_action': False,
        'corruption_scheme': ['head', 'tail'],
        'atom_embedder': 'transe',
        'state_embedder': 'mean',
        'atom_embedding_size': 250,
        'learn_embeddings': True,
        'extended_eval_info': True,
        'kge_engine': None,
        'ent_coef': 0.2,
        'clip_range': 0.2,
        'gamma': 0.99,
        'clip_range_vf': None,
        'target_kl': 0.03,
        'learning_rate': 5e-5,
        'n_steps': 128,
        'batch_size': 4096,
        'n_epochs': 20,
        'pbrs_gamma': None,
        'pbrs_beta': 0.0,
        'n_eval_envs': 128,
        'engine': 'python',
        'engine_strategy': 'cmp',
        'false_rules': False,
        'verbose': False,
        'prover_verbose': False,
        'plot': False,
        'padding_states': 150,
    }
    
    for k, v in defaults.items():
        if not hasattr(config, k):
            setattr(config, k, v)

    # Derived params matching sb3_runner logic
    if not hasattr(config, 'constant_embedding_size'):
        config.constant_embedding_size = config.atom_embedding_size
    if not hasattr(config, 'predicate_embedding_size'):
        config.predicate_embedding_size = config.atom_embedding_size
    if not hasattr(config, 'state_embedding_size'):
        config.state_embedding_size = (
            config.atom_embedding_size
            if config.state_embedder != "concat"
            else config.atom_embedding_size * config.padding_atoms
        )

    # Build data and index
    dh, im, sampler, embedder = _build_data_and_index(config, device)
    
    # Create environments
    env, eval_env, callback_env = create_environments(
        config,
        dh,
        im,
        kge_engine=None,
        detailed_eval_env=config.extended_eval_info,
    )
    
    # Policy kwargs
    policy_kwargs = {
        'features_extractor_class': CustomCombinedExtractor,
        'features_extractor_kwargs': {'features_dim': embedder.embed_dim, 'embedder': embedder},
    }

    # PPO
    ppo = PPO(
        CustomActorCriticPolicy,
        env,
        learning_rate=config.lr,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        verbose=1,
        device=device,
        ent_coef=config.ent_coef,
        clip_range=config.clip_range,
        gamma=config.gamma,
        clip_range_vf=config.clip_range_vf,
        target_kl=config.target_kl,
        policy_kwargs=policy_kwargs
    )
    
    # We want to access the policy directly too
    policy = ppo.policy
    
    return ppo, policy, env, eval_env, sampler, dh, im


def profile_learn_cprofile(config: SimpleNamespace):
    """Profile the PPO learn() function with cProfile."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\nSetting up components...")
    init_start = time()
    ppo, policy, train_env, eval_env, sampler, dh, im = setup_components(device, config)
    init_time = time() - init_start
    print(f"Initialization time: {init_time:.2f}s")
    
    print(f"\nProfiling PPO.learn() for {config.total_timesteps} timesteps...")
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time()
    ppo.learn(total_timesteps=config.total_timesteps)
    elapsed = time() - start_time
    
    profiler.disable()
    
    print(f"\nTraining completed in {elapsed:.2f}s")
    
    # Print profiling results
    n_functions = 30
    
    print("\n" + "="*80)
    print("PROFILING RESULTS - Top by Cumulative Time")
    print("="*80)
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(n_functions)
    print(s.getvalue())
    
    print("\n" + "="*80)
    print("PROFILING RESULTS - Top by Total Time")
    print("="*80)
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('tottime')
    ps.print_stats(n_functions)
    print(s.getvalue())
    
    # Save results to file
    output_path = 'tests/sb3_profile/profile_sb3_learn_results.txt'
    with open(output_path, 'w') as f:
        f.write(f"Device: {device}\n")
        f.write(f"Total timesteps: {config.total_timesteps}\n")
        f.write(f"Training time: {elapsed:.2f}s\n\n")
        
        ps = pstats.Stats(profiler, stream=f)
        ps.strip_dirs()
        f.write("="*80 + "\n")
        f.write("Top by Cumulative Time\n")
        f.write("="*80 + "\n")
        ps.sort_stats('cumulative')
        ps.print_stats(n_functions)
    
        f.write("\n\n" + "="*80 + "\n")
        f.write("Top by Total Time\n")
        f.write("="*80 + "\n")
        ps.sort_stats('tottime')
        ps.print_stats(n_functions)
    
    print(f"\nResults saved to {output_path}")


def profile_learn_gpu(config: SimpleNamespace):
    """Profile the PPO learn() function with torch.profiler for GPU analysis."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        print("CUDA not available, cannot profile GPU usage")
        return
    
    print(f"Using device: {device}")
    
    print("\nSetting up components...")
    ppo, policy, train_env, eval_env, sampler, dh, im = setup_components(device, config)
    
    print(f"\nGPU Profiling PPO.learn() for {config.total_timesteps} timesteps...")
    
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        start_time = time()
        ppo.learn(total_timesteps=config.total_timesteps)
        elapsed = time() - start_time
    
    print(f"\nTraining completed in {elapsed:.2f}s")
    
    print("\n" + "="*80)
    print("GPU PROFILING RESULTS - Top CUDA Time")
    print("="*80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    
    print("\n" + "="*80)
    print("GPU PROFILING RESULTS - Top CPU Time")
    print("="*80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
    
    # Save results to file
    output_path = 'tests/sb3_profile/profile_sb3_learn_gpu_results.txt'
    with open(output_path, 'w') as f:
        f.write(f"Device: {device}\n")
        f.write(f"Total timesteps: {config.total_timesteps}\n")
        f.write(f"Training time: {elapsed:.2f}s\n\n")
        
        f.write("="*80 + "\n")
        f.write("Top CUDA Time\n")
        f.write("="*80 + "\n")
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
        
        f.write("\n\n" + "="*80 + "\n")
        f.write("Top CPU Time\n")
        f.write("="*80 + "\n")
        f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
    
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Profile SB3 PPO learn() function')
    parser.add_argument('--use-gpu-profiler', action='store_true',
                        help='Use torch.profiler for GPU profiling instead of cProfile')
    parser.add_argument('--dataset', type=str, default='family',
                        help='Dataset name')
    parser.add_argument('--total-timesteps', type=int, default=128*5,
                        help='Total training timesteps')
    parser.add_argument('--batch-size-env', type=int, default=32,
                        help='Environment batch size (n_envs)')
    parser.add_argument('--n-steps', type=int, default=128,
                        help='Steps per rollout')
    parser.add_argument('--n-epochs', type=int, default=10,
                        help='PPO epochs per update')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='PPO minibatch size')
    
    args = parser.parse_args()
    
    # Configuration matching runner.py defaults
    config = SimpleNamespace(
        dataset_name=args.dataset,
        data_path='./data/',
        total_timesteps=args.total_timesteps,
        n_envs=args.batch_size_env, # sb3 uses n_envs instead of batch_size_env
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=5e-5,
        seed=[0], # SB3 runner expects a list for seed? Or just int? In standard it is int usually, but runner has list.
        # But setup_components uses seed_all(config.seed, ...) which likely expects int.
        # Let's double check. My setup_components passes config.seed directly.
        # sb3_runner default says 'seed': [0]
        # But _set_seeds(args.seed_run_i) is used in sb3_train. So it expects a single seed there.
        # I'll modify setup_components/main to be consistent.
    )

    # Note: runner has seed as list, but _set_seeds expects int.
    # I'll just put seed=0 in config and handle it.
    config.seed = 0
    config.seed_run_i = 0
    
    # Add other necessary fields for sb3 envs
    # In sb3_runner:
    if config.n_envs > 1:
        # We need to make sure vector env creation works. 
        pass

    # Some defaults are needed for environment creation called in setup_components
    config.reward_type = 0
    config.train_neg_ratio = 4
    config.endf_action = True
    config.endt_action = False
    config.learning_rate = 5e-5 # Duplicate but safe
    config.skip_unary_actions = True
    config.max_depth = 20
    config.memory_pruning = True
    config.num_processes = 1 # ? 
    
    if args.use_gpu_profiler:
        profile_learn_gpu(config)
    else:
        profile_learn_cprofile(config)


if __name__ == '__main__':
    main()
