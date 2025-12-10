"""
Profile the SB3 eval_corruptions() function.

This script profiles the evaluation loop to identify bottlenecks in the
SB3-based implementation. Uses both cProfile and torch.profiler for
comprehensive CPU and GPU profiling.

Usage:
    python tests/sb3_profile/profile_sb3_eval.py
    python tests/sb3_profile/profile_sb3_eval.py --use-gpu-profiler
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
from sb3.sb3_model_eval import eval_corruptions

def setup_components(device: torch.device, config: SimpleNamespace):
    """
    Initialize all components needed for evaluation.
    
    Args:
        device: Target device (cuda or cpu)
        config: Configuration namespace
        
    Returns:
        Tuple of (model, eval_env, sampler, dh, im)
    """
    # Enable TF32 for maximum performance on Ampere+ GPUs
    torch.set_float32_matmul_precision('high')
    
    # Set seeds for reproducibility
    seed_all(config.seed, deterministic=False, warn=False)
    
    # Defaults matching sb3_runner.py
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
        'dataset_name': 'family',
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
    model = PPO(
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
    
    # Ensure model is in eval mode
    model.policy.set_training_mode(False)
    
    return model, eval_env, sampler, dh, im


def run_eval_corruptions(model, eval_env, sampler, dh, config: SimpleNamespace):
    """Run eval_corruptions with the given configuration."""
    
    # Test queries
    test_queries = dh.test_queries
    if config.n_test_queries is not None:
        test_queries = test_queries[:config.n_test_queries]
        data_depths = dh.test_queries_depths[:config.n_test_queries]
    else:
        current_len = len(test_queries)
        # Check defaults or passed args for limit
        # sb3_train logic:
        # args.n_test_queries = min(args.n_test_queries, len(dh.test_queries)) if args.n_test_queries else len...
        # Here we just use what is in test_queries
        data_depths = dh.test_queries_depths

    eval_args = {
        "model": model,
        "env": eval_env,
        "data": test_queries,
        "sampler": sampler,
        "n_corruptions": config.n_corruptions,
        "verbose": config.verbose,
        "kge_inference_engine": None,
        "evaluation_mode": "rl_only",
        "plot": False,
        "corruption_scheme": config.corruption_scheme,
        "data_depths": data_depths,
        # "info_callback": None,
        "hybrid_kge_weight": 0.0,
        "hybrid_rl_weight": 1.0,
        "hybrid_success_only": True,
    }
    
    results = eval_corruptions(**eval_args)
    return results


def profile_eval_cprofile(config: SimpleNamespace):
    """Profile the eval_corruptions() function with cProfile."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\nSetting up components...")
    model, eval_env, sampler, dh, im = setup_components(device, config)
    
    n_queries = config.n_test_queries if config.n_test_queries else len(dh.test_queries)
    
    # Warmup runs? SB3 PPO setup usually doesn't need warmup as much as compiled tensor version, 
    # but if we enable compile in future it might.
    
    print(f"\nProfiling eval_corruptions() on {n_queries} queries with {config.n_corruptions} corruptions each...")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time()
    results = run_eval_corruptions(model, eval_env, sampler, dh, config)
    elapsed = time() - start_time
    
    profiler.disable()
    
    print(f"\nEvaluation completed in {elapsed:.2f}s")
    print(f"MRR: {results.get('mrr_mean', 0.0):.4f}")
    print(f"Hits@1: {results.get('h1_mean', 0.0):.4f}")
    print(f"Hits@10: {results.get('h10_mean', 0.0):.4f}")
    
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
    output_path = 'tests/sb3_profile/profile_sb3_eval_results.txt'
    with open(output_path, 'w') as f:
        f.write(f"Device: {device}\n")
        f.write(f"Number of queries: {n_queries}\n")
        f.write(f"Corruptions per query: {config.n_corruptions}\n")
        f.write(f"Evaluation time: {elapsed:.2f}s\n")
        f.write(f"MRR: {results.get('mrr_mean', 0.0):.4f}\n")
        f.write(f"Hits@1: {results.get('h1_mean', 0.0):.4f}\n\n")
        
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


def profile_eval_gpu(config: SimpleNamespace):
    """Profile the function with torch.profiler for GPU analysis."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        print("CUDA not available, cannot profile GPU usage")
        return
    
    print(f"Using device: {device}")
    
    print("\nSetting up components...")
    model, eval_env, sampler, dh, im = setup_components(device, config)
    
    print(f"\nGPU Profiling eval_corruptions...")
    
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        start_time = time()
        results = run_eval_corruptions(model, eval_env, sampler, dh, config)
        elapsed = time() - start_time
    
    print(f"\nEvaluation completed in {elapsed:.2f}s")
    
    print("\n" + "="*80)
    print("GPU PROFILING RESULTS - Top CUDA Time")
    print("="*80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    
    print("\n" + "="*80)
    print("GPU PROFILING RESULTS - Top CPU Time")
    print("="*80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
    
    # Save results to file
    output_path = 'tests/sb3_profile/profile_sb3_eval_gpu_results.txt'
    with open(output_path, 'w') as f:
        f.write(f"Device: {device}\n")
        f.write(f"Evaluation time: {elapsed:.2f}s\n\n")
        
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
    parser = argparse.ArgumentParser(description='Profile SB3 eval_corruptions() function')
    parser.add_argument('--use-gpu-profiler', action='store_true',
                        help='Use torch.profiler for GPU profiling instead of cProfile')
    parser.add_argument('--dataset', type=str, default='family',
                        help='Dataset name')
    parser.add_argument('--n-test-queries', type=int, default=10,
                        help='Number of test queries (None = all)')
    parser.add_argument('--n-corruptions', type=int, default=100,
                        help='Number of corruptions per query')
    parser.add_argument('--batch-size-env', type=int, default=100,
                        help='Environment batch size (n_envs)')
    parser.add_argument('--verbose', default=False, type=lambda x: x.lower() != 'false',
                        help='Enable verbose output during evaluation')
    
    args = parser.parse_args()
    
    # Configuration
    config = SimpleNamespace(
        dataset_name=args.dataset,
        data_path='./data/',
        n_test_queries=args.n_test_queries,
        n_corruptions=args.n_corruptions,
        n_envs=args.batch_size_env,
        verbose=args.verbose,
        seed=0,
        seed_run_i=0,
    )
    
    # Defaults required for env/setup
    config.reward_type = 0
    config.train_neg_ratio = 4
    config.endf_action = True
    config.endt_action = False
    config.skip_unary_actions = True
    config.max_depth = 20
    config.memory_pruning = True
    config.num_processes = 1
    
    if args.use_gpu_profiler:
        profile_eval_gpu(config)
    else:
        profile_eval_cprofile(config)

if __name__ == '__main__':
    main()
