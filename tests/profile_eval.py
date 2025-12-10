"""
Profile the eval_corruptions() function in the tensor version.

This script profiles the evaluation loop to identify bottlenecks in the
tensor-based implementation. Uses both cProfile and torch.profiler for
comprehensive CPU and GPU profiling.

Usage:
    python tests/profile_eval.py
    python tests/profile_eval.py --use-gpu-profiler
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import argparse
import cProfile
import pstats
import io
from types import SimpleNamespace
from time import time

import numpy as np
import torch
try:
    from torch.profiler import profile, ProfilerActivity
except (ImportError, ModuleNotFoundError):
    profile = None
    ProfilerActivity = None

from utils.seeding import seed_all


def setup_components(device: torch.device, config: SimpleNamespace):
    """
    Initialize all components needed for evaluation.
    
    Args:
        device: Target device (cuda or cpu)
        config: Configuration namespace
        
    Returns:
        Tuple of (policy, eval_env, sampler, dh, im, test_queries_tensor)
    """
    from data_handler import DataHandler
    from index_manager import IndexManager
    from unification import UnificationEngine, set_compile_mode
    from env import BatchedEnv
    from embeddings import EmbedderLearnable as TensorEmbedder
    from model import ActorCriticPolicy as TensorPolicy
    from sampler import Sampler
    
    # Enable compile mode for maximum performance
    set_compile_mode(True)
    
    # Set seeds for reproducibility (deterministic=False for performance)
    seed_all(config.seed, deterministic=False, warn=False)
    
    # Load data
    dh = DataHandler(
        dataset_name=config.dataset,
        base_path=config.data_path,
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        corruption_mode='dynamic',
    )
    
    # Index manager
    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=config.max_total_vars,
        max_arity=dh.max_arity,
        padding_atoms=config.padding_atoms,
        device=device,
        rules=dh.rules,
    )
    
    # Materialize indices
    dh.materialize_indices(im=im, device=device)
    
    # Sampler - use tail-only for countries datasets to match runner.py
    default_mode = 'tail' if 'countries' in config.dataset else 'both'
    domain2idx, entity2domain = dh.get_sampler_domain_info()
    sampler = Sampler.from_data(
        all_known_triples_idx=dh.all_known_triples_idx,
        num_entities=im.constant_no,
        num_relations=im.predicate_no,
        device=device,
        default_mode=default_mode,
        seed=config.seed,
        domain2idx=domain2idx,
        entity2domain=entity2domain,
    )
    
    # Reseed before embedder
    torch.manual_seed(config.seed)
    
    # Embedder
    embedder = TensorEmbedder(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=1000,
        max_arity=dh.max_arity,
        padding_atoms=config.padding_atoms,
        atom_embedder='transe',
        state_embedder='mean',
        constant_embedding_size=config.atom_embedding_size,
        predicate_embedding_size=config.atom_embedding_size,
        atom_embedding_size=config.atom_embedding_size,
        device=str(device),
    )
    embedder.embed_dim = config.atom_embedding_size
    
    # Create stringifier params
    stringifier_params = {
        'verbose': 0,
        'idx2predicate': im.idx2predicate,
        'idx2constant': im.idx2constant,
        'idx2template_var': im.idx2template_var,
        'padding_idx': im.padding_idx,
        'n_constants': im.constant_no
    }
    
    # Create unification engine
    engine = UnificationEngine.from_index_manager(
        im, take_ownership=True,
        stringifier_params=stringifier_params,
        end_pred_idx=im.end_pred_idx,
        end_proof_action=config.end_proof_action,
        max_derived_per_state=config.padding_states,
    )
    engine.index_manager = im
    
    # Convert queries to tensor format
    # Padded version [N, padding_atoms, 3] for BatchedEnv
    def convert_queries_padded(queries):
        tensors = []
        for q in queries:
            atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
            padded = torch.full((config.padding_atoms, 3), im.padding_idx, dtype=torch.long, device=device)
            padded[0] = atom
            tensors.append(padded)
        return torch.stack(tensors, dim=0)
    
    # Unpadded version [N, 3] for eval_corruptions sampler
    def convert_queries_unpadded(queries):
        tensors = []
        for q in queries:
            atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
            tensors.append(atom)
        return torch.stack(tensors, dim=0)
    
    test_queries_padded = convert_queries_padded(dh.test_queries)
    test_queries_unpadded = convert_queries_unpadded(dh.test_queries)
    
    # Eval environment (requires padded queries)
    eval_env = BatchedEnv(
        batch_size=config.batch_size_env,
        queries=test_queries_padded,
        labels=torch.ones(len(dh.test_queries), dtype=torch.long, device=device),
        query_depths=torch.ones(len(dh.test_queries), dtype=torch.long, device=device),
        unification_engine=engine,
        mode='eval',
        max_depth=config.max_depth,
        memory_pruning=config.memory_pruning,
        use_exact_memory=False,  # Use BloomFilter (production default)
        skip_unary_actions=config.skip_unary_actions,
        end_proof_action=config.end_proof_action,
        reward_type=config.reward_type,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        true_pred_idx=im.predicate_str2idx.get('True'),
        false_pred_idx=im.predicate_str2idx.get('False'),
        end_pred_idx=im.predicate_str2idx.get('Endf'),
        verbose=0,
        prover_verbose=0,
        device=device,
        runtime_var_start_index=im.constant_no + 1,
        total_vocab_size=im.constant_no + config.max_total_vars,
        sample_deterministic_per_env=False,  # Disabled for performance
    )
    
    # Reseed before model creation (deterministic=False for performance)
    seed_all(config.seed, deterministic=False, warn=False)
    
    # Policy - with compilation enabled for performance
    action_size = config.padding_states
    policy = TensorPolicy(
        embedder=embedder,
        embed_dim=config.atom_embedding_size,
        action_dim=action_size,
        hidden_dim=256,
        num_layers=8,
        dropout_prob=0.0,
        device=device,
        compile_policy=config.compile,  # Enable torch.compile
    ).to(device)
    
    return policy, eval_env, sampler, dh, im, test_queries_unpadded


def run_eval_corruptions(policy, eval_env, sampler, test_queries_tensor, config: SimpleNamespace):
    """Run eval_corruptions with the given configuration."""
    from model_eval import eval_corruptions as tensor_eval_corruptions
    
    # Limit queries if specified
    n_queries = config.n_test_queries or len(test_queries_tensor)
    queries = test_queries_tensor[:n_queries]
    
    # Run evaluation
    results = tensor_eval_corruptions(
        actor=policy,
        env=eval_env,
        queries=queries,
        sampler=sampler,
        n_corruptions=config.n_corruptions,
        corruption_modes=tuple(config.corruption_modes),
        verbose=config.verbose,
    )
    
    return results


def profile_eval_cprofile(config: SimpleNamespace):
    """Profile the eval_corruptions() function with cProfile."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\nSetting up components...")
    policy, eval_env, sampler, dh, im, test_queries_tensor = setup_components(device, config)
    
    n_queries = config.n_test_queries or len(test_queries_tensor)
    
    # Warmup runs to amortize torch.compile overhead
    warmup_runs = config.warmup_runs
    if warmup_runs > 0:
        print(f"\nRunning {warmup_runs} warmup iteration(s)...")
        for i in range(warmup_runs):
            # Use smaller subset for warmup to save time
            warmup_config = SimpleNamespace(**vars(config))
            warmup_config.n_test_queries = min(2, n_queries)
            warmup_config.n_corruptions = min(10, config.n_corruptions)
            _ = run_eval_corruptions(policy, eval_env, sampler, test_queries_tensor, warmup_config)
            print(f"  Warmup {i+1}/{warmup_runs} complete")
        
        # Synchronize GPU before profiling
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    print(f"\nProfiling eval_corruptions() on {n_queries} queries with {config.n_corruptions} corruptions each...")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time()
    results = run_eval_corruptions(policy, eval_env, sampler, test_queries_tensor, config)
    elapsed = time() - start_time
    
    profiler.disable()
    
    print(f"\nEvaluation completed in {elapsed:.2f}s")
    print(f"MRR: {results.get('MRR', 0.0):.4f}")
    print(f"Hits@1: {results.get('Hits@1', 0.0):.4f}")
    print(f"Hits@10: {results.get('Hits@10', 0.0):.4f}")
    
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
    output_path = 'tests/profile_eval_results.txt'
    with open(output_path, 'w') as f:
        f.write(f"Device: {device}\n")
        f.write(f"Number of queries: {n_queries}\n")
        f.write(f"Corruptions per query: {config.n_corruptions}\n")
        f.write(f"Evaluation time: {elapsed:.2f}s\n")
        f.write(f"MRR: {results.get('MRR', 0.0):.4f}\n")
        f.write(f"Hits@1: {results.get('Hits@1', 0.0):.4f}\n\n")
        
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
    """Profile the eval_corruptions() function with torch.profiler for GPU analysis."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        print("CUDA not available, cannot profile GPU usage")
        return
    
    print(f"Using device: {device}")
    
    print("\nSetting up components...")
    policy, eval_env, sampler, dh, im, test_queries_tensor = setup_components(device, config)
    
    n_queries = config.n_test_queries or len(test_queries_tensor)
    print(f"\nGPU Profiling eval_corruptions() on {n_queries} queries with {config.n_corruptions} corruptions each...")
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        start_time = time()
        results = run_eval_corruptions(policy, eval_env, sampler, test_queries_tensor, config)
        elapsed = time() - start_time
    
    print(f"\nEvaluation completed in {elapsed:.2f}s")
    print(f"MRR: {results.get('MRR', 0.0):.4f}")
    print(f"Hits@1: {results.get('Hits@1', 0.0):.4f}")
    print(f"Hits@10: {results.get('Hits@10', 0.0):.4f}")
    
    print("\n" + "="*80)
    print("GPU PROFILING RESULTS - Top CUDA Time")
    print("="*80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    
    print("\n" + "="*80)
    print("GPU PROFILING RESULTS - Top CPU Time")
    print("="*80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
    
    # Save results to file
    output_path = 'tests/profile_eval_gpu_results.txt'
    with open(output_path, 'w') as f:
        f.write(f"Device: {device}\n")
        f.write(f"Number of queries: {n_queries}\n")
        f.write(f"Corruptions per query: {config.n_corruptions}\n")
        f.write(f"Evaluation time: {elapsed:.2f}s\n")
        f.write(f"MRR: {results.get('MRR', 0.0):.4f}\n")
        f.write(f"Hits@1: {results.get('Hits@1', 0.0):.4f}\n\n")
        
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
    parser = argparse.ArgumentParser(description='Profile eval_corruptions() function')
    parser.add_argument('--use-gpu-profiler', action='store_true',
                        help='Use torch.profiler for GPU profiling instead of cProfile')
    parser.add_argument('--dataset', type=str, default='family',
                        help='Dataset name')
    parser.add_argument('--n-test-queries', type=int, default=10,
                        help='Number of test queries (None = all)')
    parser.add_argument('--n-corruptions', type=int, default=100,
                        help='Number of corruptions per query')
    parser.add_argument('--batch-size-env', type=int, default=10,
                        help='Environment batch size')
    parser.add_argument('--corruption-modes', type=str, nargs='+', default=['both'],
                        help='Corruption modes (head, tail, or both)')
    parser.add_argument('--verbose', default=True, type=lambda x: x.lower() != 'false',
                        help='Enable verbose output during evaluation')
    parser.add_argument('--compile', default=True, type=lambda x: x.lower() != 'false',
                        help='Enable torch.compile for model (default: True)')
    parser.add_argument('--warmup-runs', type=int, default=1,
                        help='Number of warmup runs before profiling (default: 1)')
    args = parser.parse_args()
    
    # Configuration matching runner.py defaults
    config = SimpleNamespace(
        dataset=args.dataset,
        data_path='./data/',
        n_test_queries=args.n_test_queries,
        n_corruptions=args.n_corruptions,
        batch_size_env=args.batch_size_env,
        corruption_modes=args.corruption_modes,
        verbose=args.verbose,
        padding_atoms=6,
        padding_states=20,
        max_depth=20,
        memory_pruning=True,
        skip_unary_actions=False,
        end_proof_action=True,
        reward_type=0,
        max_total_vars=100,
        atom_embedding_size=250,
        seed=0,
        compile=args.compile,
        warmup_runs=args.warmup_runs,
    )
    
    if args.use_gpu_profiler:
        profile_eval_gpu(config)
    else:
        profile_eval_cprofile(config)


if __name__ == '__main__':
    main()
