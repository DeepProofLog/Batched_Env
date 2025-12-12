"""
Test and Profile Optimized Evaluation.

This script compares the original and optimized eval_corruptions implementations:
1. Correctness: Verify metrics match within tolerance
2. Performance: Measure timing difference
3. Memory: Monitor GPU memory usage

Usage:
    python tests/test_eval_optimized.py
    python tests/test_eval_optimized.py --n-queries 100 --n-corruptions 1000
    python tests/test_eval_optimized.py --profile-only  # Skip correctness test
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
from types import SimpleNamespace
from typing import Dict, Any

import numpy as np
import torch


def setup_components(device: torch.device, config: SimpleNamespace):
    """Initialize all components needed for evaluation."""
    from data_handler import DataHandler
    from index_manager import IndexManager
    from unification import UnificationEngine, set_compile_mode
    from env import BatchedEnv
    from embeddings import EmbedderLearnable as TensorEmbedder
    from model import ActorCriticPolicy as TensorPolicy
    from sampler import Sampler
    
    # Enable compile mode
    set_compile_mode(True)
    
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
    
    # Sampler
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
    
    # Reseed
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
    
    # Stringifier params
    stringifier_params = {
        'verbose': 0,
        'idx2predicate': im.idx2predicate,
        'idx2constant': im.idx2constant,
        'idx2template_var': im.idx2template_var,
        'padding_idx': im.padding_idx,
        'n_constants': im.constant_no
    }
    
    # Unification engine
    engine = UnificationEngine.from_index_manager(
        im, take_ownership=True,
        stringifier_params=stringifier_params,
        end_pred_idx=im.end_pred_idx,
        end_proof_action=config.end_proof_action,
        max_derived_per_state=config.padding_states,
    )
    engine.index_manager = im
    
    # Convert queries
    def convert_queries_unpadded(queries):
        tensors = []
        for q in queries:
            atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
            tensors.append(atom)
        return torch.stack(tensors, dim=0)
    
    def convert_queries_padded(queries):
        tensors = []
        for q in queries:
            atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
            padded = torch.full((config.padding_atoms, 3), im.padding_idx, dtype=torch.long, device=device)
            padded[0] = atom
            tensors.append(padded)
        return torch.stack(tensors, dim=0)
    
    test_queries_unpadded = convert_queries_unpadded(dh.test_queries)
    test_queries_padded = convert_queries_padded(dh.test_queries)
    
    # Eval environment
    eval_env = BatchedEnv(
        batch_size=config.batch_size_env,
        queries=test_queries_padded,
        labels=torch.ones(len(dh.test_queries), dtype=torch.long, device=device),
        query_depths=torch.ones(len(dh.test_queries), dtype=torch.long, device=device),
        unification_engine=engine,
        mode='eval',
        max_depth=config.max_depth,
        memory_pruning=config.memory_pruning,
        use_exact_memory=False,
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
        sample_deterministic_per_env=False,
    )
    
    # Policy
    action_size = config.padding_states
    policy = TensorPolicy(
        embedder=embedder,
        embed_dim=config.atom_embedding_size,
        action_dim=action_size,
        hidden_dim=256,
        num_layers=8,
        dropout_prob=0.0,
        device=device,
        compile_policy=config.compile,
    ).to(device)
    
    return policy, eval_env, sampler, dh, im, test_queries_unpadded


def run_original_eval(policy, eval_env, sampler, test_queries, config):
    """Run original eval_corruptions."""
    from model_eval import eval_corruptions
    
    n_queries = config.n_test_queries or len(test_queries)
    queries = test_queries[:n_queries]
    
    return eval_corruptions(
        actor=policy,
        env=eval_env,
        queries=queries,
        sampler=sampler,
        n_corruptions=config.n_corruptions,
        corruption_modes=tuple(config.corruption_modes),
        verbose=config.verbose,
    )


def run_optimized_eval(policy, eval_env, sampler, test_queries, config):
    """Run optimized eval_corruptions."""
    from model_eval_optimized import eval_corruptions_optimized
    
    n_queries = config.n_test_queries or len(test_queries)
    queries = test_queries[:n_queries]
    
    return eval_corruptions_optimized(
        actor=policy,
        env=eval_env,
        queries=queries,
        sampler=sampler,
        n_corruptions=config.n_corruptions,
        corruption_modes=tuple(config.corruption_modes),
        verbose=config.verbose,
        vram_gb=config.vram_gb,
    )


def run_standalone_eval(policy, eval_env, sampler, test_queries, config):
    """Run standalone optimized eval_corruptions using EvalOnlyEnv."""
    from model_eval_optimized import eval_with_existing_env
    
    n_queries = config.n_test_queries or len(test_queries)
    queries = test_queries[:n_queries]
    
    return eval_with_existing_env(
        actor=policy,
        env=eval_env,
        queries=queries,
        sampler=sampler,
        n_corruptions=config.n_corruptions,
        corruption_modes=tuple(config.corruption_modes),
        verbose=config.verbose,
    )


def test_correctness(policy, eval_env, sampler, test_queries, config):
    """Test that optimized results match original within tolerance."""
    print("\n" + "="*60)
    print("CORRECTNESS TEST")
    print("="*60)
    
    # Use small test
    small_config = SimpleNamespace(**vars(config))
    small_config.n_test_queries = min(10, config.n_test_queries or 10)
    small_config.n_corruptions = min(50, config.n_corruptions or 50)
    small_config.verbose = False
    
    print(f"Testing with {small_config.n_test_queries} queries, {small_config.n_corruptions} corruptions...")
    
    # Run original
    print("Running original...")
    orig_result = run_original_eval(policy, eval_env, sampler, test_queries, small_config)
    
    # Run optimized
    print("Running optimized...")
    opt_result = run_optimized_eval(policy, eval_env, sampler, test_queries, small_config)
    
    # Compare metrics
    tolerance = 0.05  # 5% tolerance for minor differences
    metrics = ["MRR", "Hits@1", "Hits@3", "Hits@10"]
    
    print(f"\n{'Metric':<12} {'Original':>10} {'Optimized':>10} {'Diff':>10} {'Status':>8}")
    print("-" * 52)
    
    all_pass = True
    for m in metrics:
        orig_val = orig_result.get(m, 0.0)
        opt_val = opt_result.get(m, 0.0)
        diff = abs(orig_val - opt_val)
        status = "✓ PASS" if diff <= tolerance else "✗ FAIL"
        if diff > tolerance:
            all_pass = False
        print(f"{m:<12} {orig_val:>10.4f} {opt_val:>10.4f} {diff:>10.4f} {status:>8}")
    
    print("-" * 52)
    print(f"Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    
    return all_pass


def test_performance(policy, eval_env, sampler, test_queries, config):
    """Benchmark performance of both implementations."""
    print("\n" + "="*60)
    print("PERFORMANCE TEST")
    print("="*60)
    
    perf_config = SimpleNamespace(**vars(config))
    perf_config.verbose = False
    
    print(f"Testing with {perf_config.n_test_queries} queries, {perf_config.n_corruptions} corruptions...")
    
    # Warmup
    print("Warming up...")
    warmup_config = SimpleNamespace(**vars(perf_config))
    warmup_config.n_test_queries = 2
    warmup_config.n_corruptions = 10
    _ = run_original_eval(policy, eval_env, sampler, test_queries, warmup_config)
    _ = run_optimized_eval(policy, eval_env, sampler, test_queries, warmup_config)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark original
    print("Benchmarking original...")
    start = time.time()
    orig_result = run_original_eval(policy, eval_env, sampler, test_queries, perf_config)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    orig_time = time.time() - start
    
    # Benchmark optimized
    print("Benchmarking optimized...")
    start = time.time()
    opt_result = run_optimized_eval(policy, eval_env, sampler, test_queries, perf_config)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    opt_time = time.time() - start
    
    # Report
    speedup = orig_time / opt_time if opt_time > 0 else float('inf')
    
    print(f"\n{'Implementation':<15} {'Time (s)':>12} {'MRR':>10}")
    print("-" * 40)
    print(f"{'Original':<15} {orig_time:>12.2f} {orig_result.get('MRR', 0.0):>10.4f}")
    print(f"{'Optimized':<15} {opt_time:>12.2f} {opt_result.get('MRR', 0.0):>10.4f}")
    print("-" * 40)
    print(f"Speedup: {speedup:.2f}x")
    
    return {
        "original_time": orig_time,
        "optimized_time": opt_time,
        "speedup": speedup,
    }


def main():
    parser = argparse.ArgumentParser(description='Test optimized evaluation')
    parser.add_argument('--dataset', type=str, default='family', help='Dataset name')
    parser.add_argument('--n-test-queries', type=int, default=20, help='Number of test queries')
    parser.add_argument('--n-corruptions', type=int, default=100, help='Corruptions per query')
    parser.add_argument('--batch-size-env', type=int, default=100, help='Environment batch size')
    parser.add_argument('--corruption-modes', type=str, nargs='+', default=['both'], 
                        help='Corruption modes')
    parser.add_argument('--vram-gb', type=float, default=8.0, help='Available VRAM in GB')
    parser.add_argument('--profile-only', action='store_true', help='Skip correctness test')
    parser.add_argument('--correctness-only', action='store_true', help='Skip performance test')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--compile', default=True, type=lambda x: x.lower() != 'false')
    args = parser.parse_args()
    
    # Configuration
    config = SimpleNamespace(
        dataset=args.dataset,
        data_path='./data/',
        n_test_queries=args.n_test_queries,
        n_corruptions=args.n_corruptions,
        batch_size_env=args.batch_size_env,
        corruption_modes=args.corruption_modes,
        verbose=args.verbose,
        vram_gb=args.vram_gb,
        padding_atoms=6,
        padding_states=120,
        max_depth=20,
        memory_pruning=True,
        skip_unary_actions=False,
        end_proof_action=True,
        reward_type=0,
        max_total_vars=100,
        atom_embedding_size=250,
        seed=0,
        compile=args.compile,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"VRAM budget: {config.vram_gb}GB")
    
    print("\nSetting up components...")
    policy, eval_env, sampler, dh, im, test_queries = setup_components(device, config)
    
    results = {}
    
    # Correctness test
    if not args.profile_only:
        correct = test_correctness(policy, eval_env, sampler, test_queries, config)
        results["correctness"] = correct
    
    # Performance test
    if not args.correctness_only:
        perf = test_performance(policy, eval_env, sampler, test_queries, config)
        results["performance"] = perf
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if "correctness" in results:
        print(f"Correctness: {'PASS' if results['correctness'] else 'FAIL'}")
    if "performance" in results:
        print(f"Speedup: {results['performance']['speedup']:.2f}x")
    
    return results


if __name__ == '__main__':
    main()
