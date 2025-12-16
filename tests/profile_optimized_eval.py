"""
Profile the PPOOptimized.evaluate_with_corruptions() function.

This script profiles the optimized evaluation loop to identify bottlenecks.
Uses cProfile, torch.profiler, and optional line profiling for comprehensive
CPU and GPU analysis.

Measures:
- Warmup time (compilation + first run)
- Runtime (excluding warmup)
- Total time
- ms per query
- ms per candidate

Usage:
    python tests/profile_optimized_eval.py
    python tests/profile_optimized_eval.py --use-gpu-profiler
    python tests/profile_optimized_eval.py --use-line-profiler
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import cProfile
import pstats
import io
from datetime import datetime
from types import SimpleNamespace
from time import time
from typing import Optional, Tuple

import numpy as np
import torch

try:
    from torch.profiler import profile, ProfilerActivity
except (ImportError, ModuleNotFoundError):
    profile = None
    ProfilerActivity = None


def setup_components(device: torch.device, config: SimpleNamespace):
    """
    Initialize all components needed for optimized evaluation.
    
    Returns:
        Dict with all components needed for evaluation.
    """
    from data_handler import DataHandler
    from index_manager import IndexManager
    from unification import UnificationEngine, set_compile_mode
    from unification_optimized import UnificationEngineVectorized
    from embeddings import EmbedderLearnable as TensorEmbedder
    from model import ActorCriticPolicy as TensorPolicy
    from sampler import Sampler
    from env_optimized import EvalEnvOptimized
    
    # Enable compile mode
    set_compile_mode(config.compile)
    
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
    
    # Create stringifier params
    stringifier_params = {
        'verbose': 0,
        'idx2predicate': im.idx2predicate,
        'idx2constant': im.idx2constant,
        'idx2template_var': im.idx2template_var,
        'padding_idx': im.padding_idx,
        'n_constants': im.constant_no
    }
    
    # Base unification engine
    engine = UnificationEngine.from_index_manager(
        im, take_ownership=True,
        stringifier_params=stringifier_params,
        end_pred_idx=im.end_pred_idx,
        end_proof_action=config.end_proof_action,
        max_derived_per_state=config.padding_states,
    )
    engine.index_manager = im
    
    # Vectorized engine for compiled path
    vec_engine = UnificationEngineVectorized.from_base_engine(
        engine,
        max_fact_pairs=None,  # Auto-compute
        max_rule_pairs=None,  # Auto-compute
        padding_atoms=config.padding_atoms,
    )
    
    # Convert queries
    def convert_queries_unpadded(queries):
        tensors = []
        for q in queries:
            atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
            tensors.append(atom)
        return torch.stack(tensors, dim=0)
    
    test_queries = convert_queries_unpadded(dh.test_queries)
    
    # Compiled environment
    eval_env = EvalEnvOptimized(
        vec_engine=vec_engine,
        batch_size=config.batch_size_env,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_depth,
        end_proof_action=config.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
        memory_pruning=True,
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
    
    return {
        'policy': policy,
        'eval_env': eval_env,
        'sampler': sampler,
        'dh': dh,
        'im': im,
        'vec_engine': vec_engine,
        'test_queries': test_queries,
    }


def create_ppo_and_warmup(components, config) -> Tuple[any, float]:
    """Create PPOOptimized and run warmup. Returns (ppo, warmup_time_s)."""
    from ppo_optimized import PPOOptimized, compute_optimal_batch_size
    
    policy = components['policy']
    env = components['eval_env']
    queries = components['test_queries'][:config.n_test_queries].to(env.device)
    
    # Compute batch size
    effective_chunk = min(config.chunk_queries, len(queries))
    batch_size = compute_optimal_batch_size(
        chunk_queries=effective_chunk,
        n_corruptions=config.n_corruptions,
        max_vram_gb=config.vram_gb,
    )
    
    # Create PPOOptimized
    ppo = PPOOptimized(
        policy=policy,
        env=env,
        device=env.device,
        fixed_batch_size=batch_size,
        verbose=False,
    )
    
    # Warmup and compile
    warmup_start = time()
    
    env.compile(
        policy=policy,
        deterministic=True,
        mode=config.compile_mode,
        fullgraph=config.fullgraph,
        include_value=False,
    )
    
    # Trigger JIT compilation with exact batch size
    warmup_queries = queries[:1].expand(batch_size, -1)
    _ = ppo.evaluate_policy(warmup_queries, deterministic=True)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    warmup_time = time() - warmup_start
    
    return ppo, warmup_time


def run_evaluation(ppo, components, config):
    """Run the actual evaluation (post-warmup)."""
    sampler = components['sampler']
    queries = components['test_queries'][:config.n_test_queries].to(ppo.env.device)
    
    effective_chunk = min(config.chunk_queries, len(queries))
    
    results = ppo.evaluate_with_corruptions(
        queries=queries,
        sampler=sampler,
        n_corruptions=config.n_corruptions,
        corruption_modes=tuple(config.corruption_modes),
        chunk_queries=effective_chunk,
        verbose=config.verbose,
        deterministic=True,
    )
    
    return results


def profile_cprofile(config: SimpleNamespace):
    """Profile with cProfile for CPU bottlenecks."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    print("\nSetting up components...")
    components = setup_components(device, config)
    
    n_queries = config.n_test_queries
    n_corruptions = config.n_corruptions
    n_modes = len(config.corruption_modes)
    total_candidates = n_queries * (1 + n_corruptions) * n_modes
    
    print(f"\nConfiguration:")
    print(f"  Queries: {n_queries}")
    print(f"  Corruptions: {n_corruptions}")
    print(f"  Corruption modes: {config.corruption_modes}")
    print(f"  Total candidates: {total_candidates}")
    
    # Warmup
    print("\nRunning warmup (compilation)...")
    ppo, warmup_time = create_ppo_and_warmup(components, config)
    print(f"Warmup time: {warmup_time:.2f}s")
    
    # Profile the evaluation
    print(f"\nProfiling evaluation...")
    profiler = cProfile.Profile()
    profiler.enable()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time()
    
    results = run_evaluation(ppo, components, config)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    runtime = time() - start_time
    
    profiler.disable()
    
    total_time = warmup_time + runtime
    ms_per_query = (runtime / n_queries) * 1000
    ms_per_candidate = (runtime / total_candidates) * 1000
    
    # Print results
    print(f"\n{'='*80}")
    print("TIMING SUMMARY")
    print(f"{'='*80}")
    print(f"Warmup time:      {warmup_time:.4f}s")
    print(f"Runtime:          {runtime:.4f}s")
    print(f"Total time:       {total_time:.4f}s")
    print(f"ms/query:         {ms_per_query:.3f}")
    print(f"ms/candidate:     {ms_per_candidate:.3f}")
    print(f"")
    print(f"MRR:              {results.get('MRR', 0.0):.4f}")
    print(f"Hits@1:           {results.get('Hits@1', 0.0):.4f}")
    print(f"Hits@10:          {results.get('Hits@10', 0.0):.4f}")
    
    # Profile stats
    n_functions = 40
    
    print(f"\n{'='*80}")
    print("PROFILING RESULTS - Top by Cumulative Time")
    print(f"{'='*80}")
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(n_functions)
    print(s.getvalue())
    
    print(f"\n{'='*80}")
    print("PROFILING RESULTS - Top by Total Time")
    print(f"{'='*80}")
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('tottime')
    ps.print_stats(n_functions)
    print(s.getvalue())
    
    # Save to file
    output_path = 'tests/profile_optimized_eval_results.txt'
    with open(output_path, 'w') as f:
        f.write(f"Profile Optimized Eval Results\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Dataset: {config.dataset}\n")
        f.write(f"\n")
        f.write(f"Configuration:\n")
        f.write(f"  Queries: {n_queries}\n")
        f.write(f"  Corruptions: {n_corruptions}\n")
        f.write(f"  Corruption modes: {config.corruption_modes}\n")
        f.write(f"  Total candidates: {total_candidates}\n")
        f.write(f"  Compile mode: {config.compile_mode}\n")
        f.write(f"  Fullgraph: {config.fullgraph}\n")
        f.write(f"\n")
        f.write(f"{'='*80}\n")
        f.write(f"TIMING SUMMARY\n")
        f.write(f"{'='*80}\n")
        f.write(f"Warmup time:      {warmup_time:.4f}s\n")
        f.write(f"Runtime:          {runtime:.4f}s\n")
        f.write(f"Total time:       {total_time:.4f}s\n")
        f.write(f"ms/query:         {ms_per_query:.3f}\n")
        f.write(f"ms/candidate:     {ms_per_candidate:.3f}\n")
        f.write(f"\n")
        f.write(f"MRR:              {results.get('MRR', 0.0):.4f}\n")
        f.write(f"Hits@1:           {results.get('Hits@1', 0.0):.4f}\n")
        f.write(f"Hits@10:          {results.get('Hits@10', 0.0):.4f}\n")
        f.write(f"\n")
        
        f.write(f"{'='*80}\n")
        f.write(f"Top by Cumulative Time\n")
        f.write(f"{'='*80}\n")
        ps = pstats.Stats(profiler, stream=f)
        ps.strip_dirs()
        ps.sort_stats('cumulative')
        ps.print_stats(n_functions)
        
        f.write(f"\n\n{'='*80}\n")
        f.write(f"Top by Total Time\n")
        f.write(f"{'='*80}\n")
        ps.sort_stats('tottime')
        ps.print_stats(n_functions)
    
    print(f"\nResults saved to {output_path}")


def profile_gpu(config: SimpleNamespace):
    """Profile with torch.profiler for GPU analysis."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        print("CUDA not available, cannot profile GPU")
        return
    
    if profile is None:
        print("torch.profiler not available")
        return
    
    print(f"Device: {device}")
    
    print("\nSetting up components...")
    components = setup_components(device, config)
    
    n_queries = config.n_test_queries
    n_corruptions = config.n_corruptions
    n_modes = len(config.corruption_modes)
    total_candidates = n_queries * (1 + n_corruptions) * n_modes
    
    # Warmup
    print("\nRunning warmup (compilation)...")
    ppo, warmup_time = create_ppo_and_warmup(components, config)
    print(f"Warmup time: {warmup_time:.2f}s")
    
    # Profile with torch.profiler
    print("\nGPU Profiling evaluation...")
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        torch.cuda.synchronize()
        start_time = time()
        results = run_evaluation(ppo, components, config)
        torch.cuda.synchronize()
        runtime = time() - start_time
    
    total_time = warmup_time + runtime
    ms_per_query = (runtime / n_queries) * 1000
    ms_per_candidate = (runtime / total_candidates) * 1000
    
    print(f"\n{'='*80}")
    print("TIMING SUMMARY")
    print(f"{'='*80}")
    print(f"Warmup time:      {warmup_time:.4f}s")
    print(f"Runtime:          {runtime:.4f}s")
    print(f"Total time:       {total_time:.4f}s")
    print(f"ms/query:         {ms_per_query:.3f}")
    print(f"ms/candidate:     {ms_per_candidate:.3f}")
    print(f"")
    print(f"MRR:              {results.get('MRR', 0.0):.4f}")
    
    print(f"\n{'='*80}")
    print("GPU PROFILING - Top CUDA Time")
    print(f"{'='*80}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))
    
    print(f"\n{'='*80}")
    print("GPU PROFILING - Top CPU Time")
    print(f"{'='*80}")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=40))
    
    # Save to file
    output_path = 'tests/profile_optimized_eval_gpu_results.txt'
    with open(output_path, 'w') as f:
        f.write(f"Profile Optimized Eval GPU Results\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n")
        f.write(f"\n")
        f.write(f"Warmup time:      {warmup_time:.4f}s\n")
        f.write(f"Runtime:          {runtime:.4f}s\n")
        f.write(f"Total time:       {total_time:.4f}s\n")
        f.write(f"ms/query:         {ms_per_query:.3f}\n")
        f.write(f"ms/candidate:     {ms_per_candidate:.3f}\n")
        f.write(f"MRR:              {results.get('MRR', 0.0):.4f}\n")
        f.write(f"\n")
        
        f.write(f"{'='*80}\n")
        f.write(f"Top CUDA Time\n")
        f.write(f"{'='*80}\n")
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))
        
        f.write(f"\n\n{'='*80}\n")
        f.write(f"Top CPU Time\n")
        f.write(f"{'='*80}\n")
        f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=40))
    
    print(f"\nResults saved to {output_path}")


def profile_memory(config: SimpleNamespace):
    """Profile memory usage with tracemalloc."""
    import tracemalloc
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    print("\nSetting up components...")
    components = setup_components(device, config)
    
    # Warmup
    print("\nRunning warmup...")
    ppo, warmup_time = create_ppo_and_warmup(components, config)
    print(f"Warmup time: {warmup_time:.2f}s")
    
    # Profile memory
    print("\nProfiling memory...")
    tracemalloc.start()
    
    start_time = time()
    results = run_evaluation(ppo, components, config)
    runtime = time() - start_time
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Get top allocations
    snapshot = tracemalloc.take_snapshot() if hasattr(tracemalloc, 'take_snapshot') else None
    
    print(f"\n{'='*80}")
    print("MEMORY SUMMARY")
    print(f"{'='*80}")
    print(f"Current memory: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory:    {peak / 1024 / 1024:.2f} MB")
    print(f"Runtime:        {runtime:.4f}s")
    
    if torch.cuda.is_available():
        print(f"\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"  Max Alloc: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    # Save to file
    output_path = 'tests/profile_optimized_eval_memory_results.txt'
    with open(output_path, 'w') as f:
        f.write(f"Profile Optimized Eval Memory Results\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n")
        f.write(f"Current memory: {current / 1024 / 1024:.2f} MB\n")
        f.write(f"Peak memory:    {peak / 1024 / 1024:.2f} MB\n")
        f.write(f"Runtime:        {runtime:.4f}s\n")
        
        if torch.cuda.is_available():
            f.write(f"\nGPU Memory:\n")
            f.write(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB\n")
            f.write(f"  Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB\n")
            f.write(f"  Max Alloc: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB\n")
    
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Profile PPOOptimized.evaluate_with_corruptions()')
    
    # Profiler selection
    parser.add_argument('--use-gpu-profiler', action='store_true',
                        help='Use torch.profiler for GPU profiling')
    parser.add_argument('--use-memory-profiler', action='store_true',
                        help='Use tracemalloc for memory profiling')
    
    # Dataset/eval config
    parser.add_argument('--dataset', type=str, default='family')
    parser.add_argument('--n-test-queries', type=int, default=100)
    parser.add_argument('--n-corruptions', type=int, default=100)
    parser.add_argument('--corruption-modes', type=str, nargs='+', default=['both'])
    parser.add_argument('--chunk-queries', type=int, default=256)
    parser.add_argument('--batch-size-env', type=int, default=256)
    
    # Compile settings
    parser.add_argument('--compile', default=True, type=lambda x: x.lower() != 'false')
    parser.add_argument('--compile-mode', type=str, default='reduce-overhead')
    parser.add_argument('--fullgraph', default=True, type=lambda x: x.lower() != 'false')
    parser.add_argument('--vram-gb', type=float, default=6.0)
    
    # Other
    parser.add_argument('--verbose', default=False, type=lambda x: x.lower() in ('true', '1', 'yes'))
    
    args = parser.parse_args()
    
    config = SimpleNamespace(
        dataset=args.dataset,
        data_path='./data/',
        n_test_queries=args.n_test_queries,
        n_corruptions=args.n_corruptions,
        corruption_modes=args.corruption_modes,
        chunk_queries=args.chunk_queries,
        batch_size_env=args.batch_size_env,
        padding_atoms=6,
        padding_states=120,
        max_depth=20,
        end_proof_action=True,
        max_total_vars=100,
        atom_embedding_size=250,
        seed=42,
        compile=args.compile,
        compile_mode=args.compile_mode,
        fullgraph=args.fullgraph,
        vram_gb=args.vram_gb,
        verbose=args.verbose,
    )
    
    if args.use_gpu_profiler:
        profile_gpu(config)
    elif args.use_memory_profiler:
        profile_memory(config)
    else:
        profile_cprofile(config)


if __name__ == '__main__':
    main()
