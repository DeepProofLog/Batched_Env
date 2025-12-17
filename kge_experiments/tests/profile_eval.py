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


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    def isatty(self):
        return any(getattr(f, 'isatty', lambda: False)() for f in self.files)




def compute_optimal_batch_size(
    chunk_queries: int = None,
    n_corruptions: int = None,
    max_vram_gb: float = None,
    min_batch_size: int = 64,
    prefer_power_of_two: bool = False,
) -> int:
    """
    Compute optimal batch size for evaluation.
    
    Uses adaptive batch size: smaller for small evaluations, larger for large ones.
    This optimizes compilation time for small tests while supporting large-scale ranking.
    
    Memory model:
    - Each query in a batch requires ~3-4MB for state tensors:
      - derived_states: [K_max=120, M_max=26, 3] × 8 bytes ≈ 75KB per query
      - Multiple intermediate tensors during unification: ~3MB per query
    - For 8GB GPU with ~6GB usable: max ~2000 queries safely, ~1500 conservatively
    
    CRITICAL: Batch sizes > 512 have pathologically worse per-query performance
    due to GPU memory bandwidth saturation with large derived_states tensors.
    
    Args:
        chunk_queries: Number of queries per chunk (used for sizing)
        n_corruptions: Number of corruptions per query (used for sizing)
        max_vram_gb: Maximum VRAM to use (default: detected from GPU)
        min_batch_size: Minimum allowed batch size
        prefer_power_of_two: If True, round to nearest power of 2
        
    Returns:
        Batch size (adaptive based on actual needs)
    """
    # Detect available VRAM if not specified
    if max_vram_gb is None and torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        # Use ~60% of total memory to leave room for PyTorch overhead
        max_vram_gb = total_mem * 0.6
    elif max_vram_gb is None:
        max_vram_gb = 4.0  # Conservative default for CPU
    
    # Adaptive: use smaller batch for small evaluations
    # This speeds up compilation for small tests
    # NOTE: Batch sizes > 512 tend to have much worse per-query performance
    # due to GPU memory bandwidth limits with large derived_states tensors
    if chunk_queries is not None and n_corruptions is not None:
        actual_need = chunk_queries * (1 + n_corruptions)
        if actual_need <= 64:
            batch_size = 64
        elif actual_need <= 256:
            batch_size = 256
        else:
            # Cap at 512 for best throughput (larger sizes have worse per-query time)
            batch_size = 512
    else:
        # Default to moderate size if params not provided
        batch_size = 256
    
    # Clamp to VRAM limit - use more accurate memory estimation
    # Each query uses approximately 3MB during unification
    mem_per_query_mb = 3.0
    max_batch_from_vram = int(max_vram_gb * 1024 / mem_per_query_mb)
    batch_size = min(batch_size, max_batch_from_vram)
    
    # Apply minimum
    batch_size = max(batch_size, min_batch_size)
    
    # Align to multiples of 32 for GPU efficiency
    batch_size = ((batch_size + 31) // 32) * 32
    
    # Optional: round to power of 2
    if prefer_power_of_two:
        import math
        log2 = math.log2(batch_size)
        batch_size = 2 ** round(log2)
    
    return batch_size

def setup_components(device: torch.device, config: SimpleNamespace):
    """
    Initialize all components needed for optimized evaluation.
    
    Returns:
        Dict with all components needed for evaluation.
    """
    from data_handler import DataHandler
    from index_manager import IndexManager
    from unification import UnificationEngineVectorized
    from nn.embeddings import EmbedderLearnable as TensorEmbedder
    from policy import ActorCriticPolicy as TensorPolicy
    from nn.sampler import Sampler
    from env import EnvVec
    
    # Enable compile mode
    import unification
    unification.COMPILE_MODE = config.compile
    
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
    
    # Vectorized engine
    vec_engine = UnificationEngineVectorized.from_index_manager(
        im,
        padding_atoms=config.padding_atoms,
        max_derived_per_state=config.padding_states,
        end_proof_action=config.end_proof_action,
    )
    
    # Clean up index manager tensors
    im.facts_idx = None
    im.rules_idx = None
    im.rule_lens = None
    im.rules_heads_idx = None
    
    # Convert queries
    def convert_queries_unpadded(queries):
        tensors = []
        for q in queries:
            atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
            tensors.append(atom)
        return torch.stack(tensors, dim=0)
    
    test_queries = convert_queries_unpadded(dh.test_queries)
    train_queries = convert_queries_unpadded(dh.train_queries)
    
    # Compiled environment
    eval_env = EnvVec(
        vec_engine=vec_engine,
        batch_size=config.batch_size_env,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_depth,
        end_proof_action=config.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
        memory_pruning=True,
        train_queries=train_queries,
        valid_queries=test_queries,
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
    from ppo import PPO
    
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
    
    # Create PPOOptimized in eval-only mode (skips massive rollout buffer allocation)
    ppo = PPO(
        policy=policy,
        env=env,
        device=env.device,
        fixed_batch_size=batch_size,
        verbose=False,
        eval_only=True,  # Skip rollout buffer allocation - saves ~8-16GB VRAM
    )
    
    # Warmup and compile
    warmup_start = time()
    
    # Compile env step function
    env.compile(
        mode=config.compile_mode,
        fullgraph=config.fullgraph,
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
    # Output path relative to this script
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profile_eval_results.txt')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Setup tee to write to both stdout/stderr and file
    log_file = open(output_path, 'w')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = sys.stdout
    
    try:
        initial_wallclock = time()  # Measure from very start
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Profile Optimized Eval Results")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {device}")
        print(f"Dataset: {config.dataset}")
        
        print(f"\nConfiguration:")
        print(f"  Queries: {config.n_test_queries}")
        print(f"  Corruptions: {config.n_corruptions}")
        print(f"  Corruption modes: {config.corruption_modes}")
        print(f"  Compile mode: {config.compile_mode}")
        print(f"  Fullgraph: {config.fullgraph}")
        
        print("\nSetting up components...")
        components = setup_components(device, config)
        
        n_queries = config.n_test_queries
        n_corruptions = config.n_corruptions
        n_modes = len(config.corruption_modes)
        total_candidates = n_queries * (1 + n_corruptions) * n_modes
        
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
        
        # Final wallclock measurement
        final_wallclock = time()
        total_wallclock = final_wallclock - initial_wallclock
        print(f"\n{'='*80}")
        print("WALLCLOCK SUMMARY")
        print(f"{'='*80}")
        print(f"Initial wallclock: {initial_wallclock:.4f}")
        print(f"Final wallclock:   {final_wallclock:.4f}")
        print(f"Total wallclock:   {total_wallclock:.4f}s")
        
        print(f"\nResults saved to {output_path}")

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

def profile_gpu(config: SimpleNamespace):
    """Profile with torch.profiler for GPU analysis."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        print("CUDA not available, cannot profile GPU")
        return
    
    if profile is None:
        print("torch.profiler not available")
        return
        
    # Output path relative to this script
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profile_eval_results.txt')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Setup tee to write to both stdout/stderr and file
    log_file = open(output_path, 'w')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = sys.stdout
    
    try:
        print(f"Profile Optimized Eval GPU Results")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
        
        print(f"\nResults saved to {output_path}")

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

def profile_memory(config: SimpleNamespace):
    """Profile memory usage with tracemalloc."""
    import tracemalloc
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_path = './profile_eval_memory_results.txt'
    
    # Setup tee to write to both stdout/stderr and file
    log_file = open(output_path, 'w')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = sys.stdout
    
    try:
        print(f"Profile Optimized Eval Memory Results")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
        
        print(f"\nResults saved to {output_path}")

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


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
        data_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'),
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
