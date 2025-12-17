"""
Benchmark: Fused vs Separate Compilation

This benchmark measures the speed difference between:
1. FUSED: Step + Policy compiled together (current approach in env.py)
2. SEPARATE: Policy compiled separately, step compiled separately

Measures:
- Warmup time (compilation)
- Throughput (steps/second)
- Latency per step (μs)

Usage:
    python tests/benchmark_compilation.py
    python tests/benchmark_compilation.py --n-steps 1000 --batch-size 256
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from time import time
from types import SimpleNamespace
from typing import Tuple
import gc

import torch
import torch.nn as nn


def setup_components(device: torch.device, config: SimpleNamespace) -> dict:
    """Initialize all components needed for benchmarking."""
    from data_handler import DataHandler
    from index_manager import IndexManager
    from unification import UnificationEngineVectorized
    from nn.embeddings import EmbedderLearnable as TensorEmbedder
    from policy import ActorCriticPolicy as TensorPolicy
    from env import EnvVec as EnvVec
    
    # Enable compile mode
    import unification
    unification.COMPILE_MODE = True
    
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
    
    # Vectorized engine
    vec_engine = UnificationEngineVectorized.from_index_manager(
        im,
        padding_atoms=config.padding_atoms,
        max_derived_per_state=config.padding_states,
        end_proof_action=config.end_proof_action,
    )
    
    # Convert queries
    def convert_queries_unpadded(queries):
        tensors = []
        for q in queries:
            atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
            tensors.append(atom)
        return torch.stack(tensors, dim=0)
    
    test_queries = convert_queries_unpadded(dh.test_queries)
    
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
        compile_policy=False,  # We'll compile manually
    ).to(device)
    
    return {
        'policy': policy,
        'vec_engine': vec_engine,
        'test_queries': test_queries,
        'im': im,
        'dh': dh,
    }


def run_fused_benchmark(components: dict, config: SimpleNamespace) -> dict:
    """Benchmark the FUSED approach (step+policy compiled together)."""
    from env import EnvVec
    
    device = config.device
    policy = components['policy']
    vec_engine = components['vec_engine']
    queries = components['test_queries'][:config.batch_size].to(device)
    im = components['im']
    
    # Create fresh environment
    env = EnvVec(
        vec_engine=vec_engine,
        batch_size=config.batch_size,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_depth,
        end_proof_action=config.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
        memory_pruning=True,
    )
    
    # Warmup (compilation)
    print("\n=== FUSED: Compiling step+policy together ===")
    torch.cuda.synchronize() if device.type == 'cuda' else None
    warmup_start = time()
    
    env.compile(
        policy=policy,
        deterministic=True,
        mode=config.compile_mode,
        fullgraph=True,
        include_value=False,
    )
    
    # Initialize
    env.set_queries(queries)
    obs, state = env.reset()
    
    # Run a few warmup steps to trigger JIT
    for _ in range(3):
        result = env.step_with_policy(state, obs, deterministic=True, eval_mode=True)
        state, obs = result[0], result[1]
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    warmup_time = time() - warmup_start
    print(f"  Warmup time: {warmup_time:.2f}s")
    
    # Reset for benchmark
    obs, state = env.reset()
    
    # Benchmark
    print(f"  Running {config.n_steps} steps...")
    torch.cuda.synchronize() if device.type == 'cuda' else None
    bench_start = time()
    
    for step in range(config.n_steps):
        result = env.step_with_policy(state, obs, deterministic=True, eval_mode=True)
        state, obs = result[0], result[1]
        # Handle resets
        if state.done.any():
            obs, state = env.reset()
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    bench_time = time() - bench_start
    
    steps_per_sec = config.n_steps / bench_time
    us_per_step = (bench_time / config.n_steps) * 1e6
    
    return {
        'warmup_s': warmup_time,
        'runtime_s': bench_time,
        'steps_per_sec': steps_per_sec,
        'us_per_step': us_per_step,
    }


def run_separate_benchmark(components: dict, config: SimpleNamespace) -> dict:
    """Benchmark the SEPARATE approach (policy and step compiled independently)."""
    from env import EnvVec, EnvObs
    from policy import create_policy_logits_fn
    
    device = config.device
    policy = components['policy']
    vec_engine = components['vec_engine']
    queries = components['test_queries'][:config.batch_size].to(device)
    im = components['im']
    
    # Create fresh environment WITHOUT policy compilation
    env = EnvVec(
        vec_engine=vec_engine,
        batch_size=config.batch_size,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_depth,
        end_proof_action=config.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
        memory_pruning=True,
    )
    
    print("\n=== SEPARATE: Compiling policy and step independently ===")
    torch.cuda.synchronize() if device.type == 'cuda' else None
    warmup_start = time()
    
    # Compile policy SEPARATELY
    print("  Compiling policy...")
    policy_logits_fn = create_policy_logits_fn(policy)
    compiled_policy_fn = torch.compile(
        policy_logits_fn,
        mode=config.compile_mode,
        fullgraph=True,
        dynamic=False,
    )
    
    # Compile step_functional SEPARATELY
    print("  Compiling step_functional...")
    compiled_step_fn = torch.compile(
        env.step_functional,
        mode=config.compile_mode,
        fullgraph=True,
        dynamic=False,
    )
    
    # Initialize
    env.set_queries(queries)
    obs, state = env.reset()
    
    # Warmup both compiled functions
    for _ in range(3):
        # Policy forward
        logits = compiled_policy_fn(obs)
        masked_logits = torch.where(
            obs.action_mask,
            logits,
            torch.full_like(logits, float('-inf'))
        )
        actions = masked_logits.argmax(dim=-1)
        
        # Step forward
        step_result = compiled_step_fn(state, actions)
        state = step_result.state
        obs = step_result.obs
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    warmup_time = time() - warmup_start
    print(f"  Warmup time: {warmup_time:.2f}s")
    
    # Reset for benchmark
    obs, state = env.reset()
    
    # Benchmark
    print(f"  Running {config.n_steps} steps...")
    torch.cuda.synchronize() if device.type == 'cuda' else None
    bench_start = time()
    
    for step in range(config.n_steps):
        # Mark step begin for CUDA graphs
        torch.compiler.cudagraph_mark_step_begin()
        
        # SEPARATE: Policy forward (graph 1)
        logits = compiled_policy_fn(obs)
        
        # Clone logits to break aliasing with CUDA graph buffers
        logits = logits.clone()
        
        masked_logits = torch.where(
            obs.action_mask,
            logits,
            torch.full_like(logits, float('-inf'))
        )
        actions = masked_logits.argmax(dim=-1)
        
        # Clone actions before passing to next graph
        actions = actions.clone()
        
        # Mark step begin for the second graph
        torch.compiler.cudagraph_mark_step_begin()
        
        # SEPARATE: Step forward (graph 2)  
        step_result = compiled_step_fn(state, actions)
        
        # Clone all outputs to break aliasing with CUDA graph buffers
        # This is the overhead that fused compilation avoids!
        from env import EnvState, EnvObs
        state = EnvState(
            current_states=step_result.state.current_states.clone(),
            derived_states=step_result.state.derived_states.clone(),
            derived_counts=step_result.state.derived_counts.clone(),
            original_queries=step_result.state.original_queries.clone(),
            next_var_indices=step_result.state.next_var_indices.clone(),
            depths=step_result.state.depths.clone(),
            done=step_result.state.done.clone(),
            success=step_result.state.success.clone(),
            history_hashes=step_result.state.history_hashes.clone(),
            history_count=step_result.state.history_count.clone(),
        )
        obs = EnvObs(
            sub_index=step_result.obs.sub_index.clone(),
            derived_sub_indices=step_result.obs.derived_sub_indices.clone(),
            action_mask=step_result.obs.action_mask.clone(),
        )
        
        # Handle resets (simplified - just reset all)
        if state.done.any():
            obs, state = env.reset()
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    bench_time = time() - bench_start
    
    steps_per_sec = config.n_steps / bench_time
    us_per_step = (bench_time / config.n_steps) * 1e6
    
    return {
        'warmup_s': warmup_time,
        'runtime_s': bench_time,
        'steps_per_sec': steps_per_sec,
        'us_per_step': us_per_step,
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark fused vs separate compilation')
    parser.add_argument('--dataset', type=str, default='family')
    parser.add_argument('--n-steps', type=int, default=500, help='Steps to benchmark')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--compile-mode', type=str, default='reduce-overhead',
                        choices=['default', 'reduce-overhead', 'max-autotune'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = SimpleNamespace(
        dataset=args.dataset,
        data_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'),
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        padding_atoms=6,
        padding_states=120,
        max_depth=20,
        end_proof_action=True,
        max_total_vars=100,
        atom_embedding_size=250,
        seed=args.seed,
        compile_mode=args.compile_mode,
        device=device,
    )
    
    print("=" * 70)
    print("BENCHMARK: Fused vs Separate Compilation")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Dataset: {config.dataset}")
    print(f"Batch size: {config.batch_size}")
    print(f"Steps: {config.n_steps}")
    print(f"Compile mode: {config.compile_mode}")
    
    # Setup
    print("\nSetting up components...")
    components = setup_components(device, config)
    
    # Clear cache before each benchmark
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Run FUSED benchmark
    fused_results = run_fused_benchmark(components, config)
    
    # Clear and reset for separate benchmark
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch._dynamo.reset()  # Clear compiled cache
    
    # Recreate components with fresh random state
    torch.manual_seed(config.seed)
    components = setup_components(device, config)
    
    # Run SEPARATE benchmark
    separate_results = run_separate_benchmark(components, config)
    
    # Report results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n{'Metric':<25} {'Fused':>15} {'Separate':>15} {'Diff':>12}")
    print("-" * 70)
    
    warmup_diff = (separate_results['warmup_s'] - fused_results['warmup_s']) / fused_results['warmup_s'] * 100
    print(f"{'Warmup time (s)':<25} {fused_results['warmup_s']:>15.2f} {separate_results['warmup_s']:>15.2f} {warmup_diff:>+11.1f}%")
    
    runtime_diff = (separate_results['runtime_s'] - fused_results['runtime_s']) / fused_results['runtime_s'] * 100
    print(f"{'Runtime (s)':<25} {fused_results['runtime_s']:>15.4f} {separate_results['runtime_s']:>15.4f} {runtime_diff:>+11.1f}%")
    
    throughput_diff = (separate_results['steps_per_sec'] - fused_results['steps_per_sec']) / fused_results['steps_per_sec'] * 100
    print(f"{'Throughput (steps/s)':<25} {fused_results['steps_per_sec']:>15.1f} {separate_results['steps_per_sec']:>15.1f} {throughput_diff:>+11.1f}%")
    
    latency_diff = (separate_results['us_per_step'] - fused_results['us_per_step']) / fused_results['us_per_step'] * 100
    print(f"{'Latency (μs/step)':<25} {fused_results['us_per_step']:>15.1f} {separate_results['us_per_step']:>15.1f} {latency_diff:>+11.1f}%")
    
    print("\n" + "=" * 70)
    if latency_diff > 0:
        print(f"CONCLUSION: Separate compilation is {latency_diff:.1f}% SLOWER than fused.")
    else:
        print(f"CONCLUSION: Separate compilation is {-latency_diff:.1f}% FASTER than fused.")
    print("=" * 70)


if __name__ == '__main__':
    main()
