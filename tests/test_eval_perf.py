"""
Performance Benchmark for Optimized Evaluation.

This script benchmarks the performance difference between:
1. Original Evaluation (Python/Stack-based)
2. Eager Optimized (Vectorized, No Compile)
3. Compiled Optimized (Vectorized + Compile)

It reproduces the "Performance Summary" table from docs/eval_optimization.md.

Usage:
    python tests/test_eval_perf.py --dataset family --modes compiled --warmup-only
    python tests/test_eval_perf.py --dataset family --modes original eager compiled --n-test-queries 100
    python tests/test_eval_perf.py --check-compile  # Check for graph breaks
"""

import os
import sys
import argparse
import time
import torch
import torch._dynamo as dynamo
import numpy as np
from types import SimpleNamespace
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



def setup_components(device: torch.device, config: SimpleNamespace):
    """Initialize all components needed for evaluation."""
    from data_handler import DataHandler
    from index_manager import IndexManager
    from unification import UnificationEngine, set_compile_mode
    from unification_vectorized import UnificationEngineVectorized
    from env import BatchedEnv
    from embeddings import EmbedderLearnable as TensorEmbedder
    from model import ActorCriticPolicy as TensorPolicy
    from sampler import Sampler
    
    # Enable compile mode if requested (global flag for unification module)
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
    
    # Stringifier params
    stringifier_params = {
        'verbose': 0,
        'idx2predicate': im.idx2predicate,
        'idx2constant': im.idx2constant,
        'idx2template_var': im.idx2template_var,
        'padding_idx': im.padding_idx,
        'n_constants': im.constant_no
    }
    
    # Base unification engine
    base_engine = UnificationEngine.from_index_manager(
        im, take_ownership=True,
        stringifier_params=stringifier_params,
        end_pred_idx=im.end_pred_idx,
        end_proof_action=config.end_proof_action,
        max_derived_per_state=config.padding_states,
    )
    base_engine.index_manager = im
    
    # Vectorized engine (for compiled path)
    # Use None for auto-computation from data - this avoids OOM by not over-allocating
    vec_engine = UnificationEngineVectorized.from_base_engine(
        base_engine,
        max_fact_pairs=None,  # Auto-compute from data
        max_rule_pairs=None,  # Auto-compute from data
        padding_atoms=config.padding_atoms,  # Use config value for M_max alignment with original
    )
    
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
    
    # Clamp batch_size_env to available test queries
    actual_batch_size = min(config.batch_size_env, len(dh.test_queries))
    
    # Eval environment (Original)
    eval_env_orig = BatchedEnv(
        batch_size=actual_batch_size,
        queries=test_queries_padded[:actual_batch_size],
        labels=torch.ones(actual_batch_size, dtype=torch.long, device=device),
        query_depths=torch.ones(actual_batch_size, dtype=torch.long, device=device),
        unification_engine=base_engine,
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

    # Compiled Environment (for Optimized/Compiled paths)
    from env_optimized import EvalEnvOptimized
    eval_env_compiled = EvalEnvOptimized(
        vec_engine=vec_engine,
        batch_size=actual_batch_size,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_depth,
        end_proof_action=config.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
        memory_pruning=True,  # Cycle detection - verified working
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
        'eval_env_orig': eval_env_orig,
        'eval_env_compiled': eval_env_compiled,
        'sampler': sampler,
        'dh': dh,
        'im': im,
        'vec_engine': vec_engine,
        'test_queries': test_queries_unpadded,
    }


def create_and_warmup_optimized_evaluator(components, config):
    """Create fast evaluator (single-step compilation) and run warmup.
    
    Uses the new pattern: PPOOptimized with env.compile(policy).
    Returns (ppo, warmup_time_s) for compatibility with the test interface.
    """
    from ppo_optimized import PPOOptimized, compute_optimal_batch_size
    from env_optimized import EvalEnvOptimized
    
    actor = components['policy']
    
    # Create optimized environment
    vec_engine = components['vec_engine']
    im = components['im']
    
    env_fast = EvalEnvOptimized(
        vec_engine=vec_engine,
        batch_size=config.batch_size_env,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_depth,
        end_proof_action=config.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=components['eval_env_compiled'].device,
        memory_pruning=True,
    )
    
    queries = components['test_queries'][:config.n_test_queries].to(env_fast.device)
    
    # Compute batch size
    effective_chunk_queries = min(int(config.chunk_queries), int(queries.shape[0]))
    batch_size = compute_optimal_batch_size(
        chunk_queries=effective_chunk_queries,
        n_corruptions=config.n_corruptions,
        max_vram_gb=config.vram_gb,
    )
    
    compile_mode = getattr(config, 'compile_mode', 'default')
    fullgraph = getattr(config, 'fullgraph', True)
    
    # Create PPOOptimized with fixed_batch_size
    ppo = PPOOptimized(
        policy=actor,
        env=env_fast,
        device=env_fast.device,
        fixed_batch_size=batch_size,
        verbose=False,
    )
    
    # Compile environment with policy
    print(f"Starting fast warmup (batch_size={batch_size})...")
    warmup_start = time.time()
    
    env_fast.compile(
        policy=actor,
        deterministic=True,
        mode=compile_mode,
        fullgraph=fullgraph,
        include_value=False,  # Eval mode doesn't need values
    )
    
    # Warmup with the exact batch_size to trigger JIT/CUDA graph compilation
    warmup_queries = queries[:1].expand(batch_size, -1)  # [batch_size, 3] - exact size
    _ = ppo.evaluate_policy(warmup_queries, deterministic=True)
    
    warmup_time_s = time.time() - warmup_start
    print(f"Fast warmup took {warmup_time_s} s.")
    
    # Store for later use
    evaluator_like = SimpleNamespace(
        ppo=ppo,
        env=env_fast,
        warmup_time_s=warmup_time_s,
        fixed_batch_size=batch_size,
    )
    
    return evaluator_like, warmup_time_s


def run_optimized_eval(components, config, seed: Optional[int] = None, 
                       evaluator=None, return_evaluator: bool = False,
                       mode: str = 'compiled'):
    """Run optimized eval (eager or compiled).
    
    Uses the new pattern: PPOOptimized.evaluate_with_corruptions().
    
    Args:
        mode: 'eager' or 'compiled'.
    """
    from ppo_optimized import PPOOptimized, compute_optimal_batch_size
    
    actor = components['policy']
    env = components['eval_env_compiled']
    sampler = components['sampler']
    queries = components['test_queries'][:config.n_test_queries].to(env.device)
    
    if seed is not None:
        sampler.rng = np.random.RandomState(seed)
    
    warmup_time_s = 0.0
    if evaluator is None:
        # Create and warmup using the new pattern
        compile_mode = getattr(config, 'compile_mode', 'default')
        fullgraph = getattr(config, 'fullgraph', True) if mode == 'compiled' else False
        
        # Compute batch size
        effective_chunk_queries = min(int(config.chunk_queries), int(queries.shape[0]))
        batch_size = compute_optimal_batch_size(
            chunk_queries=effective_chunk_queries,
            n_corruptions=config.n_corruptions,
        )
        
        # Create PPOOptimized
        ppo = PPOOptimized(
            policy=actor,
            env=env,
            device=env.device,
            fixed_batch_size=batch_size,
            verbose=False,
        )
        
        warmup_start = time.time()
        env.compile(
            policy=actor,
            deterministic=True,
            mode=compile_mode,
            fullgraph=fullgraph,
            include_value=False,
        )
        # Warmup with exact batch_size
        warmup_queries = queries[:1].expand(batch_size, -1)  # [batch_size, 3]
        _ = ppo.evaluate_policy(warmup_queries, deterministic=True)
        warmup_time_s = time.time() - warmup_start
        
        evaluator = SimpleNamespace(ppo=ppo, env=env, warmup_time_s=warmup_time_s)

    # Compute optimal batch size (needed for chunking in eval loop)
    effective_chunk_queries = min(int(config.chunk_queries), int(queries.shape[0]))

    # Run evaluation using PPOOptimized.evaluate_with_corruptions
    results = evaluator.ppo.evaluate_with_corruptions(
        queries=queries,
        sampler=sampler,
        n_corruptions=config.n_corruptions,
        corruption_modes=tuple(config.corruption_modes),
        chunk_queries=effective_chunk_queries,
        verbose=config.verbose if seed is None else False,
        deterministic=True,
    )
    
    if return_evaluator:
        return results, evaluator, warmup_time_s
    return results



def run_original_eval(components, config, seed: Optional[int] = None):
    """Run original eval_corruptions."""
    from model_eval import eval_corruptions
    
    actor = components['policy']
    env = components['eval_env_orig']
    sampler = components['sampler']
    queries = components['test_queries'][:config.n_test_queries]
    
    # Reseed if requested
    if seed is not None:
        sampler.rng = np.random.RandomState(seed)
    
    return eval_corruptions(
        actor=actor,
        env=env,
        queries=queries,
        sampler=sampler,
        n_corruptions=config.n_corruptions,
        corruption_modes=tuple(config.corruption_modes),
        verbose=config.verbose if seed is None else False,
    )


def check_graph_breaks(vec_engine, device, config):
    """Check for graph breaks in the compiled function."""
    print("\n" + "="*60)
    print("CHECKING FOR GRAPH BREAKS")
    print("="*60)
    
    # Create sample inputs
    B = 10
    A = config.padding_atoms
    
    current_states = torch.zeros(B, A, 3, dtype=torch.long, device=device)
    current_states[:, 0, :] = 1  # Some dummy values
    next_var_indices = torch.full((B,), 1000, dtype=torch.long, device=device)
    excluded = torch.zeros(B, 1, 3, dtype=torch.long, device=device)
    
    # Explain the function
    try:
        explanation = dynamo.explain(vec_engine.get_derived_states_compiled)(
            current_states, next_var_indices, excluded
        )
        
        print(f"\nGraph Count: {explanation.graph_count}")
        print(f"Graph Break Count: {explanation.graph_break_count}")
        
        if explanation.break_reasons:
            print("\nBreak Reasons:")
            for i, reason in enumerate(explanation.break_reasons):
                print(f"  {i+1}. {reason}")
        else:
            print("\nNo graph breaks! ✓")
        
        return explanation.graph_break_count == 0
        
    except Exception as e:
        print(f"Error during explain: {e}")
        return False

def run_performance_test(components, config, modes, warmup_only):
    """
    Run performance benchmark for specified modes.
    
    Args:
        modes: List of modes to test ('original', 'eager', 'compiled')
        warmup_only: If True, only measure warmup/compile time (for compiled).
                     Evaluation will run on a minimal set of queries.
    """
    print("\n" + "="*60)
    print(f"PERFORMANCE BENCHMARK (Warmup Only: {warmup_only})")
    print(f"Modes: {modes}")
    print(f"Config: {config.n_test_queries} queries, {config.n_corruptions} corruptions")
    print("="*60)
    
    # Calculate total candidates for the header
    # Each query has 1 (original) + n_corruptions candidates, times number of modes
    n_modes = len(config.corruption_modes)
    total_candidates = config.n_test_queries * (1 + config.n_corruptions) * n_modes
    print(f"Total candidates: {total_candidates} ({config.n_test_queries} queries × {1 + config.n_corruptions} candidates × {n_modes} modes)")
    
    # Define table header
    print(f"\n{'Mode':<20} {'Compile (s)':>12} {'Runtime (s)':>12} {'ms/query':>12} {'ms/candidate':>14} {'Total (s)':>12}")
    print("-" * 95)
    
    for mode in modes:
        # Prepare run config
        run_config = SimpleNamespace(**vars(config))
        run_config.verbose = False
        
        if warmup_only:
            run_config.n_test_queries = min(config.chunk_queries, config.n_test_queries)
            
        warmup_s = 0.0
        eval_ms_per_q = 0.0
        total_s = 0.0
        
        try:
            start_t = time.time()
            
            if mode == 'original':
                if warmup_only:
                    # Original has no warmup
                    pass
                else:
                    print(f"  Running {mode} eval...", flush=True)
                    run_original_eval(components, run_config)
                    
            elif mode == 'eager':
                if warmup_only:
                    # Just run warmup for eager - use the new pattern
                    from ppo_optimized import PPOOptimized, compute_optimal_batch_size
                    env = components['eval_env_compiled']
                    actor = components['policy']
                    
                    batch_size = compute_optimal_batch_size(
                        chunk_queries=config.chunk_queries,
                        n_corruptions=config.n_corruptions,
                    )
                    ppo = PPOOptimized(
                        policy=actor, env=env, device=env.device,
                        fixed_batch_size=batch_size, verbose=False,
                    )
                    
                    warmup_start = time.time()
                    env.compile(policy=actor, deterministic=True, mode='default', fullgraph=False, include_value=False)
                    warmup_queries = components['test_queries'][:1].to(env.device).expand(batch_size, -1)
                    _ = ppo.evaluate_policy(warmup_queries, deterministic=True)
                    warmup_s = time.time() - warmup_start
                    print(f"  Warmup finished in {warmup_s:.4f}s", flush=True)
                else:
                    print(f"  Running {mode} warmup...", flush=True)
                    # Run with return_evaluator to get warmup time (same pattern as compiled)
                    _, _, start_warmup_s = run_optimized_eval(
                        components, run_config, return_evaluator=True, mode='eager'
                    )
                    warmup_s = start_warmup_s
                    print(f"  Warmup finished in {warmup_s:.4f}s", flush=True)
                    print(f"  Running {mode} eval...", flush=True)

            elif mode == 'compiled':
                if warmup_only:
                    # Just run warmup
                    evaluator, warmup_s = create_and_warmup_optimized_evaluator(components, run_config)
                    print(f"  Warmup finished in {warmup_s:.4f}s", flush=True)
                else:
                    print(f"  Running {mode} warmup...", flush=True)
                    # First: warmup only
                    evaluator, warmup_s = create_and_warmup_optimized_evaluator(components, run_config)
                    print(f"  Warmup finished in {warmup_s:.4f}s", flush=True)
                    
                    # Second: timed evaluation run (separate from warmup)
                    print(f"  Running {mode} eval...", flush=True)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    eval_start = time.time()
                    effective_chunk_queries = min(int(run_config.chunk_queries), int(run_config.n_test_queries))
                    queries = components['test_queries'][:run_config.n_test_queries].to(evaluator.env.device)
                    # Use PPOOptimized.evaluate_with_corruptions
                    _ = evaluator.ppo.evaluate_with_corruptions(
                        queries=queries,
                        sampler=components['sampler'],
                        n_corruptions=run_config.n_corruptions,
                        corruption_modes=tuple(run_config.corruption_modes),
                        chunk_queries=effective_chunk_queries,
                        verbose=False,
                        deterministic=True,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    eval_s = time.time() - eval_start
                    # Override total_s calculation below
                    total_s = warmup_s + eval_s

            # For compiled/fast mode, total_s is set above; for others, calculate from start_t
            if mode not in ('compiled') or warmup_only:
                total_s = time.time() - start_t
                eval_s = max(0.0, total_s - warmup_s)
            # else: eval_s and total_s already set for compiled mode
            
            # Calculate metrics
            n_run = run_config.n_test_queries
            n_modes_run = len(run_config.corruption_modes)
            total_candidates_run = n_run * (1 + run_config.n_corruptions) * n_modes_run
            
            if n_run > 0:
                eval_ms_per_q = (eval_s / n_run) * 1000.0
                eval_ms_per_candidate = (eval_s / total_candidates_run) * 1000.0
            else:
                eval_ms_per_q = 0.0
                eval_ms_per_candidate = 0.0
            
            print(f"  Eval finished in {eval_s:.4f}s", flush=True)
            
            # Display summary row
            result_line = f"{mode:<20} {warmup_s:>12.4f} {eval_s:>12.4f} {eval_ms_per_q:>12.2f} {eval_ms_per_candidate:>14.3f} {total_s:>12.4f}"
            print(result_line)
            
            # Append result to file
            with open("test_eval_perf.txt", "a") as f:
                f.write(result_line + "\n")
            
        except Exception as e:
            print(f"{mode:<20} {'FAILED':>12} {'FAILED':>12} {'FAILED':>12} {'FAILED':>14} {str(e):>12}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Test Evaluation Performance')
    parser.add_argument('--dataset', type=str, default='family', help='Dataset name')
    parser.add_argument('--n-test-queries', type=int, default=100, help='Number of test queries')
    parser.add_argument('--n-corruptions', type=int, default=100)
    parser.add_argument('--corruption-modes', type=str, nargs='+', default=['both'])

    parser.add_argument('--chunk-queries', type=int, default=100)
    parser.add_argument('--batch-size-env', type=int, default=100)
    
    parser.add_argument('--warmup-only', action='store_true', help='Only measure warmup/compile time')
    parser.add_argument('--check-compile', action='store_true', help='Check for graph breaks')
    
    parser.add_argument('--modes', nargs='+', default=['compiled'], 
                       choices=['original', 'eager', 'compiled'],
                       help='Modes to benchmark. compiled=single-step compile (faster warmup).')
    parser.add_argument('--compile-mode', type=str, default='reduce-overhead',
                       help='torch.compile mode: default, reduce-overhead, max-autotune')
    parser.add_argument('--vram-gb', type=float, default=6.0, help='Available VRAM budget in GB')
    parser.add_argument('--fixed-batch-size', type=int, default=None,
                       help='Fixed batch size for compilation. If None, uses adaptive sizing (smaller for fewer queries = faster compile).')
    

    args = parser.parse_args()
    
    # Calculate total candidates for file header
    n_modes = len(args.corruption_modes)
    total_candidates = args.n_test_queries * (1 + args.n_corruptions) * n_modes
    
    # Initialize result file
    with open("test_eval_perf.txt", "w") as f:
        f.write("Performance Benchmark Results\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Queries: {args.n_test_queries}, Corruptions: {args.n_corruptions}, Modes: {args.corruption_modes}\n")
        f.write(f"Total candidates: {total_candidates}\n")
        f.write(f"Compile mode: {args.compile_mode}\n")
        f.write("="*95 + "\n")
        f.write(f"{'Mode':<20} {'Compile (s)':>12} {'Runtime (s)':>12} {'ms/query':>12} {'ms/candidate':>14} {'Total (s)':>12}\n")
        f.write("-" * 95 + "\n")
    
    config = SimpleNamespace(
        dataset=args.dataset,
        data_path='./data/',
        n_test_queries=args.n_test_queries,
        n_corruptions=args.n_corruptions,
        chunk_queries=args.chunk_queries,
        batch_size_env=args.batch_size_env,
        corruption_modes=args.corruption_modes,
        verbose=True,
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
        seed=42,
        compile=True, # Will be ignored/overridden by mode logic in run_perf_test or run_optimized_eval wrapper? 
                      # Actually, run_optimized_eval handles enable/disable of compilation based on mode.
                      # Ideally we set this to True generally so components like Unification are ready for compilation if needed.
        compile_mode=args.compile_mode,
        fullgraph=True, # Default to True, but run_optimized_eval controls usage
        fixed_batch_size=args.fixed_batch_size,  # Fixed batch size to prevent recompilation
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\nSetting up components...")
    components = setup_components(device, config)
    
    if args.check_compile:
        check_graph_breaks(components['vec_engine'], device, config)
    else:
        run_performance_test(components, config, args.modes, args.warmup_only)

if __name__ == '__main__':
    main()
