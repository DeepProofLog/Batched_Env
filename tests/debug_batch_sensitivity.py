"""
Debug batch size sensitivity in CUDA graph compilation.

This script investigates why certain batch sizes fail with torch.compile
and reduce-overhead mode.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Enable CUDA graph logging
os.environ['TORCH_LOGS'] = '+cudagraphs,+recompiles'

import time
import torch
import torch.nn as nn
from typing import Tuple

# Configuration
DATASET = 'family'
DATA_PATH = 'data'  # Base path, dataset name is appended
SEED = 42


def setup_env_and_policy(batch_size: int, device: torch.device):
    """Create environment and policy for a specific batch size."""
    from data_handler import DataHandler
    from index_manager import IndexManager
    from unification import UnificationEngine
    from unification_vectorized import UnificationEngineVectorized
    from env_eval_compiled import EvalOnlyEnvCompiled, EvalObs
    from embeddings import EmbedderLearnable as TensorEmbedder
    from model import ActorCriticPolicy as TensorPolicy
    
    # Load data
    dh = DataHandler(
        dataset_name=DATASET,
        base_path=DATA_PATH,
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules_amie.txt",
        facts_file="train.txt",
        corruption_mode='dynamic',
    )
    
    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=1000,
        max_arity=dh.max_arity,
        padding_atoms=6,
        device=device,
        rules=dh.rules,
    )
    dh.materialize_indices(im=im, device=device)
    
    # Base engine
    stringifier_params = {
        'verbose': 0,
        'idx2predicate': im.idx2predicate,
        'idx2constant': im.idx2constant,
        'idx2template_var': im.idx2template_var,
        'padding_idx': im.padding_idx,
        'n_constants': im.constant_no
    }
    base_engine = UnificationEngine.from_index_manager(
        im, take_ownership=True,
        stringifier_params=stringifier_params,
        end_pred_idx=im.end_pred_idx,
        end_proof_action=True,
        max_derived_per_state=120,
    )
    base_engine.index_manager = im
    
    # Vectorized engine - auto-compute max params
    vec_engine = UnificationEngineVectorized.from_base_engine(
        base_engine,
        max_fact_pairs=None,  # Auto-compute
        max_rule_pairs=None,  # Auto-compute
    )
    
    # Environment
    env = EvalOnlyEnvCompiled(
        vec_engine=vec_engine,
        batch_size=batch_size,
        padding_atoms=6,
        padding_states=120,
        max_depth=20,
        end_proof_action=True,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
    )
    
    # Embedder
    embedder = TensorEmbedder(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=1000,
        max_arity=dh.max_arity,
        padding_atoms=6,
        atom_embedder='transe',
        state_embedder='mean',
        constant_embedding_size=128,
        predicate_embedding_size=128,
        atom_embedding_size=128,
        device=str(device),
    )
    embedder.embed_dim = 128
    
    # Policy
    policy = TensorPolicy(
        embedder=embedder,
        embed_dim=128,
        action_dim=120,
        hidden_dim=256,
        num_layers=8,
        dropout_prob=0.0,
        device=device,
        compile_policy=False,
    ).to(device)
    
    # Create policy function using the proper helper
    from model_eval_optimized import create_policy_logits_fn
    policy_fn = create_policy_logits_fn(policy, deterministic=True)
    
    # Convert queries
    def convert_queries(queries):
        tensors = []
        for q in queries:
            atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
            tensors.append(atom)
        return torch.stack(tensors, dim=0)
    
    test_queries = convert_queries(dh.test_queries)
    
    return env, policy_fn, test_queries


def test_batch_size(batch_size: int, device: torch.device) -> Tuple[bool, str, float]:
    """
    Test a specific batch size for CUDA graph compatibility.
    
    Returns:
        success: Whether the test passed
        message: Description of result or error
        time_per_query: Average time per query in ms
    """
    print(f"\n{'='*60}")
    print(f"Testing batch_size = {batch_size}")
    print(f"{'='*60}")
    
    try:
        env, policy_fn, test_queries = setup_env_and_policy(batch_size, device)
        test_queries = test_queries[:100].to(device)
        
        # Create compiled evaluator
        from model_eval_optimized import CompiledEvaluator
        
        evaluator = CompiledEvaluator(
            env=env,
            policy_logits_fn=policy_fn,
            batch_size=batch_size,
            max_steps=10,
            deterministic=True,
        )
        
        # Check static buffer data pointer
        initial_ptr = evaluator._input_buffer.data_ptr()
        print(f"Initial input_buffer ptr: {initial_ptr}")
        
        # Warmup
        print("Warming up...")
        warmup_start = time.time()
        evaluator.warmup(test_queries[:10])
        warmup_time = time.time() - warmup_start
        print(f"Warmup time: {warmup_time:.2f}s")
        
        # Check buffer ptr didn't change
        after_warmup_ptr = evaluator._input_buffer.data_ptr()
        if after_warmup_ptr != initial_ptr:
            return False, f"Buffer ptr changed after warmup: {initial_ptr} -> {after_warmup_ptr}", 0
        
        # Test with various actual batch sizes
        test_sizes = [10, 50, batch_size // 2, batch_size - 1, batch_size]
        
        for actual_size in test_sizes:
            if actual_size > batch_size:
                continue
                
            queries = test_queries[:min(actual_size, len(test_queries))]
            if len(queries) < actual_size:
                queries = queries.repeat((actual_size // len(queries)) + 1, 1)[:actual_size]
            
            print(f"  Testing actual size {actual_size}...")
            
            # Multiple runs to check stability
            times = []
            for i in range(5):
                start = time.time()
                log_probs, success, lengths, rewards = evaluator(queries)
                torch.cuda.synchronize()
                times.append(time.time() - start)
                
                # Verify outputs are valid
                if log_probs.shape[0] != actual_size:
                    return False, f"Output shape mismatch: {log_probs.shape[0]} vs {actual_size}", 0
                if not torch.isfinite(log_probs).all():
                    return False, f"Non-finite log_probs at size {actual_size}", 0
            
            avg_time = sum(times) / len(times)
            time_variance = max(times) - min(times)
            print(f"    OK: avg={avg_time*1000:.2f}ms, var={time_variance*1000:.2f}ms")
            
            # High variance indicates recompilation
            if time_variance > 0.5 and i > 0:  # Allow first run to be slow
                print(f"    WARNING: High time variance suggests recompilation!")
        
        # Final stability test
        print("\nRunning 20 iterations to check steady-state stability...")
        steady_times = []
        for i in range(20):
            queries = test_queries[:batch_size]
            if len(queries) < batch_size:
                queries = queries.repeat((batch_size // len(queries)) + 1, 1)[:batch_size]
            
            start = time.time()
            _ = evaluator(queries)
            torch.cuda.synchronize()
            steady_times.append(time.time() - start)
        
        avg_time = sum(steady_times[5:]) / len(steady_times[5:])  # Exclude first 5
        print(f"Steady-state: avg={avg_time*1000:.2f}ms per {batch_size} queries")
        
        return True, "OK", avg_time * 1000 / batch_size
        
    except Exception as e:
        import traceback
        return False, f"Error: {e}\n{traceback.format_exc()}", 0


def find_optimal_batch_size(device: torch.device, min_size: int = 50, max_size: int = 2000, step: int = 50):
    """
    Test a range of batch sizes to find optimal one and identify problematic sizes.
    """
    print("\n" + "="*70)
    print("BATCH SIZE SENSITIVITY ANALYSIS")
    print("="*70)
    
    results = []
    
    # Test specific sizes mentioned as problematic
    test_sizes = [
        50, 100, 200, 300, 400, 500, 510, 520, 550, 600, 
        750, 1000, 1020, 1100, 1500, 2000
    ]
    
    for size in test_sizes:
        if size < min_size or size > max_size:
            continue
            
        success, msg, time_per_query = test_batch_size(size, device)
        results.append({
            'batch_size': size,
            'success': success,
            'message': msg,
            'time_per_query_ms': time_per_query,
        })
        
        # Clear CUDA cache between tests
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Batch Size':<12} {'Status':<10} {'Time/Query (ms)':<18} {'Notes'}")
    print("-"*70)
    
    for r in results:
        status = "✓ OK" if r['success'] else "✗ FAIL"
        time_str = f"{r['time_per_query_ms']:.4f}" if r['time_per_query_ms'] > 0 else "N/A"
        notes = "" if r['success'] else r['message'][:40]
        print(f"{r['batch_size']:<12} {status:<10} {time_str:<18} {notes}")
    
    # Identify optimal
    successful = [r for r in results if r['success'] and r['time_per_query_ms'] > 0]
    if successful:
        best = min(successful, key=lambda x: x['time_per_query_ms'])
        print(f"\nOptimal batch size: {best['batch_size']} ({best['time_per_query_ms']:.4f} ms/query)")
    
    # Identify problematic sizes
    failed = [r for r in results if not r['success']]
    if failed:
        print(f"\nProblematic batch sizes: {[r['batch_size'] for r in failed]}")


def analyze_cuda_graph_behavior(batch_size: int, device: torch.device):
    """
    Deep analysis of CUDA graph behavior at a specific batch size.
    """
    print(f"\n{'='*70}")
    print(f"DEEP ANALYSIS: batch_size={batch_size}")
    print(f"{'='*70}")
    
    env, policy_fn, test_queries = setup_env_and_policy(batch_size, device)
    
    # Create evaluator
    from model_eval_optimized import CompiledEvaluator
    evaluator = CompiledEvaluator(
        env=env,
        policy_logits_fn=policy_fn,
        batch_size=batch_size,
        max_steps=10,
        deterministic=True,
    )
    
    # Get the compiled function directly
    compiled_fn = evaluator._get_compiled()
    
    # Check tensor allocations
    print("\n1. Checking tensor memory allocations...")
    
    # Fill buffer with sample data
    test_queries = test_queries[:batch_size].to(device)
    if len(test_queries) < batch_size:
        test_queries = test_queries.repeat((batch_size // len(test_queries)) + 1, 1)[:batch_size]
    
    evaluator._input_buffer.copy_(test_queries)
    
    # Record memory before
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    
    # Run compilation
    print("  First run (cold)...")
    start = time.time()
    out1 = compiled_fn(evaluator._input_buffer)
    torch.cuda.synchronize()
    time_cold = time.time() - start
    
    mem_after_cold = torch.cuda.max_memory_allocated()
    print(f"    Time: {time_cold*1000:.2f}ms")
    print(f"    Peak memory: {(mem_after_cold - mem_before) / 1e6:.1f} MB")
    
    # Second run
    torch.cuda.reset_peak_memory_stats()
    print("  Second run (warm 1)...")
    start = time.time()
    out2 = compiled_fn(evaluator._input_buffer)
    torch.cuda.synchronize()
    time_warm1 = time.time() - start
    print(f"    Time: {time_warm1*1000:.2f}ms")
    
    # Third run
    print("  Third run (warm 2)...")
    start = time.time()
    out3 = compiled_fn(evaluator._input_buffer)
    torch.cuda.synchronize()
    time_warm2 = time.time() - start
    print(f"    Time: {time_warm2*1000:.2f}ms")
    
    # Many more runs
    print("\n2. Steady state analysis (50 iterations)...")
    times = []
    for i in range(50):
        start = time.time()
        _ = compiled_fn(evaluator._input_buffer)
        torch.cuda.synchronize()
        times.append(time.time() - start)
    
    print(f"    Min: {min(times)*1000:.2f}ms")
    print(f"    Max: {max(times)*1000:.2f}ms")
    print(f"    Avg (last 40): {sum(times[10:]) / 40 * 1000:.2f}ms")
    print(f"    Std: {(sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5 * 1000:.2f}ms")
    
    # Check for large spikes (indicates recompilation)
    threshold = min(times[10:]) * 10  # 10x minimum is suspicious
    spikes = [i for i, t in enumerate(times) if t > threshold]
    if spikes:
        print(f"    WARNING: Large time spikes at iterations: {spikes}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=None, help='Test specific batch size')
    parser.add_argument('--analyze', type=int, default=None, help='Deep analysis of specific batch size')
    parser.add_argument('--sweep', action='store_true', help='Test range of batch sizes')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.analyze:
        analyze_cuda_graph_behavior(args.analyze, device)
    elif args.batch_size:
        success, msg, time_per = test_batch_size(args.batch_size, device)
        print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
        print(f"Message: {msg}")
    elif args.sweep:
        find_optimal_batch_size(device)
    else:
        # Default: test a few key sizes
        for size in [510, 1020, 500, 512]:
            success, msg, time_per = test_batch_size(size, device)
            print(f"\nbatch_size={size}: {'OK' if success else 'FAILED'} ({msg[:50]})")
            torch.cuda.empty_cache()
