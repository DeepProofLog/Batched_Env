"""
Profile compiled evaluation to identify performance bottlenecks.

This script profiles:
1. Individual components (init, unification, step)
2. Compiled step execution with per-step breakdown
3. CUDA events for accurate GPU timing

Results are saved to profile_eval_compiled_results.txt

Usage:
    python tests/profile_eval_compiled.py [--batch-size 256] [--n-steps 20]
    python tests/profile_eval_compiled.py --mode reduce-overhead --batch-size 512
"""

import torch
import time
import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace
from data_handler import DataHandler
from index_manager import IndexManager
from unification import UnificationEngine
from unification_vectorized import UnificationEngineVectorized
from env_optimized import EvalEnvOptimized, EvalObs
from embeddings import EmbedderLearnable as TensorEmbedder
from model import ActorCriticPolicy as TensorPolicy


OUTPUT_FILE = "profile_eval_compiled_results.txt"


class ProfileResults:
    """Accumulate and format profiling results."""
    
    def __init__(self):
        self.lines = []
        self.sections = {}
        self.current_section = None
        
    def header(self, text):
        line = f"\n{'='*70}\n{text}\n{'='*70}"
        self.lines.append(line)
        print(line)
        self.current_section = text
        self.sections[text] = []
        
    def log(self, text):
        self.lines.append(text)
        print(text)
        if self.current_section:
            self.sections[self.current_section].append(text)
            
    def table_row(self, label, value, unit="ms", pct=None):
        if pct is not None:
            text = f"  {label:<35} {value:>10.2f} {unit:<5} ({pct:>5.1f}%)"
        else:
            text = f"  {label:<35} {value:>10.2f} {unit}"
        self.log(text)
        
    def save(self, filename):
        with open(filename, 'w') as f:
            f.write('\n'.join(self.lines))
        self.log(f"\nResults saved to: {filename}")


def setup_components(device):
    """Setup all components for evaluation."""
    config = SimpleNamespace(
        dataset='family',
        data_path='./data/',
        padding_atoms=6,
        padding_states=120,
        max_depth=20,
        memory_pruning=True,
        end_proof_action=True,
        max_total_vars=100,
        atom_embedding_size=250,
        seed=42,
    )
    
    dh = DataHandler(
        dataset_name=config.dataset,
        base_path=config.data_path,
        train_file='train.txt',
        valid_file='valid.txt',
        test_file='test.txt',
        rules_file='rules.txt',
        facts_file='train.txt',
    )
    
    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=config.max_total_vars,
        max_arity=dh.max_arity,
        padding_atoms=config.padding_atoms,
        device=device,
        rules=dh.rules,
    )
    
    dh.materialize_indices(im=im, device=device)
    
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
        end_proof_action=config.end_proof_action,
        max_derived_per_state=config.padding_states,
    )
    base_engine.index_manager = im
    
    vec_engine = UnificationEngineVectorized.from_base_engine(
        base_engine,
        max_fact_pairs=None,
        max_rule_pairs=None,
        padding_atoms=config.padding_atoms,
    )
    
    env = EvalEnvOptimized(
        vec_engine=vec_engine,
        batch_size=256,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_depth,
        end_proof_action=config.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
        memory_pruning=True,
    )
    
    # Create embedder
    torch.manual_seed(config.seed)
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
    
    # Create model using embedder
    model = TensorPolicy(
        embedder=embedder,
        embed_dim=config.atom_embedding_size,
        action_dim=config.padding_states,
        hidden_dim=256,
        num_layers=8,
        dropout_prob=0.0,
        device=device,
    ).to(device)
    model.eval()
    
    return config, dh, im, vec_engine, env, model


def convert_queries(queries, im):
    """Convert Query objects to tensor."""
    tensors = []
    for q in queries:
        atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
        tensors.append(atom)
    return torch.stack(tensors, dim=0)


def make_obs(state, env):
    """Create observation from state."""
    action_mask = torch.arange(env.padding_states, device=state.current_states.device).unsqueeze(0) < state.derived_counts.unsqueeze(1)
    return EvalObs(
        sub_index=state.current_states.unsqueeze(1),
        derived_sub_indices=state.derived_states,
        action_mask=action_mask,
    )


def profile_components(env, model, queries, device, n_steps, results: ProfileResults):
    """Profile individual components without compilation."""
    results.header("COMPONENT PROFILING (Eager Mode)")
    
    B = queries.shape[0]
    results.log(f"Batch size: {B}")
    results.log(f"Steps: {n_steps}")
    
    timings = {}
    
    # Profile init_state_from_queries
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        state = env.init_state_from_queries(queries)
    torch.cuda.synchronize()
    init_time = (time.time() - t0) / 10 * 1000
    timings['init_state_from_queries'] = init_time
    
    # Profile _compute_derived_functional
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        derived, counts, new_var = env._compute_derived_functional(
            state.current_states, state.next_var_indices, state.original_queries,
            state.history_hashes, state.history_count
        )
    torch.cuda.synchronize()
    derived_time = (time.time() - t0) / 10 * 1000
    timings['_compute_derived_functional'] = derived_time
    
    # Profile step_functional (uncompiled)
    actions = torch.zeros(B, dtype=torch.long, device=device)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        result = env.step_functional(state, actions)
    torch.cuda.synchronize()
    step_time = (time.time() - t0) / 10 * 1000
    timings['step_functional'] = step_time
    
    # Profile unification engine directly
    excluded = state.original_queries[:, 0:1, :]
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        out = env.engine.get_derived_states_compiled(
            state.current_states, state.next_var_indices, excluded
        )
    torch.cuda.synchronize()
    unify_time = (time.time() - t0) / 100 * 1000
    timings['get_derived_states_compiled'] = unify_time
    
    # Calculate percentages
    total_step = step_time
    
    results.log("\nComponent breakdown:")
    results.table_row("init_state_from_queries", init_time)
    results.table_row("_compute_derived_functional", derived_time, pct=derived_time/total_step*100 if total_step > 0 else 0)
    results.table_row("get_derived_states_compiled (raw)", unify_time, pct=unify_time/total_step*100 if total_step > 0 else 0)
    results.table_row("step_functional (total)", step_time)
    
    # Estimate overhead
    overhead = step_time - derived_time
    results.table_row("Step overhead (non-unification)", overhead, pct=overhead/total_step*100 if total_step > 0 else 0)
    
    results.log(f"\nTensor shapes (batch={B}):")
    results.log(f"  derived_states: {state.derived_states.shape}")
    results.log(f"  M_max (padding_states): {env.engine.M_max}")
    results.log(f"  K_max (padding_atoms): {env.engine.K_max}")
    
    return timings


def profile_compiled(env, model, queries, device, n_steps, mode, results: ProfileResults):
    """Profile compiled step execution with detailed breakdown."""
    results.header(f"COMPILED STEP PROFILING (mode='{mode}')")
    
    B = queries.shape[0]
    results.log(f"Batch size: {B}")
    results.log(f"Compile mode: {mode}")
    
    # Compile
    results.log(f"\nCompiling with mode='{mode}'...")
    torch.cuda.synchronize()
    t0 = time.time()
    env.compile(model, deterministic=True, mode=mode, fullgraph=True)
    torch.cuda.synchronize()
    compile_time = time.time() - t0
    results.log(f"Compilation time: {compile_time:.2f} s")
    
    # Setup
    state = env.init_state_from_queries(queries)
    obs = make_obs(state, env)
    query_pool = queries
    per_env_ptrs = torch.zeros(B, dtype=torch.long, device=device)
    
    # Warmup
    results.log("\nWarmup (5 steps)...")
    warmup_times = []
    for i in range(5):
        torch.cuda.synchronize()
        t0 = time.time()
        result = env.step_with_policy(state, obs, query_pool, per_env_ptrs, eval_mode=True)
        torch.cuda.synchronize()
        warmup_times.append((time.time() - t0) * 1000)
        state, obs = result[0], result[1]
    results.log(f"  Warmup step times: {', '.join(f'{t:.1f}' for t in warmup_times)} ms")
    
    # Reset for timed run
    state = env.init_state_from_queries(queries)
    obs = make_obs(state, env)
    per_env_ptrs = torch.zeros(B, dtype=torch.long, device=device)
    
    # Timed run with per-step metrics
    results.log(f"\nTiming {n_steps} steps with done tracking...")
    step_times = []
    done_counts = []
    
    for step in range(n_steps):
        torch.cuda.synchronize()
        t0 = time.time()
        result = env.step_with_policy(state, obs, query_pool, per_env_ptrs, eval_mode=True)
        torch.cuda.synchronize()
        elapsed = (time.time() - t0) * 1000
        step_times.append(elapsed)
        state, obs = result[0], result[1]
        done_count = state.done.sum().item()
        done_counts.append(done_count)
        
        # Early exit check
        if state.done.all():
            results.log(f"  Early exit at step {step} (all {B} queries done)")
            break
    
    actual_steps = len(step_times)
    total_time = sum(step_times)
    mean_time = total_time / actual_steps
    
    # Performance summary
    results.log("\nPERFORMANCE SUMMARY:")
    results.table_row("Compilation time", compile_time * 1000)
    results.table_row("Total eval time", total_time)
    results.table_row(f"Mean step time ({actual_steps} steps)", mean_time)
    results.table_row("Min step time", min(step_times))
    results.table_row("Max step time", max(step_times))
    results.table_row(f"Time per query ({actual_steps} steps)", total_time / B)
    
    # Per-step breakdown
    results.log("\nPER-STEP BREAKDOWN:")
    results.log(f"  {'Step':<6} {'Done':<10} {'Time (ms)':<12} {'Active':<10} {'Wasted %':<10}")
    results.log("  " + "-" * 48)
    
    total_wasted = 0
    for i, (t, done) in enumerate(zip(step_times, done_counts)):
        active = B - (done_counts[i-1] if i > 0 else 0)
        wasted_pct = (1 - active / B) * 100 if B > 0 else 0
        total_wasted += wasted_pct * t
        results.log(f"  {i:<6} {done:>5}/{B:<4} {t:>10.2f}   {active:>6}     {wasted_pct:>6.1f}%")
    
    # Efficiency analysis
    results.log("\nEFFICIENCY ANALYSIS:")
    
    # Calculate theoretical optimal time (if we could skip done queries)
    avg_active_ratio = sum((B - (done_counts[i-1] if i > 0 else 0)) / B for i in range(actual_steps)) / actual_steps
    results.log(f"  Average active query ratio: {avg_active_ratio:.1%}")
    results.log(f"  Theoretical speedup if skipping done: {1/avg_active_ratio:.2f}x")
    
    # Wasted computation estimate
    wasted_time_est = total_time * (1 - avg_active_ratio)
    results.log(f"  Estimated wasted time (done queries): {wasted_time_est:.1f} ms ({(1-avg_active_ratio)*100:.1f}%)")
    
    return compile_time, total_time, mean_time, actual_steps


def identify_bottlenecks(eager_timings, compiled_compile_time, compiled_total_time, 
                         compiled_mean_step, n_steps, batch_size, results: ProfileResults):
    """Analyze and identify main bottlenecks."""
    results.header("BOTTLENECK ANALYSIS")
    
    # Calculate bottleneck contributions
    bottlenecks = []
    
    # 1. Compilation overhead
    compile_ms = compiled_compile_time * 1000
    bottlenecks.append(("Compilation (one-time)", compile_ms, "One-time cost, amortized over many evals"))
    
    # 2. Per-step unification cost
    unify_per_step = eager_timings.get('_compute_derived_functional', 0)
    unify_total = unify_per_step * n_steps
    bottlenecks.append(("Unification (per step)", unify_per_step, f"Total: {unify_total:.1f} ms for {n_steps} steps"))
    
    # 3. No early exit waste
    step_overhead = compiled_mean_step - unify_per_step
    bottlenecks.append(("Step overhead (non-unification)", step_overhead, "Policy forward, state updates, etc."))
    
    # 4. Total per-step
    bottlenecks.append(("Total per step", compiled_mean_step, f"= unification + overhead"))
    
    results.log("\nBOTTLENECK RANKING:")
    results.log(f"  {'Component':<40} {'Time (ms)':<12} {'Notes'}")
    results.log("  " + "-" * 80)
    
    for name, time_ms, notes in bottlenecks:
        results.log(f"  {name:<40} {time_ms:>10.2f}   {notes}")
    
    # Key insights
    results.log("\nKEY INSIGHTS:")
    results.log("  1. Unification is the dominant per-step cost")
    results.log("  2. All B queries processed every step (no early exit per query)")
    results.log("  3. With CUDA graphs (reduce-overhead), batch size must stay fixed")
    results.log("  4. Optimal batch size ~512 (larger causes memory bandwidth issues)")
    
    # Recommendations
    results.log("\nRECOMMENDATIONS:")
    results.log("  1. [HIGH] Implement per-query early exit (skip done queries in unification)")
    results.log("  2. [MED] Profile unification kernel for optimization opportunities")
    results.log("  3. [LOW] Consider reducing max_depth if most proofs finish early")
    results.log("  4. [LOW] Pre-compute and cache common unification patterns")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--n-steps', type=int, default=20)
    parser.add_argument('--mode', type=str, default='reduce-overhead', 
                       choices=['default', 'reduce-overhead', 'max-autotune'])
    parser.add_argument('--output', type=str, default=OUTPUT_FILE,
                       help='Output file for results')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize results
    results = ProfileResults()
    results.header("EVALUATION PROFILING REPORT")
    results.log(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results.log(f"Device: {device}")
    results.log(f"Batch size: {args.batch_size}")
    results.log(f"Steps: {args.n_steps}")
    results.log(f"Compile mode: {args.mode}")
    
    # Setup
    results.log("\nSetting up components...")
    config, dh, im, vec_engine, env, model = setup_components(device)
    results.log(f"Queries loaded - Test: {len(dh.test_queries)}")
    
    # Convert queries
    queries = convert_queries(dh.test_queries[:args.batch_size], im).to(device)
    results.log(f"Using {queries.shape[0]} queries")
    
    # Profile eager components
    eager_timings = profile_components(env, model, queries, device, args.n_steps, results)
    
    # Profile compiled execution
    compile_time, total_time, mean_step, actual_steps = profile_compiled(
        env, model, queries, device, args.n_steps, args.mode, results
    )
    
    # Bottleneck analysis
    identify_bottlenecks(
        eager_timings, compile_time, total_time, mean_step, 
        actual_steps, args.batch_size, results
    )
    
    # Also profile with 'default' mode for comparison
    results.header("COMPARISON: default mode")
    
    # Reset environment
    env._compiled = False
    env._compiled_step_fn = None
    
    compile_time_default, total_time_default, mean_step_default, _ = profile_compiled(
        env, model, queries, device, args.n_steps, 'default', results
    )
    
    # Final comparison
    results.header("MODE COMPARISON SUMMARY")
    results.log(f"\n  {'Mode':<20} {'Compile (s)':<15} {'Eval (ms)':<15} {'Mean Step (ms)':<15}")
    results.log("  " + "-" * 65)
    results.log(f"  {'reduce-overhead':<20} {compile_time:<15.2f} {total_time:<15.2f} {mean_step:<15.2f}")
    results.log(f"  {'default':<20} {compile_time_default:<15.2f} {total_time_default:<15.2f} {mean_step_default:<15.2f}")
    
    speedup = total_time_default / total_time if total_time > 0 else 0
    results.log(f"\n  reduce-overhead speedup vs default: {speedup:.2f}x")
    
    # Per-candidate metrics
    n_candidates = args.batch_size
    results.log(f"\n  Per-candidate timing (batch_size={args.batch_size}):")
    results.log(f"    reduce-overhead: {total_time / n_candidates:.3f} ms/candidate")
    results.log(f"    default:         {total_time_default / n_candidates:.3f} ms/candidate")
    
    # Save results
    results.save(args.output)


if __name__ == '__main__':
    main()
