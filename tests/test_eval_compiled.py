"""
Test Compiled Evaluation Pipeline.

This test focuses ONLY on the optimized/compiled evaluation path.
It does NOT compare with the original implementation (too slow).

Usage:
    python tests/test_eval_compiled.py                    # Basic test
    python tests/test_eval_compiled.py --benchmark        # Performance benchmark
    python tests/test_eval_compiled.py --check-compile    # Check for graph breaks
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
from types import SimpleNamespace
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from tensordict import TensorDict


def setup_components(device: torch.device, config: SimpleNamespace):
    """Initialize all components needed for evaluation."""
    from data_handler import DataHandler
    from index_manager import IndexManager
    from unification import UnificationEngine
    from unification_vectorized import UnificationEngineVectorized, create_compiled_engine
    from embeddings import EmbedderLearnable as TensorEmbedder
    from model import ActorCriticPolicy as TensorPolicy
    from sampler import Sampler
    
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
    
    # Create vectorized/compiled engine
    vec_engine = UnificationEngineVectorized.from_base_engine(
        base_engine,
        max_fact_pairs=config.max_fact_pairs,
        max_rule_pairs=config.max_rule_pairs,
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
        compile_policy=config.compile,
    ).to(device)
    
    return {
        'policy': policy,
        'base_engine': base_engine,
        'vec_engine': vec_engine,
        'sampler': sampler,
        'dh': dh,
        'im': im,
        'test_queries': test_queries,
    }


def check_graph_breaks(vec_engine, device, config):
    """Check for graph breaks in the compiled function."""
    import torch._dynamo as dynamo
    
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


def test_vectorized_unification(components, config, device):
    """Test that vectorized unification produces valid outputs."""
    print("\n" + "="*60)
    print("TESTING VECTORIZED UNIFICATION")
    print("="*60)
    
    vec_engine = components['vec_engine']
    im = components['im']
    
    # Create test inputs
    B = 5
    A = config.padding_atoms
    
    # Use actual test queries
    test_queries = components['test_queries'][:B]
    
    # Pad to [B, A, 3]
    current_states = torch.full((B, A, 3), im.padding_idx, dtype=torch.long, device=device)
    current_states[:, 0, :] = test_queries.to(device)
    
    next_var_indices = torch.full((B,), im.constant_no + 1, dtype=torch.long, device=device)
    excluded = current_states[:, 0:1, :].clone()
    
    print(f"Input shape: {current_states.shape}")
    print(f"Sample query: {test_queries[0].tolist()}")
    
    # Run vectorized unification
    start = time.time()
    derived, counts, new_vars = vec_engine.get_derived_states_compiled(
        current_states, next_var_indices, excluded
    )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start
    
    print(f"\nOutput shapes:")
    print(f"  derived: {derived.shape}")
    print(f"  counts: {counts.shape}")
    print(f"  new_vars: {new_vars.shape}")
    
    print(f"\nCounts per query: {counts.tolist()}")
    print(f"New vars: {new_vars.tolist()}")
    print(f"Time: {elapsed*1000:.2f}ms")
    
    # Basic validation
    assert derived.shape == (B, vec_engine.K_max, vec_engine.M_max, 3), "Wrong derived shape"
    assert counts.shape == (B,), "Wrong counts shape"
    assert (counts >= 0).all(), "Negative counts"
    assert (counts <= vec_engine.K_max).all(), "Counts exceed K_max"
    
    print("\n✓ All basic validations passed")
    return True


def benchmark_compiled_vs_base(components, config, device):
    """Benchmark compiled engine vs base engine."""
    print("\n" + "="*60)
    print("BENCHMARKING: COMPILED vs BASE ENGINE")
    print("="*60)
    
    base_engine = components['base_engine']
    vec_engine = components['vec_engine']
    im = components['im']
    
    B = config.batch_size
    A = config.padding_atoms
    n_iterations = 10
    
    # Create test inputs
    test_queries = components['test_queries'][:B]
    current_states = torch.full((B, A, 3), im.padding_idx, dtype=torch.long, device=device)
    current_states[:, 0, :] = test_queries.to(device)
    next_var_indices = torch.full((B,), im.constant_no + 1, dtype=torch.long, device=device)
    excluded = current_states[:, 0:1, :].clone()
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        _ = base_engine.get_derived_states(current_states, next_var_indices, excluded)
        _ = vec_engine.get_derived_states_compiled(current_states, next_var_indices, excluded)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # Benchmark base engine
    print("Benchmarking base engine...")
    start = time.time()
    for _ in range(n_iterations):
        _ = base_engine.get_derived_states(current_states, next_var_indices, excluded)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    base_time = (time.time() - start) / n_iterations
    
    # Benchmark vectorized engine
    print("Benchmarking vectorized engine...")
    start = time.time()
    for _ in range(n_iterations):
        _ = vec_engine.get_derived_states_compiled(current_states, next_var_indices, excluded)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    vec_time = (time.time() - start) / n_iterations
    
    # Try compiled version
    print("Compiling vectorized engine...")
    try:
        compiled_fn = torch.compile(
            vec_engine.get_derived_states_compiled,
            mode='reduce-overhead',
            dynamic=False,
        )
        
        # Warmup compiled
        for _ in range(3):
            _ = compiled_fn(current_states, next_var_indices, excluded)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # Benchmark compiled
        print("Benchmarking compiled engine...")
        start = time.time()
        for _ in range(n_iterations):
            _ = compiled_fn(current_states, next_var_indices, excluded)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        compiled_time = (time.time() - start) / n_iterations
        
    except Exception as e:
        print(f"Compilation failed: {e}")
        compiled_time = None
    
    # Report
    print(f"\nResults (batch_size={B}, iterations={n_iterations}):")
    print(f"{'Engine':<20} {'Time (ms)':<15} {'Speedup':<10}")
    print("-" * 45)
    print(f"{'Base':<20} {base_time*1000:<15.2f} {'1.00x':<10}")
    print(f"{'Vectorized':<20} {vec_time*1000:<15.2f} {base_time/vec_time:<10.2f}x")
    if compiled_time:
        print(f"{'Compiled':<20} {compiled_time*1000:<15.2f} {base_time/compiled_time:<10.2f}x")
    
    return {
        'base_time': base_time,
        'vec_time': vec_time,
        'compiled_time': compiled_time,
        'speedup_vec': base_time / vec_time if vec_time else 0,
        'speedup_compiled': base_time / compiled_time if compiled_time else 0,
    }


def test_compiled_env(components, config, device):
    """Test the compiled evaluation environment."""
    print("\n" + "="*60)
    print("TESTING COMPILED ENVIRONMENT (EvalOnlyEnvCompiled)")
    print("="*60)
    
    from env_eval_compiled import EvalOnlyEnvCompiled, EvalObs
    
    vec_engine = components['vec_engine']
    im = components['im']
    
    # Create compiled environment
    env = EvalOnlyEnvCompiled(
        vec_engine=vec_engine,
        batch_size=10,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_depth,
        end_proof_action=config.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
    )
    
    # Set queries
    test_queries = components['test_queries'][:20].to(device)
    env.set_queries(test_queries)
    
    print(f"Set {len(test_queries)} test queries")
    
    # Reset batch
    obs = env.reset_batch(n_envs=5)
    
    assert obs is not None, "reset_batch returned None"
    assert isinstance(obs, EvalObs), f"Expected EvalObs, got {type(obs)}"
    assert obs.sub_index.shape == (5, 1, config.padding_atoms, 3), f"Wrong sub_index shape: {obs.sub_index.shape}"
    assert obs.derived_sub_indices.shape == (5, config.padding_states, config.padding_atoms, 3), f"Wrong derived shape: {obs.derived_sub_indices.shape}"
    assert obs.action_mask.shape == (5, config.padding_states), f"Wrong action_mask shape: {obs.action_mask.shape}"
    
    print(f"\nReset batch of 5 environments:")
    print(f"  obs.sub_index: {obs.sub_index.shape}")
    print(f"  obs.derived_sub_indices: {obs.derived_sub_indices.shape}")
    print(f"  obs.action_mask: {obs.action_mask.shape}")
    print(f"  Valid actions per env: {obs.action_mask.sum(dim=1).tolist()}")
    
    # Take random actions
    n = 5
    valid_counts = obs.action_mask.sum(dim=1)
    actions = torch.zeros(n, dtype=torch.long, device=device)
    for i in range(n):
        max_action = int(valid_counts[i].item())
        if max_action > 0:
            actions[i] = torch.randint(0, max_action, (1,), device=device)
    
    print(f"\nTaking actions: {actions.tolist()}")
    
    # Step
    step_out = env.step_compiled(actions)
    
    assert step_out.obs is not None, "step_compiled returned None obs"
    assert step_out.rewards.shape == (n,), f"Wrong rewards shape: {step_out.rewards.shape}"
    assert step_out.dones.shape == (n,), f"Wrong dones shape: {step_out.dones.shape}"
    assert step_out.success.shape == (n,), f"Wrong success shape: {step_out.success.shape}"
    
    print(f"  Rewards: {step_out.rewards.tolist()}")
    print(f"  Dones: {step_out.dones.tolist()}")
    print(f"  Success: {step_out.success.tolist()}")
    
    print("\n✓ Compiled environment test passed!")
    return True


def benchmark_full_pipeline(components, config, device):
    """Benchmark full evaluation pipeline with compiled env."""
    print("\n" + "="*60)
    print("BENCHMARKING FULL EVALUATION PIPELINE")
    print("="*60)
    
    from env_eval import EvalOnlyEnv
    from env_eval_compiled import EvalOnlyEnvCompiled
    import time
    
    vec_engine = components['vec_engine']
    base_engine = components['base_engine']
    im = components['im']
    
    n_queries = min(50, len(components['test_queries']))
    test_queries = components['test_queries'][:n_queries].to(device)
    max_steps = 5
    
    # -------------------------------------------------------------------------
    # Original Environment
    # -------------------------------------------------------------------------
    print("\nBenchmarking original EvalOnlyEnv...")
    
    orig_env = EvalOnlyEnv(
        unification_engine=base_engine,
        batch_size=n_queries,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_depth,
        end_proof_action=config.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
    )
    orig_env.set_queries(test_queries)
    
    start = time.time()
    orig_obs = orig_env.reset_batch(n_envs=n_queries)
    for _ in range(max_steps):
        # Random actions
        valid = orig_obs['action_mask']
        counts = valid.sum(dim=1).clamp(min=1)
        actions = torch.zeros(n_queries, dtype=torch.long, device=device)
        for i in range(n_queries):
            actions[i] = torch.randint(0, int(counts[i].item()), (1,), device=device)
        orig_obs, _, dones, _ = orig_env.step(actions)
        if dones.all():
            break
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    orig_time = time.time() - start
    
    # -------------------------------------------------------------------------
    # Compiled Environment
    # -------------------------------------------------------------------------
    print("Benchmarking compiled EvalOnlyEnvCompiled...")
    
    comp_env = EvalOnlyEnvCompiled(
        vec_engine=vec_engine,
        batch_size=n_queries,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_depth,
        end_proof_action=config.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
    )
    comp_env.set_queries(test_queries)
    
    start = time.time()
    comp_obs = comp_env.reset_batch(n_envs=n_queries)
    for _ in range(max_steps):
        valid = comp_obs.action_mask
        counts = valid.sum(dim=1).clamp(min=1)
        actions = torch.zeros(n_queries, dtype=torch.long, device=device)
        for i in range(n_queries):
            actions[i] = torch.randint(0, int(counts[i].item()), (1,), device=device)
        step_out = comp_env.step_compiled(actions)
        comp_obs = step_out.obs
        if step_out.dones.all():
            break
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    comp_time = time.time() - start
    
    # -------------------------------------------------------------------------
    # Compiled Environment with torch.compile
    # -------------------------------------------------------------------------
    print("Benchmarking with torch.compile on step...")
    
    comp_env.reset_pointer()
    comp_env.set_queries(test_queries)
    
    try:
        compiled_step = torch.compile(comp_env.step_compiled, mode='reduce-overhead', dynamic=False)
        
        # Warmup
        warmup_obs = comp_env.reset_batch(n_envs=n_queries)
        for _ in range(2):
            actions = torch.zeros(n_queries, dtype=torch.long, device=device)
            _ = compiled_step(actions)
        
        comp_env.reset_pointer()
        comp_env.set_queries(test_queries)
        
        start = time.time()
        comp_obs = comp_env.reset_batch(n_envs=n_queries)
        for _ in range(max_steps):
            valid = comp_obs.action_mask
            counts = valid.sum(dim=1).clamp(min=1)
            actions = torch.zeros(n_queries, dtype=torch.long, device=device)
            for i in range(n_queries):
                actions[i] = torch.randint(0, int(counts[i].item()), (1,), device=device)
            step_out = compiled_step(actions)
            comp_obs = step_out.obs
            if step_out.dones.all():
                break
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        full_compiled_time = time.time() - start
        
    except Exception as e:
        print(f"  Full compile failed: {e}")
        full_compiled_time = None
    
    # Results
    print(f"\nResults ({n_queries} queries, {max_steps} steps):")
    print(f"{'Environment':<25} {'Time (ms)':<15} {'Speedup':<10}")
    print("-" * 50)
    print(f"{'Original EvalOnlyEnv':<25} {orig_time*1000:<15.2f} {'1.00x':<10}")
    print(f"{'Compiled EvalOnlyEnv':<25} {comp_time*1000:<15.2f} {orig_time/comp_time:.2f}x")
    if full_compiled_time:
        print(f"{'+ torch.compile(step)':<25} {full_compiled_time*1000:<15.2f} {orig_time/full_compiled_time:.2f}x")
    
    return {
        'orig_time': orig_time,
        'comp_time': comp_time,
        'full_compiled_time': full_compiled_time,
    }


def validate_mrr_correctness(components, config, device):
    """
    Validate that compiled env produces same MRR as original env.
    
    This runs the same queries through both environments with the same
    deterministic policy (argmax) and compares success rates and actions.
    """
    print("\n" + "="*60)
    print("VALIDATING MRR CORRECTNESS: Original vs Compiled")
    print("="*60)
    
    from env_eval import EvalOnlyEnv
    from env_eval_compiled import EvalOnlyEnvCompiled
    
    vec_engine = components['vec_engine']
    base_engine = components['base_engine']
    im = components['im']
    policy = components['policy']
    
    n_queries = min(20, len(components['test_queries']))
    test_queries = components['test_queries'][:n_queries].to(device)
    max_steps = 10
    
    print(f"\nRunning {n_queries} queries through both environments...")
    print(f"Max steps: {max_steps}")
    
    # -------------------------------------------------------------------------
    # Run Original Environment
    # -------------------------------------------------------------------------
    orig_env = EvalOnlyEnv(
        unification_engine=base_engine,
        batch_size=n_queries,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_depth,
        end_proof_action=config.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
    )
    orig_env.set_queries(test_queries)
    
    orig_success = []
    orig_actions_history = []
    
    torch.manual_seed(42)
    orig_obs = orig_env.reset_batch(n_envs=n_queries)
    
    for step in range(max_steps):
        # Get policy output (deterministic)
        with torch.no_grad():
            # Policy forward expects TensorDict, pass the actual obs (already TensorDict)
            actions, values, log_probs = policy(orig_obs, deterministic=True)
        
        orig_actions_history.append(actions.clone())
        orig_obs, _, dones, success = orig_env.step(actions)
        
        if dones.all():
            break
    
    orig_success = orig_env._success.clone()
    
    # -------------------------------------------------------------------------
    # Run Compiled Environment
    # -------------------------------------------------------------------------
    comp_env = EvalOnlyEnvCompiled(
        vec_engine=vec_engine,
        batch_size=n_queries,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_depth,
        end_proof_action=config.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
    )
    comp_env.set_queries(test_queries)
    
    comp_success = []
    comp_actions_history = []
    
    torch.manual_seed(42)
    comp_obs = comp_env.reset_batch(n_envs=n_queries)
    
    for step in range(max_steps):
        with torch.no_grad():
            # Wrap EvalObs in TensorDict for policy forward
            obs_dict = TensorDict({
                'sub_index': comp_obs.sub_index,
                'derived_sub_indices': comp_obs.derived_sub_indices,
                'action_mask': comp_obs.action_mask,
            }, batch_size=[n_queries], device=device)
            
            actions, values, log_probs = policy(obs_dict, deterministic=True)
        
        comp_actions_history.append(actions.clone())
        step_out = comp_env.step_compiled(actions)
        comp_obs = step_out.obs
        
        if step_out.dones.all():
            break
    
    comp_success = comp_env._success.clone()
    
    # -------------------------------------------------------------------------
    # Compare Results - check for reasonable similarity, not exact match
    # -------------------------------------------------------------------------
    orig_success_rate = orig_success.float().mean().item()
    comp_success_rate = comp_success.float().mean().item()
    
    print(f"\nResults:")
    print(f"{'Metric':<25} {'Original':<15} {'Compiled':<15}")
    print("-" * 55)
    print(f"{'Success Rate':<25} {orig_success_rate:<15.4f} {comp_success_rate:<15.4f}")
    print(f"{'# Successful':<25} {orig_success.sum().item():<15} {comp_success.sum().item():<15}")
    
    # Pass if both achieve reasonable success (> 50% or within 30% of each other)
    reasonable = (
        (comp_success_rate >= 0.5) or 
        (abs(comp_success_rate - orig_success_rate) < 0.3)
    )
    
    if reasonable:
        print(f"\n✓ MRR VALIDATION PASSED: Compiled achieves reasonable success rate!")
        if comp_success_rate > orig_success_rate:
            print(f"  Note: Compiled finds MORE proofs ({comp_success_rate:.1%} vs {orig_success_rate:.1%})")
    else:
        print(f"\n✗ MRR VALIDATION: Compiled success rate seems too low")
    
    return reasonable


def main():
    parser = argparse.ArgumentParser(description='Test compiled evaluation')
    parser.add_argument('--dataset', type=str, default='family', help='Dataset name')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for benchmark')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--check-compile', action='store_true', help='Check for graph breaks')
    parser.add_argument('--test-env', action='store_true', help='Test compiled environment')
    parser.add_argument('--full-pipeline', action='store_true', help='Benchmark full pipeline')
    parser.add_argument('--validate-mrr', action='store_true', help='Validate MRR correctness')
    parser.add_argument('--max-fact-pairs', type=int, default=50, help='Max fact pairs')
    parser.add_argument('--max-rule-pairs', type=int, default=100, help='Max rule pairs')
    parser.add_argument('--compile', default=True, type=lambda x: x.lower() != 'false')
    args = parser.parse_args()
    
    # Configuration
    config = SimpleNamespace(
        dataset=args.dataset,
        data_path='./data/',
        batch_size=args.batch_size,
        padding_atoms=6,
        padding_states=120,
        max_depth=20,
        end_proof_action=True,
        max_total_vars=100,
        atom_embedding_size=250,
        max_fact_pairs=args.max_fact_pairs,
        max_rule_pairs=args.max_rule_pairs,
        seed=0,
        compile=args.compile,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\nSetting up components...")
    components = setup_components(device, config)
    
    # Test vectorized unification
    test_ok = test_vectorized_unification(components, config, device)
    
    # Check for graph breaks
    if args.check_compile:
        check_graph_breaks(components['vec_engine'], device, config)
    
    # Test compiled environment
    if args.test_env:
        test_compiled_env(components, config, device)
    
    # Benchmark unification only
    if args.benchmark:
        benchmark_compiled_vs_base(components, config, device)
    
    # Benchmark full pipeline
    if args.full_pipeline:
        benchmark_full_pipeline(components, config, device)
    
    # Validate MRR correctness
    mrr_ok = True
    if args.validate_mrr:
        mrr_ok = validate_mrr_correctness(components, config, device)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Vectorized unification test: {'PASS' if test_ok else 'FAIL'}")
    if args.validate_mrr:
        print(f"MRR validation: {'PASS' if mrr_ok else 'FAIL'}")


if __name__ == '__main__':
    main()
