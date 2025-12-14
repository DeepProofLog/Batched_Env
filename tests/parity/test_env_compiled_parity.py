"""
Environment Compiled Parity Tests.

Tests verifying that EvalOnlyEnvOptimized (compiled environment) produces
EXACTLY the same step-by-step traces as the tensor BatchedEnv when both use
skip_unary_actions=False.

Usage:
    # Run with pytest
    pytest tests/parity/test_env_compiled_parity.py -v -s
    
    # Run with torch.compile enabled
    pytest tests/parity/test_env_compiled_parity.py -v -s --compile
    
    # CLI
    python tests/parity/test_env_compiled_parity.py --dataset countries_s3 --n-queries 50
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict, Tuple, Optional
import random

import pytest
import torch
import numpy as np
from tensordict import TensorDict

# Setup paths
ROOT = Path(__file__).resolve().parents[2]
TEST_ENVS_ROOT = ROOT / "tests" / "other" / "test_envs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TEST_ENVS_ROOT) not in sys.path:
    sys.path.insert(1, str(TEST_ENVS_ROOT))

from data_handler import DataHandler
from index_manager import IndexManager
from unification import UnificationEngine
from unification_vectorized import UnificationEngineVectorized
from env import BatchedEnv
from env_optimized import EvalEnvOptimized
from utils.debug_helper import DebugHelper


# ============================================================================
# Configuration
# ============================================================================

def create_default_config() -> SimpleNamespace:
    """Default parameters for parity tests matching test_env_parity.py."""
    return SimpleNamespace(
        # Dataset/query selection
        dataset="countries_s3",
        n_queries=200,
        data_path="./data/",
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        train_depth=None,
        corruption_mode="dynamic",
        
        # Environment behavior
        max_depth=20,
        padding_atoms=6,
        padding_states=40,
        max_derived_per_state=40,
        skip_unary_actions=False,  # Must be False for parity
        end_proof_action=True,
        memory_pruning=True,
        use_exact_memory=False,
        reward_type=0,
        prover_verbose=0,
        max_total_runtime_vars=1000,
        device="cpu",
        
        # Seeds / logging
        seed=42,
        verbose=False,
        debug=False,
    )


def clone_config(config: SimpleNamespace) -> SimpleNamespace:
    """Create a shallow clone of a SimpleNamespace."""
    return SimpleNamespace(**vars(config))


# ============================================================================
# Query Preparation
# ============================================================================

def prepare_queries(config: SimpleNamespace) -> List[Tuple[str, Tuple[str, str, str]]]:
    """Prepare a shuffled list of queries for the requested dataset."""
    dh = DataHandler(
        dataset_name=config.dataset,
        base_path=config.data_path,
        train_file=config.train_file,
        valid_file=config.valid_file,
        test_file=config.test_file,
        rules_file=config.rules_file,
        facts_file=config.facts_file,
        train_depth=config.train_depth,
    )
    
    all_queries = []
    for q in dh.train_queries:
        all_queries.append(('train', (q.predicate, q.args[0], q.args[1])))
    for q in dh.valid_queries:
        all_queries.append(('valid', (q.predicate, q.args[0], q.args[1])))
    for q in dh.test_queries:
        all_queries.append(('test', (q.predicate, q.args[0], q.args[1])))
    
    rng = random.Random(config.seed)
    rng.shuffle(all_queries)
    
    if config.n_queries is not None:
        all_queries = all_queries[:config.n_queries]
    
    return all_queries


# ============================================================================
# Environment Setup
# ============================================================================

def setup_environments(config: SimpleNamespace) -> Tuple:
    """
    Setup both tensor and compiled environments from the same data.
    
    Returns:
        (tensor_env, compiled_env, im, debug_helper, dh)
    """
    device = torch.device(config.device)
    
    dh = DataHandler(
        dataset_name=config.dataset,
        base_path=config.data_path,
        train_file=config.train_file,
        valid_file=config.valid_file,
        test_file=config.test_file,
        rules_file=config.rules_file,
        facts_file=config.facts_file,
        train_depth=config.train_depth,
    )
    
    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=config.max_total_runtime_vars,
        padding_atoms=config.padding_atoms,
        max_arity=dh.max_arity,
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
    
    # Create base unification engine for both environments
    base_engine = UnificationEngine.from_index_manager(
        im, take_ownership=True,
        stringifier_params=stringifier_params,
        end_pred_idx=im.end_pred_idx if config.end_proof_action else None,
        end_proof_action=config.end_proof_action,
        max_derived_per_state=config.max_derived_per_state,
        sort_states=False,
    )
    base_engine.index_manager = im
    
    # Create vectorized engine for compiled env (with parity_mode=True)
    vec_engine = UnificationEngineVectorized.from_base_engine(
        base_engine,
        max_fact_pairs=None,
        max_rule_pairs=None,
        padding_atoms=config.padding_atoms,
        parity_mode=True,  # Enable exact matching
    )
    
    debug_helper = DebugHelper(
        verbose=0,
        idx2predicate=im.idx2predicate,
        idx2constant=im.idx2constant,
        idx2template_var=im.idx2template_var,
        padding_idx=im.padding_idx,
        n_constants=im.constant_no
    )
    
    return base_engine, vec_engine, im, debug_helper, dh


def create_tensor_env(
    queries: List[Tuple[str, Tuple[str, str, str]]],
    base_engine: UnificationEngine,
    im: IndexManager,
    config: SimpleNamespace,
) -> BatchedEnv:
    """Create a tensor BatchedEnv for the given queries."""
    device = torch.device(config.device)
    batch_size = len(queries)
    
    # Convert queries to tensor format
    query_tensors = []
    for split, (p, h, t) in queries:
        query_atom = im.atom_to_tensor(p, h, t)
        query_padded = torch.full(
            (config.padding_atoms, 3), im.padding_idx,
            dtype=torch.long, device=device
        )
        query_padded[0] = query_atom
        query_tensors.append(query_padded)
    
    queries_tensor = torch.stack(query_tensors, dim=0)
    
    return BatchedEnv(
        batch_size=batch_size,
        queries=queries_tensor,
        labels=torch.ones(batch_size, dtype=torch.long, device=device),
        query_depths=torch.ones(batch_size, dtype=torch.long, device=device),
        unification_engine=base_engine,
        mode='eval',
        max_depth=config.max_depth,
        memory_pruning=config.memory_pruning,
        use_exact_memory=config.use_exact_memory,
        skip_unary_actions=config.skip_unary_actions,  # Must be False
        end_proof_action=config.end_proof_action,
        reward_type=config.reward_type,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        true_pred_idx=im.predicate_str2idx.get('True'),
        false_pred_idx=im.predicate_str2idx.get('False'),
        end_pred_idx=im.predicate_str2idx.get('Endf'),
        verbose=0,
        prover_verbose=config.prover_verbose,
        device=device,
        runtime_var_start_index=im.constant_no + 1,
        total_vocab_size=im.constant_no + config.max_total_runtime_vars,
        sample_deterministic_per_env=False,
    )


def create_compiled_env(
    vec_engine: UnificationEngineVectorized,
    im: IndexManager,
    config: SimpleNamespace,
    batch_size: int,
) -> EvalEnvOptimized:
    """Create a compiled EvalOnlyEnvOptimized environment."""
    device = torch.device(config.device)
    
    return EvalEnvOptimized(
        vec_engine=vec_engine,
        batch_size=batch_size,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_depth,
        end_proof_action=config.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
        memory_pruning=config.memory_pruning,
    )


# ============================================================================
# Query Running
# ============================================================================

def safe_item(x):
    """Safely extract scalar from numpy/torch/python scalar."""
    if isinstance(x, (np.ndarray, np.generic)):
        return x.item() if x.size == 1 else x
    elif isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x
    return x


def run_query_trace_both(
    query: Tuple[str, str, str],
    base_engine: UnificationEngine,
    compiled_env: EvalEnvOptimized,
    im: IndexManager,
    debug_helper: DebugHelper,
    config: SimpleNamespace,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run a single query through both environments and collect step traces.
    
    Returns:
        (tensor_trace, compiled_trace) - each is a list of step dicts
    """
    device = torch.device(config.device)
    pad = im.padding_idx
    
    # Build query tensor
    p, h, t = query
    query_atom = im.atom_to_tensor(p, h, t)
    
    # For tensor env: [1, A, 3]
    query_padded = torch.full(
        (1, config.padding_atoms, 3), pad,
        dtype=torch.long, device=device
    )
    query_padded[0, 0] = query_atom
    
    # Create a fresh tensor env for this single query
    tensor_env = BatchedEnv(
        batch_size=1,
        queries=query_padded,
        labels=torch.ones(1, dtype=torch.long, device=device),
        query_depths=torch.ones(1, dtype=torch.long, device=device),
        unification_engine=base_engine,
        mode='eval',
        max_depth=config.max_depth,
        memory_pruning=config.memory_pruning,
        use_exact_memory=config.use_exact_memory,
        skip_unary_actions=config.skip_unary_actions,  # Must be False
        end_proof_action=config.end_proof_action,
        reward_type=config.reward_type,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        true_pred_idx=im.predicate_str2idx.get('True'),
        false_pred_idx=im.predicate_str2idx.get('False'),
        end_pred_idx=im.predicate_str2idx.get('Endf'),
        verbose=0,
        prover_verbose=config.prover_verbose,
        device=device,
        runtime_var_start_index=im.constant_no + 1,
        total_vocab_size=im.constant_no + config.max_total_runtime_vars,
        sample_deterministic_per_env=False,
    )
    
    # Setup eval mode with proper slots
    tensor_env.set_eval_dataset(
        queries=query_padded,
        labels=torch.ones(1, dtype=torch.long, device=device),
        query_depths=torch.ones(1, dtype=torch.long, device=device),
        per_slot_lengths=torch.ones(1, dtype=torch.long, device=device),
    )
    
    tensor_obs = tensor_env.reset()
    if 'next' in tensor_obs.keys():
        tensor_obs = tensor_obs['next']
    
    # For compiled env: use init_state_from_queries
    compiled_state = compiled_env.init_state_from_queries(query_atom.unsqueeze(0))
    
    tensor_trace = []
    compiled_trace = []
    
    tensor_done = False
    compiled_done = False
    
    for step in range(config.max_depth):
        # ===== Tensor env step =====
        if not tensor_done:
            tensor_state = tensor_env.current_queries[0]
            tensor_derived = tensor_env.derived_states_batch[0]
            tensor_mask = tensor_obs['action_mask'][0]
            tensor_n_actions = int(tensor_mask.sum())
            
            state_str = debug_helper.state_to_str(tensor_state)
            
            if tensor_n_actions == 0:
                tensor_trace.append({
                    'step': step,
                    'state': state_str,
                    'derived_states': [],
                    'num_actions': 0,
                    'action': None,
                    'done': True,
                })
                tensor_done = True
            else:
                derived_strs = sorted([
                    debug_helper.state_to_str(tensor_derived[a])
                    for a in range(tensor_n_actions)
                ])
                action = 0  # Deterministic: choose first action
                
                tensor_trace.append({
                    'step': step,
                    'state': state_str,
                    'derived_states': derived_strs,
                    'num_actions': tensor_n_actions,
                    'action': action,
                    'done': False,
                })
                
                # Take step
                action_td = TensorDict({'action': torch.tensor([action], device=device)}, batch_size=[1])
                result_td = tensor_env.step(action_td)
                if 'next' in result_td.keys():
                    tensor_obs = result_td['next']
                else:
                    tensor_obs = result_td
                
                if tensor_obs['done'][0]:
                    tensor_done = True
        
        # ===== Compiled env step =====
        if not compiled_done:
            compiled_current = compiled_state.current_states[0]
            compiled_derived = compiled_state.derived_states[0]
            compiled_count = int(compiled_state.derived_counts[0])
            
            state_str = debug_helper.state_to_str(compiled_current)
            
            if compiled_count == 0:
                compiled_trace.append({
                    'step': step,
                    'state': state_str,
                    'derived_states': [],
                    'num_actions': 0,
                    'action': None,
                    'done': True,
                })
                compiled_done = True
            else:
                derived_strs = sorted([
                    debug_helper.state_to_str(compiled_derived[k])
                    for k in range(compiled_count)
                ])
                action = 0  # Deterministic: choose first action
                
                compiled_trace.append({
                    'step': step,
                    'state': state_str,
                    'derived_states': derived_strs,
                    'num_actions': compiled_count,
                    'action': action,
                    'done': False,
                })
                
                # Take step
                actions = torch.tensor([action], dtype=torch.long, device=device)
                step_result = compiled_env.step_functional(compiled_state, actions)
                compiled_state = step_result.state
                
                if compiled_state.done[0]:
                    compiled_done = True
        
        # Break if both are done
        if tensor_done and compiled_done:
            break
    
    return tensor_trace, compiled_trace


# ============================================================================
# Trace Comparison
# ============================================================================

def canonicalize_state(state_str: str) -> str:
    """
    Normalize variable names in a state string for structural comparison.
    
    Variables like Var_2972, Var_2973 are renamed to V0, V1, V2...
    in order of first appearance.
    """
    import re
    if not state_str:
        return state_str
    
    var_pattern = r'Var_\d+'
    vars_found = re.findall(var_pattern, state_str)
    
    var_map = {}
    counter = 0
    for v in vars_found:
        if v not in var_map:
            var_map[v] = f"V{counter}"
            counter += 1
    
    result = state_str
    for old, new in var_map.items():
        result = result.replace(old, new)
    
    return result


def compare_trace_step(step_tensor: Dict, step_compiled: Dict, step_idx: int) -> Tuple[bool, str]:
    """Compare a single step. Returns (match, error_message)."""
    # Compare state canonical strings
    tensor_state = canonicalize_state(step_tensor.get('state', ''))
    compiled_state = canonicalize_state(step_compiled.get('state', ''))
    
    if tensor_state != compiled_state:
        return False, f"Step {step_idx}: state mismatch:\n  Tensor:   {step_tensor.get('state')}\n  Compiled: {step_compiled.get('state')}"
    
    # Compare number of actions
    if step_tensor.get('num_actions') != step_compiled.get('num_actions'):
        return False, f"Step {step_idx}: num_actions mismatch: {step_tensor.get('num_actions')} vs {step_compiled.get('num_actions')}"
    
    # Compare derived states using canonical form
    tensor_derived = set(canonicalize_state(s) for s in step_tensor.get('derived_states', []))
    compiled_derived = set(canonicalize_state(s) for s in step_compiled.get('derived_states', []))
    
    if tensor_derived != compiled_derived:
        only_tensor = tensor_derived - compiled_derived
        only_compiled = compiled_derived - tensor_derived
        return False, f"Step {step_idx}: derived_states mismatch:\n  Only tensor: {list(only_tensor)[:2]}\n  Only compiled:  {list(only_compiled)[:2]}"
    
    # Compare done flag
    if step_tensor.get('done') != step_compiled.get('done'):
        return False, f"Step {step_idx}: done mismatch: {step_tensor.get('done')} vs {step_compiled.get('done')}"
    
    return True, ""


def compare_full_traces(trace_tensor: List[Dict], trace_compiled: List[Dict]) -> Tuple[bool, str]:
    """Compare full traces step-by-step. Returns (match, error_message)."""
    if len(trace_tensor) != len(trace_compiled):
        return False, f"Trace length mismatch: {len(trace_tensor)} vs {len(trace_compiled)}"
    
    for i, (step_t, step_c) in enumerate(zip(trace_tensor, trace_compiled)):
        match, error = compare_trace_step(step_t, step_c, i)
        if not match:
            return False, error
    
    return True, ""


def compare_all_traces(
    all_tensor_traces: List[List[Dict]],
    all_compiled_traces: List[List[Dict]],
    queries: List[Tuple[str, Tuple[str, str, str]]]
) -> Tuple[int, List[Tuple[int, str, str]]]:
    """Compare all query traces. Returns (num_matches, list of mismatches)."""
    mismatches = []
    matches = 0
    
    for i, (tensor_trace, compiled_trace, query) in enumerate(zip(all_tensor_traces, all_compiled_traces, queries)):
        split, (pred, head, tail) = query
        query_str = f"{pred}({head}, {tail}) [{split}]"
        
        match, error = compare_full_traces(tensor_trace, compiled_trace)
        if match:
            matches += 1
        else:
            mismatches.append((i, query_str, error))
    
    return matches, mismatches


# ============================================================================
# Main Test Function
# ============================================================================

def run_parity_test(
    dataset: str,
    n_queries: int,
    max_depth: int = 20,
    seed: int = 42,
    compile_mode: bool = False,
    verbose: bool = False,
) -> Tuple[bool, Dict]:
    """Run parity test comparing tensor and compiled environments."""
    
    config = create_default_config()
    config.dataset = dataset
    config.n_queries = n_queries
    config.max_depth = max_depth
    config.seed = seed
    config.verbose = verbose
    config.skip_unary_actions = False  # Critical for parity
    
    # Set seeds
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*80}")
    print(f"ENVIRONMENT COMPILED PARITY TEST")
    print(f"{'='*80}")
    print(f"Dataset: {dataset}")
    print(f"Queries: {n_queries}")
    print(f"Max depth: {max_depth}")
    print(f"skip_unary_actions: {config.skip_unary_actions}")
    print(f"Compile mode: {compile_mode}")
    print(f"{'='*80}\n")
    
    # Setup both environments
    base_engine, vec_engine, im, debug_helper, dh = setup_environments(config)
    
    # Prepare queries
    queries = prepare_queries(config)
    print(f"Testing {len(queries)} queries...\n")
    
    # Create compiled environment (tensor env is created per-query in run_query_trace_both)
    compiled_env = create_compiled_env(vec_engine, im, config, 1)
    
    # Optionally compile
    if compile_mode:
        print("Compiling environment step function...")
        compiled_env.step_functional = torch.compile(
            compiled_env.step_functional,
            mode='default', dynamic=False,
        )
        print("Compilation complete.\n")
    
    # Run all queries
    all_tensor_traces = []
    all_compiled_traces = []
    
    for i, (split, query) in enumerate(queries):
        tensor_trace, compiled_trace = run_query_trace_both(
            query, base_engine, compiled_env, im, debug_helper, config
        )
        all_tensor_traces.append(tensor_trace)
        all_compiled_traces.append(compiled_trace)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(queries)} queries...")
    
    # Compare all traces
    matches, mismatches = compare_all_traces(all_tensor_traces, all_compiled_traces, queries)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Total queries:  {len(queries)}")
    print(f"Matches:        {matches}")
    print(f"Mismatches:     {len(mismatches)}")
    print(f"{'='*80}")
    
    if mismatches:
        print(f"\nFirst {min(5, len(mismatches))} mismatches:")
        for idx, query_str, error in mismatches[:5]:
            print(f"\n  Query {idx}: {query_str}")
            print(f"    {error}")
    
    all_passed = len(mismatches) == 0
    
    if all_passed:
        print(f"\n✓ ALL {len(queries)} QUERIES MATCH")
    else:
        print(f"\n✗ {len(mismatches)} QUERIES HAVE MISMATCHES")
    
    return all_passed, {
        'total': len(queries),
        'matches': matches,
        'mismatches': mismatches,
    }


# ============================================================================
# Pytest Fixtures and Tests
# ============================================================================

@pytest.fixture(scope="module")
def base_config():
    return create_default_config()


class TestEnvCompiledParity:
    """Test that compiled environment produces same results as tensor environment."""
    
    @pytest.mark.parametrize("dataset,n_queries", [
        ("countries_s3", 50),
        ("countries_s3", 111),  # All queries
        ("family", 50),
        ("family", 200),
    ])
    def test_env_traces_match(self, dataset: str, n_queries: int, base_config):
        """Test that tensor and compiled environments produce identical traces."""
        passed, results = run_parity_test(
            dataset=dataset,
            n_queries=n_queries,
            max_depth=20,
            seed=base_config.seed,
            compile_mode=False,
        )
        
        if not passed:
            pytest.fail(f"Found {len(results['mismatches'])} mismatches out of {results['total']} queries")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test environment compiled parity')
    parser.add_argument('--dataset', type=str, default='countries_s3')
    parser.add_argument('--n-queries', type=int, default=50)
    parser.add_argument('--max-depth', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    passed, results = run_parity_test(
        dataset=args.dataset,
        n_queries=args.n_queries,
        max_depth=args.max_depth,
        seed=args.seed,
        compile_mode=args.compile,
        verbose=args.verbose,
    )
    
    sys.exit(0 if passed else 1)
