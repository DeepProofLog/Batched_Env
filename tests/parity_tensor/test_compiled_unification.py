"""
Compiled Unification Engine Parity Tests.

Tests verifying that UnificationEngineVectorized.get_derived_states_compiled()
produces EXACTLY the same results as UnificationEngine.get_derived_states().

Usage:
    # Run with 50 queries in eager mode
    pytest tests/parity/test_unification_compiled_parity.py -v -s
    
    # Run with torch.compile enabled
    pytest tests/parity/test_unification_compiled_parity.py -v -s --compile
    
    # CLI: 
    python tests/parity/test_unification_compiled_parity.py --dataset countries_s3 --n-queries 111
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

# Setup paths
ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_handler import DataHandler
from index_manager import IndexManager
from tensor.tensor_unification import UnificationEngine
from unification import UnificationEngineVectorized
from unification import UnificationEngineVectorized
import utils.utils as utils_funcs
import re


def canonicalize_state(state_str: str) -> str:
    """
    Normalize variable names in a state string for structural comparison.
    
    Variables like Var_2972, Var_2973 are renamed to V0, V1, V2...
    in order of first appearance. This enables comparing states that
    are structurally identical but have different variable numbering.
    
    Example:
        "father(Var_2972,Var_2973)|sister(Var_2973,353)"
        -> "father(V0,V1)|sister(V1,353)"
    """
    if not state_str:
        return state_str
    
    # Find all Var_N patterns
    var_pattern = r'Var_\d+'
    vars_found = re.findall(var_pattern, state_str)
    
    # Build mapping (first appearance order)
    var_map = {}
    counter = 0
    for v in vars_found:
        if v not in var_map:
            var_map[v] = f"V{counter}"
            counter += 1
    
    # Replace all variables with canonical names
    result = state_str
    for old, new in var_map.items():
        result = result.replace(old, new)
    
    return result


# ============================================================================
# Configuration
# ============================================================================

def create_default_config() -> SimpleNamespace:
    """Default parameters for parity tests."""
    return SimpleNamespace(
        dataset="countries_s3",
        data_path="./data/",
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        n_queries=50,
        max_depth=20,
        padding_atoms=6,
        padding_states=40,
        max_derived_per_state=40,
        max_total_runtime_vars=1000,
        seed=42,
        device="cpu",
        verbose=False,
    )


def clone_config(config: SimpleNamespace) -> SimpleNamespace:
    """Clone config for customization."""
    return SimpleNamespace(**vars(config))


# ============================================================================
# Engine Setup
# ============================================================================

def setup_engines(config: SimpleNamespace) -> Tuple[UnificationEngine, UnificationEngineVectorized, IndexManager, Dict]:
    """Setup both original and vectorized engines from the same data."""
    device = torch.device(config.device)
    
    dh = DataHandler(
        dataset_name=config.dataset,
        base_path=config.data_path,
        train_file=config.train_file,
        valid_file=config.valid_file,
        test_file=config.test_file,
        rules_file=config.rules_file,
        facts_file=config.facts_file,
        train_depth=None,
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
    
    original_engine = UnificationEngine.from_index_manager(
        im, take_ownership=False, stringifier_params=stringifier_params,
        max_derived_per_state=config.max_derived_per_state, sort_states=False
    )
    
    vectorized_engine = UnificationEngineVectorized.from_base_engine(
        original_engine, max_fact_pairs=None, max_rule_pairs=None,
        padding_atoms=config.padding_atoms, parity_mode=True,  # Enable exact matching
    )
    
    return original_engine, vectorized_engine, im, stringifier_params


def get_next_var_start(engine: UnificationEngine) -> int:
    """Compute starting variable index."""
    max_template_var = engine.constant_no
    if engine.rules_idx.numel() > 0:
        rule_max = engine.rules_idx.max().item()
        if rule_max > max_template_var:
            max_template_var = int(rule_max)
    return max_template_var + 1


# ============================================================================
# Query Preparation
# ============================================================================

def prepare_queries(config: SimpleNamespace) -> List[Tuple[str, Tuple[str, str, str]]]:
    """Prepare queries from dataset (following test_unification_parity pattern)."""
    dh = DataHandler(
        dataset_name=config.dataset,
        base_path=config.data_path,
        train_file=config.train_file,
        valid_file=config.valid_file,
        test_file=config.test_file,
        rules_file=config.rules_file,
        facts_file=config.facts_file,
        train_depth=None,
    )
    
    all_queries = []
    for q in dh.train_queries:
        all_queries.append(('train', (q.predicate, q.args[0], q.args[1])))
    for q in dh.valid_queries:
        all_queries.append(('valid', (q.predicate, q.args[0], q.args[1])))
    for q in dh.test_queries:
        all_queries.append(('test', (q.predicate, q.args[0], q.args[1])))
    
    # Shuffle and take first n
    rng = random.Random(config.seed)
    rng.shuffle(all_queries)
    
    if config.n_queries is not None:
        all_queries = all_queries[:config.n_queries]
    
    return all_queries


# ============================================================================
# Batched Parity Testing
# ============================================================================

def run_batched_parity_step(
    queries: List[Tuple[str, str, str]],
    original_engine: UnificationEngine,
    vectorized_engine: UnificationEngineVectorized,
    im: IndexManager,
    stringifier_params: Dict,
    config: SimpleNamespace,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run a batch of queries through both engines for a SINGLE step and compare.
    
    This is the batched version - we pass all queries as a single batch to both engines.
    
    Args:
        queries: List of (pred, head, tail) tuples
        
    Returns:
        (original_results, vectorized_results) - each is a list of result dicts per query
    """
    device = original_engine.device
    pad = original_engine.padding_idx
    next_var_start = get_next_var_start(original_engine)
    max_atoms = config.padding_atoms + 10
    B = len(queries)
    
    # Build batched initial states from queries
    states = torch.full((B, max_atoms, 3), pad, dtype=torch.long, device=device)
    for i, (pred, head, tail) in enumerate(queries):
        query_tensor = im.atom_to_tensor(pred, head, tail)
        states[i, 0] = query_tensor
    
    next_var = torch.full((B,), next_var_start, dtype=torch.long, device=device)
    
    # Initialize excluded_queries with the original queries (to prevent cycles)
    excluded_queries = torch.full((B, max_atoms, 3), pad, dtype=torch.long, device=device)
    excluded_queries[:, 0, :] = states[:, 0, :]  # First atom is the original query
    
    # Prepare excluded query in correct shape for vectorized engine: [B, 1, 3]
    excluded_for_vec = excluded_queries[:, 0:1, :]  # [B, 1, 3]
    
    # Run original engine (batched)
    orig_derived, orig_counts, orig_new_vars = original_engine.get_derived_states(
        states, next_var, excluded_queries=excluded_queries, verbose=0
    )
    
    # Run vectorized engine (batched)
    vec_derived, vec_counts, vec_new_vars = vectorized_engine.get_derived_states_compiled(
        states, next_var, excluded_queries=excluded_for_vec
    )
    
    # Collect results per query
    orig_results = []
    vec_results = []
    
    for i in range(B):
        state_str = utils_funcs.state_to_str(states[i], **stringifier_params)
        
        # Original engine results
        orig_count = orig_counts[i].item()
        orig_derived_strs = set()
        for k in range(orig_count):
            orig_derived_strs.add(utils_funcs.state_to_str(orig_derived[i, k], **stringifier_params))
        
        orig_results.append({
            'query_idx': i,
            'state': state_str,
            'num_derived': orig_count,
            'derived_states': sorted(orig_derived_strs),
        })
        
        # Vectorized engine results
        vec_count = vec_counts[i].item()
        vec_derived_strs = set()
        for k in range(vec_count):
            vec_derived_strs.add(utils_funcs.state_to_str(vec_derived[i, k], **stringifier_params))
        
        vec_results.append({
            'query_idx': i,
            'state': state_str,
            'num_derived': vec_count,
            'derived_states': sorted(vec_derived_strs),
        })
    
    return orig_results, vec_results


def run_multi_step_batched_parity(
    queries: List[Tuple[str, str, str]],
    original_engine: UnificationEngine,
    vectorized_engine: UnificationEngineVectorized,
    im: IndexManager,
    stringifier_params: Dict,
    config: SimpleNamespace,
) -> Tuple[List[List[Dict]], List[List[Dict]]]:
    """
    Run a batch of queries through both engines for MULTIPLE steps.
    
    At each step, we pass the entire batch to both engines simultaneously.
    Each query in the batch proceeds independently (taking action 0).
    
    Args:
        queries: List of (pred, head, tail) tuples
        
    Returns:
        (all_orig_traces, all_vec_traces) - traces for each query in the batch
    """
    device = original_engine.device
    pad = original_engine.padding_idx
    next_var_start = get_next_var_start(original_engine)
    max_atoms = config.padding_atoms + 10
    B = len(queries)
    
    # Build batched initial states from queries
    states = torch.full((B, max_atoms, 3), pad, dtype=torch.long, device=device)
    for i, (pred, head, tail) in enumerate(queries):
        query_tensor = im.atom_to_tensor(pred, head, tail)
        states[i, 0] = query_tensor
    
    next_vars = torch.full((B,), next_var_start, dtype=torch.long, device=device)
    
    # Initialize excluded_queries with the original queries (to prevent cycles)
    excluded_queries = torch.full((B, max_atoms, 3), pad, dtype=torch.long, device=device)
    excluded_queries[:, 0, :] = states[:, 0, :]  # First atom is the original query
    
    # Track which queries are still active (not terminal)
    active = torch.ones(B, dtype=torch.bool, device=device)
    
    # Initialize traces
    all_orig_traces = [[] for _ in range(B)]
    all_vec_traces = [[] for _ in range(B)]
    
    for step in range(config.max_depth):
        # Check if any queries are still active
        if not active.any():
            break
        
        # Check for terminal states per query
        query_preds = states[:, 0, 0]  # [B]
        is_empty = (query_preds == pad)
        is_true = (query_preds == original_engine.true_pred_idx) if original_engine.true_pred_idx is not None else torch.zeros_like(is_empty)
        is_false = (query_preds == original_engine.false_pred_idx) if original_engine.false_pred_idx is not None else torch.zeros_like(is_empty)
        is_terminal = is_empty | is_true | is_false
        
        # Record terminal states
        for i in range(B):
            if active[i] and is_terminal[i]:
                state_str = utils_funcs.state_to_str(states[i], **stringifier_params)
                result = 'TRUE' if is_true[i] else ('FALSE' if is_false[i] else 'EMPTY')
                all_orig_traces[i].append({'step': step, 'state': state_str, 'done': True, 'result': result})
                all_vec_traces[i].append({'step': step, 'state': state_str, 'done': True, 'result': result})
                active[i] = False
        
        # If no active queries left, break
        if not active.any():
            break
        
        # Prepare excluded query in correct shape for vectorized engine: [B, 1, 3]
        excluded_for_vec = excluded_queries[:, 0:1, :]  # [B, 1, 3]
        
        # Run original engine (batched)
        orig_derived, orig_counts, orig_new_vars = original_engine.get_derived_states(
            states, next_vars, excluded_queries=excluded_queries, verbose=0
        )
        
        # Run vectorized engine (batched)
        vec_derived, vec_counts, vec_new_vars = vectorized_engine.get_derived_states_compiled(
            states, next_vars, excluded_queries=excluded_for_vec
        )
        
        # Update max_atoms if needed
        M_out = orig_derived.shape[2]
        if M_out > max_atoms:
            new_states = torch.full((B, M_out, 3), pad, dtype=torch.long, device=device)
            new_states[:, :max_atoms, :] = states
            states = new_states
            
            new_excluded = torch.full((B, M_out, 3), pad, dtype=torch.long, device=device)
            new_excluded[:, :max_atoms, :] = excluded_queries[:, :max_atoms, :]
            excluded_queries = new_excluded
            max_atoms = M_out
        
        # Collect results and advance state for each active query
        for i in range(B):
            if not active[i]:
                continue
            
            state_str = utils_funcs.state_to_str(states[i], **stringifier_params)
            
            # Original engine results
            orig_count = orig_counts[i].item()
            orig_derived_strs = set()
            for k in range(orig_count):
                orig_derived_strs.add(utils_funcs.state_to_str(orig_derived[i, k], **stringifier_params))
            
            all_orig_traces[i].append({
                'step': step,
                'state': state_str,
                'num_derived': orig_count,
                'derived_states': sorted(orig_derived_strs),
                'done': False,
            })
            
            # Vectorized engine results
            vec_count = vec_counts[i].item()
            vec_derived_strs = set()
            for k in range(vec_count):
                vec_derived_strs.add(utils_funcs.state_to_str(vec_derived[i, k], **stringifier_params))
            
            all_vec_traces[i].append({
                'step': step,
                'state': state_str,
                'num_derived': vec_count,
                'derived_states': sorted(vec_derived_strs),
                'done': False,
            })
            
            # Take first derived state for next step (deterministic action)
            if orig_count == 0:
                active[i] = False
            else:
                # Update state to first derived state
                states[i] = pad
                states[i, :orig_derived.shape[2]] = orig_derived[i, 0]
                next_vars[i] = orig_new_vars[i]
    
    return all_orig_traces, all_vec_traces


# ============================================================================
# Trace Comparison
# ============================================================================

def compare_trace_step(step_orig: Dict, step_vec: Dict, step_idx: int) -> Tuple[bool, str]:
    """Compare a single step. Returns (match, error_message)."""
    if step_orig.get('state') != step_vec.get('state'):
        return False, f"Step {step_idx}: state mismatch:\n  Orig: {step_orig.get('state')}\n  Vec:  {step_vec.get('state')}"
    
    if step_orig.get('num_derived') != step_vec.get('num_derived'):
        return False, f"Step {step_idx}: num_derived mismatch: {step_orig.get('num_derived')} vs {step_vec.get('num_derived')}"
    
    # Compare derived states using canonical form (normalizes variable names)
    orig_derived = step_orig.get('derived_states', [])
    vec_derived = step_vec.get('derived_states', [])
    
    # With parity_mode + input_states seeding, variable names should match exactly
    # No canonicalization needed
    if set(orig_derived) != set(vec_derived):
        # Fall back to canonical for detailed error message
        orig_canonical = set(canonicalize_state(s) for s in orig_derived)
        vec_canonical = set(canonicalize_state(s) for s in vec_derived)
        if orig_canonical != vec_canonical:
            only_orig = orig_canonical - vec_canonical
            only_vec = vec_canonical - orig_canonical
            return False, f"Step {step_idx}: derived_states mismatch (structural):\n  Only orig: {list(only_orig)[:2]}\n  Only vec:  {list(only_vec)[:2]}"
        else:
            # Structural match but variable names differ
            only_orig = set(orig_derived) - set(vec_derived)
            only_vec = set(vec_derived) - set(orig_derived)
            return False, f"Step {step_idx}: variable naming mismatch:\n  Only orig: {list(only_orig)[:2]}\n  Only vec:  {list(only_vec)[:2]}"
    
    if step_orig.get('done') != step_vec.get('done'):
        return False, f"Step {step_idx}: done mismatch: {step_orig.get('done')} vs {step_vec.get('done')}"
    
    return True, ""


def compare_full_traces(trace_orig: List[Dict], trace_vec: List[Dict]) -> Tuple[bool, str]:
    """Compare full traces step-by-step. Returns (match, error_message)."""
    if len(trace_orig) != len(trace_vec):
        return False, f"Trace length mismatch: {len(trace_orig)} vs {len(trace_vec)}"
    
    for i, (step_o, step_v) in enumerate(zip(trace_orig, trace_vec)):
        match, error = compare_trace_step(step_o, step_v, i)
        if not match:
            return False, error
    
    return True, ""


def compare_all_traces(
    all_orig_traces: List[List[Dict]],
    all_vec_traces: List[List[Dict]],
    queries: List[Tuple[str, Tuple[str, str, str]]]
) -> Tuple[int, List[Tuple[int, str, str]]]:
    """Compare all query traces. Returns (num_matches, list of mismatches)."""
    mismatches = []
    matches = 0
    
    for i, (orig_trace, vec_trace, query) in enumerate(zip(all_orig_traces, all_vec_traces, queries)):
        split, (pred, head, tail) = query
        query_str = f"{pred}({head}, {tail}) [{split}]"
        
        match, error = compare_full_traces(orig_trace, vec_trace)
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
    batch_size: int = 100,
) -> Tuple[bool, Dict]:
    """Run parity test comparing original and vectorized engines.
    
    Args:
        dataset: Dataset name
        n_queries: Number of queries to test
        max_depth: Maximum proof depth
        seed: Random seed
        compile_mode: Whether to use torch.compile
        verbose: Enable verbose output
        batch_size: Number of queries to process in each batch (default: 100)
    """
    
    config = create_default_config()
    config.dataset = dataset
    config.n_queries = n_queries
    config.max_depth = max_depth
    config.seed = seed
    config.verbose = verbose
    
    # Set seeds
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*80}")
    print(f"COMPILED UNIFICATION ENGINE PARITY TEST (BATCHED)")
    print(f"{'='*80}")
    print(f"Dataset: {dataset}")
    print(f"Queries: {n_queries}")
    print(f"Batch size: {batch_size}")
    print(f"Max depth: {max_depth}")
    print(f"Compile mode: {compile_mode}")
    print(f"{'='*80}\n")
    
    # Setup engines
    original, vectorized, im, stringifier_params = setup_engines(config)
    
    # Optionally compile
    if compile_mode:
        print("Compiling vectorized engine...")
        vectorized.get_derived_states_compiled = torch.compile(
            vectorized.get_derived_states_compiled,
            mode='default', dynamic=False,
        )
        print("Compilation complete.\n")
    
    # Prepare queries
    queries = prepare_queries(config)
    print(f"Testing {len(queries)} queries in batches of {batch_size}...\n")
    
    # Run queries in batches
    all_orig_traces = []
    all_vec_traces = []
    
    total_queries = len(queries)
    for batch_start in range(0, total_queries, batch_size):
        batch_end = min(batch_start + batch_size, total_queries)
        batch_queries = queries[batch_start:batch_end]
        
        # Extract just the (pred, head, tail) tuples for batched processing
        batch_query_tuples = [q[1] for q in batch_queries]
        
        print(f"  Processing batch {batch_start//batch_size + 1}: queries {batch_start} to {batch_end}...")
        
        # Run batched parity test (multi-step)
        orig_traces, vec_traces = run_multi_step_batched_parity(
            batch_query_tuples, original, vectorized, im, stringifier_params, config
        )
        
        all_orig_traces.extend(orig_traces)
        all_vec_traces.extend(vec_traces)
    
    # Compare all traces
    matches, mismatches = compare_all_traces(all_orig_traces, all_vec_traces, queries)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Total queries:  {len(queries)}")
    print(f"Batches:        {(len(queries) + batch_size - 1) // batch_size}")
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


class TestCompiledUnificationParity:
    """Test that vectorized engine produces same results as original."""
    
    @pytest.mark.parametrize("dataset,n_queries", [
        ("countries_s3", 111),  # All queries
        ("family", 50),
        ("family", 200),
    ])
    def test_engine_traces_match(self, dataset: str, n_queries: int, base_config):
        """Test that original and vectorized engines produce identical traces."""
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
    
    parser = argparse.ArgumentParser(description='Test compiled unification parity')
    parser.add_argument('--dataset', type=str, default='family')
    parser.add_argument('--n-queries', type=int, default=600)
    parser.add_argument('--batch-size', type=int, default=200,
                        help='Number of queries to process in each batch (default: 100)')
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
        batch_size=args.batch_size,
    )
    
    sys.exit(0 if passed else 1)

