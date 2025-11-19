"""
Test to compare derived states between str engine and tensor engine for countries_s3.

Usage:
  # Test 10 queries
  python test_engines.py --n_queries 10
  
  # Start from specific query (useful for debugging)
  python test_engines.py --start_query 2047 --n_queries 1

Arguments:
  --dataset: Dataset name (default: family)
  --start_query: Starting query index (default: 0)
  --n_queries: Number of queries to test (default: all)
  --seed: Random seed for reproducibility (default: 42)
  --memory_pruning: Enable memory pruning (default: False)
  --skip_unary_actions: Enable skip unary actions (default: False)
"""
import os
import sys
import argparse
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_path)

import random
import torch
from typing import List

# String-engine stack
from sb3.dataset import DataHandler as StrDataHandler
from sb3.index_manager import IndexManager as StrIndexManager
from sb3.utils import Term as StrTerm, Rule as StrRule
from sb3.unification import get_next_unification_python, canonicalize_state_to_str, canonical_states_to_str

# Tensor-engine stack
from data_handler import DataHandler
from index_manager import IndexManager
from unification_engine import UnificationEngine
from debug_helper import DebugHelper


def _parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Compare string and tensor versions")
    parser.add_argument('--dataset', type=str, default="family",
                        help='Dataset name to use')
    parser.add_argument('--max_derived_states', type=int, default=500,
                        help='Maximum derived states to generate per step (default: 200)')
    parser.add_argument('--start_query', type=int, default=0,
                        help='Index of the first query to test (default: 0)')
    parser.add_argument('--n_queries', type=int, default=100,
                        help='Number of queries to test (default: None, meaning all)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling queries (default: 42)')
    parser.add_argument('--memory_pruning', action='store_true',
                        help='Enable memory pruning (default: False)')
    parser.add_argument('--skip_unary_actions', action='store_true',
                        help='Enable skip unary actions (default: False)')
    return parser.parse_args()



def load_str_engine(dataset: str):
    """Load and configure the string-based engine components.
    
    Returns:
        Tuple of (dh_str, im_str, fact_index_str, rules_by_pred)
    """
    dh_str = StrDataHandler(
        dataset_name=dataset,
        base_path="./data/",
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        train_depth=None,
    )

    im_str = StrIndexManager(
        constants=dh_str.constants,
        predicates=dh_str.predicates,
        max_total_vars=1000000,
        rules=dh_str.rules,
        padding_atoms=20,
        max_arity=dh_str.max_arity,
        device=torch.device('cpu'),
    )
    fact_index_str = im_str.build_fact_index(dh_str.facts)

    rules_by_pred: dict = {}
    for r in dh_str.rules:
        rules_by_pred.setdefault(r.head.predicate, []).append(r)
    
    return dh_str, im_str, fact_index_str, rules_by_pred


def load_tensor_engine(dataset: str, max_derived_states: int):
    """Load and configure the tensor-based engine components.
    
    Returns:
        Tuple of (dh_non, im_non, engine)
    """
    dh_non = DataHandler(
        dataset_name=dataset,
        base_path="./data/",
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        train_depth=None,
    )

    im_non = IndexManager(
        constants=dh_non.constants,
        predicates=dh_non.predicates,
        max_total_runtime_vars=1000000,
        padding_atoms=20,
        max_arity=dh_non.max_arity,
        device=torch.device('cpu'),
        rules=dh_non.rules,
    )
    dh_non.materialize_indices(im=im_non, device=torch.device('cpu'))

    # Create stringifier params for engine initialization
    stringifier_params = {
        'verbose': 0,
        'idx2predicate': im_non.idx2predicate,
        'idx2constant': im_non.idx2constant,
        'idx2template_var': im_non.idx2template_var,
        'padding_idx': im_non.padding_idx,
        'n_constants': im_non.constant_no
    }
    
    engine = UnificationEngine.from_index_manager(
        im_non, take_ownership=True, stringifier_params=stringifier_params,
        max_derived_per_state=max_derived_states,
        sort_states=True
    )
    
    return dh_non, im_non, engine


def print_debug_info(step: int, query, str_state, tensor_state, str_derived, tensor_derived, 
                     str_canonical, tensor_canonical, error_msg: str, debug_helper):
    """Print detailed debug information when a mismatch occurs."""
    print(f"\n{'='*60}")
    print(f"DEBUG INFO - STEP {step}")
    print(f"{'='*60}")
    print(f"\nQuery: {query}")
    print(f"\nCurrent State:")
    print(f"  Str:     {str_canonical}")
    print(f"  Tensor:  {tensor_canonical}")
    
    print(f"\nAvailable Actions: {len(str_derived)} (str) / {len(tensor_derived)} (tensor)")
    
    print(f"\nDerived States (first 20):")
    print(f"  Str:")
    for i, state in enumerate(str_derived[:20]):
        print(f"    [{i}] {state}")
    if len(str_derived) > 20:
        print(f"    ... ({len(str_derived) - 20} more)")
    
    print(f"  Tensor:")
    for i, state in enumerate(tensor_derived[:20]):
        print(f"    [{i}] {debug_helper.state_to_str(state.squeeze(0))}")
    if len(tensor_derived) > 20:
        print(f"    ... ({len(tensor_derived) - 20} more)")
    
    print(f"\nError Message:")
    print(f"  {error_msg}")
    print(f"{'='*60}\n")


def run_full_proof(initial_state, get_derived_fn, is_true_fn, is_false_fn, 
                   canonicalize_fn, max_depth=10, compare_fn=None):
    """Run full proof, choosing first canonical action. Works for both engines."""
    current_state = initial_state
    steps = 0
    
    while steps < max_depth:
        # Check if already proved
        if is_true_fn(current_state):
            return steps, True, None
        
        # Get derived states
        valid_derived = get_derived_fn(current_state)
        
        if not valid_derived:
            return steps, False, None
        
        # Canonicalize and sort - choose first canonical state
        state_canon = [(canonicalize_fn(s), s) for s in valid_derived]
        state_canon.sort(key=lambda x: x[0])
        
        # If compare function provided, call it for cross-engine comparison
        if compare_fn:
            compare_fn(steps, valid_derived, state_canon)
        
        chosen_state = state_canon[0][1]
        
        current_state = chosen_state
        steps += 1
    
    # Check final state
    success = is_true_fn(current_state)
    return steps, success, None


def test_single_query(p: str, h: str, t: str, 
                      str_engine_data, 
                      tensor_engine_data, 
                      facts_str,
                      verbose=False, 
                      max_derived_states: int = 500,
                      count=None):
    """Test a single query with both engines, comparing at each step.
    
    Returns: (match, str_success, tensor_success, reason, str_reward, tensor_reward)
    """
    dh_str, im_str, fact_index_str, rules_by_pred = str_engine_data
    dh_non, im_non, engine, next_var_start = tensor_engine_data
        
    if verbose:
        print(f"\n{'-'*60}\nQuery: {p}({h}, {t})")
    
    # Setup for both engines
    q_str = StrTerm(predicate=p, args=(h, t))
    query_tensor = im_non.atom_to_tensor(p, h, t)
    query_padded_tensor = query_tensor.unsqueeze(0).unsqueeze(0)
    padding = torch.zeros(1, 19, 3, dtype=torch.long, device='cpu')
    padding[:, :, 0] = engine.padding_idx
    query_padded = torch.cat([query_padded_tensor, padding], dim=1)
        
    # Initialize next_var to skip past all template variables
    # Template variables are used in rule templates and occupy indices [constant_no+1, max_template_var]
    # Runtime variables for proof search must start AFTER the template space to avoid collisions
    next_var_tracker = torch.tensor([next_var_start], dtype=torch.long, device='cpu')
       
    # === Run both proofs in parallel, comparing at each step ===
    str_state = [q_str]
    tensor_state = query_padded
    max_depth = 20
    
    for step in range(max_depth):
        # DEBUG: Print current state at beginning of each step
        if verbose:
            print(f"\n[DEBUG] === STEP {step} ===")
            # print(f"  Str state:    {str_state}")
            # print(f"  Tensor state: {state_to_str(tensor_state)}")
        
        # Check if both are done
        str_done = str_state[0].predicate == 'True' and len(str_state[0].args) == 0
        tensor_done = engine.is_true_state(tensor_state[0])
        
        if str_done and tensor_done:
            if verbose:
                print(f"  ✓ Both proved at step {step}")
            return True, True, True, "match"
        
        if str_done != tensor_done:
            error_msg = f"\n{'='*60}\nSUCCESS MISMATCH at STEP {step} for query: {p}({h}, {t})\n{'='*60}\n"
            error_msg += f"String engine:  is_true={str_done}\n"
            error_msg += f"Tensor engine:  is_true={tensor_done}\n\n"
            error_msg += f"Current states:\n"
            error_msg += f"  Str:    {str_state}\n"
            error_msg += f"  Tensor: {engine.deb.state_to_str(tensor_state.squeeze(0))}\n"
            raise ValueError(error_msg)
        
        # Get derived states string engine
        branch_next_states, _ = get_next_unification_python(
            str_state, facts_str, fact_index_str, rules_by_pred, 
            excluded_fact=q_str, verbose=0, next_var_index=1,
            max_derived_states=max_derived_states,
            canonical_order=True,
            index_manager=im_str
        )
        str_derived = []
        for s in branch_next_states:
            str_derived.append(s) 

        # Get derived states tensor engine
        # Note: deduplicate=False to match string engine behavior which doesn't deduplicate
        # Note: sort_states=True to use _sort_candidates_by_str_order for test compatibility
        derived, derived_counts, next_var_tracker = engine.get_derived_states(
            tensor_state, next_var_tracker,
            excluded_queries=query_padded, verbose=0
        )
        # Convert derived states to list of tensors
        num_derived = derived_counts[0].item()
        tensor_derived = []
        for i in range(num_derived):
            s = derived[0, i]
            tensor_derived.append(s.unsqueeze(0))
        
        # Canonicalization comparison using standardized canonical_states_to_str
        canon_str = canonical_states_to_str(str_derived)
        if len(tensor_derived) > 0:
            canon_tensor = engine.deb.canonical_states_to_str(torch.stack([s.squeeze(0) for s in tensor_derived]))
        else:
            raise ValueError("No derived states from tensor engine")

        
        # Check length mismatch
        if len(str_derived) != len(tensor_derived):
            str_canonical = canonicalize_state_to_str(str_state) if str_state else None
            tensor_canonical = engine.deb.state_to_str(tensor_state.squeeze(0))
            error_msg = f"\n{'='*60}\nLENGTH MISMATCH at STEP {step} for query: {p}({h}, {t})\n{'='*60}\n"
            print_debug_info(step, f"{p}({h}, {t})", str_state, tensor_state, str_derived, tensor_derived, 
                             str_canonical, tensor_canonical, error_msg, engine.deb)
            raise ValueError(error_msg)
        
        if canon_str != canon_tensor:
            str_canonical = canonicalize_state_to_str(str_state) if str_state else None
            tensor_canonical = engine.deb.state_to_str(tensor_state.squeeze(0))
            error_msg = f"\n{'='*60}\nCANONICALIZATION MISMATCH for query {count} at STEP {step}: {p}({h}, {t})\n{'='*60}\n"
            print_debug_info(step, f"{p}({h}, {t})", str_state, tensor_state, str_derived, tensor_derived, 
                             str_canonical, tensor_canonical, error_msg, engine.deb)
            raise ValueError(error_msg)
        
        if verbose:
            print(f"  ✓ Step {step}:")
            print(f"   [STR] Current state: {str_state}")
            print(f"\n   [TENSOR] Current state: {engine.deb.state_to_str(tensor_state.squeeze(0))}")
            print(f"\n\n   [STR] Derived states ({len(str_derived)}):")
            for i, s in enumerate(str_derived):
                print(f"           {i}: {s}")
            print(f"\n   [TENSOR] Derived states ({len(tensor_derived)}):")
            for i, s in enumerate(tensor_derived):
                print(f"           {i}: {engine.deb.state_to_str(s.squeeze(0))}")
        
        # Choose first canonical state for both (they should choose the same)
        _, str_indices = canonical_states_to_str(str_derived, return_indices=True)
        str_state = str_derived[str_indices[0]]

        
        # Use canonical_states_to_str with return_indices to select the same state as str_engine
        if len(tensor_derived) > 0:
            # tensor_derived contains tensors with shape [1, M, 3], squeeze batch dim before stacking
            tensor_states_squeezed = [s.squeeze(0) for s in tensor_derived]
            tensor_states_stacked = torch.stack(tensor_states_squeezed, dim=0)  # [N, M, 3]
            
            _, tensor_indices = engine.deb.canonical_states_to_str(tensor_states_stacked, return_indices=True)
            # Keep batch dimension [1, M, 3] for next iteration
            tensor_state = tensor_derived[tensor_indices[0]]
        else:
            raise ValueError("No derived states from tensor engine")
    
    # Reached max depth - check final state
    str_success =  str_state[0].predicate == 'True' and len(str_state[0].args) == 0
    tensor_success =  engine.is_true_state(tensor_state[0])
    
    if str_success != tensor_success:
        error_msg = f"\n{'='*60}\nSUCCESS MISMATCH after MAX DEPTH for query: {p}({h}, {t})\n{'='*60}\n"
        error_msg += f"String engine:  success={str_success}\n"
        error_msg += f"Tensor engine:  success={tensor_success}\n\n"
        error_msg += f"Final states:\n"
        error_msg += f"  Str:    {str_state}\n"
        error_msg += f"  Tensor: {engine.deb.state_to_str(tensor_state.squeeze(0))}\n"
        raise ValueError(error_msg)
    
    if verbose:
        print(f"  ✓ Both reached max depth with same result: success={str_success}")
    
    return True, str_success, tensor_success, "match"


def main():
    
    args = _parse_args()

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    SEED = args.seed

    print("Loading data...")

    # Load engines
    dh_str, im_str, fact_index_str, rules_by_pred = load_str_engine(args.dataset)
    dh_non, im_non, engine = load_tensor_engine(args.dataset, args.max_derived_states)

    # Compute max template variable index for initializing next_var during proof search
    # Template variables are stored in engine.rules_idx
    max_template_var = engine.constant_no
    if engine.rules_idx.numel() > 0:
        rule_max = engine.rules_idx.max().item()
        if rule_max > max_template_var:
            max_template_var = rule_max
    next_var_start_for_proofs = max_template_var + 1

    str_engine_data = (dh_str, im_str, fact_index_str, rules_by_pred)
    tensor_engine_data = (dh_non, im_non, engine, next_var_start_for_proofs)

    # Collect all queries to test
    all_queries = []
    for q in dh_str.train_queries:
        all_queries.append(("train", q.predicate, q.args[0], q.args[1]))
    for q in dh_str.valid_queries:
        all_queries.append(("valid", q.predicate, q.args[0], q.args[1]))
    for q in dh_str.test_queries:
        all_queries.append(("test", q.predicate, q.args[0], q.args[1]))

    # shuffle queries for randomness
    random.Random(SEED).shuffle(all_queries)
    
    # Apply start_query and n_queries filtering
    start_idx = args.start_query
    end_idx = start_idx + args.n_queries if args.n_queries is not None else len(all_queries)
    all_queries = all_queries[start_idx:end_idx]

    total_queries = len(all_queries)
    queries_to_test = total_queries

    
    print(f"\nTesting {total_queries} total queries:")
    print(f"  Train: {len(dh_str.train_queries)}")
    print(f"  Valid: {len(dh_str.valid_queries)}")
    print(f"  Test: {len(dh_str.test_queries)}")
    print(f"  Will test {queries_to_test} queries (indices {start_idx} to {start_idx + total_queries - 1})")
    print()

    # Statistics
    stats = {
        "tested": 0,
        "match": 0,
        "length_mismatch": 0,
        "canonicalization_mismatch": 0,
        "success_mismatch": 0,
        "steps_mismatch": 0,
        "str_proven": 0,
        "tensor_proven": 0,
        "both_proven": 0,
        "both_failed": 0,
    }
    facts_set_str = frozenset(dh_str.facts)
    for query_idx, (split, p, h, t) in enumerate(all_queries):
            
        stats["tested"] += 1
        
        # This will raise an error immediately on any mismatch
        match, str_success, tensor_success, reason = test_single_query(
            p, h, t, str_engine_data, tensor_engine_data, facts_set_str, verbose=False,
            max_derived_states=args.max_derived_states,
            count=query_idx
        )
        
        stats["match"] += bool(match)
        
        # Track proof success statistics
        # removed skipped logic: reason == "skipped" is no longer tracked
        if str_success is not None and tensor_success is not None:
            if str_success:
                stats["str_proven"] += 1
            if tensor_success:
                stats["tensor_proven"] += 1
            if str_success and tensor_success:
                stats["both_proven"] += 1
            if not str_success and not tensor_success:
                stats["both_failed"] += 1
        
        # Progress report every 10 queries
        if stats["tested"] % 10 == 0:
            print(f"  Tested {stats['tested']}/{queries_to_test} queries (current index: {query_idx})... all match so far")

    # Print summary - only reached if all tests pass
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print(f"Total queries tested: {stats['tested']}")
    # We always run the full query range
    print(f"Query range: 0 to {total_queries-1}")
    print(f"All {stats['match']} queries have identical behavior in both engines:")
    print(f"  - Same derived states")
    print(f"  - Same canonicalization")
    print(f"  - Same proof success/failure")
    print(f"  - Same number of proof steps")
    
    # Print proof success statistics
    # All tested queries are considered for success rate calculations
    non_skipped = stats["tested"]
    if non_skipped > 0:
        str_success_rate = stats["str_proven"] / non_skipped
        tensor_success_rate = stats["tensor_proven"] / non_skipped
        print("\n" + "="*60)
        print("PROOF SUCCESS STATISTICS")
        print("="*60)
        print(f"Queries proven (both engines): {stats['both_proven']}/{non_skipped} ({stats['both_proven']/non_skipped*100:.2f}%)")
        print(f"Queries failed (both engines):  {stats['both_failed']}/{non_skipped} ({stats['both_failed']/non_skipped*100:.2f}%)")
        print(f"String engine success rate:     {stats['str_proven']}/{non_skipped} ({str_success_rate*100:.2f}%)")
        print(f"Tensor engine success rate:     {stats['tensor_proven']}/{non_skipped} ({tensor_success_rate*100:.2f}%)")
        # Skipped queries are not tracked in this test harness
        print(f"\nAverage success rate: {(str_success_rate + tensor_success_rate) / 2 * 100:.2f}%")
        print(f"(Success rate = queries proven within max depth)")
        if str_success_rate > 0.001 and str_success_rate < 0.999:
            print(f"✓ Success rate is between 0% and 100% (non-trivial: {str_success_rate*100:.2f}%)")
    
    print("\nBoth string and tensor engines are equivalent!")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())