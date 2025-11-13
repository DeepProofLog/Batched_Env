"""
Test to compare derived states between str engine and tensor engine for countries_s3.
"""
import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

import random
import torch
from typing import List

# String-engine stack
from str_based.str_dataset import DataHandler as StrDataHandler
from str_based.str_index_manager import IndexManager as StrIndexManager
from str_based.str_utils import Term as StrTerm, Rule as StrRule
from str_based.str_unification import get_next_unification_python

# Tensor-engine stack
from data_handler import DataHandler
from index_manager import IndexManager
from unification_engine import UnificationEngine
from debug_helper import DebugHelper


DATASET = "wn18rr"

def canonicalize_str_state(state: List[StrTerm]) -> str:
    """Convert state to canonical string representation with canonicalized variables.
    
    Atoms are sorted by structure (predicate and constant positions), then variables
    are renumbered by order of first appearance in this sorted sequence.
    """
    # Pre-compute sort keys and canonical strings in a single pass
    # This avoids multiple iterations and reduces string operations
    atoms_data = []
    for term in state:
        # Build sort key by replacing variables with 'V'
        sort_key_parts = [term.predicate, '(']
        for i, arg in enumerate(term.args):
            if i > 0:
                sort_key_parts.append(',')
            if isinstance(arg, str) and arg.startswith('Var'):
                sort_key_parts.append('V')
            else:
                sort_key_parts.append(str(arg))
        sort_key_parts.append(')')
        sort_key = ''.join(sort_key_parts)
        atoms_data.append((sort_key, term))
    
    # Sort by structure
    atoms_data.sort(key=lambda x: x[0])
    
    # Find variables in order of first appearance in sorted atoms
    var_mapping = {}
    next_var_num = 1
    for _, term in atoms_data:
        for arg in term.args:
            if isinstance(arg, str) and arg.startswith('Var') and arg not in var_mapping:
                var_mapping[arg] = f"Var_{next_var_num}"
                next_var_num += 1
    
    # Build final canonical strings (single pass, pre-allocated list)
    atoms_str = []
    for _, term in atoms_data:
        # Build atom string directly without intermediate list
        atom_parts = [term.predicate, '(']
        for i, arg in enumerate(term.args):
            if i > 0:
                atom_parts.append(',')
            if isinstance(arg, str) and arg.startswith('Var'):
                atom_parts.append(var_mapping[arg])
            else:
                atom_parts.append(str(arg))
        atom_parts.append(')')
        atoms_str.append(''.join(atom_parts))
    
    # Sort final canonical atoms
    atoms_str.sort()
    return '|'.join(atoms_str)



def canonicalize_state_by_appearance(state: torch.Tensor, constant_no: int, debug_helper=None) -> torch.Tensor:
    """Canonicalize variables in state by order of first appearance.
    
    Atoms are sorted by structure (predicate and constant positions), then variables
    are renumbered by order of first appearance in this sorted sequence.
    """
    # Find non-padding atoms using vectorized operation
    non_padding_mask = state[:, 0] != 0
    n_atoms = non_padding_mask.sum().item()
    
    if n_atoms == 0:
        return state
    
    # Extract non-padding atoms - work directly with torch tensors for speed
    atoms = state[:n_atoms]
    
    # Create sort keys with sentinel for variables (vectorized)
    # Use 999999 as sentinel for variables to group them together in sorting
    sort_keys = atoms.clone()
    sort_keys[:, 1] = torch.where(atoms[:, 1] > constant_no, 999999, atoms[:, 1])
    sort_keys[:, 2] = torch.where(atoms[:, 2] > constant_no, 999999, atoms[:, 2])
    
    # Sort using argsort on computed keys (p, k1, k2)
    # Create a combined key for sorting: p * 1e12 + k1 * 1e6 + k2
    combined_keys = sort_keys[:, 0] * 1000000000000 + sort_keys[:, 1] * 1000000 + sort_keys[:, 2]
    sort_indices = torch.argsort(combined_keys)
    sorted_atoms = atoms[sort_indices]
    
    # Find unique variables in order of first appearance
    # Flatten arguments and filter for variables
    args = sorted_atoms[:, 1:].flatten()
    var_mask = args > constant_no
    vars_in_order = args[var_mask]
    
    # Get unique variables while preserving order - use dict for O(1) lookup
    seen = {}
    var_order = []
    for v in vars_in_order.tolist():
        if v not in seen:
            seen[v] = len(var_order)
            var_order.append(v)
    
    # Create mapping tensor for fast vectorized lookup
    if var_order:
        max_var = max(var_order)
        var_map = torch.zeros(max_var + 1, dtype=torch.long, device=state.device)
        for new_idx, old_var in enumerate(var_order):
            var_map[old_var] = constant_no + 1 + new_idx
        
        # Apply mapping using vectorized operations
        canonical_atoms = sorted_atoms.clone()
        for col in [1, 2]:
            var_mask = canonical_atoms[:, col] > constant_no
            if var_mask.any():
                canonical_atoms[var_mask, col] = var_map[canonical_atoms[var_mask, col]]
    else:
        canonical_atoms = sorted_atoms.clone()
    
    # Final sort by canonical representation using combined key
    final_keys = canonical_atoms[:, 0] * 1000000000000 + canonical_atoms[:, 1] * 1000000 + canonical_atoms[:, 2]
    final_indices = torch.argsort(final_keys)
    
    result = state.clone()
    result[:n_atoms] = canonical_atoms[final_indices]
    
    return result

def convert_state_to_str(state: torch.Tensor, debug_helper) -> str:
    """Convert tensor state to string representation."""
    valid = state[:, 0] != 0  # assuming padding_idx = 0
    atoms_str = []
    for i in range(valid.sum().item()):
        atom_str = debug_helper.atom_to_str(state[i])
        atoms_str.append(atom_str)
    atoms_str.sort()
    return '|'.join(atoms_str)

def atom_to_str_canonical(atom_idx: torch.LongTensor, debug_helper, constant_no: int, 
                          idx2predicate_cache: list, idx2constant_cache: list) -> str:
    """Convert an atom index tensor to string, using Var_1, Var_2, etc. for variables > constant_no.
    
    Uses pre-fetched caches to avoid repeated dict/list lookups.
    """
    p, a, b = atom_idx[0].item(), atom_idx[1].item(), atom_idx[2].item()
    ps = idx2predicate_cache[p] if 0 <= p < len(idx2predicate_cache) else str(p)

    # Special case for True and False predicates - they don't have meaningful arguments
    if ps in ['True', 'False']:
        return f"{ps}()"

    # Inline term_str logic to avoid function call overhead
    # Convert a
    if 1 <= a <= constant_no:
        a_str = idx2constant_cache[a] if 0 <= a < len(idx2constant_cache) else f"c{a}"
    elif a > constant_no:
        a_str = f"Var_{a - constant_no}"
    else:
        a_str = f"_{a}"
    
    # Convert b
    if 1 <= b <= constant_no:
        b_str = idx2constant_cache[b] if 0 <= b < len(idx2constant_cache) else f"c{b}"
    elif b > constant_no:
        b_str = f"Var_{b - constant_no}"
    else:
        b_str = f"_{b}"

    return f"{ps}({a_str},{b_str})"

def canonicalize_tensor_state(state: torch.Tensor, debug_helper, constant_no: int) -> str:
    """Convert tensor state to canonical string representation after canonicalizing vars by appearance."""
    # First canonicalize
    canonical_state = canonicalize_state_by_appearance(state, constant_no, debug_helper)
    
    # Pre-fetch lookup tables once to avoid repeated dict/list access
    idx2predicate_cache = debug_helper.idx2predicate if debug_helper.idx2predicate else []
    idx2constant_cache = debug_helper.idx2constant if debug_helper.idx2constant else []
    
    # Find number of valid atoms
    n_atoms = (canonical_state[:, 0] != 0).sum().item()
    
    # Build strings in a single pass with pre-allocated list
    atoms_str = []
    for i in range(n_atoms):
        atom_str = atom_to_str_canonical(canonical_state[i], debug_helper, constant_no,
                                         idx2predicate_cache, idx2constant_cache)
        atoms_str.append(atom_str)
    
    atoms_str.sort()
    return '|'.join(atoms_str)


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


def test_single_query(p: str, h: str, t: str, str_engine_data, tensor_engine_data, split='train', verbose=False):
    """Test a single query with both engines, comparing at each step."""
    dh_str, im_str, fact_index_str, rules_by_pred = str_engine_data
    dh_non, im_non, engine, debug_helper, next_var_start = tensor_engine_data
    
    if verbose:
        print(f"\nQuery: {p}({h}, {t})")
    
    # Convert facts list to frozenset for O(1) lookup performance
    facts_set_str = frozenset(dh_str.facts)
    
    # Setup for both engines
    q_str = StrTerm(predicate=p, args=(h, t))
    query_tensor = im_non.atom_to_tensor(p, h, t)
    query_padded_tensor = query_tensor.unsqueeze(0).unsqueeze(0)
    padding = torch.zeros(1, 19, 3, dtype=torch.long, device='cpu')
    padding[:, :, 0] = engine.padding_idx
    query_padded = torch.cat([query_padded_tensor, padding], dim=1)
    
    # For training queries, exclude the query from facts to test actual proof mechanism
    excluded_fact_str = q_str if split == 'train' else None
    excluded_query_tensor = query_padded if split == 'train' else None
    
    # String engine proof functions
    def str_get_derived(state):
        branch_next_states, _ = get_next_unification_python(
            state, facts_set_str, fact_index_str, rules_by_pred, 
            excluded_fact=excluded_fact_str, verbose=0, next_var_index=1,
            max_derived_states=500
        )
        valid = []
        for s in branch_next_states:
            if s and not any(term.predicate == 'False' for term in s) and len(s) <= 20:
                valid.append(s)
        return valid
    
    def str_is_true(state):
        return all(term.predicate == 'True' for term in state)
    
    def str_is_false(state):
        return any(term.predicate == 'False' for term in state)
    
    # Tensor engine proof functions
    # Initialize next_var to skip past all template variables
    # Template variables are used in rule templates and occupy indices [constant_no+1, max_template_var]
    # Runtime variables for proof search must start AFTER the template space to avoid collisions
    next_var_tracker = torch.tensor([next_var_start], dtype=torch.long, device='cpu')
    
    def tensor_get_derived(state, next_var):
        if state.dim() == 2:
            state = state.unsqueeze(0)
        
        # Exclude training queries from facts to test actual proof mechanism
        derived, derived_counts, updated_next_var = engine.get_derived_states(
            state, next_var,
            excluded_queries=excluded_query_tensor, verbose=0,
            max_derived_per_state=500
        )
        num_derived = derived_counts[0].item()
        valid = []
        for i in range(num_derived):
            s = derived[0, i]
            if not engine.is_false_state(s):
                non_padding = (s[:, 0] != engine.padding_idx).sum().item()
                if non_padding <= 20:
                    valid.append(s.unsqueeze(0))
        return valid, updated_next_var
    
    def tensor_is_true(state):
        # is_true_state expects 2D tensor [max_atoms, 3]
        if state.dim() == 3:
            state = state.squeeze(0)
        return engine.is_true_state(state)
    
    def tensor_is_false(state):
        # is_false_state expects 2D tensor [max_atoms, 3]
        if state.dim() == 3:
            state = state.squeeze(0)
        return engine.is_false_state(state)
    
    def tensor_canonicalize(state):
        s = state.squeeze(0) if state.dim() > 2 else state
        return canonicalize_tensor_state(s, debug_helper, engine.constant_no)
    
    # === Run both proofs in parallel, comparing at each step ===
    str_state = [q_str]
    tensor_state = query_padded
    max_depth = 10
    
    for step in range(max_depth):
        # Check if both are done
        str_done = str_is_true(str_state)
        tensor_done = tensor_is_true(tensor_state)
        
        if str_done and tensor_done:
            if verbose:
                print(f"  ✓ Both proved at step {step}")
            return True, True, True, "match"
        
        if str_done != tensor_done:
            error_msg = f"\n{'='*60}\nSUCCESS MISMATCH at STEP {step} for query: {p}({h}, {t})\n{'='*60}\n"
            error_msg += f"String engine:  is_true={str_done}\n"
            error_msg += f"Tensor engine:  is_true={tensor_done}\n"
            raise ValueError(error_msg)
        
        # Get derived states for both
        str_derived = str_get_derived(str_state)
        tensor_derived, next_var_tracker = tensor_get_derived(tensor_state, next_var_tracker)
        
        # Check if both have no derived states (dead end)
        if not str_derived and not tensor_derived:
            if verbose:
                print(f"  ✓ Both failed at step {step} (no derived states)")
            return True, False, False, "match"
        
        # Check length mismatch
        if len(str_derived) != len(tensor_derived):
            error_msg = f"\n{'='*60}\nLENGTH MISMATCH at STEP {step} for query: {p}({h}, {t})\n{'='*60}\n"
            error_msg += f"Current state at step {step}:\n"
            error_msg += f"  Str:    {str_state}\n"
            error_msg += f"  Tensor: {convert_state_to_str(tensor_state.squeeze(0) if tensor_state.dim() > 2 else tensor_state, debug_helper)}\n\n"
            error_msg += f"String engine: {len(str_derived)} derived states\n"
            error_msg += f"Tensor engine: {len(tensor_derived)} derived states\n"
            error_msg += f"\n[ORIG] Str states:\n"
            for i, s in enumerate(str_derived):
                error_msg += f"  {i}: {s}\n"
            error_msg += f"\n[ORIG] Tensor states:\n"
            for i, s in enumerate(tensor_derived):
                error_msg += f"  {i}: {convert_state_to_str(s.squeeze(0) if s.dim() > 2 else s, debug_helper)}\n"
            raise ValueError(error_msg)
        
        # Canonicalize and compare
        canon_str = [canonicalize_str_state(s) for s in str_derived]
        canon_str.sort()
        
        canon_tensor = [tensor_canonicalize(s) for s in tensor_derived]
        canon_tensor.sort()
        
        if canon_str != canon_tensor:
            error_msg = f"\n{'='*60}\nCANONICALIZATION MISMATCH at STEP {step} for query: {p}({h}, {t})\n{'='*60}\n"
            error_msg += f"Current state at step {step}:\n"
            error_msg += f"  Str:    {str_state}\n"
            error_msg += f"  Tensor: {convert_state_to_str(tensor_state.squeeze(0) if tensor_state.dim() > 2 else tensor_state, debug_helper)}\n\n"
            error_msg += f"[CANON] Str:    {canon_str}\n"
            error_msg += f"[CANON] Tensor: {canon_tensor}\n\n"
            error_msg += f"[ORIG] Str:\n"
            for i, s in enumerate(str_derived):
                error_msg += f"  {i}: {s}\n"
            error_msg += f"\n[ORIG] Tensor:\n"
            for i, s in enumerate(tensor_derived):
                error_msg += f"  {i}: {convert_state_to_str(s.squeeze(0) if s.dim() > 2 else s, debug_helper)}\n"
            raise ValueError(error_msg)
        
        if verbose and step == 0:
            print(f"  ✓ Step {step}: Canonicalization matches ({len(canon_str)} states)")
        
        # Choose first canonical state for both (they should choose the same)
        str_state_canon = [(canonicalize_str_state(s), s) for s in str_derived]
        str_state_canon.sort(key=lambda x: x[0])
        str_state = str_state_canon[0][1]
        
        tensor_state_canon = [(tensor_canonicalize(s), s) for s in tensor_derived]
        tensor_state_canon.sort(key=lambda x: x[0])
        tensor_state = tensor_state_canon[0][1]
    
    # Reached max depth - check final state
    str_success = str_is_true(str_state)
    tensor_success = tensor_is_true(tensor_state)
    
    if str_success != tensor_success:
        error_msg = f"\n{'='*60}\nSUCCESS MISMATCH after MAX DEPTH for query: {p}({h}, {t})\n{'='*60}\n"
        error_msg += f"String engine:  success={str_success}\n"
        error_msg += f"Tensor engine:  success={tensor_success}\n"
        raise ValueError(error_msg)
    
    if verbose:
        print(f"  ✓ Both reached max depth with same result: success={str_success}")
    
    return True, str_success, tensor_success, "match"


def main():
    import sys
    
    # Parse command-line arguments
    start_query = 0
    if len(sys.argv) > 1:
        try:
            start_query = int(sys.argv[1])
            print(f"Starting from query index: {start_query}")
        except ValueError:
            print(f"Invalid query index '{sys.argv[1]}', starting from 0")
            start_query = 0
    
    # Reproducibility
    random.seed(42)
    torch.manual_seed(42)

    print("Loading data...")

    # Str engine setup
    dh_str = StrDataHandler(
        dataset_name=DATASET,
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

    # Tensor engine setup
    dh_non = DataHandler(
        dataset_name=DATASET,
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
    
    engine = UnificationEngine.from_index_manager(im_non, take_ownership=True, stringifier_params=stringifier_params)

    debug_helper = DebugHelper(
        verbose=0,
        idx2predicate=im_non.idx2predicate,
        idx2constant=im_non.idx2constant,
        idx2template_var=im_non.idx2template_var,
        padding_idx=engine.padding_idx,
        n_constants=im_non.constant_no
    )

    # Compute max template variable index for initializing next_var during proof search
    # Template variables are stored in engine.rules_idx
    max_template_var = engine.constant_no
    if engine.rules_idx.numel() > 0:
        rule_max = engine.rules_idx.max().item()
        if rule_max > max_template_var:
            max_template_var = rule_max
    next_var_start_for_proofs = max_template_var + 1

    # Package engine data for testing
    str_engine_data = (dh_str, im_str, fact_index_str, rules_by_pred)
    tensor_engine_data = (dh_non, im_non, engine, debug_helper, next_var_start_for_proofs)

    # Collect all queries to test
    all_queries = []
    for q in dh_str.train_queries:
        all_queries.append(("train", q.predicate, q.args[0], q.args[1]))
    for q in dh_str.valid_queries:
        all_queries.append(("valid", q.predicate, q.args[0], q.args[1]))
    for q in dh_str.test_queries:
        all_queries.append(("test", q.predicate, q.args[0], q.args[1]))

    total_queries = len(all_queries)
    queries_to_test = total_queries - start_query
    
    print(f"\nTesting {total_queries} total queries:")
    print(f"  Train: {len(dh_str.train_queries)}")
    print(f"  Valid: {len(dh_str.valid_queries)}")
    print(f"  Test: {len(dh_str.test_queries)}")
    if start_query > 0:
        print(f"  Starting from query index {start_query} (skipping first {start_query} queries)")
        print(f"  Will test {queries_to_test} queries (indices {start_query} to {total_queries-1})")
    print()

    # Statistics
    stats = {
        "tested": 0,
        "match": 0,
        "length_mismatch": 0,
        "canonicalization_mismatch": 0,
        "success_mismatch": 0,
        "steps_mismatch": 0,
    }
    
    failed_queries = []
    
    # Test each query
    for query_idx, (split, p, h, t) in enumerate(all_queries):
        # Skip queries before start_query
        if query_idx < start_query:
            continue
            
        stats["tested"] += 1
        
        # Test query (verbose for first 5 after start)
        verbose = stats["tested"] <= 5
        
        # This will raise an error immediately on any mismatch
        match, str_success, tensor_success, reason = test_single_query(
            p, h, t, str_engine_data, tensor_engine_data, split=split, verbose=verbose
        )
        
        stats["match"] += 1
        if not verbose:
            if stats["tested"] % 10 == 0:
                print(f"  Tested {stats['tested']}/{queries_to_test} queries (current index: {query_idx})... all match so far")

    # Print summary - only reached if all tests pass
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print(f"Total queries tested: {stats['tested']}")
    if start_query > 0:
        print(f"Query range: {start_query} to {total_queries-1}")
    print(f"All {stats['match']} queries have identical behavior in both engines:")
    print(f"  - Same derived states")
    print(f"  - Same canonicalization")
    print(f"  - Same proof success/failure")
    print(f"  - Same number of proof steps")
    print("\nBoth string and tensor engines are equivalent!")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())