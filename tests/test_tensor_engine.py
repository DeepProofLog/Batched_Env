"""
Test module for tensor-based unification engine.

This module contains functions to test the tensor-based unification engine
with both deterministic (canonical) and random action selection.
Supports both non-batched and batched tensor engines.
"""
import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_path)

import random
import torch
from typing import List, Tuple, Dict

# Tensor-engine stack
from data_handler import DataHandler
from index_manager import IndexManager
from unification_engine import UnificationEngine
from debug_helper import DebugHelper


def canonicalize_state_by_appearance(state: torch.Tensor, constant_no: int, debug_helper=None) -> torch.Tensor:
    """Canonicalize variables in state by order of first appearance.
    
    IMPORTANT: This function ONLY renames variables to canonical names.
    It does NOT reorder atoms - atom order must be preserved for proof search!
    """
    # Find non-padding atoms using vectorized operation
    non_padding_mask = state[:, 0] != 0
    n_atoms = non_padding_mask.sum().item()
    
    if n_atoms == 0:
        return state
    
    # Extract non-padding atoms - work directly with torch tensors for speed
    atoms = state[:n_atoms]
    
    # Find unique variables in order of first appearance (row-major: atom by atom, arg by arg)
    # Flatten arguments and filter for variables
    args = atoms[:, 1:].flatten()  # Process in order: atom0_arg1, atom0_arg2, atom1_arg1, atom1_arg2, ...
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
        
        # Apply mapping using vectorized operations - preserve atom order!
        canonical_atoms = atoms.clone()
        for col in [1, 2]:
            var_mask = canonical_atoms[:, col] > constant_no
            if var_mask.any():
                canonical_atoms[var_mask, col] = var_map[canonical_atoms[var_mask, col]]
    else:
        canonical_atoms = atoms.clone()
    
    # Return with atom order preserved
    result = state.clone()
    result[:n_atoms] = canonical_atoms
    
    return result


def atom_to_str_canonical(atom_idx: torch.LongTensor, debug_helper, constant_no: int, 
                          idx2predicate_cache: list, idx2constant_cache: list) -> str:
    """Convert an atom index tensor to string, using Var_1, Var_2, etc. for variables > constant_no."""
    p, a, b = atom_idx[0].item(), atom_idx[1].item(), atom_idx[2].item()
    ps = idx2predicate_cache[p] if 0 <= p < len(idx2predicate_cache) else str(p)

    # Special case for True and False predicates
    if ps in ['True', 'False']:
        return f"{ps}()"

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
    
    # DO NOT SORT - preserve original atom order for proper comparison
    return '|'.join(atoms_str)


def setup_tensor_engine(dataset: str = "countries_s3", base_path: str = "./data/", batched: bool = False) -> Tuple:
    """
    Setup the tensor-based engine with dataset.
    
    Args:
        dataset: Dataset name
        base_path: Base path to data directory
        batched: If True, setup for batched operations
    
    Returns:
        (dh, im, engine, debug_helper, next_var_start)
    """
    dh_non = DataHandler(
        dataset_name=dataset,
        base_path=base_path,
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
    max_template_var = engine.constant_no
    if engine.rules_idx.numel() > 0:
        rule_max = engine.rules_idx.max().item()
        if rule_max > max_template_var:
            max_template_var = rule_max
    next_var_start_for_proofs = max_template_var + 1

    return dh_non, im_non, engine, debug_helper, next_var_start_for_proofs


def test_tensor_engine_single_query(
    query_tuple: Tuple[str, str, str],
    engine_data: Tuple,
    split: str = 'train',
    deterministic: bool = True,
    max_depth: int = 10,
    max_derived_states: int = 200,
    verbose: bool = False,
    seed: int = 42
) -> Dict:
    """
    Test a single query using the tensor engine.
    
    Args:
        query_tuple: (predicate, head, tail)
        engine_data: Tuple from setup_tensor_engine()
        split: 'train', 'valid', or 'test'
        deterministic: If True, use canonical ordering; if False, random actions
        max_depth: Maximum proof depth
        max_derived_states: Maximum derived states per step
        verbose: Print detailed information
        seed: Random seed for reproducible random actions
        
    Returns:
        Dict with keys:
            - success: bool
            - steps: int
            - reward: float (1.0 if success, 0.0 otherwise)
            - trace: List[Dict] with state, derived_states, action at each step
    """
    p, h, t = query_tuple
    dh_non, im_non, engine, debug_helper, next_var_start = engine_data
    
    if verbose:
        print(f"\nQuery: {p}({h}, {t}) [split={split}, deterministic={deterministic}]")
    
    # Setup query
    query_tensor = im_non.atom_to_tensor(p, h, t)
    query_padded_tensor = query_tensor.unsqueeze(0).unsqueeze(0)
    padding = torch.zeros(1, 19, 3, dtype=torch.long, device='cpu')
    padding[:, :, 0] = engine.padding_idx
    query_padded = torch.cat([query_padded_tensor, padding], dim=1)
    
    # For training queries, exclude the query from facts
    excluded_query_tensor = query_padded if split == 'train' else None
    
    # Initialize next_var tracker
    next_var_tracker = torch.tensor([next_var_start], dtype=torch.long, device='cpu')
    
    # Proof functions
    def tensor_get_derived(state, next_var):
        if state.dim() == 2:
            state = state.unsqueeze(0)
        
        derived, derived_counts, updated_next_var = engine.get_derived_states(
            state, next_var,
            excluded_queries=excluded_query_tensor, verbose=0,
            max_derived_per_state=max_derived_states
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
        if state.dim() == 3:
            state = state.squeeze(0)
        return engine.is_true_state(state)
    
    def tensor_is_false(state):
        if state.dim() == 3:
            state = state.squeeze(0)
        return engine.is_false_state(state)
    
    def tensor_canonicalize(state):
        s = state.squeeze(0) if state.dim() > 2 else state
        return canonicalize_tensor_state(s, debug_helper, engine.constant_no)
    
    # Run proof
    current_state = query_padded
    steps = 0
    trace = []
    rng = random.Random(seed)
    
    while steps < max_depth:
        # Check if already proved
        if tensor_is_true(current_state):
            trace.append({
                'step': steps,
                'state': tensor_canonicalize(current_state),
                'derived_states': [],
                'num_actions': 0,
                'action': None,
                'done': True
            })
            if verbose:
                print(f"  ✓ Proved at step {steps}")
            return {
                'success': True,
                'steps': steps,
                'reward': 1.0,
                'trace': trace
            }
        
        # Get derived states
        tensor_derived, next_var_tracker = tensor_get_derived(current_state, next_var_tracker)
        
        if not tensor_derived:
            trace.append({
                'step': steps,
                'state': tensor_canonicalize(current_state),
                'derived_states': [],
                'num_actions': 0,
                'action': None,
                'done': True
            })
            if verbose:
                print(f"  ✗ Failed at step {steps} (no derived states)")
            return {
                'success': False,
                'steps': steps,
                'reward': 0.0,
                'trace': trace
            }
        
        # Canonicalize and sort
        canon_states = [(tensor_canonicalize(s), s) for s in tensor_derived]
        canon_states.sort(key=lambda x: x[0])
        
        if deterministic:
            # Choose first canonical state
            chosen_idx = 0
        else:
            # Choose random state
            chosen_idx = rng.randint(0, len(canon_states) - 1)
        
        chosen_state = canon_states[chosen_idx][1]
        
        trace.append({
            'step': steps,
            'state': tensor_canonicalize(current_state),
            'derived_states': [c[0] for c in canon_states],
            'num_actions': len(tensor_derived),
            'action': chosen_idx,
            'done': False
        })
        
        if verbose and steps < 3:
            print(f"  Step {steps}: {len(tensor_derived)} actions, chose {chosen_idx}")
        
        current_state = chosen_state
        steps += 1
    
    # Reached max depth
    success = tensor_is_true(current_state)
    trace.append({
        'step': steps,
        'state': tensor_canonicalize(current_state),
        'derived_states': [],
        'num_actions': 0,
        'action': None,
        'done': True
    })
    
    if verbose:
        print(f"  Max depth reached: success={success}")
    
    return {
        'success': success,
        'steps': steps,
        'reward': 1.0 if success else 0.0,
        'trace': trace
    }


def test_tensor_engine_batch(
    queries: List[Tuple[str, Tuple[str, str, str]]],
    engine_data: Tuple,
    deterministic: bool = True,
    max_depth: int = 10,
    seed: int = 42,
    verbose: bool = False
) -> Dict:
    """
    Test multiple queries using the tensor engine.
    
    Args:
        queries: List of (split, (predicate, head, tail))
        engine_data: Tuple from setup_tensor_engine()
        deterministic: If True, use canonical ordering; if False, random actions
        max_depth: Maximum proof depth
        seed: Random seed for reproducible random actions
        verbose: Print detailed information
        
    Returns:
        Dict with keys:
            - total_queries: int
            - successful: int
            - avg_reward: float
            - avg_steps: float
            - traces: List of trace dicts
    """
    results = []
    
    for idx, (split, query_tuple) in enumerate(queries):
        result = test_tensor_engine_single_query(
            query_tuple, engine_data, split=split,
            deterministic=deterministic, max_depth=max_depth,
            verbose=verbose and idx < 3, seed=seed + idx
        )
        results.append(result)
        
        if not verbose and (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(queries)} queries...")
    
    # Aggregate statistics
    successful = sum(1 for r in results if r['success'])
    total_reward = sum(r['reward'] for r in results)
    total_steps = sum(r['steps'] for r in results)
    
    return {
        'total_queries': len(queries),
        'successful': successful,
        'avg_reward': total_reward / len(queries) if queries else 0.0,
        'avg_steps': total_steps / len(queries) if queries else 0.0,
        'traces': results
    }
