"""
Test module for SB3 (string-based) engine.

Simple and modular testing for the string-based unification engine.
"""
import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_path)

import random
import torch
from typing import Tuple, Dict, List

from str_based.str_dataset import DataHandler as StrDataHandler
from str_based.str_index_manager import IndexManager as StrIndexManager
from str_based.str_utils import Term as StrTerm
from str_based.str_unification import get_next_unification_python, canonicalize_state_to_str


def setup_sb3_engine(dataset: str = "countries_s3", base_path: str = "./data/") -> Tuple:
    """
    Setup the string-based (SB3) engine with dataset.
    
    Returns:
        (dh_str, im_str, fact_index_str, rules_by_pred, facts_set)
    """
    dh_str = StrDataHandler(
        dataset_name=dataset,
        base_path=base_path,
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

    # Convert facts list to frozenset for O(1) lookup performance
    facts_set_str = frozenset(dh_str.facts)

    return dh_str, im_str, fact_index_str, rules_by_pred, facts_set_str


def test_sb3_engine_single_query(
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
    Test a single query using the SB3 engine.
    
    Args:
        query_tuple: (predicate, head, tail)
        engine_data: Tuple from setup_sb3_engine()
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
    dh_str, im_str, fact_index_str, rules_by_pred, facts_set_str = engine_data
    
    if verbose:
        print(f"\nQuery: {p}({h}, {t}) [split={split}, deterministic={deterministic}]")
    
    # Setup query
    q_str = StrTerm(predicate=p, args=(h, t))
    
    # For training queries, exclude the query from facts to test actual proof mechanism
    excluded_fact_str = q_str if split == 'train' else None
    
    # Proof functions
    def str_get_derived(state):
        branch_next_states, _ = get_next_unification_python(
            state, facts_set_str, fact_index_str, rules_by_pred, 
            excluded_fact=excluded_fact_str, verbose=0, next_var_index=1,
            max_derived_states=max_derived_states
        )
        valid = []
        for s in branch_next_states:
            if s and not any(term.predicate == 'False' for term in s) and len(s) <= 100:
                valid.append(s)
        return valid
    
    def str_is_true(state):
        return all(term.predicate == 'True' for term in state)
    
    def str_is_false(state):
        return any(term.predicate == 'False' for term in state)
    
    # Run proof
    current_state = [q_str]
    steps = 0
    trace = []
    rng = random.Random(seed)
    
    while steps < max_depth:
        # Check if already proved
        if str_is_true(current_state):
            trace.append({
                'step': steps,
                'state': canonicalize_state_to_str(current_state),
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
        valid_derived = str_get_derived(current_state)
        
        if not valid_derived:
            trace.append({
                'step': steps,
                'state': canonicalize_state_to_str(current_state),
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
        
        # Canonicalize and sort - choose action
        state_canon = [(canonicalize_state_to_str(s), s) for s in valid_derived]
        state_canon.sort(key=lambda x: x[0])
        
        if deterministic:
            # Choose first canonical state
            chosen_idx = 0
        else:
            # Choose random state
            chosen_idx = rng.randint(0, len(state_canon) - 1)
        
        chosen_state = state_canon[chosen_idx][1]
        
        trace.append({
            'step': steps,
            'state': canonicalize_state_to_str(current_state),
            'derived_states': [c[0] for c in state_canon],
            'num_actions': len(valid_derived),
            'action': chosen_idx,
            'done': False
        })
        
        if verbose and steps < 3:
            print(f"  Step {steps}: {len(valid_derived)} actions, chose {chosen_idx}")
        
        current_state = chosen_state
        steps += 1
    
    # Reached max depth
    success = str_is_true(current_state)
    trace.append({
        'step': steps,
        'state': canonicalize_state_to_str(current_state),
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


def test_sb3_engine_batch(
    queries: List[Tuple[str, Tuple[str, str, str]]],
    engine_data: Tuple,
    deterministic: bool = True,
    max_depth: int = 10,
    seed: int = 42,
    verbose: bool = False
) -> Dict:
    """
    Test multiple queries using the SB3 engine.
    
    Args:
        queries: List of (split, (predicate, head, tail))
        engine_data: Tuple from setup_sb3_engine()
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
        result = test_sb3_engine_single_query(
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
