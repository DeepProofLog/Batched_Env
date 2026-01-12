"""
Test module for SB3 (string-based) engine.

Simple and modular testing for the string-based unification engine.
"""
import os
import sys

# Navigate from tests/parity_sb3/envs/ to kge_experiments/
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sb3_path = os.path.join(root_path, 'sb3')
# Ensure all SB3 modules resolve to the SAME top-level module paths.
# Mixing package and top-level imports creates distinct Term classes and breaks parity.
sys.path.insert(0, root_path)
sys.path.insert(0, sb3_path)

import random
import torch
from typing import Tuple, Dict, List
from types import SimpleNamespace

from sb3.sb3_dataset import DataHandler as StrDataHandler
from sb3.sb3_index_manager import IndexManager as StrIndexManager
from sb3.sb3_utils import Term as StrTerm
from sb3.sb3_unification import get_next_unification_python, state_to_str




def setup_sb3_engine(
    config: SimpleNamespace = None
) -> Tuple:
    """
    Setup the string-based (SB3) engine with dataset.
    
    Returns:
        (dh_str, im_str, fact_index_str, rules_by_pred, facts_set)
    """
    cfg = config
    base_path = getattr(cfg, 'base_path', "./data/")
    dataset = getattr(cfg, 'dataset', 'family')
    device_value = getattr(cfg, 'device', 'cpu')
    device = device_value if isinstance(device_value, torch.device) else torch.device(device_value)
    padding_atoms = getattr(cfg, 'padding_atoms', 100)
    max_total_vars = getattr(cfg, 'max_total_runtime_vars', 1_000_000)

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
        max_total_vars=max_total_vars,
        rules=dh_str.rules,
        padding_atoms=padding_atoms,
        max_arity=dh_str.max_arity,
        device=device,
    )
    fact_index_str = im_str.build_fact_index(dh_str.facts, deterministic=True)

    rules_by_pred: dict = {}
    for r in dh_str.rules:
        rules_by_pred.setdefault(r.head.predicate, []).append(r)

    # Convert facts list to frozenset for O(1) lookup performance
    facts_set_str = frozenset(dh_str.facts)

    return dh_str, im_str, fact_index_str, rules_by_pred, facts_set_str


def run_sb3_engine_single_query(
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
    
    # Setup query - use the Term class from the loaded facts to avoid class mismatch issues
    # The sb3_dataset may import Term as 'sb3_utils.Term' while we import 'sb3.sb3_utils.Term'
    # which creates different classes that don't compare equal
    TermClass = type(dh_str.facts[0]) if dh_str.facts else StrTerm
    q_str = TermClass(predicate=p, args=(h, t))
    
    # For training queries, exclude the query from facts to test actual proof mechanism
    excluded_fact_str = q_str if split == 'train' else None
    
   
    def str_is_true(state):
        return all(term.predicate == 'True' for term in state)
    
    def str_is_false(state):
        return any(term.predicate == 'False' for term in state)
    
    # Run proof
    current_state = [q_str]
    steps = 0
    trace = []
    rng = random.Random(seed)
    
    # Calculate correct next_var_index to match tensor engine
    # Use the IndexManager's computed next_var_start_for_proofs property
    # This ensures SB3 and tensor engines start runtime variables at the same index
    start_next_var_index = im_str.next_var_start_for_proofs
    if verbose:
        print(f"DEBUG: constant_no={im_str.constant_no}")
        print(f"DEBUG: max_template_var_idx={im_str.max_template_var_idx}")
        print(f"DEBUG: start_next_var_index={start_next_var_index}")
    
    current_next_var = start_next_var_index

    while steps < max_depth:
        # Check if already proved
        if str_is_true(current_state):
            trace.append({
                'step': steps,
                'state': state_to_str(current_state),
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
        valid_derived, updated_next_var = get_next_unification_python(
            current_state, facts_set_str, fact_index_str, rules_by_pred, 
            excluded_fact=excluded_fact_str, verbose=0, next_var_index=current_next_var,
            max_derived_states=max_derived_states,
            canonical_order=False, index_manager=im_str  # Natural ordering matches tensor engine
        )
        
        # In Tensor engine, next_var_indices is updated.
        # So we should update it here too.
        current_next_var = updated_next_var
        
        # States are in natural order (matching tensor engine)
        # Choose action based on deterministic flag
        if deterministic:
            chosen_idx = 0
        else:
            # Choose random state
            chosen_idx = rng.randint(0, len(valid_derived) - 1)
        
        chosen_state = valid_derived[chosen_idx]
        
        trace.append({
            'step': steps,
            'state': state_to_str(current_state),
            'derived_states': [state_to_str(s) for s in valid_derived],
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
        'state': state_to_str(current_state),
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


def run_sb3_engine(
    queries: List[Tuple[str, Tuple[str, str, str]]],
    engine_data: Tuple,
    config: SimpleNamespace
) -> Dict:
    """
    Test multiple queries using the SB3 engine.
    
    Args:
        queries: List of (split, (predicate, head, tail))
        engine_data: Tuple from setup_sb3_engine()
        config: Configuration namespace with deterministic, max_depth, seed, verbose, etc.
        
    Returns:
        Dict with keys:
            - total_queries: int
            - successful: int
            - avg_reward: float
            - avg_steps: float
            - traces: List of trace dicts
    """
    deterministic = config.deterministic if hasattr(config, 'deterministic') else True
    max_depth = config.max_depth
    seed = config.seed
    verbose = config.verbose
    results = []
    
    for idx, (split, query_tuple) in enumerate(queries):
        # Get max_derived_states from config if available, else use default
        max_derived_states = getattr(config, 'max_derived_per_state', 200)
        result = run_sb3_engine_single_query(
            query_tuple, engine_data, split=split, 
            deterministic=deterministic, max_depth=max_depth,
            verbose=verbose and idx < 3, seed=seed + idx,
            max_derived_states=max_derived_states
        )
        results.append(result)
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(queries)} queries...")
    
    # Aggregate statistics
    successful = sum(1 for r in results if r['success'])
    total_reward = sum(r['reward'] for r in results)
    total_steps = sum(r['steps'] for r in results)
    
    # Compute average actions (branching factor)
    total_actions = 0
    total_action_steps = 0
    for r in results:
        for step in r['trace']:
            if 'num_actions' in step:
                total_actions += step['num_actions']
                total_action_steps += 1
    
    avg_actions = total_actions / total_action_steps if total_action_steps > 0 else 0.0
    
    return {
        'total_queries': len(queries),
        'successful': successful,
        'avg_reward': total_reward / len(queries) if queries else 0.0,
        'avg_steps': total_steps / len(queries) if queries else 0.0,
        'avg_actions': avg_actions,
        'traces': results
    }
