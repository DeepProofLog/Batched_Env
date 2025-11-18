"""
Test module for tensor-based unification engine.

Simple and modular testing for the tensor-based unification engine in eval mode.
"""
import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_path)

import random
import torch
from typing import Tuple, Dict, List
from types import SimpleNamespace

from data_handler import DataHandler
from index_manager import IndexManager
from unification_engine import UnificationEngine
from debug_helper import DebugHelper


def get_default_tensor_engine_config() -> SimpleNamespace:
    return SimpleNamespace(
        padding_atoms=20,
        max_total_runtime_vars=1_000_000,
        max_derived_per_state=500,
        device='cpu'
    )


def setup_tensor_engine(
    dataset: str = "countries_s3",
    base_path: str = "./data/",
    batched: bool = False,
    config: SimpleNamespace = None
) -> Tuple:
    """
    Setup the tensor-based engine with dataset.
    
    Args:
        dataset: Dataset name
        base_path: Base path to data directory
        batched: If True, setup for batched operations
    
    Returns:
        (dh, im, engine, debug_helper, next_var_start)
    """
    cfg = config or get_default_tensor_engine_config()
    device_value = getattr(cfg, 'device', 'cpu')
    device = device_value if isinstance(device_value, torch.device) else torch.device(device_value)
    padding_atoms = getattr(cfg, 'padding_atoms', 20)
    max_total_runtime_vars = getattr(cfg, 'max_total_runtime_vars', 1_000_000)
    max_derived_per_state = getattr(cfg, 'max_derived_per_state', 500)

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
        max_total_runtime_vars=max_total_runtime_vars,
        padding_atoms=padding_atoms,
        max_arity=dh_non.max_arity,
        device=device,
        rules=dh_non.rules,
    )
    dh_non.materialize_indices(im=im_non, device=device)

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
        max_derived_per_state=max_derived_per_state,  # Set max derived states for eval mode
        sort_states=True
    )

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
            excluded_queries=excluded_query_tensor, verbose=0
        )
        num_derived = derived_counts[0].item()
        
        valid = []
        for i in range(num_derived):
            s = derived[0, i]
            if not engine.is_false_state(s):
                non_padding = (s[:, 0] != engine.padding_idx).sum().item()
                if non_padding <= 100:
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
        return debug_helper.canonical_state_to_str(s)
    
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
    config: SimpleNamespace
) -> Dict:
    """
    Test multiple queries using the tensor engine.
    
    Args:
        queries: List of (split, (predicate, head, tail))
        engine_data: Tuple from setup_tensor_engine()
        config: Configuration namespace with deterministic, max_depth, seed, verbose, etc.
        
    Returns:
        Dict with keys:
            - total_queries: int
            - successful: int
            - avg_reward: float
            - avg_steps: float
            - traces: List of trace dicts
    """
    deterministic = config.deterministic
    max_depth = config.max_depth
    seed = config.seed
    verbose = config.verbose
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
