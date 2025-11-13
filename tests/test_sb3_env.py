"""
Test module for SB3 (string-based) environment.

This module contains functions to test the string-based environment
with both deterministic (canonical) and random action selection.
"""
import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_path)
sys.path.insert(0, os.path.join(root_path, 'str_based'))

import random
import torch
import numpy as np
from typing import List, Tuple, Dict

# String-engine stack
from str_based.str_dataset import DataHandler as StrDataHandler
from str_based.str_index_manager import IndexManager as StrIndexManager
from str_based.str_env import LogicEnv_gym as StrEnv
from str_based.str_utils import Term as StrTerm


def safe_item(x):
    """Safely extract scalar from numpy/torch/python scalar."""
    if isinstance(x, (np.ndarray, np.generic)):
        return x.item() if x.size == 1 else x
    elif isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x
    return x


def canonicalize_str_state(state: List[StrTerm]) -> str:
    """Convert str state to canonical string with variable renaming by first appearance.
    
    IMPORTANT: This function ONLY renames variables to canonical names.
    It does NOT reorder atoms - atom order must be preserved for proof search!
    Variables are renamed by order of first appearance (atom by atom, arg by arg).
    """
    var_mapping = {}
    next_var_num = 1
    
    # Find variables in order of first appearance (preserve atom order!)
    for term in state:
        for arg in term.args:
            if isinstance(arg, str) and arg.startswith('Var') and arg not in var_mapping:
                var_mapping[arg] = f"Var_{next_var_num}"
                next_var_num += 1
    
    # Build canonical strings preserving atom order
    canonical_atoms = []
    for term in state:
        new_args = []
        for arg in term.args:
            if isinstance(arg, str) and arg.startswith('Var'):
                new_args.append(var_mapping[arg])
            else:
                new_args.append(arg)
        canonical_atoms.append(f"{term.predicate}({','.join(new_args)})")
    
    # Return with atoms in original order, no spaces for consistency
    return '|'.join(canonical_atoms)


def setup_sb3_env(dataset: str = "countries_s3", base_path: str = "./data/", seed: int = 42) -> Tuple:
    """
    Setup the string-based (SB3) environment with dataset.
    
    Returns:
        (str_env, im_str, dh_str)
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
        padding_atoms=100,
        max_arity=dh_str.max_arity,
        device=torch.device('cpu'),
    )
    
    facts_set = set(dh_str.facts)
    
    # Build fact index for efficient unification
    im_str.build_fact_index(list(facts_set))
    
    # Create str environment
    str_env = StrEnv(
        index_manager=im_str,
        data_handler=dh_str,
        queries=dh_str.train_queries,
        labels=[1] * len(dh_str.train_queries),
        query_depths=[None] * len(dh_str.train_queries),
        facts=facts_set,
        mode='eval_with_restart',
        seed=seed,
        max_depth=20,
        memory_pruning=True,
        padding_atoms=100,
        padding_states=500,
        verbose=0,
        prover_verbose=0,
        device=torch.device('cpu'),
        engine='python',
        engine_strategy='complete',
        skip_unary_actions=True,
        endf_action=False,
        reward_type=0,
    )
    
    return str_env, im_str, dh_str


def test_sb3_env_single_query(
    query_tuple: Tuple[str, str, str],
    env_data: Tuple,
    split: str = 'train',
    deterministic: bool = True,
    max_depth: int = 20,
    verbose: bool = False,
    seed: int = 42
) -> Dict:
    """
    Test a single query using the SB3 environment.
    
    Args:
        query_tuple: (predicate, head, tail)
        env_data: Tuple from setup_sb3_env()
        split: 'train', 'valid', or 'test'
        deterministic: If True, use canonical ordering; if False, random actions
        max_depth: Maximum proof depth
        verbose: Print detailed information
        seed: Random seed for reproducible random actions
        
    Returns:
        Dict with keys:
            - success: bool
            - steps: int
            - reward: float
            - trace: List[Dict] with state, derived_states, action at each step
    """
    p, h, t = query_tuple
    str_env, im_str, dh_str = env_data
    
    if verbose:
        print(f"\nQuery: {p}({h}, {t}) [split={split}, deterministic={deterministic}]")
    
    # Reset environment with the query
    q_str = StrTerm(predicate=p, args=(h, t))
    label = 1  # All queries are true (ground truth label)
    
    str_env.current_query = q_str
    str_env.current_label = label
    str_env.current_query_depth_value = None
    str_obs, _ = str_env._reset([q_str], label)
    str_env.current_label = label  # Re-ensure after reset
    
    # Setup RNG
    rng = random.Random(seed)
    
    # Run episode
    total_reward = 0.0
    steps = 0
    trace = []
    done_flag = False
    str_info = {}  # Initialize to empty dict
    
    while steps < max_depth and not done_flag:
        # Get current state info
        state = str_env.tensordict['state']
        derived_states = str_env.tensordict['derived_states']
        action_mask = str_obs['action_mask']
        num_actions = safe_item(action_mask.sum())
        
        if num_actions == 0:
            # Terminal state - check if it's a success (all atoms are True)
            is_success_state = all(term.predicate == 'True' for term in state)
            trace.append({
                'step': steps,
                'state': canonicalize_str_state(state),
                'derived_states': [],
                'num_actions': 0,
                'action': None,
                'done': True,
                'reward': safe_item(str_env.tensordict.get('reward', 0))
            })
            # Set done_flag and info for proper success detection after loop
            done_flag = True
            # Check if is_success already in str_obs, otherwise compute it
            if 'is_success' in str_obs:
                str_info = {'is_success': str_obs['is_success']}
            else:
                str_info = {'is_success': is_success_state}
            break
        
        # Canonicalize and sort derived states
        canon_states = [(canonicalize_str_state(ds), i) for i, ds in enumerate(derived_states)]
        canon_states.sort(key=lambda x: x[0])
        
        if deterministic:
            # Choose first canonical action
            action = canon_states[0][1]
        else:
            # Choose random action
            valid_actions = [i for i in range(len(action_mask)) if action_mask[i]]
            action = rng.choice(valid_actions)
        
        trace.append({
            'step': steps,
            'state': canonicalize_str_state(state),
            'derived_states': [c[0] for c in canon_states],
            'num_actions': num_actions,
            'action': action,
            'done': False,
            'reward': 0.0  # Intermediate reward
        })
        
        if verbose and steps < 3:
            print(f"  Step {steps}: {num_actions} actions, chose {action}")
        
        # Take step
        str_obs, str_reward, done_flag, truncated, str_info = str_env.step(action)
        total_reward += safe_item(str_reward)
        
        # Update last trace entry with reward
        if trace:
            trace[-1]['reward'] = safe_item(str_reward)
        
        steps += 1
    
    # Check final state
    # Note: str_info['is_success'] is set by env.step() when the proof succeeds,
    # regardless of whether done_flag is set (done_flag might be False if we hit max_depth)
    success = str_info.get('is_success', False)
    
    # NOTE: Do NOT add extra terminal state here - it's already in the trace
    # The last trace entry (either from num_actions==0 or final iteration) is the terminal state
    
    if verbose:
        print(f"  Debug: done_flag={done_flag}, str_info={str_info}")
        print(f"  Finished: success={success}, total_reward={total_reward:.2f}")
    
    return {
        'success': success,
        'steps': steps,
        'reward': total_reward,
        'trace': trace
    }


def test_sb3_env_batch(
    queries: List[Tuple[str, Tuple[str, str, str]]],
    env_data: Tuple,
    deterministic: bool = True,
    max_depth: int = 20,
    seed: int = 42,
    verbose: bool = False
) -> Dict:
    """
    Test multiple queries using the SB3 environment.
    
    Args:
        queries: List of (split, (predicate, head, tail))
        env_data: Tuple from setup_sb3_env()
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
        result = test_sb3_env_single_query(
            query_tuple, env_data, split=split,
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
