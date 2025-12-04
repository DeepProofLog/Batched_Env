"""
Test module for SB3 (string-based) environment.

Simple and modular testing for the string-based environment.
"""
import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_path)
# CRITICAL: Add sb3 folder to path so that relative imports within sb3 work correctly
# This ensures sb3_utils.Term is the same class everywhere
sb3_path = os.path.join(root_path, 'sb3')
sys.path.insert(0, sb3_path)

import random
import torch
import numpy as np
from typing import Tuple, Dict, List
from types import SimpleNamespace

# Import using relative names (same as sb3_env.py does internally)
from sb3_dataset import DataHandler as StrDataHandler
from sb3_index_manager import IndexManager as StrIndexManager
from sb3_env import LogicEnv_gym as StrEnv
from sb3_utils import Term as StrTerm
from sb3_unification import state_to_str


def get_default_sb3_env_config() -> SimpleNamespace:
    return SimpleNamespace(
        padding_atoms=100,
        padding_states=500,
        skip_unary_actions=True,
        end_proof_action=True,
        reward_type=0,
        memory_pruning=True,
        max_total_runtime_vars=1000,
        verbose=0,
        prover_verbose=0,
        device='cpu'
    )


def safe_item(x):
    """Safely extract scalar from numpy/torch/python scalar."""
    if isinstance(x, (np.ndarray, np.generic)):
        return x.item() if x.size == 1 else x
    elif isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x
    return x


def setup_sb3_env(
    dataset: str = "countries_s3",
    base_path: str = "./data/",
    seed: int = 42,
    config: SimpleNamespace = None
) -> Tuple:
    """
    Setup the string-based (SB3) environment with dataset.
    
    Returns:
        (str_env, im_str, dh_str)
    """
    cfg = config or get_default_sb3_env_config()
    device_value = getattr(cfg, 'device', 'cpu')
    device = device_value if isinstance(device_value, torch.device) else torch.device(device_value)

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
        max_total_vars=getattr(cfg, 'max_total_runtime_vars', 1000000),
        rules=dh_str.rules,
        padding_atoms=getattr(cfg, 'padding_atoms', 100),
        max_arity=dh_str.max_arity,
        device=device,
    )
    
    facts_set = set(dh_str.facts)
    
    # Build fact index for efficient unification
    # Use deterministic=True to ensure consistent iteration order (sorted by (pred, head, tail))
    # This matches the tensor engine's fact ordering
    im_str.build_fact_index(list(facts_set), deterministic=True)
    
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
        memory_pruning=getattr(cfg, 'memory_pruning', True),
        padding_atoms=getattr(cfg, 'padding_atoms', 100),
        padding_states=getattr(cfg, 'padding_states', 500),
        verbose=getattr(cfg, 'verbose', 0),
        prover_verbose=getattr(cfg, 'prover_verbose', 0),
        device=device,
        engine='python',
        engine_strategy='complete',
        skip_unary_actions=getattr(cfg, 'skip_unary_actions', True),
        endf_action=getattr(cfg, 'end_proof_action', True),
        reward_type=getattr(cfg, 'reward_type', 0),
        canonical_action_order=getattr(cfg, 'canonical_action_order', False),
    )
    
    return str_env, im_str, dh_str


def run_sb3_env_single_query(
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
                'state': state_to_str(state),
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
        
        # Just take first action in deterministic mode
        if deterministic:
            # Choose first action
            action = 0
        else:
            # Choose random action
            action = rng.choice(range(num_actions))
        
        # Build derived states list for trace 
        derived_states_ = [state_to_str(derived_states[i]) for i in range(num_actions)]
        
        trace.append({
            'step': steps,
            'state': state_to_str(state),
            'derived_states': derived_states_,
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


def run_sb3_env(
    queries: List[Tuple[str, Tuple[str, str, str]]],
    env_data: Tuple,
    config: SimpleNamespace
) -> Dict:
    """
    Test multiple queries using the SB3 environment.
    
    Uses proper reset/step calls directly (not calling _single_query).
    SB3 environment doesn't have true batching, so queries are processed sequentially.
    
    Args:
        queries: List of (split, (predicate, head, tail))
        env_data: Tuple from setup_sb3_env()
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
    
    str_env, im_str, dh_str = env_data
    results = []
    
    for idx, (split, query_tuple) in enumerate(queries):
        p, h, t = query_tuple
        
        # Setup query
        q_str = StrTerm(predicate=p, args=(h, t))
        label = 1  # All queries are true
        
        # Reset environment with the query using proper reset call
        str_env.current_query = q_str
        str_env.current_label = label
        str_env.current_query_depth_value = None
        str_obs, _ = str_env._reset([q_str], label)
        str_env.current_label = label  # Re-ensure after reset
        
        # Setup RNG for this query
        rng = random.Random(seed + idx)
        
        # Run episode using step loop
        total_reward = 0.0
        steps = 0
        trace = []
        done_flag = False
        str_info = {}
        
        while steps < max_depth and not done_flag:
            # Get current state info
            state = str_env.tensordict['state']
            derived_states = str_env.tensordict['derived_states']
            action_mask = str_obs['action_mask']
            num_actions = safe_item(action_mask.sum())
            
            if num_actions == 0:
                # Terminal state - check if it's a success
                is_success_state = all(term.predicate == 'True' for term in state)
                trace.append({
                    'step': steps,
                    'state': state_to_str(state),
                    'derived_states': [],
                    'num_actions': 0,
                    'action': None,
                    'done': True,
                    'reward': safe_item(str_env.tensordict.get('reward', 0))
                })
                done_flag = True
                if 'is_success' in str_obs:
                    str_info = {'is_success': str_obs['is_success']}
                else:
                    str_info = {'is_success': is_success_state}
                break
            
            # Choose action
            if deterministic:
                action = 0  # First action 
            else:
                action = rng.choice(range(num_actions))
            
            # Build derived states list for trace
            derived_states_ = [state_to_str(derived_states[i]) for i in range(num_actions)]
            
            trace.append({
                'step': steps,
                'state': state_to_str(state),
                'derived_states': derived_states_,
                'num_actions': num_actions,
                'action': action,
                'done': False,
                'reward': 0.0
            })
            
            if verbose and idx < 3 and steps < 3:
                print(f"  Query {idx+1}, Step {steps}: {num_actions} actions, chose {action}")
            
            # Take step
            str_obs, str_reward, done_flag, truncated, str_info = str_env.step(action)
            total_reward += safe_item(str_reward)
            
            # Update last trace entry with reward
            if trace:
                trace[-1]['reward'] = safe_item(str_reward)
            
            steps += 1
        
        # Get success status
        success = str_info.get('is_success', False)
        
        results.append({
            'success': success,
            'steps': steps,
            'reward': total_reward,
            'trace': trace
        })
        
        if not verbose and (idx + 1) % 100 == 0:
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
