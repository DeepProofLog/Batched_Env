"""
Test module for tensor-based batched environment.

This module contains functions to test the batched tensor environment
with both deterministic (canonical) and random action selection.
"""
import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_path)

import random
import torch
import numpy as np
from typing import List, Tuple, Dict
from tensordict import TensorDict

# Tensor-engine stack
from data_handler import DataHandler
from index_manager import IndexManager
from unification_engine import UnificationEngine
from env import BatchedEnv
from debug_helper import DebugHelper


def safe_item(x):
    """Safely extract scalar from numpy/torch/python scalar."""
    if isinstance(x, (np.ndarray, np.generic)):
        return x.item() if x.size == 1 else x
    elif isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x
    return x


# Note: canonicalize_tensor_state is replaced by engine.canonical_state_to_str()
# Use the engine's built-in method instead of this custom implementation


def setup_tensor_env(dataset: str = "countries_s3", base_path: str = "./data/", seed: int = 42, batch_size: int = 1) -> Tuple:
    """
    Setup the batched tensor environment with dataset.
    
    Returns:
        (batched_env, debug_helper, constant_no, im_batched, dh_batched)
    """
    dh_batched = DataHandler(
        dataset_name=dataset,
        base_path=base_path,
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        train_depth=None,
    )
    
    im_batched = IndexManager(
        constants=dh_batched.constants,
        predicates=dh_batched.predicates,
        max_total_runtime_vars=1000000,
        padding_atoms=100,
        max_arity=dh_batched.max_arity,
        device=torch.device('cpu'),
        rules=dh_batched.rules,
    )
    dh_batched.materialize_indices(im=im_batched, device=torch.device('cpu'))
    
    stringifier_params = {
        'verbose': 0,
        'idx2predicate': im_batched.idx2predicate,
        'idx2constant': im_batched.idx2constant,
        'idx2template_var': im_batched.idx2template_var,
        'padding_idx': im_batched.padding_idx,
        'n_constants': im_batched.constant_no
    }
    
    engine = UnificationEngine.from_index_manager(im_batched, take_ownership=True, 
                                                   stringifier_params=stringifier_params)
    engine.index_manager = im_batched
    
    debug_helper = DebugHelper(
        verbose=0,
        idx2predicate=im_batched.idx2predicate,
        idx2constant=im_batched.idx2constant,
        idx2template_var=im_batched.idx2template_var,
        padding_idx=engine.padding_idx,
        n_constants=im_batched.constant_no
    )
    
    dummy_query = torch.full((batch_size, 100, 3), im_batched.padding_idx, dtype=torch.long, device='cpu')
    
    batched_env = BatchedEnv(
        batch_size=batch_size,
        queries=dummy_query,
        labels=torch.ones(batch_size, dtype=torch.long, device='cpu'),
        query_depths=torch.ones(batch_size, dtype=torch.long, device='cpu'),
        unification_engine=engine,
        mode='eval',
        max_depth=20,
        memory_pruning=True,
        eval_pruning=True,
        padding_atoms=100,
        padding_states=500,
        true_pred_idx=im_batched.predicate_str2idx.get('True'),
        false_pred_idx=im_batched.predicate_str2idx.get('False'),
        end_pred_idx=im_batched.predicate_str2idx.get('End'),
        verbose=0,
        prover_verbose=0,
        device=torch.device('cpu'),
        runtime_var_start_index=im_batched.constant_no + 1,
        total_vocab_size=im_batched.constant_no + 1000000,
        skip_unary_actions=True,
        end_proof_action=False,
        reward_type=0,
    )
    
    return batched_env, debug_helper, im_batched.constant_no, im_batched, dh_batched


def test_tensor_env_single_query(
    query_tuple: Tuple[str, str, str],
    env_data: Tuple,
    split: str = 'train',
    deterministic: bool = True,
    max_depth: int = 20,
    verbose: bool = False,
    seed: int = 42
) -> Dict:
    """
    Test a single query using the batched tensor environment (batch_size=1).
    
    Args:
        query_tuple: (predicate, head, tail)
        env_data: Tuple from setup_tensor_env()
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
    batched_env, debug_helper, constant_no, im_batched, dh_batched = env_data
    
    if verbose:
        print(f"\nQuery: {p}({h}, {t}) [split={split}, deterministic={deterministic}]")
    
    # Setup query
    query_atom = batched_env.unification_engine.index_manager.atom_to_tensor(p, h, t)
    query_padded = torch.full((1, batched_env.padding_atoms, 3), batched_env.padding_idx, 
                               dtype=torch.long, device='cpu')
    query_padded[0, 0] = query_atom
    
    label = 1
    batched_env._all_queries_padded = query_padded
    batched_env._all_labels = torch.tensor([label], dtype=torch.long, device='cpu')
    batched_env._all_depths = torch.tensor([1], dtype=torch.long, device='cpu')
    batched_env._all_first_atoms = query_atom.unsqueeze(0)
    batched_env._num_all = 1
    
    batched_obs_td = batched_env.reset()
    
    # Setup RNG
    rng = random.Random(seed)
    
    # Run episode
    total_reward = 0.0
    steps = 0
    trace = []
    done_flag = False
    
    while steps < max_depth and not done_flag:
        # Get current state info
        batched_state = batched_env.current_queries[0]  # [A, D]
        batched_derived_states = batched_env.derived_states_batch[0]  # [S, A, D]
        
        # Extract action mask
        batched_action_mask = batched_obs_td['action_mask'][0]
        num_batched_actions = safe_item(batched_action_mask.sum())
        
        if num_batched_actions == 0:
            # Terminal state - check if it's a success (True state)
            is_success_state = batched_env.unification_engine.is_true_state(batched_state)
            trace.append({
                'step': steps,
                'state': batched_env.unification_engine.canonical_state_to_str(batched_state),
                'derived_states': [],
                'num_actions': 0,
                'action': None,
                'done': True,
                'reward': 0.0
            })
            # Set done_flag and is_success for proper success detection after loop
            done_flag = True
            # Check if is_success already in observation, otherwise compute it
            if 'is_success' not in batched_obs_td or batched_obs_td['is_success'][0] == 0:
                batched_obs_td['is_success'] = torch.tensor([is_success_state])
            break
        
        # Canonicalize and sort derived states using engine method
        batched_canon_states = []
        for i in range(num_batched_actions):
            ds = batched_derived_states[i]
            canon = batched_env.unification_engine.canonical_state_to_str(ds)
            batched_canon_states.append((canon, i))
        batched_canon_states.sort(key=lambda x: x[0])
        
        if deterministic:
            # Choose first canonical action
            action = batched_canon_states[0][1]
        else:
            # Choose random action
            valid_actions = [i for i in range(num_batched_actions) if batched_action_mask[i]]
            action = rng.choice(valid_actions)
        
        trace.append({
            'step': steps,
            'state': batched_env.unification_engine.canonical_state_to_str(batched_state),
            'derived_states': [c[0] for c in batched_canon_states],
            'num_actions': num_batched_actions,
            'action': action,
            'done': False,
            'reward': 0.0  # Intermediate reward
        })
        
        if verbose and steps < 3:
            print(f"  Step {steps}: {num_batched_actions} actions, chose {action}")
        
        # Take step
        batched_action_td = TensorDict({
            'action': torch.tensor([action], dtype=torch.long, device='cpu')
        }, batch_size=[1])
        batched_result_td = batched_env.step(batched_action_td)
        
        # Extract observation, reward, done
        if 'next' in batched_result_td.keys():
            batched_obs_td = batched_result_td['next']
            done_flag = safe_item(batched_obs_td['done'][0])
            batched_reward = safe_item(batched_obs_td.get('reward', batched_result_td.get('reward', torch.tensor([0.0])))[0])
        else:
            batched_obs_td = batched_result_td
            done_flag = safe_item(batched_result_td['done'][0])
            batched_reward = safe_item(batched_result_td.get('reward', torch.tensor([0.0]))[0])
        
        total_reward += batched_reward
        
        # Update last trace entry with reward
        if trace:
            trace[-1]['reward'] = batched_reward
        
        steps += 1
    
    # Check final state
    # Note: batched_obs_td['is_success'] is set by env.step() when the proof succeeds,
    # regardless of whether done_flag is set (done_flag might be False if we hit max_depth)
    success = safe_item(batched_obs_td.get('is_success', torch.tensor([False]))[0])
    
    # NOTE: Do NOT add extra terminal state here - it's already in the trace
    # The last trace entry (either from num_actions==0 or final iteration) is the terminal state
    
    if verbose:
        print(f"  Finished: success={success}, total_reward={total_reward:.2f}")
    
    return {
        'success': success,
        'steps': steps,
        'reward': total_reward,
        'trace': trace
    }


def test_tensor_env_batch(
    queries: List[Tuple[str, Tuple[str, str, str]]],
    base_env_data: Tuple,
    deterministic: bool = True,
    max_depth: int = 20,
    seed: int = 42,
    verbose: bool = False
) -> Dict:
    """
    Test multiple queries using the batched tensor environment in true batch mode.
    
    Args:
        queries: List of (split, (predicate, head, tail))
        base_env_data: Tuple from setup_tensor_env() (will be re-setup with proper batch size)
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
    # Extract dataset info from base_env_data
    _, _, _, im_batched, dh_batched = base_env_data
    
    # Setup batched env with proper batch size
    batch_size = len(queries)
    batched_env, debug_helper, constant_no, im_batched_new, _ = setup_tensor_env(
        dataset=dh_batched.dataset_name,
        base_path="./data/",
        seed=seed,
        batch_size=batch_size
    )
    
    # Prepare batch queries tensor
    A = 100
    D = im_batched_new.max_arity + 1
    pad = im_batched_new.padding_idx
    batch_q = torch.full((batch_size, A, D), pad, dtype=torch.long)
    labels = torch.zeros(batch_size, dtype=torch.long)
    depths = torch.ones(batch_size, dtype=torch.long)
    
    for i, (split, query_tuple) in enumerate(queries):
        p, h, t = query_tuple
        atom = batched_env.unification_engine.index_manager.atom_to_tensor(p, h, t)
        batch_q[i, 0] = atom
        labels[i] = 1  # All queries are true
    
    # Set eval dataset
    per_slot = torch.ones(batch_size, dtype=torch.long)
    batched_env.set_eval_dataset(batch_q, labels, depths, per_slot_lengths=per_slot)
    obs_td = batched_env.reset()
    if 'next' in obs_td.keys():
        obs_td = obs_td['next']
    
    # Track per-query results
    results = [{
        'success': False,
        'steps': 0,
        'reward': 0.0,
        'trace': []
    } for _ in range(batch_size)]
    
    done_mask = torch.zeros(batch_size, dtype=torch.bool)
    rng = random.Random(seed)
    
    for step in range(max_depth):
        action_mask = obs_td['action_mask'].cpu()  # [B, S]
        derived_batch = batched_env.derived_states_batch.cpu()  # [B, S, A, D]
        current_queries = batched_env.current_queries.cpu()  # [B, A, D]
        
        actions = torch.zeros(batch_size, dtype=torch.long)
        
        for i in range(batch_size):
            if done_mask[i]:
                actions[i] = 0
                continue
            
            # Get state info
            cur_state = current_queries[i]
            n_actions = int(action_mask[i].sum())
            
            if n_actions == 0:
                done_mask[i] = True
                # Check if this is a success state (True state)
                is_success_state = batched_env.unification_engine.is_true_state(cur_state)
                results[i]['success'] = is_success_state
                results[i]['trace'].append({
                    'step': step,
                    'state': batched_env.unification_engine.canonical_state_to_str(cur_state),
                    'derived_states': [],
                    'num_actions': 0,
                    'action': None,
                    'done': True,
                    'reward': 0.0
                })
                continue
            
            # Build canonical list using engine method
            canon_list = []
            for a in range(n_actions):
                ds = derived_batch[i, a]
                canon = batched_env.unification_engine.canonical_state_to_str(ds)
                canon_list.append((canon, a))
            canon_list.sort(key=lambda x: x[0])
            
            if deterministic:
                chosen = canon_list[0][1]
            else:
                valid_actions = [a for a in range(n_actions) if action_mask[i, a]]
                chosen = rng.choice(valid_actions)
            
            actions[i] = int(chosen)
            
            results[i]['trace'].append({
                'step': step,
                'state': batched_env.unification_engine.canonical_state_to_str(cur_state),
                'derived_states': [c[0] for c in canon_list],
                'num_actions': n_actions,
                'action': chosen,
                'done': False,
                'reward': 0.0
            })
            results[i]['steps'] += 1
        
        if done_mask.all():
            break
        
        # Step batched env
        td = TensorDict({'action': actions}, batch_size=[batch_size])
        result = batched_env.step(td)
        next_td = result['next'] if 'next' in result.keys() else result
        
        rewards = next_td.get('reward', torch.zeros(batch_size, 1)).squeeze(-1).cpu()
        dones = next_td['done'].squeeze(-1).cpu()
        is_success = next_td.get('is_success', torch.zeros(batch_size, 1)).squeeze(-1).cpu()
        
        # Handle 0-dim tensors for batch_size=1
        if rewards.dim() == 0:
            rewards = rewards.unsqueeze(0)
        if dones.dim() == 0:
            dones = dones.unsqueeze(0)
        if is_success.dim() == 0:
            is_success = is_success.unsqueeze(0)
        
        # Update results
        for i in range(batch_size):
            if not done_mask[i]:
                reward_val = float(rewards[i])
                results[i]['reward'] += reward_val
                
                if len(results[i]['trace']) > 0:
                    results[i]['trace'][-1]['reward'] = reward_val
                
                if dones[i]:
                    done_mask[i] = True
                    results[i]['success'] = bool(is_success[i])
                    
                    # NOTE: Do NOT add extra terminal state - it's already in the trace
                    # The last trace entry is the terminal state
        
        obs_td = next_td
        
        if not verbose and (step + 1) % 5 == 0:
            n_done = done_mask.sum().item()
            print(f"  Step {step + 1}: {n_done}/{batch_size} queries done")
    
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
