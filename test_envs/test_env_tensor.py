"""
Test module for tensor-based batched environment.

Simple and modular testing for the batched tensor environment.
Uses the tensor engine in eval mode with one query instead of creating from scratch.
"""
import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_path)

import random
import torch
import numpy as np
from typing import Tuple, Dict, List
from tensordict import TensorDict

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
    
    engine = UnificationEngine.from_index_manager(
        im_batched, take_ownership=True, 
        stringifier_params=stringifier_params,
        max_derived_per_state=500  # Set max derived states for eval mode
    )
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
        mode='train',  # Start in train mode, set_eval_dataset will switch to eval
        max_depth=20,
        memory_pruning=False,
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
        skip_unary_actions=False,
        end_proof_action=False,
        reward_type=0,
        use_exact_memory=True,
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
                'state': debug_helper.canonical_state_to_str(batched_state),
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
        
        # States are already in canonical order from environment
        # Just take first action in deterministic mode
        if deterministic:
            # Choose first action (already in canonical order)
            action = 0
        else:
            # Choose random action
            valid_actions = [i for i in range(num_batched_actions) if batched_action_mask[i]]
            action = rng.choice(valid_actions)
        
        # Build derived states list for trace (already in canonical order)
        derived_states_canonical = [debug_helper.canonical_state_to_str(batched_derived_states[i]) for i in range(num_batched_actions)]
        
        trace.append({
            'step': steps,
            'state': debug_helper.canonical_state_to_str(batched_state),
            'derived_states': derived_states_canonical,
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



def test_tensor_env(
    queries: List[Tuple[str, Tuple[str, str, str]]],
    base_env_data: Tuple,
    deterministic: bool = True,
    max_depth: int = 20,
    seed: int = 42,
    verbose: bool = False
) -> Dict:
    """
    Test multiple queries sequentially using the tensor environment (batch_size=1).
    
    This function loops through all queries one-by-one, testing each with test_tensor_env_single_query.
    This is the proper way to test the tensor_env config (non-batched mode).
    
    Args:
        queries: List of (split, (predicate, head, tail))
        base_env_data: Tuple from setup_tensor_env() with batch_size=1
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
            - traces: List of trace dicts (one per query)
    """
    all_results = []
    
    if verbose:
        print(f"Testing {len(queries)} queries sequentially...")
    
    for idx, (split, query_tuple) in enumerate(queries):
        if verbose and idx < 3:
            print(f"\nQuery {idx+1}/{len(queries)}: {query_tuple[0]}({query_tuple[1]}, {query_tuple[2]}) [split={split}]")
        
        result = test_tensor_env_single_query(
            query_tuple=query_tuple,
            env_data=base_env_data,
            split=split,
            deterministic=deterministic,
            max_depth=max_depth,
            verbose=verbose and idx < 3,
            seed=seed + idx  # Different seed per query for random mode
        )
        all_results.append(result)
    
    # Aggregate statistics
    successful = sum(1 for r in all_results if r['success'])
    total_reward = sum(r['reward'] for r in all_results)
    total_steps = sum(r['steps'] for r in all_results)
    
    if verbose or True:
        print(f"\nSequential evaluation complete:")
        print(f"  Total queries: {len(queries)}")
        print(f"  Successful: {successful}/{len(queries)} ({100*successful/len(queries):.1f}%)")
        print(f"  Avg steps: {total_steps/len(queries):.2f}")
        print(f"  Avg reward: {total_reward/len(queries):.2f}")
    
    return {
        'total_queries': len(queries),
        'successful': successful,
        'avg_reward': total_reward / len(queries) if queries else 0.0,
        'avg_steps': total_steps / len(queries) if queries else 0.0,
        'traces': all_results
    }


def test_tensor_env_batched(
    queries: List[Tuple[str, Tuple[str, str, str]]],
    base_env_data: Tuple,
    deterministic: bool = True,
    max_depth: int = 20,
    seed: int = 42,
    verbose: bool = False
) -> Dict:
    """
    Test multiple queries using the batched tensor environment in TRUE BATCH MODE using eval mode.
    All queries are loaded into eval mode and processed in parallel batch slots.
    
    This is the proper way to test the batched tensor environment - all queries run in parallel.
    
    Args:
        queries: List of (split, (predicate, head, tail))
        base_env_data: Tuple from setup_tensor_env()
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
            - traces: List of trace dicts (one per query)
    """
    # Extract dataset info from base_env_data
    _, _, _, im_batched, dh_batched = base_env_data
    
    batch_size = len(queries)
    
    # Setup batched env with proper batch size
    batched_env, debug_helper, constant_no, im_batched_new, _ = setup_tensor_env(
        dataset=dh_batched.dataset_name,
        base_path="./data/",
        seed=seed,
        batch_size=batch_size
    )
    
    # Prepare all queries as tensors
    all_query_tensors = []
    for split, (p, h, t) in queries:
        query_atom = im_batched_new.atom_to_tensor(p, h, t)
        query_padded = torch.full((batched_env.padding_atoms, 3), batched_env.padding_idx,
                                   dtype=torch.long, device='cpu')
        query_padded[0] = query_atom
        all_query_tensors.append(query_padded)
    
    # Stack into [M, A, D] tensor
    queries_tensor = torch.stack(all_query_tensors, dim=0)
    labels_tensor = torch.ones(batch_size, dtype=torch.long, device='cpu')
    depths_tensor = torch.ones(batch_size, dtype=torch.long, device='cpu')
    
    # Use per_slot_lengths for proper slot-based scheduling in eval mode
    # Each slot gets exactly 1 query (queries are laid out sequentially: slot0's query, slot1's query, ...)
    per_slot_lengths = torch.ones(batch_size, dtype=torch.long, device='cpu')
    
    # Load dataset into eval mode
    batched_env.set_eval_dataset(
        queries=queries_tensor,
        labels=labels_tensor,
        query_depths=depths_tensor,
        per_slot_lengths=per_slot_lengths
    )
    
    # Reset to load first batch of queries
    obs_td = batched_env.reset()
    if 'next' in obs_td.keys():
        obs_td = obs_td['next']
    
    # Setup RNG
    rng = random.Random(seed)
    
    # Track per-slot traces
    slot_traces = [[] for _ in range(batch_size)]
    slot_rewards = [0.0 for _ in range(batch_size)]
    slot_steps = [0 for _ in range(batch_size)]
    slot_done = [False for _ in range(batch_size)]
    
    # Run episode until all slots are done
    global_step = 0
    while global_step < max_depth and not all(slot_done):
        # Get current state info for all slots
        current_queries = batched_env.current_queries  # [B, A, D]
        derived_states_batch = batched_env.derived_states_batch  # [B, S, A, D]
        action_mask = obs_td['action_mask']  # [B, S]
        
        # Select actions for each slot
        actions = torch.zeros(batch_size, dtype=torch.long, device='cpu')
        
        for slot_idx in range(batch_size):
            if slot_done[slot_idx]:
                actions[slot_idx] = 0  # Dummy action for done slots
                continue
            
            mask = action_mask[slot_idx]
            n_actions = int(mask.sum())
            
            if n_actions == 0:
                # Terminal state
                slot_done[slot_idx] = True
                cur_state = current_queries[slot_idx]
                slot_traces[slot_idx].append({
                    'step': slot_steps[slot_idx],
                    'state': debug_helper.canonical_state_to_str(cur_state),
                    'derived_states': [],
                    'num_actions': 0,
                    'action': None,
                    'done': True,
                    'reward': 0.0
                })
                actions[slot_idx] = 0  # Dummy action
                continue
            
            # Choose action
            if deterministic:
                chosen = 0
            else:
                valid_actions = [a for a in range(n_actions) if mask[a]]
                chosen = rng.choice(valid_actions)
            
            actions[slot_idx] = chosen
            
            # Record trace for this slot
            cur_state = current_queries[slot_idx]
            derived_states = derived_states_batch[slot_idx]
            derived_states_canonical = [
                debug_helper.canonical_state_to_str(derived_states[a]) 
                for a in range(n_actions)
            ]
            
            slot_traces[slot_idx].append({
                'step': slot_steps[slot_idx],
                'state': debug_helper.canonical_state_to_str(cur_state),
                'derived_states': derived_states_canonical,
                'num_actions': n_actions,
                'action': chosen,
                'done': False,
                'reward': 0.0
            })
            
            slot_steps[slot_idx] += 1
        
        if verbose and global_step < 3:
            active_slots = sum(1 for done in slot_done if not done)
            print(f"  Global step {global_step}: {active_slots} active slots")
        
        # Take batched step
        action_td = TensorDict({'action': actions}, batch_size=[batch_size])
        result_td = batched_env.step(action_td)
        
        # Extract next observation
        if 'next' in result_td.keys():
            obs_td = result_td['next']
        else:
            obs_td = result_td
        
        # Update rewards and done flags
        rewards = obs_td.get('reward', torch.zeros(batch_size, device='cpu'))
        dones = obs_td['done']
        
        for slot_idx in range(batch_size):
            if not slot_done[slot_idx]:
                reward_val = float(rewards[slot_idx])
                slot_rewards[slot_idx] += reward_val
                
                # Update last trace entry with reward
                if slot_traces[slot_idx]:
                    slot_traces[slot_idx][-1]['reward'] = reward_val
                
                if dones[slot_idx]:
                    slot_done[slot_idx] = True
        
        global_step += 1
    
    # Collect results from all slots
    all_results = []
    is_success = obs_td.get('is_success', torch.zeros(batch_size, dtype=torch.bool, device='cpu'))
    
    for slot_idx in range(batch_size):
        success = bool(is_success[slot_idx])
        all_results.append({
            'success': success,
            'steps': slot_steps[slot_idx],
            'reward': slot_rewards[slot_idx],
            'trace': slot_traces[slot_idx]
        })
    
    # Aggregate statistics
    successful = sum(1 for r in all_results if r['success'])
    total_reward = sum(r['reward'] for r in all_results)
    total_steps = sum(r['steps'] for r in all_results)
    
    if verbose or True:
        print(f"\n  TRUE BATCHED evaluation complete:")
        print(f"    Total queries: {len(queries)}")
        print(f"    Successful: {successful}/{len(queries)} ({100*successful/len(queries):.1f}%)")
        print(f"    Avg steps: {total_steps/len(queries):.2f}")
        print(f"    Avg reward: {total_reward/len(queries):.2f}")
    
    return {
        'total_queries': len(queries),
        'successful': successful,
        'avg_reward': total_reward / len(queries) if queries else 0.0,
        'avg_steps': total_steps / len(queries) if queries else 0.0,
        'traces': all_results
    }
