"""
Test to compare str-based environment with batched tensor environment for equivalence.

This module provides two types of tests:

MAIN TEST (main function):
--------------------------
Runs both environments step-by-step on the same queries (interleaved comparison).
Compares:
1. Action masks (number of available actions)
2. Derived states (after canonicalization)
3. Final success/failure outcomes

Usage:
  # Test 10 queries
  python test_envs/test_envs.py --n_queries 10
  
  # Start from specific query (useful for debugging)
  python test_envs/test_envs.py --start_query 2047 --n_queries 1

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
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_path)
sys.path.insert(0, os.path.join(root_path, 'str_based'))

import argparse
import random
import torch
import numpy as np
from typing import List, Tuple

# String-engine stack
from str_based.str_dataset import DataHandler as StrDataHandler
from str_based.str_index_manager import IndexManager as StrIndexManager
from str_based.str_env import LogicEnv_gym as StrEnv
from str_based.str_utils import Term as StrTerm
from str_based.str_unification import canonicalize_state_to_str, canonical_states_to_str

# Tensor-engine stack
from data_handler import DataHandler
from index_manager import IndexManager
from unification_engine import UnificationEngine
from env import BatchedEnv
from tensordict import TensorDict


def _parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Compare string and tensor versions")
    parser.add_argument('--dataset', type=str, default="family",
                        help='Dataset name to use')
    parser.add_argument('--max_derived_states', type=int, default=500,
                        help='Maximum derived states to generate per step (default: 200)')
    parser.add_argument('--start_query', type=int, default=0,
                        help='Index of the first query to test (default: 0)')
    parser.add_argument('--n_queries', type=int, default=500,
                        help='Number of queries to test (default: None, meaning all)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling queries (default: 42)')
    parser.add_argument('--memory_pruning', type=bool, default=True,
                        help='Enable memory pruning (default: False)') 
    parser.add_argument('--skip_unary_actions', type=bool, default=True,
                        help='Enable skip unary actions (default: False)')
    return parser.parse_args()


def safe_item(x):
    """Safely extract scalar from numpy/torch/python scalar."""
    if isinstance(x, (np.ndarray, np.generic)):
        return x.item() if x.size == 1 else x
    elif isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x
    return x


def print_debug_info(step: int, str_env, batched_env, 
                     str_obs, batched_obs, batched_obs_td, 
                     str_state_canonical: str, batched_state_canonical: str,
                     str_normalized_states, batched_normalized_states, 
                     error_msg: str, query: str, n_query: int):
    """Print detailed debug information when a mismatch occurs."""
    print(f"\n{'*'*60}\nDEBUG INFO for query {n_query}: {query}\n{'*'*60}\n")
    print(f"{'='*60}")
    print(f"DEBUG INFO - STEP {step}")
    print(f"{'='*60}")
    print(f"\nCurrent State:")
    print(f"  Str:     {str_state_canonical}")
    print(f"  Batched: {batched_state_canonical}")
    
    print(f"\nAvailable Actions: {len(str_normalized_states)} (str) / {len(batched_normalized_states)} (batched)")
    
    print(f"\nDerived States (first 20):")
    print(f"  Str:")
    for i, state in enumerate(str_normalized_states[:20]):
        print(f"    [{i}] {state}")
    if len(str_normalized_states) > 20:
        print(f"    ... ({len(str_normalized_states) - 20} more)")
    
    print(f"  Batched:")
    for i, state in enumerate(batched_normalized_states[:20]):
        print(f"    [{i}] {state}")
    if len(batched_normalized_states) > 20:
        print(f"    ... ({len(batched_normalized_states) - 20} more)")
    
    # Get done flags
    str_done = safe_item(str_obs.get('done', False))
    batched_done = safe_item(batched_obs_td.get('done', torch.tensor([False]))[0])
    
    # Get success flags
    str_success = str_obs.get('is_success', False)
    batched_success = safe_item(batched_obs_td.get('is_success', torch.tensor([False]))[0])
    
    # Get labels
    str_label = getattr(str_env, 'current_label', None)
    batched_label = batched_env._all_labels[0].item() if hasattr(batched_env, '_all_labels') else None
    
    print(f"\nEnvironment State:")
    print(f"  Label:      {str_label} (str) / {batched_label} (batched)")
    print(f"  Done:       {str_done} (str) / {batched_done} (batched)")
    print(f"  Success:    {str_success} (str) / {batched_success} (batched)")
    print(f"  Truncated:  {str_obs.get('truncated', False)} (str)")
    
    print(f"\nError Message:")
    print(f"  {error_msg}")
    print(f"{'='*60}\n")


def compare_query(p: str, h: str, t: str, 
                  split: str,
                  str_env_data: Tuple, 
                  batched_env_data: Tuple,
                  count=None,
                  verbose=False) -> Tuple[bool, bool, bool, str, float, float]:
    """
    Compare str and batched environments for a single query step-by-step.
    
    Returns: (match, str_success, batched_success, error_msg, str_total_reward, batched_total_reward)
    """
    if verbose: 
        print(f"\n{'*'*60}\nComparing query {count}: {p}({h}, {t}) [{split}]\n{'*'*60}\n")
    str_env, str_im = str_env_data
    batched_env, engine = batched_env_data
    padding_idx = batched_env.padding_idx

    # Per-query reward tracking
    str_total_reward = 0.0
    batched_total_reward = 0.0
    
    # Reset both environments with the query
    q_str = StrTerm(predicate=p, args=(h, t))
    label = 1  # All queries are true (ground truth label)
    
    # Set up str env - ensure current_query and current_label are set before and maintained after _reset
    str_env.current_query = q_str
    str_env.current_label = label
    str_env.current_query_depth_value = None
    str_obs, _ = str_env._reset([q_str], label)
    # Re-ensure label is set after reset (in case _reset clears it)
    str_env.current_label = label
    
    # Set up batched env (batch_size=1)
    query_atom = batched_env.unification_engine.index_manager.atom_to_tensor(p, h, t)
    query_padded = torch.full((1, batched_env.padding_atoms, 3), batched_env.padding_idx, 
                               dtype=torch.long, device='cpu')
    query_padded[0, 0] = query_atom
    
    batched_env._all_queries_padded = query_padded
    batched_env._all_labels = torch.tensor([label], dtype=torch.long, device='cpu')
    batched_env._all_depths = torch.tensor([1], dtype=torch.long, device='cpu')
    batched_env._all_first_atoms = query_atom.unsqueeze(0)
    batched_env._num_all = 1
    
    batched_obs_td = batched_env.reset()
    batched_obs = {
        'sub_index': batched_obs_td['sub_index'].cpu().numpy(),
        'derived_sub_indices': batched_obs_td['derived_sub_indices'].cpu().numpy(),
        'action_mask': batched_obs_td['action_mask'].cpu().numpy()
    }
    
    max_depth = 20
    str_done_flag = False
    batched_done_flag = False
    
    for step in range(max_depth):
        # Check if already done from previous step
        if str_done_flag and batched_done_flag:
            # Both were marked done in previous step
            str_success = str_info.get('is_success', False)
            batched_success = safe_item(batched_obs_td.get('is_success', torch.tensor([False]))[0])
            return True, str_success, batched_success, "match", float(str_total_reward), float(batched_total_reward)
        elif str_done_flag or batched_done_flag:
            error_msg = f"TERMINATION MISMATCH: one env done but not the other (str={str_done_flag}, batched={batched_done_flag})"
            print_debug_info(step, str_env, batched_env, str_obs, batched_obs, batched_obs_td,
                           "(already done)", "(already done)", [], [], error_msg, f"{p}({h}, {t})", count)
            return False, False, False, error_msg, float(str_total_reward), float(batched_total_reward)
        
        # Get current states
        str_state = str_env.tensordict['state']
        batched_state = batched_env.current_queries[0]  # [A, D]
        
        # Use engine methods to canonicalize states
        str_state_canonical = canonicalize_state_to_str(str_state)
        batched_state_canonical = engine.deb.canonical_state_to_str(batched_state)
        
        # Get derived states (available actions)
        str_derived_states = str_env.tensordict['derived_states']
        batched_derived_states = batched_env.derived_states_batch[0]  # [S, A, D]
        
        # Extract action masks
        str_action_mask = str_obs['action_mask']
        batched_action_mask = batched_obs['action_mask'][0]
        
        num_str_actions = safe_item(str_action_mask.sum())
        num_batched_actions = safe_item(batched_action_mask.sum())
        
        # Check if both are done (done flag is True)
        str_done = str_done_flag
        batched_done = batched_done_flag

        if verbose: 
            # print_debug_info(step, str_env, batched_env, str_obs, batched_obs, batched_obs_td,
            #                str_state_canonical, batched_state_canonical, 
            #                [], [], "Step debug info")
            print(f"\n{'='*80}")
            print(f"STEP {step}")
            print(f"{'='*80}")
            print(f"\n  [STR] Current state:")
            for line in str_state_canonical.split('\n'):
                print(f"    {line}")    
            print(f"\n  [BATCHED] Current state:")
            for line in batched_state_canonical.split('\n'):
                print(f"    {line}")
            print(f"\n  [STR] Derived states:")
            for i in range(len(str_derived_states)):
                str_canon = canonicalize_state_to_str(str_derived_states[i])
                print(f"    [{i}] {str_canon}")
            print(f"\n  [BATCHED] Derived states:")
            for i in range(len(batched_derived_states)):
                batched_canon = engine.deb.canonical_state_to_str(batched_derived_states[i])
                # if it is not empty
                if batched_canon.strip():
                    print(f"    [{i}] {batched_canon}")

        
        if str_done and batched_done:
            # Both terminated - check success flags
            str_success = str_obs.get('is_success', False)
            batched_success = safe_item(batched_obs_td.get('is_success', torch.tensor([False]))[0])
            
            if str_success != batched_success:
                error_msg = f"SUCCESS MISMATCH: str={str_success}, batched={batched_success}"
                print_debug_info(step, str_env, batched_env, str_obs, batched_obs, batched_obs_td,
                               str_state_canonical, batched_state_canonical, 
                               str_normalized_states if 'str_normalized_states' in locals() else [],
                               batched_normalized_states if 'batched_normalized_states' in locals() else [],
                               error_msg, f"{p}({h}, {t})", count)
                return False, str_success, batched_success, error_msg, float(str_total_reward), float(batched_total_reward)
            
            return True, str_success, batched_success, "match", float(str_total_reward), float(batched_total_reward)
        
        if str_done or batched_done:
            error_msg = f"TERMINATION MISMATCH at step {step}: str_done={str_done}, batched_done={batched_done}"
            print_debug_info(step, str_env, batched_env, str_obs, batched_obs, batched_obs_td,
                           str_state_canonical, batched_state_canonical, [], [], error_msg, f"{p}({h}, {t})", count)
            return False, False, False, error_msg, float(str_total_reward), float(batched_total_reward)
        
        # States are already in canonical order from the engine (sorted before memory pruning)
        # Memory pruning preserves the relative order, so we should NOT sort again
        # Just compare the states directly in their existing order
        
        # Strict comparison: check action count
        if num_str_actions != num_batched_actions:
            error_msg = (
                f"ACTION COUNT MISMATCH at step {step}: "
                f"str={num_str_actions} vs batched={num_batched_actions}"
            )
            # For debug info, canonicalize states for display only
            str_normalized_states = [canonicalize_state_to_str(s) for s in str_derived_states[:num_str_actions]]
            batched_normalized_states = [engine.deb.canonical_state_to_str(batched_derived_states[i]) 
                                          for i in range(num_batched_actions)]
            print_debug_info(step, str_env, batched_env, str_obs, batched_obs, batched_obs_td,
                           str_state_canonical, batched_state_canonical,
                           str_normalized_states, batched_normalized_states, error_msg, f"{p}({h}, {t})", count)
            return False, False, False, error_msg, float(str_total_reward), float(batched_total_reward)

        # Compare states directly in their existing order (already canonical from engine)
        for idx in range(num_str_actions):
            str_canonical = canonicalize_state_to_str(str_derived_states[idx])
            batched_canonical = engine.deb.canonical_state_to_str(batched_derived_states[idx])
            if str_canonical != batched_canonical:
                error_msg = (f"STATE MISMATCH at step {step}, action {idx}:\n"
                             f"  Str:     {str_canonical}\n"
                             f"  Batched: {batched_canonical}")
                # For debug info, canonicalize all states for display
                str_normalized_states = [canonicalize_state_to_str(s) for s in str_derived_states[:num_str_actions]]
                batched_normalized_states = [engine.deb.canonical_state_to_str(batched_derived_states[i]) 
                                              for i in range(num_batched_actions)]
                print_debug_info(step, str_env, batched_env, str_obs, batched_obs, batched_obs_td,
                               str_state_canonical, batched_state_canonical,
                               str_normalized_states, batched_normalized_states, error_msg, f"{p}({h}, {t})", count)
                return False, False, False, error_msg, float(str_total_reward), float(batched_total_reward)

        # Deterministic: take first available action (already in canonical order from engine)
        str_action = 0
        batched_action = 0
        
        # Step str environment
        str_obs, str_reward, str_done_flag, str_truncated, str_info = str_env.step(str_action)
        str_total_reward += safe_item(str_reward)
        
        # Step batched environment
        batched_action_td = TensorDict({
            'action': torch.tensor([batched_action], dtype=torch.long, device='cpu')
        }, batch_size=[1])
        batched_result_td = batched_env.step(batched_action_td)
        
        # TorchRL's EnvBase wraps output in 'next' for observations
        # Reward and done are at the top level of the returned TD
        if 'next' in batched_result_td.keys():
            batched_obs_td = batched_result_td['next']
            batched_done_flag = safe_item(batched_obs_td['done'][0])
            # Reward should be in next as well
            batched_reward = safe_item(batched_obs_td.get('reward', batched_result_td.get('reward', torch.tensor([0.0])))[0])
        else:
            batched_obs_td = batched_result_td
            batched_done_flag = safe_item(batched_result_td['done'][0])
            batched_reward = safe_item(batched_result_td.get('reward', torch.tensor([0.0]))[0])
        
        batched_obs = {
            'sub_index': batched_obs_td['sub_index'].cpu().numpy(),
            'derived_sub_indices': batched_obs_td['derived_sub_indices'].cpu().numpy(),
            'action_mask': batched_obs_td['action_mask'].cpu().numpy()
        }
        
        # Check that rewards match
        str_reward_val = safe_item(str_reward)
        if str_reward_val != batched_reward:
            error_msg = f"REWARD MISMATCH at step {step}: str_reward={str_reward_val}, batched_reward={batched_reward}"
            print_debug_info(step, str_env, batched_env, str_obs, batched_obs, batched_obs_td,
                           str_state_canonical, batched_state_canonical,
                           str_normalized_states, batched_normalized_states, error_msg, f"{p}({h}, {t})", count)
            raise AssertionError(error_msg)

        batched_total_reward += float(batched_reward)
        
        # Check if both are done after step
        if str_done_flag and batched_done_flag:
            str_success = str_info.get('is_success', False)
            batched_success = safe_item(batched_obs_td.get('is_success', torch.tensor([False]))[0])
            
            if str_success != batched_success:
                error_msg = f"SUCCESS FLAG MISMATCH: str={str_success}, batched={batched_success}"
                print_debug_info(step, str_env, batched_env, str_obs, batched_obs, batched_obs_td,
                               str_state_canonical, batched_state_canonical,
                               str_normalized_states, batched_normalized_states, error_msg, f"{p}({h}, {t})", count)
                return False, str_success, batched_success, error_msg, float(str_total_reward), float(batched_total_reward)
            
            return True, str_success, batched_success, "match", float(str_total_reward), float(batched_total_reward)
        

    
    # Reached max depth
    return True, False, False, "max_depth", float(str_total_reward), float(batched_total_reward)



def load_str_environment(dataset: str, max_derived_states: int, memory_pruning: bool = False, skip_unary_actions: bool = False) -> Tuple:
    """Load and configure the string-based environment.
    
    Returns:
        Tuple of (str_env, im_str)
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
        padding_atoms=100,
        max_arity=dh_str.max_arity,
        device=torch.device('cpu'),
    )
    
    facts_set = set(dh_str.facts)
    im_str.build_fact_index(list(facts_set))
    
    str_env = StrEnv(
        index_manager=im_str,
        data_handler=dh_str,
        queries=dh_str.train_queries,
        labels=[1] * len(dh_str.train_queries),
        query_depths=[None] * len(dh_str.train_queries),
        facts=facts_set,
        mode='eval_with_restart',
        seed=42,
        max_depth=20,
        memory_pruning=memory_pruning,
        padding_atoms=100,
        padding_states=max_derived_states,
        verbose=0,
        prover_verbose=0,
        device=torch.device('cpu'),
        engine='python',
        engine_strategy='complete',
        skip_unary_actions=skip_unary_actions,
        endf_action=False,
        canonical_action_order=True,
    )
    
    return str_env, im_str, dh_str


def load_tensor_environment(dataset: str, max_derived_states: int, memory_pruning: bool = False, skip_unary_actions: bool = False) -> Tuple:
    """Load and configure the batched tensor environment.
    
    Returns:
        Tuple of (batched_env, engine, im_batched)
    """
    dh_batched = DataHandler(
        dataset_name=dataset,
        base_path="./data/",
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
        max_derived_per_state=max_derived_states,
        sort_states=True
    )
    engine.index_manager = im_batched
    
    dummy_query = torch.full((1, 100, 3), im_batched.padding_idx, dtype=torch.long, device='cpu')
    
    batched_env = BatchedEnv(
        batch_size=1,
        queries=dummy_query,
        labels=torch.ones(1, dtype=torch.long, device='cpu'),
        query_depths=torch.ones(1, dtype=torch.long, device='cpu'),
        unification_engine=engine,
        mode='train',
        max_depth=20,
        memory_pruning=memory_pruning,
        padding_atoms=100,
        padding_states=max_derived_states,
        true_pred_idx=im_batched.predicate_str2idx.get('True'),
        false_pred_idx=im_batched.predicate_str2idx.get('False'),
        end_pred_idx=im_batched.predicate_str2idx.get('End'),
        verbose=0,
        prover_verbose=0,
        device=torch.device('cpu'),
        runtime_var_start_index=im_batched.constant_no + 1,
        total_vocab_size=im_batched.constant_no + 1000000,
        skip_unary_actions=skip_unary_actions,
        end_proof_action=False,
        use_exact_memory=True,
    )
    
    return batched_env, engine, dh_batched


def main():
    args = _parse_args()
    start_query = args.start_query
    n_queries = args.n_queries
    dataset = args.dataset
    seed = args.seed
    
    # Reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    SEED = seed

    print("="*60)
    print("Environment Equivalence Test")
    print("="*60)
    print(f"Dataset: {dataset}")
    print(f"Seed: {seed}")
    if n_queries is not None:
        print(f"Testing {n_queries} queries (starting from {start_query})")
    print()
    
    print("Loading environments...")
    str_env, im_str, dh_str = load_str_environment(dataset, args.max_derived_states, args.memory_pruning, args.skip_unary_actions)
    batched_env, engine, dh_batched = load_tensor_environment(dataset, args.max_derived_states, args.memory_pruning, args.skip_unary_actions)
    
    # Prepare test data
    all_queries = (
        [('train', q) for q in dh_str.train_queries] +
        [('valid', q) for q in dh_str.valid_queries] +
        [('test', q) for q in dh_str.test_queries]
    )
    random.Random(SEED).shuffle(all_queries)
    
    # Apply filtering
    end_idx = start_query + n_queries if n_queries is not None else len(all_queries)
    all_queries = all_queries[start_query:end_idx]

    # # use the query uncle(1515, 1398)
    # all_queries = [('test', StrTerm(predicate='uncle', args=('1515', '1398')))]
    
    print(f"Testing {len(all_queries)} queries (range: {start_query}-{start_query + len(all_queries) - 1})")
    print()
    
    str_env_data = (str_env, im_str)
    batched_env_data = (batched_env, engine)
    
    matches = 0
    mismatches = 0
    str_proven = 0
    batched_proven = 0
    both_proven = 0
    both_failed = 0
    total_str_reward = 0.0
    total_batched_reward = 0.0
    reward_mismatches = 0
    
    for i, (split, query) in enumerate(all_queries):
        p, h, t = query.predicate, query.args[0], query.args[1]
        
        try:
            match, str_success, batched_success, error, str_q_reward, batched_q_reward = compare_query(
                p, h, t, split, str_env_data, batched_env_data, count=start_query + i
            )
        except TypeError as e:
            # Handle label errors
            print(f"\nTypeError at query {i} [{split}]: {p}({h}, {t})")
            print(f"Error: {e}")
            print(f"str_env.current_label: {str_env.current_label}")
            print(f"str_env.current_query: {str_env.current_query}")
            raise
        except AssertionError as e:
            # Print totals up-to-now for debugging, then re-raise
            print("\nAssertionError during comparison:", e)
            print("Totals up to failure:")
            print(f"  Total str reward: {total_str_reward:.2f}")
            print(f"  Total batched reward: {total_batched_reward:.2f}")
            # If this was a reward mismatch, increment count for the printed summary
            if 'REWARD MISMATCH' in str(e):
                reward_mismatches += 1
            print(f"  Reward mismatches: {reward_mismatches}")
            raise
        
        if match:
            print(f"✓ Query {start_query + i} MATCH [{split}]: {p}({h}, {t})")
            matches += 1
            total_str_reward += float(str_q_reward)
            total_batched_reward += float(batched_q_reward)
            
            # Track proof success statistics
            if str_success:
                str_proven += 1
            if batched_success:
                batched_proven += 1
            if str_success and batched_success:
                both_proven += 1
            if not str_success and not batched_success:
                both_failed += 1
        else:
            raise AssertionError(
                f"Environment mismatch at query {start_query + i} [{split}] {p}({h}, {t}): {error}"
            )

    
    print()
    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"Matches: {matches}/{matches+mismatches}")
    print(f"Mismatches: {mismatches}/{matches+mismatches}")
    
    # Print proof success statistics
    if matches > 0:
        str_success_rate = str_proven / matches
        batched_success_rate = batched_proven / matches
        print()
        print("="*60)
        print("PROOF SUCCESS STATISTICS")
        print("="*60)
        print(f"Queries proven (both envs): {both_proven}/{matches} ({both_proven/matches*100:.2f}%)")
        print(f"Queries failed (both envs):  {both_failed}/{matches} ({both_failed/matches*100:.2f}%)")
        print(f"Str env success rate:        {str_proven}/{matches} ({str_success_rate*100:.2f}%)")
        print(f"Batched env success rate:    {batched_proven}/{matches} ({batched_success_rate*100:.2f}%)")
        print(f"\nAverage success rate: {(str_success_rate + batched_success_rate) / 2 * 100:.2f}%")
        print(f"(Success rate = queries proven within max depth)")
        if str_success_rate > 0.001 and str_success_rate < 0.999:
            print(f"✓ Success rate is between 0% and 100% (non-trivial: {str_success_rate*100:.2f}%)")
        elif str_success_rate < 0.001:
            print(f"⚠ Success rate is 0% - all queries require proof beyond max_depth or are unprovable")
    
    # Print reward statistics
    if matches > 0:
        avg_str_reward = total_str_reward / matches
        avg_batched_reward = total_batched_reward / matches
        print()
        print("="*60)
        print("REWARD STATISTICS")
        print("="*60)
        print(f"Total str reward:       {total_str_reward:.2f}")
        print(f"Total batched reward:   {total_batched_reward:.2f}")
        print(f"Avg str reward/query:   {avg_str_reward:.4f}")
        print(f"Avg batched reward/query: {avg_batched_reward:.4f}")
        print(f"Reward mismatches:      {reward_mismatches} (should be 0)")
        print(f"Seed: 42, Dataset: {dataset}")
        
        # Check if rewards are non-trivial
        if avg_str_reward > 0.001 and avg_str_reward < 0.999:
            print(f"✓ Average reward is between 0 and 1 (non-trivial: {avg_str_reward:.4f})")
        elif avg_str_reward < 0.001:
            print(f"⚠ Average reward is ~0 (no successful proofs with reward)")
        elif avg_str_reward > 0.999:
            print(f"✓ Average reward is ~1 (all queries successfully proven)")


if __name__ == "__main__":
    exit(main())
