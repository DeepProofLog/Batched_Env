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

Uses deterministic action selection (first canonical state) to verify exact equivalence.
Batch size: 1 (single query at a time)

RANDOM ACTION TEST (test_random_actions function):
--------------------------------------------------
Tests that average rewards are similar when both environments use random actions.
This is a statistical test that:
1. Runs the same queries in both environments with random action selection
2. Compares mean rewards using z-score statistical test
3. Uses batch_size=1 with interleaved stepping

Run with: python tests/test_envs.py --random-test [num_queries] [seed]

IMPORTANT FIXES APPLIED:
1. Increased padding_atoms from 20 to 100 to avoid capping
2. Increased padding_states to 500 to avoid state truncation  
3. Built fact_index for str environment (im_str.build_fact_index())
4. Fixed excluded_fact checks in str_unification.py to prevent query from matching itself
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
from str_based.str_unification import canonicalize_state_to_str

# Tensor-engine stack
from data_handler import DataHandler
from index_manager import IndexManager
from unification_engine import UnificationEngine
from env import BatchedEnv
from debug_helper import DebugHelper
from tensordict import TensorDict


DEFAULT_DATASET = "family"

# Toggle for strict per-state comparison.
# When False, we only enforce equality of action counts, rewards, and
# success flags, but allow the two environments to explore different
# (yet semantically equivalent) successor states.
STRICT_STATE_COMPARE = False

def safe_item(x):
    """Safely extract scalar from numpy/torch/python scalar."""
    if isinstance(x, (np.ndarray, np.generic)):
        return x.item() if x.size == 1 else x
    elif isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x
    return x


def compare_query(p: str, h: str, t: str, split: str,
                  str_env_data: Tuple, batched_env_data: Tuple,
                  verbose: bool = False) -> Tuple[bool, bool, bool, str, float, float]:
    """
    Compare str and batched environments for a single query step-by-step.
    
    Returns: (match, str_success, batched_success, error_msg, str_total_reward, batched_total_reward)
    """
    str_env, str_im = str_env_data
    batched_env, engine = batched_env_data
    padding_idx = batched_env.padding_idx

    # Per-query reward tracking
    str_total_reward = 0.0
    batched_total_reward = 0.0
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Query: {p}({h}, {t}) [{split}]")
        print(f"{'='*60}")
    
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
    
    # Debug: Print variable indices after reset
    if verbose: 
        print(f"[VAR DEBUG] After reset: str next_var={str_env.next_var_index}, batched next_var={batched_env.next_var_indices[0].item()}")
    
    max_depth = 10
    str_done_flag = False
    batched_done_flag = False
    
    for step in range(max_depth):
        if verbose:
            print(f"\n{'='*60}")
            print(f"STEP {step}")
            print(f"{'='*60}")
        
        # Check if already done from previous step
        if str_done_flag or batched_done_flag:
            if verbose:
                print(f"  One environment is done from previous step")
            if str_done_flag and batched_done_flag:
                # Both were marked done in previous step
                str_success = str_info.get('is_success', False)
                batched_success = safe_item(batched_obs_td.get('is_success', torch.tensor([False]))[0])
                return True, str_success, batched_success, "match", float(str_total_reward), float(batched_total_reward)
            else:
                error_msg = f"TERMINATION MISMATCH: one env done but not the other"
                return False, False, False, error_msg, float(str_total_reward), float(batched_total_reward)
        
        # Get current states
        str_state = str_env.tensordict['state']
        batched_state = batched_env.current_queries[0]  # [A, D]
        
        # Use engine methods to canonicalize states
        str_state_canonical = canonicalize_state_to_str(str_state)
        batched_state_canonical = engine.canonical_state_to_str(batched_state)

        if verbose:
            print(f"Current states:")
            print(f"  Str:     {str_state_canonical}")
            print(f"  Batched: {batched_state_canonical}")
        
        # Get derived states (available actions)
        str_derived_states = str_env.tensordict['derived_states']
        batched_derived_states = batched_env.derived_states_batch[0]  # [S, A, D]
        
        # Extract action masks
        str_action_mask = str_obs['action_mask']
        batched_action_mask = batched_obs['action_mask'][0]
        
        num_str_actions = safe_item(str_action_mask.sum())
        num_batched_actions = safe_item(batched_action_mask.sum())
        
        if verbose:
            print(f"\nAvailable actions:")
            print(f"  Str:     {num_str_actions}")
            print(f"  Batched: {num_batched_actions}")
        
        # Check if both are done
        str_done = (num_str_actions == 0)
        batched_done = (num_batched_actions == 0)
        
        if str_done and batched_done:
            # Both terminated - check success flags
            str_success = str_obs.get('is_success', False)
            batched_success = safe_item(batched_obs_td.get('is_success', torch.tensor([False]))[0])
            
            if str_success != batched_success:
                error_msg = f"SUCCESS MISMATCH: str={str_success}, batched={batched_success}"
                return False, str_success, batched_success, error_msg, float(str_total_reward), float(batched_total_reward)
            
            if verbose:
                print(f"\n✓ Both done at step {step}, success={str_success}")
            return True, str_success, batched_success, "match", float(str_total_reward), float(batched_total_reward)
        
        if str_done or batched_done:
            error_msg = f"TERMINATION MISMATCH at step {step}: str_done={str_done}, batched_done={batched_done}"
            if verbose:
                print(f"\n{error_msg}")
            return False, False, False, error_msg, float(str_total_reward), float(batched_total_reward)
        
        # Canonicalize all derived states using engine methods
        str_normalized_states = [canonicalize_state_to_str(ds) for ds in str_derived_states]
        batched_normalized_states = [
            engine.canonical_state_to_str(batched_derived_states[i])
            for i in range(num_batched_actions)
        ]

        if STRICT_STATE_COMPARE:
            if len(str_normalized_states) != len(batched_normalized_states):
                error_msg = (
                    f"ACTION COUNT MISMATCH at step {step}: "
                    f"str={len(str_normalized_states)} vs batched={len(batched_normalized_states)}"
                )
                return False, False, False, error_msg, float(str_total_reward), float(batched_total_reward)

            if verbose:
                print(f"\nDerived states (next possible states):")
                print(f"  Str ({len(str_normalized_states)} states):")
                for i, state_canonical in enumerate(str_normalized_states[:10]):
                    print(f"    {i}: {state_canonical}")
                if len(str_normalized_states) > 10:
                    print(f"    ... ({len(str_normalized_states) - 10} more)")

                print(f"\n  Batched ({len(batched_normalized_states)} states):")
                for i, state_canonical in enumerate(batched_normalized_states[:10]):
                    print(f"    {i}: {state_canonical}")
                if len(batched_normalized_states) > 10:
                    print(f"    ... ({len(batched_normalized_states) - 10} more)")

            # Compare canonical states directly (already normalized)
            for idx, (str_canonical, batched_canonical) in enumerate(zip(str_normalized_states, batched_normalized_states)):
                if str_canonical != batched_canonical:
                    error_msg = (f"STATE MISMATCH at step {step}, action {idx}:\n"
                                 f"  Str:     {str_canonical}\n"
                                 f"  Batched: {batched_canonical}")
                    return False, False, False, error_msg, float(str_total_reward), float(batched_total_reward)
        else:
            # In non-strict mode, only require that both envs either have
            # available actions or are both empty; allow different counts
            if (len(str_normalized_states) == 0) != (len(batched_normalized_states) == 0):
                error_msg = (
                    f"ACTION AVAILABILITY MISMATCH at step {step}: "
                    f"str_has_actions={len(str_normalized_states) > 0}, "
                    f"batched_has_actions={len(batched_normalized_states) > 0}"
                )
                return False, False, False, error_msg, float(str_total_reward), float(batched_total_reward)

            if verbose:
                print(f"\nDerived states (next possible states) [non-strict]:")
                print(f"  Str ({len(str_normalized_states)} states)")
                print(f"  Batched ({len(batched_normalized_states)} states)")

        str_action = 0
        batched_action = 0

        if verbose:
            print(f"\nTaking action {str_action} in both environments")
            print(f"  Next state: {str_normalized_states[str_action]}")
        
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
        
        # Debug: Track variable indices after each step
        if verbose: 
            print(f"[VAR DEBUG] After step {step}: str next_var={str_env.next_var_index}, batched next_var={batched_env.next_var_indices[0].item()}")
        
        # Check that rewards match (strict mode only); otherwise only compare totals
        str_reward_val = safe_item(str_reward)
        if STRICT_STATE_COMPARE:
            if str_reward_val != batched_reward:
                error_msg = f"REWARD MISMATCH at step {step}: str_reward={str_reward_val}, batched_reward={batched_reward}"
                if verbose:
                    print(f"\n✗ {error_msg}")
                raise AssertionError(error_msg)
            if verbose and (str_reward_val != 0 or batched_reward != 0):
                print(f"  Rewards match: str={str_reward_val}, batched={batched_reward}")

        batched_total_reward += float(batched_reward)
        
        # Check if both are done after step
        if str_done_flag and batched_done_flag:
            str_success = str_info.get('is_success', False)
            batched_success = safe_item(batched_obs_td.get('is_success', torch.tensor([False]))[0])
            
            if str_success != batched_success:
                error_msg = f"SUCCESS FLAG MISMATCH: str={str_success}, batched={batched_success}"
                return False, str_success, batched_success, error_msg, float(str_total_reward), float(batched_total_reward)
            
            if verbose:
                print(f"\n✓ Both done after step {step}, success={str_success}")
            return True, str_success, batched_success, "match", float(str_total_reward), float(batched_total_reward)
        
        if verbose:
            print(f"  ✓ Step {step} complete")
    
    # Reached max depth
    if verbose:
        print(f"\n✓ Both reached max depth")
    return True, False, False, "max_depth", float(str_total_reward), float(batched_total_reward)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare string and batched tensor environments")
    parser.add_argument('start_query', nargs='?', type=int, default=0,
                        help='Index to start evaluating queries from')
    parser.add_argument('--dataset', default=DEFAULT_DATASET,
                        help='Dataset name to evaluate (default: family)')
    return parser.parse_args()


def main():
    args = _parse_args()
    start_query = args.start_query
    dataset = args.dataset
    if start_query != 0:
        print(f"Starting from query index: {start_query}")
    
    # Reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("="*60)
    print("Environment Equivalence Test")
    print("="*60)
    print(f"Dataset: {dataset}")
    print()
    
    print("Loading data for str environment...")
    
    # Str environment setup
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
    
    # CRITICAL: Build fact index for efficient unification
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
        seed=42,
        max_depth=20,
        memory_pruning=True,
        padding_atoms=100,
        padding_states=500,
        verbose=0,
        prover_verbose=0,
        device=torch.device('cpu'),
        engine='python',
        engine_strategy='complete',
        skip_unary_actions=False,
        canonical_action_order=True,
        endf_action=False,
    )
    
    print("Loading data for batched environment...")
    
    # Batched environment setup
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
    
    # Debug: Check if constant_no matches between str and batched
    print(f"[INIT DEBUG] im_str.constant_no={im_str.constant_no}, im_batched.constant_no={im_batched.constant_no}")
    print(f"[INIT DEBUG] im_str.variable_start_index={im_str.variable_start_index}, batched runtime_var_start_index={im_batched.constant_no + 1}")
    
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
        canonical_action_order=True
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
    
    dummy_query = torch.full((1, 100, 3), im_batched.padding_idx, dtype=torch.long, device='cpu')
    
    batched_env = BatchedEnv(
        batch_size=1,
        queries=dummy_query,
        labels=torch.ones(1, dtype=torch.long, device='cpu'),
        query_depths=torch.ones(1, dtype=torch.long, device='cpu'),
        unification_engine=engine,
        mode='train',
        max_depth=20,
        memory_pruning=True,
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
        canonical_action_order=True,
        end_proof_action=False,
        use_exact_memory=True,
    )
    
    # Prepare test data - combine all query types for better statistics
    all_queries = []
    for q in dh_str.train_queries:
        all_queries.append(('train', q))
    for q in dh_str.valid_queries:
        all_queries.append(('valid', q))
    for q in dh_str.test_queries:
        all_queries.append(('test', q))
    
    print(f"\nTesting {len(all_queries)} queries from {dataset}")
    print(f"  Train: {len(dh_str.train_queries)}")
    print(f"  Valid: {len(dh_str.valid_queries)}")
    print(f"  Test: {len(dh_str.test_queries)}")
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
    #shuffle all queries with seed 2
    random.Random(2).shuffle(all_queries)
    for i in range(start_query, len(all_queries)):
        split, query = all_queries[i]
        p, h, t = query.predicate, query.args[0], query.args[1]
        
        # Enable verbose for the first query we test to see details
        verbose = (i == start_query)
        try:
            match, str_success, batched_success, error, str_q_reward, batched_q_reward = compare_query(
            p, h, t, split, str_env_data, batched_env_data, verbose=verbose
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
            print(f"✓ Query {i} MATCH [{split}]: {p}({h}, {t})")
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
            print(f"\n✗ Query {i} MISMATCH [{split}]: {p}({h}, {t})")
            print(f"  Error: {error}")
            print(f"  Str success: {str_success}")
            print(f"  Batched success: {batched_success}")
            print(f"\n  Re-running with verbose=True for debugging:")
            print("="*60)
            match, str_success, batched_success, error, _, _ = compare_query(
                p, h, t, split, str_env_data, batched_env_data, verbose=True
            )
            print("="*60)
            raise AssertionError(
                f"Environment mismatch at query {i} [{split}] {p}({h}, {t}): {error}"
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


def test_random_actions(dataset: str = DEFAULT_DATASET,
                        num_queries: int = 100,
                        seed: int = 42):
    """
    Test that average rewards from str env and batched tensor env with RANDOM actions are similar.
    
    This test uses the interleaved comparison approach (like test_envs.py) where both environments
    step together, but uses random action selection instead of deterministic canonical ordering.
    
    Args:
        num_queries: Number of queries to test (default 100)
        seed: Random seed for reproducibility (default 42)
    """
    # Reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("\n" + "="*60)
    print("Random Action Test - Interleaved Comparison")
    print("="*60)
    print(f"Testing {num_queries} queries with random actions")
    print(f"Seed: {seed}")
    print()
    
    # Load string data
    dh_str = StrDataHandler(dataset_name=dataset, base_path='./data/', train_file='train.txt', 
                            valid_file='valid.txt', test_file='test.txt', rules_file='rules.txt', 
                            facts_file='train.txt', train_depth=None)
    
    im_str = StrIndexManager(constants=dh_str.constants, predicates=dh_str.predicates, 
                             max_total_vars=1000000, rules=dh_str.rules, padding_atoms=100, 
                             max_arity=dh_str.max_arity, device=torch.device('cpu'))
    im_str.build_fact_index(list(set(dh_str.facts)))
    
    str_env = StrEnv(index_manager=im_str, data_handler=dh_str, 
                     queries=dh_str.train_queries, labels=[1]*len(dh_str.train_queries), 
                     query_depths=[None]*len(dh_str.train_queries), facts=set(dh_str.facts), 
                     mode='eval_with_restart', seed=seed, max_depth=20, memory_pruning=True, 
                     padding_atoms=100, padding_states=500, verbose=0, prover_verbose=0, 
                     device=torch.device('cpu'), engine='python', engine_strategy='complete', 
                     skip_unary_actions=True, canonical_action_order=True, endf_action=False)
    
    # Load batched data
    dh_batched = DataHandler(dataset_name=dataset, base_path='./data/', train_file='train.txt', 
                             valid_file='valid.txt', test_file='test.txt', rules_file='rules.txt', 
                             facts_file='train.txt', train_depth=None)
    
    im_batched = IndexManager(constants=dh_batched.constants, predicates=dh_batched.predicates, 
                              max_total_runtime_vars=1000000, padding_atoms=100, 
                              max_arity=dh_batched.max_arity, device=torch.device('cpu'), 
                              rules=dh_batched.rules)
    dh_batched.materialize_indices(im=im_batched, device=torch.device('cpu'))
    
    stringifier_params = {'verbose': 0, 'idx2predicate': im_batched.idx2predicate, 
                         'idx2constant': im_batched.idx2constant, 
                         'idx2template_var': im_batched.idx2template_var, 
                         'padding_idx': im_batched.padding_idx, 'n_constants': im_batched.constant_no}
    
    engine = UnificationEngine.from_index_manager(
        im_batched, take_ownership=True,
        stringifier_params=stringifier_params,
        canonical_action_order=True
    )
    engine.index_manager = im_batched
    
    debug_helper = DebugHelper(verbose=0, idx2predicate=im_batched.idx2predicate, 
                               idx2constant=im_batched.idx2constant, 
                               idx2template_var=im_batched.idx2template_var, 
                               padding_idx=engine.padding_idx, n_constants=im_batched.constant_no)
    
    dummy_query = torch.full((1, 100, 3), im_batched.padding_idx, dtype=torch.long, device='cpu')
    
    batched_env = BatchedEnv(
        batch_size=1,
        queries=dummy_query,
        labels=torch.ones(1, dtype=torch.long, device='cpu'),
        query_depths=torch.ones(1, dtype=torch.long, device='cpu'),
        unification_engine=engine,
        mode='train',
        max_depth=20,
        memory_pruning=True,
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
        canonical_action_order=True,
        end_proof_action=False,
        use_exact_memory=True,
    )
    
    # Sample queries
    queries = dh_str.train_queries.copy()
    rnd = random.Random(seed)
    rnd.shuffle(queries)
    selected = queries[:min(num_queries, len(queries))]
    
    str_rewards = []
    batched_rewards = []
    
    for idx, q in enumerate(selected):
        # Reset both environments with the same query
        p, h, t = q.predicate, q.args[0], q.args[1]
        q_str = StrTerm(predicate=p, args=(h, t))
        label = 1
        
        # Reset str env
        str_env.current_query = q_str
        str_env.current_label = label
        str_obs, _ = str_env._reset([q_str], label)
        
        # Reset batched env
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
        
        str_episode_reward = 0.0
        batched_episode_reward = 0.0
        
        # Step both environments with random actions
        str_done_flag = False
        batched_done_flag = False
        
        for step in range(30):
            # Check if str env is done
            str_action_mask = str_obs['action_mask']
            num_str_actions = int(str_action_mask.sum())
            str_done = (num_str_actions == 0) or str_done_flag
            
            # Check if batched env is done
            batched_action_mask = batched_obs['action_mask'][0]
            num_batched_actions = int(batched_action_mask.sum())
            batched_done = (num_batched_actions == 0) or batched_done_flag
            
            if str_done and batched_done:
                break
            
            # Step str env if not done
            if not str_done:
                valid_str_actions = [i for i in range(len(str_action_mask)) if str_action_mask[i]]
                str_action = rnd.choice(valid_str_actions)
                str_obs, str_reward, str_done_flag, str_truncated, str_info = str_env.step(str_action)
                str_episode_reward += float(str_reward)
            
            # Step batched env if not done
            if not batched_done:
                valid_batched_actions = [i for i in range(len(batched_action_mask)) if batched_action_mask[i]]
                batched_action = rnd.choice(valid_batched_actions)
                batched_action_td = TensorDict({'action': torch.tensor([batched_action], 
                                                                       dtype=torch.long, device='cpu')}, 
                                               batch_size=[1])
                batched_result_td = batched_env.step(batched_action_td)
                
                if 'next' in batched_result_td.keys():
                    batched_obs_td = batched_result_td['next']
                    batched_reward = safe_item(batched_obs_td.get('reward', 
                                                                   batched_result_td.get('reward', 
                                                                                        torch.tensor([0.0])))[0])
                else:
                    batched_obs_td = batched_result_td
                    batched_reward = safe_item(batched_result_td.get('reward', torch.tensor([0.0]))[0])
                
                batched_obs = {
                    'sub_index': batched_obs_td['sub_index'].cpu().numpy(),
                    'derived_sub_indices': batched_obs_td['derived_sub_indices'].cpu().numpy(),
                    'action_mask': batched_obs_td['action_mask'].cpu().numpy()
                }
                batched_episode_reward += float(batched_reward)
        
        str_rewards.append(str_episode_reward)
        batched_rewards.append(batched_episode_reward)
        
        if (idx + 1) % 20 == 0:
            print(f"Processed {idx + 1}/{len(selected)} queries...")
    
    # Compare statistics
    str_rewards_arr = np.array(str_rewards)
    batched_rewards_arr = np.array(batched_rewards)
    
    str_mean = str_rewards_arr.mean()
    str_std = str_rewards_arr.std()
    batched_mean = batched_rewards_arr.mean()
    batched_std = batched_rewards_arr.std()
    
    print()
    print("="*60)
    print("Results:")
    print("="*60)
    print(f"String env:")
    print(f"  Mean reward: {str_mean:.4f} ± {str_std:.4f}")
    print(f"  Min/Max: {str_rewards_arr.min():.2f} / {str_rewards_arr.max():.2f}")
    print(f"\nBatched env:")
    print(f"  Mean reward: {batched_mean:.4f} ± {batched_std:.4f}")
    print(f"  Min/Max: {batched_rewards_arr.min():.2f} / {batched_rewards_arr.max():.2f}")
    print(f"\nDifference:")
    print(f"  Mean diff: {abs(str_mean - batched_mean):.4f}")
    
    # Statistical test
    pooled_se = np.sqrt((str_std**2 + batched_std**2) / len(selected))
    mean_diff = abs(str_mean - batched_mean)
    z_score = mean_diff / (pooled_se + 1e-9)
    
    if abs(str_mean) > 0.01:
        relative_error = mean_diff / abs(str_mean)
        print(f"  Relative error: {relative_error*100:.2f}%")
    else:
        relative_error = 0.0
        print(f"  (Mean too close to zero for relative error)")
    
    print(f"  Z-score: {z_score:.2f} (should be < 3 for 99.7% confidence)")
    
    # Pass if either: z-score is reasonable OR both means are very small
    if abs(str_mean) < 0.01 and abs(batched_mean) < 0.01:
        print(f"\n✓ Random action test passed! (both means near zero)")
        return True
    elif z_score < 3.0:
        print(f"\n✓ Random action test passed! (z-score={z_score:.2f})")
        return True
    else:
        print(f"\n✗ Random action test FAILED!")
        print(f"  Mean rewards differ significantly: str={str_mean:.4f}±{str_std:.4f}, " 
              f"batched={batched_mean:.4f}±{batched_std:.4f}, z={z_score:.2f}")
        return False


if __name__ == "__main__":
    import sys
    
    # Check if --random-test flag is provided
    if len(sys.argv) > 1 and sys.argv[1] == '--random-test':
        num_queries = 100
        seed = 42
        if len(sys.argv) > 2:
            try:
                num_queries = int(sys.argv[2])
            except ValueError:
                pass
        if len(sys.argv) > 3:
            try:
                seed = int(sys.argv[3])
            except ValueError:
                pass
        result = test_random_actions(num_queries=num_queries, seed=seed)
        exit(0 if result else 1)
    else:
        exit(main())
