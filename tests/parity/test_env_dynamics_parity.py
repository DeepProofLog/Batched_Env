"""
Environment Step-by-Step Parity Tests.

Tests verifying that the tensor-based environment produces the SAME
state transitions, rewards, dones, etc. step-by-step as the SB3 string-based 
environment when given the SAME actions.

This is a direct environment comparison, independent of the PPO policy.

Usage:
    pytest tests/parity/test_env_step_parity.py -v
    python tests/parity/test_env_step_parity.py --dataset countries_s3 --n-envs 4 --n-steps 10
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import pytest
import torch
import numpy as np

# Setup paths
ROOT = Path(__file__).resolve().parents[2]
SB3_ROOT = ROOT / "sb3"
TEST_ENVS_ROOT = ROOT / "tests" / "other" / "test_envs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SB3_ROOT) not in sys.path:
    sys.path.insert(1, str(SB3_ROOT))
if str(TEST_ENVS_ROOT) not in sys.path:
    sys.path.insert(2, str(TEST_ENVS_ROOT))

# SB3 imports
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# sb3 imports
from sb3_dataset import DataHandler as StrDataHandler
from sb3_index_manager import IndexManager as StrIndexManager
from sb3_env import LogicEnv_gym as StrEnv

# Tensor imports
from data_handler import DataHandler
from index_manager import IndexManager
from unification import UnificationEngine
from env import BatchedEnv


def tensor_obs_to_canonical_state(obs, im, env_idx: int = 0) -> str:
    """Convert tensor observation to canonical string representation."""
    try:
        # Handle dict obs from batched env
        if isinstance(obs, dict):
            sub_idx = obs.get('sub_index', obs.get('state'))
            if sub_idx is None:
                return "<no state>"
            
            # Tensor env returns shape [n_envs, 1, A, D] - select env and squeeze
            if len(sub_idx.shape) == 4:  # [n_envs, 1, A, D]
                sub_idx = sub_idx[env_idx, 0]  # [A, D]
            elif len(sub_idx.shape) == 3:  # [n_envs, A, D] or [1, A, D]
                sub_idx = sub_idx[env_idx]
            elif len(sub_idx.shape) == 2:  # [A, D]
                pass  # Already single env
            
            # Convert to list of atoms
            atoms = []
            padding_idx = im.padding_idx
            for atom in sub_idx:
                if isinstance(atom, np.ndarray):
                    pred_idx, arg0_idx, arg1_idx = int(atom[0]), int(atom[1]), int(atom[2])
                elif isinstance(atom, torch.Tensor):
                    pred_idx, arg0_idx, arg1_idx = int(atom[0].item()), int(atom[1].item()), int(atom[2].item())
                else:
                    pred_idx, arg0_idx, arg1_idx = int(atom[0]), int(atom[1]), int(atom[2])
                if pred_idx == padding_idx:
                    continue
                # Get strings - idx2predicate and idx2constant are lists
                pred = im.idx2predicate[pred_idx] if pred_idx < len(im.idx2predicate) else f"<pred_{pred_idx}>"
                arg0 = im.idx2constant[arg0_idx] if arg0_idx < len(im.idx2constant) else f"<C{arg0_idx}>"
                arg1 = im.idx2constant[arg1_idx] if arg1_idx < len(im.idx2constant) else f"<C{arg1_idx}>"
                atoms.append(f"{pred}({arg0},{arg1})")
            return " ; ".join(atoms) if atoms else "<empty>"
        else:
            return f"<unknown obs type: {type(obs)}>"
    except Exception as e:
        return f"<error: {e}>"


def sb3_obs_to_canonical_state(obs, im, env_idx: int = 0) -> str:
    """Convert SB3 observation to canonical string representation."""
    try:
        sub_idx = obs.get('sub_index')
        if sub_idx is None:
            return "<no state>"
        
        # SB3 VecEnv returns shape [n_envs, 1, A, D] - select env and squeeze
        if len(sub_idx.shape) == 4:  # [n_envs, 1, A, D]
            sub_idx = sub_idx[env_idx, 0]  # [A, D]
        elif len(sub_idx.shape) == 3:  # [n_envs, A, D]
            sub_idx = sub_idx[env_idx]
        
        # SB3 IndexManager has str2idx dicts, need to create reverse mappings
        idx2constant = {v: k for k, v in im.constant_str2idx.items()}
        idx2predicate = {v: k for k, v in im.predicate_str2idx.items()}
        
        # Convert to list of atoms
        atoms = []
        padding_idx = im.padding_idx
        for atom in sub_idx:
            if isinstance(atom, np.ndarray):
                pred_idx, arg0_idx, arg1_idx = int(atom[0]), int(atom[1]), int(atom[2])
            else:
                pred_idx, arg0_idx, arg1_idx = int(atom[0]), int(atom[1]), int(atom[2])
            if pred_idx == padding_idx:
                continue
            # Get strings using reverse mappings
            pred = idx2predicate.get(pred_idx, f"<pred_{pred_idx}>")
            arg0 = idx2constant.get(arg0_idx, f"<C{arg0_idx}>")
            arg1 = idx2constant.get(arg1_idx, f"<C{arg1_idx}>")
            atoms.append(f"{pred}({arg0},{arg1})")
        return " ; ".join(atoms) if atoms else "<empty>"
    except Exception as e:
        return f"<error: {e}>"


def create_aligned_environments(dataset: str, n_envs: int):
    """Create SB3 and tensor environments with the SAME queries (all train queries)."""
    base_path = "./data/"
    device = torch.device("cpu")
    padding_atoms = 100
    padding_states = 500
    
    # ===== SB3 Setup =====
    dh_sb3 = StrDataHandler(
        dataset_name=dataset,
        base_path=base_path,
        train_file="train.txt",
        valid_file="valid.txt", 
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        train_depth=None,
    )
    
    im_sb3 = StrIndexManager(
        constants=dh_sb3.constants,
        predicates=dh_sb3.predicates,
        max_total_vars=1000000,
        rules=dh_sb3.rules,
        padding_atoms=padding_atoms,
        max_arity=dh_sb3.max_arity,
        device=device,
    )
    
    facts_set = set(dh_sb3.facts)
    im_sb3.build_fact_index(list(facts_set))
    
    # ===== Tensor Setup =====
    dh_tensor = DataHandler(
        dataset_name=dataset,
        base_path=base_path,
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        train_depth=None,
    )
    
    im_tensor = IndexManager(
        constants=dh_tensor.constants,
        predicates=dh_tensor.predicates,
        max_total_runtime_vars=1000000,
        padding_atoms=padding_atoms,
        max_arity=dh_tensor.max_arity,
        device=device,
        rules=dh_tensor.rules,
    )
    dh_tensor.materialize_indices(im=im_tensor, device=device)
    
    # Use ALL train queries for both (for proper query cycling after episode resets)
    queries = dh_sb3.train_queries
    tensor_queries = dh_tensor.train_queries
    
    # Verify first few queries match (all should match since same dataset)
    for i, (q_sb3, q_tensor) in enumerate(zip(queries[:min(10, len(queries))], tensor_queries[:min(10, len(tensor_queries))])):
        assert q_sb3.predicate == q_tensor.predicate, f"Query {i} predicate mismatch"
        assert q_sb3.args[0] == q_tensor.args[0], f"Query {i} arg0 mismatch"
        assert q_sb3.args[1] == q_tensor.args[1], f"Query {i} arg1 mismatch"
    
    return {
        'sb3': {
            'dh': dh_sb3,
            'im': im_sb3,
            'facts_set': facts_set,
        },
        'tensor': {
            'dh': dh_tensor,
            'im': im_tensor,
        },
        'queries': queries,
        'tensor_queries': tensor_queries,
    }


def create_sb3_env(env_data: Dict, queries: List, n_envs: int):
    """Create SB3 DummyVecEnv with specific queries using train mode."""
    device = torch.device("cpu")
    
    dh = env_data['dh']
    im = env_data['im']
    facts_set = env_data['facts_set']
    
    labels = [1] * len(queries)
    depths = [None] * len(queries)
    
    # For train mode with n_envs, we need all envs to share the same query order
    # The first reset will sample queries 0..(n_envs-1) for envs 0..(n_envs-1)
    def make_env(env_idx: int):
        def _init():
            env = StrEnv(
                index_manager=im,
                data_handler=dh,
                queries=queries,  # All envs get same query list
                labels=labels,
                query_depths=depths,
                facts=facts_set,
                mode='train',  # Use train mode for deterministic round-robin
                sample_deterministic=True,  # Enable deterministic sampling like tensor env
                seed=42,
                max_depth=20,
                memory_pruning=False,
                padding_atoms=100,
                padding_states=500,
                verbose=0,
                prover_verbose=0,
                device=device,
                engine='python',
                engine_strategy='complete',
                skip_unary_actions=False,
                endf_action=False,
                reward_type=0,
                canonical_action_order=True,
            )
            # Set initial train pointer to env_idx so env i starts with query i
            # This matches the tensor env's batched reset which assigns query i to slot i
            env._train_ptr = env_idx
            return Monitor(env)
        return _init
    
    env_fns = [make_env(env_idx=i) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    return vec_env, im


def create_tensor_env(env_data: Dict, queries: List, n_envs: int):
    """Create tensor BatchedEnv with ALL queries (for proper cycling)."""
    device = torch.device("cpu")
    padding_atoms = 100
    padding_states = 500
    
    dh = env_data['dh']
    im = env_data['im']
    
    stringifier_params = {
        'verbose': 0,
        'idx2predicate': im.idx2predicate,
        'idx2constant': im.idx2constant,
        'idx2template_var': im.idx2template_var,
        'padding_idx': im.padding_idx,
        'n_constants': im.constant_no
    }
    
    engine = UnificationEngine.from_index_manager(
        im, take_ownership=True,
        stringifier_params=stringifier_params,
        end_pred_idx=None,
        end_proof_action=False,
        max_derived_per_state=padding_states,
        sort_states=True
    )
    engine.index_manager = im
    
    # Convert ALL queries to tensor format (not just n_envs)
    # This ensures proper query cycling like SB3
    query_tensors = []
    for q in queries:
        query_atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
        query_padded = torch.full((padding_atoms, 3), im.padding_idx, dtype=torch.long, device=device)
        query_padded[0] = query_atom
        query_tensors.append(query_padded)
    
    queries_tensor = torch.stack(query_tensors, dim=0)
    
    env = BatchedEnv(
        batch_size=n_envs,
        queries=queries_tensor,
        labels=torch.ones(len(queries), dtype=torch.long, device=device),
        query_depths=torch.ones(len(queries), dtype=torch.long, device=device),
        unification_engine=engine,
        mode='train',
        max_depth=20,
        memory_pruning=False,
        use_exact_memory=True,
        skip_unary_actions=False,
        end_proof_action=False,
        reward_type=0,
        padding_atoms=padding_atoms,
        padding_states=padding_states,
        true_pred_idx=im.predicate_str2idx.get('True'),
        false_pred_idx=im.predicate_str2idx.get('False'),
        end_pred_idx=im.predicate_str2idx.get('End'),
        verbose=0,
        prover_verbose=0,
        device=device,
        runtime_var_start_index=im.constant_no + 1,
        total_vocab_size=im.constant_no + 1000000,
        sample_deterministic_per_env=True,  # Per-env round-robin for SB3 parity
    )
    
    return env, im, engine


def compare_observations(sb3_obs: Dict, tensor_obs: Dict, sb3_im, tensor_im, n_envs: int) -> List[Dict]:
    """Compare observations from both environments."""
    results = []
    
    for env_idx in range(n_envs):
        env_result = {'env': env_idx, 'matches': {}, 'mismatches': []}
        
        # Compare states
        sb3_state = sb3_obs_to_canonical_state(sb3_obs, sb3_im, env_idx)
        tensor_state = tensor_obs_to_canonical_state(tensor_obs, tensor_im, env_idx)
        
        if sb3_state == tensor_state:
            env_result['matches']['state'] = sb3_state[:50]
        else:
            env_result['mismatches'].append(f"state: SB3='{sb3_state[:50]}', Tensor='{tensor_state[:50]}'")
        
        # Compare action masks
        sb3_mask = sb3_obs['action_mask'][env_idx]
        tensor_mask = tensor_obs['action_mask'][env_idx] if isinstance(tensor_obs['action_mask'], np.ndarray) and len(tensor_obs['action_mask'].shape) == 2 else tensor_obs['action_mask']
        if isinstance(tensor_mask, torch.Tensor):
            tensor_mask = tensor_mask.cpu().numpy()
        
        sb3_valid = int(np.sum(sb3_mask))
        tensor_valid = int(np.sum(tensor_mask))
        
        if sb3_valid == tensor_valid:
            env_result['matches']['num_actions'] = sb3_valid
        else:
            env_result['mismatches'].append(f"num_actions: SB3={sb3_valid}, Tensor={tensor_valid}")
        
        results.append(env_result)
    
    return results


def run_step_by_step_parity_test(dataset: str, n_envs: int, n_steps: int, verbose: bool = False) -> bool:
    """
    Run step-by-step environment comparison with controlled actions.
    
    This test:
    1. Creates aligned SB3 and tensor environments with the same queries
    2. Resets both environments
    3. Takes the SAME action in both (always action 0)
    4. Compares states, rewards, dones at each step
    
    NOTE: With n_envs > 1, query scheduling may differ after episode resets because:
    - SB3's DummyVecEnv: Each env independently cycles through its query list
    - Tensor BatchedEnv: All envs share a global query pointer in 'train' mode
    
    For perfect step-by-step parity, use n_envs=1 or compare only within episodes.
    """
    print(f"\n{'='*70}")
    print(f"Environment Step-by-Step Parity Test")
    print(f"Dataset: {dataset}, n_envs: {n_envs}, n_steps: {n_steps}")
    print(f"{'='*70}")
    
    # Create aligned environments
    print("\nCreating aligned environments...")
    env_data = create_aligned_environments(dataset, n_envs)
    
    # Create environments
    print("\nCreating SB3 environment...")
    sb3_env, sb3_im = create_sb3_env(env_data['sb3'], env_data['queries'], n_envs)
    
    print("Creating tensor environment...")
    tensor_env, tensor_im, engine = create_tensor_env(
        env_data['tensor'], env_data['tensor_queries'], n_envs
    )
    
    # Reset both environments
    print("\nResetting environments...")
    sb3_obs = sb3_env.reset()
    tensor_obs_td = tensor_env.reset()
    
    # Extract tensor obs as dict
    tensor_obs = {
        'sub_index': tensor_obs_td['sub_index'].cpu().numpy(),
        'derived_sub_indices': tensor_obs_td['derived_sub_indices'].cpu().numpy(),
        'action_mask': tensor_obs_td['action_mask'].cpu().numpy(),
    }
    
    # Compare initial observations
    print("\n--- Initial State Comparison ---")
    init_results = compare_observations(sb3_obs, tensor_obs, sb3_im, tensor_im, n_envs)
    
    all_match = True
    total_mismatches = 0
    
    for res in init_results:
        has_mismatch = len(res['mismatches']) > 0
        if has_mismatch:
            all_match = False
            total_mismatches += 1
        
        if verbose or has_mismatch:
            status = "MISMATCH" if has_mismatch else "OK"
            state_info = res['matches'].get('state', res['mismatches'][0] if res['mismatches'] else 'unknown')[:40]
            print(f"  Env {res['env']}: {status} - {state_info}")
            if has_mismatch:
                for m in res['mismatches']:
                    print(f"    {m}")
    
    # Step through both environments
    print("\n--- Step-by-Step Comparison ---")
    
    # Import TensorDict for tensor env actions
    from tensordict import TensorDict
    
    step_results = []
    for step in range(n_steps):
        # Always take action 0 in all environments
        actions = np.zeros(n_envs, dtype=np.int64)
        
        # Step SB3 env (DummyVecEnv auto-resets on done)
        sb3_obs_new, sb3_rewards, sb3_dones, sb3_infos = sb3_env.step(actions)
        
        # Step tensor env using step_and_maybe_reset for auto-reset behavior
        tensor_action_td = TensorDict(
            {"action": torch.from_numpy(actions).long()},
            batch_size=[n_envs]
        )
        tensor_step_td, tensor_next_obs = tensor_env.step_and_maybe_reset(tensor_action_td)
        
        # step_and_maybe_reset returns (step_result, next_obs_for_next_iteration)
        # The next_obs already has reset observations for done envs
        tensor_obs_new = {
            'sub_index': tensor_next_obs['sub_index'].cpu().numpy(),
            'derived_sub_indices': tensor_next_obs['derived_sub_indices'].cpu().numpy(),
            'action_mask': tensor_next_obs['action_mask'].cpu().numpy(),
        }
        # Get rewards/dones from the step result (before reset)
        tensor_next = tensor_step_td['next']
        tensor_rewards = tensor_next['reward'].cpu().numpy()
        tensor_dones = tensor_next['done'].cpu().numpy()
        
        # Compare
        step_result = {'step': step, 'envs': []}
        
        for env_idx in range(n_envs):
            env_result = {'env': env_idx, 'matches': {}, 'mismatches': []}
            
            # Compare states
            sb3_state = sb3_obs_to_canonical_state(sb3_obs_new, sb3_im, env_idx)
            tensor_state = tensor_obs_to_canonical_state(tensor_obs_new, tensor_im, env_idx)
            
            if sb3_state == tensor_state:
                env_result['matches']['state'] = sb3_state[:40]
            else:
                env_result['mismatches'].append(f"state: SB3='{sb3_state[:40]}', Tensor='{tensor_state[:40]}'")
            
            # Compare rewards
            if abs(sb3_rewards[env_idx] - tensor_rewards[env_idx]) < 1e-6:
                env_result['matches']['reward'] = sb3_rewards[env_idx]
            else:
                env_result['mismatches'].append(f"reward: SB3={sb3_rewards[env_idx]}, Tensor={tensor_rewards[env_idx]}")
            
            # Compare dones
            sb3_done = bool(sb3_dones[env_idx])
            tensor_done = bool(tensor_dones[env_idx])
            if sb3_done == tensor_done:
                env_result['matches']['done'] = sb3_done
            else:
                env_result['mismatches'].append(f"done: SB3={sb3_done}, Tensor={tensor_done}")
            
            step_result['envs'].append(env_result)
            
            if env_result['mismatches']:
                all_match = False
                total_mismatches += 1
        
        step_results.append(step_result)
        
        # Print step results
        if verbose:
            print(f"\nStep {step}:")
            for env_result in step_result['envs']:
                has_mismatch = len(env_result['mismatches']) > 0
                status = "MISMATCH" if has_mismatch else "OK"
                info = env_result['matches'].get('state', '')[:30] if not has_mismatch else env_result['mismatches'][0][:50]
                print(f"  Env {env_result['env']}: {status} - {info}")
                if has_mismatch:
                    for m in env_result['mismatches']:
                        print(f"    {m}")
        else:
            # Just print step number and summary
            step_mismatches = sum(1 for e in step_result['envs'] if e['mismatches'])
            if step_mismatches > 0:
                print(f"Step {step}: {step_mismatches}/{n_envs} envs have mismatches")
            elif step % 10 == 0 or step == n_steps - 1:
                print(f"Step {step}: all {n_envs} envs match")
        
        # Update current obs
        sb3_obs = sb3_obs_new
        tensor_obs = tensor_obs_new
    
    # Summarize results
    total_comparisons = (n_steps + 1) * n_envs  # +1 for initial state
    
    print(f"\n{'='*70}")
    print(f"Results: {total_comparisons - total_mismatches}/{total_comparisons} state-env pairs match")
    
    if all_match:
        print("\n✓ ALL STATES, REWARDS, DONES MATCH STEP-BY-STEP")
    else:
        print(f"\n✗ {total_mismatches} mismatches found")
    print(f"{'='*70}")
    
    # Cleanup
    sb3_env.close()
    
    return all_match


class TestEnvStepParity:
    """Pytest class for environment step-by-step parity tests."""
    
    @pytest.mark.parametrize("dataset,n_envs,n_steps", [
        ("countries_s3", 4, 10),
        ("family", 4, 10),
    ])
    def test_env_step_parity_small(self, dataset: str, n_envs: int, n_steps: int):
        """Test environment step parity with small configurations."""
        assert run_step_by_step_parity_test(dataset, n_envs, n_steps), \
            f"Environment step parity test failed for {dataset}"


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test environment step-by-step parity")
    parser.add_argument("--dataset", type=str, default="countries_s3", help="Dataset name")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of environments")
    parser.add_argument("--n-steps", type=int, default=10, help="Number of steps")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    success = run_step_by_step_parity_test(args.dataset, args.n_envs, args.n_steps, args.verbose)
    exit(0 if success else 1)
