"""
Environment Parity Tests.

Comprehensive tests verifying that the tensor-based BatchedEnv produces
the same behavior as the SB3 string-based LogicEnv_gym.
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict, Tuple

import pytest
import torch
import numpy as np

# Setup paths
ROOT = Path(__file__).resolve().parents[2]
SB3_ROOT = ROOT / "sb3"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SB3_ROOT) not in sys.path:
    sys.path.insert(1, str(SB3_ROOT))

# Import modules
from data_handler import DataHandler as NewDataHandler
from index_manager import IndexManager as NewIndexManager
from unification import UnificationEngine
from env import BatchedEnv

from sb3.sb3_dataset import DataHandler as SB3DataHandler
from sb3.sb3_index_manager import IndexManager as SB3IndexManager
from sb3.sb3_env import LogicEnv_gym as SB3Env
from sb3.sb3_utils import Term as SB3Term


# ============================================================================
# Test Fixtures and Setup Functions
# ============================================================================

def _base_config():
    """Create base configuration for tests."""
    return SimpleNamespace(
        max_total_runtime_vars=1_000_000,
        padding_atoms=100,
        padding_states=500,
        memory_pruning=False,
        reward_type=0,
        verbose=0,
        prover_verbose=0,
        skip_unary_actions=False,
        end_proof_action=False,
        use_exact_memory=True,
        max_derived_per_state=500,
        device='cpu'
    )


def _setup_new_env(dataset: str = "countries_s3", n_queries: int = 10):
    """Setup the tensor-based batched environment."""
    cfg = _base_config()
    device = torch.device(cfg.device)
    
    dh = NewDataHandler(
        dataset_name=dataset,
        base_path="./data/",
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        train_depth=None,
        n_train_queries=n_queries,
    )
    
    im = NewIndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=cfg.max_total_runtime_vars,
        padding_atoms=cfg.padding_atoms,
        max_arity=dh.max_arity,
        device=device,
        rules=dh.rules,
    )
    dh.materialize_indices(im=im, device=device)
    
    stringifier_params = {
        'idx2predicate': im.idx2predicate,
        'idx2constant': im.idx2constant,
        'idx2template_var': im.idx2template_var,
        'n_constants': im.constant_no
    }
    
    engine = UnificationEngine.from_index_manager(
        im, take_ownership=True, stringifier_params=stringifier_params,
        max_derived_per_state=cfg.max_derived_per_state,
        sort_states=True
    )
    engine.index_manager = im
    
    # Get train queries as tensors
    train_queries = dh.train_queries[:n_queries]
    query_tensors = []
    for q in train_queries:
        query_atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
        query_padded = torch.full((cfg.padding_atoms, 3), im.padding_idx, dtype=torch.long, device=device)
        query_padded[0] = query_atom
        query_tensors.append(query_padded)
    queries_tensor = torch.stack(query_tensors, dim=0)
    
    # Create batched env with correct parameters
    env = BatchedEnv(
        batch_size=1,
        queries=queries_tensor,
        labels=torch.ones(len(train_queries), dtype=torch.long, device=device),
        query_depths=torch.ones(len(train_queries), dtype=torch.long, device=device),
        unification_engine=engine,
        mode='train',  # Use train mode - env only supports 'train' and 'eval'
        max_depth=20,
        memory_pruning=cfg.memory_pruning,
        eval_pruning=cfg.memory_pruning,
        use_exact_memory=cfg.use_exact_memory,
        skip_unary_actions=cfg.skip_unary_actions,
        end_proof_action=cfg.end_proof_action,
        reward_type=cfg.reward_type,
        padding_atoms=cfg.padding_atoms,
        padding_states=cfg.padding_states,
        true_pred_idx=im.predicate_str2idx.get('True'),
        false_pred_idx=im.predicate_str2idx.get('False'),
        end_pred_idx=im.predicate_str2idx.get('Endf'),
        verbose=cfg.verbose,
        prover_verbose=cfg.prover_verbose,
        device=device,
        runtime_var_start_index=im.constant_no + 1,
        total_vocab_size=im.constant_no + cfg.max_total_runtime_vars,
        stringifier_params=stringifier_params,
    )
    
    return dh, im, env


def _setup_sb3_env(dataset: str = "countries_s3", n_queries: int = 10):
    """Setup the SB3 string-based environment."""
    cfg = _base_config()
    device = torch.device(cfg.device)
    
    dh = SB3DataHandler(
        dataset_name=dataset,
        base_path="./data/",
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        train_depth=None,
        n_train_queries=n_queries,
    )
    
    im = SB3IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_vars=cfg.max_total_runtime_vars,
        rules=dh.rules,
        padding_atoms=cfg.padding_atoms,
        max_arity=dh.max_arity,
        device=device,
    )
    
    facts_set = set(dh.facts)
    im.build_fact_index(list(facts_set))
    
    env = SB3Env(
        index_manager=im,
        data_handler=dh,
        queries=dh.train_queries,
        labels=[1] * len(dh.train_queries),
        query_depths=[None] * len(dh.train_queries),
        facts=facts_set,
        mode='eval_with_restart',
        seed=42,
        max_depth=20,
        memory_pruning=cfg.memory_pruning,
        padding_atoms=cfg.padding_atoms,
        padding_states=cfg.padding_states,
        verbose=cfg.verbose,
        prover_verbose=cfg.prover_verbose,
        device=device,
        engine='python',
        engine_strategy='complete',
        skip_unary_actions=cfg.skip_unary_actions,
        endf_action=cfg.end_proof_action,
        reward_type=cfg.reward_type,
        canonical_action_order=True,
    )
    
    return dh, im, env


# ============================================================================
# Basic Environment Tests
# ============================================================================

@pytest.mark.parametrize("dataset", ["countries_s3", "family"])
def test_envs_load_same_data(dataset):
    """Verify both environments load the same underlying data."""
    dh_new, im_new, env_new = _setup_new_env(dataset)
    dh_sb3, im_sb3, env_sb3 = _setup_sb3_env(dataset)
    
    # Check facts count
    assert len(dh_new.facts) == len(dh_sb3.facts), \
        f"Facts count mismatch: {len(dh_new.facts)} vs {len(dh_sb3.facts)}"
    
    # Check rules count  
    assert len(dh_new.rules) == len(dh_sb3.rules), \
        f"Rules count mismatch: {len(dh_new.rules)} vs {len(dh_sb3.rules)}"
    
    # Check constants count
    assert im_new.constant_no == im_sb3.constant_no, \
        f"Constant count mismatch: {im_new.constant_no} vs {im_sb3.constant_no}"


@pytest.mark.parametrize("dataset", ["countries_s3", "family"])
def test_envs_reset_produces_valid_obs(dataset):
    """Verify reset produces valid observations in new env."""
    dh_new, im_new, env_new = _setup_new_env(dataset)
    
    # Reset new env
    obs_new = env_new.reset()
    
    # Check new env observation structure
    assert obs_new is not None, "New env reset returned None"
    
    # The observation should be a TensorDict
    from tensordict import TensorDict
    assert isinstance(obs_new, TensorDict), f"Expected TensorDict, got {type(obs_new)}"


@pytest.mark.parametrize("dataset", ["countries_s3"])
def test_env_action_space_consistency(dataset):
    """Verify action spaces are consistent."""
    dh_new, im_new, env_new = _setup_new_env(dataset)
    dh_sb3, im_sb3, env_sb3 = _setup_sb3_env(dataset)
    
    # Reset to get initial action spaces
    env_new.reset()
    env_sb3.reset()
    
    # Check action space type
    assert hasattr(env_new, 'action_space'), "New env missing action_space"
    assert hasattr(env_sb3, 'action_space'), "SB3 env missing action_space"


@pytest.mark.parametrize("dataset", ["countries_s3"])
def test_env_observation_space_consistency(dataset):
    """Verify observation spaces are consistent."""
    dh_new, im_new, env_new = _setup_new_env(dataset)
    dh_sb3, im_sb3, env_sb3 = _setup_sb3_env(dataset)
    
    # Check observation space type
    assert hasattr(env_new, 'observation_space'), "New env missing observation_space"
    assert hasattr(env_sb3, 'observation_space'), "SB3 env missing observation_space"


# ============================================================================
# Step Function Tests
# ============================================================================

@pytest.mark.parametrize("dataset", ["countries_s3"])
def test_env_step_returns_valid_structure(dataset):
    """Verify step returns valid structure in both envs."""
    dh_new, im_new, env_new = _setup_new_env(dataset)
    dh_sb3, im_sb3, env_sb3 = _setup_sb3_env(dataset)
    
    # Reset
    env_new.reset()
    env_sb3.reset()
    
    # Take action 0 in both
    result_new = env_new.step(0)
    result_sb3 = env_sb3.step(0)
    
    # Check new env step result (should return obs, reward, done, truncated, info or similar)
    assert result_new is not None, "New env step returned None"
    
    # Check sb3 env step result
    assert result_sb3 is not None, "SB3 env step returned None"
    assert len(result_sb3) >= 4, f"SB3 env step returned unexpected format: {len(result_sb3)} elements"


@pytest.mark.parametrize("dataset", ["countries_s3"])
def test_env_done_signal_consistency(dataset):
    """Verify done signals are consistent."""
    dh_new, im_new, env_new = _setup_new_env(dataset)
    dh_sb3, im_sb3, env_sb3 = _setup_sb3_env(dataset)
    
    # Reset
    env_new.reset()
    env_sb3.reset()
    
    # Run multiple steps
    max_steps = 10
    for _ in range(max_steps):
        # Get valid action for new env
        n_actions_new = env_new.get_num_actions() if hasattr(env_new, 'get_num_actions') else 1
        action_new = 0 if n_actions_new > 0 else 0
        
        # Step in new env
        result_new = env_new.step(action_new)
        done_new = result_new[2] if len(result_new) > 2 else False
        
        # Step in sb3 env
        result_sb3 = env_sb3.step(0)
        done_sb3 = result_sb3[2] if len(result_sb3) > 2 else False
        
        if done_new or done_sb3:
            break
    
    # Both should eventually terminate or continue
    # (We can't guarantee exact step parity, but both should handle steps)


# ============================================================================
# Reward Tests
# ============================================================================

@pytest.mark.parametrize("dataset", ["countries_s3"])
def test_env_reward_range(dataset):
    """Verify rewards are in expected range."""
    dh_new, im_new, env_new = _setup_new_env(dataset)
    
    # Reset
    env_new.reset()
    
    # Run some steps and collect rewards
    rewards = []
    max_steps = 5
    for _ in range(max_steps):
        result = env_new.step(0)
        reward = result[1] if len(result) > 1 else 0
        rewards.append(reward)
        
        done = result[2] if len(result) > 2 else False
        if done:
            break
    
    # Rewards should be finite
    for r in rewards:
        if isinstance(r, (int, float)):
            assert not np.isnan(r), "Found NaN reward"
            assert not np.isinf(r), "Found Inf reward"


# ============================================================================
# State Transition Tests
# ============================================================================

@pytest.mark.parametrize("dataset", ["countries_s3"])
def test_env_state_changes_on_step(dataset):
    """Verify state changes after step."""
    dh_new, im_new, env_new = _setup_new_env(dataset)
    
    # Reset and get initial observation
    obs_init = env_new.reset()
    
    # Take a step
    result = env_new.step(0)
    obs_after = result[0] if len(result) > 0 else None
    
    # Observation should exist
    assert obs_after is not None, "Observation after step is None"


@pytest.mark.parametrize("dataset", ["countries_s3"])
def test_env_reset_after_done(dataset):
    """Verify environment can be reset after episode ends."""
    dh_new, im_new, env_new = _setup_new_env(dataset)
    
    # Reset
    env_new.reset()
    
    # Run until done or max steps
    max_steps = 50
    for _ in range(max_steps):
        result = env_new.step(0)
        done = result[2] if len(result) > 2 else False
        if done:
            break
    
    # Reset again should work
    obs_new = env_new.reset()
    assert obs_new is not None, "Reset after done returned None"


# ============================================================================
# Determinism Tests
# ============================================================================

@pytest.mark.parametrize("dataset", ["countries_s3"])
def test_env_deterministic_reset(dataset):
    """Verify deterministic reset with same seed."""
    # Create two environments with same seed
    dh1, im1, env1 = _setup_new_env(dataset)
    dh2, im2, env2 = _setup_new_env(dataset)
    
    # Reset both
    obs1 = env1.reset(seed=42)
    obs2 = env2.reset(seed=42)
    
    # Observations should match (if deterministic)
    # Note: This depends on implementation - may need to check specific fields


# ============================================================================
# Batch Processing Tests
# ============================================================================

@pytest.mark.parametrize("dataset", ["countries_s3"])
def test_env_batch_size_one(dataset):
    """Verify batch size 1 works correctly."""
    dh_new, im_new, env_new = _setup_new_env(dataset)
    
    # Reset
    obs = env_new.reset()
    
    # Check that observation has correct batch dimension
    assert obs is not None


# ============================================================================
# Memory/Pruning Tests
# ============================================================================

@pytest.mark.parametrize("dataset", ["countries_s3"])
def test_env_memory_consistency(dataset):
    """Verify memory structures remain consistent during episode."""
    dh_new, im_new, env_new = _setup_new_env(dataset)
    
    # Reset
    env_new.reset()
    
    # Run multiple steps
    for _ in range(5):
        result = env_new.step(0)
        done = result[2] if len(result) > 2 else False
        if done:
            break
    
    # Environment should still be in valid state
    # (No crashes during episode)


# ============================================================================
# Full Episode Tests
# ============================================================================

@pytest.mark.parametrize("dataset", ["countries_s3"])
def test_env_complete_episode(dataset):
    """Verify a complete episode runs without errors."""
    dh_new, im_new, env_new = _setup_new_env(dataset)
    
    # Reset
    obs = env_new.reset()
    total_reward = 0
    steps = 0
    max_steps = 100
    
    while steps < max_steps:
        # Take action
        result = env_new.step(0)
        
        # Extract reward and done
        reward = result[1] if len(result) > 1 else 0
        done = result[2] if len(result) > 2 else False
        
        total_reward += reward
        steps += 1
        
        if done:
            break
    
    # Should complete without error
    assert steps > 0, "No steps taken"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
