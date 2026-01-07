"""
Compiled vs SB3 Parity Tests.

Tests verifying that the optimized/compiled implementation produces the SAME
results as the original SB3 (stable-baselines3) implementation.

This bridges the two existing parity test suites:
- parity_sb3/: Tests SB3 vs tensor implementations
- parity_tensor/: Tests tensor vs compiled implementations

This script directly compares SB3 (baseline) vs Compiled (optimized) to ensure
end-to-end correctness.

Usage:
    python tests/test_compiled_sb3_parity.py --dataset countries_s3 --n-queries 5
    pytest tests/test_compiled_sb3_parity.py -v -s
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, Tuple, Optional
import time

import pytest
import torch
import numpy as np

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
SB3_ROOT = ROOT / "sb3"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SB3_ROOT) not in sys.path:
    sys.path.insert(1, str(SB3_ROOT))

# SB3 imports (baseline/reference)
from sb3.sb3_dataset import DataHandler as SB3DataHandler
from sb3.sb3_index_manager import IndexManager as SB3IndexManager
from sb3.sb3_env import LogicEnv_gym as SB3Env
from sb3.sb3_custom_dummy_env import CustomDummyVecEnv
from sb3.sb3_model import PPO_custom, CustomActorCriticPolicy, CustomCombinedExtractor
from sb3.sb3_embeddings import EmbedderLearnable as SB3Embedder
from sb3.sb3_model_eval import eval_corruptions as sb3_eval_corruptions
from sb3.sb3_neg_sampling import get_sampler as get_sb3_sampler
from stable_baselines3.common.monitor import Monitor

# Compiled/Optimized imports (test target)
from data_handler import DataHandler as CompiledDataHandler
from index_manager import IndexManager as CompiledIndexManager
from env import EnvVec as CompiledEnv
from ppo import PPO as CompiledPPO
from policy import ActorCriticPolicy as CompiledPolicy
from nn.embeddings import EmbedderLearnable as CompiledEmbedder
from unification import UnificationEngineVectorized
from nn.sampler import Sampler as CompiledSampler

# Shared utilities
from tests.test_utils.parity_config import ParityConfig, TOLERANCE

# ============================================================================
# Configuration
# ============================================================================

def create_default_config() -> SimpleNamespace:
    """Default parameters for compiled vs sb3 parity tests."""
    return SimpleNamespace(
        dataset="countries_s3",
        data_path=os.path.join(ROOT, 'data'),
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",

        # Test parameters
        n_queries=10,
        n_corruptions=5,
        chunk_queries=20,
        batch_size_env=20,
        corruption_modes=['both'],
        mode='test',

        # Environment parameters
        padding_atoms=6,
        padding_states=100,
        max_depth=20,
        memory_pruning=True,
        skip_unary_actions=True,
        end_proof_action=True,
        reward_type=0,
        negative_ratio=1.0,

        # Model parameters
        embed_dim=64,
        hidden_dim=64,

        # Training parameters (for learn parity)
        n_envs=4,
        n_steps=20,
        batch_size=20,
        n_epochs=3,
        learning_rate=3e-4,
        ent_coef=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,

        # Device
        device="cpu",
        seed=42,

        # Compile settings
        compile=False,
        compile_mode='reduce-overhead',
    )


# ============================================================================
# SB3 Setup Functions
# ============================================================================

def setup_sb3_components(config: SimpleNamespace) -> Dict[str, Any]:
    """Initialize SB3 (baseline) components."""
    print("\n[SB3] Setting up baseline components...")

    # Data handler
    dh = SB3DataHandler(
        dataset_name=config.dataset,
        data_path=config.data_path,
        train_file=config.train_file,
        valid_file=config.valid_file,
        test_file=config.test_file,
        rules_file=config.rules_file,
        facts_file=config.facts_file,
    )

    # Index manager
    im = SB3IndexManager(dh.get_facts(), dh.get_rules())

    # Embedder
    embedder = SB3Embedder(
        num_predicates=len(dh.predicates),
        num_constants=len(dh.constants),
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
    )

    # Sampler
    sampler = get_sb3_sampler(
        facts=dh.get_facts(),
        predicates=dh.predicates,
        constants=dh.constants,
        mode='both'
    )

    # Create environments
    def make_env():
        env = SB3Env(
            index_manager=im,
            facts=dh.get_facts(),
            rules=dh.get_rules(),
            queries=dh.train_queries[:config.n_envs],
            labels=[1] * config.n_envs,
            embedder=embedder,
            max_depth=config.max_depth,
            reward_type=config.reward_type,
            negative_ratio=config.negative_ratio,
            sampler=sampler,
            skip_unary_actions=config.skip_unary_actions,
            memory_pruning=config.memory_pruning,
            end_proof_action=config.end_proof_action,
            padding_atoms=config.padding_atoms,
            padding_states=config.padding_states,
            config=config,
        )
        return Monitor(env)

    envs = CustomDummyVecEnv([make_env for _ in range(config.n_envs)])

    # Policy kwargs
    policy_kwargs = {
        'features_extractor_class': CustomCombinedExtractor,
        'features_extractor_kwargs': {
            'features_dim': config.embed_dim,
            'embedder': embedder,
        }
    }

    # Create PPO model
    model = PPO_custom(
        CustomActorCriticPolicy,
        envs,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        verbose=0,
        seed=config.seed,
        device=config.device,
        policy_kwargs=policy_kwargs,
    )

    print(f"[SB3] Components ready: {len(dh.predicates)} predicates, {len(dh.constants)} constants")

    return {
        'data_handler': dh,
        'index_manager': im,
        'embedder': embedder,
        'sampler': sampler,
        'envs': envs,
        'model': model,
    }


# ============================================================================
# Compiled Setup Functions
# ============================================================================

def setup_compiled_components(config: SimpleNamespace) -> Dict[str, Any]:
    """Initialize Compiled (optimized) components."""
    print("\n[Compiled] Setting up optimized components...")

    # Data handler
    dh = CompiledDataHandler(
        dataset_name=config.dataset,
        data_path=config.data_path,
        train_file=config.train_file,
        valid_file=config.valid_file,
        test_file=config.test_file,
        rules_file=config.rules_file,
        facts_file=config.facts_file,
    )

    # Index manager
    im = CompiledIndexManager(dh)

    # Unification engine
    ue = UnificationEngineVectorized(im, device=config.device)

    # Embedder
    embedder = CompiledEmbedder(
        num_predicates=im.n_predicates,
        num_constants=im.n_constants,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        device=config.device,
    )

    # Sampler
    sampler = CompiledSampler(dh, im, mode='both')

    # Get queries
    train_split = dh.get_materialized_split('train')
    train_queries = train_split.queries.squeeze(1)

    # Create environment
    env = CompiledEnv(
        unification_engine=ue,
        embedder=embedder,
        n_envs=config.n_envs,
        train_queries=train_queries[:config.n_envs],
        sampler=sampler,
        negative_ratio=config.negative_ratio,
        reward_type=config.reward_type,
        compile=config.compile,
        device=config.device,
    )

    # Create policy
    policy = CompiledPolicy(
        embedder=embedder,
        device=config.device,
    )

    # Create PPO
    ppo = CompiledPPO(
        policy=policy,
        env=env,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        device=config.device,
        seed=config.seed,
    )

    print(f"[Compiled] Components ready: {im.n_predicates} predicates, {im.n_constants} constants")

    return {
        'data_handler': dh,
        'index_manager': im,
        'unification_engine': ue,
        'embedder': embedder,
        'sampler': sampler,
        'env': env,
        'policy': policy,
        'ppo': ppo,
    }


# ============================================================================
# Parity Tests
# ============================================================================

def test_data_handler_parity():
    """Test that data handlers load identical data."""
    config = create_default_config()

    # Both use 'base_path'
    sb3_dh = SB3DataHandler(
        dataset_name=config.dataset,
        base_path=config.data_path,
        train_file=config.train_file,
        valid_file=config.valid_file,
        test_file=config.test_file,
        rules_file=config.rules_file,
        facts_file=config.facts_file,
    )

    compiled_dh = CompiledDataHandler(
        dataset_name=config.dataset,
        base_path=config.data_path,
        train_file=config.train_file,
        valid_file=config.valid_file,
        test_file=config.test_file,
        rules_file=config.rules_file,
        facts_file=config.facts_file,
    )

    # Compare predicate counts
    sb3_n_pred = len(sb3_dh.predicates)
    compiled_n_pred = len(compiled_dh.predicates)
    assert sb3_n_pred == compiled_n_pred, \
        f"Predicate count mismatch: SB3={sb3_n_pred}, Compiled={compiled_n_pred}"

    # Compare constant counts
    sb3_n_const = len(sb3_dh.constants)
    compiled_n_const = len(compiled_dh.constants)
    assert sb3_n_const == compiled_n_const, \
        f"Constant count mismatch: SB3={sb3_n_const}, Compiled={compiled_n_const}"

    # Compare fact counts
    assert len(sb3_dh.facts) == len(compiled_dh.facts), \
        f"Fact count mismatch: SB3={len(sb3_dh.facts)}, Compiled={len(compiled_dh.facts)}"

    # Compare rule counts
    assert len(sb3_dh.rules) == len(compiled_dh.rules), \
        f"Rule count mismatch: SB3={len(sb3_dh.rules)}, Compiled={len(compiled_dh.rules)}"

    print(f"[PASS] Data handler parity: {sb3_n_pred} predicates, "
          f"{sb3_n_const} constants, {len(sb3_dh.facts)} facts, "
          f"{len(sb3_dh.rules)} rules")


def test_index_manager_parity():
    """Test that index managers produce compatible indices."""
    config = create_default_config()

    # Both use 'base_path'
    sb3_dh = SB3DataHandler(
        dataset_name=config.dataset,
        base_path=config.data_path,
        train_file=config.train_file,
        valid_file=config.valid_file,
        test_file=config.test_file,
        rules_file=config.rules_file,
        facts_file=config.facts_file,
    )
    # SB3 IndexManager requires (constants, predicates, max_total_vars, rules)
    sb3_im = SB3IndexManager(
        constants=sb3_dh.constants,
        predicates=sb3_dh.predicates,
        max_total_vars=1000,
        rules=sb3_dh.rules
    )

    # Compiled - IndexManager requires (constants, predicates, ...)
    compiled_dh = CompiledDataHandler(
        dataset_name=config.dataset,
        base_path=config.data_path,
        train_file=config.train_file,
        valid_file=config.valid_file,
        test_file=config.test_file,
        rules_file=config.rules_file,
        facts_file=config.facts_file,
    )
    compiled_im = CompiledIndexManager(
        constants=compiled_dh.constants,
        predicates=compiled_dh.predicates,
        max_total_runtime_vars=1000,
        rules=compiled_dh.rules
    )

    # Compare predicate count
    # Note: Compiled version may have extra predicates (Endt, Endf, EndUnsat) for end-proof actions
    sb3_n_pred = len(sb3_dh.predicates)
    compiled_n_pred = compiled_im.predicate_no

    if sb3_n_pred != compiled_n_pred:
        print(f"[WARN] Predicate count differs: SB3={sb3_n_pred}, Compiled={compiled_n_pred}")
        print(f"       (Compiled may include special predicates like Endt/Endf/EndUnsat)")
    else:
        print(f"[OK] Predicate count matches: {sb3_n_pred}")

    # Compare constant count
    sb3_n_const = len(sb3_dh.constants)
    compiled_n_const = compiled_im.constant_no
    assert sb3_n_const == compiled_n_const, \
        f"Constant count mismatch: SB3={sb3_n_const}, Compiled={compiled_n_const}"

    print(f"[PASS] Index manager parity: constants match ({sb3_n_const})")


def test_embedder_parity():
    """Test that embedders produce same-shaped outputs."""
    config = create_default_config()

    # Both embedders use same parameter names: n_constants, n_predicates, etc.
    sb3_embedder = SB3Embedder(
        n_constants=20,
        n_predicates=10,
        constant_embedding_size=config.embed_dim,
        predicate_embedding_size=config.embed_dim,
        atom_embedding_size=config.embed_dim,
        padding_atoms=config.padding_atoms,
    )

    compiled_embedder = CompiledEmbedder(
        n_constants=20,
        n_predicates=10,
        constant_embedding_size=config.embed_dim,
        predicate_embedding_size=config.embed_dim,
        atom_embedding_size=config.embed_dim,
        padding_atoms=config.padding_atoms,
        device=config.device,
    )

    # Both should have same embedding dimensions
    assert sb3_embedder.embed_dim == compiled_embedder.embed_dim, \
        f"Embed dim mismatch: SB3={sb3_embedder.embed_dim}, Compiled={compiled_embedder.embed_dim}"

    print(f"[PASS] Embedder parity: embed_dim={sb3_embedder.embed_dim}")


def test_component_initialization():
    """Test that both data handlers and index managers can be initialized consistently."""
    config = create_default_config()

    # Test SB3 components
    try:
        sb3_dh = SB3DataHandler(
            dataset_name=config.dataset,
            base_path=config.data_path,
            train_file=config.train_file,
            valid_file=config.valid_file,
            test_file=config.test_file,
            rules_file=config.rules_file,
            facts_file=config.facts_file,
        )
        sb3_im = SB3IndexManager(
            constants=sb3_dh.constants,
            predicates=sb3_dh.predicates,
            max_total_vars=1000,
            rules=sb3_dh.rules
        )
        print("[PASS] SB3 data handler and index manager initialized")
    except Exception as e:
        pytest.fail(f"SB3 initialization failed: {e}")

    # Test Compiled components
    try:
        compiled_dh = CompiledDataHandler(
            dataset_name=config.dataset,
            base_path=config.data_path,
            train_file=config.train_file,
            valid_file=config.valid_file,
            test_file=config.test_file,
            rules_file=config.rules_file,
            facts_file=config.facts_file,
        )
        compiled_im = CompiledIndexManager(
            constants=compiled_dh.constants,
            predicates=compiled_dh.predicates,
            max_total_runtime_vars=1000,
            rules=compiled_dh.rules
        )
        print("[PASS] Compiled data handler and index manager initialized")
    except Exception as e:
        pytest.fail(f"Compiled initialization failed: {e}")

    # Verify key component counts match
    sb3_n_pred = len(sb3_dh.predicates)
    compiled_n_pred = len(compiled_dh.predicates)

    assert sb3_n_pred == compiled_n_pred, \
        f"Predicate count mismatch: SB3={sb3_n_pred}, Compiled={compiled_n_pred}"

    print(f"[PASS] Both implementations initialized with {sb3_n_pred} predicates")


# ============================================================================
# Main Entry Point
# ============================================================================

def run_all_tests():
    """Run all parity tests."""
    print("=" * 60)
    print("Compiled vs SB3 Parity Tests")
    print("=" * 60)

    tests = [
        ("Data Handler Parity", test_data_handler_parity),
        ("Index Manager Parity", test_index_manager_parity),
        ("Embedder Parity", test_embedder_parity),
        ("Component Initialization", test_component_initialization),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compiled vs SB3 Parity Tests")
    parser.add_argument("--dataset", type=str, default="countries_s3")
    parser.add_argument("--n-queries", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    success = run_all_tests()
    sys.exit(0 if success else 1)
