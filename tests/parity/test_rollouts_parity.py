"""
Rollout Collection Parity Tests.

Tests verifying that the tensor-based PPO's collect_rollouts produces the SAME
buffer contents (states, derived states, rewards, dones, actions, values, log_probs)
step-by-step as the SB3 PPO's collect_rollouts when using deterministic action selection.

This tests the full rollout collection pipeline, not just the environment.
Uses the new `deterministic` and `return_traces` flags in both collect_rollouts implementations.

Usage:
    pytest tests/parity/test_rollouts_parity.py -v
    python tests/parity/test_rollouts_parity.py --dataset countries_s3 --n-envs 4 --n-steps 20
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any

import pytest
import torch
import numpy as np

# Setup paths
ROOT = Path(__file__).resolve().parents[2]
SB3_ROOT = ROOT / "sb3"
TEST_ENVS_ROOT = ROOT / "test_envs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SB3_ROOT) not in sys.path:
    sys.path.insert(1, str(SB3_ROOT))
if str(TEST_ENVS_ROOT) not in sys.path:
    sys.path.insert(2, str(TEST_ENVS_ROOT))

# SB3 imports
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.buffers import RolloutBuffer as SB3RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback

# sb3 imports
from sb3_dataset import DataHandler as StrDataHandler
from sb3_index_manager import IndexManager as StrIndexManager
from sb3_env import LogicEnv_gym as StrEnv
from sb3_model import PPO_custom, CustomActorCriticPolicy, CustomCombinedExtractor
from sb3_embeddings import EmbedderLearnable as SB3Embedder

# Tensor imports
from data_handler import DataHandler
from index_manager import IndexManager
from unification import UnificationEngine
from env import BatchedEnv
from embeddings import EmbedderLearnable as TensorEmbedder
from model import ActorCriticPolicy as TensorPolicy
from ppo import PPO as TensorPPO
from rollout import RolloutBuffer as TensorRolloutBuffer


def create_aligned_environments(dataset: str, n_envs: int):
    """Create SB3 and tensor environments with aligned queries."""
    base_path = "./data/"
    device = torch.device("cpu")
    padding_atoms = 6
    padding_states = 100  # Reduced from 500 to lower memory usage
    
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
    im_sb3.build_fact_index(list(facts_set), deterministic=True)
    
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
    
    # Use ALL train queries for both
    queries = dh_sb3.train_queries
    tensor_queries = dh_tensor.train_queries
    
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
        'padding_atoms': padding_atoms,
        'padding_states': padding_states,
    }


def create_sb3_ppo(env_data: Dict, queries: List, n_envs: int, n_steps: int):
    """Create SB3 PPO with DummyVecEnv."""
    device = torch.device("cpu")
    padding_atoms = env_data.get('padding_atoms', 6)
    padding_states = env_data.get('padding_states', 100)
    
    dh = env_data['dh']
    im = env_data['im']
    facts_set = env_data['facts_set']
    
    labels = [1] * len(queries)
    depths = [None] * len(queries)
    
    def make_env(env_idx: int):
        def _init():
            env = StrEnv(
                index_manager=im,
                data_handler=dh,
                queries=queries,
                labels=labels,
                query_depths=depths,
                facts=facts_set,
                mode='train',
                sample_deterministic=True,
                seed=42,
                max_depth=20,
                memory_pruning=True,
                padding_atoms=padding_atoms,
                padding_states=padding_states,
                verbose=0,
                prover_verbose=0,
                device=device,
                engine='python',
                engine_strategy='complete',
                skip_unary_actions=True,
                endf_action=True,
                reward_type=0,
            )
            env._train_ptr = env_idx
            return Monitor(env)
        return _init
    
    env_fns = [make_env(env_idx=i) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    
    # Create embedder with fixed seed for reproducibility
    torch.manual_seed(42)
    embedder = SB3Embedder(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=im.variable_no if hasattr(im, 'variable_no') else 1000,
        max_arity=dh.max_arity,
        padding_atoms=padding_atoms,
        atom_embedder='transe',
        state_embedder='sum',
        constant_embedding_size=64,
        predicate_embedding_size=64,
        atom_embedding_size=64,
        device=device,
    )
    embedder.embed_dim = 64  # Set embed_dim
    
    # Create PPO with fixed seed for reproducibility
    torch.manual_seed(42)
    ppo = PPO_custom(
        policy=CustomActorCriticPolicy,
        env=vec_env,
        n_steps=n_steps,
        batch_size=64,
        n_epochs=1,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        device=device,
        policy_kwargs={
            "features_extractor_class": CustomCombinedExtractor,
            "features_extractor_kwargs": {
                "embedder": embedder,
                "features_dim": 64,
            },
        },
    )
    
    return ppo, vec_env, im


def create_tensor_ppo(env_data: Dict, queries: List, n_envs: int, n_steps: int):
    """Create tensor PPO with BatchedEnv."""
    device = torch.device("cpu")
    padding_atoms = env_data.get('padding_atoms', 6)
    padding_states = env_data.get('padding_states', 100)
    
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
        end_pred_idx=im.end_pred_idx,
        end_proof_action=True,
        max_derived_per_state=padding_states,
    )
    engine.index_manager = im
    
    # Convert queries to tensor format
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
        memory_pruning=True,
        eval_pruning=False,
        use_exact_memory=True,
        skip_unary_actions=True,
        end_proof_action=False,
        reward_type=0,
        padding_atoms=padding_atoms,
        padding_states=padding_states,
        true_pred_idx=im.predicate_str2idx.get('True'),
        false_pred_idx=im.predicate_str2idx.get('False'),
        end_pred_idx=im.predicate_str2idx.get('Endf'),
        verbose=0,
        prover_verbose=0,
        device=device,
        runtime_var_start_index=im.constant_no + 1,
        total_vocab_size=im.constant_no + 1000000,
        sample_deterministic_per_env=True,
    )
    
    # Create embedder with fixed seed for reproducibility (same seed as SB3)
    torch.manual_seed(42)
    embedder = TensorEmbedder(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=im.variable_no,
        max_arity=dh.max_arity,
        padding_atoms=padding_atoms,
        atom_embedder='transe',
        state_embedder='sum',
        constant_embedding_size=64,
        predicate_embedding_size=64,
        atom_embedding_size=64,
        device=str(device),
    )
    embedder.embed_dim = 64
    
    # Create policy with fixed seed for reproducibility (same seed as SB3)
    torch.manual_seed(42)
    action_size = padding_states  # action space size is padding_states
    policy = TensorPolicy(
        embedder=embedder,
        embed_dim=64,
        action_dim=action_size,
        hidden_dim=256,
        num_layers=8,
        dropout_prob=0.0,
        device=device,
    ).to(device)
    
    # Create PPO
    ppo = TensorPPO(
        policy=policy,
        env=env,
        n_steps=n_steps,
        learning_rate=3e-4,
        n_epochs=1,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=device,
        verbose=False,
    )
    
    return ppo, env, im, engine


def collect_sb3_rollout_traces(
    ppo: PPO_custom,
    n_steps: int,
) -> Tuple[List[Dict], SB3RolloutBuffer]:
    """
    Collect rollouts using SB3 PPO with trace collection.
    Uses the modified collect_rollouts with return_traces=True.
    """
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from collections import deque
    
    # Initialize required buffers
    ppo._last_obs = ppo.env.reset()
    ppo._last_episode_starts = np.ones((ppo.env.num_envs,), dtype=bool)
    ppo.ep_info_buffer = deque(maxlen=100)
    ppo.ep_success_buffer = deque(maxlen=100)
    ppo.num_timesteps = 0
    
    # Initialize logger (required by collect_rollouts)
    ppo._logger = configure(None, [""])  # Silent logger
    
    # Create dummy callback
    class DummyCallback(BaseCallback):
        def _on_step(self) -> bool:
            return True
    callback = DummyCallback()
    callback.init_callback(ppo)
    
    # Collect rollouts with traces
    success, traces = ppo.collect_rollouts(
        env=ppo.env,
        callback=callback,
        rollout_buffer=ppo.rollout_buffer,
        n_rollout_steps=n_steps,
        return_traces=True,
    )
    
    return traces, ppo.rollout_buffer


def collect_tensor_rollout_traces(
    ppo: TensorPPO,
    n_steps: int,
    im: IndexManager,
) -> Tuple[List[Dict], TensorRolloutBuffer]:
    """
    Collect rollouts using tensor PPO with trace collection.
    Uses the modified collect_rollouts with return_traces=True.
    """
    # Initialize
    current_obs = ppo.env.reset()
    episode_starts = torch.ones(ppo.n_envs, dtype=torch.float32, device=ppo.device)
    current_episode_reward = torch.zeros(ppo.n_envs, dtype=torch.float32, device=ppo.device)
    current_episode_length = torch.zeros(ppo.n_envs, dtype=torch.long, device=ppo.device)
    episode_rewards = []
    episode_lengths = []
    
    # Collect rollouts with traces
    result = ppo.collect_rollouts(
        current_obs=current_obs,
        episode_starts=episode_starts,
        current_episode_reward=current_episode_reward,
        current_episode_length=current_episode_length,
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        iteration=0,
        return_traces=True,
    )
    
    # Result now has traces as 6th element
    traces = result[5]
    
    return traces, ppo.rollout_buffer


def compare_traces(sb3_traces: List[Dict], tensor_traces: List[Dict], 
                   n_steps: int, n_envs: int, verbose: bool = True) -> Dict[str, Any]:
    """Compare traces from SB3 and tensor rollouts."""
    results = {
        'total_comparisons': 0,
        'state_matches': 0,
        'reward_matches': 0,
        'done_matches': 0,
        'action_matches': 0,
        'value_matches': 0,
        'log_prob_matches': 0,
        'mismatches': [],
    }
    
    # Create lookup dicts by (step, env)
    sb3_lookup = {(t['step'], t['env']): t for t in sb3_traces}
    tensor_lookup = {(t['step'], t['env']): t for t in tensor_traces}
    
    for step in range(n_steps):
        for env_idx in range(n_envs):
            sb3_trace = sb3_lookup.get((step, env_idx))
            tensor_trace = tensor_lookup.get((step, env_idx))
            
            if sb3_trace is None or tensor_trace is None:
                continue
            
            results['total_comparisons'] += 1
            
            # Compare state observation (sub_index)
            sb3_sub = sb3_trace['state_obs']['sub_index']
            tensor_sub = tensor_trace['state_obs']['sub_index']
            state_match = np.array_equal(np.array(sb3_sub).flatten()[:20], 
                                         np.array(tensor_sub).flatten()[:20])
            if state_match:
                results['state_matches'] += 1
            
            # Compare reward
            reward_match = abs(sb3_trace['reward'] - tensor_trace['reward']) < 1e-5
            if reward_match:
                results['reward_matches'] += 1
            
            # Compare done
            done_match = sb3_trace['done'] == tensor_trace['done']
            if done_match:
                results['done_matches'] += 1
            
            # Compare action
            action_match = sb3_trace['action'] == tensor_trace['action']
            if action_match:
                results['action_matches'] += 1
            
            # Compare value (with tolerance)
            value_match = abs(sb3_trace['value'] - tensor_trace['value']) < 0.1
            if value_match:
                results['value_matches'] += 1
            
            # Compare log_prob (with tolerance)
            log_prob_match = abs(sb3_trace['log_prob'] - tensor_trace['log_prob']) < 0.1
            if log_prob_match:
                results['log_prob_matches'] += 1
            
            # Record mismatches
            if not (state_match and reward_match and done_match and action_match):
                mismatch = {
                    'step': step,
                    'env_idx': env_idx,
                    'state_match': state_match,
                    'reward_match': reward_match,
                    'done_match': done_match,
                    'action_match': action_match,
                    'sb3_reward': sb3_trace['reward'],
                    'tensor_reward': tensor_trace['reward'],
                    'sb3_done': sb3_trace['done'],
                    'tensor_done': tensor_trace['done'],
                    'sb3_action': sb3_trace['action'],
                    'tensor_action': tensor_trace['action'],
                }
                results['mismatches'].append(mismatch)
            
            if verbose:
                status = "OK" if (state_match and reward_match and done_match and action_match) else "MISMATCH"
                if status == "MISMATCH" or step < 3:
                    print(f"Step {step}, Env {env_idx}: {status}")
                    print(f"  SB3   : action={sb3_trace['action']}, reward={sb3_trace['reward']:.3f}, done={sb3_trace['done']}")
                    print(f"  Tensor: action={tensor_trace['action']}, reward={tensor_trace['reward']:.3f}, done={tensor_trace['done']}")
    
    return results


def run_rollout_parity_test(dataset: str, n_envs: int, n_steps: int, verbose: bool = True) -> bool:
    """Run the full rollout parity test."""
    print("=" * 70)
    print(f"Rollout Collection Parity Test")
    print(f"Dataset: {dataset}, n_envs: {n_envs}, n_steps: {n_steps}")
    print("=" * 70)
    
    # Create aligned environments
    print("\nCreating aligned environments...")
    env_data = create_aligned_environments(dataset, n_envs)
    
    # Create SB3 PPO
    print("Creating SB3 PPO...")
    sb3_ppo, sb3_env, sb3_im = create_sb3_ppo(
        env_data['sb3'], env_data['queries'], n_envs, n_steps
    )
    
    # Create tensor PPO
    print("Creating tensor PPO...")
    tensor_ppo, tensor_env, tensor_im, engine = create_tensor_ppo(
        env_data['tensor'], env_data['tensor_queries'], n_envs, n_steps
    )
    
    # Collect rollouts with traces using the new flags
    # Set same seed before each collect to ensure identical sampling
    print("\nCollecting SB3 rollouts with return_traces=True...")
    torch.manual_seed(123)  # Seed before sampling
    sb3_traces, sb3_buffer = collect_sb3_rollout_traces(sb3_ppo, n_steps)
    
    print("Collecting tensor rollouts with return_traces=True...")
    torch.manual_seed(123)  # Same seed for identical sampling
    tensor_traces, tensor_buffer = collect_tensor_rollout_traces(tensor_ppo, n_steps, tensor_im)
    
    # Compare traces
    print("\n--- Trace Comparison ---")
    results = compare_traces(sb3_traces, tensor_traces, n_steps, n_envs, verbose=verbose)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Results Summary:")
    print(f"  Total comparisons: {results['total_comparisons']}")
    print(f"  State matches: {results['state_matches']}/{results['total_comparisons']}")
    print(f"  Reward matches: {results['reward_matches']}/{results['total_comparisons']}")
    print(f"  Done matches: {results['done_matches']}/{results['total_comparisons']}")
    print(f"  Action matches: {results['action_matches']}/{results['total_comparisons']}")
    
    success = (
        results['state_matches'] == results['total_comparisons'] and
        results['reward_matches'] == results['total_comparisons'] and
        results['done_matches'] == results['total_comparisons'] and
        results['action_matches'] == results['total_comparisons']
    )
    
    if success:
        print("\n✓ ALL ROLLOUT TRACES MATCH")
    else:
        print(f"\n✗ MISMATCHES FOUND: {len(results['mismatches'])}")
        if results['mismatches'] and verbose:
            print("\nFirst 5 mismatches:")
            for m in results['mismatches'][:5]:
                print(f"  Step {m['step']}, Env {m['env_idx']}:")
                print(f"    SB3: action={m['sb3_action']}, reward={m['sb3_reward']}, done={m['sb3_done']}")
                print(f"    Tensor: action={m['tensor_action']}, reward={m['tensor_reward']}, done={m['tensor_done']}")
    
    print("=" * 70)
    
    return success


# ============================================================
# Pytest Tests
# ============================================================

@pytest.mark.parametrize("n_envs,n_steps", [
    (1, 10),
    (4, 10),
])
def test_rollout_parity(n_envs, n_steps):
    """Test rollout collection parity between SB3 and tensor implementations."""
    success = run_rollout_parity_test(
        dataset="countries_s3",
        n_envs=n_envs,
        n_steps=n_steps,
        verbose=False
    )
    assert success, f"Rollout parity test failed for n_envs={n_envs}, n_steps={n_steps}"


# ============================================================
# CLI Interface
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Rollout Collection Parity Test")
    parser.add_argument("--dataset", type=str, default="countries_s3",
                       help="Dataset name (default: countries_s3)")
    parser.add_argument("--n-envs", type=int, default=4,
                       help="Number of environments (default: 4)")
    parser.add_argument("--n-steps", type=int, default=20,
                       help="Number of rollout steps (default: 20)")
    parser.add_argument("--verbose", action="store_true", default=False,
                       help="Enable verbose output (show each step comparison)")
    
    args = parser.parse_args()
    
    success = run_rollout_parity_test(
        dataset=args.dataset,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        verbose=args.verbose
    )
    
    sys.exit(0 if success else 1)
