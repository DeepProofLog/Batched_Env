"""
Evaluation Parity Tests.

Tests verifying that the tensor-based evaluation functions produce the SAME
results as the SB3 evaluation functions, including:
- evaluate_policy: Rollout-based policy evaluation
- eval_corruptions: Link prediction evaluation with corruption sampling

Uses deterministic action selection and aligned environments to ensure
reproducible comparisons.

Usage:
    pytest tests/parity/test_eval_parity.py -v
    python tests/parity/test_eval_parity.py --dataset countries_s3 --n-eval-episodes 10
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field

import pytest
import torch
import torch.nn as nn
import numpy as np
from collections import deque

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
from stable_baselines3.common.logger import configure

# sb3 imports
from sb3_custom_dummy_env import CustomDummyVecEnv
from sb3_dataset import DataHandler as StrDataHandler
from sb3_index_manager import IndexManager as StrIndexManager
from sb3_env import LogicEnv_gym as StrEnv
from sb3_model import PPO_custom, CustomActorCriticPolicy, CustomCombinedExtractor
from sb3_embeddings import EmbedderLearnable as SB3Embedder
from sb3_model_eval import evaluate_policy as sb3_evaluate_policy
from sb3_model_eval import eval_corruptions as sb3_eval_corruptions
from sb3_neg_sampling import BasicNegativeSamplerDomain, get_sampler as get_sb3_sampler

# Tensor imports
from data_handler import DataHandler
from index_manager import IndexManager
from unification import UnificationEngine
from env import BatchedEnv
from embeddings import EmbedderLearnable as TensorEmbedder
from model import ActorCriticPolicy as TensorPolicy
from ppo import PPO as TensorPPO
from model_eval import evaluate_policy as tensor_evaluate_policy
from model_eval import eval_corruptions as tensor_eval_corruptions
from sampler import Sampler, SamplerConfig


@dataclass
class EvalParityResults:
    """Results container for evaluation parity comparison."""
    # evaluate_policy comparison
    rewards_match: bool = False
    lengths_match: bool = False
    success_match: bool = False
    logps_match: bool = False
    
    # eval_corruptions comparison  
    mrr_match: bool = False
    hits1_match: bool = False
    hits3_match: bool = False
    hits10_match: bool = False
    
    # Raw values
    sb3_rewards: Optional[np.ndarray] = None
    tensor_rewards: Optional[np.ndarray] = None
    sb3_lengths: Optional[np.ndarray] = None
    tensor_lengths: Optional[np.ndarray] = None
    sb3_logps: Optional[np.ndarray] = None
    tensor_logps: Optional[np.ndarray] = None
    
    sb3_mrr: float = 0.0
    tensor_mrr: float = 0.0
    sb3_hits1: float = 0.0
    tensor_hits1: float = 0.0
    sb3_hits3: float = 0.0
    tensor_hits3: float = 0.0
    sb3_hits10: float = 0.0
    tensor_hits10: float = 0.0
    
    # Overall
    evaluate_policy_success: bool = False
    eval_corruptions_success: bool = False


def create_aligned_environments(dataset: str, n_envs: int, mode: str = 'valid'):
    """
    Create SB3 and tensor environments with aligned queries.
    
    Args:
        dataset: Dataset name
        n_envs: Number of parallel environments
        mode: 'valid' or 'test' for query selection
    
    Returns:
        Dict with environment data for both implementations
    """
    base_path = "./data/"
    device = torch.device("cpu")
    padding_atoms = 6
    padding_states = 100
    
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
        corruption_mode=True,  # Enable domain loading for negative sampling
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
    
    # Select queries based on mode
    if mode == 'valid':
        queries_sb3 = dh_sb3.valid_queries
        queries_tensor = dh_tensor.valid_queries
    elif mode == 'test':
        queries_sb3 = dh_sb3.test_queries
        queries_tensor = dh_tensor.test_queries
    else:
        queries_sb3 = dh_sb3.train_queries
        queries_tensor = dh_tensor.train_queries
    
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
        'queries_sb3': queries_sb3,
        'queries_tensor': queries_tensor,
        'padding_atoms': padding_atoms,
        'padding_states': padding_states,
    }


def create_sb3_eval_env(env_data: Dict, queries: List, n_envs: int, seed: int = 42):
    """Create SB3 environment and PPO for evaluation."""
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
                mode='eval',  # Use eval mode
                sample_deterministic=True,
                seed=seed,
                max_depth=20,
                memory_pruning=False,
                padding_atoms=padding_atoms,
                padding_states=padding_states,
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
            env._train_ptr = env_idx
            return Monitor(env)
        return _init
    
    env_fns = [make_env(env_idx=i) for i in range(n_envs)]
    vec_env = CustomDummyVecEnv(env_fns)
    
    # CustomDummyVecEnv already has the episode tracking attributes built-in
    # Just reset them for this evaluation run
    vec_env._episode_target = np.zeros(n_envs, dtype=int)
    vec_env._episode_count = np.zeros(n_envs, dtype=int)
    vec_env.active_envs = np.ones(n_envs, dtype=bool)
    vec_env._episodes_done = 0
    
    # Create embedder with fixed seed
    n_vars_for_embedder = 1000000
    torch.manual_seed(seed)
    embedder = SB3Embedder(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=n_vars_for_embedder,
        max_arity=dh.max_arity,
        padding_atoms=padding_atoms,
        atom_embedder='transe',
        state_embedder='sum',
        constant_embedding_size=64,
        predicate_embedding_size=64,
        atom_embedding_size=64,
        device=device,
    )
    embedder.embed_dim = 64
    
    # Create PPO with fixed seed
    torch.manual_seed(seed)
    ppo = PPO_custom(
        policy=CustomActorCriticPolicy,
        env=vec_env,
        n_steps=20,
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
        seed=seed,
        policy_kwargs={
            "features_extractor_class": CustomCombinedExtractor,
            "features_extractor_kwargs": {
                "embedder": embedder,
                "features_dim": 64,
            },
        },
    )
    
    return ppo, vec_env, im


def create_tensor_eval_env(env_data: Dict, queries: List, n_envs: int, seed: int = 42, n_eval_episodes: int = None):
    """Create tensor environment and PPO for evaluation.
    
    Args:
        env_data: Environment data dictionary
        queries: List of query objects
        n_envs: Number of parallel environments
        seed: Random seed
        n_eval_episodes: Total number of episodes to evaluate (if None, uses len(queries))
    """
    device = torch.device("cpu")
    padding_atoms = env_data.get('padding_atoms', 6)
    padding_states = env_data.get('padding_states', 100)
    
    # Use n_eval_episodes if provided, otherwise default to number of queries
    if n_eval_episodes is None:
        n_eval_episodes = len(queries)
    
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
    
    # Convert queries to tensor format
    query_tensors = []
    for q in queries:
        query_atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
        query_padded = torch.full((padding_atoms, 3), im.padding_idx, dtype=torch.long, device=device)
        query_padded[0] = query_atom
        query_tensors.append(query_padded)
    
    queries_tensor = torch.stack(query_tensors, dim=0)
    
    # Create environment in train mode initially (will switch to eval via set_eval_dataset)
    env = BatchedEnv(
        batch_size=n_envs,
        queries=queries_tensor,
        labels=torch.ones(len(queries), dtype=torch.long, device=device),
        query_depths=torch.ones(len(queries), dtype=torch.long, device=device),
        unification_engine=engine,
        mode='train',  # Start in train mode
        max_depth=20,
        memory_pruning=False,
        eval_pruning=False,
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
        sample_deterministic_per_env=True,
    )
    
    # Set up the evaluation dataset with per_slot_lengths
    # SB3's DummyVecEnv gives each sub-env the SAME query list.
    # Each sub-env has its own _train_ptr which starts at 0.
    # So all envs get queries in the same order: 0, 1, 2, ... (mod n_queries)
    
    # Compute target episodes per env using n_eval_episodes (total episodes to run)
    # This matches SB3's target distribution: [(n_eval_episodes + i) // n_envs for i in range(n_envs)]
    n_queries = len(queries)
    targets = [(n_eval_episodes + i) // n_envs for i in range(n_envs)]
    total_episodes = sum(targets)
    
    # Reorder queries to match SB3's pattern:
    # Each env independently iterates through queries starting from index 0.
    # So env0 gets queries [0, 1, 2, ...] and env1 also gets [0, 1, 2, ...]
    reordered_queries = []
    
    for env_idx in range(n_envs):
        # Each env starts at ptr=0 and increments through the query list
        ptr = 0  # All envs start at 0, just like SB3's _train_ptr initialization
        for i in range(targets[env_idx]):
            query_idx = ptr % len(query_tensors)
            reordered_queries.append(query_tensors[query_idx])
            ptr += 1  # Each reset increments the pointer
    
    per_slot_lengths = torch.tensor(targets, dtype=torch.long, device=device)
    reordered_queries_tensor = torch.stack(reordered_queries, dim=0)
    
    env.set_eval_dataset(
        queries=reordered_queries_tensor,
        labels=torch.ones(len(reordered_queries), dtype=torch.long, device=device),
        query_depths=torch.ones(len(reordered_queries), dtype=torch.long, device=device),
        per_slot_lengths=per_slot_lengths,
    )
    
    # Create embedder with fixed seed
    n_vars_for_embedder = 1000000
    torch.manual_seed(seed)
    embedder = TensorEmbedder(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=n_vars_for_embedder,
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
    
    # Create policy with fixed seed
    action_size = padding_states
    torch.manual_seed(seed)
    policy = TensorPolicy(
        embedder=embedder,
        embed_dim=64,
        action_dim=action_size,
        hidden_dim=256,
        num_layers=8,
        dropout_prob=0.0,
        device=device,
    ).to(device)
    
    # Create PPO with fixed seed
    torch.manual_seed(seed)
    ppo = TensorPPO(
        policy=policy,
        env=env,
        n_steps=20,
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
        sb3_determinism=True,
    )
    
    return ppo, env, im, engine


def create_sb3_sampler(dh, im, device, seed: int = 42, corruption_scheme=['tail']):
    """Create SB3 negative sampler using the official get_sampler function."""
    return get_sb3_sampler(
        data_handler=dh,
        index_manager=im,
        corruption_scheme=corruption_scheme,
        device=device,
        corruption_mode=True,
    )


def create_tensor_sampler(dh, im, device, seed: int = 42):
    """Create tensor negative sampler with domain constraints matching SB3."""
    import os
    
    # Build domain constraints from domain file (same as SB3)
    domain_heads = {}
    domain_tails = {}
    
    # Load domain file directly (like SB3 does)
    # Construct path from dataset name
    domain_file = os.path.join("./data/", dh.dataset_name, "domain2constants.txt")
    if os.path.exists(domain_file):
        pred2domains = {}  # predicate -> (head_domain, tail_domain)
        domain2entities = {}  # domain_name -> list of entities
        
        with open(domain_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if ':' in line:
                    # Format: predicate:head_domain:tail_domain
                    parts = line.split(':')
                    if len(parts) == 3:
                        pred, head_dom, tail_dom = parts
                        pred2domains[pred] = (head_dom, tail_dom)
                else:
                    # Format: domain_name entity1 entity2 ...
                    parts = line.split()
                    if len(parts) >= 2:
                        domain_name = parts[0]
                        entities = parts[1:]
                        domain2entities[domain_name] = entities
        
        # Build domain_heads and domain_tails per predicate
        for pred_str, (head_dom, tail_dom) in pred2domains.items():
            if pred_str not in im.predicate_str2idx:
                continue
            rel_idx = im.predicate_str2idx[pred_str]
            
            # Head domain
            if head_dom in domain2entities:
                head_ents = domain2entities[head_dom]
                head_indices = torch.tensor(
                    [im.constant_str2idx[e] for e in head_ents if e in im.constant_str2idx],
                    dtype=torch.long
                )
                if head_indices.numel() > 0:
                    domain_heads[rel_idx] = head_indices
            
            # Tail domain
            if tail_dom in domain2entities:
                tail_ents = domain2entities[tail_dom]
                tail_indices = torch.tensor(
                    [im.constant_str2idx[e] for e in tail_ents if e in im.constant_str2idx],
                    dtype=torch.long
                )
                if tail_indices.numel() > 0:
                    domain_tails[rel_idx] = tail_indices
    
    # Build known triples for filtering
    all_triples = []
    for q in dh.train_queries + dh.valid_queries + dh.test_queries:
        pred_idx = im.predicate_str2idx[q.predicate]
        head_idx = im.constant_str2idx[q.args[0]]
        tail_idx = im.constant_str2idx[q.args[1]]
        all_triples.append([pred_idx, head_idx, tail_idx])
    
    all_triples_tensor = torch.tensor(all_triples, dtype=torch.long, device=device)
    
    sampler = Sampler.from_data(
        all_known_triples_idx=all_triples_tensor,
        num_entities=im.constant_no,
        num_relations=im.predicate_no,
        device=device,
        default_mode='both',
        seed=seed,
        domain_heads=domain_heads if domain_heads else None,
        domain_tails=domain_tails if domain_tails else None,
    )
    
    return sampler


def run_evaluate_policy_parity(
    dataset: str,
    n_envs: int,
    n_eval_episodes: int,
    seed: int = 42,
    verbose: bool = True,
    mode: str = 'valid',
) -> EvalParityResults:
    """
    Run evaluate_policy parity test comparing SB3 and tensor implementations.
    
    Args:
        dataset: Dataset name
        n_envs: Number of parallel environments
        n_eval_episodes: Number of episodes to evaluate
        seed: Random seed
        verbose: Whether to print detailed output
        mode: Query set to use ('train', 'valid', 'test')
    """
    results = EvalParityResults()
    
    if verbose:
        print("=" * 70)
        print(f"Evaluate Policy Parity Test")
        print(f"Dataset: {dataset}, n_envs: {n_envs}, n_episodes: {n_eval_episodes}")
        print("=" * 70)
    
    # Create aligned environments
    if verbose:
        print("\nCreating aligned environments...")
    env_data = create_aligned_environments(dataset, n_envs, mode=mode)
    
    # Limit queries to n_eval_episodes
    queries_sb3 = env_data['queries_sb3'][:n_eval_episodes]
    queries_tensor = env_data['queries_tensor'][:n_eval_episodes]
    
    if verbose:
        print(f"Using {len(queries_sb3)} queries for evaluate_policy parity")
        for i, (q_sb3, q_tensor) in enumerate(zip(queries_sb3, queries_tensor)):
            print(f"  Query {i}: SB3={q_sb3.predicate}({q_sb3.args[0]}, {q_sb3.args[1]}), Tensor={q_tensor.predicate}({q_tensor.args[0]}, {q_tensor.args[1]})")
    
    # Create SB3 environment and model
    if verbose:
        print("Creating SB3 eval environment...")
    sb3_ppo, sb3_env, sb3_im = create_sb3_eval_env(
        env_data['sb3'], queries_sb3, n_envs, seed
    )
    
    # Create tensor environment and model
    if verbose:
        print("Creating tensor eval environment...")
    tensor_ppo, tensor_env, tensor_im, engine = create_tensor_eval_env(
        env_data['tensor'], queries_tensor, n_envs, seed, n_eval_episodes=n_eval_episodes
    )
    
    # Run SB3 evaluate_policy
    if verbose:
        print("\nRunning SB3 evaluate_policy...")
    
    try:
        result = sb3_evaluate_policy(
            model=sb3_ppo,
            env=sb3_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            track_logprobs=True,
        )
        # When track_logprobs=True, returns 9 values
        sb3_rewards, sb3_lengths, sb3_logps, sb3_mask, sb3_success = result[:5]
        results.sb3_rewards = sb3_rewards
        results.sb3_lengths = sb3_lengths
        results.sb3_logps = sb3_logps
        
        if verbose:
            print(f"  SB3 rewards shape: {sb3_rewards.shape}")
            print(f"  SB3 lengths shape: {sb3_lengths.shape}")
            print(f"  SB3 rewards: {sb3_rewards}")
            print(f"  SB3 lengths: {sb3_lengths}")
            print(f"  SB3 avg reward: {np.mean(sb3_rewards[sb3_mask]):.4f}")
            print(f"  SB3 avg length: {np.mean(sb3_lengths[sb3_mask]):.4f}")
    except Exception as e:
        if verbose:
            print(f"  SB3 evaluate_policy error: {e}")
        import traceback
        traceback.print_exc()
        return results
    
    # Run tensor evaluate_policy
    if verbose:
        print("\nRunning tensor evaluate_policy...")
    
    try:
        # Compute target episodes per env to match SB3 distribution:
        # targets = [(n_eval_episodes + i) // n_envs for i in range(n_envs)]
        targets = [(n_eval_episodes + i) // n_envs for i in range(n_envs)]
        
        tensor_results = tensor_evaluate_policy(
            actor=tensor_ppo.policy,
            env=tensor_env,
            target_episodes=targets,  # Use per-env targets like SB3
            deterministic=True,
            track_logprobs=True,
        )
        
        tensor_rewards = tensor_results['rewards'].numpy()
        tensor_lengths = tensor_results['lengths'].numpy()
        tensor_logps = tensor_results.get('logps', torch.zeros_like(tensor_results['rewards'])).numpy()
        tensor_mask = tensor_results['mask'].numpy()
        tensor_success = tensor_results['success'].numpy()
        
        results.tensor_rewards = tensor_rewards
        results.tensor_lengths = tensor_lengths
        results.tensor_logps = tensor_logps
        
        if verbose:
            print(f"  Tensor rewards shape: {tensor_rewards.shape}")
            print(f"  Tensor lengths shape: {tensor_lengths.shape}")
            print(f"  Tensor rewards: {tensor_rewards}")
            print(f"  Tensor lengths: {tensor_lengths}")
            print(f"  Tensor avg reward: {np.mean(tensor_rewards[tensor_mask]):.4f}")
            print(f"  Tensor avg length: {np.mean(tensor_lengths[tensor_mask]):.4f}")
    except Exception as e:
        if verbose:
            print(f"  Tensor evaluate_policy error: {e}")
        import traceback
        traceback.print_exc()
        return results
    
    # Compare results - focus on aggregate metrics since episode ordering may differ
    # due to different reset semantics (SB3 VecEnv auto-reset vs TorchRL step_and_maybe_reset)
    if verbose:
        print("\n--- Results Comparison ---")
    
    # Handle different output shapes by flattening valid entries
    sb3_valid_rewards = sb3_rewards[sb3_mask]
    tensor_valid_rewards = tensor_rewards[tensor_mask]
    
    sb3_valid_lengths = sb3_lengths[sb3_mask]
    tensor_valid_lengths = tensor_lengths[tensor_mask]
    
    # Compare number of completed episodes
    n_sb3_episodes = len(sb3_valid_rewards)
    n_tensor_episodes = len(tensor_valid_rewards)
    
    if verbose:
        print(f"  SB3 completed episodes: {n_sb3_episodes}")
        print(f"  Tensor completed episodes: {n_tensor_episodes}")
    
    # Compare element-wise without sorting (strict parity check)
    if n_sb3_episodes == n_tensor_episodes and n_sb3_episodes > 0:
        # Element-wise reward comparison (no sorting)
        rewards_match_elementwise = np.allclose(sb3_valid_rewards, tensor_valid_rewards, atol=1e-4)
        
        # Element-wise length comparison (no sorting)
        lengths_match_elementwise = np.array_equal(sb3_valid_lengths, tensor_valid_lengths)
        
        results.rewards_match = rewards_match_elementwise
        results.lengths_match = lengths_match_elementwise
        
        if verbose:
            print(f"  Rewards match (element-wise, no sorting): {rewards_match_elementwise}")
            print(f"  Lengths match (element-wise, no sorting): {lengths_match_elementwise}")
            
            if not rewards_match_elementwise:
                print(f"    SB3 rewards: {sb3_valid_rewards}")
                print(f"    Tensor rewards: {tensor_valid_rewards}")
                diff_mask = ~np.isclose(sb3_valid_rewards, tensor_valid_rewards, atol=1e-4)
                diff_indices = np.where(diff_mask)[0]
                print(f"    Mismatched indices: {diff_indices}")
                for idx in diff_indices[:10]:  # Show first 10 mismatches
                    print(f"      Episode {idx}: SB3={sb3_valid_rewards[idx]}, Tensor={tensor_valid_rewards[idx]}")
            
            if not lengths_match_elementwise:
                print(f"    SB3 lengths: {sb3_valid_lengths}")
                print(f"    Tensor lengths: {tensor_valid_lengths}")
                diff_mask = sb3_valid_lengths != tensor_valid_lengths
                diff_indices = np.where(diff_mask)[0]
                print(f"    Mismatched indices: {diff_indices}")
                for idx in diff_indices[:10]:  # Show first 10 mismatches
                    print(f"      Episode {idx}: SB3={sb3_valid_lengths[idx]}, Tensor={tensor_valid_lengths[idx]}")
    else:
        results.rewards_match = False
        results.lengths_match = False
        if verbose:
            print(f"  Episode count mismatch - cannot compare rewards/lengths")
    
    # Compare logprobs if available
    if sb3_logps is not None and tensor_logps is not None:
        sb3_valid_logps = sb3_logps[sb3_mask]
        tensor_valid_logps = tensor_logps[tensor_mask]
        
        if len(sb3_valid_logps) == len(tensor_valid_logps) and len(sb3_valid_logps) > 0:
            results.logps_match = np.allclose(sb3_valid_logps, tensor_valid_logps, atol=0.1)
            if verbose:
                print(f"  Log probs match: {results.logps_match}")
                if not results.logps_match:
                    print(f"    Max logp diff: {np.max(np.abs(sb3_valid_logps - tensor_valid_logps)):.6f}")
        else:
            results.logps_match = False
    else:
        results.logps_match = True  # Not comparing
    
    # Overall success
    results.evaluate_policy_success = (
        results.rewards_match and
        results.lengths_match
    )
    
    if verbose:
        print("\n" + "=" * 70)
        if results.evaluate_policy_success:
            print("✓ EVALUATE_POLICY PARITY TEST PASSED")
        else:
            print("✗ EVALUATE_POLICY PARITY TEST FAILED")
        print("=" * 70)
    
    return results


def run_eval_corruptions_parity(
    dataset: str,
    n_envs: int,
    n_eval_episodes: int,
    seed: int = 42,
    verbose: bool = True,
    mode: str = 'valid',
    corruption_mode: str = 'tail',
    k_negatives: int = 50,
) -> EvalParityResults:
    """
    Run eval_corruptions parity test comparing SB3 and tensor implementations.
    
    Args:
        dataset: Dataset name
        n_envs: Number of parallel environments
        n_eval_episodes: Number of episodes to evaluate
        seed: Random seed
        verbose: Whether to print detailed output
        mode: Query set to use ('train', 'valid', 'test')
        corruption_mode: 'head', 'tail', or 'both'
        k_negatives: Number of negative samples per positive
    """
    results = EvalParityResults()
    
    if verbose:
        print("=" * 70)
        print(f"Eval Corruptions Parity Test")
        print(f"Dataset: {dataset}, n_envs: {n_envs}, n_episodes: {n_eval_episodes}")
        print(f"Corruption mode: {corruption_mode}, K negatives: {k_negatives}")
        print("=" * 70)
    
    # Create aligned environments
    if verbose:
        print("\nCreating aligned environments...")
    env_data = create_aligned_environments(dataset, n_envs, mode=mode)
    
    # Limit queries to n_eval_episodes
    queries_sb3 = env_data['queries_sb3'][:n_eval_episodes]
    queries_tensor = env_data['queries_tensor'][:n_eval_episodes]
    
    if verbose:
        print(f"Using {len(queries_sb3)} queries for evaluation")
    
    # Create SB3 environment and model
    if verbose:
        print("Creating SB3 eval environment...")
    sb3_ppo, sb3_env, sb3_im = create_sb3_eval_env(
        env_data['sb3'], queries_sb3, n_envs, seed
    )
    
    # Create tensor environment and model
    if verbose:
        print("Creating tensor eval environment...")
    tensor_ppo, tensor_env, tensor_im, engine = create_tensor_eval_env(
        env_data['tensor'], queries_tensor, n_envs, seed, n_eval_episodes=n_eval_episodes
    )
    
    # Create samplers
    if verbose:
        print("Creating negative samplers...")
    
    device = torch.device("cpu")
    # Map corruption_mode to corruption_scheme list
    if corruption_mode == 'both':
        corruption_scheme = ['head', 'tail']
    else:
        corruption_scheme = [corruption_mode]
    sb3_sampler = create_sb3_sampler(env_data['sb3']['dh'], sb3_im, device, seed, corruption_scheme=corruption_scheme)
    tensor_sampler = create_tensor_sampler(env_data['tensor']['dh'], tensor_im, device, seed)
    
    # Run SB3 eval_corruptions
    if verbose:
        print("\nRunning SB3 eval_corruptions...")
    
    try:
        # SB3 API: eval_corruptions(model, env, data, sampler, n_corruptions, ...)
        sb3_metrics = sb3_eval_corruptions(
            model=sb3_ppo,
            env=sb3_env,
            data=queries_sb3,  # list of query objects
            sampler=sb3_sampler,
            n_corruptions=k_negatives,
            deterministic=True,
            corruption_scheme=corruption_scheme,  # ['head'] or ['tail'] or ['head', 'tail']
            verbose=0,
        )
        
        # SB3 returns keys like 'mrr_mean', 'hits1_mean', etc.
        results.sb3_mrr = sb3_metrics.get('mrr_mean', sb3_metrics.get('MRR', 0.0))
        results.sb3_hits1 = sb3_metrics.get('hits1_mean', sb3_metrics.get('Hits@1', 0.0))
        results.sb3_hits3 = sb3_metrics.get('hits3_mean', sb3_metrics.get('Hits@3', 0.0))
        results.sb3_hits10 = sb3_metrics.get('hits10_mean', sb3_metrics.get('Hits@10', 0.0))
        
        if verbose:
            print(f"  SB3 MRR: {results.sb3_mrr:.4f}")
            print(f"  SB3 Hits@1: {results.sb3_hits1:.4f}")
            print(f"  SB3 Hits@3: {results.sb3_hits3:.4f}")
            print(f"  SB3 Hits@10: {results.sb3_hits10:.4f}")
    except Exception as e:
        if verbose:
            print(f"  SB3 eval_corruptions error: {e}")
        import traceback
        traceback.print_exc()
        return results
    
    # Run tensor eval_corruptions
    if verbose:
        print("\nRunning tensor eval_corruptions...")
    
    try:
        # Tensor API: eval_corruptions(actor, env, queries, sampler, ...)
        # Convert queries to tensor format - just the query atom, not padded
        tensor_im = env_data['tensor']['im']
        query_tensors = []
        for q in queries_tensor:
            query_atom = tensor_im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
            query_tensors.append(query_atom)
        queries_t = torch.stack(query_tensors, dim=0)  # shape (B, 3)
        
        tensor_metrics = tensor_eval_corruptions(
            actor=tensor_ppo.policy,
            env=tensor_env,
            queries=queries_t,
            sampler=tensor_sampler,
            n_corruptions=k_negatives,
            corruption_modes=tuple(corruption_scheme),  # ('head',) or ('tail',) or ('head', 'tail')
            deterministic=True,
            verbose=False,
        )
        
        results.tensor_mrr = tensor_metrics.get('MRR', 0.0)
        results.tensor_hits1 = tensor_metrics.get('Hits@1', 0.0)
        results.tensor_hits3 = tensor_metrics.get('Hits@3', 0.0)
        results.tensor_hits10 = tensor_metrics.get('Hits@10', 0.0)
        
        if verbose:
            print(f"  Tensor MRR: {results.tensor_mrr:.4f}")
            print(f"  Tensor Hits@1: {results.tensor_hits1:.4f}")
            print(f"  Tensor Hits@3: {results.tensor_hits3:.4f}")
            print(f"  Tensor Hits@10: {results.tensor_hits10:.4f}")
    except Exception as e:
        if verbose:
            print(f"  Tensor eval_corruptions error: {e}")
        import traceback
        traceback.print_exc()
        return results
    
    # Compare results (with 1% relative tolerance)
    if verbose:
        print("\n--- Results Comparison ---")
    
    def rel_close(a, b, rtol=0.01):
        """Check if two values are relatively close."""
        max_val = max(abs(a), abs(b))
        if max_val < 1e-8:
            return True
        return abs(a - b) / max_val < rtol
    
    results.mrr_match = rel_close(results.sb3_mrr, results.tensor_mrr)
    results.hits1_match = rel_close(results.sb3_hits1, results.tensor_hits1)
    results.hits3_match = rel_close(results.sb3_hits3, results.tensor_hits3)
    results.hits10_match = rel_close(results.sb3_hits10, results.tensor_hits10)
    
    if verbose:
        print(f"  MRR match: {results.mrr_match}")
        print(f"  Hits@1 match: {results.hits1_match}")
        print(f"  Hits@3 match: {results.hits3_match}")
        print(f"  Hits@10 match: {results.hits10_match}")
        
        if not results.mrr_match:
            print(f"    MRR diff: {abs(results.sb3_mrr - results.tensor_mrr):.6f}")
        if not results.hits1_match:
            print(f"    Hits@1 diff: {abs(results.sb3_hits1 - results.tensor_hits1):.6f}")
    
    # Overall success
    results.eval_corruptions_success = (
        results.mrr_match and
        results.hits1_match and
        results.hits3_match and
        results.hits10_match
    )
    
    if verbose:
        print("\n" + "=" * 70)
        if results.eval_corruptions_success:
            print("✓ EVAL_CORRUPTIONS PARITY TEST PASSED")
        else:
            print("✗ EVAL_CORRUPTIONS PARITY TEST FAILED")
        print("=" * 70)
    
    return results


# ============================================================
# Pytest Tests
# ============================================================

class TestEvaluatePolicyParity:
    """Tests for evaluate_policy parity."""
    
    @pytest.mark.parametrize("n_envs,n_episodes", [
        (1, 5),
        (2, 10),
    ])
    def test_evaluate_policy_basic(self, n_envs, n_episodes):
        """Basic evaluate_policy parity test."""
        results = run_evaluate_policy_parity(
            dataset="countries_s3",
            n_envs=n_envs,
            n_eval_episodes=n_episodes,
            seed=42,
            verbose=False,
        )
        assert results.evaluate_policy_success, \
            f"evaluate_policy parity failed for n_envs={n_envs}, n_episodes={n_episodes}"
    
    def test_evaluate_policy_rewards_match(self):
        """Test that rewards match between implementations."""
        results = run_evaluate_policy_parity(
            dataset="countries_s3",
            n_envs=2,
            n_eval_episodes=10,
            seed=42,
            verbose=False,
        )
        assert results.rewards_match, "Rewards don't match between SB3 and tensor"
    
    def test_evaluate_policy_lengths_match(self):
        """Test that episode lengths match between implementations."""
        results = run_evaluate_policy_parity(
            dataset="countries_s3",
            n_envs=2,
            n_eval_episodes=10,
            seed=42,
            verbose=False,
        )
        assert results.lengths_match, "Episode lengths don't match between SB3 and tensor"


class TestEvalCorruptionsParity:
    """Tests for eval_corruptions parity.
    
    Tests cover:
    - countries_s3: Uses domain constraints (tail corruption only, regions domain has 5 entities -> max 4 negatives)
    - family: No domain constraints (both head and tail corruptions)
    """
    
    def test_eval_corruptions_countries_s3_tail(self):
        """Test eval_corruptions for countries_s3 with tail corruption (domain-constrained).
        
        Note: countries_s3 locatedin predicate has tail domain = regions (5 entities),
        so we use num_negs=None to get all possible corruptions (4 negatives per positive).
        """
        results = run_eval_corruptions_parity(
            dataset="countries_s3",
            n_envs=2,
            n_eval_episodes=5,
            seed=42,
            verbose=True,
            corruption_mode="tail",
            k_negatives=None,  # Use None to get all possible corruptions (domain has only 5 entities)
        )
        assert results.eval_corruptions_success, \
            f"eval_corruptions parity failed for countries_s3 tail: MRR SB3={results.sb3_mrr:.4f}, Tensor={results.tensor_mrr:.4f}"
    
    def test_eval_corruptions_family_tail(self):
        """Test eval_corruptions for family with tail corruption."""
        results = run_eval_corruptions_parity(
            dataset="family",
            n_envs=2,
            n_eval_episodes=5,
            seed=42,
            verbose=True,
            corruption_mode="tail",
            k_negatives=20,
        )
        assert results.eval_corruptions_success, \
            f"eval_corruptions parity failed for family tail: MRR SB3={results.sb3_mrr:.4f}, Tensor={results.tensor_mrr:.4f}"
    
    def test_eval_corruptions_family_head(self):
        """Test eval_corruptions for family with head corruption."""
        results = run_eval_corruptions_parity(
            dataset="family",
            n_envs=2,
            n_eval_episodes=5,
            seed=42,
            verbose=True,
            corruption_mode="head",
            k_negatives=20,
        )
        assert results.eval_corruptions_success, \
            f"eval_corruptions parity failed for family head: MRR SB3={results.sb3_mrr:.4f}, Tensor={results.tensor_mrr:.4f}"
    
    def test_eval_corruptions_family_both(self):
        """Test eval_corruptions for family with both head and tail corruption."""
        results = run_eval_corruptions_parity(
            dataset="family",
            n_envs=2,
            n_eval_episodes=5,
            seed=42,
            verbose=True,
            corruption_mode="both",
            k_negatives=20,
        )
        assert results.eval_corruptions_success, \
            f"eval_corruptions parity failed for family both: MRR SB3={results.sb3_mrr:.4f}, Tensor={results.tensor_mrr:.4f}"


# ============================================================
# CLI Interface
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluation Parity Tests")
    parser.add_argument("--dataset", type=str, default="countries_s3",
                       help="Dataset name (default: countries_s3)")
    parser.add_argument("--n-envs", type=int, default=2,
                       help="Number of environments (default: 2)")
    parser.add_argument("--n-eval-episodes", type=int, default=10,
                       help="Number of evaluation episodes (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--mode", type=str, default="valid",
                       choices=["train", "valid", "test"],
                       help="Query set to use (default: valid)")
    parser.add_argument("--test", type=str, default="both",
                       choices=["evaluate_policy", "eval_corruptions", "both"],
                       help="Which test to run (default: both)")
    parser.add_argument("--corruption-mode", type=str, default="tail",
                       choices=["head", "tail", "both"],
                       help="Corruption mode for eval_corruptions (default: tail)")
    parser.add_argument("--k-negatives", type=int, default=50,
                       help="Number of negatives for eval_corruptions (default: 50)")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Enable verbose output")
    parser.add_argument("--quiet", action="store_true", default=False,
                       help="Disable verbose output")
    
    args = parser.parse_args()
    verbose = args.verbose and not args.quiet
    
    success = True
    
    if args.test in ["evaluate_policy", "both"]:
        print("\n" + "=" * 80)
        print("RUNNING: evaluate_policy parity test")
        print("=" * 80)
        
        results = run_evaluate_policy_parity(
            dataset=args.dataset,
            n_envs=args.n_envs,
            n_eval_episodes=args.n_eval_episodes,
            seed=args.seed,
            verbose=verbose,
            mode=args.mode,
        )
        
        if not results.evaluate_policy_success:
            success = False
    
    if args.test in ["eval_corruptions", "both"]:
        print("\n" + "=" * 80)
        print("RUNNING: eval_corruptions parity test")
        print("=" * 80)
        
        results = run_eval_corruptions_parity(
            dataset=args.dataset,
            n_envs=args.n_envs,
            n_eval_episodes=args.n_eval_episodes,
            seed=args.seed,
            verbose=verbose,
            mode=args.mode,
            corruption_mode=args.corruption_mode,
            k_negatives=args.k_negatives,
        )
        
        if not results.eval_corruptions_success:
            success = False
    
    sys.exit(0 if success else 1)
