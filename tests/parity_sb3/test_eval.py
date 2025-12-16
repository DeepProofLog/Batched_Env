"""
Evaluation Parity Tests.

Tests verifying that the tensor-based evaluation functions produce the SAME
results as the SB3 evaluation functions, including:
- evaluate_policy: Rollout-based policy evaluation
- eval_corruptions: Link prediction evaluation with corruption sampling

Uses deterministic action selection and aligned environments to ensure
reproducible comparisons.

Usage:
    python tests/parity/test_eval_parity.py --dataset countries_s3 --n-eval-episodes 10
"""
import os
import sys
from pathlib import Path
import pytest
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch
import torch.nn as nn
import numpy as np
from collections import deque

# Setup paths
ROOT = Path(__file__).resolve().parents[2]
SB3_ROOT = ROOT / "sb3"
TEST_ENVS_ROOT = ROOT / "tests" / "other" / "test_envs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SB3_ROOT) not in sys.path:
    # Use index 1 to allow local imports over sb3 modules if needed, or keep 1
    sys.path.insert(1, str(SB3_ROOT))
if str(TEST_ENVS_ROOT) not in sys.path:
    sys.path.insert(2, str(TEST_ENVS_ROOT))

# Import seeding utilities (must be before other local imports to set up paths correctly)
from tensor.utils.tensor_seeding import ParityTestSeeder, ParityTestConfig, seed_all
from tests.test_utils.parity_config import ParityConfig, TOLERANCE, create_parser, config_from_args

from stable_baselines3.common.monitor import Monitor

# sb3 imports
from sb3.sb3_custom_dummy_env import CustomDummyVecEnv
from sb3.sb3_dataset import DataHandler as StrDataHandler
from sb3.sb3_index_manager import IndexManager as StrIndexManager
from sb3.sb3_env import LogicEnv_gym as StrEnv
from sb3.sb3_model import PPO_custom, CustomActorCriticPolicy, CustomCombinedExtractor
from sb3.sb3_embeddings import EmbedderLearnable as SB3Embedder
from sb3.sb3_model_eval import evaluate_policy as sb3_evaluate_policy
from sb3.sb3_model_eval import eval_corruptions as sb3_eval_corruptions
from sb3.sb3_model_eval import SB3EvalStepTrace, SB3EvalCorruptionsTrace
from sb3.sb3_neg_sampling import BasicNegativeSamplerDomain, get_sampler as get_sb3_sampler

# Tensor imports
from tensor.tensor_data_handler import DataHandler
from tensor.tensor_index_manager import IndexManager
from tensor.tensor_unification import UnificationEngine
from tensor.tensor_env import BatchedEnv
from tensor.tensor_embeddings import EmbedderLearnable as TensorEmbedder
from tensor.tensor_model import ActorCriticPolicy as TensorPolicy
from tensor.tensor_ppo import PPO as TensorPPO
from tensor.tensor_model_eval import evaluate_policy as tensor_evaluate_policy
from tensor.tensor_model_eval import eval_corruptions as tensor_eval_corruptions
from tensor.tensor_model_eval import EvalStepTrace, EvalCorruptionsTrace
from tensor.tensor_sampler import Sampler, SamplerConfig


# ============================================================================
# Default Configuration
# ============================================================================

def create_default_config() -> SimpleNamespace:
    """Centralized defaults for eval parity tests."""
    nmsp=  SimpleNamespace(
        # Dataset/files
        dataset="countries_s3",
        data_path="./data/",
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        train_depth=None,
        max_total_vars=1000000,
        
        # Environment
        padding_atoms=6,
        padding_states=100,
        max_depth=20,
        memory_pruning=True,
        use_exact_memory=True,
        skip_unary_actions=True,
        end_proof_action=True,
        reward_type=0,
        device="cpu",
        
        # PPO/training
        n_envs=2,
        n_steps=20,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        
        # Embedding/model
        embed_dim=64,
        atom_embedder="transe",
        state_embedder="sum",
        hidden_dim=256,
        num_layers=8,
        dropout_prob=0.0,
        
        # Eval-specific
        n_eval_episodes=10,
        corruption_mode="tail",
        k_negatives=50,
        mode="valid",  # which split to evaluate
        
        # Seeds
        seed=42,
    )
    nmsp.n_vars_for_embedder = nmsp.max_total_vars
    if nmsp.dataset == "countries_s3":
        nmsp.corruption_mode = "tail"
    return nmsp


def clone_config(config: SimpleNamespace) -> SimpleNamespace:
    """Clone a config SimpleNamespace."""
    return SimpleNamespace(**vars(config))


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


def compare_traces(
    sb3_traces: List[SB3EvalCorruptionsTrace],
    tensor_traces: List[EvalCorruptionsTrace],
    max_traces: int = 5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compare traces from SB3 and tensor eval_corruptions to find differences.
    
    Args:
        sb3_traces: List of SB3EvalCorruptionsTrace
        tensor_traces: List of EvalCorruptionsTrace
        max_traces: Maximum number of traces to compare in detail
        verbose: Whether to print comparison output
        
    Returns:
        Dict with comparison results and first differences found
    """
    result = {
        "num_sb3_traces": len(sb3_traces),
        "num_tensor_traces": len(tensor_traces),
        "matches": [],
        "mismatches": [],
    }
    
    if verbose:
        print("\n" + "=" * 70)
        print("TRACE COMPARISON")
        print(f"SB3 traces: {len(sb3_traces)}, Tensor traces: {len(tensor_traces)}")
        print("=" * 70)
    
    # Compare traces pairwise
    num_to_compare = min(len(sb3_traces), len(tensor_traces), max_traces)
    for i in range(num_to_compare):
        sb3_t = sb3_traces[i]
        tensor_t = tensor_traces[i]
        
        # Extract comparable fields
        sb3_rank = sb3_t.get("rank", -1)
        tensor_rank = tensor_t.get("rank", -1)
        
        sb3_pos_logp = sb3_t.get("pos_logp", float('-inf'))
        tensor_pos_logp = tensor_t.get("pos_logp", float('-inf'))
        
        sb3_mode = sb3_t.get("mode", "")
        tensor_mode = tensor_t.get("mode", "")
        
        sb3_query_idx = sb3_t.get("query_idx", -1)
        tensor_query_idx = tensor_t.get("query_idx", -1)
        
        sb3_num_negs = sb3_t.get("num_negatives", 0)
        tensor_num_negs = tensor_t.get("num_negatives", 0)
        
        # Check for rank mismatch
        rank_match = sb3_rank == tensor_rank
        logp_match = abs(sb3_pos_logp - tensor_pos_logp) < 0.001
        
        trace_info = {
            "idx": i,
            "sb3_rank": sb3_rank,
            "tensor_rank": tensor_rank,
            "rank_match": rank_match,
            "sb3_pos_logp": sb3_pos_logp,
            "tensor_pos_logp": tensor_pos_logp,
            "logp_match": logp_match,
            "sb3_mode": sb3_mode,
            "tensor_mode": tensor_mode,
            "sb3_query_idx": sb3_query_idx,
            "tensor_query_idx": tensor_query_idx,
            "sb3_num_negs": sb3_num_negs,
            "tensor_num_negs": tensor_num_negs,
        }
        
        if rank_match and logp_match:
            result["matches"].append(trace_info)
        else:
            result["mismatches"].append(trace_info)
        
        if verbose:
            match_symbol = "✓" if (rank_match and logp_match) else "✗"
            print(f"\nTrace {i} [{match_symbol}]:")
            print(f"  Mode: SB3={sb3_mode}, Tensor={tensor_mode}")
            print(f"  Query idx: SB3={sb3_query_idx}, Tensor={tensor_query_idx}")
            print(f"  Num negs: SB3={sb3_num_negs}, Tensor={tensor_num_negs}")
            print(f"  Rank: SB3={sb3_rank}, Tensor={tensor_rank} {'✓' if rank_match else '✗'}")
            print(f"  Pos logp: SB3={sb3_pos_logp:.4f}, Tensor={tensor_pos_logp:.4f} {'✓' if logp_match else '✗'}")
            
            # If mismatch, print more details
            if not rank_match or not logp_match:
                sb3_neg_logps = sb3_t.get("neg_logps", [])
                tensor_neg_logps = tensor_t.get("neg_logps", [])
                
                print(f"  Neg logps (first 5):")
                print(f"    SB3: {[f'{x:.4f}' for x in sb3_neg_logps[:5]]}")
                print(f"    Tensor: {[f'{x:.4f}' for x in tensor_neg_logps[:5]]}")
                
                sb3_pos_succ = sb3_t.get("pos_success", False)
                tensor_pos_succ = tensor_t.get("pos_success", False)
                print(f"  Pos success: SB3={sb3_pos_succ}, Tensor={tensor_pos_succ}")
                
                # Look at episode traces if available
                sb3_ep_traces = sb3_t.get("episode_traces", [])
                tensor_ep_traces = tensor_t.get("episode_traces", [])
                
                if sb3_ep_traces or tensor_ep_traces:
                    print(f"  Episode traces: SB3 has {len(sb3_ep_traces)}, Tensor has {len(tensor_ep_traces)}")
                    
                    # DETAILED COMPARISON: Compare episode traces step by step
                    max_steps_to_compare = min(10, len(sb3_ep_traces), len(tensor_ep_traces))
                    for j in range(max_steps_to_compare):
                        sb3_ep = sb3_ep_traces[j]
                        tensor_ep = tensor_ep_traces[j]
                        
                        sb3_action = sb3_ep.get("action", -1)
                        tensor_action = tensor_ep.get("action", -1)
                        sb3_reward = sb3_ep.get("reward", 0.0)
                        tensor_reward = tensor_ep.get("reward", 0.0)
                        sb3_lp = sb3_ep.get("log_prob", 0.0)
                        tensor_lp = tensor_ep.get("log_prob", 0.0)
                        sb3_done = sb3_ep.get("done", False)
                        tensor_done = tensor_ep.get("done", False)
                        
                        # Check if this step differs
                        step_differs = (sb3_action != tensor_action or 
                                       abs(sb3_lp - tensor_lp) > 0.001 or
                                       abs(sb3_reward - tensor_reward) > 1e-6 or
                                       sb3_done != tensor_done)
                        
                        if step_differs or j < 3:  # Always show first 3 steps
                            marker = "***DIFF***" if step_differs else ""
                            print(f"\n    === Step {j} {marker} ===")
                            print(f"      Action:  SB3={sb3_action}, Tensor={tensor_action}")
                            print(f"      Reward:  SB3={sb3_reward:.6f}, Tensor={tensor_reward:.6f}")
                            print(f"      LogProb: SB3={sb3_lp:.6f}, Tensor={tensor_lp:.6f}")
                            print(f"      Done:    SB3={sb3_done}, Tensor={tensor_done}")
                            
                            # Print state observations if available
                            sb3_state_obs = sb3_ep.get("state_obs", {})
                            tensor_state_obs = tensor_ep.get("state_obs", {})
                            
                            if sb3_state_obs or tensor_state_obs:
                                # Sub-index (current state)
                                sb3_sub = sb3_state_obs.get("sub_index")
                                tensor_sub = tensor_state_obs.get("sub_index")
                                
                                if sb3_sub is not None:
                                    sb3_sub_arr = np.array(sb3_sub) if not isinstance(sb3_sub, np.ndarray) else sb3_sub
                                    # Get first non-padding atom
                                    print(f"      SB3 sub_index[0]: {sb3_sub_arr[0].tolist() if len(sb3_sub_arr.shape) > 0 else sb3_sub_arr}")
                                if tensor_sub is not None:
                                    tensor_sub_arr = np.array(tensor_sub) if not isinstance(tensor_sub, np.ndarray) else tensor_sub
                                    print(f"      Tensor sub_index[0]: {tensor_sub_arr[0].tolist() if len(tensor_sub_arr.shape) > 0 else tensor_sub_arr}")
                                
                                # Compare sub_index match
                                if sb3_sub is not None and tensor_sub is not None:
                                    sb3_arr = np.array(sb3_sub)
                                    tensor_arr = np.array(tensor_sub)
                                    if sb3_arr.shape == tensor_arr.shape:
                                        match = np.allclose(sb3_arr, tensor_arr)
                                        print(f"      sub_index MATCH: {match}")
                                    else:
                                        print(f"      sub_index SHAPE MISMATCH: SB3={sb3_arr.shape}, Tensor={tensor_arr.shape}")
                                
                                # Action mask
                                sb3_mask = sb3_state_obs.get("action_mask")
                                tensor_mask = tensor_state_obs.get("action_mask")
                                
                                if sb3_mask is not None and tensor_mask is not None:
                                    sb3_mask_arr = np.array(sb3_mask)
                                    tensor_mask_arr = np.array(tensor_mask)
                                    sb3_valid = int(np.sum(sb3_mask_arr))
                                    tensor_valid = int(np.sum(tensor_mask_arr))
                                    print(f"      Valid actions: SB3={sb3_valid}, Tensor={tensor_valid}")
                                    
                                    if sb3_mask_arr.shape == tensor_mask_arr.shape:
                                        mask_match = np.array_equal(sb3_mask_arr, tensor_mask_arr)
                                        print(f"      action_mask MATCH: {mask_match}")
                                        if not mask_match:
                                            # Find differing positions
                                            diff_pos = np.where(sb3_mask_arr != tensor_mask_arr)[0]
                                            print(f"      Differing positions: {diff_pos[:10].tolist()}")
                                
                                # Derived states
                                sb3_derived = sb3_state_obs.get("derived_sub_indices")
                                tensor_derived = tensor_state_obs.get("derived_sub_indices")
                                
                                if sb3_derived is not None:
                                    sb3_derived_arr = np.array(sb3_derived)
                                    # Count non-padding derived states
                                    if len(sb3_derived_arr.shape) >= 2:
                                        sb3_num_derived = np.sum(sb3_derived_arr[:, 0, 0] != 0)  # Assuming padding_idx=0
                                        print(f"      SB3 num derived states: {sb3_num_derived}")
                                if tensor_derived is not None:
                                    tensor_derived_arr = np.array(tensor_derived)
                                    if len(tensor_derived_arr.shape) >= 2:
                                        tensor_num_derived = np.sum(tensor_derived_arr[:, 0, 0] != 0)
                                        print(f"      Tensor num derived states: {tensor_num_derived}")
                            
                            # Print logits if available
                            sb3_logits = sb3_ep.get("logits")
                            tensor_logits = tensor_ep.get("logits")
                            
                            if sb3_logits is not None or tensor_logits is not None:
                                if sb3_logits is not None:
                                    sb3_logits_arr = np.array(sb3_logits)
                                    print(f"      SB3 logits (first 5): {sb3_logits_arr[:5].tolist()}")
                                if tensor_logits is not None:
                                    tensor_logits_arr = np.array(tensor_logits)
                                    print(f"      Tensor logits (first 5): {tensor_logits_arr[:5].tolist()}")
                            
                            if step_differs:
                                break  # Stop at first difference for clarity
    
    if verbose:
        print("\n" + "-" * 70)
        print(f"Summary: {len(result['matches'])} matches, {len(result['mismatches'])} mismatches")
        print("-" * 70)
    
    return result


def create_aligned_environments(config: SimpleNamespace):
    """
    Create SB3 and tensor environments with aligned queries.
    
    Args:
        dataset: Dataset name
        n_envs: Number of parallel environments
        mode: 'valid' or 'test' for query selection
    
    Returns:
        Dict with environment data for both implementations
    """
    device = torch.device(config.device)
    
    # ===== SB3 Setup =====
    dh_sb3 = StrDataHandler(
        dataset_name=config.dataset,
        base_path=config.data_path,
        train_file=config.train_file,
        valid_file=config.valid_file, 
        test_file=config.test_file,
        rules_file=config.rules_file,
        facts_file=config.facts_file,
        train_depth=config.train_depth,
        corruption_mode=True,  # Enable domain loading for negative sampling
    )
    
    im_sb3 = StrIndexManager(
        constants=dh_sb3.constants,
        predicates=dh_sb3.predicates,
        max_total_vars=config.max_total_vars,
        rules=dh_sb3.rules,
        padding_atoms=config.padding_atoms,
        max_arity=dh_sb3.max_arity,
        device=device,
    )
    
    facts_set = set(dh_sb3.facts)
    im_sb3.build_fact_index(list(facts_set), deterministic=True)
    
    # ===== Tensor Setup =====
    dh_tensor = DataHandler(
        dataset_name=config.dataset,
        base_path=config.data_path,
        train_file=config.train_file,
        valid_file=config.valid_file,
        test_file=config.test_file,
        rules_file=config.rules_file,
        facts_file=config.facts_file,
        train_depth=config.train_depth,
        corruption_mode='dynamic',  # Enable domain loading for negative sampling
    )
    
    im_tensor = IndexManager(
        constants=dh_tensor.constants,
        predicates=dh_tensor.predicates,
        max_total_runtime_vars=config.max_total_vars,
        padding_atoms=config.padding_atoms,
        max_arity=dh_tensor.max_arity,
        device=device,
        rules=dh_tensor.rules,
    )
    dh_tensor.materialize_indices(im=im_tensor, device=device)
    
    # Select queries based on mode
    if config.mode == 'valid':
        queries_sb3 = dh_sb3.valid_queries
        queries_tensor = dh_tensor.valid_queries
    elif config.mode == 'test':
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
        'padding_atoms': config.padding_atoms,
        'padding_states': config.padding_states,
    }


def create_sb3_eval_env(config: SimpleNamespace, env_data: Dict, queries: List):
    """Create SB3 environment and PPO for evaluation."""
    device = torch.device(config.device)
    padding_atoms = env_data.get('padding_atoms', config.padding_atoms)
    padding_states = env_data.get('padding_states', config.padding_states)
    
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
                seed=config.seed,
                max_depth=config.max_depth,
                memory_pruning=config.memory_pruning,
                padding_atoms=padding_atoms,
                padding_states=padding_states,
                verbose=0,
                prover_verbose=0,
                device=device,
                engine='python',
                engine_strategy='complete',
                skip_unary_actions=config.skip_unary_actions,
                endf_action=config.end_proof_action,
                reward_type=config.reward_type,
            )
            env._train_ptr = env_idx
            return Monitor(env)
        return _init
    
    env_fns = [make_env(env_idx=i) for i in range(config.n_envs)]
    vec_env = CustomDummyVecEnv(env_fns)
    
    # CustomDummyVecEnv already has the episode tracking attributes built-in
    # Just reset them for this evaluation run
    vec_env._episode_target = np.zeros(config.n_envs, dtype=int)
    vec_env._episode_count = np.zeros(config.n_envs, dtype=int)
    vec_env.active_envs = np.ones(config.n_envs, dtype=bool)
    vec_env._episodes_done = 0
    
    # Create embedder with fixed seed
    torch.manual_seed(config.seed)
    embedder = SB3Embedder(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=config.n_vars_for_embedder,
        max_arity=dh.max_arity,
        padding_atoms=padding_atoms,
        atom_embedder=config.atom_embedder,
        state_embedder=config.state_embedder,
        constant_embedding_size=config.embed_dim,
        predicate_embedding_size=config.embed_dim,
        atom_embedding_size=config.embed_dim,
        device=device,
    )
    embedder.embed_dim = config.embed_dim
    
    # Create PPO with fixed seed
    torch.manual_seed(config.seed)
    ppo = PPO_custom(
        policy=CustomActorCriticPolicy,
        env=vec_env,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=1,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        verbose=0,
        device=device,
        seed=config.seed,
        policy_kwargs={
            "features_extractor_class": CustomCombinedExtractor,
            "features_extractor_kwargs": {
                "embedder": embedder,
                "features_dim": config.embed_dim,
            },
        },
    )
    
    return ppo, vec_env, im


def create_tensor_eval_env(
    config: SimpleNamespace,
    env_data: Dict,
    queries: List,
    n_eval_episodes: int = None,
):
    """Create tensor environment and PPO for evaluation.
    
    Args:
        env_data: Environment data dictionary
        queries: List of query objects
        n_eval_episodes: Total number of episodes to evaluate (if None, uses len(queries))
    """
    device = torch.device(config.device)
    padding_atoms = env_data.get('padding_atoms', config.padding_atoms)
    padding_states = env_data.get('padding_states', config.padding_states)
    
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
        end_pred_idx=im.end_pred_idx,
        end_proof_action=config.end_proof_action,
        max_derived_per_state=padding_states,
    )
    engine.index_manager = im
    
    n_envs = config.n_envs
    
    # Convert queries to tensor format
    query_tensors = []
    for q in queries:
        query_atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
        query_padded = torch.full((padding_atoms, 3), im.padding_idx, dtype=torch.long, device=device)
        query_padded[0] = query_atom
        query_tensors.append(query_padded)
    
    queries_tensor = torch.stack(query_tensors, dim=0)
    
    # Create environment in train mode initially (set_eval_dataset will switch to eval)
    # Note: We must start in 'train' mode because 'eval' mode requires set_eval_dataset
    # to be called first to initialize the evaluation slots
    env = BatchedEnv(
        batch_size=n_envs,
        queries=queries_tensor,
        labels=torch.ones(len(queries), dtype=torch.long, device=device),
        query_depths=torch.ones(len(queries), dtype=torch.long, device=device),
        unification_engine=engine,
        mode='train',  # Start in train mode, set_eval_dataset will switch to eval
        max_depth=config.max_depth,
        memory_pruning=config.memory_pruning,
        use_exact_memory=config.use_exact_memory,
        skip_unary_actions=config.skip_unary_actions,
        end_proof_action=config.end_proof_action,
        reward_type=config.reward_type,
        padding_atoms=padding_atoms,
        padding_states=padding_states,
        true_pred_idx=im.predicate_str2idx.get('True'),
        false_pred_idx=im.predicate_str2idx.get('False'),
        end_pred_idx=im.predicate_str2idx.get('Endf'),
        verbose=0,
        prover_verbose=0,
        device=device,
        runtime_var_start_index=im.constant_no + 1,
        total_vocab_size=im.constant_no + config.max_total_vars,
        sample_deterministic_per_env=True,
    )
    
    # Set up the evaluation dataset with per_slot_lengths
    # SB3's DummyVecEnv gives each sub-env the SAME query list.
    # Each sub-env has its own _train_ptr which starts at 0.
    # So all envs get queries in the same order: 0, 1, 2, ... (mod n_queries)
    
    # Compute target episodes per env using n_eval_episodes (total episodes to run)
    # This matches SB3's target distribution: [(n_eval_episodes + i) // n_envs for i in range(n_envs)]
    n_queries = len(queries)
    targets = [(n_eval_episodes + i) // config.n_envs for i in range(config.n_envs)]
    total_episodes = sum(targets)
    
    # Reorder queries to match SB3's pattern:
    # Each env independently iterates through queries starting from index 0.
    # So env0 gets queries [0, 1, 2, ...] and env1 also gets [0, 1, 2, ...]
    reordered_queries = []
    
    for env_idx in range(config.n_envs):
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
    torch.manual_seed(config.seed)
    embedder = TensorEmbedder(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=config.n_vars_for_embedder,
        max_arity=dh.max_arity,
        padding_atoms=padding_atoms,
        atom_embedder=config.atom_embedder,
        state_embedder=config.state_embedder,
        constant_embedding_size=config.embed_dim,
        predicate_embedding_size=config.embed_dim,
        atom_embedding_size=config.embed_dim,
        device=str(device),
    )
    embedder.embed_dim = config.embed_dim
    
    # Create policy with fixed seed
    action_size = padding_states
    torch.manual_seed(config.seed)
    policy = TensorPolicy(
        embedder=embedder,
        embed_dim=config.embed_dim,
        action_dim=action_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout_prob=config.dropout_prob,
        device=device,
    ).to(device)
    
    # Create PPO with fixed seed
    torch.manual_seed(config.seed)
    ppo = TensorPPO(
        policy=policy,
        env=env,
        n_steps=config.n_steps,
        learning_rate=config.learning_rate,
        n_epochs=1,
        batch_size=config.batch_size,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        normalize_advantage=True,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        device=device,
        verbose=False,
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


def create_tensor_sampler(dh, im, dh_sb3, device, seed: int = 42, corruption_scheme=['head', 'tail']):
    """Create tensor negative sampler with domain constraints matching SB3.
    
    Uses the same domain2idx and entity2domain format as SB3's get_sampler function.
    
    Args:
        dh: Tensor DataHandler
        im: Tensor IndexManager  
        dh_sb3: SB3 DataHandler (needed for domain2entity which is loaded by SB3)
        device: Target device
        seed: Random seed
        corruption_scheme: List of corruption modes ['head'], ['tail'], or ['head', 'tail']
    """
    domain2idx, entity2domain = dh.get_sampler_domain_info()

    # Map corruption_scheme to default_mode
    if corruption_scheme == ['head']:
        default_mode = 'head'
    elif corruption_scheme == ['tail']:
        default_mode = 'tail'
    else:
        default_mode = 'both'

    # Use the pre-computed all_known_triples_idx from the data handler
    sampler = Sampler.from_data(
        all_known_triples_idx=dh.all_known_triples_idx,
        num_entities=im.constant_no,
        num_relations=im.predicate_no,
        device=device,
        default_mode=default_mode,
        seed=seed,
        domain2idx=domain2idx,
        entity2domain=entity2domain,
    )
    
    return sampler


def run_evaluate_policy_parity(
    dataset: str,
    n_envs: int,
    n_eval_episodes: int,
    seed: int = 42,
    verbose: bool = True,
    mode: str = 'valid',
    config: Optional[SimpleNamespace] = None,
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
        config: Optional base config to clone and override
    """
    results = EvalParityResults()
    cfg = clone_config(config or create_default_config())
    cfg.dataset = dataset
    cfg.n_envs = n_envs
    cfg.n_eval_episodes = n_eval_episodes
    cfg.seed = seed
    cfg.verbose = verbose
    cfg.mode = mode
    
    if verbose:
        print("=" * 70)
        print("Evaluate Policy Parity Test")
        print(f"Dataset: {cfg.dataset}, n_envs: {cfg.n_envs}, n_episodes: {cfg.n_eval_episodes}")
        print("=" * 70)
    
    if verbose:
        print("\nCreating aligned environments...")
    env_data = create_aligned_environments(cfg)
    
    # Limit queries to n_eval_episodes
    queries_sb3 = env_data['queries_sb3'][:cfg.n_eval_episodes]
    queries_tensor = env_data['queries_tensor'][:cfg.n_eval_episodes]
    
    if verbose:
        print(f"Using {len(queries_sb3)} queries for evaluate_policy parity")
        for i, (q_sb3, q_tensor) in enumerate(zip(queries_sb3, queries_tensor)):
            print(f"  Query {i}: SB3={q_sb3.predicate}({q_sb3.args[0]}, {q_sb3.args[1]}), Tensor={q_tensor.predicate}({q_tensor.args[0]}, {q_tensor.args[1]})")
    
    # Create SB3 environment and model
    if verbose:
        print("Creating SB3 eval environment...")
    sb3_ppo, sb3_env, sb3_im = create_sb3_eval_env(cfg, env_data['sb3'], queries_sb3)
    
    # Create tensor environment and model
    if verbose:
        print("Creating tensor eval environment...")
    tensor_ppo, tensor_env, tensor_im, engine = create_tensor_eval_env(
        cfg, env_data['tensor'], queries_tensor, n_eval_episodes=cfg.n_eval_episodes
    )
    
    # Run SB3 evaluate_policy
    if verbose:
        print("\nRunning SB3 evaluate_policy...")
    
    try:
        result = sb3_evaluate_policy(
            model=sb3_ppo,
            env=sb3_env,
            n_eval_episodes=cfg.n_eval_episodes,
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
        targets = [(cfg.n_eval_episodes + i) // cfg.n_envs for i in range(cfg.n_envs)]
        
        tensor_results = tensor_evaluate_policy(
            actor=tensor_ppo.policy,
            env=tensor_env,
            target_episodes=targets,  # Use per-env targets like SB3
            track_logprobs=True,
            verbose=True
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
    config: Optional[SimpleNamespace] = None,
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
    cfg = clone_config(config or create_default_config())
    cfg.dataset = dataset
    cfg.n_envs = n_envs
    cfg.n_eval_episodes = n_eval_episodes
    cfg.seed = seed
    cfg.verbose = verbose
    cfg.mode = mode
    cfg.corruption_mode = corruption_mode
    cfg.k_negatives = k_negatives
    
    if verbose:
        print("=" * 70)
        print("Eval Corruptions Parity Test")
        print(f"Dataset: {cfg.dataset}, n_envs: {cfg.n_envs}, n_episodes: {cfg.n_eval_episodes}")
        print(f"Corruption mode: {cfg.corruption_mode}, K negatives: {cfg.k_negatives}")
        print("=" * 70)
    
    # Create aligned environments
    if verbose:
        print("\nCreating aligned environments...")
    env_data = create_aligned_environments(cfg)
    
    # Limit queries to n_eval_episodes
    queries_sb3 = env_data['queries_sb3'][:cfg.n_eval_episodes]
    queries_tensor = env_data['queries_tensor'][:cfg.n_eval_episodes]
    
    if verbose:
        print(f"Using {len(queries_sb3)} queries for evaluation")
    
    # Create SB3 environment and model
    if verbose:
        print("Creating SB3 eval environment...")
    sb3_ppo, sb3_env, sb3_im = create_sb3_eval_env(cfg, env_data['sb3'], queries_sb3)
    
    # Create tensor environment and model
    if verbose:
        print("Creating tensor eval environment...")
    tensor_ppo, tensor_env, tensor_im, engine = create_tensor_eval_env(
        cfg, env_data['tensor'], queries_tensor, n_eval_episodes=cfg.n_eval_episodes
    )
    
    # Create samplers
    if verbose:
        print("Creating negative samplers...")
    
    device = torch.device(cfg.device)
    # Map corruption_mode to corruption_scheme list
    if cfg.corruption_mode == 'both':
        corruption_scheme = ['head', 'tail']
    else:
        corruption_scheme = [cfg.corruption_mode]
    sb3_sampler = create_sb3_sampler(env_data['sb3']['dh'], sb3_im, device, cfg.seed, corruption_scheme=corruption_scheme)
    tensor_sampler = create_tensor_sampler(env_data['tensor']['dh'], tensor_im, env_data['sb3']['dh'], device, cfg.seed, corruption_scheme=corruption_scheme)

    # Debug: Compare generated negatives for the first query
    # Use a separate seed for debug to avoid polluting RNG state
    debug_rng_state = torch.get_rng_state()
    DEBUG_SEED = 99999
    torch.manual_seed(DEBUG_SEED)
    
    print("\n--- Debug: Negative Sampling Comparison ---")
    q0 = queries_sb3[0]
    print(f"Query 0: {q0}")
    
    # Handle -1 for debug sampling
    debug_k_negatives = cfg.k_negatives if cfg.k_negatives != -1 else None
    
    # SB3 Negatives - use get_negatives_from_states_separate
    sb3_head_negs, sb3_tail_negs = sb3_sampler.get_negatives_from_states_separate([[q0]], device=device, num_negs=debug_k_negatives)
    
    if corruption_mode == 'tail' or corruption_mode == 'both':
        if isinstance(sb3_tail_negs, list):
            print(f"SB3 Tail Negs (first 3): {[str(t) for t in sb3_tail_negs[:3]]}")
        else:
            print(f"SB3 Tail Negs (first 3): {sb3_tail_negs[:3] if sb3_tail_negs is not None else 'None'}")
    if corruption_mode == 'head' or corruption_mode == 'both':
        if isinstance(sb3_head_negs, list):
            print(f"SB3 Head Negs (first 3): {[str(t) for t in sb3_head_negs[:3]]}")
        else:
            print(f"SB3 Head Negs (first 3): {sb3_head_negs[:3] if sb3_head_negs is not None else 'None'}")

    # Tensor Negatives - use get_negatives_from_states_separate to match SB3
    tensor_im = env_data['tensor']['im']
    query_tensors = []
    for q in queries_tensor:
        query_atom = tensor_im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
        query_tensors.append(query_atom)
    queries_t_debug = torch.stack(query_tensors, dim=0)
    
    t_q0 = queries_t_debug[0].unsqueeze(0)
    # Use get_negatives_from_states_separate for tensor too
    t_head_negs, t_tail_negs = tensor_sampler.get_negatives_from_states_separate(
        t_q0, num_negatives=debug_k_negatives, device=device
    )
    
    if cfg.corruption_mode == 'tail' or cfg.corruption_mode == 'both':
        if t_tail_negs and len(t_tail_negs) > 0 and t_tail_negs[0].numel() > 0:
            print(f"Tensor Tail Negs (first 3): {t_tail_negs[0][:3].tolist()}")
        else:
            print(f"Tensor Tail Negs: empty")
    if cfg.corruption_mode == 'head' or cfg.corruption_mode == 'both':
        if t_head_negs and len(t_head_negs) > 0 and t_head_negs[0].numel() > 0:
            print(f"Tensor Head Negs (first 3): {t_head_negs[0][:3].tolist()}")
        else:
            print(f"Tensor Head Negs: empty")
    print("-------------------------------------------\n")
    
    # Restore RNG state after debug
    torch.set_rng_state(debug_rng_state)
    
    # Synchronize RNG state before SB3 eval_corruptions
    eval_seed = 12345  # Fixed seed for eval_corruptions parity
    torch.manual_seed(eval_seed)
    
    sb3_traces = []  # Store SB3 traces for comparison
    
    try:
        # SB3 API: eval_corruptions(model, env, data, sampler, n_corruptions, ...)
        sb3_metrics = sb3_eval_corruptions(
            model=sb3_ppo,
            env=sb3_env,
            data=queries_sb3,  # list of query objects
            sampler=sb3_sampler,
            n_corruptions=cfg.k_negatives,
            corruption_scheme=corruption_scheme,  # ['head'] or ['tail'] or ['head', 'tail']
            verbose=0,
            return_traces=True,  # Enable trace collection
        )
        
        # Extract traces
        sb3_traces = sb3_metrics.get('traces', [])
        
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
            print(f"  SB3 traces collected: {len(sb3_traces)}")
    except Exception as e:
        if verbose:
            print(f"  SB3 eval_corruptions error: {e}")
        import traceback
        traceback.print_exc()
        return results
    
    # Run tensor eval_corruptions
    if verbose:
        print("\nRunning tensor eval_corruptions...")
    
    # Synchronize RNG state before tensor eval_corruptions (SAME seed as SB3)
    torch.manual_seed(eval_seed)
    
    tensor_traces = []  # Store tensor traces for comparison
    
    try:
        # Tensor API: eval_corruptions(actor, env, queries, sampler, ...)
        # Convert queries to tensor format - just the query atom, not padded
        tensor_im = env_data['tensor']['im']
        query_tensors = []
        for q in queries_tensor:
            query_atom = tensor_im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
            query_tensors.append(query_atom)
        queries_t = torch.stack(query_tensors, dim=0)  # shape (B, 3)
        
        # Handle -1 as None (all corruptions) for Tensor implementation
        tensor_k_negatives = cfg.k_negatives if cfg.k_negatives != -1 else None
        
        tensor_metrics = tensor_eval_corruptions(
            actor=tensor_ppo.policy,
            env=tensor_env,
            queries=queries_t,
            sampler=tensor_sampler,
            n_corruptions=tensor_k_negatives,
            corruption_modes=tuple(corruption_scheme),  # ('head',) or ('tail',) or ('head', 'tail')
            verbose=False,
            return_traces=True,  # Enable trace collection
        )
        
        # Extract traces
        tensor_traces = tensor_metrics.get('traces', [])
        
        results.tensor_mrr = tensor_metrics.get('MRR', 0.0)
        results.tensor_hits1 = tensor_metrics.get('Hits@1', 0.0)
        results.tensor_hits3 = tensor_metrics.get('Hits@3', 0.0)
        results.tensor_hits10 = tensor_metrics.get('Hits@10', 0.0)
        
        if verbose:
            print(f"  Tensor MRR: {results.tensor_mrr:.4f}")
            print(f"  Tensor Hits@1: {results.tensor_hits1:.4f}")
            print(f"  Tensor Hits@3: {results.tensor_hits3:.4f}")
            print(f"  Tensor Hits@10: {results.tensor_hits10:.4f}")
            print(f"  Tensor traces collected: {len(tensor_traces)}")
    except Exception as e:
        if verbose:
            print(f"  Tensor eval_corruptions error: {e}")
        import traceback
        traceback.print_exc()
        return results
    
    # Compare traces if results don't match
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
    
    # Compare traces if there's a mismatch and we have traces
    if not results.eval_corruptions_success and sb3_traces and tensor_traces:
        if verbose:
            compare_traces(sb3_traces, tensor_traces, max_traces=30, verbose=True)
    
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
    
    Parametrized test covering various configurations:
    - dataset: Dataset name (family, countries_s3, etc.)
    - corruption_mode: 'head', 'tail', or 'both'
    - n_queries: Number of test queries to evaluate
    - k_negatives: Number of corruptions per query (None for all possible corruptions)
    """
    
    @pytest.mark.parametrize("dataset,corruption_mode,n_queries,k_negatives", [
        # Family dataset: head and tail corruption, 10 test queries, 10 corruptions
        ("family", "both", 10, 10),
        # Countries_s3 dataset: tail corruption, all test queries (24), 3 corruptions
        ("countries_s3", "tail", 24, 3),
        # Countries_s3 dataset: tail corruption, all test queries (24), all corruptions (None)
        ("countries_s3", "tail", 24, None),
    ])
    def test_eval_corruptions(self, dataset, corruption_mode, n_queries, k_negatives):
        """General eval_corruptions parity test with configurable parameters."""
        # Convert None to -1 for the internal API (which uses -1 to signal "all corruptions")
        k_neg_internal = -1 if k_negatives is None else k_negatives
        
        results = run_eval_corruptions_parity(
            dataset=dataset,
            n_envs=2,
            n_eval_episodes=n_queries,
            seed=42,
            verbose=True,
            mode="test",
            corruption_mode=corruption_mode,
            k_negatives=k_neg_internal,
        )
        
        k_neg_str = "all" if k_negatives is None else str(k_negatives)
        assert results.eval_corruptions_success, \
            f"eval_corruptions parity failed for {dataset} {corruption_mode} ({n_queries} queries, {k_neg_str} corruptions): " \
            f"MRR SB3={results.sb3_mrr:.4f}, Tensor={results.tensor_mrr:.4f}"


# ============================================================
# CLI Interface
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluation Parity Tests")
    parser.add_argument("--dataset", type=str, default="countries_s3",
                       help="Dataset name (default: countries_s3)")
    parser.add_argument("--n-envs", type=int, default=20,
                       help="Number of environments (default: 2)")
    parser.add_argument("--n-eval-episodes", type=int, default=24,
                       help="Number of evaluation episodes (default: 24)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--mode", type=str, default="test",
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
