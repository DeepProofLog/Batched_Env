"""
Learn Compiled Parity Tests.

Tests verifying that the tensor PPO's learn function produces the SAME
behavior as the optimized PPO's learn function, including:
- Rollout collection step-by-step
- Training metrics (policy loss, value loss, entropy, clip fraction)
- Buffer contents after rollout collection
- Policy parameter updates after training

Uses deterministic action selection and aligned environments to ensure
reproducible comparisons.

This test runs the optimized PPO in eager mode (for parity testing) but
ensures the code paths are compatible with torch.compile in production.

Usage:
    python tests/parity/test_learn_compiled_parity.py --dataset countries_s3 --n-envs 4 --n-steps 20
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import numpy as np
import pytest
from types import SimpleNamespace

# Setup paths
ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import seeding utilities
from tensor.utils.tensor_seeding import ParityTestSeeder, ParityTestConfig, ParityTolerances, seed_all
from tests.test_utils.parity_config import ParityConfig, TOLERANCE, create_parser, config_from_args

# Tensor imports
from data_handler import DataHandler
from index_manager import IndexManager
from tensor.tensor_unification import UnificationEngine
from unification import UnificationEngineVectorized
from tensor.tensor_env import BatchedEnv
from env import EvalEnvOptimized, EvalObs, EvalState
from tensor.tensor_embeddings import EmbedderLearnable as TensorEmbedder
from tensor.tensor_model import ActorCriticPolicy as TensorPolicy
from tensor.tensor_ppo import PPO as TensorPPO
from ppo import PPOOptimized
from tensor.tensor_rollout import RolloutBuffer as TensorRolloutBuffer
from rollout import RolloutBufferOptimized


# ============================================================================
# Default Configuration - using shared parity_config
# ============================================================================

def get_default_config() -> ParityConfig:
    """Get default config from shared parity_config, with skip_unary_actions=False."""
    return ParityConfig(skip_unary_actions=False)


@dataclass
class LearnCompiledParityResults:
    """Results container for learn compiled parity comparison."""
    # Rollout comparison
    rollout_traces_match: bool = False
    rollout_n_mismatches: int = 0
    rollout_mismatches: List[Dict] = field(default_factory=list)
    
    # Buffer comparison
    buffer_obs_match: bool = False
    buffer_actions_match: bool = False
    buffer_rewards_match: bool = False
    buffer_values_match: bool = False
    buffer_log_probs_match: bool = False
    buffer_advantages_match: bool = False
    buffer_returns_match: bool = False
    
    # Training metrics comparison
    tensor_train_metrics: Dict[str, float] = field(default_factory=dict)
    optimized_train_metrics: Dict[str, float] = field(default_factory=dict)
    metrics_match: bool = False
    
    # Training traces comparison
    tensor_train_traces: List[Dict] = field(default_factory=list)
    optimized_train_traces: List[Dict] = field(default_factory=list)
    train_traces_match: bool = False
    train_traces_n_mismatches: int = 0
    
    # Weight parity
    weights_match: bool = False
    weights_max_diff: float = 0.0
    
    # Done/state parity
    dones_match: bool = False
    states_match: bool = False
    
    # Overall
    success: bool = False


def create_aligned_environments(config: SimpleNamespace):
    """Create tensor and optimized environments with aligned data."""
    base_path = config.data_path
    device = torch.device(config.device)
    padding_atoms = config.padding_atoms
    padding_states = config.padding_states
    
    # Create data handler
    dh = DataHandler(
        dataset_name=config.dataset,
        base_path=base_path,
        train_file=config.train_file,
        valid_file=config.valid_file,
        test_file=config.test_file,
        rules_file=config.rules_file,
        facts_file=config.facts_file,
        train_depth=config.train_depth,
    )
    
    # Create index manager
    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=config.max_total_vars,
        padding_atoms=padding_atoms,
        max_arity=dh.max_arity,
        device=device,
        rules=dh.rules,
    )
    dh.materialize_indices(im=im, device=device)
    
    stringifier_params = {
        'verbose': 0,
        'idx2predicate': im.idx2predicate,
        'idx2constant': im.idx2constant,
        'idx2template_var': im.idx2template_var,
        'padding_idx': im.padding_idx,
        'n_constants': im.constant_no
    }
    
    # Create base unification engine
    base_engine = UnificationEngine.from_index_manager(
        im, take_ownership=True,
        stringifier_params=stringifier_params,
        end_pred_idx=im.end_pred_idx,
        end_proof_action=config.end_proof_action,
        max_derived_per_state=padding_states,
    )
    base_engine.index_manager = im
    
    # Create vectorized engine for optimized env (parity_mode=True for determinism)
    vec_engine = UnificationEngineVectorized.from_base_engine(
        base_engine,
        max_fact_pairs=None,
        max_rule_pairs=None,
        padding_atoms=padding_atoms,
        parity_mode=True,  # Enable parity mode for determinism
    )
    
    queries = dh.train_queries
    
    return {
        'dh': dh,
        'im': im,
        'base_engine': base_engine,
        'vec_engine': vec_engine,
        'queries': queries,
        'padding_atoms': padding_atoms,
        'padding_states': padding_states,
    }


def create_tensor_ppo(config: SimpleNamespace, env_data: Dict, queries: List):
    """Create tensor PPO with BatchedEnv."""
    device = torch.device(config.device)
    padding_atoms = env_data.get('padding_atoms', config.padding_atoms)
    padding_states = env_data.get('padding_states', config.padding_states)
    
    dh = env_data['dh']
    im = env_data['im']
    engine = env_data['base_engine']
    
    # Convert queries to tensor format
    query_tensors = []
    for q in queries[:config.n_envs]:
        query_atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
        query_padded = torch.full((padding_atoms, 3), im.padding_idx, dtype=torch.long, device=device)
        query_padded[0] = query_atom
        query_tensors.append(query_padded)
    
    queries_tensor = torch.stack(query_tensors, dim=0)
    
    env = BatchedEnv(
        batch_size=config.n_envs,
        queries=queries_tensor,
        labels=torch.ones(len(query_tensors), dtype=torch.long, device=device),
        query_depths=torch.ones(len(query_tensors), dtype=torch.long, device=device),
        unification_engine=engine,
        mode='train',
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
    
    # Create embedder with fixed seed
    torch.manual_seed(config.seed)
    embedder = TensorEmbedder(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=config.n_vars_for_embedder,
        max_arity=dh.max_arity,
        padding_atoms=padding_atoms,
        atom_embedder='transe',
        state_embedder='sum',
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
        hidden_dim=256,
        num_layers=8,
        dropout_prob=0.0,
        device=device,
        parity=True,  # Use SB3-identical initialization
        use_l2_norm=False,
        sqrt_scale=True,
        temperature=None,
    ).to(device)
    
    # Create PPO with fixed seed
    torch.manual_seed(config.seed)
    ppo = TensorPPO(
        policy=policy,
        env=env,
        n_steps=config.n_steps,
        learning_rate=config.learning_rate,
        n_epochs=config.n_epochs,
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
        use_compile=False,  # Disable compile for parity testing
        parity=True,  # Enable parity mode for consistent shuffling
    )
    
    return ppo, env, im, engine, queries_tensor


def create_optimized_ppo(config: SimpleNamespace, env_data: Dict, queries: List, 
                         tensor_policy_state: dict):
    """Create optimized PPO with EvalEnvOptimized, using same initial weights."""
    device = torch.device(config.device)
    padding_atoms = env_data.get('padding_atoms', config.padding_atoms)
    padding_states = env_data.get('padding_states', config.padding_states)
    
    dh = env_data['dh']
    im = env_data['im']
    vec_engine = env_data['vec_engine']
    
    # Create optimized environment
    env = EvalEnvOptimized(
        vec_engine=vec_engine,
        batch_size=config.n_envs,
        padding_atoms=padding_atoms,
        padding_states=padding_states,
        max_depth=config.max_depth,
        end_proof_action=config.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
        memory_pruning=config.memory_pruning,
    )
    
    # Create embedder with SAME fixed seed
    torch.manual_seed(config.seed)
    embedder = TensorEmbedder(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=config.n_vars_for_embedder,
        max_arity=dh.max_arity,
        padding_atoms=padding_atoms,
        atom_embedder='transe',
        state_embedder='sum',
        constant_embedding_size=config.embed_dim,
        predicate_embedding_size=config.embed_dim,
        atom_embedding_size=config.embed_dim,
        device=str(device),
    )
    embedder.embed_dim = config.embed_dim
    
    # Create policy with SAME fixed seed
    action_size = padding_states
    torch.manual_seed(config.seed)
    policy = TensorPolicy(
        embedder=embedder,
        embed_dim=config.embed_dim,
        action_dim=action_size,
        hidden_dim=256,
        num_layers=8,
        dropout_prob=0.0,
        device=device,
        parity=True,
        use_l2_norm=False,
        sqrt_scale=True,
        temperature=None,
    ).to(device)
    
    # Load tensor policy weights for exact parity
    policy.load_state_dict(tensor_policy_state)
    
    # Create PPO with fixed seed
    torch.manual_seed(config.seed)
    ppo = PPOOptimized(
        policy=policy,
        env=env,
        n_steps=config.n_steps,
        learning_rate=config.learning_rate,
        n_epochs=config.n_epochs,
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
        parity=True,  # Enable parity mode
    )
    
    # Convert queries to tensor format [N, 3]
    # IMPORTANT: Only use the first n_envs queries to match tensor env's query cycling
    query_atoms = []
    for q in queries[:config.n_envs]:
        query_atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
        query_atoms.append(query_atom)
    query_pool = torch.stack(query_atoms, dim=0).to(device)
    
    return ppo, env, im, query_pool


def collect_tensor_rollout_with_traces(
    ppo: TensorPPO,
    n_steps: int,
    return_train_traces: bool = False,
    seeder: ParityTestSeeder = None,
) -> Tuple[List[Dict], TensorRolloutBuffer, Dict[str, float]]:
    """
    Collect rollouts using tensor PPO with trace collection, then perform training.
    Returns traces, buffer, and training metrics.
    """
    device = ppo.device
    
    # Initialize
    current_obs = ppo.env.reset()
    episode_starts = torch.ones(ppo.n_envs, dtype=torch.float32, device=device)
    current_episode_reward = torch.zeros(ppo.n_envs, dtype=torch.float32, device=device)
    current_episode_length = torch.zeros(ppo.n_envs, dtype=torch.long, device=device)
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
        iteration=1,
        return_traces=True,
    )
    
    # Result has traces as 6th element
    traces = result[5]
    
    # Seed RNG before training for shuffling parity
    if seeder is not None:
        seeder.seed_for_training()
    else:
        seed_all(123)
    
    # Perform training and get metrics
    train_metrics = ppo.train(return_traces=return_train_traces)
    
    return traces, ppo.rollout_buffer, train_metrics


def collect_optimized_rollout_with_traces(
    ppo: PPOOptimized,
    im: IndexManager,
    query_pool: torch.Tensor,
    n_steps: int,
    return_train_traces: bool = False,
    seeder: ParityTestSeeder = None,
) -> Tuple[List[Dict], RolloutBufferOptimized, Dict[str, float]]:
    """
    Collect rollouts using optimized PPO with trace collection, then perform training.
    Returns traces, buffer, and training metrics.
    """
    device = ppo.device
    n_envs = ppo.n_envs
    
    # Setup policy for step_with_policy (eager mode for parity tests)
    from model import create_policy_logits_fn
    ppo.env._policy_logits_fn = create_policy_logits_fn(ppo.policy)
    ppo.env._compile_deterministic = False  # Training uses sampling
    ppo.env._compiled = False
    
    # Set query pool in environment for round-robin cycling (new API)
    ppo.env.set_queries(query_pool)
    
    # Initialize state with first n_envs queries
    init_queries = query_pool[:n_envs]
    state = ppo.env.init_state_from_queries(init_queries)
    
    # Initialize per-env pointers to match tensor env's round-robin
    # Must be done after set_queries since it initializes pointers
    ppo.env._per_env_ptrs = torch.arange(n_envs, device=device) + 1
    
    # Create initial observation
    action_mask = torch.arange(ppo.padding_states, device=device).unsqueeze(0) < state.derived_counts.unsqueeze(1)
    obs = EvalObs(
        sub_index=state.current_states.unsqueeze(1),
        derived_sub_indices=state.derived_states,
        action_mask=action_mask,
    )
    
    episode_starts = torch.ones(n_envs, dtype=torch.float32, device=device)
    current_episode_reward = torch.zeros(n_envs, dtype=torch.float32, device=device)
    current_episode_length = torch.zeros(n_envs, dtype=torch.long, device=device)
    episode_rewards = []
    episode_lengths = []
    
    # Collect rollouts with traces (query cycling now handled internally)
    result = ppo.collect_rollouts(
        current_state=state,
        current_obs=obs,
        episode_starts=episode_starts,
        current_episode_reward=current_episode_reward,
        current_episode_length=current_episode_length,
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        iteration=1,
        return_traces=True,
    )

    
    # Result has traces as last element (7 total elements: state, obs, episode_starts, reward, length, n_steps, traces)
    traces = result[-1]

    
    # Seed RNG before training for shuffling parity
    if seeder is not None:
        seeder.seed_for_training()
    else:
        seed_all(123)
    
    # Perform training and get metrics
    train_metrics = ppo.train(return_traces=return_train_traces)
    
    return traces, ppo.rollout_buffer, train_metrics


def compare_traces(tensor_traces: List[Dict], optimized_traces: List[Dict], 
                   n_steps: int, n_envs: int) -> Tuple[bool, int, List[Dict]]:
    """Compare traces from tensor and optimized rollouts."""
    mismatches = []
    
    # Create lookup dicts by (step, env)
    tensor_lookup = {(t['step'], t['env']): t for t in tensor_traces}
    optimized_lookup = {(t['step'], t['env']): t for t in optimized_traces}
    
    total_comparisons = 0
    matches = 0
    
    for step in range(n_steps):
        for env_idx in range(n_envs):
            tensor_trace = tensor_lookup.get((step, env_idx))
            optimized_trace = optimized_lookup.get((step, env_idx))
            
            if tensor_trace is None or optimized_trace is None:
                continue
            
            total_comparisons += 1
            
            # Compare key fields
            # Compare state observation - compare the sub_index arrays
            tensor_sub = tensor_trace['state_obs']['sub_index']
            optimized_sub = optimized_trace['state_obs']['sub_index']
            state_match = np.array_equal(
                np.array(tensor_sub).flatten()[:20],
                np.array(optimized_sub).flatten()[:20]
            )
            reward_match = abs(tensor_trace['reward'] - optimized_trace['reward']) < 1e-5
            done_match = tensor_trace['done'] == optimized_trace['done']
            action_match = tensor_trace['action'] == optimized_trace['action']
            
            if state_match and reward_match and done_match and action_match:
                matches += 1
            else:
                mismatches.append({
                    'step': step,
                    'env_idx': env_idx,
                    'state_match': state_match,
                    'reward_match': reward_match,
                    'done_match': done_match,
                    'action_match': action_match,
                    'tensor': tensor_trace,
                    'optimized': optimized_trace,
                })
    
    all_match = (matches == total_comparisons)
    n_mismatches = total_comparisons - matches
    
    return all_match, n_mismatches, mismatches


def compare_buffers(tensor_buffer, optimized_buffer, 
                    n_steps: int, n_envs: int, atol: float = 1e-4) -> Dict[str, bool]:
    """Compare rollout buffer contents between tensor and optimized implementations."""
    results = {}
    
    # Get tensor buffer data
    tensor_rewards = tensor_buffer.rewards.cpu().numpy()
    tensor_values = tensor_buffer.values.cpu().numpy()
    tensor_log_probs = tensor_buffer.log_probs.cpu().numpy()
    tensor_advantages = tensor_buffer.advantages.cpu().numpy()
    tensor_returns = tensor_buffer.returns.cpu().numpy()
    tensor_actions = tensor_buffer.actions.cpu().numpy()
    
    # Get optimized buffer data
    optimized_rewards = optimized_buffer.rewards.cpu().numpy()
    optimized_values = optimized_buffer.values.cpu().numpy()
    optimized_log_probs = optimized_buffer.log_probs.cpu().numpy()
    optimized_advantages = optimized_buffer.advantages.cpu().numpy()
    optimized_returns = optimized_buffer.returns.cpu().numpy()
    optimized_actions = optimized_buffer.actions.cpu().numpy()
    
    # Compare each field
    results['rewards_match'] = np.allclose(tensor_rewards.flatten(), optimized_rewards.flatten(), atol=atol)
    results['values_match'] = np.allclose(tensor_values.flatten(), optimized_values.flatten(), atol=0.1)
    results['log_probs_match'] = np.allclose(tensor_log_probs.flatten(), optimized_log_probs.flatten(), atol=0.1)
    results['advantages_match'] = np.allclose(tensor_advantages.flatten(), optimized_advantages.flatten(), atol=0.1)
    results['returns_match'] = np.allclose(tensor_returns.flatten(), optimized_returns.flatten(), atol=0.1)
    results['actions_match'] = np.array_equal(tensor_actions.flatten().astype(int), optimized_actions.flatten().astype(int))
    
    return results


def compare_train_metrics(tensor_metrics: Dict[str, float], optimized_metrics: Dict[str, float], 
                         rtol: float = 0.01) -> Tuple[bool, Dict[str, Tuple[float, float, bool]]]:
    """Compare training metrics between tensor and optimized implementations."""
    comparison = {}
    all_match = True
    
    for key in ['policy_loss', 'value_loss', 'entropy', 'clip_fraction']:
        tensor_val = tensor_metrics.get(key, 0.0)
        optimized_val = optimized_metrics.get(key, 0.0)
        
        # Use numpy-style isclose
        max_abs = max(abs(tensor_val), abs(optimized_val))
        if max_abs < 1e-8:
            matches = True
        else:
            abs_diff = abs(tensor_val - optimized_val)
            matches = abs_diff <= rtol + rtol * max_abs
        
        comparison[key] = (tensor_val, optimized_val, matches)
        if not matches:
            all_match = False
    
    return all_match, comparison


def compare_train_traces(
    tensor_traces: List[Dict], 
    optimized_traces: List[Dict],
    atol: float = 0.01,
    verbose: bool = False
) -> Tuple[bool, int, List[Dict]]:
    """Compare per-batch training traces between tensor and optimized implementations."""
    if not tensor_traces or not optimized_traces:
        return len(tensor_traces) == len(optimized_traces), 0, []
    
    mismatches = []
    n_comparisons = min(len(tensor_traces), len(optimized_traces))
    
    fields_to_compare = {
        'epoch': 0,
        'batch_size': 0,
        'policy_loss': atol,
        'value_loss': atol,
        'entropy_loss': atol,
        'clip_fraction': atol,
    }
    
    for i in range(n_comparisons):
        tensor_trace = tensor_traces[i]
        optimized_trace = optimized_traces[i]
        
        trace_mismatches = {}
        for field, tolerance in fields_to_compare.items():
            tensor_val = tensor_trace.get(field)
            optimized_val = optimized_trace.get(field)
            
            if tensor_val is None or optimized_val is None:
                continue
                
            if tolerance == 0:
                if tensor_val != optimized_val:
                    trace_mismatches[field] = (tensor_val, optimized_val)
            else:
                if abs(float(tensor_val) - float(optimized_val)) > tolerance:
                    trace_mismatches[field] = (tensor_val, optimized_val)
        
        if trace_mismatches:
            mismatches.append({
                'batch_idx': i,
                'mismatches': trace_mismatches,
                'tensor_trace': tensor_trace,
                'optimized_trace': optimized_trace,
            })
            
            if verbose:
                print(f"Batch {i} mismatches:")
                for field, (t_val, o_val) in trace_mismatches.items():
                    print(f"  {field}: Tensor={t_val}, Optimized={o_val}")
    
    all_match = len(mismatches) == 0 and len(tensor_traces) == len(optimized_traces)
    return all_match, len(mismatches), mismatches


def compare_weights(tensor_policy, optimized_policy, atol: float = 1e-3) -> Tuple[bool, float]:
    """Compare policy weights between tensor and optimized implementations."""
    max_diff = 0.0
    
    tensor_state = tensor_policy.state_dict()
    optimized_state = optimized_policy.state_dict()
    
    for key in tensor_state:
        if key not in optimized_state:
            continue
        
        tensor_param = tensor_state[key].cpu()
        optimized_param = optimized_state[key].cpu()
        
        diff = (tensor_param - optimized_param).abs().max().item()
        max_diff = max(max_diff, diff)
    
    weights_match = max_diff <= atol
    return weights_match, max_diff


def run_learn_compiled_parity_test(
    dataset: str,
    n_envs: int,
    n_steps: int,
    n_epochs: int = 1,
    seed: int = 42,
    verbose: bool = True,
    compare_training_traces: bool = True,
    config: Optional[ParityConfig] = None,
) -> LearnCompiledParityResults:
    """Run the full learn compiled parity test comparing tensor and optimized PPO."""
    results = LearnCompiledParityResults()
    base_cfg = config or get_default_config()
    
    # Note: PPOOptimized will auto-adjust batch_size if > buffer_size
    cfg = base_cfg.update(
        data_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data'),
        dataset=dataset,
        n_envs=n_envs,
        n_steps=n_steps,
        n_epochs=n_epochs,
        seed=seed,
        verbose=verbose,
        skip_unary_actions=False,  # Required for parity
    ).to_namespace()
    
    # Initialize seeder
    seeder = ParityTestSeeder(seed=seed)
    
    if verbose:
        print("=" * 70)
        print(f"Learn Compiled Parity Test")
        print(f"Dataset: {cfg.dataset}, n_envs: {cfg.n_envs}, n_steps: {cfg.n_steps}, n_epochs: {cfg.n_epochs}")
        print(f"skip_unary_actions: {cfg.skip_unary_actions}")
        print(f"Using ParityTestSeeder with base seed: {seed}")
        print("=" * 70)
    
    # Create aligned environments
    if verbose:
        print("\nCreating aligned environments...")
    env_data = create_aligned_environments(cfg)
    
    # Create tensor PPO
    if verbose:
        print("Creating tensor PPO...")
    tensor_ppo, tensor_env, tensor_im, engine, queries_tensor = create_tensor_ppo(
        cfg, env_data, env_data['queries']
    )
    
    # Save tensor policy initial state for weight comparison
    tensor_policy_initial_state = {k: v.clone() for k, v in tensor_ppo.policy.state_dict().items()}
    
    # Create optimized PPO with same initial weights
    if verbose:
        print("Creating optimized PPO...")
    optimized_ppo, optimized_env, optimized_im, query_pool = create_optimized_ppo(
        cfg, env_data, env_data['queries'], tensor_policy_initial_state
    )
    
    # Collect rollouts and train for tensor
    if verbose:
        print("\nCollecting tensor rollouts and training...")
    seeder.seed_for_rollout_collection()
    tensor_traces, tensor_buffer, tensor_train_metrics = collect_tensor_rollout_with_traces(
        tensor_ppo, cfg.n_steps, return_train_traces=compare_training_traces, seeder=seeder
    )
    results.tensor_train_metrics = tensor_train_metrics
    if compare_training_traces and "traces" in tensor_train_metrics:
        results.tensor_train_traces = tensor_train_metrics["traces"]
    
    # Collect rollouts and train for optimized (same seed for identical sampling)
    if verbose:
        print("Collecting optimized rollouts and training...")
    seeder.seed_for_rollout_collection()
    optimized_traces, optimized_buffer, optimized_train_metrics = collect_optimized_rollout_with_traces(
        optimized_ppo, optimized_im, query_pool, cfg.n_steps, 
        return_train_traces=compare_training_traces, seeder=seeder
    )
    results.optimized_train_metrics = optimized_train_metrics
    if compare_training_traces and "traces" in optimized_train_metrics:
        results.optimized_train_traces = optimized_train_metrics["traces"]
    
    # Compare traces
    if verbose:
        print("\n--- Trace Comparison ---")
    traces_match, n_mismatches, mismatches = compare_traces(tensor_traces, optimized_traces, cfg.n_steps, cfg.n_envs)
    results.rollout_traces_match = traces_match
    results.rollout_n_mismatches = n_mismatches
    results.rollout_mismatches = mismatches[:10]
    
    if verbose:
        print(f"Traces match: {traces_match} ({n_mismatches} mismatches)")
    
    # Compare buffers
    if verbose:
        print("\n--- Buffer Comparison ---")
    buffer_results = compare_buffers(tensor_buffer, optimized_buffer, cfg.n_steps, cfg.n_envs, atol=TOLERANCE)
    results.buffer_rewards_match = buffer_results['rewards_match']
    results.buffer_values_match = buffer_results['values_match']
    results.buffer_log_probs_match = buffer_results['log_probs_match']
    results.buffer_advantages_match = buffer_results['advantages_match']
    results.buffer_returns_match = buffer_results['returns_match']
    results.buffer_actions_match = buffer_results['actions_match']
    
    if verbose:
        for key, match in buffer_results.items():
            print(f"  {key}: {match}")
    
    # Compare training metrics
    if verbose:
        print("\n--- Training Metrics Comparison ---")
    metrics_match, metrics_comparison = compare_train_metrics(tensor_train_metrics, optimized_train_metrics, rtol=TOLERANCE)
    results.metrics_match = metrics_match
    
    if verbose:
        for key, (t_val, o_val, match) in metrics_comparison.items():
            status = "✓" if match else "✗"
            print(f"  {status} {key}: Tensor={t_val:.6f}, Optimized={o_val:.6f}")
    
    # Compare training traces
    if compare_training_traces and results.tensor_train_traces and results.optimized_train_traces:
        if verbose:
            print("\n--- Training Traces Comparison (Per-Batch) ---")
        train_traces_match, train_traces_n_mismatches, train_trace_mismatches = compare_train_traces(
            results.tensor_train_traces, results.optimized_train_traces, atol=TOLERANCE, verbose=verbose
        )
        results.train_traces_match = train_traces_match
        results.train_traces_n_mismatches = train_traces_n_mismatches
        
        if verbose:
            print(f"  Training traces match: {train_traces_match} ({train_traces_n_mismatches} batch mismatches)")
            print(f"  Tensor batches: {len(results.tensor_train_traces)}, Optimized batches: {len(results.optimized_train_traces)}")
    else:
        results.train_traces_match = True
    
    # Compare weights after training
    if verbose:
        print("\n--- Weight Comparison (After Training) ---")
    weights_match, max_diff = compare_weights(tensor_ppo.policy, optimized_ppo.policy, atol=0.01)
    results.weights_match = weights_match
    results.weights_max_diff = max_diff
    
    if verbose:
        status = "✓" if weights_match else "✗"
        print(f"  {status} Weights match (max diff: {max_diff:.6f})")
    
    # Determine overall success
    results.success = (
        traces_match and
        buffer_results['actions_match'] and
        buffer_results['rewards_match'] and
        buffer_results['values_match'] and
        buffer_results['log_probs_match'] and
        buffer_results['advantages_match'] and
        buffer_results['returns_match'] and
        metrics_match and
        results.train_traces_match and
        weights_match
    )
    
    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("Results Summary:")
        print(f"  Rollout traces match: {results.rollout_traces_match}")
        print(f"  Buffer actions match: {results.buffer_actions_match}")
        print(f"  Buffer rewards match: {results.buffer_rewards_match}")
        print(f"  Buffer values match: {results.buffer_values_match}")
        print(f"  Training metrics match: {results.metrics_match}")
        print(f"  Training traces match: {results.train_traces_match}")
        print(f"  Weights match: {results.weights_match} (max diff: {results.weights_max_diff:.6f})")
        
        if results.success:
            print("\n✓ LEARN COMPILED PARITY TEST PASSED")
        else:
            print(f"\n✗ LEARN COMPILED PARITY TEST FAILED")
            if not traces_match:
                print(f"  - {n_mismatches} trace mismatches")
            if not buffer_results['actions_match']:
                print("  - Buffer actions don't match")
            if not buffer_results['rewards_match']:
                print("  - Buffer rewards don't match")
            if not results.train_traces_match:
                print(f"  - {results.train_traces_n_mismatches} training trace batch mismatches")
            if not weights_match:
                print(f"  - Weights don't match (max diff: {max_diff:.6f})")
        
        print("=" * 70)
    
    return results


# ============================================================
# Test Functions (callable from __main__)
# ============================================================

@pytest.mark.parametrize("n_envs,n_steps", [(1, 10), (4, 10)])
def test_learn_compiled_parity(n_envs: int, n_steps: int) -> bool:
    """Test learn parity between tensor and optimized implementations."""
    results = run_learn_compiled_parity_test(
        dataset="countries_s3",
        n_envs=n_envs,
        n_steps=n_steps,
        n_epochs=1,
        seed=42,
        verbose=True,
    )
    if not results.success:
        print(f"Learn compiled parity test failed for n_envs={n_envs}, n_steps={n_steps}")
        return False
    return True


@pytest.mark.parametrize("n_epochs", [1, 2])
def test_learn_compiled_parity_multiple_epochs(n_epochs: int) -> bool:
    """Test learn parity with multiple training epochs."""
    results = run_learn_compiled_parity_test(
        dataset="countries_s3",
        n_envs=2,
        n_steps=10,
        n_epochs=n_epochs,
        seed=42,
        verbose=True,
    )
    if not results.rollout_traces_match:
        print(f"Rollout traces don't match for n_epochs={n_epochs}")
        return False
    if not results.buffer_actions_match:
        print(f"Buffer actions don't match for n_epochs={n_epochs}")
        return False
    return True


def run_all_tests(args) -> bool:
    """Run all learn compiled parity tests."""
    all_passed = True
    verbose = args.verbose and not getattr(args, 'quiet', False)
    
    # Test 1: Basic parity tests with different n_envs/n_steps
    test_params = [(1, 10), (4, 10)]
    if args.n_envs is not None and args.n_steps is not None:
        test_params = [(args.n_envs, args.n_steps)]
    
    for n_envs, n_steps in test_params:
        print(f"\n{'='*70}")
        print(f"Running test_learn_compiled_parity[{n_envs}-{n_steps}]")
        print(f"{'='*70}")
        
        if test_learn_compiled_parity(n_envs, n_steps):
            print(f"✓ PASSED: test_learn_compiled_parity[{n_envs}-{n_steps}]")
        else:
            print(f"✗ FAILED: test_learn_compiled_parity[{n_envs}-{n_steps}]")
            all_passed = False
    
    # Test 2: Multiple epochs tests
    for n_epochs in [1, 2]:
        print(f"\n{'='*70}")
        print(f"Running test_learn_compiled_parity_multiple_epochs[{n_epochs}]")
        print(f"{'='*70}")
        
        if test_learn_compiled_parity_multiple_epochs(n_epochs):
            print(f"✓ PASSED: test_learn_compiled_parity_multiple_epochs[{n_epochs}]")
        else:
            print(f"✗ FAILED: test_learn_compiled_parity_multiple_epochs[{n_epochs}]")
            all_passed = False
    
    return all_passed


# ============================================================
# CLI Interface
# ============================================================

if __name__ == "__main__":
    parser = create_parser(description="Learn Compiled Parity Test")
    parser.add_argument("--run-all", action="store_true", default=True,
                        help="Run all test functions (default: True)")
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("LEARN COMPILED PARITY TESTS")
    print(f"Tolerance: {TOLERANCE}")
    print(f"{'='*70}")
    
    all_passed = run_all_tests(args)
    
    print(f"\n{'='*70}")
    if all_passed:
        print("All learn compiled parity tests PASSED")
    else:
        print("Some learn compiled parity tests FAILED")
    print(f"{'='*70}")
    
    sys.exit(0 if all_passed else 1)
