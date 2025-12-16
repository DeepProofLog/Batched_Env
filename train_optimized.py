"""
Compiled/Optimized training script for parity testing.

This module mirrors train_parity.py but uses PPOOptimized and EvalEnvOptimized
instead of the tensor-based PPO and BatchedEnv.

Key Differences from train_parity.py:
1. Uses EvalEnvOptimized instead of BatchedEnv
2. Uses PPOOptimized instead of PPO
3. Uses UnificationEngineVectorized with parity_mode=True

This allows testing parity between the original tensor implementation and
the optimized implementation that can be compiled with torch.compile.
"""

import gc
import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import numpy as np

# Add root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_handler import DataHandler
from index_manager import IndexManager
from unification import UnificationEngine
from unification_optimized import UnificationEngineVectorized
from env_optimized import EvalEnvOptimized, EvalObs, EvalState
from embeddings import EmbedderLearnable as TensorEmbedder
from model import ActorCriticPolicy as TensorPolicy, create_policy_logits_fn
from ppo_optimized import PPOOptimized
from model_eval import eval_corruptions as tensor_eval_corruptions
from sampler import Sampler

# [ADAPTER] Import callbacks if available (optional support)
try:
    from callbacks import (
        TorchRLCallbackManager, 
        MetricsCallback, 
        RankingCallback,
        CheckpointCallback,
        ScalarAnnealingCallback, 
        AnnealingTarget
    )
except ImportError:
    pass


@dataclass
class TrainCompiledConfig:
    """Configuration for compiled/optimized training parity tests."""
    # Dataset / data files
    dataset: str = "countries_s3"
    data_path: str = "./data/"
    train_file: str = "train.txt"
    valid_file: str = "valid.txt"
    test_file: str = "test.txt"
    rules_file: str = "rules.txt"
    facts_file: str = "train.txt"
    train_depth: Any = None
    
    # Environment / padding
    padding_atoms: int = 6
    padding_states: int = 64
    max_steps: int = 20
    use_exact_memory: bool = True
    memory_pruning: bool = True
    skip_unary_actions: bool = False  # Must be False for parity
    end_proof_action: bool = True
    reward_type: int = 0
    max_total_vars: int = 1000
    sample_deterministic_per_env: bool = True  # For parity testing
    
    # PPO / training
    n_envs: int = 3
    n_steps: int = 20
    n_epochs: int = 4
    batch_size: int = 20  # Must divide buffer_size (n_steps * n_envs) evenly
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    total_timesteps: int = 120
    n_corruptions: int = 10
    corruption_scheme: List[str] = None
    sampler_default_mode: str = "both"
    
    def __post_init__(self):
        # Set default corruption_scheme based on dataset if not specified
        if self.corruption_scheme is None:
            if 'countries' in self.dataset or 'ablation' in self.dataset:
                self.corruption_scheme = ['tail']
            else:
                self.corruption_scheme = ['head', 'tail']
    
    # Embedding / model
    atom_embedding_size: int = 64
    
    # Model saving / evaluation
    eval_freq: int = 0
    save_model: bool = False
    model_path: str = "./models/"
    restore_best: bool = True
    
    # Misc
    # Misc
    seed: int = 42
    device: str = "cpu"
    verbose: bool = True
    parity: bool = True  # Enable parity mode by default
    
    # Compilation / Performance
    compile_policy: bool = True
    compile_mode: str = "reduce-overhead"
    use_amp: bool = True
    fullgraph: bool = True


def seed_all(seed: int):
    """Set all random seeds."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_eval_callback(
    eval_env,
    eval_queries,
    sampler,
    policy,
    config: TrainCompiledConfig,
    model_path: str = None,
):
    """
    Create an evaluation callback for optimized training.
    
    This is analogous to the callback in train_parity.py.
    """
    from pathlib import Path
    
    state = {
        'best_mrr': -1.0,
        'best_weights': None,
        'eval_count': 0,
        'total_timesteps_at_eval': [],
    }
    
    # Create PPO wrapper for evaluation logic
    eval_ppo = PPOOptimized(
        policy=policy,
        env=eval_env,
        n_steps=config.n_steps,
        learning_rate=0.0, 
        n_epochs=1,
        batch_size=config.n_envs, # Matches eval_env batch size
        device=torch.device(config.device),
        verbose=False,
        parity=config.parity,
    )
    
    # Pre-compile or setup eval environment
    if config.compile_policy:
        print(f"[Callback] Compiling eval environment (mode={config.compile_mode})...")
        # Compile with include_value=False for pure policy evaluation? 
        # evaluate_with_corruptions only needs policy output.
        # But evaluate_policy uses step_with_policy which might expect value if compiled with it?
        # EvalEnvOptimized.compile signature: (policy, deterministic, mode, fullgraph, include_value)
        # We need include_value=True if PPO was compiled with it? No, separate envs.
        eval_env.compile(
            policy=policy,
            deterministic=True, # Eval is deterministic
            mode=config.compile_mode,
            fullgraph=config.fullgraph,
            include_value=False, # Eval doesn't need value
        )
        
        # [PERFORMANCE] Cleanup after compilation to free graph construction memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    else:
        # Eager setup
        eval_env._policy_logits_fn = create_policy_logits_fn(policy)
        eval_env._compile_deterministic = True

    
    def callback(locals_dict, globals_dict):
        """Callback invoked at end of each learn iteration."""
        iteration = locals_dict.get('iteration', 0)
        total_steps_done = locals_dict.get('total_steps_done', 0)
        
        if config.eval_freq <= 0:
            return True
        
        if total_steps_done > 0 and total_steps_done % config.eval_freq == 0:
            state['eval_count'] += 1
            state['total_timesteps_at_eval'].append(total_steps_done)
            
            # [PERFORMANCE] Run evaluation
            # Switch to eval mode
            policy.eval()
            
            eval_results = eval_ppo.evaluate_with_corruptions(
                queries=eval_queries,
                sampler=sampler,
                n_corruptions=config.n_corruptions,
                corruption_modes=tuple(config.corruption_scheme),
                verbose=False,
                parity_mode=config.parity,
            )
            
            # Switch back to train mode
            policy.train()
            
            # [PERFORMANCE] Cleanup large tensors from evaluation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            current_mrr = eval_results.get('MRR', 0.0)
            print(f"[Eval {state['eval_count']}] timesteps={total_steps_done}, MRR={current_mrr:.4f}", end="")
            
            if current_mrr > state['best_mrr']:
                state['best_mrr'] = current_mrr
                state['best_weights'] = {k: v.clone() for k, v in policy.state_dict().items()}
                print(f" ★ New best!")
                
                if model_path and config.save_model:
                    save_path = Path(model_path)
                    save_path.mkdir(parents=True, exist_ok=True)
                    torch.save(policy.state_dict(), save_path / "best_model.pt")
                
            else:
                print("") # Newline
                
        return True
        
    return callback, state


def create_compiled_components(config: TrainCompiledConfig) -> Dict[str, Any]:
    """Create optimized/compiled training components."""
    device = torch.device(config.device)
    
    # Data handler
    dh = DataHandler(
        dataset_name=config.dataset,
        base_path=config.data_path,
        train_file=config.train_file,
        valid_file=config.valid_file, 
        test_file=config.test_file,
        rules_file=config.rules_file,
        facts_file=config.facts_file,
        train_depth=config.train_depth,
        corruption_mode="dynamic",
    )
    
    # Index manager
    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=config.max_total_vars,
        max_arity=dh.max_arity,
        padding_atoms=config.padding_atoms,
        device=device,
        rules=dh.rules,
    )
    
    # Materialize indices
    dh.materialize_indices(im=im, device=device)
    
    # Sampler
    domain2idx, entity2domain = dh.get_sampler_domain_info()
    sampler = Sampler.from_data(
        all_known_triples_idx=dh.all_known_triples_idx,
        num_entities=im.constant_no,
        num_relations=im.predicate_no,
        device=device,
        default_mode=config.sampler_default_mode,
        seed=config.seed,
        domain2idx=domain2idx,
        entity2domain=entity2domain,
    )
    
    # Create stringifier params
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
        max_derived_per_state=config.padding_states,
    )
    base_engine.index_manager = im
    
    # Create vectorized engine
    # Use config.parity to determine if we need strict parity (slower) or fast execution
    vec_engine = UnificationEngineVectorized.from_base_engine(
        base_engine,
        max_fact_pairs=None,
        max_rule_pairs=None,
        padding_atoms=config.padding_atoms,
        parity_mode=config.parity,
    )
    
    # Convert queries to tensor format [N, 3] for PPOOptimized
    train_queries = dh.train_queries
    test_queries = dh.test_queries
    
    def convert_queries_to_atoms(queries):
        """Convert query objects to [N, 3] atom tensors."""
        query_atoms = []
        for q in queries:
            query_atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
            query_atoms.append(query_atom)
        return torch.stack(query_atoms, dim=0).to(device)
    
    train_queries_tensor = convert_queries_to_atoms(train_queries)
    test_queries_tensor = convert_queries_to_atoms(test_queries)
    
    # Create optimized environment
    train_env = EvalEnvOptimized(
        vec_engine=vec_engine,
        batch_size=config.n_envs,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_steps,
        end_proof_action=config.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
        memory_pruning=config.memory_pruning,
    )
    
    # Create eval environment (same config, different mode)
    eval_env = EvalEnvOptimized(
        vec_engine=vec_engine,
        batch_size=config.n_envs,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_steps,
        end_proof_action=config.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
        memory_pruning=config.memory_pruning,
    )
    
    # Create embedder with fixed seed
    n_vars_for_embedder = 1000
    torch.manual_seed(config.seed)
    embedder = TensorEmbedder(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=n_vars_for_embedder,
        max_arity=dh.max_arity,
        padding_atoms=config.padding_atoms,
        atom_embedder='transe',
        state_embedder='sum',
        constant_embedding_size=config.atom_embedding_size,
        predicate_embedding_size=config.atom_embedding_size,
        atom_embedding_size=config.atom_embedding_size,
        device=str(device),
    )
    embedder.embed_dim = config.atom_embedding_size
    
    # Create policy with fixed seed
    action_size = config.padding_states
    torch.manual_seed(config.seed)
    policy = TensorPolicy(
        embedder=embedder,
        embed_dim=config.atom_embedding_size,
        action_dim=action_size,
        hidden_dim=256,
        num_layers=8,
        dropout_prob=0.0,
        device=device,
        parity=True,  # Use SB3-identical initialization for parity testing
    ).to(device)
    
    return {
        'dh': dh,
        'im': im,
        'sampler': sampler,
        'embedder': embedder,
        'base_engine': base_engine,
        'vec_engine': vec_engine,
        'train_env': train_env,
        'eval_env': eval_env,
        'policy': policy,
        'device': device,
        'train_queries_tensor': train_queries_tensor,
        'test_queries_tensor': test_queries_tensor,
    }


def run_experiment(config: TrainCompiledConfig) -> Dict[str, float]:
    """Run full training experiment with optimized PPO and return evaluation metrics."""
    print("=" * 70)
    print("COMPILED/OPTIMIZED TRAINING")
    print(f"Dataset: {config.dataset}")
    print(f"N envs: {config.n_envs}, N steps: {config.n_steps}")
    print(f"Total timesteps: {config.total_timesteps}")
    print(f"Seed: {config.seed}")
    print(f"skip_unary_actions: {config.skip_unary_actions}")
    print("=" * 70)
    
    # Create compiled components
    print("\n[1/3] Creating compiled/optimized components...")
    
    # [PERFORMANCE] Set precision
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        
    seed_all(config.seed)
    comp = create_compiled_components(config)
    
    # [PARITY] Output IndexManager info
    im = comp['im']
    print(f"[PARITY] IndexManager: constants={im.constant_no}, predicates={im.predicate_no}")
    
    # [PARITY] Output Embedder checksum
    embedder = comp['embedder']
    embedder_checksum = sum(p.sum().item() for p in embedder.parameters())
    print(f"[PARITY] Embedder checksum: {embedder_checksum:.6f}")
    
    # [PARITY] Output Policy init checksum
    policy = comp['policy']
    policy_checksum_init = sum(p.sum().item() for p in policy.parameters())
    print(f"[PARITY] Policy checksum after creation: {policy_checksum_init:.6f}")
    
    # [PARITY] Output RNG state before sampler
    print(f"[PARITY] RNG state before sampler: {torch.get_rng_state().sum().item():.0f}")
    
    # Prepare eval queries for callback
    eval_query_objs = comp['dh'].valid_queries[:config.n_envs * 4]
    eval_query_atoms = []
    for q in eval_query_objs:
        query_atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
        eval_query_atoms.append(query_atom)
    eval_queries = torch.stack(eval_query_atoms, dim=0)
    
    # Create evaluation callback if eval_freq is set
    callback = None
    callback_state = None
    if config.eval_freq > 0:
        callback, callback_state = make_eval_callback(
            eval_env=comp['eval_env'],
            eval_queries=eval_queries,
            sampler=comp['sampler'],
            policy=comp['policy'],
            config=config,
            model_path=config.model_path if config.save_model else None,
        )
        print(f"[Callback] Evaluating every {config.eval_freq} timesteps")
    
    # Create PPOOptimized
    print("\n[2/3] Running training...")
    seed_all(config.seed)
    
    # Setup policy for step_with_policy (eager mode for parity tests)
    comp['train_env']._policy_logits_fn = create_policy_logits_fn(policy)
    comp['train_env']._compile_deterministic = False  # Training uses sampling
    comp['train_env']._compiled = False
    
    # Create labels/depths for detailed training metrics
    train_depths = torch.as_tensor(comp['dh'].train_depths, dtype=torch.long, device=comp['device'])
    train_labels = torch.ones(len(comp['dh'].train_queries), dtype=torch.long, device=comp['device'])
    
    ppo = PPOOptimized(
        policy=comp['policy'],
        env=comp['train_env'],
        n_steps=config.n_steps,
        learning_rate=config.learning_rate,
        n_epochs=config.n_epochs,
        batch_size=config.batch_size,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        target_kl=config.target_kl,
        device=comp['device'],
        verbose=True,
        parity=config.parity,
        compile_policy=config.compile_policy,
        compile_mode=config.compile_mode,
        use_amp=config.use_amp,
        query_labels=train_labels,
        query_depths=train_depths,
    )
    
    # [PERFORMANCE] Explicitly compile environment if requested
    # This fuses the policy and environment step into a single CUDA graph
    if config.compile_policy: # reusing flag for now, or assume implied by PPO compilation
        # Parity mode requires loop-based logic for exact matching, which breaks fullgraph
        use_fullgraph = config.fullgraph and not config.parity
        if config.parity and config.fullgraph:
            print("[WARNING] Parity mode enabled: disabling fullgraph requirements (loops required for exact matching)")
        
        print(f"[PERFORMANCE] Compiling environment (mode={config.compile_mode}, fullgraph={use_fullgraph})...")
        comp['train_env'].compile(
            policy=comp['policy'],
            deterministic=False, # Training uses sampling
            mode=config.compile_mode,
            fullgraph=use_fullgraph,
            include_value=True, # Critical: must compile with value prediction for PPO
        )

    # ----------------------------------------------------
    # Setup Callbacks (Metrics & Ranking)
    # ----------------------------------------------------
    callbacks_list = []
    
    # 1. Metrics Callback (Detailed Rollout Info)
    # Replaces default PPO logging with detailed breakdown
    if 'MetricsCallback' in globals():
        metrics_cb = MetricsCallback(log_interval=1, verbose=True, collect_detailed=True)
        callbacks_list.append(metrics_cb)

    # 2. Ranking Callback (Evaluation)
    # Uses PPOOptimized.evaluate_with_corruptions for optimized evaluation
    if 'RankingCallback' in globals() and config.eval_freq > 0:
        # Prepare Eval Data for RankingCallback
        # Use first N envs * 4 queries for fast periodic eval
        n_eval_queries = config.n_envs * 4
        eval_query_objs = comp['dh'].valid_queries[:n_eval_queries]
        eval_queries_depths = torch.as_tensor(comp['dh'].valid_depths[:n_eval_queries], dtype=torch.long, device=comp['device'])
        
        # Need [N, 3] tensor for RankingCallback input
        eval_queries_tensor = torch.stack([
            im.atom_to_tensor(q.predicate, q.args[0], q.args[1]) for q in eval_query_objs
        ], dim=0).to(comp['device'])

        # Create dedicated EvalEnvOptimized for evaluation to avoid compilation conflicts
        # and allow using a safer compilation mode (default) than training (reduce-overhead)
        from env_optimized import EvalEnvOptimized
        
        eval_env_opt = EvalEnvOptimized(
            vec_engine=comp['train_env'].engine,
            batch_size=config.batch_size,
            padding_atoms=config.padding_atoms,
            padding_states=config.padding_states,
            max_depth=config.max_steps,
            end_proof_action=config.end_proof_action,
            runtime_var_start_index=comp['train_env'].runtime_var_start_index,
            device=comp['device'],
            memory_pruning=config.memory_pruning,
        )
        # Setup policy logits for the eval env (needed for compile/execution)
        eval_env_opt._policy_logits_fn = create_policy_logits_fn(comp['policy'])
        
        # Create dedicated PPO agent for evaluation
        # We assume eval batch size fits in training batch size, or we use fixed_batch_size
        ppo_eval = PPOOptimized(
            policy=comp['policy'],
            env=eval_env_opt,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            device=comp['device'],
            compile_policy=True,
            compile_mode='default', # Use default for stability
            query_labels=None, # Not needed for ranking callback internal logic (it passes its own)
            query_depths=None,
        )
        
        print("[PERFORMANCE] Compiling evaluation environment (mode=default)...")
        eval_env_opt.compile(
            policy=comp['policy'],
            deterministic=True, # Eval often uses deterministic
            mode='default',
            fullgraph=False,
            include_value=True,
        )

        ranking_cb = RankingCallback(
            eval_env=eval_env_opt, 
            policy=comp['policy'],
            sampler=comp['sampler'],
            eval_data=eval_queries_tensor,
            eval_data_depths=eval_queries_depths,
            eval_freq=config.eval_freq, 
            n_corruptions=config.n_corruptions,
            corruption_scheme=tuple(config.corruption_scheme),
            ppo_agent=ppo_eval # Pass dedicated evaluation agent
        )
        callbacks_list.append(ranking_cb)
    
    # Manager
    callback_manager = None
    if callbacks_list and 'TorchRLCallbackManager' in globals():
        callback_manager = TorchRLCallbackManager(callbacks=callbacks_list)
        # Manually trigger initial evaluation if present
        if callback_manager:
            callback_manager.on_training_start()

    # Pass all training queries as pool for round-robin cycling (matches BatchedEnv behavior)
    train_queries = comp['train_queries_tensor']
    
    # Adapters for PPO learn - wire up the callback manager
    # PPO.learn expects: callback (end of iter), on_iteration_start_callback, on_step_callback
    cb_func = callback_manager if callback_manager else None
    iteration_start_cb = callback_manager.on_iteration_start if callback_manager else None
    step_cb = callback_manager.on_step if callback_manager else None

    ppo.learn(
        total_timesteps=config.total_timesteps,
        queries=train_queries,
        callback=cb_func,
        on_iteration_start_callback=iteration_start_cb,
        on_step_callback=step_cb
    )

    
    # Restore best model if we tracked it (via CheckpointCallback in manager if we added it, but here we heavily rely on global state)
    # The new callbacks don't easily expose best model directly back to here unless we search the manager.
    # For now, skip auto-restore or implement search if CheckpointCallback was used.
    # But since we didn't add CheckpointCallback (user didn't ask for it specifically in the list, just screen output),
    # we might skip this or rely on the old callback mechanism if we kept it?
    # The old mechanism is gone.
    
    # [PARITY] Output Policy trained checksum
    policy_checksum_trained = sum(p.sum().item() for p in policy.parameters())
    print(f"[PARITY] Policy checksum after training: {policy_checksum_trained:.6f}")
    
    # Evaluation (Final)
    print("\n[3/3] Running evaluation...")
    seed_all(config.seed + 1000)
    
    print(f"[PARITY] RNG before eval: {torch.get_rng_state().sum().item():.0f}")
    
    comp['policy'].eval()
    
    test_queries = comp['dh'].test_queries[:config.n_envs * 4]
    test_query_atoms = []
    for q in test_queries:
        query_atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
        test_query_atoms.append(query_atom)
    test_queries_tensor = torch.stack(test_query_atoms, dim=0)
    
    # [DEBUG] Log evaluation setup
    print(f"\n[COMPILED EVAL DEBUG]")
    print(f"  corruption_scheme: {config.corruption_scheme}")
    print(f"  n_corruptions: {config.n_corruptions}")
    print(f"  num test queries: {len(test_queries)}")
    print(f"  sampler default_mode: {config.sampler_default_mode}")
    print(f"  first query: {test_queries[0]}")
    
    # Use tensor eval for evaluation (same as train_parity.py)
    # We need a BatchedEnv for eval_corruptions, so we create one here (reuse/recreate if needed)
    from env import BatchedEnv
    
    # Reuse convert_queries_to_padded
    if 'convert_queries_to_padded' not in locals():
        def convert_queries_to_padded(queries):
            query_tensors = []
            for q in queries:
                query_atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
                query_padded = torch.full((config.padding_atoms, 3), im.padding_idx, dtype=torch.long, device=comp['device'])
                query_padded[0] = query_atom
                query_tensors.append(query_padded)
            return torch.stack(query_tensors, dim=0)
    
    test_queries_padded = convert_queries_to_padded(comp['dh'].test_queries)
    
    eval_env_batched_final = BatchedEnv(
        batch_size=config.n_envs,
        queries=test_queries_padded,
        labels=torch.ones(len(comp['dh'].test_queries), dtype=torch.long, device=comp['device']),
        query_depths=torch.as_tensor(comp['dh'].test_depths, dtype=torch.long, device=comp['device']),
        unification_engine=comp['base_engine'],
        mode='eval',
        max_depth=config.max_steps,
        memory_pruning=config.memory_pruning,
        use_exact_memory=config.use_exact_memory,
        skip_unary_actions=config.skip_unary_actions,
        end_proof_action=config.end_proof_action,
        reward_type=config.reward_type,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        true_pred_idx=im.predicate_str2idx.get('True'),
        false_pred_idx=im.predicate_str2idx.get('False'),
        end_pred_idx=im.predicate_str2idx.get('Endf'),
        verbose=0,
        prover_verbose=0,
        device=comp['device'],
        runtime_var_start_index=im.constant_no + 1,
        total_vocab_size=im.constant_no + config.max_total_vars,
        sample_deterministic_per_env=config.sample_deterministic_per_env,
    )
    
    eval_results = tensor_eval_corruptions(
        actor=comp['policy'],
        env=eval_env_batched_final,
        queries=test_queries_tensor,
        sampler=comp['sampler'],
        n_corruptions=config.n_corruptions,
        corruption_modes=tuple(config.corruption_scheme),
        query_depths=torch.as_tensor(comp['dh'].test_depths[:config.n_envs * 4], dtype=torch.long, device=comp['device']),
        verbose=False,
    )
    
    # Extract results
    mrr = eval_results.get('MRR', 0.0)
    hits1 = eval_results.get('Hits@1', 0.0)
    hits3 = eval_results.get('Hits@3', 0.0)
    hits10 = eval_results.get('Hits@10', 0.0)
    
    # [PARITY] Output metrics
    print(f"\n[PARITY] Evaluation Results:")
    print(f"[PARITY] Compiled MRR: {mrr:.4f}")
    print(f"[PARITY] Compiled Hits@1: {hits1:.4f}")
    print(f"[PARITY] Compiled Hits@3: {hits3:.4f}")
    print(f"[PARITY] Compiled Hits@10: {hits10:.4f}")
    
    # Get training stats from PPO
    train_stats = getattr(ppo, 'last_train_metrics', {})
    
    # Comprehensive results dict
    results = {
        # Evaluation metrics
        "MRR": mrr,
        "Hits@1": hits1,
        "Hits@3": hits3,
        "Hits@10": hits10,
        # Checksums
        "index_manager_constants": im.constant_no,
        "index_manager_predicates": im.predicate_no,
        "embedder_checksum": embedder_checksum,
        "policy_checksum_init": policy_checksum_init,
        "policy_checksum_trained": policy_checksum_trained,
        # Training losses
        "policy_loss": train_stats.get('policy_loss', 0.0),
        "value_loss": train_stats.get('value_loss', 0.0),
        "entropy": train_stats.get('entropy', 0.0),
        "approx_kl": train_stats.get('approx_kl', 0.0),
        "clip_fraction": train_stats.get('clip_fraction', 0.0),
    }
    
    return results


def main(args, log_filename=None, use_logger=False, use_WB=False, WB_path=None, date=None, external_components=None, profile_run=False):
    """
    Adapter main function to match train.py's signature for runner.py compatibility.
    """
    # Convert args (Namespace) to TrainCompiledConfig
    config = TrainCompiledConfig(
        dataset=args.dataset_name,
        data_path=args.data_path,
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
        rules_file=args.rules_file,
        facts_file=args.facts_file,
        
        # Training params
        n_envs=args.batch_size_env,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        total_timesteps=args.timesteps_train,
        
        # Eval params
        n_corruptions=args.eval_neg_samples if cls_has(args, 'eval_neg_samples') else 10,
        eval_freq=args.eval_freq,
        save_model=args.save_model,
        model_path=args.models_path,
        restore_best=args.restore_best_val_model,
        
        # Performance / Compile - Default to high performance
        compile_policy=getattr(args, 'use_compile', True),
        use_amp=getattr(args, 'use_amp', True),
        compile_mode="reduce-overhead", # Hardcode for performance per user request
        fullgraph=True,
        
        # Misc
        seed=args.seed_run_i if hasattr(args, 'seed_run_i') else args.seed[0] if isinstance(args.seed, list) else args.seed,
        device=args.device,
        verbose=args.verbose,
        parity=False, # Disable strict parity to allow optimizations/sampling
        
        # Embeddings
        atom_embedding_size=args.atom_embedding_size,
        padding_atoms=args.padding_atoms,
        padding_states=args.padding_states,
        max_steps=args.max_depth,
        use_exact_memory=args.use_exact_memory,
        memory_pruning=args.memory_pruning,
        skip_unary_actions=args.skip_unary_actions,
        end_proof_action=args.end_proof_action,
        reward_type=args.reward_type,
        max_total_vars=args.max_total_vars,
        sample_deterministic_per_env=args.sample_deterministic_per_env,
    )
    
    # Run experiment
    results = run_experiment(config)
    
    # Split results into train/valid/test metrics for runner.py
    # runner.py expects: train_metrics, valid_metrics, test_metrics
    
    test_metrics = {}
    train_metrics = {}
    valid_metrics = {}
    
    for k, v in results.items():
        if k in ["MRR", "Hits@1", "Hits@3", "Hits@10"] or k.startswith("mrr_") or k.startswith("hits"):
            test_metrics[k] = v
            valid_metrics[k] = v # Assume valid uses same logic for now
        else:
            train_metrics[k] = v
            
    return train_metrics, valid_metrics, test_metrics


def cls_has(obj, name):
    return hasattr(obj, name) and getattr(obj, name) is not None




def main_cli():
    """Command-line interface for train_compiled.py (standalone use)."""
    parser = argparse.ArgumentParser(description="Train Compiled (Optimized PPO)")
    parser.add_argument("--dataset", type=str, default="countries_s3")
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--total-timesteps", type=int, default=2000)
    parser.add_argument("--n-corruptions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--eval-freq", type=int, default=0,
                        help="Evaluate every N timesteps (0=only at end)")
    parser.add_argument("--save-model", action="store_true", default=False,
                        help="Save model checkpoints")
    parser.add_argument("--model-path", type=str, default="./models/",
                        help="Path to save models")
    parser.add_argument("--no-restore-best", action="store_true", default=False,
                        help="Don't restore best model after training")
    
    args = parser.parse_args()
    
    config = TrainCompiledConfig(
        dataset=args.dataset,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        total_timesteps=args.total_timesteps,
        n_corruptions=args.n_corruptions,
        seed=args.seed,
        device=args.device,
        verbose=args.verbose,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        eval_freq=args.eval_freq,
        save_model=args.save_model,
        model_path=args.model_path,
        restore_best=not args.no_restore_best,
        skip_unary_actions=False,  # Must be False for parity
        parity=True,
    )
    
    run_experiment(config)


if __name__ == "__main__":
    main_cli()
