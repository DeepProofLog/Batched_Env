"""
Training script for Neural-Guided Logical Reasoning.

Usage:
    from train import run_experiment
    from config import TrainConfig
    results = run_experiment(TrainConfig(dataset="countries_s3", total_timesteps=1000))
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np

# Add root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_handler import DataHandler
from index_manager import IndexManager
from unification import UnificationEngineVectorized
from env import EnvVec
from nn.embeddings import EmbedderLearnable
from policy import ActorCriticPolicy
from ppo import PPO
from nn.sampler import Sampler
from config import TrainConfig

from callbacks import (
    TorchRLCallbackManager, 
    MetricsCallback, 
    RankingCallback,
    CheckpointCallback,
    ScalarAnnealingCallback, 
    AnnealingTarget
)

from utils import seed_all
from kge_module import build_kge_inference
from kge_module.pbrs import create_pbrs_module, PBRSWrapper
from kge_module import create_neural_bridge, create_predicate_type_bridge


def build_callbacks(config, ppo, policy, sampler, dh, eval_env=None, date: str = None, im=None):
    """Build callbacks: MetricsCallback, RankingCallback, CheckpointCallback, ScalarAnnealingCallback."""
    # Build predicate vocabulary for per-predicate metrics (index -> name)
    predicate_vocab = None
    if im is not None and hasattr(im, 'idx2predicate'):
        predicate_vocab = {i: name for i, name in enumerate(im.idx2predicate)}

    callbacks = [MetricsCallback(
        log_interval=1,
        verbose=getattr(config, 'verbose', True),
        predicate_vocab=predicate_vocab,
        log_per_depth=getattr(config, 'log_per_depth', True),
        log_per_predicate=getattr(config, 'log_per_predicate', True),
    )]
    best_model_path_train, best_model_path_eval = None, None
    
    save_model = getattr(config, 'save_model', False)
    eval_freq = getattr(config, 'eval_freq', 0)
    
    if save_model or eval_freq > 0:
        models_path = getattr(config, 'models_path', getattr(config, 'model_path', './models/'))
        run_sig = getattr(config, 'run_signature', getattr(config, 'dataset', 'run'))
        save_path = Path(models_path) / run_sig
        best_model_path_train = save_path / "best_model_train.pt"
        best_model_path_eval = save_path / "best_model_eval.pt"
        
    if save_model:
        best_metric = getattr(config, 'eval_best_metric', 'mrr_mean')
        if best_metric == 'mrr':
            best_metric = 'mrr_mean'
        callbacks.append(CheckpointCallback(
            save_path=save_path, policy=policy,
            train_metric="ep_rew_mean", eval_metric=best_metric,
            verbose=True, date=date,
            restore_best=getattr(config, 'restore_best', False),
            load_best_metric=getattr(config, 'load_best_metric', 'eval'),
            load_model=getattr(config, 'load_model', False)
        ))

    if eval_freq > 0 and eval_env is not None:
        valid_split = dh.get_materialized_split('valid')
        valid_queries = valid_split.queries.squeeze(1)
        valid_depths = valid_split.depths
        n_eval = getattr(config, 'n_eval_queries', None)
        if n_eval:
            valid_queries, valid_depths = valid_queries[:n_eval], valid_depths[:n_eval]
        
        n_corruptions = getattr(config, 'eval_neg_samples')
        scheme = getattr(config, 'corruption_scheme', ('head', 'tail'))
        
        callbacks.append(RankingCallback(
            eval_env=eval_env, policy=policy, sampler=sampler,
            eval_data=valid_queries, eval_data_depths=valid_depths,
            eval_freq=int(eval_freq), n_corruptions=n_corruptions,
            corruption_scheme=tuple(scheme), ppo_agent=ppo
        ))
    
    annealing_targets = []
    total_timesteps = getattr(config, 'timesteps_train', getattr(config, 'total_timesteps', 0))
    
    if getattr(config, 'lr_decay', False):
        lr_init = getattr(config, 'lr_init_value', getattr(config, 'lr', getattr(config, 'learning_rate', 3e-4)))
        lr_final = getattr(config, 'lr_final_value', 1e-6)
        lr_warmup_steps = getattr(config, 'lr_warmup_steps', 0.0) if getattr(config, 'lr_warmup', False) else 0.0
        def _set_lr(v):
            for pg in ppo.optimizer.param_groups:
                pg['lr'] = float(v)
            ppo.learning_rate = float(v)
        annealing_targets.append(AnnealingTarget(
            name='lr', setter=_set_lr, initial=float(lr_init), final=float(lr_final),
            start_point=float(getattr(config, 'lr_start', 0.0)),
            end_point=float(getattr(config, 'lr_end', 1.0)),
            transform=getattr(config, 'lr_transform', 'linear'), value_type='float',
            warmup_steps=float(lr_warmup_steps),
        ))
    
    if getattr(config, 'ent_coef_decay', False):
        ent_init = getattr(config, 'ent_coef_init_value', getattr(config, 'ent_coef', 0.01))
        ent_final = getattr(config, 'ent_coef_final_value', 0.01)
        def _set_ent(v):
            ppo.ent_coef = float(v)
        annealing_targets.append(AnnealingTarget(
            name='ent_coef', setter=_set_ent, initial=float(ent_init), final=float(ent_final),
            start_point=float(getattr(config, 'ent_coef_start', 0.0)),
            end_point=float(getattr(config, 'ent_coef_end', 1.0)),
            transform=getattr(config, 'ent_coef_transform', 'linear'), value_type='float',
        ))
    
    if annealing_targets:
        callbacks.append(ScalarAnnealingCallback(
            total_timesteps=total_timesteps, targets=annealing_targets, verbose=1
        ))
    
    callback_manager = TorchRLCallbackManager(callbacks=callbacks) if callbacks else None
    return callback_manager, best_model_path_train, best_model_path_eval


def create_components(config: TrainConfig) -> Dict[str, Any]:
    """Create training components (data handler, index manager, env, model)."""
    device = torch.device(config.device)
    seed_all(config.seed)
    
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
        filter_queries_by_rules=getattr(config, 'filter_queries_by_rules', True),
        # KGE Integration: Probabilistic Facts
        prob_facts=config.prob_facts,
        topk_facts=config.prob_facts_topk,
        topk_facts_threshold=config.prob_facts_threshold,
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
    
    # Convert queries
    train_queries_tensor = im.queries_to_tensor(dh.train_queries, device)
    valid_queries_tensor = im.queries_to_tensor(dh.valid_queries, device)

    
    # Create vectorized engine
    vec_engine = UnificationEngineVectorized.from_index_manager(
        im,
        padding_atoms=config.padding_atoms,
        parity_mode=False,
        max_derived_per_state=config.padding_states,
        end_proof_action=config.end_proof_action,
        max_fact_pairs_cap=getattr(config, 'max_fact_pairs_cap', None),
    )
    
    # Create environment
    env = EnvVec(
        vec_engine=vec_engine,
        batch_size=config.n_envs,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_steps,
        end_proof_action=config.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
        memory_pruning=config.memory_pruning,
        train_queries=train_queries_tensor,
        valid_queries=valid_queries_tensor,
        sample_deterministic_per_env=config.sample_deterministic_per_env,
        sampler=sampler,
        order=False,  # Random query selection (production)
        negative_ratio=config.negative_ratio,
        reward_type=config.reward_type,
        skip_unary_actions=config.skip_unary_actions,  # AAAI26 parity: auto-advance when only 1 action
    )
    
    # Create embedder
    embedder = EmbedderLearnable(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=1000,
        max_arity=dh.max_arity,
        padding_atoms=config.padding_atoms,
        atom_embedder=config.atom_embedder,
        state_embedder=config.state_embedder,
        constant_embedding_size=config.atom_embedding_size,
        predicate_embedding_size=config.atom_embedding_size,
        atom_embedding_size=config.atom_embedding_size,
        device=str(device),
    )
    embedder.embed_dim = config.atom_embedding_size
    
    # Create policy
    policy = ActorCriticPolicy(
        embedder=embedder,
        embed_dim=config.atom_embedding_size,
        action_dim=config.padding_states,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout_prob=config.dropout_prob,
        device=device,
        parity=config.parity,
        # Critical: pass use_l2_norm and temperature to control logit magnitudes
        use_l2_norm=config.use_l2_norm,
        temperature=config.temperature,
        sqrt_scale=config.sqrt_scale,
    ).to(device)
    
    return {
        'dh': dh,
        'im': im,
        'sampler': sampler,
        'embedder': embedder,
        'vec_engine': vec_engine,
        'env': env,
        'policy': policy,
        'device': device,
    }


def run_experiment(config: TrainConfig, return_traces: bool = False) -> Dict[str, Any]:
    """Run training experiment and return evaluation metrics."""
    print("=" * 70)
    print(f"Training: {config.dataset}")
    print(f"Envs: {config.n_envs}, Steps: {config.n_steps}, Timesteps: {config.total_timesteps}")
    print(f"Seed: {config.seed}")
    print("=" * 70)
    
    # Create components
    print("\n[1/3] Creating components...")
    comp = create_components(config)

    config._components = comp

    im = comp['im']
    policy = comp['policy']
    env = comp['env']
    device = comp['device']
    
    # Create PPO
    print("\n[2/3] Running training...")
    
    # Extract metadata for metrics
    query_labels, query_depths = None, None
    dh = comp.get('dh')
    if dh and hasattr(dh, 'get_materialized_split'):
        try:
            train_split = dh.get_materialized_split('train')
            query_depths = train_split.depths
            if hasattr(train_split, 'labels') and train_split.labels is not None:
                query_labels = train_split.labels
        except Exception as e:
            print(f"Warning: Could not extract training metadata: {e}")

    kge_engine = build_kge_inference(config, index_manager=im)

    # Create PBRS wrapper if enabled
    pbrs_wrapper = None
    if getattr(config, 'pbrs_beta', 0.0) != 0.0:
        pbrs_module = create_pbrs_module(
            config=config,
            kge_engine=kge_engine,
            index_manager=im,
            device=torch.device(config.device),
        )
        if pbrs_module is not None:
            pbrs_wrapper = PBRSWrapper(pbrs_module, im)
            print(f"[PBRS] Enabled with beta={config.pbrs_beta}, gamma={getattr(config, 'pbrs_gamma', 0.99)}")

    # Create neural bridge if enabled
    neural_bridge = None
    if getattr(config, 'neural_bridge', False):
        neural_bridge = create_neural_bridge(
            config=config,
            device=torch.device(config.device),
            n_predicates=im.predicate_no,
        )

    # Create predicate-type bridge if enabled (different weights for symmetric vs chain predicates)
    predicate_type_bridge = None
    if getattr(config, 'predicate_aware_scoring', False):
        predicate_type_bridge = create_predicate_type_bridge(
            rules_str=dh.rules_str,
            predicate_str2idx=im.predicate_str2idx,
            n_predicates=im.predicate_no,
            symmetric_weight=getattr(config, 'predicate_aware_symmetric_weight', 0.7),
            chain_weight=getattr(config, 'predicate_aware_chain_weight', 0.0),
            kge_weight=getattr(config, 'kge_eval_kge_weight', 1.0),
            fail_penalty=getattr(config, 'kge_fail_penalty', 0.5),
            device=torch.device(config.device),
            verbose=getattr(config, 'verbose', True),
        )

    ppo = PPO(
        policy,
        env,
        config,
        query_labels=query_labels,
        query_depths=query_depths,
        kge_inference_engine=kge_engine,
        kge_index_manager=im,
        pbrs_wrapper=pbrs_wrapper,
        neural_bridge=neural_bridge,
        predicate_type_bridge=predicate_type_bridge,
    )
    
    # Build callbacks
    date = None
    if config.use_callbacks:
        import datetime
        date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        
        # Create eval env if needed (reuse training env or create new one)
        # For simplicity, we use the same env instance but could create a dedicated one
        eval_env = env 
        
        callback_manager, _, _ = build_callbacks(
            config, ppo, policy, comp['sampler'], comp['dh'],
            eval_env=eval_env, date=date, im=comp['im']
        )
        ppo.callback = callback_manager
     
    # Train
    if config.total_timesteps > 0:
        learn_result = ppo.learn(
            total_timesteps=config.total_timesteps,
            return_traces=return_traces,
        )
    else:
        # If timesteps=0, skip learn but still trigger start callback for loading
        if ppo.callback and hasattr(ppo.callback, 'on_training_start'):
            ppo.callback.on_training_start(total_timesteps=0)
        learn_result = {}
    
    # Normalize corruption_scheme: handle both list/tuple and string
    corruption_scheme = config.corruption_scheme
    if isinstance(corruption_scheme, str):
        corruption_scheme = (corruption_scheme,)
    else:
        corruption_scheme = tuple(corruption_scheme)

    # Train neural bridge on validation data if enabled
    if neural_bridge is not None and getattr(config, 'neural_bridge', False):
        print("\n[2.5/3] Training neural bridge on validation data...")
        n_val_bridge = min(100, len(comp['dh'].valid_queries))  # Use subset for bridge training
        valid_queries_bridge = comp['dh'].valid_queries[:n_val_bridge]
        valid_queries_tensor = im.queries_to_tensor(valid_queries_bridge, device)

        ppo.train_neural_bridge(
            queries=valid_queries_tensor,
            sampler=comp['sampler'],
            n_corruptions=getattr(config, 'eval_neg_samples', 10),
            corruption_modes=corruption_scheme,
            epochs=getattr(config, 'neural_bridge_train_epochs', 100),
            lr=getattr(config, 'neural_bridge_lr', 0.01),
        )

    # Evaluation
    print("\n[3/3] Running evaluation...")
    policy.eval()

    # Warmup evaluation (needed for reduce-overhead compile mode)
    print("Warmup...")
    ppo.evaluate(
        queries=im.queries_to_tensor(comp['dh'].test_queries[:5], device),
        sampler=comp['sampler'],
        n_corruptions=5,
        corruption_modes=('head',),
        verbose=False,
    )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    print("Warmup complete")

    n_test = getattr(config, 'n_test_queries', None)
    if n_test:
        test_queries = comp['dh'].test_queries[:n_test]
        test_depths = comp['dh'].test_depths[:n_test]
    else:
        test_queries = comp['dh'].test_queries
        test_depths = comp['dh'].test_depths

    queries_tensor = im.queries_to_tensor(test_queries, device)

    
    eval_results = ppo.evaluate(
        queries=queries_tensor,
        sampler=comp['sampler'],
        n_corruptions=config.test_neg_samples,
        corruption_modes=corruption_scheme,
        query_depths=torch.as_tensor(
            test_depths,
            dtype=torch.long, device=device
        ),
        verbose=True,
    )
    
    # Extract results
    mrr = eval_results.get('MRR', 0.0)
    hits1 = eval_results.get('Hits@1', 0.0)
    hits3 = eval_results.get('Hits@3', 0.0)
    hits10 = eval_results.get('Hits@10', 0.0)
    
    proven_pos = eval_results.get('proven_pos', 0.0)
    proven_neg = eval_results.get('proven_neg', 0.0)
    print(f"\nResults: MRR={mrr:.4f}, Hits@1={hits1:.4f}, Hits@3={hits3:.4f}, Hits@10={hits10:.4f}")
    print(f"Proven: pos={proven_pos:.4f}, neg={proven_neg:.4f}")
    
    train_stats = getattr(ppo, 'last_train_metrics', {})
    results = {
        "MRR": mrr, "Hits@1": hits1, "Hits@3": hits3, "Hits@10": hits10,
        "policy_loss": train_stats.get('policy_loss', 0.0),
        "value_loss": train_stats.get('value_loss', 0.0),
        "entropy": train_stats.get('entropy', 0.0),
    }
    # Merge all detailed eval metrics (e.g. depth breakdown)
    results.update(eval_results)
    
    if return_traces:
        results['rollout_traces'] = learn_result.get('rollout_traces', [])
        results['train_traces'] = learn_result.get('train_traces', [])
    
    return results
