"""
KGE Experiment Builder - Factory functions for KGE components.

This module hides internal complexity (DataHandler, IndexManager,
UnificationEngine, Sampler) from the user. The main entry point is
through runner.py which calls these factories.

Example:
    from kge_experiments.builder import get_default_config, create_env, create_policy

    config = get_default_config(dataset='countries_s3')
    env = create_env(config)
    policy = create_policy(config)
"""
import torch
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

from .config import TrainConfig
from .data_handler import DataHandler
from .index_manager import IndexManager
from .unification import UnificationEngineVectorized
from .env import EnvVec
from .nn.sampler import Sampler
from .nn.embeddings import EmbedderLearnable as Embedder
from .policy import ActorCriticPolicy as Policy
from .kge_module import build_kge_inference
from .kge_module.pbrs import create_pbrs_module, PBRSWrapper
from .kge_module import create_neural_bridge, create_predicate_type_bridge
from utils import seed_all


@dataclass 
class KGEConfig(TrainConfig):
    """KGE experiment configuration with experiment_type marker."""
    experiment_type: str = field(default='kge', init=False, repr=False)
    
    # Cached internal components (populated by create_env)
    _components: Dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


def get_default_config(**overrides) -> KGEConfig:
    """
    Get default KGE config with optional overrides.

    Args:
        **overrides: Any TrainConfig field to override (e.g., dataset='fb15k')

    Returns:
        KGEConfig with defaults and overrides applied.
    """
    config = KGEConfig()
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Update run_signature based on dataset and other params
    import datetime
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    config.run_signature = f"{config.dataset}-{config.atom_embedding_size}-{config.n_envs}-{date_str}"

    return config


def create_env(config: KGEConfig) -> EnvVec:
    """
    Create KGE environment with all internal components.
    
    Internally initializes: DataHandler, IndexManager, UnificationEngine, Sampler.
    These are cached in config._components for use by create_policy().
    
    Args:
        config: KGEConfig with experiment parameters.
        
    Returns:
        EnvVec environment ready for training/evaluation.
    """
    device = torch.device(config.device)
    seed_all(config.seed, deterministic=config.parity)
    
    # =========================================================================
    # DataHandler - loads facts, rules, queries from disk
    # =========================================================================
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
        filter_queries_by_rules=getattr(config, 'filter_queries_by_rules', True),
        # KGE Integration: Probabilistic Facts
        prob_facts=getattr(config, 'prob_facts', False),
        topk_facts=getattr(config, 'prob_facts_topk', None),
        topk_facts_threshold=getattr(config, 'prob_facts_threshold', None),
    )
    
    # =========================================================================
    # IndexManager - converts strings to indices, manages vocabulary
    # =========================================================================
    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=config.max_total_vars,
        max_arity=dh.max_arity,
        padding_atoms=config.padding_atoms,
        device=device,
        rules=dh.rules,
    )
    dh.materialize_indices(im=im, device=device)
    
    # =========================================================================
    # Sampler - generates negative samples for training/evaluation
    # =========================================================================
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
    
    # =========================================================================
    # UnificationEngine - performs logical unification for reasoning
    # =========================================================================
    vec_engine = UnificationEngineVectorized.from_index_manager(
        im,
        padding_atoms=config.padding_atoms,
        parity_mode=config.parity,
        max_derived_per_state=config.padding_states,
        end_proof_action=config.end_proof_action,
        max_fact_pairs_cap=getattr(config, 'max_fact_pairs_cap', None),
    )
    
    # Clean up index manager tensors (already transferred to engine)
    im.facts_idx = None
    im.rules_idx = None
    im.rule_lens = None
    im.rules_heads_idx = None
    
    # =========================================================================
    # Convert queries to tensor format
    # =========================================================================
    train_queries = torch.stack([
        im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
        for q in dh.train_queries
    ], dim=0)

    valid_queries = torch.stack([
        im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
        for q in dh.valid_queries
    ], dim=0) if dh.valid_queries else None

    test_queries = torch.stack([
        im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
        for q in dh.test_queries
    ], dim=0)
    
    # =========================================================================
    # EnvVec - the actual environment
    # =========================================================================
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
        train_queries=train_queries,
        valid_queries=valid_queries if valid_queries is not None else test_queries,
        sample_deterministic_per_env=config.parity,  # Only use parity path when testing
        sampler=sampler,
        order=True if config.parity else False,
        negative_ratio=config.negative_ratio,
        reward_type=getattr(config, 'reward_type', 4),
        skip_unary_actions=getattr(config, 'skip_unary_actions', False),
    )
    
    # Cache components for policy creation and evaluation
    config._components = {
        'dh': dh,
        'im': im,
        'sampler': sampler,
        'vec_engine': vec_engine,
        'train_queries': train_queries,
        'valid_queries': valid_queries if valid_queries is not None else test_queries,
        'test_queries': test_queries,
        'device': device,
    }
    
    return env


def create_policy(config: KGEConfig, env: Optional[EnvVec] = None) -> Policy:
    """
    Create KGE policy network.
    
    Must be called after create_env() to access cached components.
    
    Args:
        config: KGEConfig (must have _components populated by create_env).
        env: Optional EnvVec (unused, for API compatibility).
        
    Returns:
        ActorCriticPolicy ready for training.
    """
    if not config._components:
        raise RuntimeError("create_env() must be called before create_policy()")
    
    device = config._components['device']
    im = config._components['im']
    dh = config._components['dh']
    
    # Create embedder
    torch.manual_seed(config.seed)
    embedder = Embedder(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=1000,  # Fixed to match train.py
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
    torch.manual_seed(config.seed)
    policy = Policy(
        embedder=embedder,
        embed_dim=config.atom_embedding_size,
        action_dim=config.padding_states,
        hidden_dim=getattr(config, 'hidden_dim', 256),
        num_layers=getattr(config, 'num_layers', 8),
        dropout_prob=getattr(config, 'dropout_prob', 0.0),
        separate_value_network=getattr(config, 'separate_value_network', False),
        value_head_scale=getattr(config, 'value_head_scale', 1.0),
        device=device,
        parity=config.parity,  # Use parity when in parity mode, otherwise not
    ).to(device)
    
    return policy


def create_algorithm(policy: Policy, env: EnvVec, config: KGEConfig):
    """
    Create PPO algorithm and set up callbacks for training.

    Args:
        policy: Policy network
        env: Environment
        config: KGEConfig with _components populated

    Returns:
        PPO algorithm instance with callbacks configured
    """
    from ppo import PPO
    from callbacks import (
        TorchRLCallbackManager,
        MetricsCallback,
        RankingCallback,
        CheckpointCallback,
        ScalarAnnealingCallback,
        AnnealingTarget,
    )

    if not config._components:
        raise RuntimeError("create_env() must be called before create_algorithm()")

    im = config._components['im']
    dh = config._components['dh']
    device = config._components['device']
    sampler = config._components['sampler']

    # =========================================================================
    # Extract query metadata for PPO
    # =========================================================================
    query_labels, query_depths = None, None
    if hasattr(dh, 'get_materialized_split'):
        try:
            train_split = dh.get_materialized_split('train')
            query_depths = train_split.depths
            if hasattr(train_split, 'labels') and train_split.labels is not None:
                query_labels = train_split.labels
        except Exception as e:
            print(f"Warning: Could not extract training metadata: {e}")

    # =========================================================================
    # Build KGE inference engine if enabled
    # =========================================================================
    kge_engine = None
    if getattr(config, 'kge_inference', False):
        kge_engine = build_kge_inference(config, index_manager=im)

    # =========================================================================
    # Create PBRS wrapper if enabled
    # =========================================================================
    pbrs_wrapper = None
    if getattr(config, 'pbrs_beta', 0.0) != 0.0:
        pbrs_module = create_pbrs_module(
            config=config,
            kge_engine=kge_engine,
            index_manager=im,
            device=device,
        )
        if pbrs_module is not None:
            pbrs_wrapper = PBRSWrapper(pbrs_module, im)
            print(f"[PBRS] Enabled with beta={config.pbrs_beta}, gamma={getattr(config, 'pbrs_gamma', 0.99)}")

    # =========================================================================
    # Create neural bridge if enabled
    # =========================================================================
    neural_bridge = None
    if getattr(config, 'neural_bridge', False):
        neural_bridge = create_neural_bridge(
            config=config,
            device=device,
            n_predicates=im.predicate_no,
        )

    # =========================================================================
    # Create predicate-type bridge if enabled
    # =========================================================================
    predicate_type_bridge = None
    if getattr(config, 'predicate_aware_scoring', False):
        predicate_type_bridge = create_predicate_type_bridge(
            rules_str=dh.rules_str,
            predicate_str2idx=im.predicate_str2idx,
            n_predicates=im.predicate_no,
            symmetric_weight=getattr(config, 'predicate_aware_symmetric_weight', 0.7),
            chain_weight=getattr(config, 'predicate_aware_chain_weight', 0.0),
            kge_weight=getattr(config, 'kge_eval_kge_weight', 1.0),
            fail_penalty=getattr(config, 'kge_fail_penalty', 100),
            device=device,
            verbose=getattr(config, 'verbose', True),
        )

    # =========================================================================
    # Create PPO
    # =========================================================================
    algorithm = PPO(
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

    # Store neural_bridge in config for later use in run_evaluation
    config._components['neural_bridge'] = neural_bridge

    # =========================================================================
    # Build callbacks
    # =========================================================================
    callbacks = []

    # MetricsCallback for basic logging
    callbacks.append(MetricsCallback(
        log_interval=1,
        verbose=getattr(config, 'verbose', False),
        collect_detailed=True,
    ))

    # CheckpointCallback for model saving
    save_model = getattr(config, 'save_model', False)
    eval_freq = getattr(config, 'eval_freq', 0)

    if save_model:
        from pathlib import Path
        import datetime
        date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        models_path = getattr(config, 'models_path', getattr(config, 'model_path', './models/'))
        run_sig = getattr(config, 'run_signature', getattr(config, 'dataset', 'run'))
        save_path = Path(models_path) / run_sig

        best_metric = getattr(config, 'eval_best_metric', 'mrr_mean')
        if best_metric == 'mrr':
            best_metric = 'mrr_mean'

        callbacks.append(CheckpointCallback(
            save_path=save_path,
            policy=policy,
            train_metric="ep_rew_mean",
            eval_metric=best_metric,
            verbose=True,
            date=date,
            restore_best=getattr(config, 'restore_best', False),
            load_best_metric=getattr(config, 'load_best_metric', 'eval'),
            load_model=getattr(config, 'load_model', False),
        ))

    # RankingCallback for evaluation during training
    if eval_freq > 0:
        valid_queries = config._components.get('valid_queries', config._components['test_queries'])
        valid_depths = None

        # Try to get validation depths
        if hasattr(dh, 'get_materialized_split'):
            try:
                valid_split = dh.get_materialized_split('valid')
                valid_depths = valid_split.depths
            except Exception:
                pass

        # Limit eval queries
        n_eval = getattr(config, 'n_eval_queries', 20)
        if n_eval is not None and n_eval < len(valid_queries):
            valid_queries = valid_queries[:n_eval]
            if valid_depths is not None:
                valid_depths = valid_depths[:n_eval]

        # Get corruption scheme
        corruption_scheme = getattr(config, 'corruption_scheme', ['head', 'tail'])
        if isinstance(corruption_scheme, str):
            corruption_scheme = (corruption_scheme,)
        else:
            corruption_scheme = tuple(corruption_scheme)

        n_corruptions = getattr(config, 'eval_neg_samples', 100)

        callbacks.append(RankingCallback(
            eval_env=env,
            policy=algorithm._uncompiled_policy,
            sampler=sampler,
            eval_data=valid_queries,
            eval_data_depths=valid_depths,
            eval_freq=int(eval_freq),
            n_corruptions=n_corruptions,
            corruption_scheme=corruption_scheme,
            ppo_agent=algorithm,
            verbose=False,
        ))

    # ScalarAnnealingCallback for learning rate / entropy decay
    annealing_targets = []
    total_timesteps = getattr(config, 'timesteps_train', getattr(config, 'total_timesteps', 0))

    if getattr(config, 'lr_decay', False):
        lr_init = getattr(config, 'lr_init_value', getattr(config, 'lr', getattr(config, 'learning_rate', 3e-4)))
        lr_final = getattr(config, 'lr_final_value', 1e-6)

        def _set_lr(v):
            for pg in algorithm.optimizer.param_groups:
                pg['lr'] = float(v)
            algorithm.learning_rate = float(v)

        annealing_targets.append(AnnealingTarget(
            name='lr',
            setter=_set_lr,
            initial=float(lr_init),
            final=float(lr_final),
            start_point=float(getattr(config, 'lr_start', 0.0)),
            end_point=float(getattr(config, 'lr_end', 1.0)),
            transform=getattr(config, 'lr_transform', 'linear'),
            value_type='float',
        ))

    if getattr(config, 'ent_coef_decay', False):
        ent_init = getattr(config, 'ent_coef_init_value', getattr(config, 'ent_coef', 0.01))
        ent_final = getattr(config, 'ent_coef_final_value', 0.01)

        def _set_ent(v):
            algorithm.ent_coef = float(v)

        annealing_targets.append(AnnealingTarget(
            name='ent_coef',
            setter=_set_ent,
            initial=float(ent_init),
            final=float(ent_final),
            start_point=float(getattr(config, 'ent_coef_start', 0.0)),
            end_point=float(getattr(config, 'ent_coef_end', 1.0)),
            transform=getattr(config, 'ent_coef_transform', 'linear'),
            value_type='float',
        ))

    if annealing_targets:
        callbacks.append(ScalarAnnealingCallback(
            total_timesteps=total_timesteps,
            targets=annealing_targets,
            verbose=1,
        ))

    # Set callback manager
    if callbacks:
        algorithm.callback = TorchRLCallbackManager(callbacks=callbacks)

    return algorithm


# =============================================================================
# Helper accessors (internal use)
# =============================================================================

def run_evaluation(algorithm, config: KGEConfig) -> dict:
    """
    Run evaluation with all KGE-specific logic.

    Handles warmup, test queries, sampler, corruption scheme, neural bridge training, etc.

    Args:
        algorithm: PPO algorithm instance
        config: KGEConfig with _components populated

    Returns:
        Dictionary of evaluation results (MRR, Hits@K, etc.)
    """
    import time

    if not config._components:
        raise RuntimeError("create_env() must be called first")

    # Get components
    sampler = config._components['sampler']
    test_queries = config._components['test_queries']
    dh = config._components['dh']
    im = config._components['im']
    device = config._components['device']
    neural_bridge = config._components.get('neural_bridge')

    # Get corruption scheme
    corruption_scheme = getattr(config, 'corruption_scheme', ['head', 'tail'])
    if isinstance(corruption_scheme, str):
        corruption_scheme = (corruption_scheme,)
    else:
        corruption_scheme = tuple(corruption_scheme)

    # =========================================================================
    # Trigger callback for model loading if training was skipped
    # =========================================================================
    if algorithm.num_timesteps == 0 and algorithm.callback is not None:
        if hasattr(algorithm.callback, 'on_training_start'):
            algorithm.callback.on_training_start(total_timesteps=0)

    # =========================================================================
    # Restore best model if available (from RankingCallback during training)
    # =========================================================================
    if algorithm.callback is not None:
        from callbacks import RankingCallback
        for cb in getattr(algorithm.callback, 'callbacks', []):
            if isinstance(cb, RankingCallback) and cb.best_model_state is not None:
                best_mrr = cb.mrr_tracker.best_mrr
                best_iter = cb.mrr_tracker.best_iteration
                print(f"[Eval] Restoring best model from iter {best_iter} (MRR={best_mrr:.3f})")
                algorithm._uncompiled_policy.load_state_dict(cb.best_model_state)
                break

    # =========================================================================
    # Train neural bridge on validation data if enabled
    # =========================================================================
    if neural_bridge is not None and getattr(config, 'neural_bridge', False):
        print("\n[Eval] Training neural bridge on validation data...")
        n_val_bridge = min(100, len(dh.valid_queries))
        valid_queries_bridge = dh.valid_queries[:n_val_bridge]
        valid_queries_tensor = im.queries_to_tensor(valid_queries_bridge, device)

        algorithm.train_neural_bridge(
            queries=valid_queries_tensor,
            sampler=sampler,
            n_corruptions=getattr(config, 'eval_neg_samples', 10),
            corruption_modes=corruption_scheme,
            epochs=getattr(config, 'neural_bridge_train_epochs', 100),
            lr=getattr(config, 'neural_bridge_lr', 0.01),
        )

    # =========================================================================
    # Get test queries and depths
    # =========================================================================
    test_depths = None
    if hasattr(dh, 'test_depths') and dh.test_depths is not None:
        test_depths = dh.test_depths

    # Limit test queries if specified (None or >= total means full evaluation)
    n_test = getattr(config, 'n_test_queries', None)
    if n_test is not None and n_test < len(test_queries):
        test_queries = test_queries[:n_test]
        if test_depths is not None:
            test_depths = test_depths[:n_test]

    # Get negative samples
    test_neg_samples = getattr(config, 'test_neg_samples', 100)

    # =========================================================================
    # Warmup evaluation for torch.compile
    # =========================================================================
    print("Warmup...")
    algorithm.evaluate(
        queries=test_queries[:5],
        sampler=sampler,
        n_corruptions=5,
        corruption_modes=('head',),
        verbose=False,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("Warmup complete")

    # =========================================================================
    # Run evaluation
    # =========================================================================
    algorithm._uncompiled_policy.eval()
    eval_start = time.time()

    # Prepare query_depths tensor if available
    query_depths_tensor = None
    if test_depths is not None:
        query_depths_tensor = torch.as_tensor(test_depths, dtype=torch.long, device=device)

    results = algorithm.evaluate(
        queries=test_queries,
        sampler=sampler,
        n_corruptions=test_neg_samples,
        corruption_modes=corruption_scheme,
        query_depths=query_depths_tensor,
        verbose=True,
    )
    eval_time = time.time() - eval_start

    print(f"[Eval] Took {eval_time:.2f} seconds")

    # Print ranking metrics
    mrr = results.get('MRR', 0.0)
    hits1 = results.get('Hits@1', 0.0)
    hits3 = results.get('Hits@3', 0.0)
    hits10 = results.get('Hits@10', 0.0)
    proven_pos = results.get('proven_pos', 0.0)
    proven_neg = results.get('proven_neg', 0.0)

    print(f"[Ranking] MRR: {mrr:.3f}, Hits@1: {hits1:.3f}, Hits@3: {hits3:.3f}, Hits@10: {hits10:.3f}")
    print(f"[Proven] pos: {proven_pos:.3f}, neg: {proven_neg:.3f}")

    return results
