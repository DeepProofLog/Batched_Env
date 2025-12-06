import json
from typing import Any, Callable, List, Optional, Tuple
import torch
from pathlib import Path

try:
    # Try relative import first (when sb3/ is in sys.path)
    from sb3_index_manager import IndexManager
    from sb3_utils import (
        get_device, 
        print_eval_info, 
        profile_code, 
        _set_seeds,
        _freeze_dropout_layernorm,
        _warn_non_reproducible,
        # _maybe_enable_wandb,
    )
    from sb3_callbacks import (
        SB3TrainCheckpoint,
        CustomEvalCallbackMRR,
        CustomEvalCallback,
        DepthProofStatsCallback,
        ScalarAnnealingCallback,
        AnnealingTarget,
        _EvalDepthRewardTracker
    )
    from sb3_custom_dummy_env import create_environments
    from sb3_dataset import DataHandler
    from sb3_model import CustomActorCriticPolicy, CustomCombinedExtractor, PPO_custom as PPO
    from sb3_embeddings import get_embedder
    from sb3_neg_sampling import get_sampler
    from sb3_model_eval import eval_corruptions
except ImportError:
    # Fallback to package import (when imported as sb3.sb3_train)
    from sb3.sb3_index_manager import IndexManager
    from sb3.sb3_utils import (
        get_device, 
        print_eval_info, 
        profile_code, 
        _set_seeds,
        _freeze_dropout_layernorm,
        _warn_non_reproducible,
        # _maybe_enable_wandb,
    )
    from sb3.sb3_callbacks import (
        SB3TrainCheckpoint,
        CustomEvalCallbackMRR,
        CustomEvalCallback,
        DepthProofStatsCallback,
        ScalarAnnealingCallback,
        AnnealingTarget,
        _EvalDepthRewardTracker
    )
    from sb3.sb3_custom_dummy_env import create_environments
    from sb3.sb3_dataset import DataHandler
    from sb3.sb3_model import CustomActorCriticPolicy, CustomCombinedExtractor, PPO_custom as PPO
    from sb3.sb3_embeddings import get_embedder
    from sb3.sb3_neg_sampling import get_sampler
    from sb3.sb3_model_eval import eval_corruptions
from stable_baselines3.common.callbacks import (
    StopTrainingOnRewardThreshold,
    CallbackList,
    StopTrainingOnNoModelImprovement,
)
# from kge_integration import (
#     _init_kge_engine,
#     _attach_kge_to_policy,
# )

def _attach_kge_to_policy(
    model: PPO,
    im: IndexManager,
    engine,
    device: torch.device,
    args: Any,
) -> None:
    """ Placeholder for KGE attachment logic. """
    policy = model.policy
    policy.enable_kge_action = False
    policy.enable_logit_fusion = False
    policy.kge_inference_engine = None
    policy.index_manager = None
    policy.kge_indices_tensor = torch.empty(0, dtype=torch.int32, device=device)
    return None
# ------------------------------
# Initialization helpers
# ------------------------------

def _build_data_and_index(args: Any, device: torch.device) -> Tuple[DataHandler, IndexManager, Any, Any]:
    """Prepare DataHandler, IndexManager, sampler and embedder."""
    
    # PARITY: Reseed at start for deterministic alignment
    deterministic_parity = getattr(args, 'deterministic_parity', False)
    if deterministic_parity:
        _set_seeds(args.seed_run_i)
    
    # Dataset
    dh = DataHandler(
        dataset_name=args.dataset_name,
        base_path=args.data_path,
        janus_file=args.janus_file,
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
        rules_file=args.rules_file,
        facts_file=args.facts_file,
        n_train_queries=args.n_train_queries,
        n_eval_queries=args.n_eval_queries,
        n_test_queries=args.n_test_queries,
        corruption_mode=args.corruption_mode,
        train_depth=args.train_depth,
        valid_depth=args.valid_depth,
        test_depth=args.test_depth,
        prob_facts=args.prob_facts,
        topk_facts=args.topk_facts,
        topk_facts_threshold=args.topk_facts_threshold,
    )
    print(f"DEBUG: SB3 DataHandler n_train_queries={args.n_train_queries}")
    print(f"DEBUG: SB3 DataHandler predicates ({len(dh.predicates)}): {sorted(list(dh.predicates))}")

    # Respect caps from args while ensuring >1 eval query for callbacks
    args.n_train_queries = (
        len(dh.train_queries)
        if args.n_train_queries is None
        else min(args.n_train_queries, len(dh.train_queries))
    )
    args.n_eval_queries = (
        len(dh.valid_queries)
        if args.n_eval_queries is None
        else min(args.n_eval_queries, len(dh.valid_queries))
    )
    assert (
        args.n_eval_queries > 1
    ), "Number of evaluation queries must be greater than 1 for callbacks."
    args.n_test_queries = (
        len(dh.test_queries)
        if args.n_test_queries is None
        else min(args.n_test_queries, len(dh.test_queries))
    )

    # Index manager
    im = IndexManager(
        dh.constants,
        dh.predicates,
        args.max_total_vars,
        constants_images=dh.constants_images if args.dataset_name == "mnist_addition" else set(),
        constant_images_no=dh.constant_images_no if args.dataset_name == "mnist_addition" else 0,
        rules=dh.rules,
        max_arity=dh.max_arity,
        device="cpu",
        padding_atoms=args.padding_atoms,
        include_kge_predicates=args.kge_action,
    )
    im.build_fact_index(dh.facts)

    # Negative sampler
    dh.sampler = get_sampler(
        data_handler=dh,
        index_manager=im,
        corruption_scheme=args.corruption_scheme,
        device=device,
        corruption_mode=args.corruption_mode,
    )
    sampler = dh.sampler

    # Embedder
    # Seed is already set at the beginning of main(), but reseed to align with torchrl stack
    torch.manual_seed(args.seed_run_i)
    embedder_getter = get_embedder(args, dh, im, device)
    embedder = embedder_getter.embedder

    # Derived dims for concat options
    args.atom_embedding_size = (
        args.atom_embedding_size
        if args.atom_embedder != "concat"
        else (1 + dh.max_arity) * args.atom_embedding_size
    )
    args.state_embedding_size = (
        args.atom_embedding_size
        if args.state_embedder != "concat"
        else args.atom_embedding_size * args.padding_atoms
    )
    embedder.embed_dim = args.state_embedding_size

    return dh, im, sampler, embedder

# ------------------------------
# Checkpoint helpers
# ------------------------------

def _model_dir(args: Any, date: str) -> Path:
    return Path(args.models_path) / args.run_signature / f"seed_{args.seed_run_i}"


def _resolve_ckpt_to_load(root: Path, restore_best: bool) -> Optional[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {root}")
    keyword = "best_eval" if restore_best else "last_epoch"
    candidates = sorted([p for p in root.glob(f"*{keyword}*.zip")])
    return candidates[-1] if candidates else None


def _maybe_load_model(args: Any, device: torch.device, eval_env, date: str, model: PPO) -> Tuple[bool, PPO]:
    """Load a checkpoint when args.load_model is True. Returns (loaded, model)."""
    if not args.load_model:
        return False, model

    try:
        ckpt_dir = _model_dir(args, date)
        ckpt = _resolve_ckpt_to_load(ckpt_dir, restore_best=args.restore_best_val_model)
        if ckpt is None:
            raise FileNotFoundError(
                f"No suitable '{'best_eval' if args.restore_best_val_model else 'last_epoch'}' model found in {ckpt_dir}"
        )
        print(f"Loading model from {ckpt}")
        loaded = PPO.load(str(ckpt), env=eval_env, device=device)
        info_path = ckpt.with_name(f"info_{ckpt.stem}.json")
        checkpoint_info = None
        if info_path.exists():
            try:
                with info_path.open("r", encoding="utf-8") as info_file:
                    checkpoint_info = json.load(info_file)
                print(f"Loaded checkpoint metadata from {info_path}")
            except json.JSONDecodeError as exc:
                print(f"Warning: Failed to parse checkpoint metadata at {info_path}: {exc}")
        if checkpoint_info is not None:
            setattr(loaded, "checkpoint_info", checkpoint_info)
            setattr(args, "loaded_model_info", checkpoint_info)
        return True, loaded
    except (FileNotFoundError, NotADirectoryError) as e:
        if args.timesteps_train > 0:
            print(
                f"Warning: Could not load pre-existing model ({e}). A new model will be trained from scratch."
            )
            args.load_model = False
            return False, model
        raise ValueError(
            f"Error: In evaluation-only mode but could not load model. Reason: {e}"
        )
    



# ------------------------------
# Training helpers
# ------------------------------

def _build_callbacks(
    args: Any,
    model: PPO,
    callback_env,
    eval_env,
    sampler,
    data_handler: DataHandler,
    log_filename: Optional[str],
    model_path: Path,
    model_name: str,
):
    callbacks = []
    reward_threshold_cb = StopTrainingOnRewardThreshold(reward_threshold=1, verbose=1)
    # timing_callback = EpochTimingCallback(verbose=1)
    # no_improvement_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=7, verbose=1)

    if hasattr(callback_env, "type_") and callback_env.type_ == "custom_dummy":
        eval_cb = CustomEvalCallbackMRR(
            eval_env=callback_env,
            sampler=sampler,
            eval_data=data_handler.valid_queries[: args.n_eval_queries],
            eval_data_depths=data_handler.valid_queries_depths[: args.n_eval_queries],
            n_corruptions=args.eval_neg_samples,
            model_path=str(model_path) if args.save_model else None,
            log_path=log_filename,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_queries - 1,
            deterministic=True,
            render=False,
            name=model_name,
            callback_on_new_best=(
                reward_threshold_cb if args.restore_best_val_model and not args.kge_action else None
            ),
            corruption_scheme=args.corruption_scheme,
            best_metric=args.eval_best_metric,
            # callback_after_eval=no_improvement_callback,
            verbose=0,
        )
    else:
        eval_cb = CustomEvalCallback(
            eval_env=callback_env,
            model_path=str(model_path) if args.save_model else None,
            log_path=log_filename,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_queries - 1,
            deterministic=True,
            render=False,
            name=model_name,
            # callback_on_new_best=reward_threshold_callback if args.restore_best_val_model \
            #                                        and not args.kge_action else None,
            # callback_after_eval=no_improvement_callback,
        )

    train_ckpt_cb = SB3TrainCheckpoint(
        model,
        monitor="rollout/ep_rew_mean",
        frequency=5,
        model_path=str(model_path) if args.save_model else None,
        name=model_name,
    )

    depth_stats_cb = DepthProofStatsCallback(prefix="rollout", track_negative=True)

    callbacks.extend([eval_cb, train_ckpt_cb, depth_stats_cb])

    # Add annealing callbacks if configured
    annealing_specs = getattr(args, "annealing_specs", {}) or {}
    if not isinstance(annealing_specs, dict):
        annealing_specs = {}

    annealing_targets: List[AnnealingTarget] = []
    total_timesteps = max(int(args.timesteps_train or 0), 1)
    policy = model.policy

    top_k_spec = annealing_specs.get('top_k_value')
    if policy is not None and isinstance(top_k_spec, dict) and args.enable_top_k:
        top_k_initial = top_k_spec.get('initial')
        top_k_final = top_k_spec.get('final')
        if top_k_initial is not None and top_k_final is not None:
            def _set_top_k_value(value: float) -> None:
                if value is None or int(value) <= 0:
                    policy.top_k_value = None
                else:
                    policy.top_k_value = int(value)

            annealing_targets.append(
                AnnealingTarget(
                    name='top_k_value',
                    setter=_set_top_k_value,
                    initial=float(top_k_initial),
                    final=float(top_k_final),
                    start_point=float(top_k_spec.get('start_point', 0.0)),
                    end_point=float(top_k_spec.get('end_point', 1.0)),
                    transform=str(top_k_spec.get('transform', 'linear')),
                    value_type='int',
                )
            )
            print(
                f"Top-K annealing configured: {top_k_initial} -> {top_k_final} "
                f"({top_k_spec.get('transform', 'linear')})"
            )

    logit_spec = annealing_specs.get('kge_logit_gain')
    if isinstance(logit_spec, dict) and (args.logit_fusion or args.kge_action):
        def _set_kge_gain(value: float) -> None:
            if hasattr(policy, 'set_kge_logit_gain'):
                policy.set_kge_logit_gain(float(value))

        annealing_targets.append(
            AnnealingTarget(
                name='kge_logit_gain',
                setter=_set_kge_gain,
                initial=float(logit_spec.get('initial', args.kge_logit_init_value)),
                final=float(logit_spec.get('final', args.kge_logit_final_value)),
                start_point=float(logit_spec.get('start_point', 0.0)),
                end_point=float(logit_spec.get('end_point', 1.0)),
                transform=str(logit_spec.get('transform', 'linear')),
                value_type='float',
            )
        )

    # --- Entropy coefficient decay ---
    ent_coef_spec = annealing_specs.get('ent_coef')
    if isinstance(ent_coef_spec, dict):
        def _set_ent_coef(value: float) -> None:
            model.ent_coef = float(value)

        annealing_targets.append(
            AnnealingTarget(
                name='ent_coef',
                setter=_set_ent_coef,
                initial=float(ent_coef_spec.get('initial', args.ent_coef)),
                final=float(ent_coef_spec.get('final', 0.01)),
                start_point=float(ent_coef_spec.get('start_point', 0.0)),
                end_point=float(ent_coef_spec.get('end_point', 1.0)),
                transform=str(ent_coef_spec.get('transform', 'linear')),
                value_type='float',
            )
        )
        print(
            f"Entropy coefficient annealing: {ent_coef_spec.get('initial')} -> {ent_coef_spec.get('final')} "
            f"({ent_coef_spec.get('transform', 'linear')})"
        )

    # --- Learning rate decay ---
    lr_spec = annealing_specs.get('lr')
    if isinstance(lr_spec, dict):
        def _set_lr(value: float) -> None:
            # Update the learning rate for all parameter groups in the optimizer
            for param_group in model.policy.optimizer.param_groups:
                param_group['lr'] = float(value)

        annealing_targets.append(
            AnnealingTarget(
                name='lr',
                setter=_set_lr,
                initial=float(lr_spec.get('initial', args.lr)),
                final=float(lr_spec.get('final', 1e-6)),
                start_point=float(lr_spec.get('start_point', 0.0)),
                end_point=float(lr_spec.get('end_point', 1.0)),
                transform=str(lr_spec.get('transform', 'linear')),
                value_type='float',
            )
        )
        print(
            f"Learning rate annealing: {lr_spec.get('initial')} -> {lr_spec.get('final')} "
            f"({lr_spec.get('transform', 'linear')})"
        )

    if annealing_targets and args.timesteps_train > 0:
        # Ensure environments reflect initial values before training starts.
        for target in annealing_targets:
            target.setter(target.initial)
        callbacks.append(
            ScalarAnnealingCallback(
                total_timesteps=total_timesteps,
                targets=annealing_targets,
                verbose=0,
            )
        )
    
    return CallbackList(callbacks), eval_cb, train_ckpt_cb


def _train_if_needed(
    args: Any,
    model: PPO,
    callbacks: CallbackList,
    eval_cb,
    train_ckpt_cb,
    use_WB: bool,
    WB_path: str,
    model_name: str,
):
    if args.timesteps_train <= 0 or args.load_model:
        return model

    run = None  # Placeholder for wandb run (currently disabled)
    # run = _maybe_enable_wandb(use_WB, args, WB_path, model_name)

    training_fn = model.learn
    training_args = {"total_timesteps": args.timesteps_train, "callback": callbacks}
    profile_code('False', training_fn, **training_args)  # cProfile
    # exit(0)
    # Restore desired checkpoint (if model saving is enabled)
    if args.save_model:
        if args.restore_best_val_model:
            restored_model = eval_cb.restore_best_ckpt(model.get_env())
            if restored_model is not None:
                model = restored_model
        else:
            restored_model = train_ckpt_cb.restore_last_ckpt(model.get_env())
            if restored_model is not None:
                model = restored_model

    if run is not None:
        run.finish()
    return model


def _evaluate(args: Any, model: PPO, eval_env, kge_engine, sampler, data_handler: DataHandler) -> Tuple[dict, dict, dict]:
    print("\nTest set evaluation...")
    eval_mode = "hybrid" if bool(getattr(args, "inference_fusion", False)) else "rl_only"
    
    # Reseed before evaluation for deterministic parity testing
    # This ensures negative sampling in eval_corruptions matches tensor implementation
    deterministic_parity = getattr(args, "deterministic_parity", False)
    if deterministic_parity:
        import numpy as np
        eval_seed = 12345  # Fixed seed for eval_corruptions parity (same as test_eval_parity.py)
        torch.manual_seed(eval_seed)
        np.random.seed(eval_seed)

    depth_reward_tracker = _EvalDepthRewardTracker()

    eval_args = {
        "model": model,
        "env": eval_env,
        "data": data_handler.test_queries,
        "sampler": sampler,
        "n_corruptions": args.test_neg_samples,
        "verbose": 0,
        "kge_inference_engine": kge_engine,
        "evaluation_mode": eval_mode,
        "plot": args.plot,
        "corruption_scheme": args.corruption_scheme,
        "data_depths": data_handler.test_queries_depths,
        "info_callback": depth_reward_tracker,
        "hybrid_kge_weight": args.eval_hybrid_kge_weight,
        "hybrid_rl_weight": args.eval_hybrid_rl_weight,
        "hybrid_success_only": args.eval_hybrid_success_only,
    }
    metrics_test = profile_code(False, eval_corruptions, **eval_args)
    metrics_test.update(depth_reward_tracker.metrics())
    print("results for:", args.run_signature)
    print_eval_info("Test", metrics_test)

    # Optionally evaluate on val/train (kept default same as original)
    eval_only_test = True
    if not eval_only_test:
        metrics_valid = eval_corruptions(
            model,
            eval_env,
            data_handler.valid_queries,
            sampler,
            n_corruptions=args.eval_neg_samples,
            evaluation_mode=eval_mode,
            kge_inference_engine=kge_engine,
            corruption_scheme=args.corruption_scheme,
            data_depths=data_handler.valid_queries_depths,
            hybrid_kge_weight=args.eval_hybrid_kge_weight,
            hybrid_rl_weight=args.eval_hybrid_rl_weight,
            hybrid_success_only=args.eval_hybrid_success_only,
        )
        print_eval_info("Validation", metrics_valid)

        metrics_train = eval_corruptions(
            model,
            eval_env,
            data_handler.train_queries,
            sampler,
            n_corruptions=args.train_neg_ratio,
            evaluation_mode=eval_mode,
            kge_inference_engine=kge_engine,
            corruption_scheme=args.corruption_scheme,
            data_depths=data_handler.train_queries_depths,
            hybrid_kge_weight=args.eval_hybrid_kge_weight,
            hybrid_rl_weight=args.eval_hybrid_rl_weight,
            hybrid_success_only=args.eval_hybrid_success_only,
        )
        print_eval_info("Train", metrics_train)
    else:
        # Mirror original behavior: zero dicts with same keys as test
        metrics_train = {k: 0 for k in metrics_test.keys()}
        metrics_valid = {k: 0 for k in metrics_test.keys()}

    # Mirror torchrl trace logging: record eval metrics when trace recorder is available
    trace_recorder = getattr(model, "trace_recorder", None)
    if trace_recorder is not None:
        if metrics_valid:
            trace_recorder.log_eval("valid", metrics_valid)
        if metrics_test:
            trace_recorder.log_eval("test", metrics_test)
        trace_recorder.flush(f"{model.trace_prefix}_trace.jsonl")

    return metrics_train, metrics_valid, metrics_test



def main(args, log_filename, use_logger, use_WB, WB_path, date, external_components=None):
    """Main training function.
    
    Args:
        args: Configuration namespace
        log_filename: Log file path
        use_logger: Whether to use logging
        use_WB: Whether to use Weights & Biases
        WB_path: W&B path
        date: Date string
        external_components: Optional dict with pre-created components for parity testing:
            {'dh': DataHandler, 'index_manager': IndexManager, 'sampler': Sampler, 'embedder': Embedder}
    """
    _warn_non_reproducible(args)
    _set_seeds(args.seed_run_i)
    
    # Deterministic parity mode for exact alignment with tensor implementation
    deterministic_parity = getattr(args, 'deterministic_parity', False)

    # Normalize KGE flags
    args.kge_action = bool(getattr(args, "kge_action", False))
    args.logit_fusion = bool(getattr(args, "logit_fusion", False))
    args.inference_fusion = bool(getattr(args, "inference_fusion", False))
    args.inference_success_only = bool(getattr(args, "inference_success_only", True))
    args.pbrs = bool(getattr(args, "pbrs", False)) or float(getattr(args, "pbrs_beta", 0.0)) != 0.0
    args.enable_top_k = bool(getattr(args, "enable_top_k", False))

    device = get_device(args.device)
    print(f"Device: {device}. CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}")

    # Build pieces - use external components if provided (for parity testing)
    kge_engine = None
    if external_components is not None:
        dh = external_components['dh']
        index_manager = external_components['index_manager']
        sampler = external_components['sampler']
        embedder = external_components['embedder']
    else:
        dh, index_manager, sampler, embedder = _build_data_and_index(args, device)
    
    # PARITY: Reseed before environment creation for deterministic alignment
    if deterministic_parity:
        _set_seeds(args.seed_run_i)
    
    env, eval_env, callback_env = create_environments(
        args,
        dh,
        index_manager,
        kge_engine=kge_engine,
        detailed_eval_env=args.extended_eval_info,
    )

    # --- INIT MODEL ---
    enable_top_k = getattr(args, 'enable_top_k', False)
    top_k_value = getattr(args, 'top_k_value', None) if enable_top_k else None
    
    policy_kwargs = {
        'features_extractor_class': CustomCombinedExtractor,
        'features_extractor_kwargs': {'features_dim': embedder.embed_dim, 'embedder': embedder},
        # 'enable_top_k': enable_top_k,
        # 'top_k_value': top_k_value,
        # 'enable_kge_action': args.kge_action,
        # 'enable_logit_fusion': args.logit_fusion,
        # 'kge_logit_gain_initial': getattr(args, 'kge_logit_init_value', 1.0),
        # 'kge_logit_transform': getattr(args, 'kge_logit_transform', 'log'),
        # 'kge_logit_eps': getattr(args, 'kge_logit_eps', 1e-6),
    }

    # PARITY: Reseed before model creation for deterministic alignment
    if deterministic_parity:
        _set_seeds(args.seed_run_i)
    
    model = PPO(
        CustomActorCriticPolicy,
        env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        verbose=1,
        device=device,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        gamma=args.gamma,
        clip_range_vf=args.clip_range_vf,
        target_kl=args.target_kl,
        policy_kwargs=policy_kwargs,
        trace_dir=getattr(args, "trace_dir", None),
        trace_prefix="sb3",
        seed=args.seed_run_i,
    )

    if getattr(args, "trace_dir", None):
        print(f"[TRACE] SB3 trace recorder: {model.trace_recorder}")
    
    # Optional policy KGE wiring
    _attach_kge_to_policy(
        model,
        index_manager,
        kge_engine,
        device,
        args,
    )



    # --- TRAIN ---

    # Preload if requested
    model_path = _model_dir(args, date)
    _loaded, model = _maybe_load_model(args, device, eval_env, date, model)

    # Train if needed
    if args.timesteps_train > 0 and not args.load_model:
        model_name = f"{date}"
        callbacks, eval_cb, train_ckpt_cb = _build_callbacks(
            args,
            model,
            callback_env,
            eval_env,
            sampler,
            dh,
            log_filename,
            model_path,
            model_name,
        )
        model = _train_if_needed(
            args, model, callbacks, eval_cb, train_ckpt_cb, use_WB, WB_path, model_name
        )


    # --- TEST ---

    # --- freeze Dropout & LayerNorm ---
    model.policy.apply(_freeze_dropout_layernorm)

    # Re-attach KGE (if loaded model or after compile) and compile policy
    _attach_kge_to_policy(
        model,
        index_manager,
        kge_engine,
        device,
        args,
    )

    model.policy = torch.compile(model.policy, mode="reduce-overhead", fullgraph=False)
    model.policy.set_training_mode(False)

    # ------- Evaluate -------
    metrics_train, metrics_valid, metrics_test = _evaluate(
        args, model, eval_env, kge_engine, sampler, dh
    )

    return metrics_train, metrics_valid, metrics_test
