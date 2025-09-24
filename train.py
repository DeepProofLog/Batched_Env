from typing import Any, Optional, Tuple
import numpy as np
import os
import random
import torch
from pathlib import Path
import wandb
from wandb.integration.sb3 import WandbCallback
import torch.nn as nn

from index_manager import IndexManager
from utils import get_device, print_eval_info, profile_code
from callbacks import SB3TrainCheckpoint, CustomEvalCallbackMRR, CustomEvalCallback
from custom_dummy_env import create_environments
from dataset import DataHandler
from model import CustomActorCriticPolicy, CustomCombinedExtractor, PPO_custom as PPO
from embeddings import get_embedder
from neg_sampling import get_sampler
from model_eval import eval_corruptions
from kge_inference import KGEInference
from stable_baselines3.common.callbacks import (
    StopTrainingOnRewardThreshold,
    CallbackList,
    StopTrainingOnNoModelImprovement,
)


def _freeze_dropout_layernorm(m: nn.Module):
    if isinstance(m, (nn.Dropout, nn.LayerNorm)):
        m.eval()          # freeze statistics
        m.training = False

def _set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def _warn_non_reproducible(args: Any) -> None:
    if args.restore_best_val_model is False:
        print(
            "Warning: This setting is not reproducible when creating 2 models from scratch, "
            "but it is when loading pretrained models. You can use\n"
            "  export CUBLAS_WORKSPACE_CONFIG=:16:8; export PYTHONHASHSEED=0\n"
            "to make runs reproducible."
        )

def _init_kge_engine(args: Any) -> Optional[KGEInference]:
    """Create KGE inference engine when requested."""
    if args.use_kge_action or args.kge_integration_strategy == "sum_eval":
        print("\nInitializing KGE Inference Engine...", flush=True)
        engine = KGEInference(
            dataset_name=args.dataset_name,
            base_path=args.data_path,
            checkpoint_dir=args.kge_checkpoint_dir,
            run_signature=args.kge_run_signature,
            seed=0,
            scores_file_path=args.kge_scores_file,
        )
        print("KGE Inference Engine Initialized.\n")
        return engine
    return None

def _build_data_and_index(args: Any, device: torch.device) -> Tuple[DataHandler, IndexManager, Any, Any]:
    """Prepare DataHandler, IndexManager, sampler and embedder."""
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
        n_eval_queries=args.n_eval_queries,
        n_test_queries=args.n_test_queries,
        corruption_mode=args.corruption_mode,
        train_depth=args.train_depth,
        valid_depth=args.valid_depth,
        test_depth=args.test_depth,
    )

    # Respect caps from args while ensuring >1 eval query for callbacks
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
    )
    im.build_fact_index(dh.facts)

    # Negative sampler
    dh.sampler = get_sampler(
        data_handler=dh,
        index_manager=im,
        corruption_scheme=args.corruption_scheme,
        device=device,
    )
    sampler = dh.sampler

    # Embedder
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
            n_corruptions=args.eval_neg_samples,
            model_path=str(model_path) if args.save_model else None,
            log_path=log_filename,
            eval_freq=max(int(args.eval_freq // args.n_envs), 1),
            n_eval_episodes=args.n_eval_queries - 1,
            deterministic=True,
            render=False,
            name=model_name,
            callback_on_new_best=(
                reward_threshold_cb if args.restore_best_val_model and not args.use_kge_action else None
            ),
            corruption_scheme=args.corruption_scheme,
            # callback_after_eval=no_improvement_callback,
            verbose=0,
        )
    else:
        eval_cb = CustomEvalCallback(
            eval_env=callback_env,
            model_path=str(model_path) if args.save_model else None,
            log_path=log_filename,
            eval_freq=max(int(args.eval_freq // args.n_envs), 1),
            n_eval_episodes=args.n_eval_queries - 1,
            deterministic=True,
            render=False,
            name=model_name,
            # callback_on_new_best=reward_threshold_callback if args.restore_best_val_model \
            #                                        and not args.use_kge_action else None,
            # callback_after_eval=no_improvement_callback,
        )

    train_ckpt_cb = SB3TrainCheckpoint(
        model,
        monitor="rollout/ep_rew_mean",
        frequency=5,
        model_path=str(model_path) if args.save_model else None,
        name=model_name,
    )

    callbacks.extend([eval_cb, train_ckpt_cb])
    return CallbackList(callbacks), eval_cb, train_ckpt_cb


def _maybe_enable_wandb(use_WB: bool, args: Any, WB_path: str, model_name: str):
    if not use_WB:
        return None
    return wandb.init(
        project="RL-NeSy",
        group=args.run_signature,
        name=model_name,
        dir=WB_path,
        sync_tensorboard=True,
        config=vars(args),
    )


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

    run = _maybe_enable_wandb(use_WB, args, WB_path, model_name)

    training_fn = model.learn
    training_args = {"total_timesteps": args.timesteps_train, "callback": callbacks}
    profile_code(False, training_fn, **training_args)  # cProfile
    # exit(0)
    # Restore desired checkpoint
    if args.restore_best_val_model:
        model = eval_cb.restore_best_ckpt(model.get_env())
    else:
        model = train_ckpt_cb.restore_last_ckpt(model.get_env())

    if run is not None:
        run.finish()
    return model



def _attach_kge_to_policy(model: PPO, im: IndexManager, engine: Optional[KGEInference], device: torch.device, use_kge_action: bool) -> None:
    if not use_kge_action:
        return
    model.policy.kge_inference_engine = engine
    model.policy.index_manager = im
    kge_indices = [idx for pred, idx in im.predicate_str2idx.items() if pred.endswith("_kge")]
    model.policy.kge_indices_tensor = torch.tensor(kge_indices, device=device, dtype=torch.int32)


def _eval_mode_from_args(args: Any) -> str:
    # For sum_eval, we fuse externally; otherwise model handles fusion
    return "hybrid" if args.kge_integration_strategy == "sum_eval" else "rl_only"


def _evaluate(args: Any, model: PPO, eval_env, kge_engine, sampler, data_handler: DataHandler) -> Tuple[dict, dict, dict]:
    print("\nTest set evaluation...")
    eval_mode = _eval_mode_from_args(args)

    eval_args = {
        "model": model,
        "env": eval_env,
        "data": data_handler.test_queries,
        "sampler": sampler,
        "n_corruptions": args.test_neg_samples,
        "verbose": 2,
        "kge_inference_engine": kge_engine,
        "evaluation_mode": eval_mode,
        "plot": args.plot,
        "corruption_scheme": args.corruption_scheme,
    }
    metrics_test = profile_code(False, eval_corruptions, **eval_args)
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
        )
        print_eval_info("Train", metrics_train)
    else:
        # Mirror original behavior: zero dicts with same keys as test
        metrics_train = {k: 0 for k in metrics_test.keys()}
        metrics_valid = {k: 0 for k in metrics_test.keys()}

    return metrics_train, metrics_valid, metrics_test



def main(args, log_filename, use_logger, use_WB, WB_path, date):

    _warn_non_reproducible(args)
    _set_seeds(args.seed_run_i)

    device = get_device(args.device)
    print(f"Device: {device}. CUDA available: {torch.cuda.is_available()},\
                            Device count: {torch.cuda.device_count()}")

    # Build pieces
    kge_engine = _init_kge_engine(args)
    dh, index_manager, sampler, embedder = _build_data_and_index(args, device)
    env, eval_env, callback_env = create_environments(args, dh, index_manager, detailed_eval_env=args.extended_eval_info)

    # --- INIT MODEL ---
    policy_kwargs = {
        'features_extractor_class': CustomCombinedExtractor,
        'features_extractor_kwargs': {'features_dim': embedder.embed_dim, 'embedder': embedder}
    }

    # if args.kge_integration_strategy and args.kge_integration_strategy is not 'sum_eval':
    #     policy_kwargs.update({
    #         'kge_inference_engine': kge_inference_engine,
    #         'index_manager': index_manager,
    #         'kge_integration_strategy': args.kge_integration_strategy,
    #     })

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
        policy_kwargs=policy_kwargs)
    
    # Optional policy KGE wiring
    _attach_kge_to_policy(model, index_manager, kge_engine, device, args.use_kge_action)



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
    _attach_kge_to_policy(model, index_manager, kge_engine, device, args.use_kge_action)

    model.policy = torch.compile(model.policy, mode="reduce-overhead", fullgraph=False)
    model.policy.set_training_mode(False)

    # ------- Evaluate -------
    metrics_train, metrics_valid, metrics_test = _evaluate(
        args, model, eval_env, kge_engine, sampler, dh
    )

    return metrics_train, metrics_valid, metrics_test