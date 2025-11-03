"""
TorchRL-based training script for Neural-guided Grounding.

This module provides the main training loop using TorchRL's PPO implementation,
migrated from the original Stable-Baselines3 version.
"""

import json
import time
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from index_manager import IndexManager
from utils import (
    get_device,
    print_eval_info,
    profile_code,
    _set_seeds,
    _freeze_dropout_layernorm,
    _warn_non_reproducible,
    _maybe_enable_wandb,
    FileLogger,
)
from custom_env import create_environments
from dataset import DataHandler
from ppo import create_torchrl_modules, PPOAgent
from embeddings import get_embedder
from neg_sampling import get_sampler
from model_eval import eval_corruptions_torchrl, TorchRLPolicyWrapper
from callbacks import (
    RolloutProgressCallback,
    EvaluationCallback,
    TrainingMetricsCallback,
    TorchRLCallbackManager,
)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install tensorboard for advanced logging.")


# ------------------------------
# Initialization helpers
# ------------------------------

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

    # Respect caps from args while ensuring >1 eval query
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
    ), "Number of evaluation queries must be greater than 1."
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
    candidates = sorted([p for p in root.glob(f"*{keyword}*.pt")])
    return candidates[-1] if candidates else None



def _load_checkpoint(
    actor: nn.Module,
    critic: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[int, int, dict]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    timestep = checkpoint.get('timestep', 0)
    metrics = checkpoint.get('metrics', {})
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Epoch: {epoch}, Timestep: {timestep}")
    
    return epoch, timestep, metrics


# ------------------------------
# Logging utilities
# ------------------------------

class TrainingLogger:
    """
    Advanced logging for training metrics.
    
    Supports console output, file logging, and TensorBoard.
    """
    def __init__(
        self,
        log_dir: Path,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # File logger
        self.log_file = self.log_dir / "training_log.txt"
        
        # TensorBoard
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.tb_writer = None
        if self.use_tensorboard:
            tb_dir = self.log_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            print(f"TensorBoard logging to: {tb_dir}")
        
        # WandB (placeholder for future)
        self.use_wandb = use_wandb
        
        # Metrics history
        self.history = {
            'train': [],
            'eval': [],
        }
    
    def log_scalar(
        self,
        name: str,
        value: float,
        step: int,
        category: str = "train",
    ) -> None:
        """Log a scalar value."""
        # TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(f"{category}/{name}", value, step)
        
        # File
        with open(self.log_file, 'a') as f:
            f.write(f"[Step {step}] {category}/{name}: {value:.6f}\n")
    
    def log_dict(
        self,
        metrics: dict,
        step: int,
        category: str = "train",
        prefix: str = "",
    ) -> None:
        """Log a dictionary of metrics."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                full_name = f"{prefix}{key}" if prefix else key
                self.log_scalar(full_name, value, step, category)
    
    def log_training_step(
        self,
        iteration: int,
        global_step: int,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        mean_reward: float = None,
    ) -> None:
        """Log training step metrics."""
        metrics = {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
        }
        if mean_reward is not None:
            metrics['mean_reward'] = mean_reward
        
        self.log_dict(metrics, global_step, category="train")
    
    def log_evaluation(
        self,
        iteration: int,
        global_step: int,
        metrics: dict,
        prefix: str = "eval",
    ) -> None:
        """Log evaluation metrics."""
        self.log_dict(metrics, global_step, category="eval", prefix=prefix + "/")
        
        # Store in history
        self.history['eval'].append({
            'iteration': iteration,
            'step': global_step,
            'metrics': metrics.copy(),
        })
        
        # Console output (key metrics only)
        key_metrics = ['mrr_mean', 'hits1_mean', 'hits10_mean', 'rewards_pos_mean']
        msg = f"[Eval at Iter {iteration}]"
        for key in key_metrics:
            if key in metrics:
                msg += f" {key}={metrics[key]:.4f}"
        print(msg)
    
    def save_history(self) -> None:
        """Save training history to JSON file."""
        import json
        history_file = self.log_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved training history to: {history_file}")
    
    def close(self) -> None:
        """Close loggers."""
        if self.tb_writer is not None:
            self.tb_writer.close()
        self.save_history()


# ------------------------------
# Training loop
# ------------------------------

def _train(
    args: Any,
    actor: nn.Module,
    critic: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_env,
    eval_env,
    sampler,
    data_handler: DataHandler,
    model_path: Path,
    device: torch.device,
) -> Tuple[nn.Module, nn.Module]:
    """
    Main TorchRL training loop with full metrics and logging.
    
    This function uses the modular PPOAgent class to handle training.
    """
    if args.timesteps_train <= 0:
        print("No training steps requested (timesteps_train <= 0)")
        return actor, critic
    
    # Training hyperparameters
    n_steps = args.n_steps
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    total_timesteps = args.timesteps_train
    
    # Initialize logger
    logger = TrainingLogger(
        log_dir=model_path,
        use_tensorboard=True,
        use_wandb=False,
    )
    
    # Initialize callbacks
    rollout_callback = RolloutProgressCallback(
        total_steps=n_steps * args.n_envs,
        n_envs=args.n_envs,
        update_interval=25,
        verbose=True,
    )
    
    eval_callback = EvaluationCallback(
        eval_env=eval_env,
        sampler=sampler,
        eval_data=data_handler.valid_queries[:args.n_eval_queries],
        eval_data_depths=(
            data_handler.valid_depths[:args.n_eval_queries]
            if hasattr(data_handler, 'valid_depths') and data_handler.valid_depths is not None
            else None
        ),
        n_corruptions=args.eval_neg_samples,
        eval_freq=1,  # Evaluate every iteration
        best_metric=args.eval_best_metric if hasattr(args, 'eval_best_metric') else "mrr_mean",
        save_path=model_path,
        verbose=True,
        collect_detailed=args.depth_info,
        verbose_cb=getattr(args, 'verbose_cb', False),  # Enable verbose callback debugging
    )
    
    train_callback = TrainingMetricsCallback(
        log_interval=1,
        verbose=True,
        verbose_cb=getattr(args, 'verbose_cb', False),  # Enable verbose callback debugging
        collect_detailed=args.depth_info 
    )
    
    callback_manager = TorchRLCallbackManager(
        rollout_callback=rollout_callback,
        eval_callback=eval_callback,
        train_callback=train_callback,
    )
    
    # Notify callbacks of training start
    callback_manager.on_training_start()
    
    # Create PPO agent
    ppo_agent = PPOAgent(
        actor=actor,
        critic=critic,
        optimizer=optimizer,
        train_env=train_env,
        eval_env=eval_env,
        sampler=sampler,
        data_handler=data_handler,
        args=args,
        n_envs=args.n_envs,
        n_steps=n_steps,
        n_epochs=n_epochs,
        batch_size=batch_size,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        value_coef=0.5,
        max_grad_norm=0.5,
        device=device,
        model_save_path=model_path,
        eval_best_metric=args.eval_best_metric if hasattr(args, 'eval_best_metric') else "mrr_mean",
        verbose_cb=getattr(args, 'verbose_cb', False),  # Enable verbose callback debugging
    )
    
    # Train the agent
    actor, critic = ppo_agent.train(
        total_timesteps=total_timesteps,
        eval_callback=callback_manager,
        rollout_callback=callback_manager.rollout_callback,
        callback_manager=callback_manager,
        logger=logger,
    )
    
    # Close logger
    logger.close()
    
    # Restore best model if requested
    if args.restore_best_val_model and ppo_agent.best_model_path is not None:
        print(f"Restoring best model from {ppo_agent.best_model_path}")
        ppo_agent.load_checkpoint(ppo_agent.best_model_path)
    
    return actor, critic


def _evaluate(
    args: Any,
    actor: nn.Module,
    eval_env,
    kge_engine,
    sampler,
    data_handler: DataHandler
) -> Tuple[dict, dict, dict]:
    """
    Final evaluation on train/valid/test sets with full corruption-based metrics.
    
    Returns:
        Tuple of (train_metrics, valid_metrics, test_metrics)
    """
    from sb3_code.sb3_callbacks import _EvalDepthRewardTracker
    
    print("\n" + "="*60)
    print("Running Final Evaluation with Corruption-based Metrics")
    print("="*60)
    
    actor.eval()
    
    # Prepare datasets
    train_data = data_handler.train_queries[:args.n_train_queries]
    valid_data = data_handler.valid_queries[:args.n_eval_queries]
    test_data = data_handler.test_queries[:args.n_test_queries]
    
    train_depths = (
        data_handler.train_depths[:args.n_train_queries]
        if hasattr(data_handler, 'train_depths') and data_handler.train_depths is not None
        else None
    )
    valid_depths = (
        data_handler.valid_depths[:args.n_eval_queries]
        if hasattr(data_handler, 'valid_depths') and data_handler.valid_depths is not None
        else None
    )
    test_depths = (
        data_handler.test_depths[:args.n_test_queries]
        if hasattr(data_handler, 'test_depths') and data_handler.test_depths is not None
        else None
    )
    
    # Evaluate on each split
    metrics_train = {}
    metrics_valid = {}
    metrics_test = {}
    
    # Valid set
    if len(valid_data) > 0:
        print("\n--- Evaluating on VALID set ---")
        try:
            # Create depth tracker for this evaluation
            depth_tracker = _EvalDepthRewardTracker()
            
            metrics_valid = eval_corruptions_torchrl(
                actor=actor,
                env=eval_env,
                data=valid_data,
                sampler=sampler,
                n_corruptions=args.eval_neg_samples,
                deterministic=True,
                verbose=1,
                plot=False,
                evaluation_mode='rl_only',
                corruption_scheme=['head', 'tail'],
                info_callback=depth_tracker,
                data_depths=valid_depths,
            )
            
            # Merge depth-based metrics into results
            depth_metrics = depth_tracker.metrics()
            metrics_valid.update(depth_metrics)
            
            print_eval_info("VALID", metrics_valid)
        except Exception as e:
            print(f"Warning: Valid evaluation failed: {e}")
    
    # Test set (full evaluation)
    if len(test_data) > 0:
        print("\n--- Evaluating on TEST set ---")
        try:
            # Create depth tracker for this evaluation
            depth_tracker = _EvalDepthRewardTracker()
            
            metrics_test = eval_corruptions_torchrl(
                actor=actor,
                env=eval_env,
                data=test_data,
                sampler=sampler,
                n_corruptions=args.test_neg_samples,
                deterministic=True,
                verbose=1,
                plot=args.plot_trajectories if hasattr(args, 'plot_trajectories') else False,
                evaluation_mode='rl_only',
                corruption_scheme=['head', 'tail'],
                info_callback=depth_tracker,
                data_depths=test_depths,
            )
            
            # Merge depth-based metrics into results
            depth_metrics = depth_tracker.metrics()
            metrics_test.update(depth_metrics)
            
            print_eval_info("TEST", metrics_test)
        except Exception as e:
            print(f"Warning: Test evaluation failed: {e}")
    
    print("\n" + "="*60)
    print("Final Evaluation Complete")
    print("="*60 + "\n")
    
    return metrics_train, metrics_valid, metrics_test


# ------------------------------
# Main training function
# ------------------------------

def main(args, log_filename, use_logger, use_WB, WB_path, date):
    """Main training function for TorchRL-based PPO."""
    
    _warn_non_reproducible(args)
    _set_seeds(args.seed_run_i)
    
    # Normalize flags (no KGE integration in this version)
    args.kge_action = False
    args.logit_fusion = False
    args.inference_fusion = False
    args.pbrs = False
    args.enable_top_k = False
    
    device = get_device(args.device)
    print(f"Device: {device}. CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}")
    
    # Build data and index
    kge_engine = None  # No KGE in this version
    dh, index_manager, sampler, embedder = _build_data_and_index(args, device)
    
    # Create environments
    env, eval_env, callback_env = create_environments(
        args,
        dh,
        index_manager,
        kge_engine=kge_engine,
        detailed_eval_env=args.extended_eval_info,
    )
    
    # --- CREATE MODEL ---
    print("\nCreating TorchRL actor-critic model...")
    
    actor, critic = create_torchrl_modules(
        embedder=embedder,
        num_actions=args.padding_states,
        embed_dim=args.state_embedding_size,
        hidden_dim=128,
        num_layers=8,
        dropout_prob=0.2,
        device=device,
        enable_kge_action=False,
        kge_inference_engine=None,
        index_manager=None,
    )
    
    print(f"Actor parameters: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"Critic parameters: {sum(p.numel() for p in critic.parameters()):,}")
    
    # --- CREATE OPTIMIZER ---
    # Deduplicate parameters since actor and critic share the same underlying model
    params_dict = {id(p): p for p in list(actor.parameters()) + list(critic.parameters())}
    optimizer = torch.optim.Adam(params_dict.values(), lr=args.lr)
    
    # --- LOAD MODEL IF REQUESTED ---
    model_path = _model_dir(args, date)
    start_epoch = 0
    
    if args.load_model:
        try:
            ckpt = _resolve_ckpt_to_load(model_path, restore_best=args.restore_best_val_model)
            if ckpt is not None:
                start_epoch, _, _ = _load_checkpoint(actor, critic, optimizer, ckpt, device)
            else:
                print(f"No checkpoint found in {model_path}, starting from scratch.")
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            if args.timesteps_train == 0:
                raise ValueError("In eval-only mode but could not load model.")
    
    # --- TRAIN ---
    if args.timesteps_train > 0 and not args.load_model:
        actor, critic = _train(
            args,
            actor,
            critic,
            optimizer,
            env,
            eval_env,
            sampler,
            dh,
            model_path,
            device,
        )
    
    # --- TEST ---
    print("\nFreezing dropout and layer normalization for evaluation...")
    actor.apply(_freeze_dropout_layernorm)
    critic.apply(_freeze_dropout_layernorm)
    
    actor.eval()
    critic.eval()
    
    # Evaluate
    metrics_train, metrics_valid, metrics_test = _evaluate(
        args, actor, eval_env, kge_engine, sampler, dh
    )
    
    return metrics_train, metrics_valid, metrics_test
