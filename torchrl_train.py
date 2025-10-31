"""
TorchRL-based training script for Neural-guided Grounding.

This module provides the main training loop using TorchRL's PPO implementation,
migrated from the original Stable-Baselines3 version.
"""

import json
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

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
from torchrl_custom_env import create_environments
from dataset import DataHandler
from torchrl_model import create_torchrl_modules
from embeddings import get_embedder
from neg_sampling import get_sampler
from torchrl_model_eval import eval_corruptions_torchrl, TorchRLPolicyWrapper

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


def _save_checkpoint(
    actor: nn.Module,
    critic: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    timestep: int,
    metrics: dict,
    save_path: Path,
    prefix: str = "checkpoint",
):
    """Save model checkpoint."""
    save_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'timestep': timestep,
        'metrics': metrics,
    }
    
    filename = save_path / f"{prefix}_epoch_{epoch}_step_{timestep}.pt"
    torch.save(checkpoint, filename)
    print(f"Saved checkpoint to {filename}")
    return filename


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
    loss_module: ClipPPOLoss,
    advantage_module: GAE,
    train_env,
    eval_env,
    sampler,
    data_handler: DataHandler,
    model_path: Path,
    device: torch.device,
) -> Tuple[nn.Module, nn.Module]:
    """
    Main TorchRL training loop with full metrics and logging.
    """
    if args.timesteps_train <= 0:
        print("No training steps requested (timesteps_train <= 0)")
        return actor, critic
    
    # Training hyperparameters
    n_steps = args.n_steps
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    total_timesteps = args.timesteps_train
    
    # Calculate training parameters
    steps_per_epoch = n_steps * args.n_envs
    n_iterations = total_timesteps // steps_per_epoch
    
    print(f"\n{'='*60}")
    print(f"Starting TorchRL PPO Training")
    print(f"{'='*60}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Steps per iteration: {steps_per_epoch}")
    print(f"Number of iterations: {n_iterations}")
    print(f"Mini-batch size: {batch_size}")
    print(f"Optimization epochs per iteration: {n_epochs}")
    print(f"{'='*60}\n")
    
    # Initialize logger
    logger = TrainingLogger(
        log_dir=model_path,
        use_tensorboard=True,
        use_wandb=False,
    )
    
    # Tracking variables
    global_step = 0
    best_eval_metric = float('-inf')
    best_model_path = None
    
    # Training loop
    for iteration in range(n_iterations):
        iteration_start_time = time.time()
        
        # ==================== Data Collection ====================
        print(f"\nIteration {iteration + 1}/{n_iterations}")
        print(f"  Collecting {steps_per_epoch} steps...")
        
        experiences = []
        td = train_env.reset()
        
        for step in range(n_steps):
            # Get actions from policy
            with torch.no_grad():
                # Get the underlying actor-critic model
                # The critic is simpler: TensorDictModule(TorchRLValueModule(actor_critic_model))
                # Access it from the critic module
                critic_inner = critic._module if hasattr(critic, '_module') else critic.module
                actor_critic_model = critic_inner.actor_critic_model
                
                # Get logits and values directly from the model
                logits = actor_critic_model.forward_actor(td.clone())
                values = actor_critic_model.forward_critic(td.clone())
                
                # Remove extra dimensions if present
                if logits.dim() == 3 and logits.shape[0] == logits.shape[1]:
                    # Shape is [batch, batch, actions] - squeeze first dim
                    logits = logits[:,0,:]  # Take first slice
                if values.dim() == 2 and values.shape[0] == values.shape[1]:
                    values = values[:,0] if values.shape[1] == 1 else values[0,:]
                
                # Sample actions from logits
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                action_indices = dist.sample()  # Should be shape (n_envs,)
                log_probs = dist.log_prob(action_indices)
                
                # DEBUG
                # print(f"DEBUG: logits.shape={logits.shape}, action_indices.shape={action_indices.shape}, n_envs={args.n_envs}")
                
                # Convert to one-hot for storage
                action_one_hot = torch.nn.functional.one_hot(action_indices, num_classes=logits.shape[-1]).float()
                
                # Store current state with action and value
                experience = TensorDict({
                    "sub_index": td["sub_index"],
                    "derived_sub_indices": td["derived_sub_indices"],
                    "action_mask": td["action_mask"],
                    "action": action_one_hot,
                    "sample_log_prob": log_probs,
                    "state_value": values,
                }, batch_size=torch.Size([args.n_envs]))
                
                # Ensure actions are valid by checking mask
                action_mask = td["action_mask"]
                for env_idx in range(args.n_envs):
                    action_idx = action_indices[env_idx].item()
                    if not action_mask[env_idx, action_idx].item():
                        # If chosen action is invalid, pick first valid action
                        valid_actions = torch.where(action_mask[env_idx])[0]
                        if len(valid_actions) > 0:
                            action_indices[env_idx] = valid_actions[0]
                        else:
                            # No valid actions, use 0 (will likely fail but avoid crash)
                            action_indices[env_idx] = 0
            
            # Step environment
            # Update the td with actions for stepping
            td["action"] = action_indices
            next_td = train_env.step(td)
            
            # Add next state info to experience
            # TorchRL's step() puts reward/done in ("next", "reward") structure
            experience["next"] = TensorDict({
                "sub_index": next_td["next"]["sub_index"],
                "derived_sub_indices": next_td["next"]["derived_sub_indices"],
                "action_mask": next_td["next"]["action_mask"],
                "reward": next_td["next", "reward"],
                "done": next_td["next", "done"],
            }, batch_size=torch.Size([args.n_envs]))
            
            experiences.append(experience)
            
            # Update state for next iteration
            # Extract the next state from the nested structure
            td = TensorDict({
                "sub_index": next_td["next"]["sub_index"],
                "derived_sub_indices": next_td["next"]["derived_sub_indices"],
                "action_mask": next_td["next"]["action_mask"],
            }, batch_size=torch.Size([args.n_envs]))
            global_step += args.n_envs
        
        # Stack experiences
        # Shape: (n_steps, n_envs, ...)
        batch = torch.stack(experiences, dim=0)
        
        # Flatten batch for training: (n_steps * n_envs, ...)
        flat_batch_size = n_steps * args.n_envs
        
        print(f"  Collected {flat_batch_size} transitions")
        
        # ==================== Compute Advantages ====================
        with torch.no_grad():
            # Extract values, rewards, dones
            # Shape: (n_steps, n_envs)
            rewards = torch.stack([batch[i]["next"]["reward"] for i in range(n_steps)]).to(device)
            values = torch.stack([batch[i]["state_value"] for i in range(n_steps)]).squeeze(-1).to(device)
            dones = torch.stack([batch[i]["next"]["done"] for i in range(n_steps)]).float().to(device)
            
            # Squeeze extra dimensions if present
            if rewards.dim() == 3 and rewards.shape[-1] == 1:
                rewards = rewards.squeeze(-1)
            if values.dim() == 3 and values.shape[-1] == 1:
                values = values.squeeze(-1)
            if dones.dim() == 3 and dones.shape[-1] == 1:
                dones = dones.squeeze(-1)
            
            # Compute GAE
            advantages = torch.zeros_like(rewards, device=device)
            returns = torch.zeros_like(rewards, device=device)
            gae = torch.zeros(args.n_envs, device=device)
            next_value = torch.zeros(args.n_envs, device=device)  # Assume 0 for terminal states
            
            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    next_non_terminal = 1.0 - dones[t]
                    next_value_t = next_value
                else:
                    next_non_terminal = 1.0 - dones[t]
                    next_value_t = values[t + 1]
                
                delta = rewards[t] + args.gamma * next_value_t * next_non_terminal - values[t]
                gae = delta + args.gamma * 0.95 * next_non_terminal * gae  # lambda = 0.95
                advantages[t] = gae
                returns[t] = gae + values[t]
        
        # Normalize advantages
        advantages_flat = advantages.reshape(-1)
        advantages_normalized = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
        
        print(f"  Advantage mean: {advantages_flat.mean().item():.4f}, std: {advantages_flat.std().item():.4f}")
        
        # ==================== Policy Optimization ====================
        print(f"  Optimizing policy for {n_epochs} epochs...")
        
        # Flatten batch for training
        obs_flat = torch.cat([batch[i]["sub_index"] for i in range(n_steps)], dim=0).to(device)
        actions_flat = torch.cat([batch[i]["derived_sub_indices"] for i in range(n_steps)], dim=0).to(device)
        masks_flat = torch.cat([batch[i]["action_mask"] for i in range(n_steps)], dim=0).to(device)
        old_actions_flat = torch.cat([batch[i]["action"] for i in range(n_steps)], dim=0).to(device)
        old_log_probs_flat = torch.cat([batch[i]["sample_log_prob"] for i in range(n_steps)], dim=0).to(device)
        returns_flat = returns.reshape(-1)
        
        # Create mini-batches
        indices = torch.randperm(flat_batch_size, device=device)
        num_batches = flat_batch_size // batch_size
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for epoch in range(n_epochs):
            for i in range(num_batches):
                batch_indices = indices[i * batch_size:(i + 1) * batch_size]
                
                # Get mini-batch (ensure all on device)
                mb_obs = obs_flat[batch_indices].to(device)
                mb_actions = actions_flat[batch_indices].to(device)
                mb_masks = masks_flat[batch_indices].to(device)
                mb_old_actions = old_actions_flat[batch_indices].to(device)
                mb_old_log_probs = old_log_probs_flat[batch_indices].to(device)
                mb_advantages = advantages_normalized[batch_indices].to(device)
                mb_returns = returns_flat[batch_indices].to(device)
                
                # Forward pass
                mb_td = TensorDict({
                    "sub_index": mb_obs,
                    "derived_sub_indices": mb_actions,
                    "action_mask": mb_masks,
                }, batch_size=torch.Size([batch_size]))
                
                td_with_action = actor(mb_td)
                td_with_value = critic(mb_td)
                
                new_log_probs = td_with_action.get("sample_log_prob", torch.zeros(batch_size, device=device)).to(device)
                new_values = td_with_value["state_value"].to(device)
                
                # Policy loss (PPO clip)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = 0.5 * ((new_values.squeeze() - mb_returns) ** 2).mean()
                
                # Entropy bonus (approximate)
                entropy = -new_log_probs.mean()
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss - args.ent_coef * entropy
                
                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()),
                    max_norm=0.5
                )
                optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        avg_policy_loss = total_policy_loss / (n_epochs * num_batches)
        avg_value_loss = total_value_loss / (n_epochs * num_batches)
        avg_entropy = total_entropy / (n_epochs * num_batches)
        
        iteration_time = time.time() - iteration_start_time
        
        # Compute mean reward from collected experiences
        mean_reward = rewards.mean().item()
        
        # Log training metrics
        logger.log_training_step(
            iteration=iteration + 1,
            global_step=global_step,
            policy_loss=avg_policy_loss,
            value_loss=avg_value_loss,
            entropy=avg_entropy,
            mean_reward=mean_reward,
        )
        
        print(f"  Mean reward: {mean_reward:.4f}")
        print(f"  Iteration time: {iteration_time:.2f}s")
        
        # ==================== Evaluation ====================
        if (iteration + 1) % 5 == 0 or (iteration + 1) == n_iterations:
            print(f"\n  Running corruption-based evaluation...")
            eval_metrics = _evaluate_during_training(
                args, actor, eval_env, sampler, data_handler, verbose=1
            )
            
            # Log evaluation metrics
            logger.log_evaluation(
                iteration=iteration + 1,
                global_step=global_step,
                metrics=eval_metrics,
                prefix="valid",
            )
            
            # Check if this is the best model
            eval_metric = eval_metrics.get(args.eval_best_metric, 0)
            print(f"  Eval {args.eval_best_metric}: {eval_metric:.4f} (best: {best_eval_metric:.4f})")
            
            if eval_metric > best_eval_metric:
                best_eval_metric = eval_metric
                best_model_path = _save_checkpoint(
                    actor, critic, optimizer,
                    epoch=iteration + 1,
                    timestep=global_step,
                    metrics=eval_metrics,
                    save_path=model_path,
                    prefix="best_eval",
                )
                print(f"  ★ New best model saved!")
        
        # ==================== Periodic Checkpoint ====================
        if (iteration + 1) % 10 == 0:
            _save_checkpoint(
                actor, critic, optimizer,
                epoch=iteration + 1,
                timestep=global_step,
                metrics={'policy_loss': avg_policy_loss, 'value_loss': avg_value_loss, 'entropy': avg_entropy},
                save_path=model_path,
                prefix="last_epoch",
            )
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}\n")
    
    # Close logger
    logger.close()
    
    # Restore best model if requested
    if args.restore_best_val_model and best_model_path is not None:
        print(f"Restoring best model from {best_model_path}")
        _load_checkpoint(actor, critic, optimizer, best_model_path, device)
    
    return actor, critic


def _evaluate_during_training(
    args: Any,
    actor: nn.Module,
    eval_env,
    sampler,
    data_handler: DataHandler,
    verbose: int = 0,
) -> dict:
    """
    Run corruption-based evaluation during training.
    
    This function uses the full eval_corruptions_torchrl pipeline to compute
    MRR, Hits@K, and other ranking metrics on the validation set.
    """
    actor.eval()
    
    # Prepare evaluation data
    eval_data = data_handler.valid_queries[:args.n_eval_queries]
    eval_depths = (
        data_handler.valid_depths[:args.n_eval_queries]
        if hasattr(data_handler, 'valid_depths') and data_handler.valid_depths is not None
        else None
    )
    
    # Run corruption-based evaluation
    try:
        metrics = eval_corruptions_torchrl(
            actor=actor,
            env=eval_env,
            data=eval_data,
            sampler=sampler,
            n_corruptions=args.eval_neg_samples,
            deterministic=True,
            verbose=verbose,
            plot=False,
            kge_inference_engine=None,
            evaluation_mode='rl_only',
            corruption_scheme=['head', 'tail'],
            info_callback=None,
            data_depths=eval_depths,
        )
    except Exception as e:
        print(f"Warning: Evaluation failed with error: {e}")
        print("Returning dummy metrics.")
        metrics = {
            'mrr_mean': 0.0,
            'hits1_mean': 0.0,
            'hits3_mean': 0.0,
            'hits10_mean': 0.0,
            'rewards_pos_mean': 0.0,
            'success_rate': 0.0,
        }
    
    actor.train()
    
    return metrics


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
    
    # Train set (quick check, fewer corruptions)
    if len(train_data) > 0:
        print("\n--- Evaluating on TRAIN set ---")
        try:
            metrics_train = eval_corruptions_torchrl(
                actor=actor,
                env=eval_env,
                data=train_data[:min(100, len(train_data))],  # Sample for speed
                sampler=sampler,
                n_corruptions=args.eval_neg_samples,
                deterministic=True,
                verbose=1,
                plot=False,
                evaluation_mode='rl_only',
                corruption_scheme=['head', 'tail'],
                data_depths=train_depths[:min(100, len(train_data))] if train_depths else None,
            )
            print_eval_info("TRAIN", metrics_train)
        except Exception as e:
            print(f"Warning: Train evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            metrics_train = _get_dummy_metrics()
    
    # Valid set
    if len(valid_data) > 0:
        print("\n--- Evaluating on VALID set ---")
        try:
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
                data_depths=valid_depths,
            )
            print_eval_info("VALID", metrics_valid)
        except Exception as e:
            print(f"Warning: Valid evaluation failed: {e}")
            metrics_valid = _get_dummy_metrics()
    
    # Test set (full evaluation)
    if len(test_data) > 0:
        print("\n--- Evaluating on TEST set ---")
        try:
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
                data_depths=test_depths,
            )
            print_eval_info("TEST", metrics_test)
        except Exception as e:
            print(f"Warning: Test evaluation failed: {e}")
            metrics_test = _get_dummy_metrics()
    
    print("\n" + "="*60)
    print("Final Evaluation Complete")
    print("="*60 + "\n")
    
    return metrics_train, metrics_valid, metrics_test


def _get_dummy_metrics() -> dict:
    """Return dummy metrics in case of evaluation failure."""
    return {
        'mrr_mean': 0.0,
        'hits1_mean': 0.0,
        'hits3_mean': 0.0,
        'hits10_mean': 0.0,
        'rewards_pos_mean': 0.0,
        'rewards_neg_mean': 0.0,
        'episode_len_pos_mean': 0.0,
        'episode_len_neg_mean': 0.0,
    }


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
    
    # --- CREATE LOSS MODULES ---
    advantage_module = GAE(
        gamma=args.gamma,
        lmbda=0.95,
        value_network=critic,
        average_gae=True,
    )
    
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=args.clip_range,
        entropy_coeff=args.ent_coef,
        critic_coeff=0.5,
        loss_critic_type="smooth_l1",
    )
    
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
            loss_module,
            advantage_module,
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
