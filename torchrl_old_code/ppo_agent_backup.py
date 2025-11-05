"""
PPO Agent

This module provides the main PPO agent class that coordinates training.
"""

import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .ppo_rollout_backup import collect_rollouts
from .ppo_learner import PPOLearner


class PPOAgent:
    """
    PPO Agent for Neural-guided Grounding.
    
    This class coordinates the training process, managing rollout collection,
    learning, and evaluation.
    """
    
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_env: Any,
        eval_env: Any,
        sampler: Any,
        data_handler: Any,
        args: Any = None,
        n_envs: int = 128,
        n_steps: int = 128,
        n_epochs: int = 10,
        batch_size: int = 128,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: torch.device = torch.device('cpu'),
        model_save_path: Optional[Path] = None,
        eval_best_metric: str = "mrr_mean",
        verbose_cb: bool = False,  # NEW: Verbose callback debugging
    ):
        """
        Initialize the PPO agent.
        
        Args:
            actor: Actor network
            critic: Critic network
            optimizer: Optimizer
            train_env: Training environment
            eval_env: Evaluation environment
            sampler: Negative sampler
            data_handler: Data handler
            args: Training arguments (optional, for accessing eval parameters)
            n_envs: Number of parallel environments
            n_steps: Steps per rollout
            n_epochs: Optimization epochs per update
            batch_size: Mini-batch size
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clipping range
            ent_coef: Entropy coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Max gradient norm
            device: Device to use
            model_save_path: Path to save models
            eval_best_metric: Metric to use for best model selection
            verbose_cb: If True, print debug information during callback collection
        """
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.train_env = train_env
        self.eval_env = eval_env
        self.sampler = sampler
        self.data_handler = data_handler
        self.args = args
        self.verbose_cb = verbose_cb
        
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.device = device
        self.model_save_path = model_save_path
        self.eval_best_metric = eval_best_metric
        
        # Create learner
        self.learner = PPOLearner(
            actor=actor,
            critic=critic,
            optimizer=optimizer,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            value_coef=value_coef,
            max_grad_norm=max_grad_norm,
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device,
        )
        
        # Tracking
        self.best_eval_metric = float('-inf')
        self.best_model_path = None
        self.global_step = 0
    
    def train(
        self,
        total_timesteps: int,
        eval_callback: Optional[Callable] = None,
        rollout_callback: Optional[Callable] = None,
        callback_manager: Optional[Any] = None,
        logger: Optional[Any] = None,
    ) -> Tuple[nn.Module, nn.Module]:
        """
        Train the PPO agent.
        
        Args:
            total_timesteps: Total timesteps to train
            eval_callback: Optional evaluation callback
            rollout_callback: Optional rollout callback
            callback_manager: Optional callback manager for stats accumulation
            logger: Optional logger
            
        Returns:
            Tuple of (trained_actor, trained_critic)
        """
        if total_timesteps <= 0:
            print("No training steps requested (total_timesteps <= 0)")
            return self.actor, self.critic
        
        # Calculate training parameters
        steps_per_iteration = self.n_steps * self.n_envs
        n_iterations = total_timesteps // steps_per_iteration
        
        print(f"\n{'='*60}")
        print(f"Starting TorchRL PPO Training")
        print(f"{'='*60}")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Steps per iteration: {steps_per_iteration}")
        print(f"Number of iterations: {n_iterations}")
        print(f"Mini-batch size: {self.learner.batch_size}")
        print(f"Optimization epochs per iteration: {self.learner.n_epochs}")
        print(f"{'='*60}\n")
        
        # Training loop
        for iteration in range(n_iterations):
            iteration_start_time = time.time()
            print(f"\nIteration {iteration + 1}/{n_iterations}")
            
            # ==================== Data Collection ====================
            if rollout_callback is not None:
                rollout_callback.on_rollout_start()
            
            experiences, stats = collect_rollouts(
                env=self.train_env,
                actor=self.actor,
                critic=self.critic,
                n_envs=self.n_envs,
                n_steps=self.n_steps,
                device=self.device,
                rollout_callback=(
                    rollout_callback.on_step if rollout_callback is not None else None
                ),
            )
            
            if rollout_callback is not None:
                rollout_callback.on_rollout_end()
            
            # Update global step counter
            self.global_step += steps_per_iteration
            
            print(f"  Collected {steps_per_iteration} transitions")
            
            # Accumulate episode stats
            if callback_manager is not None and stats["episode_info"]:
                if self.verbose_cb:
                    print(f"[PPOAgent] Accumulating {len(stats['episode_info'])} episode stats for training")
                callback_manager.accumulate_episode_stats(stats["episode_info"], mode="train")
            print(f"  Rollout time: {time.time() - iteration_start_time:.2f}s\n")
            # ==================== Policy Optimization ====================
            print(f"Training model")
            training_start_time = time.time()
            
            train_metrics = self.learner.learn(
                experiences=experiences,
                n_steps=self.n_steps,
                n_envs=self.n_envs,
                metrics_callback=callback_manager.train_callback if callback_manager else None,
            )
            
            # Log training metrics
            if logger is not None:
                logger.log_training_step(
                    iteration=iteration + 1,
                    global_step=self.global_step,
                    policy_loss=train_metrics["policy_loss"],
                    value_loss=train_metrics["value_loss"],
                    entropy=train_metrics["entropy"],
                    mean_reward=None,
                )
            
            # Prepare metrics for callback
            formatted_metrics = {
                "approx_kl": f"{train_metrics['approx_kl']:.4f}",
                "clip_fraction": f"{train_metrics['clip_fraction']:.3f}",
                "clip_range": f"{self.learner.clip_range:.1f}",
                "entropy_loss": f"{-train_metrics['entropy']:.5f}",
                "explained_variance": f"{train_metrics['explained_variance']:.2f}",
                "learning_rate": f"{self.optimizer.param_groups[0]['lr']:.4f}",
                "loss": f"{train_metrics['policy_loss'] + train_metrics['value_loss']:.3f}",
                "n_updates": str(train_metrics["n_updates"]),
                "policy_gradient_loss": f"{train_metrics['policy_loss']:.3f}",
                "value_loss": f"{train_metrics['value_loss']:.2f}",
            }
            training_time = time.time() - training_start_time
            print(f"Time to train {training_time:.2f}s\n")
            # ==================== Evaluation ====================
            eval_start_time = time.time()
            print('---------------evaluation started---------------')
            if eval_callback is not None and eval_callback.should_evaluate(iteration + 1):
                eval_callback.on_evaluation_start(iteration + 1, self.global_step)
                
                # Run evaluation to get metrics
                eval_metrics = self._run_evaluation(
                    actor=self.actor,
                    eval_env=self.eval_env,
                    sampler=self.sampler,
                    data_handler=self.data_handler,
                    callback=eval_callback,
                )
                
                # Notify callback of evaluation end with actual metrics
                is_new_best = eval_callback.on_evaluation_end(
                    iteration + 1, self.global_step, eval_metrics
                )
                
                if is_new_best and self.model_save_path is not None:
                    self.best_model_path = self._save_checkpoint(
                        epoch=iteration + 1,
                        timestep=self.global_step,
                        metrics=eval_metrics,
                        prefix="best_eval",
                    )
                    print(f"  ★ New best model saved!")
            print(f'---------------evaluation finished---------------  took {time.time() - eval_start_time:.2f} seconds')
            # ==================== End of Iteration ====================
            if callback_manager is not None:
                callback_manager.on_iteration_end(
                    iteration=iteration + 1,
                    global_step=self.global_step,
                    train_metrics=formatted_metrics,
                    n_envs=self.n_envs,
                )
            
            # ==================== Periodic Checkpoint ====================
            if (iteration + 1) % 10 == 0 and self.model_save_path is not None:
                self._save_checkpoint(
                    epoch=iteration + 1,
                    timestep=self.global_step,
                    metrics=train_metrics,
                    prefix="last_epoch",
                )
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"{'='*60}\n")
        
        return self.actor, self.critic
    
    def _run_evaluation(
        self,
        actor: nn.Module,
        eval_env: Any,
        sampler: Any,
        data_handler: Any,
        callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Run evaluation on the validation set.
        
        Args:
            actor: Actor network to evaluate
            eval_env: Evaluation environment
            sampler: Negative sampler
            data_handler: Data handler with validation queries
            callback: Optional callback for accumulating episode stats
            
        Returns:
            Dictionary of evaluation metrics
        """
        from model_eval import eval_corruptions_torchrl
        
        actor.eval()
        
        # Prepare evaluation data
        # Get number of eval queries from args if available
        n_eval_queries = len(data_handler.valid_queries)
        if self.args and hasattr(self.args, 'n_eval_queries') and self.args.n_eval_queries:
            n_eval_queries = min(self.args.n_eval_queries, n_eval_queries)
        
        eval_data = data_handler.valid_queries[:n_eval_queries]
        eval_depths = (
            data_handler.valid_queries_depths[:n_eval_queries]
            if hasattr(data_handler, 'valid_queries_depths') and data_handler.valid_queries_depths is not None
            else None
        )
        
        # Determine n_corruptions from args
        if self.args and hasattr(self.args, 'eval_neg_samples'):
            n_corruptions = self.args.eval_neg_samples
        else:
            raise ValueError("n_corruptions not specified in args")
                
        # Run evaluation to get metrics
        try:
            # Create info callback with verbose logging
            def info_callback_with_verbose(infos):
                if callback is not None:
                    if self.verbose_cb:
                        print(f"[PPOAgent._run_evaluation] Callback receiving {len(infos)} infos")
                    callback.accumulate_episode_stats(infos, mode="eval")
            
            metrics = eval_corruptions_torchrl(
                actor=actor,
                env=eval_env,
                data=eval_data,
                sampler=sampler,
                n_corruptions=n_corruptions,
                deterministic=True,
                verbose=0,
                plot=False,
                kge_inference_engine=None,
                evaluation_mode='rl_only',
                corruption_scheme=['head', 'tail'],
                info_callback=info_callback_with_verbose,
                data_depths=eval_depths,
            )
            
        except Exception as e:
            print(f"Warning: Evaluation failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise e
        
        actor.train()
        
        return metrics
    
    def _save_checkpoint(
        self,
        epoch: int,
        timestep: int,
        metrics: dict,
        prefix: str = "checkpoint",
    ) -> Path:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            timestep: Current timestep
            metrics: Training metrics
            prefix: Filename prefix
            
        Returns:
            Path to saved checkpoint
        """
        if self.model_save_path is None:
            raise ValueError("model_save_path not set")
        
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'timestep': timestep,
            'metrics': metrics,
        }
        
        filename = self.model_save_path / f"{prefix}_epoch_{epoch}_step_{timestep}.pt"
        torch.save(checkpoint, filename)
        print(f"Saved checkpoint to {filename}")
        
        return filename
    
    def load_checkpoint(
        self,
        checkpoint_path: Path,
    ) -> Tuple[int, int, dict]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Tuple of (epoch, timestep, metrics)
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        timestep = checkpoint.get('timestep', 0)
        metrics = checkpoint.get('metrics', {})
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {epoch}, Timestep: {timestep}")
        
        return epoch, timestep, metrics
