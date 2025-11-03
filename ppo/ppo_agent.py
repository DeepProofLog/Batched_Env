"""
PPO Agent

This module provides the main PPO agent class that coordinates training.
"""

import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, List
from tensordict import TensorDict

import torch
import torch.nn as nn

from .ppo_rollout import collect_rollouts


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

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Tracking
        self.best_eval_metric = float('-inf')
        self.best_model_path = None
        self.global_step = 0

    
    
    def compute_advantages(
        self,
        experiences: List[TensorDict],
        n_steps: int,
        n_envs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE(γ, λ) using **TorchRL's GAE** on a single stacked TensorDict.

        Returns
        -------
        advantages : Tensor shaped [n_steps, n_envs]
        returns    : Tensor shaped [n_steps, n_envs] (a.k.a. value targets)
        """
        # Stack to [T, B]
        batch_td_time = torch.stack(experiences, dim=0)

        # Ensure value predictions exist for both current and next observations
        with torch.no_grad():
            # value at current state
            if "state_value" not in batch_td_time.keys():
                self.critic(batch_td_time)
            # value at next state
            nxt = batch_td_time.get("next")
            if "state_value" not in nxt.keys():
                self.critic(nxt)

            # TorchRL's GAE writes 'advantage' and 'value_target' in-place
            try:
                from torchrl.objectives.value import GAE
            except Exception:
                # Fallback API path for older versions
                from torchrl.objectives.advantages import GAE  # type: ignore

            gae = GAE(
                gamma=self.gamma,
                lmbda=self.gae_lambda,
                value_key="state_value",
                reward_key=("next", "reward"),
                done_key=("next", "done"),
            )
            batch_td_time = gae(batch_td_time)

        advantages = batch_td_time.get("advantage")
        value_targets = batch_td_time.get("value_target")

        # Reshape to [T, B] if they came out flattened
        if advantages.dim() == 1:
            advantages = advantages.view(n_steps, n_envs)
        if value_targets.dim() == 1:
            value_targets = value_targets.view(n_steps, n_envs)

        return advantages, value_targets

    def learn(
        self,
        experiences: List[TensorDict],
        n_steps: int,
        n_envs: int,
        metrics_callback: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Optimize with **TorchRL's PPOLoss** on in-place TensorDict minibatches."""
        import math
        from copy import deepcopy

        # Stack to [T, B]
        batch_td_time = torch.stack(experiences, dim=0)

        # Compute advantages (and value targets) with TorchRL GAE
        advantages, value_targets = self.compute_advantages(experiences, n_steps, n_envs)

        # Attach to the stacked TD then flatten to [T*B]
        batch_td_time.set("advantage", advantages)
        batch_td_time.set("value_target", value_targets)
        flat_td = batch_td_time.reshape(n_steps * n_envs)

        # Advantage normalization (PPOLoss can also normalize internally; we do it here for stability)
        adv = flat_td.get("advantage")
        adv_norm = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        flat_td.set("advantage", adv_norm)

        # Build loss module
        try:
            from torchrl.objectives import ClipPPOLoss as PPOLossCls
        except Exception:
            try:
                from torchrl.objectives import PPOLoss as PPOLossCls  # older alias
            except Exception:
                from torchrl.objectives.ppo import ClipPPOLoss as PPOLossCls  # very old fallback

        loss_module = PPOLossCls(
            actor=self.actor,
            value_network=self.critic,
            clip_epsilon=self.clip_range,
            entropy_coef=self.ent_coef,
            normalize_advantage=False,  # we normalized above
            value_coef=self.value_coef,
            advantage_key="advantage",
            value_target_key="value_target",
            value_key="state_value",
        )
        # Key mapping for compatibility across TorchRL versions
        if hasattr(loss_module, "set_keys"):
            try:
                loss_module.set_keys(
                    action="action",
                    sample_log_prob="sample_log_prob",
                    value="state_value",
                    advantage="advantage",
                    value_target="value_target",
                )
            except TypeError:
                # Some versions only accept a subset (action, sample_log_prob)
                loss_module.set_keys(action="action", sample_log_prob="sample_log_prob")

        # Optimization
        flat_bs = n_steps * n_envs
        num_minibatches = max(1, flat_bs // int(self.batch_size))
        indices = torch.arange(flat_bs, device=self.device)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        num_updates = 0

        for epoch in range(self.n_epochs):
            perm = indices[torch.randperm(flat_bs, device=self.device)]
            for mb in range(num_minibatches):
                mb_idx = perm[mb * int(self.batch_size):(mb + 1) * int(self.batch_size)]
                mb_td = flat_td[mb_idx]

                # Compute PPO loss terms
                loss_out = loss_module(mb_td)

                # Robust extraction across TorchRL versions
                policy_loss = (
                    loss_out.get("loss_objective", None)
                    or loss_out.get("loss_policy", None)
                    or loss_out.get("loss_actor", None)
                    or loss_out.get("loss", None)
                )
                if policy_loss is None:
                    # fall back to 0 to avoid crashing; shouldn't happen on recent versions
                    policy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

                value_loss = loss_out.get("loss_critic", torch.tensor(0.0, device=self.device))
                entropy = loss_out.get("entropy", None)
                if entropy is None:
                    # if module returns loss_entropy instead
                    entropy = -loss_out.get("loss_entropy", torch.tensor(0.0, device=self.device))

                # Build total loss (value and entropy terms already scaled in some versions;
                # we scale explicitly here for consistency).
                loss = policy_loss + self.value_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    max_norm=self.max_grad_norm
                )
                self.optimizer.step()

                # ----- Metrics (approx_kl & clip_fraction) -----
                approx_kl = loss_out.get("approx_kl", None)
                clip_fraction = loss_out.get("clip_fraction", None)

                if (approx_kl is None) or (clip_fraction is None):
                    # recompute cheaply from logits to keep metrics informative
                    with torch.no_grad():
                        mb_logits_td = mb_td.clone(False)
                        # Run the actor's underlying net without resampling
                        try:
                            self.actor.module(mb_logits_td)  # writes 'logits'
                            from torchrl.modules.distributions import OneHotCategorical
                            dist = OneHotCategorical(logits=mb_logits_td.get("logits"))
                        except Exception:
                            # Fallback: compute logits/value via the shared model, if exposed
                            # This branch is rarely needed.
                            from torch.distributions import Categorical
                            # assume logits already in td (collector path)
                            dist = Categorical(logits=mb_logits_td.get("logits"))

                        new_log_prob = dist.log_prob(mb_td.get("action"))
                        old_log_prob = mb_td.get("sample_log_prob")
                        log_ratio = new_log_prob - old_log_prob
                        ratio = torch.exp(log_ratio)
                        approx_kl = ((ratio - 1) - log_ratio).mean()
                        clip_fraction = (ratio.sub(1).abs() > self.clip_range).float().mean()
                        if not torch.isfinite(approx_kl):
                            approx_kl = torch.tensor(0.0, device=self.device)
                        if not torch.isfinite(clip_fraction):
                            clip_fraction = torch.tensor(0.0, device=self.device)

                # Accumulate
                total_policy_loss += float(policy_loss.detach().cpu().item())
                total_value_loss += float(value_loss.detach().cpu().item())
                total_entropy += float(entropy.detach().cpu().item())
                total_approx_kl += float(approx_kl.detach().cpu().item())
                total_clip_fraction += float(clip_fraction.detach().cpu().item())
                num_updates += 1

            # Per-epoch callback
            if metrics_callback is not None and num_minibatches > 0:
                metrics_callback.on_training_epoch(
                    epoch=epoch + 1,
                    n_epochs=self.n_epochs,
                    policy_loss=total_policy_loss / max(1, num_updates),
                    value_loss=total_value_loss / max(1, num_updates),
                    entropy=total_entropy / max(1, num_updates),
                )

        # Aggregate metrics
        avg_policy_loss = total_policy_loss / max(1, num_updates)
        avg_value_loss = total_value_loss / max(1, num_updates)
        avg_entropy = total_entropy / max(1, num_updates)
        avg_approx_kl = total_approx_kl / max(1, num_updates)
        avg_clip_fraction = total_clip_fraction / max(1, num_updates)

        # Explained variance
        with torch.no_grad():
            vpred = flat_td.get("state_value").flatten()
            vtarget = flat_td.get("value_target").flatten()
            var_vtarget = torch.var(vtarget)
            var_residual = torch.var(vtarget - vpred)
            explained_var = (1 - (var_residual / (var_vtarget + 1e-8))).item()

        return {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
            "approx_kl": avg_approx_kl,
            "clip_fraction": avg_clip_fraction,
            "explained_variance": explained_var,
            "n_updates": int(num_updates),
        }
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
            print(f"Mini-batch size: {self.batch_size}")
            print(f"Optimization epochs per iteration: {self.n_epochs}")
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
                
                train_metrics = self.learn(
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
                    "clip_range": f"{self.clip_range:.1f}",
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
