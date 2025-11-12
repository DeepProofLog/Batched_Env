from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
from contextlib import nullcontext
import time
import torch.nn as nn
import warnings

import torch
import torch.nn.functional as F

try:  # PyTorch >= 2.1 preferred AMP API
    from torch.amp import GradScaler as _TorchGradScaler, autocast as _torch_autocast
    _AMP_DEVICE_TYPE = "cuda"
except (ImportError, AttributeError):  # Fallback for older versions
    from torch.cuda.amp import GradScaler as _TorchGradScaler, autocast as _torch_autocast  # type: ignore
    _AMP_DEVICE_TYPE = None
from tensordict import TensorDict

from ppo.ppo_rollout import RolloutCollector



class PPOAgent():
    """
    TorchRL-free PPO update that reuses the existing rollout/evaluation logic from ``PPOAgent``
    but replaces the optimisation step with a fully vectorised PyTorch implementation.

    The core idea is to reuse the underlying ``ActorCriticModel`` (shared by both the actor
    and the critic modules) so we can recompute logits and values on-device, apply action
    masking explicitly, and operate only on non-padded actions.
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
        index_manager: Any = None,  # Add index_manager parameter
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
        verbose_cb: bool = False,
        verbose: bool = False,
        debug_mode: bool = False,
        min_multiaction_ratio: float = 0.05,
        use_amp: bool = False,
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
            debug_mode: When True, emit verbose rollout/optimization diagnostics
            use_amp: Enable mixed-precision training on CUDA for faster compute
        """
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.train_env = train_env
        self.eval_env = eval_env
        self.sampler = sampler
        self.data_handler = data_handler
        self.index_manager = index_manager  # Store index_manager
        self.args = args
        self.verbose_cb = verbose_cb
        self.verbose = verbose
        self.debug_mode = debug_mode
        self.device = device
        self.use_amp = bool(use_amp and device.type == "cuda" and torch.cuda.is_available())
        self.grad_scaler = self._create_grad_scaler(self.use_amp)
        self.min_multiaction_ratio = float(min_multiaction_ratio)
        
        self.n_envs = n_envs
        self.n_steps = n_steps
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
        
        # Persistent rollout collector (created on first use)
        self._rollout_collector: Optional[Any] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _stack_experiences(experiences: List[TensorDict]) -> TensorDict:
        return torch.stack(experiences, dim=0)

    def _amp_context(self):
        if not self.use_amp:
            return nullcontext()
        if _AMP_DEVICE_TYPE is not None:
            return _torch_autocast(device_type=_AMP_DEVICE_TYPE, enabled=True)
        return _torch_autocast(enabled=True)

    @staticmethod
    def _create_grad_scaler(enabled: bool):
        try:
            if _AMP_DEVICE_TYPE is not None:
                return _TorchGradScaler(device=_AMP_DEVICE_TYPE, enabled=enabled)
        except TypeError:
            pass
        return _TorchGradScaler(enabled=enabled)

    @staticmethod
    def _masked_log_softmax(
        logits: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probabilities and entropy for masked categorical distribution - OPTIMIZED.
        
        Args:
            logits: Raw logits (batch, num_actions)
            mask: Boolean mask indicating valid actions (batch, num_actions)
            eps: Small epsilon for numerical stability
            
        Returns:
            Tuple of (log_probs, entropy)
        """
        # Ensure mask is boolean
        mask = mask.bool()
        
        # Mask invalid actions with very negative value
        masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        
        # Compute log softmax - this is numerically stable
        log_probs = torch.log_softmax(masked_logits, dim=-1)
        
        # Compute probabilities for entropy calculation
        probs = log_probs.exp()
        
        # Compute entropy: -sum(p * log(p)) = -sum(p * log_probs)
        # Only sum over valid actions (masked invalid ones will be ~0 after exp)
        entropy = -(probs * log_probs * mask.float()).sum(dim=-1)
        
        return log_probs, entropy

    def _actor_critic_forward(self, obs: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor and critic networks.
        
        Args:
            obs: Observation TensorDict
            
        Returns:
            Tuple of (logits, values)
        """
        # Get logits from actor
        actor_out = self.actor(obs)
        if isinstance(actor_out, TensorDict):
            logits = actor_out.get("logits")
        else:
            logits = actor_out
            
        # Get values from critic
        critic_out = self.critic(obs)
        if isinstance(critic_out, TensorDict):
            values = critic_out.get("state_value")
        else:
            values = critic_out
            
        # Ensure proper shapes
        if values.dim() > 1:
            values = values.squeeze(-1)
            
        return logits, values

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE) - VECTORIZED.
        
        Note: values and next_values should already be detached when passed in,
        but we ensure no gradients flow through this computation.
        
        Vectorized implementation using reverse cumulative product for massive speedup.
        """
        # Ensure inputs don't track gradients
        rewards = rewards.detach()
        dones = dones.detach()
        values = values.detach()
        next_values = next_values.detach()
        
        # Compute masks and deltas for all timesteps at once
        masks = 1.0 - dones.float()  # [n_steps, n_envs]
        deltas = rewards + self.gamma * next_values * masks - values  # [n_steps, n_envs]
        
        # Vectorized GAE computation using reverse cumsum with decay
        # The GAE at time t is: delta_t + (gamma * lambda * mask_t) * GAE_{t+1}
        # This can be computed by reversing, applying cumsum with decay, then reversing back
        
        # Compute discount factors: gamma * lambda * mask
        discount_factors = self.gamma * self.gae_lambda * masks  # [n_steps, n_envs]
        
        # Reverse tensors for backward computation
        deltas_rev = torch.flip(deltas, dims=[0])  # Reverse time dimension
        discount_rev = torch.flip(discount_factors, dims=[0])
        
        # Initialize advantages in reverse order
        advantages_rev = torch.zeros_like(deltas_rev)
        advantages_rev[0] = deltas_rev[0]  # Last timestep (first in reversed)
        
        # Vectorized accumulation - still need loop but much simpler
        for t in range(1, self.n_steps):
            advantages_rev[t] = deltas_rev[t] + discount_rev[t-1] * advantages_rev[t-1]
        
        # Reverse back to get advantages in correct order
        advantages = torch.flip(advantages_rev, dims=[0])
        returns = advantages + values
        
        return advantages, returns

    @staticmethod
    def _explained_variance(values: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return targets.new_zeros(())
        var_y = torch.var(targets)
        # OPTIMIZED: Replace expensive torch.allclose with simple threshold check
        if var_y < 1e-8:
            return var_y.new_zeros(())
        return 1 - torch.var(targets - values) / var_y

    def _select_obs(self, td: TensorDict) -> TensorDict:
        return td.select("sub_index", "derived_sub_indices", "action_mask")

    # ------------------------------------------------------------------
    # Overridden PPO optimise step
    # ------------------------------------------------------------------
    def learn(
        self,
        experiences: List[TensorDict],
        n_steps: int,
        n_envs: int,
        metrics_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        assert n_steps == self.n_steps and n_envs == self.n_envs

        batch_td = self._stack_experiences(experiences).to(self.device)
        flat_td = batch_td.reshape(self.n_steps * self.n_envs)

        next_td = batch_td.get("next")
        rewards = next_td["reward"].squeeze(-1).detach()  # Detach early - no gradients needed
        dones = next_td["done"].squeeze(-1).detach()  # Detach early - no gradients needed

        with torch.no_grad():
            obs = self._select_obs(batch_td)
            obs_flat = obs.reshape(-1)
            with self._amp_context():
                _, values = self._actor_critic_forward(obs_flat)
            values = values.view(self.n_steps, self.n_envs).to(torch.float32)

            next_obs = self._select_obs(next_td)
            next_obs_flat = next_obs.reshape(-1)
            with self._amp_context():
                _, next_values = self._actor_critic_forward(next_obs_flat)
            next_values = next_values.view(self.n_steps, self.n_envs).to(torch.float32)

        # GAE computation - values are already detached from no_grad context
        advantages, returns = self._compute_gae(rewards, dones, values, next_values)
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)
        
        # Normalizing advantages - optimized in-place
        with torch.no_grad():  # advantages don’t need grads in PPO
            adv_std, adv_mean = torch.std_mean(advantages, unbiased=False)  # both are GPU tensors
        # normalize on GPU (no sync)
        advantages.sub_(adv_mean).div_(adv_std.clamp_min(1e-8))

        # Extract old log probs and actions - detach since they're from the old policy
        old_log_probs = flat_td.get("sample_log_prob").reshape(-1).detach()
        actions = flat_td.get("action").reshape(-1).detach()

        flat_bs = self.n_steps * self.n_envs
        batch_size = min(self.batch_size, flat_bs)

        total_policy_loss = torch.zeros((), device=self.device)
        total_value_loss = torch.zeros((), device=self.device)
        total_entropy = torch.zeros((), device=self.device)
        total_kl = torch.zeros((), device=self.device)
        total_clip_fraction = torch.zeros((), device=self.device)
        num_updates = 0

        indices = torch.arange(flat_bs, device=self.device)
        
        # Create a generator for CPU randperm (much faster than GPU randperm)
        generator = torch.Generator()

        params_to_clip = [
            p for p in list(self.actor.parameters()) + list(self.critic.parameters()) if p.requires_grad
        ]
        has_params_to_clip = len(params_to_clip) > 0

        for epoch in range(self.n_epochs):
            # Print epoch progress (causes minor GPU-CPU sync overhead but useful for monitoring)
            print(f"Starting epoch {epoch+1}/{self.n_epochs}...")
            # Generate permutation on CPU (much faster) then move to device
            perm_cpu = torch.randperm(flat_bs, generator=generator)
            perm = perm_cpu.to(self.device, non_blocking=True)
            # Initialize separate tensors for each metric to avoid aliasing
            epoch_pl = torch.zeros((), device=self.device)
            epoch_vl = torch.zeros((), device=self.device)
            epoch_ent = torch.zeros((), device=self.device)
            epoch_kl = torch.zeros((), device=self.device)
            epoch_clip = torch.zeros((), device=self.device)
            epoch_updates = 0

            for mb_idx in torch.split(perm, batch_size):
                mb_td = flat_td[mb_idx]
                obs_mb = self._select_obs(mb_td)

                with self._amp_context():
                    logits, value_pred = self._actor_critic_forward(obs_mb)
                    mask_mb = obs_mb["action_mask"].bool()
                    log_probs, entropy = self._masked_log_softmax(logits, mask_mb)

                    mb_actions = actions[mb_idx].unsqueeze(-1)
                    new_log_prob = log_probs.gather(1, mb_actions).squeeze(-1).float()
                    old_log_prob_mb = old_log_probs[mb_idx].float()
                    ratio = (new_log_prob - old_log_prob_mb).exp()

                    adv_mb = advantages[mb_idx]
                    returns_mb = returns[mb_idx]
                    surr1 = ratio * adv_mb
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_mb
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(value_pred.float(), returns_mb)
                    entropy_loss = entropy.mean()
                    loss = policy_loss + self.value_coef * value_loss - self.ent_coef * entropy_loss

                # Compute metrics BEFORE backward pass to avoid stale values
                with torch.no_grad():
                    approx_kl = (old_log_prob_mb - new_log_prob.detach()).mean()
                    clip_fraction = (torch.abs(ratio.detach() - 1.0) > self.clip_range).float().mean()

                self.optimizer.zero_grad(set_to_none=True)
                if self.use_amp:
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.unscale_(self.optimizer)
                    if has_params_to_clip:
                        torch.nn.utils.clip_grad_norm_(params_to_clip, self.max_grad_norm)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    loss.backward()
                    if has_params_to_clip:
                        torch.nn.utils.clip_grad_norm_(params_to_clip, self.max_grad_norm)
                    self.optimizer.step()

                total_policy_loss += policy_loss.detach()
                total_value_loss += value_loss.detach()
                total_entropy += entropy_loss.detach()
                total_kl += approx_kl  # Already detached via no_grad context
                total_clip_fraction += clip_fraction  # Already detached via no_grad context
                num_updates += 1

                epoch_pl += policy_loss.detach()
                epoch_vl += value_loss.detach()
                epoch_ent += entropy_loss.detach()
                epoch_kl += approx_kl  # Already detached via no_grad context
                epoch_clip += clip_fraction  # Already detached via no_grad context
                epoch_updates += 1

            # Using division to ensure we get scalar tensors, then .item()
            epoch_pl_f = float((epoch_pl / epoch_updates).item())
            epoch_vl_f = float((epoch_vl / epoch_updates).item())
            epoch_ent_f = float((epoch_ent / epoch_updates).item())
            epoch_kl_f = float((epoch_kl / epoch_updates).item())
            epoch_clip_f = float((epoch_clip / epoch_updates).item())
            print(f"Epoch {epoch+1}. Losses: policy {epoch_pl_f:.4f}, value {epoch_vl_f:.4f}, entropy {epoch_ent_f:.4f}, KL {epoch_kl_f:.4f}, clip {epoch_clip_f:.4f}")

        total_metrics = torch.stack((
            total_policy_loss,
            total_value_loss,
            total_entropy,
            total_kl,
            total_clip_fraction,
        ))
        total_metrics = total_metrics / max(1, num_updates)
        policy_loss, value_loss, entropy, approx_kl, clip_fraction = total_metrics.detach().unbind()

        metrics = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
            "explained_variance": self._explained_variance(
                values.reshape(-1), returns
            ),
            "n_updates": int(num_updates),
        }
        return metrics

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
                raise ValueError("total_timesteps must be positive")

            def _to_float(value):
                if isinstance(value, torch.Tensor):
                    return float(value.detach().to('cpu'))  # one controlled sync at the call site
                return float(value)
           
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
                
                # Create persistent collector on first iteration
                print("  Creating persistent rollout collector (one-time setup)...")
                self._rollout_collector = RolloutCollector(
                        env=self.train_env,
                        actor=self.actor,
                        n_envs=self.n_envs,
                        n_steps=self.n_steps,
                        device=self.device,
                        debug=self.debug_mode,
                    )
                print("  Collector ready!\n")
                
                # Use the persistent collector
                experiences, stats = self._rollout_collector.collect(
                    critic=self.critic,
                    rollout_callback=(
                        rollout_callback.on_step if rollout_callback is not None else None
                    ),
                )
                if self.debug_mode:
                    self._debug_print_rollout_stats(experiences, stats)
                
                if rollout_callback is not None:
                    rollout_callback.on_rollout_end()
                
                # Update global step counter
                self.global_step += steps_per_iteration
                
                print(f"  Collected {steps_per_iteration} transitions")
                
                # Accumulate episode stats
                episode_info = stats.get("episode_info")
                has_episodes = False
                if isinstance(episode_info, dict):
                    rewards_tensor = episode_info.get("reward")
                    has_episodes = rewards_tensor is not None and rewards_tensor.numel() > 0
                elif episode_info:
                    has_episodes = True

                if callback_manager is not None and has_episodes:
                    if self.verbose_cb and isinstance(episode_info, dict):
                        reward_count = int(episode_info.get("reward").shape[0]) if episode_info.get("reward") is not None else 0
                        print(f"[PPOAgent] Accumulating {reward_count} episode stats for training")
                    callback_manager.accumulate_episode_stats(episode_info, mode="train")
                print(f"  Rollout time: {time.time() - iteration_start_time:.2f}s\n")
                # ==================== Policy Optimization ====================
                print(f"Training model")
                training_start_time = time.time()

                train_metrics = self.learn(
                    experiences=experiences,
                    n_steps=self.n_steps,
                    n_envs=self.n_envs,
                    metrics_callback=None,
                )
                if self.debug_mode:
                    self._debug_print_optimizer_stats(train_metrics)
                
                # # Log training metrics
                # if logger is not None:
                #     logger.log_training_step(
                #         global_step=self.global_step,
                #         policy_loss=_to_float(train_metrics["policy_loss"]),
                #         value_loss=_to_float(train_metrics["value_loss"]),
                #         entropy=_to_float(train_metrics["entropy"]),
                #     )
                
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
        from model_eval import evaluate_ranking_metrics
        
        actor.eval()
        
        # Prepare evaluation data
        # Get the materialized valid split
        valid_split = data_handler.get_materialized_split('valid')
        eval_queries_tensor = valid_split.queries
        
        # Get number of eval queries from args if available
        n_eval_queries = eval_queries_tensor.shape[0]
        if self.args and hasattr(self.args, 'n_eval_queries') and self.args.n_eval_queries:
            n_eval_queries = min(self.args.n_eval_queries, n_eval_queries)
        
        eval_data = eval_queries_tensor[:n_eval_queries]
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
            # Prepare queries for evaluation
            queries = eval_data
            
            # queries is already a tensor from the materialized split
            # It should be [N, A, D] where A is atoms per query, D is dimensions per atom
            if queries.dim() == 2:  # [N, 3] -> [N, 1, 3]
                queries = queries.unsqueeze(1)
            
            # Get corruption scheme from args
            corruption_scheme = ['head', 'tail']
            if self.args and hasattr(self.args, 'corruption_scheme'):
                corruption_scheme = self.args.corruption_scheme
            
            print(f"  Evaluating on {queries.shape[0]} queries with {n_corruptions} corruptions per query...")
            
            # Call evaluate_ranking_metrics with the correct signature
            metrics = evaluate_ranking_metrics(
                actor=actor,
                env=eval_env,
                queries=queries,
                sampler=sampler,
                n_corruptions=n_corruptions,
                corruption_modes=corruption_scheme,
                deterministic=True,
                verbose=False,
            )
            
            print(f"  Evaluation complete: MRR={metrics.get('MRR', 0.0):.4f}, Hits@1={metrics.get('Hits@1', 0.0):.4f}, Hits@10={metrics.get('Hits@10', 0.0):.4f}")
            
            # Transform keys to match expected format (lowercase with _mean suffix)
            # evaluate_ranking_metrics returns: MRR, Hits@1, Hits@3, Hits@10, per_mode
            transformed_metrics = {
                'mrr_mean': metrics.get('MRR', 0.0),
                'hits1_mean': metrics.get('Hits@1', 0.0),
                'hits3_mean': metrics.get('Hits@3', 0.0),
                'hits10_mean': metrics.get('Hits@10', 0.0),
            }
            
            # Add per-mode metrics if available
            if 'per_mode' in metrics:
                per_mode = metrics['per_mode']
                for mode, mode_metrics in per_mode.items():
                    transformed_metrics[f'{mode}_mrr_mean'] = mode_metrics.get('MRR', 0.0)
                    transformed_metrics[f'{mode}_hits1_mean'] = mode_metrics.get('Hits@1', 0.0)
                    transformed_metrics[f'{mode}_hits3_mean'] = mode_metrics.get('Hits@3', 0.0)
                    transformed_metrics[f'{mode}_hits10_mean'] = mode_metrics.get('Hits@10', 0.0)
            
            metrics = transformed_metrics
            
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
        
        epoch = checkpoint['epoch']
        timestep = checkpoint['timestep']
        metrics = checkpoint.get('metrics', {})
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {epoch}, Timestep: {timestep}")
        
        return epoch, timestep, metrics
    
    def cleanup(self):
        """Clean up resources, especially the persistent collector."""
        if self._rollout_collector is not None:
            print("Shutting down persistent rollout collector...")
            self._rollout_collector.shutdown()
            self._rollout_collector = None
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        self.cleanup()
