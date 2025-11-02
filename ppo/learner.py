"""
PPO Learner Module

This module handles the policy and value function optimization using PPO.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict


class PPOLearner:
    """
    PPO learning algorithm implementation.
    
    This class handles the computation of advantages using GAE and
    the optimization of policy and value networks using the PPO objective.
    """
    
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Initialize the PPO learner.
        
        Args:
            actor: Actor network
            critic: Critic network
            optimizer: Optimizer for both networks
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clipping range
            ent_coef: Entropy coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            n_epochs: Number of optimization epochs per update
            batch_size: Mini-batch size for optimization
            device: Device to use
        """
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
    
    def compute_advantages(
        self,
        experiences: List[TensorDict],
        n_steps: int,
        n_envs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            experiences: List of experience TensorDicts
            n_steps: Number of steps
            n_envs: Number of parallel environments
            
        Returns:
            Tuple of (advantages, returns)
        """
        with torch.no_grad():
            # Stack batch
            batch = torch.stack(experiences, dim=0)
            
            # Extract values, rewards, dones
            rewards = torch.stack([batch[i]["next"]["reward"] for i in range(n_steps)]).to(self.device)
            values = torch.stack([batch[i]["state_value"] for i in range(n_steps)]).squeeze(-1).to(self.device)
            dones = torch.stack([batch[i]["next"]["done"] for i in range(n_steps)]).float().to(self.device)
            
            # Squeeze extra dimensions if present
            if rewards.dim() == 3 and rewards.shape[-1] == 1:
                rewards = rewards.squeeze(-1)
            if values.dim() == 3 and values.shape[-1] == 1:
                values = values.squeeze(-1)
            if dones.dim() == 3 and dones.shape[-1] == 1:
                dones = dones.squeeze(-1)
            
            # Compute GAE
            advantages = torch.zeros_like(rewards, device=self.device)
            returns = torch.zeros_like(rewards, device=self.device)
            gae = torch.zeros(n_envs, device=self.device)
            next_value = torch.zeros(n_envs, device=self.device)
            
            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    next_non_terminal = 1.0 - dones[t]
                    next_value_t = next_value
                else:
                    next_non_terminal = 1.0 - dones[t]
                    next_value_t = values[t + 1]
                
                delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - values[t]
                gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
                advantages[t] = gae
                returns[t] = gae + values[t]
        
        return advantages, returns
    
    def learn(
        self,
        experiences: List[TensorDict],
        n_steps: int,
        n_envs: int,
        metrics_callback: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Perform PPO optimization on collected experiences.
        
        Args:
            experiences: List of experience TensorDicts
            n_steps: Number of steps
            n_envs: Number of parallel environments
            metrics_callback: Optional callback for logging metrics
            
        Returns:
            Dict containing training metrics
        """
        # Stack batch
        batch = torch.stack(experiences, dim=0)
        flat_batch_size = n_steps * n_envs
        
        # Compute advantages
        advantages, returns = self.compute_advantages(experiences, n_steps, n_envs)
        
        # Normalize advantages
        advantages_flat = advantages.reshape(-1)
        advantages_normalized = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
        
        # Flatten batch for training
        obs_flat = torch.cat([batch[i]["sub_index"] for i in range(n_steps)], dim=0).to(self.device)
        actions_flat = torch.cat([batch[i]["derived_sub_indices"] for i in range(n_steps)], dim=0).to(self.device)
        masks_flat = torch.cat([batch[i]["action_mask"] for i in range(n_steps)], dim=0).to(self.device)
        old_actions_flat = torch.cat([batch[i]["action"] for i in range(n_steps)], dim=0).to(self.device)
        old_log_probs_flat = torch.cat([batch[i]["sample_log_prob"] for i in range(n_steps)], dim=0).to(self.device)
        returns_flat = returns.reshape(-1)
        
        # Get the underlying actor-critic model
        critic_inner = self.critic._module if hasattr(self.critic, '_module') else self.critic.module
        actor_critic_model = critic_inner.actor_critic_model
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_approx_kl = 0
        total_clip_fraction = 0
        
        # Optimization epochs
        for epoch in range(self.n_epochs):
            # Create mini-batches
            indices = torch.randperm(flat_batch_size, device=self.device)
            num_batches = flat_batch_size // self.batch_size
            
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_entropy = 0
            epoch_approx_kl = 0
            epoch_clip_fraction = 0
            
            for i in range(num_batches):
                batch_indices = indices[i * self.batch_size:(i + 1) * self.batch_size]
                
                # Get mini-batch
                mb_obs = obs_flat[batch_indices].to(self.device)
                mb_actions = actions_flat[batch_indices].to(self.device)
                mb_masks = masks_flat[batch_indices].to(self.device)
                mb_old_actions = old_actions_flat[batch_indices].to(self.device)
                mb_old_log_probs = old_log_probs_flat[batch_indices].to(self.device)
                mb_advantages = advantages_normalized[batch_indices].to(self.device)
                mb_returns = returns_flat[batch_indices].to(self.device)
                
                # Forward pass
                mb_td = TensorDict({
                    "sub_index": mb_obs,
                    "derived_sub_indices": mb_actions,
                    "action_mask": mb_masks,
                }, batch_size=torch.Size([self.batch_size]))
                
                # Get logits and values
                logits = actor_critic_model.forward_actor(mb_td)
                new_values = actor_critic_model.forward_critic(mb_td)
                
                # Apply action mask
                mb_action_mask = mb_td["action_mask"].to(logits.device)
                masked_logits = logits.masked_fill(~mb_action_mask.bool(), float("-inf"))
                masked_logits = torch.nan_to_num(masked_logits, nan=float("-inf"))

                # Create distribution
                dist = torch.distributions.Categorical(logits=masked_logits)
                
                # Get action indices from one-hot encoded old actions
                old_action_indices = torch.argmax(mb_old_actions, dim=-1)
                
                # Evaluate log probability of old actions under current policy
                new_log_probs = dist.log_prob(old_action_indices)
                
                # Compute entropy
                entropy = dist.entropy().mean()
                
                # Policy loss (PPO clip)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = 0.5 * ((new_values.squeeze() - mb_returns) ** 2).mean()
                
                # PPO metrics
                with torch.no_grad():
                    # Approximate KL divergence
                    log_ratio = new_log_probs - mb_old_log_probs
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    
                    # Clip fraction
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.ent_coef * entropy
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    max_norm=self.max_grad_norm
                )
                self.optimizer.step()
                
                # Accumulate epoch metrics
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_entropy += entropy.item()
                epoch_approx_kl += approx_kl.item()
                epoch_clip_fraction += clip_fraction.item()
            
            # Average epoch metrics
            avg_epoch_policy_loss = epoch_policy_loss / num_batches
            avg_epoch_value_loss = epoch_value_loss / num_batches
            avg_epoch_entropy = epoch_entropy / num_batches
            
            # Accumulate for overall average
            total_policy_loss += epoch_policy_loss
            total_value_loss += epoch_value_loss
            total_entropy += epoch_entropy
            total_approx_kl += epoch_approx_kl
            total_clip_fraction += epoch_clip_fraction
            
            # Report epoch progress
            if metrics_callback is not None:
                metrics_callback.on_training_epoch(
                    epoch=epoch + 1,
                    n_epochs=self.n_epochs,
                    policy_loss=avg_epoch_policy_loss,
                    value_loss=avg_epoch_value_loss,
                    entropy=avg_epoch_entropy,
                )
        
        # Compute average metrics across all epochs
        num_total_updates = self.n_epochs * num_batches
        avg_policy_loss = total_policy_loss / num_total_updates
        avg_value_loss = total_value_loss / num_total_updates
        avg_entropy = total_entropy / num_total_updates
        avg_approx_kl = total_approx_kl / num_total_updates
        avg_clip_fraction = total_clip_fraction / num_total_updates
        
        # Compute explained variance
        with torch.no_grad():
            values_all = torch.stack([batch[i]["state_value"] for i in range(n_steps)]).squeeze(-1).to(self.device)
            if values_all.dim() == 3 and values_all.shape[-1] == 1:
                values_all = values_all.squeeze(-1)
            values_flat = values_all.reshape(-1)
            
            var_returns = torch.var(returns_flat)
            var_residual = torch.var(returns_flat - values_flat)
            explained_var = 1 - (var_residual / (var_returns + 1e-8))
            explained_var = explained_var.item()
        
        # Return metrics
        metrics = {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
            "approx_kl": avg_approx_kl,
            "clip_fraction": avg_clip_fraction,
            "explained_variance": explained_var,
            "n_updates": num_total_updates,
        }
        
        return metrics
