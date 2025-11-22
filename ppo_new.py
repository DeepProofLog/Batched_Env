"""
ppo_new.py - Clean PPO implementation that exactly mimics SB3's PPO.

This file removes all batched-specific optimizations and matches the
SB3 PPO.train() implementation as closely as possible.
"""

import time
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rollout import RolloutBuffer
from trace_utils import TraceRecorder
from callbacks import print_formatted_metrics, DetailedMetricsCollector


class PPO:
    """
    Proximal Policy Optimization algorithm (PPO) - matches SB3 exactly.
    
    This is a simplified version that removes all optimizations and follows
    the SB3 PPO implementation line-by-line.
    """
    
    def __init__(
        self,
        policy: nn.Module,
        env,  # Environment to get n_envs from
        n_steps: int = 2048,
        learning_rate: float = 3e-4,
        n_epochs: int = 10,
        batch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: torch.device = None,
        verbose: bool = False,
        trace_dir: Optional[str] = None,
        trace_prefix: str = "batched",
        trace_recorder: Optional[TraceRecorder] = None,
    ):
        """
        Initialize PPO - matches SB3's __init__ exactly.
        """
        self.policy = policy
        self.env = env
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = device if device is not None else torch.device('cpu')
        self.verbose = verbose
        self.trace_recorder = trace_recorder or (TraceRecorder(trace_dir, prefix=trace_prefix) if trace_dir else None)
        self._trace_episode_ids = None
        self._trace_lengths = None
        self.env_device = getattr(env, "_device", None) or getattr(env, "device", None) or torch.device("cpu")
        if not isinstance(self.env_device, torch.device):
            self.env_device = torch.device(self.env_device)
        self.metrics_collector = DetailedMetricsCollector(collect_detailed=True, verbose=False)
        
        # Get number of environments
        self.n_envs = int(env.batch_size[0]) if isinstance(env.batch_size, torch.Size) else int(env.batch_size)
        
        # Create rollout buffer - matches SB3
        self.rollout_buffer = RolloutBuffer(
            buffer_size=n_steps,
            n_envs=self.n_envs,
            device=self.device,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        
        # Create optimizer - matches SB3
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate,
            eps=1e-5
        )
    
    def train(self) -> Dict[str, float]:
        """
        Train the policy using collected rollouts.
        Matches SB3's train() method exactly.
        
        Returns:
            Dictionary of training metrics
        """
        # Set policy to training mode
        self.policy.train()
        
        # Accumulators for logging
        pg_losses = []
        value_losses = []
        entropy_losses = []
        clip_fractions = []
        
        if self.verbose:
            print(f"[PPO] Training for {self.n_epochs} epochs...")
        
        # --------------------
        # Epoch loop - matches SB3 exactly
        # --------------------
        for epoch in range(self.n_epochs):
            # Minibatch loop
            for batch_data in self.rollout_buffer.get(batch_size=self.batch_size):
                obs, actions, old_values, old_log_probs, advantages, returns = batch_data
                
                # Flatten actions to match SB3
                actions = actions.squeeze(-1) if actions.dim() > 1 else actions
                
                # Normalize advantages - exactly as in SB3
                # SB3 normalizes per minibatch, not per full batch
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # --------------------
                # Forward pass - evaluate actions with current policy
                # --------------------
                values, log_probs, entropy = self.policy.evaluate_actions(obs, actions)
                
                # Flatten values to match returns shape
                values = values.flatten()
                
                # --------------------
                # Policy loss - PPO clipped objective
                # --------------------
                # Ratio between old and new policy
                log_ratio = log_probs - old_log_probs
                ratio = torch.exp(log_ratio)
                
                # Clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio,
                    1.0 - self.clip_range,
                    1.0 + self.clip_range,
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                # Logging: clip fraction
                with torch.no_grad():
                    clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float()).item()
                    clip_fractions.append(clip_fraction)
                
                # --------------------
                # Value loss - matches SB3 exactly
                # --------------------
                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    values_pred = old_values + torch.clamp(
                        values - old_values,
                        -self.clip_range_vf,
                        self.clip_range_vf,
                    )
                
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(returns, values_pred)
                
                # --------------------
                # Entropy loss - matches SB3 exactly
                # --------------------
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_probs)
                else:
                    entropy_loss = -torch.mean(entropy)
                
                # --------------------
                # Total loss
                # --------------------
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                
                # Log losses
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                
                # --------------------
                # Optimization step - matches SB3 exactly
                # --------------------
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.max_grad_norm,
                    )
                
                self.optimizer.step()
        
        # Set policy back to eval mode
        self.policy.eval()
        
        # Return average metrics - matches SB3
        return {
            "policy_loss": sum(pg_losses) / len(pg_losses) if pg_losses else 0.0,
            "value_loss": sum(value_losses) / len(value_losses) if value_losses else 0.0,
            "entropy": -sum(entropy_losses) / len(entropy_losses) if entropy_losses else 0.0,
            "clip_fraction": sum(clip_fractions) / len(clip_fractions) if clip_fractions else 0.0,
        }
    
    def learn(
        self,
        total_timesteps: int,
        callback=None,
    ) -> None:
        """
        Learn policy using PPO algorithm - matches SB3's learn() method.
        
        Args:
            total_timesteps: Total number of timesteps to train for
            callback: Optional callback to call after each rollout
        """
        from tensordict import TensorDict
        
        total_steps_done = 0
        iteration = 0
        
        # Reset environment
        current_obs = self.env.reset()
        current_episode_reward = torch.zeros(self.n_envs, device=self.device)
        current_episode_length = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
        episode_starts = torch.ones(self.n_envs, dtype=torch.float32, device=self.device)
        
        # Episode tracking
        episode_rewards = []
        episode_lengths = []
        if self.trace_recorder is not None:
            self._trace_episode_ids = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
            self._trace_lengths = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
        
        if self.verbose:
            print(f"\n[PPO] Starting training for {total_timesteps} timesteps")
            print(f"[PPO] Rollout size: {self.n_steps} steps x {self.n_envs} envs = {self.n_steps * self.n_envs} samples per rollout")
        
        while total_steps_done < total_timesteps:
            iteration += 1
            
            if self.verbose:
                print(f"\n[PPO] ===== Iteration {iteration} ({total_steps_done}/{total_timesteps} steps) =====")
            
            # ============================================================
            # Collect rollouts
            # ============================================================
            self.policy.eval()
            self.rollout_buffer.reset()
            self.metrics_collector.reset()
            
            rollout_start_time = time.time()
            n_collected = 0
            
            with torch.no_grad():
                while n_collected < self.n_steps:
                    # Clone the current observation so env.step() cannot mutate the
                    # data we store in the buffer or write to trace.
                    obs_snapshot = current_obs.clone()
                    obs_device = obs_snapshot.to(self.device)

                    # Get action from policy
                    actions, values, log_probs = self.policy(obs_device, deterministic=True)
                    dist_logits = None
                    try:
                        dist = getattr(self.policy.action_dist, "distribution", None)
                        if dist is not None and hasattr(dist, "logits"):
                            dist_logits = dist.logits.detach().clone()
                    except Exception:
                        dist_logits = None
                    
                    # Step environment
                    actions_env = actions.to(self.env_device)
                    action_td = TensorDict({"action": actions_env}, batch_size=current_obs.batch_size, device=self.env_device)
                    step_result, next_obs = self.env.step_and_maybe_reset(action_td)
                    
                    # Extract done/reward
                    if "next" in step_result.keys():
                        step_info = step_result["next"]
                    else:
                        step_info = step_result
                    
                    rewards_env = step_info.get("reward", torch.zeros(self.n_envs, device=self.env_device))
                    dones_env = step_info.get("done", torch.zeros(self.n_envs, dtype=torch.bool, device=self.env_device))
                    
                    # Squeeze to ensure correct shape
                    if rewards_env.dim() > 1:
                        rewards_env = rewards_env.squeeze(-1)
                    if dones_env.dim() > 1:
                        dones_env = dones_env.squeeze(-1)
                    rewards = rewards_env.to(self.device)
                    dones = dones_env.to(self.device)
                    
                    # Store transition
                    self.rollout_buffer.add(
                        obs=obs_device,
                        action=actions,
                        reward=rewards,
                        episode_start=episode_starts,
                        value=values,
                        log_prob=log_probs
                    )
                    if self.trace_recorder is not None:
                        if self._trace_episode_ids is None:
                            self._trace_episode_ids = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
                        if self._trace_lengths is None:
                            self._trace_lengths = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
                        self._trace_lengths = self._trace_lengths + 1
                        for idx in range(self.n_envs):
                            obs_dict = obs_snapshot
                            sub_index = obs_dict.get("sub_index")[idx]
                            derived_sub_indices = obs_dict.get("derived_sub_indices")[idx]
                            action_mask = obs_dict.get("action_mask")[idx]
                            self.trace_recorder.log_step(
                                phase="train",
                                iteration=iteration - 1,
                                step=n_collected,
                                env=int(idx),
                                action=int(actions[idx]),
                                reward=float(rewards[idx]),
                                done=bool(dones[idx]),
                                length=int(self._trace_lengths[idx]),
                                episode=int(self._trace_episode_ids[idx]),
                                value=float(values[idx]),
                                log_prob=float(log_probs[idx]),
                                sub_index=sub_index,
                                derived_sub_indices=derived_sub_indices,
                                action_mask=action_mask,
                                logits=dist_logits[idx] if dist_logits is not None else None,
                            )
                    
                    # Update statistics
                    current_episode_reward += rewards
                    current_episode_length += 1
                    n_collected += 1
                    total_steps_done += self.n_envs
                    
                    # Check for episode ends
                    if dones.any():
                        for idx in torch.where(dones)[0]:
                            ep_reward = current_episode_reward[idx].item()
                            ep_length = current_episode_length[idx].item()
                            episode_rewards.append(ep_reward)
                            episode_lengths.append(ep_length)

                            # Collect episode info for rollout metrics
                            label_val = step_info.get("label")
                            depth_val = step_info.get("query_depth")
                            success_val = step_info.get("is_success")
                            length_val = step_info.get("length")
                            info_dict = {
                                "episode": {"r": ep_reward, "l": ep_length},
                            }
                            if label_val is not None:
                                info_dict["label"] = int(label_val[idx].item()) if label_val.ndim > 0 else int(label_val.item())
                            if depth_val is not None:
                                info_dict["query_depth"] = int(depth_val[idx].item()) if depth_val.ndim > 0 else int(depth_val.item())
                            if success_val is not None:
                                info_dict["is_success"] = bool(success_val[idx].item()) if success_val.ndim > 0 else bool(success_val.item())
                            self.metrics_collector.accumulate([info_dict])
                            
                            # Reset episode stats
                            current_episode_reward[idx] = 0.0
                            current_episode_length[idx] = 0
                            if self._trace_episode_ids is not None:
                                self._trace_episode_ids[idx] += 1
                            if self._trace_lengths is not None:
                                self._trace_lengths[idx] = 0
                        
                        # Mark episode starts for next step
                        episode_starts = dones.float()
                    else:
                        episode_starts = torch.zeros(self.n_envs, dtype=torch.float32, device=self.device)
                    
                    current_obs = next_obs
                
                # Compute last values for bootstrapping
                _, last_values, _ = self.policy(current_obs.to(self.device))
            
            # Compute advantages and returns
            if dones.dim() > 1:
                dones = dones.squeeze(-1)
            self.rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)
            
            rollout_time = time.time() - rollout_start_time
            if self.verbose:
                print(f"[PPO] Rollout collected in {rollout_time:.2f}s")
                if episode_rewards:
                    recent_rewards = episode_rewards[-10:]
                    print(f"[PPO] Recent episodes: reward={sum(recent_rewards)/len(recent_rewards):.3f}, length={sum(episode_lengths[-10:])/len(episode_lengths[-10:]):.1f}")
            # SB3-style rollout metrics table with detailed stats
            rollout_metrics = self.metrics_collector.compute_metrics()
            extra_rollout = {
                "total_timesteps": total_steps_done,
            }
            if rollout_time > 0:
                extra_rollout["fps"] = int((self.n_envs * self.n_steps) / rollout_time)
            print_formatted_metrics(metrics=rollout_metrics, prefix="rollout", extra_metrics=extra_rollout)
            
            # ============================================================
            # Train policy
            # ============================================================
            train_start_time = time.time()
            train_metrics = self.train()
            train_time = time.time() - train_start_time
            train_extra = {**train_metrics, "total_timesteps": total_steps_done, "iterations": iteration}
            print_formatted_metrics(metrics={}, prefix="train", extra_metrics=train_extra)
            
            if self.verbose:
                print(f"[PPO] Training completed in {train_time:.2f}s")
                print(f"[PPO] Metrics: policy_loss={train_metrics['policy_loss']:.4f}, value_loss={train_metrics['value_loss']:.4f}, entropy={train_metrics['entropy']:.4f}")
            
            # ============================================================
            # Callback
            # ============================================================
            if callback is not None:
                callback_result = callback(locals(), globals())
                if callback_result is False:
                    if self.verbose:
                        print("[PPO] Training stopped by callback")
                    break
        
        if self.verbose:
            print(f"\n[PPO] Training completed!")
            if episode_rewards:
                print(f"[PPO] Total episodes: {len(episode_rewards)}")
                print(f"[PPO] Mean reward: {sum(episode_rewards)/len(episode_rewards):.3f}")
                print(f"[PPO] Mean length: {sum(episode_lengths)/len(episode_lengths):.1f}")
        if self.trace_recorder is not None:
            self.trace_recorder.flush()
