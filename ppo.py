"""
GPU-based PPO Algorithm using TensorDict.

This module provides a PPO implementation that mimics the SB3 PPO
but works entirely on GPU with torch tensors and tensordicts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Union
from tensordict import TensorDict
import time
import contextlib

from rollout import RolloutBuffer
from model import ActorCriticPolicy
from utils.debug_config import DebugConfig


class PPO:
    """
    Proximal Policy Optimization (PPO) algorithm.
    
    This implementation mimics SB3's PPO but works entirely on GPU with
    tensordicts for batched environments.
    
    Args:
        policy: Actor-critic policy
        env: Batched environment
        n_steps: Number of steps per rollout
        n_epochs: Number of epochs for policy update
        batch_size: Minibatch size for training
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clipping range
        clip_range_vf: Value function clipping range (None = no clipping)
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Maximum gradient norm for clipping
        learning_rate: Learning rate
        device: PyTorch device for training
        rollout_device: PyTorch device for rollout collection (defaults to device)
        verbose: Verbosity level
    """
    
    def __init__(
        self,
        policy: ActorCriticPolicy,
        env,
        n_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        learning_rate: float = 3e-4,
        device: torch.device = None,
        rollout_device: torch.device = None,
        verbose: int = 1,
        debug_config: Optional[DebugConfig] = None,
    ):
        self.policy = policy
        self.env = env
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.device = device if device is not None else torch.device('cpu')
        self.rollout_device = rollout_device if rollout_device is not None else self.device
        self.verbose = verbose
        self.debug_config = debug_config or DebugConfig()
        
        # Initialize optimizer with fused option for better performance
        try:
            # Try to use fused Adam for significant speedup
            self.optimizer = torch.optim.Adam(
                self.policy.parameters(),
                lr=self.learning_rate,
                fused=True if self.device.type == 'cuda' else False
            )
            if self.device.type == 'cuda':
                print("[OPTIMIZATION] Using fused Adam optimizer")
        except Exception as e:
            # Fallback to standard Adam if fused is not available
            print(f"[WARNING] Fused Adam not available: {e}. Using standard Adam.")
            self.optimizer = torch.optim.Adam(
                self.policy.parameters(),
                lr=self.learning_rate
            )
        
        # Get number of environments
        self.n_envs = int(env.batch_size[0]) if isinstance(env.batch_size, torch.Size) else int(env.batch_size)
        
        # Initialize rollout buffer
        self.rollout_buffer = RolloutBuffer(
            buffer_size=n_steps,
            n_envs=self.n_envs,
            device=self.device,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        
        # Training statistics
        self.num_timesteps = 0
        self._last_obs = None
        self._last_episode_starts = None
        self.ep_info_buffer = TensorDict({}, batch_size=torch.Size([0]), device=self.device)
        self.episode_lengths = torch.zeros(self.n_envs, dtype=torch.int32, device=self.device)
        
        # Logging
        self.logger_dict = {}
    
    def _setup_model(self) -> None:
        """Initialize the model and reset the environment."""
        # Reset environment and get initial observation
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        self._last_obs = obs
        self._last_episode_starts = torch.ones(self.n_envs, dtype=torch.float32, device=self.device)
        self.episode_lengths = torch.zeros(self.n_envs, dtype=torch.int32, device=self.device)
    
    def collect_rollouts(self, collect_action_stats: bool = False) -> bool:
        """
        Collect rollouts using the current policy.
        
        Args:
            collect_action_stats: If True, collect action statistics during rollout
            
        Returns:
            True if rollout collection was successful
        """
        assert self._last_obs is not None, "No previous observation"
        
        self.policy.eval()
        self.rollout_buffer.reset()
        
        # Store action statistics if requested
        action_stats = [] if collect_action_stats else None
        
        n_steps = 0
        rollout_start_time = time.time()
        
        # Reset episode lengths
        self.episode_lengths.fill_(0)
        
        print(f"Collecting rollouts for {self.n_steps} steps...")
        
        with torch.no_grad():
            while n_steps < self.n_steps:
                # Get actions, values, and log probs from policy
                # Move observation to training device if different from rollout device
                obs_for_policy = self._last_obs
                if self.rollout_device != self.device:
                    obs_for_policy = obs_for_policy.to(self.device)
                
                actions, values, log_probs = self.policy(obs_for_policy)
                
                # Collect action statistics if requested
                if collect_action_stats:
                    action_mask = self._last_obs.get('action_mask')
                    if action_mask is not None:
                        # Count valid actions for each environment
                        n_actions = action_mask.sum(dim=-1).cpu().numpy()
                        action_stats.append(n_actions)
                
                # Wrap actions in TensorDict for environment
                action_td = TensorDict(
                    {"action": actions.to(self.rollout_device)},
                    batch_size=self.n_envs,
                    device=self.rollout_device
                )
                
                # Take step in environment using step_and_maybe_reset
                # This automatically resets done environments
                step_result, obs = self.env.step_and_maybe_reset(action_td)
                
                # DEBUG: Check keys
                # print(f"step_result keys: {step_result.keys()}")

                # Extract next state info from step result
                if "next" in step_result.keys():
                    next_info = step_result['next']
                else:
                    next_info = step_result
                
                # Get reward, done, and success (all have shape [n_envs, 1])
                rewards = next_info['reward'].squeeze(-1).to(self.device)
                dones = next_info['done'].squeeze(-1).float().to(self.device)
                is_success = next_info.get('is_success')
                if is_success is not None:
                    is_success = is_success.squeeze(-1).to(self.device)
                
                # Update timesteps
                self.num_timesteps += self.n_envs
                
                # Add to buffer
                self.rollout_buffer.add(
                    obs=self._last_obs,
                    action=actions,
                    reward=rewards,
                    episode_start=self._last_episode_starts,
                    value=values,
                    log_prob=log_probs,
                )
                
                # Update last obs and episode starts
                self._last_obs = obs
                self._last_episode_starts = dones
                
                n_steps += 1
                
                # Progress logging every 10%
                progress = (n_steps / self.n_steps) * 100
                if int(progress) % 10 == 0 and int(progress) > 0:
                    elapsed_time = time.time() - rollout_start_time
                    print(f"  Rollout progress: {int(progress)}% complete ({n_steps}/{self.n_steps} steps) - Time: {elapsed_time:.2f}s")
                
                # Collect episode info with label and depth
                if is_success is not None:
                    done_mask = dones.bool()
                    if done_mask.any():
                        done_rewards = rewards[done_mask]
                        done_lengths = self.episode_lengths[done_mask]
                        done_successes = is_success[done_mask]
                        
                        # Extract label and depth from observation
                        done_labels = next_info.get('label')
                        done_depths = next_info.get('query_depth')
                        
                        # Build episode info dict (use 'r', 'l', 's' for compatibility)
                        ep_info = {
                            'r': done_rewards,
                            'l': done_lengths,
                            's': done_successes
                        }
                        
                        if done_labels is not None:
                            ep_info['label'] = done_labels[done_mask]
                        if done_depths is not None:
                            ep_info['query_depth'] = done_depths[done_mask]
                        
                        # Avoid .item() sync - let TensorDict infer batch_size from tensors
                        new_td = TensorDict(ep_info, batch_size=done_rewards.shape[0], device=self.device)
                        
                        # Check if buffer is empty (batch_size is torch.Size, so check first dimension)
                        if len(self.ep_info_buffer.keys()) == 0 or self.ep_info_buffer.batch_size[0] == 0:
                            self.ep_info_buffer = new_td
                        else:
                            self.ep_info_buffer = torch.cat([self.ep_info_buffer, new_td], dim=0)
                        
                        # Accumulate for callbacks if provided
                        if hasattr(self, '_callback_manager') and self._callback_manager is not None:
                            self._callback_manager.accumulate_episode_stats(ep_info, mode="train")
                
                # Update episode lengths (vectorized)
                done_mask = dones.bool()
                self.episode_lengths[done_mask] = 0
                self.episode_lengths[~done_mask] += 1
        
        # Compute returns and advantages
        with torch.no_grad():
            # Move observation to training device if different from rollout device
            obs_for_values = self._last_obs
            if self.rollout_device != self.device:
                obs_for_values = obs_for_values.to(self.device)
            values = self.policy.predict_values(obs_for_values)
        
        self.rollout_buffer.compute_returns_and_advantage(
            last_values=values,
            dones=self._last_episode_starts
        )
        
        total_rollout_time = time.time() - rollout_start_time
        print(f"Rollout collection complete. Collected {n_steps}x{self.n_envs}={n_steps * self.n_envs} Timesteps in {total_rollout_time:.2f}s.\n\n")
        
        # Debug rollout statistics
        if self.debug_config.is_enabled('agent') and self.debug_config.debug_agent_rollout_stats:
            self._debug_rollout_stats(action_stats)
        
        # Store action stats for later retrieval
        if collect_action_stats:
            self._last_action_stats = action_stats
        
        return True
    
    def collect_rollouts_with_results(self, return_processed_results: bool = False, collect_action_stats: bool = False) -> Union[bool, Dict]:
        """
        Collect rollouts and optionally return processed results.
        
        Args:
            return_processed_results: If True, return processed results dict
            collect_action_stats: If True, collect action statistics
            
        Returns:
            If return_processed_results=False: True if successful
            If return_processed_results=True: results dict
        """
        result = self.collect_rollouts(collect_action_stats=collect_action_stats)
        
        if return_processed_results:
            # Process the buffer to extract per-slot results
            n_total = self.n_envs
            
            # Track per-slot cumulative rewards, steps, and episode counts
            slot_rewards = [0.0 for _ in range(n_total)]
            slot_steps = [0 for _ in range(n_total)]
            slot_done = [False for _ in range(n_total)]
            slot_success = [False for _ in range(n_total)]
            slot_episode_count = [0 for _ in range(n_total)]
            slot_action_counts = [[] for _ in range(n_total)] if collect_action_stats else None
            
            # First, determine where each environment's episode ends by scanning the buffer
            # This is approximate - we assume episodes end when episode_starts is True on next step
            for step in range(self.rollout_buffer.pos):
                for env in range(n_total):
                    if slot_episode_count[env] == 0:  # Only track first episode per slot
                        reward = float(self.rollout_buffer.rewards[step, env])
                        episode_start = bool(self.rollout_buffer.episode_starts[step, env])
                        
                        slot_rewards[env] += reward
                        slot_steps[env] += 1
                        
                        # Check if this is the end of episode (next step has episode_start=True or last step)
                        is_last_step = step == self.rollout_buffer.pos - 1
                        next_episode_start = False
                        if step < self.rollout_buffer.pos - 1:
                            next_episode_start = bool(self.rollout_buffer.episode_starts[step + 1, env])
                        
                        if is_last_step or next_episode_start:
                            # Assume success if reward > 0 (approximate)
                            slot_success[env] = slot_rewards[env] > 0
                            slot_done[env] = True
                            slot_episode_count[env] = 1
            
            # Now collect action statistics only up to where each environment's episode ended
            if collect_action_stats and hasattr(self, '_last_action_stats') and self._last_action_stats:
                # Reset episode counts to track which episodes we've collected stats for
                episode_collection_tracker = [0 for _ in range(n_total)]
                
                # Group action stats by environment, only collecting up to episode end
                for step_idx, step_stats in enumerate(self._last_action_stats):
                    for env_idx, n_actions in enumerate(step_stats):
                        # Only collect if this environment's episode hasn't ended yet or just ended at this step
                        if episode_collection_tracker[env_idx] == 0 and step_idx < slot_steps[env_idx]:
                            slot_action_counts[env_idx].append(n_actions)
                        
                        # Mark as complete if we've collected all steps for this environment
                        if step_idx + 1 >= slot_steps[env_idx]:
                            episode_collection_tracker[env_idx] = 1
            
            # Create traces from per-slot results
            traces = []
            for i in range(n_total):
                trace = {
                    'success': slot_success[i],
                    'steps': slot_steps[i],
                    'reward': slot_rewards[i],
                    'trace': []
                }
                
                # Add action statistics if collected
                if collect_action_stats and slot_action_counts and slot_action_counts[i]:
                    for step, num_actions in enumerate(slot_action_counts[i]):
                        trace['trace'].append({
                            'step': step,
                            'num_actions': num_actions,
                            'done': step == len(slot_action_counts[i]) - 1
                        })
                
                traces.append(trace)
            
            # Calculate statistics
            successful = sum(1 for t in traces if t['success'])
            total_reward = sum(t['reward'] for t in traces)
            total_steps = sum(t['steps'] for t in traces)
            
            # Compute average actions (branching factor)
            if collect_action_stats and slot_action_counts:
                total_actions = 0
                total_action_steps = 0
                for counts in slot_action_counts:
                    total_actions += sum(counts)
                    total_action_steps += len(counts)
                avg_actions = total_actions / total_action_steps if total_action_steps > 0 else 0.0
            else:
                avg_actions = 0.0
            
            results_dict = {
                'total_queries': n_total,
                'successful': successful,
                'avg_reward': total_reward / n_total if n_total > 0 else 0.0,
                'avg_steps': total_steps / n_total if n_total > 0 else 0.0,
                'avg_actions': avg_actions,
                'traces': traces
            }
            
            return results_dict
        
        return result
    
    @staticmethod
    def _explained_variance(values: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return targets.new_zeros(())
        var_y = torch.var(targets)
        # OPTIMIZED: Replace expensive torch.allclose with simple threshold check
        if var_y < 1e-8:
            return var_y.new_zeros(())
        return 1 - torch.var(targets - values) / var_y
    
    def train(self) -> Dict[str, float]:
        """
        Train the policy using collected rollouts.
        
        Returns:
            Dictionary of training metrics
        """
        training_start_time = time.time()
        self.policy.train()

        device = self.device

        # Accumulators kept as tensors on device
        total_policy_loss = torch.zeros((), device=device)
        total_value_loss = torch.zeros((), device=device)
        total_entropy_loss = torch.zeros((), device=device)
        total_loss = torch.zeros((), device=device)
        n_updates = 0

        clip_fractions = []
        approx_kl_divs = []

        # AMP setup
        use_amp = getattr(self.policy, "use_amp", False) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
        autocast_ctx = torch.amp.autocast("cuda") if use_amp else contextlib.nullcontext()

        if use_amp and self.verbose:
            print(f"[OPTIMIZATION] Using automatic mixed precision (AMP))")

        print(f"Training for {self.n_epochs} epochs... (AMP={use_amp})")

        # --------------------
        # Epoch loop
        # --------------------
        for epoch in range(self.n_epochs):
            epoch_start_time = time.time()
            epoch_policy_loss = torch.zeros((), device=device)
            epoch_value_loss = torch.zeros((), device=device)
            epoch_entropy = torch.zeros((), device=device)
            epoch_n_batches = 0

            # Minibatch loop
            for batch_data in self.rollout_buffer.get(batch_size=self.batch_size):
                obs, actions, old_values, old_log_probs, advantages, returns = batch_data
                actions = actions.squeeze(-1)

                # Normalize advantages
                adv_mean = advantages.mean()
                adv_std = advantages.std()
                if adv_std > 1e-8:
                    advantages = (advantages - adv_mean) / (adv_std + 1e-8)
                else:
                    advantages = advantages - adv_mean

                # --------------------
                # Forward + loss
                # --------------------
                with autocast_ctx:
                    values, log_probs, entropy = self.policy.evaluate_actions(obs, actions)

                    # Optionally, you can guard these checks with a debug flag if needed
                    if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                        raise RuntimeError("NaN or Inf detected in log_probs during PPO.train()")
                    if torch.isnan(values).any() or torch.isinf(values).any():
                        raise RuntimeError("NaN or Inf detected in values during PPO.train()")

                    # PPO clipped policy loss
                    log_ratio = log_probs - old_log_probs
                    ratio = log_ratio.exp()

                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * torch.clamp(
                        ratio,
                        1.0 - self.clip_range,
                        1.0 + self.clip_range,
                    )
                    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                    # Value loss
                    values_flat = values.flatten()
                    returns_flat = returns.flatten()

                    if self.clip_range_vf is not None:
                        # Clipped value loss
                        values_clipped = old_values + torch.clamp(
                            values_flat - old_values,
                            -self.clip_range_vf,
                            self.clip_range_vf,
                        )
                        value_loss_1 = F.mse_loss(values_flat, returns_flat)
                        value_loss_2 = F.mse_loss(values_clipped, returns_flat)
                        value_loss = torch.max(value_loss_1, value_loss_2)
                    else:
                        value_loss = F.mse_loss(values_flat, returns_flat)

                    # Entropy loss
                    entropy_mean = entropy.mean()
                    entropy_loss = -entropy_mean
                    # Total loss
                    loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                # --------------------
                # Backward + step
                # --------------------
                # set_to_none=True is a bit faster / more memory friendly
                self.optimizer.zero_grad(set_to_none=True)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    if self.max_grad_norm is not None:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.policy.parameters(),
                            self.max_grad_norm,
                        )
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.policy.parameters(),
                            self.max_grad_norm,
                        )
                    self.optimizer.step()

                # --------------------
                # Accumulate stats as tensors (no .item() here)
                # --------------------
                epoch_policy_loss += policy_loss.detach()
                epoch_value_loss += value_loss.detach()
                epoch_entropy += entropy_mean.detach()
                epoch_n_batches += 1

                # Approx KL + clip fraction (reuse log_ratio / ratio)
                with torch.no_grad():
                    approx_kl_divs.append(((ratio - 1.0) - log_ratio).mean())
                    clip_fractions.append(
                        (ratio - 1.0).abs().gt(self.clip_range).float().mean()
                    )

            epoch_time = time.time() - epoch_start_time
            
            # Epoch logging
            if epoch_n_batches > 0:
                total_policy_loss += epoch_policy_loss
                total_value_loss += epoch_value_loss
                total_entropy_loss += epoch_entropy
                n_updates += epoch_n_batches

                # Only print per-epoch details in higher verbosity
                avg_policy_loss = (epoch_policy_loss / epoch_n_batches).item()
                avg_value_loss = (epoch_value_loss / epoch_n_batches).item()
                avg_entropy = (epoch_entropy / epoch_n_batches).item()
                print(
                    f"  Epoch {epoch + 1}/{self.n_epochs}: "
                    f"policy_loss={avg_policy_loss:.7f}, "
                    f"value_loss={avg_value_loss:.7f}, "
                    f"entropy={avg_entropy:.7f}, "
                    f"time={epoch_time:.2f}s"
                )
        
        # --------------------
        # Final metrics
        # --------------------
        if n_updates == 0:
            metrics = {
                "train/policy_loss": 0.0,
                "train/value_loss": 0.0,
                "train/entropy": 0.0,
                "train/total_loss": 0.0,
                "train/approx_kl": 0.0,
                "train/clip_fraction": 0.0,
                "train/n_updates": 0,
            }
            self.logger_dict.update(metrics)
            return metrics

        total_loss = (
            total_policy_loss
            + self.vf_coef * total_value_loss
            + self.ent_coef * total_entropy_loss
        )

        with torch.no_grad():
            mean_losses = torch.stack(
                [total_policy_loss, total_value_loss, total_entropy_loss, total_loss]
            ) / float(n_updates)
            mean_pol, mean_val, mean_ent, mean_total = mean_losses.cpu().tolist()

            mean_kl = (
                torch.stack(approx_kl_divs).mean().cpu().item()
                if approx_kl_divs
                else 0.0
            )
            mean_clip_frac = (
                torch.stack(clip_fractions).mean().cpu().item()
                if clip_fractions
                else 0.0
            )

        metrics = {
            "train/policy_loss": float(mean_pol),
            "train/value_loss": float(mean_val),
            # total_entropy_loss was accumulated as -entropy
            "train/entropy": float(-mean_ent),
            "train/total_loss": float(mean_total),
            "train/approx_kl": mean_kl,
            "train/clip_fraction": mean_clip_frac,
            "train/n_updates": int(n_updates),
        }

        # Compute explained variance
        explained_var = self._explained_variance(
            self.rollout_buffer.flat_values, self.rollout_buffer.flat_returns
        ).item()
        metrics["train/explained_variance"] = explained_var

        self.logger_dict.update(metrics)

        # Optional: debug hook (uses last batch tensors)
        if self.debug_config.is_enabled('agent') and self.debug_config.debug_agent_train_stats:
            self._debug_train_stats(metrics, advantages, values, returns)
        
        training_time = time.time() - training_start_time
        print(f"Training complete in {training_time:.2f}s over {self.n_epochs} epochs")
        return metrics
    
    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 1,
        reset_num_timesteps: bool = True,
        callback_manager=None,
    ) -> "PPO":
        """
        Train the agent for a given number of timesteps.
        
        Args:
            total_timesteps: Total number of timesteps to train for
            log_interval: Logging interval (in iterations)
            reset_num_timesteps: Whether to reset timestep counter
            callback_manager: Optional TorchRLCallbackManager for callbacks
        
        Returns:
            Self
        """
        if reset_num_timesteps:
            self.num_timesteps = 0
        
        # Setup model
        self._setup_model()
        
        iteration = 0
        start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"Starting PPO training for {total_timesteps} timesteps")
        print(f"{'='*80}\n")
        
        while self.num_timesteps < total_timesteps:
            iteration += 1
            
            print(f"\n--- Iteration {iteration} ---")
            print(f"Timesteps: {self.num_timesteps}/{total_timesteps}")
            
            # Collect rollouts
            rollout_start = time.time()
            continue_training = self.collect_rollouts()
            rollout_time = time.time() - rollout_start
            
            if not continue_training:
                break
            
            # Train
            train_start = time.time()
            metrics = self.train()
            train_time = time.time() - train_start
            
            # Prepare rollout metrics for callbacks
            rollout_metrics = {}
            has_ep_info = len(self.ep_info_buffer.keys()) > 0 and len(self.ep_info_buffer.batch_size) > 0 and self.ep_info_buffer.batch_size[0] > 0
            if has_ep_info:
                rollout_metrics['ep_rew'] = self.ep_info_buffer['r'].mean().item()
                rollout_metrics['ep_len'] = self.ep_info_buffer['l'].float().mean().item()
                rollout_metrics['success_rate'] = self.ep_info_buffer['s'].float().mean().item()
            
            # Log
            if log_interval > 0 and iteration % log_interval == 0:
                elapsed_time = time.time() - start_time
                # num_timesteps increases by n_envs per cicle (while loop until total_timesteps)
                # fps is measured as how long it takes, to do rollouts + training every num_timesteps
                fps = int(self.num_timesteps / elapsed_time) if elapsed_time > 0 else 0
                # from metrics, get relevant info such as explained variance
                explained_variance = metrics.get("train/explained_variance", 0.0)
                print(f"\nIteration {iteration} summary:")
                print(f"  Explained variance: {explained_variance:.4f}")
                print(f"  Timesteps: {self.num_timesteps}/{total_timesteps}")
                print(f"  FPS: {fps}")
                print(f"  Rollout time: {rollout_time:.2f}s")
                print(f"  Train time: {train_time:.2f}s")
                
                # Check if ep_info_buffer has data (keys and non-zero batch size)
                # batch_size is always a torch.Size object (tuple-like)
                if has_ep_info:
                    print(f"  Episode reward mean: {rollout_metrics['ep_rew']:.2f}")
                    print(f"  Episode length mean: {rollout_metrics['ep_len']:.2f}")
                    print(f"  Success rate: {rollout_metrics['success_rate']*100:.2f}%")
            
            # Store callback manager for use during rollout
            self._callback_manager = callback_manager
            
            # Call callbacks
            if callback_manager is not None:
                # Iteration end callback (with metrics for checkpointing)
                callback_manager.on_iteration_end(iteration, self.num_timesteps, self.n_envs, rollout_metrics)
                
                # Evaluation callback
                if callback_manager.should_evaluate(iteration):
                    eval_start = time.time()
                    print("\n" + "="*80)
                    print("Starting evaluation")
                    print("="*80)
                    
                    callback_manager.on_evaluation_start(iteration, self.num_timesteps)
                    
                    # Perform MRR evaluation if using MRREvaluationCallback
                    eval_metrics = {}
                    if hasattr(callback_manager.eval_callback, 'evaluate_mrr'):
                        # MRR evaluation - pass the full policy
                        eval_metrics = callback_manager.eval_callback.evaluate_mrr(self.policy)
                    
                    # Call evaluation end
                    is_best = callback_manager.on_evaluation_end(iteration, self.num_timesteps, eval_metrics)
                    
                    eval_time = time.time() - eval_start
                    print(f"Evaluation took {eval_time:.2f}s")
                    print("="*80 + "\n")
            
            # Clear buffer after callbacks
            if has_ep_info:
                self.ep_info_buffer = TensorDict({}, batch_size=torch.Size([0]), device=self.device)
        
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"Training complete!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Final timesteps: {self.num_timesteps}")
        print(f"Explained variance: {self.logger_dict.get('train/explained_variance', 0.0):.4f}")
        print(f"{'='*80}\n")
        
        return self
    
    def predict(
        self,
        obs: TensorDict,
        deterministic: bool = False
    ) -> TensorDict:
        """
        Predict actions for given observations.
        
        Args:
            obs: Observations as TensorDict
            deterministic: If True, select argmax action
        
        Returns:
            Actions wrapped in TensorDict
        """
        with torch.no_grad():
            actions, _, _ = self.policy(obs, deterministic=deterministic)
        
        # Return TensorDict for compatibility with environment
        return TensorDict(
            {"action": actions},
            batch_size=self.n_envs,
            device=self.device
        )
    
    def get_logger_dict(self) -> Dict[str, Any]:
        """Get the current logger dictionary."""
        return self.logger_dict.copy()
    
    def _debug_rollout_stats(self, action_stats):
        """Debug output for rollout statistics."""
        import numpy as np
        
        print(f"\n{self.debug_config.debug_prefix} [AGENT ROLLOUT STATS]")
        
        if action_stats:
            # Convert to numpy for easier stats
            action_counts = np.concatenate(action_stats, axis=0)  # Shape: (total_steps, n_envs)
            
            # Per-step statistics
            avg_actions_per_step = action_counts.mean(axis=1)  # Average across envs per step
            print(f"  Actions available per step:")
            print(f"    Mean: {avg_actions_per_step.mean():.2f}")
            print(f"    Min: {avg_actions_per_step.min():.2f}")
            print(f"    Max: {avg_actions_per_step.max():.2f}")
            print(f"    Std: {avg_actions_per_step.std():.2f}")
            
            # Overall statistics
            print(f"  Overall action availability:")
            print(f"    Mean: {action_counts.mean():.2f}")
            print(f"    Min: {action_counts.min():.0f}")
            print(f"    Max: {action_counts.max():.0f}")
            
            # Distribution analysis (helps understand entropy issues)
            hist, bins = np.histogram(action_counts.flatten(), bins=10)
            print(f"  Distribution of available actions:")
            for i in range(len(hist)):
                print(f"    [{bins[i]:.1f}-{bins[i+1]:.1f}]: {hist[i]} ({100*hist[i]/hist.sum():.1f}%)")
        
        # Episode statistics from buffer
        if len(self.ep_info_buffer.keys()) > 0 and self.ep_info_buffer.batch_size[0] > 0:
            print(f"  Episodes completed: {self.ep_info_buffer.batch_size[0]}")
            print(f"    Avg reward: {self.ep_info_buffer['r'].mean().item():.2f}")
            print(f"    Avg length: {self.ep_info_buffer['l'].float().mean().item():.2f}")
            if 's' in self.ep_info_buffer.keys():
                print(f"    Success rate: {self.ep_info_buffer['s'].float().mean().item()*100:.2f}%")
    
    def _debug_train_stats(self, metrics, advantages, values, returns):
        """Debug output for training statistics."""
        print(f"\n{self.debug_config.debug_prefix} [AGENT TRAIN STATS]")
        
        # Core metrics
        print(f"  Training metrics:")
        print(f"    Policy loss: {metrics['train/policy_loss']:.6f}")
        print(f"    Value loss: {metrics['train/value_loss']:.6f}")
        print(f"    Entropy: {metrics['train/entropy']:.6f}")
        print(f"    Total loss: {metrics['train/total_loss']:.6f}")
        print(f"    Approx KL: {metrics['train/approx_kl']:.6f}")
        print(f"    Clip fraction: {metrics['train/clip_fraction']:.4f}")
        
        # Advantage statistics
        print(f"  Advantages:")
        print(f"    Mean: {advantages.mean().item():.4f}")
        print(f"    Std: {advantages.std().item():.4f}")
        print(f"    Min: {advantages.min().item():.4f}")
        print(f"    Max: {advantages.max().item():.4f}")
        
        # Value statistics
        print(f"  Values:")
        print(f"    Mean: {values.mean().item():.4f}")
        print(f"    Std: {values.std().item():.4f}")
        print(f"    Min: {values.min().item():.4f}")
        print(f"    Max: {values.max().item():.4f}")
        
        # Return statistics
        print(f"  Returns:")
        print(f"    Mean: {returns.mean().item():.4f}")
        print(f"    Std: {returns.std().item():.4f}")
        print(f"    Min: {returns.min().item():.4f}")
        print(f"    Max: {returns.max().item():.4f}")
        
        # Entropy analysis (key for debugging low entropy)
        if metrics['train/entropy'] < 0.1:
            print(f"\n  ⚠️  WARNING: Low entropy detected ({metrics['train/entropy']:.6f})")
            print(f"      This suggests the policy is very confident/deterministic.")
            print(f"      Check:")
            print(f"        - Are there enough valid actions? (see rollout stats)")
            print(f"        - Is the action space too small?")
            print(f"        - Are logits too extreme? (see model debug)")
            print(f"        - Is entropy coefficient too low? (current: {self.ent_coef})")
