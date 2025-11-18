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

from ppo.pposb3_rollout import RolloutBuffer
from ppo.pposb3_model import ActorCriticPolicy


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
        device: PyTorch device
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
        verbose: int = 1,
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
        self.verbose = verbose
        
        # Initialize optimizer
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
                actions, values, log_probs = self.policy(self._last_obs)
                
                # Collect action statistics if requested
                if collect_action_stats:
                    action_mask = self._last_obs.get('action_mask')
                    if action_mask is not None:
                        # Count valid actions for each environment
                        n_actions = action_mask.sum(dim=-1).cpu().numpy()
                        action_stats.append(n_actions)
                
                # Wrap actions in TensorDict for environment
                action_td = TensorDict(
                    {"action": actions},
                    batch_size=self.n_envs,
                    device=self.device
                )
                
                # Take step in environment (returns TensorDict)
                # TorchRL's EnvBase wraps the output in a 'next' key
                step_result = self.env.step(action_td)
                
                # Extract next state info
                next_info = step_result['next']
                obs = next_info  # Next observation (contains all obs keys)
                
                # Get reward, done, and success (all have shape [n_envs, 1])
                rewards = next_info['reward'].squeeze(-1).to(self.device)
                dones = next_info['done'].squeeze(-1).float().to(self.device)
                is_success = next_info.get('is_success')
                if is_success is not None:
                    is_success = is_success.squeeze(-1)
                
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
                
                # Collect episode info
                if is_success is not None:
                    done_mask = dones.bool()
                    if done_mask.any():
                        done_rewards = rewards[done_mask]
                        done_lengths = self.episode_lengths[done_mask]
                        done_successes = is_success[done_mask]
                        
                        new_td = TensorDict({
                            'r': done_rewards,
                            'l': done_lengths,
                            's': done_successes
                        }, batch_size=int(done_mask.sum().item()), device=self.device)
                        
                        # Check if buffer is empty (batch_size is torch.Size, so check first dimension)
                        if len(self.ep_info_buffer.keys()) == 0 or self.ep_info_buffer.batch_size[0] == 0:
                            self.ep_info_buffer = new_td
                        else:
                            self.ep_info_buffer = torch.cat([self.ep_info_buffer, new_td], dim=0)
                
                # Update episode lengths (vectorized)
                done_mask = dones.bool()
                self.episode_lengths[done_mask] = 0
                self.episode_lengths[~done_mask] += 1
        
        # Compute returns and advantages
        with torch.no_grad():
            values = self.policy.predict_values(self._last_obs)
        
        self.rollout_buffer.compute_returns_and_advantage(
            last_values=values,
            dones=self._last_episode_starts
        )
        
        total_rollout_time = time.time() - rollout_start_time
        print(f"Rollout collection complete. Collected {n_steps}x{self.n_envs}={n_steps * self.n_envs} Timesteps in {total_rollout_time:.2f}s.")
        
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
    
    def train(self) -> Dict[str, float]:
        """
        Train the policy using collected rollouts.
        
        Returns:
            Dictionary of training metrics
        """
        self.policy.train()
        
        # Initialize logging
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_loss = 0.0
        n_updates = 0
        
        clip_fractions = []
        approx_kl_divs = []
        
        print(f"Training for {self.n_epochs} epochs...")
        
        for epoch in range(self.n_epochs):
            epoch_start_time = time.time()
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_entropy = 0.0
            epoch_n_batches = 0
            
            # Iterate over minibatches
            for batch_data in self.rollout_buffer.get(batch_size=self.batch_size):
                obs, actions, old_values, old_log_probs, advantages, returns = batch_data
                
                # Normalize advantages
                adv_mean = advantages.mean()
                adv_std = advantages.std()
                if adv_std > 1e-8:
                    advantages = (advantages - adv_mean) / (adv_std + 1e-8)
                else:
                    # If std is too small, just center the advantages
                    advantages = advantages - adv_mean
                
                # Evaluate actions with current policy
                values, log_probs, entropy = self.policy.evaluate_actions(obs, actions.squeeze(-1))
                
                # Check for NaN/Inf in outputs
                if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                    raise RuntimeError("NaN or Inf detected in log_probs during PPO.train()")
                if torch.isnan(values).any() or torch.isinf(values).any():
                    raise RuntimeError("NaN or Inf detected in values during PPO.train()")

                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - old_log_probs)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                # Value loss
                # Ensure values and returns have the same shape
                values_flat = values.flatten()
                returns_flat = returns.flatten()
                
                if self.clip_range_vf is not None:
                    # Clipped value loss
                    values_clipped = old_values + torch.clamp(
                        values_flat - old_values,
                        -self.clip_range_vf,
                        self.clip_range_vf
                    )
                    value_loss_1 = F.mse_loss(values_flat, returns_flat)
                    value_loss_2 = F.mse_loss(values_clipped, returns_flat)
                    value_loss = torch.max(value_loss_1, value_loss_2)
                else:
                    value_loss = F.mse_loss(values_flat, returns_flat)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.max_grad_norm
                    )
                
                self.optimizer.step()
                
                # Logging
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_entropy += entropy.mean().item()
                epoch_n_batches += 1
                
                # Compute approximate KL divergence and clip fraction
                with torch.no_grad():
                    log_ratio = log_probs - old_log_probs
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item()
                    approx_kl_divs.append(approx_kl_div)
                    
                    clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float()).item()
                    clip_fractions.append(clip_fraction)
            
            epoch_time = time.time() - epoch_start_time
            
            # Epoch logging
            total_policy_loss += epoch_policy_loss
            total_value_loss += epoch_value_loss
            total_entropy_loss += epoch_entropy
            n_updates += epoch_n_batches
            
            avg_policy_loss = epoch_policy_loss / epoch_n_batches if epoch_n_batches > 0 else 0.0
            avg_value_loss = epoch_value_loss / epoch_n_batches if epoch_n_batches > 0 else 0.0
            avg_entropy = epoch_entropy / epoch_n_batches if epoch_n_batches > 0 else 0.0
            print(f"  Epoch {epoch + 1}/{self.n_epochs}: "
                    f"policy_loss={avg_policy_loss:.7f}, "
                    f"value_loss={avg_value_loss:.7f}, "
                    f"entropy={avg_entropy:.7f}, "
                    f"time={epoch_time:.2f}s")
        
        total_loss = total_policy_loss + self.vf_coef * total_value_loss + self.ent_coef * total_entropy_loss
        
        # Prepare logging dict
        # Note: total_entropy_loss is accumulated as -entropy (line 283), so we negate it here to get actual entropy
        metrics = {
            'train/policy_loss': total_policy_loss / n_updates if n_updates > 0 else 0.0,
            'train/value_loss': total_value_loss / n_updates if n_updates > 0 else 0.0,
            'train/entropy': -total_entropy_loss / n_updates if n_updates > 0 else 0.0,  # Negate to get actual entropy
            'train/total_loss': total_loss / n_updates if n_updates > 0 else 0.0,
            'train/approx_kl': sum(approx_kl_divs) / len(approx_kl_divs) if approx_kl_divs else 0.0,
            'train/clip_fraction': sum(clip_fractions) / len(clip_fractions) if clip_fractions else 0.0,
            'train/n_updates': n_updates,
        }
        
        self.logger_dict.update(metrics)
        
        print(f"Training complete. Average losses:")
        print(f"  Policy loss: {metrics['train/policy_loss']:.7f}")
        print(f"  Value loss: {metrics['train/value_loss']:.7f}")
        print(f"  Entropy: {metrics['train/entropy']:.7f}")
        print(f"  Total loss: {metrics['train/total_loss']:.7f}")
        print(f"  Approx KL: {metrics['train/approx_kl']:.7f}")
        print(f"  Clip fraction: {metrics['train/clip_fraction']:.7f}")
        
        return metrics
    
    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 1,
        reset_num_timesteps: bool = True,
    ) -> "PPO":
        """
        Train the agent for a given number of timesteps.
        
        Args:
            total_timesteps: Total number of timesteps to train for
            log_interval: Logging interval (in iterations)
            reset_num_timesteps: Whether to reset timestep counter
        
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
            
            # Log
            if log_interval > 0 and iteration % log_interval == 0:
                elapsed_time = time.time() - start_time
                # num_timesteps increases by n_envs per cicle (while loop until total_timesteps)
                # fps is measured as how long it takes, to do rollouts + training every num_timesteps
                fps = int(self.num_timesteps / elapsed_time) if elapsed_time > 0 else 0
                
                print(f"\nIteration {iteration} summary:")
                print(f"  Timesteps: {self.num_timesteps}/{total_timesteps}")
                print(f"  FPS: {fps}")
                print(f"  Rollout time: {rollout_time:.2f}s")
                print(f"  Train time: {train_time:.2f}s")
                
                # Check if ep_info_buffer has data (keys and non-zero batch size)
                # batch_size is always a torch.Size object (tuple-like)
                has_ep_info = len(self.ep_info_buffer.keys()) > 0 and len(self.ep_info_buffer.batch_size) > 0 and self.ep_info_buffer.batch_size[0] > 0
                if has_ep_info:
                    print(f"  Episode reward mean: {self.ep_info_buffer['r'].mean().item():.2f}")
                    print(f"  Episode length mean: {self.ep_info_buffer['l'].float().mean().item():.2f}")
                    print(f"  Success rate: {self.ep_info_buffer['s'].float().mean().item()*100:.2f}%")
                    
                    # Clear buffer after logging
                    self.ep_info_buffer = TensorDict({}, batch_size=torch.Size([0]), device=self.device)
        
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"Training complete!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Final timesteps: {self.num_timesteps}")
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
