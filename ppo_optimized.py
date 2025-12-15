"""
PPO (Proximal Policy Optimization) for Optimized Environment.

This module implements PPO for use with EvalEnvOptimized which uses
EvalObs/EvalState instead of TensorDict.

Key Differences from ppo.py:
    - Uses step_functional() instead of step_and_maybe_reset()
    - Works with EvalObs NamedTuples
    - Uses RolloutBufferOptimized
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Dict, List, Tuple, Any
from tensordict import TensorDict

from rollout_optimized import RolloutBufferOptimized
from env_optimized import EvalEnvOptimized, EvalObs, EvalState


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Compute fraction of variance that y_pred explains about y_true.
    Returns 1 - Var[y_true - y_pred] / Var[y_true]
    """
    var_y = torch.var(y_true)
    return 1.0 - torch.var(y_true - y_pred) / (var_y + 1e-8)


class PPOOptimized:
    """
    Proximal Policy Optimization for EvalEnvOptimized.
    
    This implementation works with the functional/immutable state approach
    of EvalEnvOptimized rather than the TensorDict-based BatchedEnv.
    
    Args:
        policy: Actor-critic policy network
        env: EvalEnvOptimized instance
        n_steps: Steps per environment per rollout
        learning_rate: Adam optimizer learning rate
        n_epochs: Number of optimization epochs per rollout
        batch_size: Minibatch size for gradient updates
        gamma: Discount factor
        gae_lambda: GAE smoothing parameter
        clip_range: PPO clipping parameter
        normalize_advantage: Whether to normalize advantages
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Gradient clipping norm
    """
    
    def __init__(
        self,
        policy: nn.Module,
        env: EvalEnvOptimized,
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
        target_kl: Optional[float] = None,
        device: torch.device = None,
        verbose: bool = True,
        seed: Optional[int] = None,
        parity: bool = False,
    ):
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
        self.target_kl = target_kl
        self.device = device if device is not None else torch.device('cpu')
        self.verbose = verbose
        self.seed = seed
        self.parity = parity
        
        # Get environment parameters
        self.n_envs = env.batch_size
        self.padding_atoms = env.padding_atoms
        self.padding_states = env.padding_states
        
        # Create rollout buffer
        self.rollout_buffer = RolloutBufferOptimized(
            buffer_size=n_steps,
            n_envs=self.n_envs,
            device=self.device,
            gamma=gamma,
            gae_lambda=gae_lambda,
            padding_atoms=self.padding_atoms,
            padding_states=self.padding_states,
            parity=parity,
        )
        
        # Persistent state
        self._last_state: Optional[EvalState] = None
        self._last_obs: Optional[EvalObs] = None
        self.num_timesteps = 0
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate,
            eps=1e-5,
        )
    
    def _obs_to_tensordict(self, obs: EvalObs) -> TensorDict:
        """Convert EvalObs to TensorDict for policy forward pass."""
        return TensorDict({
            'sub_index': obs.sub_index,
            'derived_sub_indices': obs.derived_sub_indices,
            'action_mask': obs.action_mask,
        }, batch_size=[obs.sub_index.shape[0]], device=self.device)
    
    def collect_rollouts(
        self,
        current_state: EvalState,
        current_obs: EvalObs,
        episode_starts: torch.Tensor,
        current_episode_reward: torch.Tensor,
        current_episode_length: torch.Tensor,
        episode_rewards: list,
        episode_lengths: list,
        iteration: int,
        return_traces: bool = False,
        query_pool: Optional[torch.Tensor] = None,
        per_env_ptrs: Optional[torch.Tensor] = None,
    ) -> Tuple[EvalState, EvalObs, torch.Tensor, torch.Tensor, torch.Tensor, int, Optional[List], Optional[torch.Tensor]]:
        """
        Collect experiences using the current policy and fill the rollout buffer.
        
        Uses EvalEnvOptimized's functional step interface.
        
        Args:
            current_state: Current EvalState from previous rollout
            current_obs: Current EvalObs observation
            episode_starts: [N] Binary mask for episode starts
            current_episode_reward: [N] Accumulator for rewards
            current_episode_length: [N] Accumulator for lengths
            episode_rewards: List to append completed episode rewards
            episode_lengths: List to append completed episode lengths
            iteration: Current global iteration number
            return_traces: If True, collect step-by-step traces
            
        Returns:
            Tuple containing:
                - next_state: Latest EvalState
                - next_obs: Latest observation
                - episode_starts: Updated start masks
                - current_episode_reward: Updated reward accumulators
                - current_episode_length: Updated length accumulators
                - total_steps: Total steps collected
                - traces: Optional list of trace dictionaries
        """
        self.policy.eval()
        self.rollout_buffer.reset()
        
        traces = [] if return_traces else None
        n_collected = 0
        
        state = current_state
        obs = current_obs
        
        with torch.no_grad():
            while n_collected < self.n_steps:
                if self.verbose and n_collected % max(1, self.n_steps // 5) == 0:
                    print(f"Collecting rollouts: {n_collected}/{self.n_steps} steps")
                
                # Snapshot observations before step
                obs_snapshot_sub = obs.sub_index.clone()
                obs_snapshot_derived = obs.derived_sub_indices.clone()
                obs_snapshot_mask = obs.action_mask.clone()
                
                # Convert EvalObs to TensorDict for policy value prediction
                obs_td = self._obs_to_tensordict(obs)
                
                # Predict values (critic) separate from actor (step_with_policy)
                # This keeps the training rollout collected via the same compiled path as eval
                values = self.policy.predict_values(obs_td)
                
                # Step environment using unified compiled step
                # This handles policy forward, action selection, env step, and auto-reset
                step_result = self.env.step_with_policy(
                    state, obs, query_pool, per_env_ptrs,
                    deterministic=False,
                    eval_mode=False
                )
                
                # step_with_policy returns: state, obs, actions, log_probs, values, rewards, dones, ptrs, mask
                # We ignore the inline values since we computed them separately above
                new_state, new_obs, actions, log_probs, _step_values, rewards, dones, new_ptrs, _ = step_result
                
                # Update pointers
                if query_pool is not None:
                    per_env_ptrs = new_ptrs
                
                # Store transition
                self.rollout_buffer.add(
                    sub_index=obs_snapshot_sub,
                    derived_sub_indices=obs_snapshot_derived,
                    action_mask=obs_snapshot_mask,
                    action=actions,
                    reward=rewards,
                    episode_start=episode_starts,
                    value=values.flatten(),  # predict_values returns [B, 1]
                    log_prob=log_probs,
                )
                
                # Collect traces if requested
                if return_traces:
                    for idx in range(self.n_envs):
                        trace_entry = {
                            "step": n_collected,
                            "env": idx,
                            "state_obs": {
                                "sub_index": obs_snapshot_sub[idx].cpu().numpy().copy(),
                                "derived_sub_indices": obs_snapshot_derived[idx].cpu().numpy().copy(),
                                "action_mask": obs_snapshot_mask[idx].cpu().numpy().copy(),
                            },
                            "action": int(actions[idx]),
                            "reward": float(rewards[idx]),
                            "done": bool(dones[idx]),
                            "value": float(values[idx]),
                            "log_prob": float(log_probs[idx]),
                        }
                        traces.append(trace_entry)
                
                # Update statistics
                current_episode_reward += rewards
                current_episode_length += 1
                n_collected += 1
                
                # Handle episode ends
                if dones.any():
                    done_indices = torch.where(dones)[0]
                    
                    for idx in done_indices:
                        ep_reward = float(current_episode_reward[idx])
                        ep_length = int(current_episode_length[idx])
                        episode_rewards.append(ep_reward)
                        episode_lengths.append(ep_length)
                        
                        # Reset episode stats
                        current_episode_reward[idx] = 0.0
                        current_episode_length[idx] = 0
                    
                    # Mark episode starts for next step
                    episode_starts = dones.float()
                else:
                    episode_starts = torch.zeros(self.n_envs, dtype=torch.float32, device=self.device)
                
                state = new_state
                obs = new_obs

            
            # Compute last values for bootstrapping
            last_obs_td = self._obs_to_tensordict(obs)
            last_values = self.policy.predict_values(last_obs_td)
        
        # Compute advantages and returns
        self.rollout_buffer.compute_returns_and_advantage(
            last_values=last_values,
            dones=state.done.float()
        )
        
        result = (
            state,
            obs,
            episode_starts,
            current_episode_reward,
            current_episode_length,
            n_collected * self.n_envs,
        )
        
        if return_traces:
            result = result + (traces,)
        else:
            result = result + (None,)
        
        # Return per_env_ptrs if query_pool was provided
        if query_pool is not None:
            result = result + (per_env_ptrs,)
        
        return result
        
    def train(self, return_traces: bool = False) -> Dict[str, float]:
        """
        Update policy using the currently collected rollout buffer.
        
        Returns:
            Dict containing average training metrics
        """
        self.policy.train()
        
        # Accumulators
        pg_losses = []
        value_losses = []
        entropy_losses = []
        clip_fractions = []
        approx_kl_divs = []
        
        train_traces = [] if return_traces else None
        
        continue_training = True
        for epoch in range(self.n_epochs):
            epoch_kl_divs = []
            
            for batch_data in self.rollout_buffer.get(batch_size=self.batch_size):
                (sub_index, derived_sub_indices, action_mask,
                 actions, old_values, old_log_probs, advantages, returns) = batch_data
                
                # Build observation TensorDict
                batch_size = sub_index.shape[0]
                obs_td = TensorDict({
                    'sub_index': sub_index,
                    'derived_sub_indices': derived_sub_indices,
                    'action_mask': action_mask,
                }, batch_size=[batch_size], device=self.device)
                
                # Normalize advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Forward pass
                _, values, log_probs, entropy = self.policy.evaluate_actions(obs_td, actions)
                values = values.flatten()
                
                # Compute losses
                log_ratio = log_probs - old_log_probs
                ratio = torch.exp(log_ratio)
                
                # Clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                policy_loss = -torch.minimum(policy_loss_1, policy_loss_2).mean()
                
                # Clip fraction
                clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float())
                
                # Value loss
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = old_values + torch.clamp(
                        values - old_values, -self.clip_range_vf, self.clip_range_vf
                    )
                value_loss = F.mse_loss(returns, values_pred)
                
                # Entropy loss
                if entropy is None:
                    entropy_loss = -torch.mean(-log_probs)
                else:
                    entropy_loss = -torch.mean(entropy)
                
                # Total loss
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                
                # Approx KL
                approx_kl_div = torch.mean((ratio - 1.0) - log_ratio)
                epoch_kl_divs.append(approx_kl_div.item())
                
                # Log losses
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fractions.append(clip_fraction.item())
                
                # Collect traces
                if return_traces:
                    train_traces.append({
                        "epoch": epoch,
                        "batch_size": batch_size,
                        "policy_loss": policy_loss.item(),
                        "value_loss": value_loss.item(),
                        "entropy_loss": entropy_loss.item(),
                        "clip_fraction": clip_fraction.item(),
                    })
                
                # KL divergence early stopping
                if self.target_kl is not None and approx_kl_div.item() > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch} due to KL divergence: {approx_kl_div.item():.4f}")
                    break
                
                # Optimizer step
                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            approx_kl_divs.extend(epoch_kl_divs)
            
            if not continue_training:
                break
            
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.n_epochs}")
        
        # Compute metrics
        with torch.no_grad():
            ev = explained_variance(self.rollout_buffer.values, self.rollout_buffer.returns)
        
        metrics = {
            "policy_loss": sum(pg_losses) / len(pg_losses) if pg_losses else 0.0,
            "value_loss": sum(value_losses) / len(value_losses) if value_losses else 0.0,
            "entropy": -sum(entropy_losses) / len(entropy_losses) if entropy_losses else 0.0,
            "clip_fraction": sum(clip_fractions) / len(clip_fractions) if clip_fractions else 0.0,
            "approx_kl": sum(approx_kl_divs) / len(approx_kl_divs) if approx_kl_divs else 0.0,
            "explained_var": ev.item(),
        }
        
        if return_traces:
            metrics["traces"] = train_traces
        
        return metrics
    
    def learn(
        self,
        total_timesteps: int,
        queries: torch.Tensor,
        reset_num_timesteps: bool = True,
    ) -> None:
        """
        Execute the PPO main loop: alternate between collecting rollouts and training.
        
        Args:
            total_timesteps: Total number of environment steps to train for
            queries: [N, 3] Query tensor to initialize environments
            reset_num_timesteps: If True, reset the timestep counter
        """
        if reset_num_timesteps:
            self.num_timesteps = 0
        else:
            total_timesteps += self.num_timesteps
        
        iteration = 0
        
        # Initialize environment state
        state = self.env.init_state_from_queries(queries)
        
        # Create initial observation
        action_mask = torch.arange(self.padding_states, device=self.device).unsqueeze(0) < state.derived_counts.unsqueeze(1)
        obs = EvalObs(
            sub_index=state.current_states.unsqueeze(1),
            derived_sub_indices=state.derived_states,
            action_mask=action_mask,
        )
        
        episode_starts = torch.ones(self.n_envs, dtype=torch.float32, device=self.device)
        current_episode_reward = torch.zeros(self.n_envs, dtype=torch.float32, device=self.device)
        current_episode_length = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
        episode_rewards = []
        episode_lengths = []
        
        while self.num_timesteps < total_timesteps:
            iteration += 1
            
            # Collect rollouts
            result = self.collect_rollouts(
                current_state=state,
                current_obs=obs,
                episode_starts=episode_starts,
                current_episode_reward=current_episode_reward,
                current_episode_length=current_episode_length,
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                iteration=iteration,
            )
            
            state, obs, episode_starts, current_episode_reward, current_episode_length, n_steps, _ = result
            self.num_timesteps += n_steps
            
            # Train
            train_metrics = self.train()
            
            if self.verbose:
                print(f"Iteration {iteration}, timesteps: {self.num_timesteps}/{total_timesteps}")
                print(f"  policy_loss: {train_metrics['policy_loss']:.4f}, "
                      f"value_loss: {train_metrics['value_loss']:.4f}, "
                      f"entropy: {train_metrics['entropy']:.4f}")

    def evaluate(
        self,
        queries: torch.Tensor,
        sampler: Any,
        n_corruptions: int = 50,
        corruption_modes: Tuple[str, ...] = ("head", "tail"),
        chunk_queries: int = 50,
        deterministic: bool = True,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate the current policy using the optimized evaluation loop.
        
        Args:
            queries: [N, 3] Query tensor to evaluate on
            sampler: Corruption sampler
            n_corruptions: Number of corruptions per query
            corruption_modes: Modes to evaluate ('head', 'tail')
            chunk_queries: Chunk size handling
            deterministic: Use deterministic actions
            verbose: Print progress
            
        Returns:
            Dictionary of metrics
        """
        from model_eval_optimized import evaluate_policy
        
        results = evaluate_policy(
            env=self.env,
            queries=queries,
            sampler=sampler,
            n_corruptions=n_corruptions,
            corruption_modes=corruption_modes,
            chunk_queries=chunk_queries,
            verbose=verbose,
            deterministic=deterministic,
        )
        return results
