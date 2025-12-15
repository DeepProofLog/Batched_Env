"""
PPO (Proximal Policy Optimization) for Optimized Environment.

This module implements PPO for use with EvalEnvOptimized which uses
EvalObs/EvalState instead of TensorDict.

Key Differences from ppo.py:
    - Uses step_functional() instead of step_and_maybe_reset()
    - Works with EvalObs NamedTuples
    - Uses RolloutBufferOptimized
    
Evaluation Methods:
    - evaluate_policy(): Run trajectories for a batch of queries
    - evaluate_with_corruptions(): Full MRR/Hits@K evaluation with corruptions
    
The PPOOptimized class maintains a fixed_batch_size that is set during
compile() and enforced during evaluation to avoid recompilation with
CUDA graphs (reduce-overhead mode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Optional, Callable, Dict, List, Tuple, Any, Sequence
from tensordict import TensorDict

from rollout_optimized import RolloutBufferOptimized
from env_optimized import EvalEnvOptimized, EvalObs, EvalState


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_metrics_from_ranks(ranks: Tensor) -> Dict[str, float]:
    """Compute MRR and Hits@K from ranks tensor."""
    if ranks.numel() == 0:
        return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}
    ranks_float = ranks.float()
    return {
        "MRR": float(torch.mean(1.0 / ranks_float).item()),
        "Hits@1": float(torch.mean((ranks_float <= 1.0).float()).item()),
        "Hits@3": float(torch.mean((ranks_float <= 3.0).float()).item()),
        "Hits@10": float(torch.mean((ranks_float <= 10.0).float()).item()),
    }


def compute_optimal_batch_size(
    chunk_queries: int = None,
    n_corruptions: int = None,
    max_vram_gb: float = None,
    min_batch_size: int = 64,
    prefer_power_of_two: bool = False,
) -> int:
    """
    Compute optimal batch size for evaluation.
    
    Uses adaptive batch size: smaller for small evaluations, larger for large ones.
    This optimizes compilation time for small tests while supporting large-scale ranking.
    
    Memory model:
    - Each query in a batch requires ~3-4MB for state tensors:
      - derived_states: [K_max=120, M_max=26, 3] × 8 bytes ≈ 75KB per query
      - Multiple intermediate tensors during unification: ~3MB per query
    - For 8GB GPU with ~6GB usable: max ~2000 queries safely, ~1500 conservatively
    
    CRITICAL: Batch sizes > 512 have pathologically worse per-query performance
    due to GPU memory bandwidth saturation with large derived_states tensors.
    
    Args:
        chunk_queries: Number of queries per chunk (used for sizing)
        n_corruptions: Number of corruptions per query (used for sizing)
        max_vram_gb: Maximum VRAM to use (default: detected from GPU)
        min_batch_size: Minimum allowed batch size
        prefer_power_of_two: If True, round to nearest power of 2
        
    Returns:
        Batch size (adaptive based on actual needs)
    """
    # Detect available VRAM if not specified
    if max_vram_gb is None and torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        # Use ~60% of total memory to leave room for PyTorch overhead
        max_vram_gb = total_mem * 0.6
    elif max_vram_gb is None:
        max_vram_gb = 4.0  # Conservative default for CPU
    
    # Adaptive: use smaller batch for small evaluations
    # This speeds up compilation for small tests
    # NOTE: Batch sizes > 512 tend to have much worse per-query performance
    # due to GPU memory bandwidth limits with large derived_states tensors
    if chunk_queries is not None and n_corruptions is not None:
        actual_need = chunk_queries * (1 + n_corruptions)
        if actual_need <= 64:
            batch_size = 64
        elif actual_need <= 256:
            batch_size = 256
        else:
            # Cap at 512 for best throughput (larger sizes have worse per-query time)
            batch_size = 512
    else:
        # Default to moderate size if params not provided
        batch_size = 256
    
    # Clamp to VRAM limit - use more accurate memory estimation
    # Each query uses approximately 3MB during unification
    mem_per_query_mb = 3.0
    max_batch_from_vram = int(max_vram_gb * 1024 / mem_per_query_mb)
    batch_size = min(batch_size, max_batch_from_vram)
    
    # Apply minimum
    batch_size = max(batch_size, min_batch_size)
    
    # Align to multiples of 32 for GPU efficiency
    batch_size = ((batch_size + 31) // 32) * 32
    
    # Optional: round to power of 2
    if prefer_power_of_two:
        import math
        log2 = math.log2(batch_size)
        batch_size = 2 ** round(log2)
    
    return batch_size


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Compute fraction of variance that y_pred explains about y_true.
    Returns 1 - Var[y_true - y_pred] / Var[y_true]
    """
    var_y = torch.var(y_true)
    return 1.0 - torch.var(y_true - y_pred) / (var_y + 1e-8)


# ============================================================================
# PPO Implementation
# ============================================================================

class PPOOptimized:
    """
    Proximal Policy Optimization for EvalEnvOptimized.
    
    This implementation works with the functional/immutable state approach
    of EvalEnvOptimized rather than the TensorDict-based BatchedEnv.
    
    Key Features:
        - Fixed batch size for CUDA graph compatibility (reduce-overhead mode)
        - Integrated evaluation methods (evaluate_policy, evaluate_with_corruptions)
        - Single-step compilation for fast warmup
    
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
        fixed_batch_size: Fixed batch size for evaluation (auto-computed if None)
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
        fixed_batch_size: Optional[int] = None,
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
        
        # Fixed batch size for evaluation (enforced to avoid recompilation)
        # If not specified, will be set during compile() based on env.batch_size
        self._fixed_batch_size = fixed_batch_size
        
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
    
    @property
    def fixed_batch_size(self) -> int:
        """Get the fixed batch size for evaluation.
        
        If not explicitly set, returns env.batch_size as the default.
        This property ensures consistent batch sizes for CUDA graph compatibility.
        """
        if self._fixed_batch_size is not None:
            return self._fixed_batch_size
        return self.env.batch_size
    
    @fixed_batch_size.setter
    def fixed_batch_size(self, value: int):
        """Set the fixed batch size for evaluation.
        
        Once set, all evaluation calls must use this batch size to avoid
        recompilation with CUDA graphs (reduce-overhead mode).
        """
        self._fixed_batch_size = value
    
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

    @torch.inference_mode()
    def evaluate_policy(
        self,
        queries: Tensor,
        max_steps: int = None,
        deterministic: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Run policy evaluation over trajectories for a single batch.
        
        This is the core evaluation loop. It expects queries to already be
        at the correct batch size (fixed_batch_size). Use evaluate_with_corruptions()
        for the full evaluation pipeline with chunking and padding.
        
        Must call env.compile(policy) before using this method.
        
        Args:
            queries: [B, 3] Query triples (B should equal fixed_batch_size)
            max_steps: Maximum trajectory length (default: env.max_depth)
            deterministic: Use argmax for action selection
            
        Returns:
            log_probs: [B] Accumulated log probs per query
            success: [B] Whether proof succeeded
            lengths: [B] Trajectory lengths
            rewards: [B] Accumulated rewards
        """
        if self.env._policy_logits_fn is None:
            raise RuntimeError("Must call env.compile(policy) before evaluate_policy()")
        
        device = self.device
        max_steps = max_steps or self.env.max_depth
        
        # Initialize state
        state = self.env.init_state_from_queries(queries)
        B = state.current_states.shape[0]
        
        # Pre-allocate accumulators
        total_log_probs = torch.zeros(B, device=device)
        total_rewards = torch.zeros(B, device=device)
        
        # Create initial observation
        action_mask = self.env._positions_S < state.derived_counts.unsqueeze(1)
        obs = EvalObs(
            sub_index=state.current_states.unsqueeze(1),
            derived_sub_indices=state.derived_states,
            action_mask=action_mask,
        )
        
        # Empty query pool and pointers for eval mode (no resets)
        empty_pool = torch.empty((0, 3), dtype=torch.long, device=device)
        empty_ptrs = torch.zeros(B, dtype=torch.long, device=device)
        eval_done_mask = torch.zeros(B, dtype=torch.bool, device=device)
        
        # Python loop over transitions
        for step_idx in range(max_steps):
            # Early exit if all done
            if state.done.all():
                break
            
            state, obs, actions, step_log_probs, _values, rewards, dones, _, _ = self.env.step_with_policy(
                state, obs, empty_pool, empty_ptrs,
                deterministic=deterministic,
                eval_mode=True,
                eval_done_mask=eval_done_mask,
            )
            
            # Accumulate
            total_log_probs = total_log_probs + step_log_probs
            total_rewards = total_rewards + rewards
        
        return total_log_probs, state.success, state.depths, total_rewards
    
    def _pad_queries(self, queries: Tensor) -> Tuple[Tensor, int]:
        """Pad queries to fixed_batch_size. Returns (padded_queries, original_size)."""
        B = queries.shape[0]
        fixed_batch_size = self.fixed_batch_size
        
        if B >= fixed_batch_size:
            return queries[:fixed_batch_size], min(B, fixed_batch_size)
        
        padded = torch.zeros(fixed_batch_size, 3, dtype=queries.dtype, device=self.device)
        padded[:B] = queries
        # Fill padding with last query (valid but results ignored)
        padded[B:] = queries[-1]
        return padded, B
    
    @torch.inference_mode()
    def evaluate_with_corruptions(
        self,
        queries: Tensor,
        sampler: Any,
        *,
        n_corruptions: int = 50,
        corruption_modes: Sequence[str] = ("head", "tail"),
        chunk_queries: int = 50,
        verbose: bool = False,
        deterministic: bool = True,
        parity_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate policy on queries with corruptions for ranking metrics (MRR, Hits@K).
        
        For each query, generates corruptions and evaluates all candidates 
        (positive + corruptions) to compute ranking metrics.
        
        This method handles ALL chunking and padding:
        1. Chunks positive queries into batches of chunk_queries
        2. For each chunk, generates corruptions per corruption mode
        3. Flattens candidates and pads to fixed_batch_size
        4. Calls evaluate_policy with padded batches
        5. Computes ranks and aggregates metrics
        
        Args:
            queries: [N, 3] Tensor of test triples
            sampler: Sampler for generating corruptions
            n_corruptions: Number of corruptions per query
            corruption_modes: Tuple of modes ('head', 'tail')
            chunk_queries: Number of positive queries to process at once
            verbose: Print progress
            deterministic: Use deterministic action selection
            parity_mode: If True, use numpy RNG for tie-breaking (slower but matches
                         model_eval.py exactly). Default False uses fast torch RNG.
            
        Returns:
            Dictionary with MRR and Hits@K metrics
        """
        if not self.env._compiled and self.env._policy_logits_fn is None:
            raise RuntimeError("Must call env.compile(policy) before evaluate_with_corruptions()")
        
        device = self.device
        N = queries.shape[0]
        K = n_corruptions
        fixed_batch_size = self.fixed_batch_size
        
        # Accumulate ranks per mode
        per_mode_ranks: Dict[str, list] = {m: [] for m in corruption_modes}
        
        # Process positive queries in chunks
        for start in range(0, N, chunk_queries):
            end = min(start + chunk_queries, N)
            Q = end - start
            
            if verbose:
                print(f"Processing queries {start}-{end} / {N}")
            
            chunk_queries_tensor = queries[start:end]  # [Q, 3]
            
            # parity_mode: Match model_eval.py corruption generation order exactly
            # Non-parity: Fast path with single sampler.corrupt call per mode
            if parity_mode:
                # Generate corruptions based on sampler.default_mode, not corruption_modes
                # This ensures RNG parity since both paths consume RNG in the same order
                sampler_mode = getattr(sampler, 'default_mode', 'both')
                
                # Pre-generate corruptions in head-then-tail order like model_eval.py
                head_corruptions = None
                tail_corruptions = None
                
                if sampler_mode in ('head', 'both'):
                    head_corruptions = sampler.corrupt(
                        chunk_queries_tensor, num_negatives=K, mode='head', device=device
                    )
                if sampler_mode in ('tail', 'both'):
                    tail_corruptions = sampler.corrupt(
                        chunk_queries_tensor, num_negatives=K, mode='tail', device=device
                    )
            
            for mode in corruption_modes:
                if parity_mode:
                    # Select corruptions based on mode (matching model_eval.py line 475 logic)
                    # model_eval.py: corrs_list = head_corrs_list if mode == "head" else tail_corrs_list
                    if mode == 'head':
                        corruptions = head_corruptions if head_corruptions is not None else torch.zeros(Q, K, 3, dtype=torch.long, device=device)
                    else:
                        # For mode='tail' or mode='both', use tail_corruptions (matching model_eval.py fallback)
                        corruptions = tail_corruptions if tail_corruptions is not None else torch.zeros(Q, K, 3, dtype=torch.long, device=device)
                else:
                    # Fast path: generate corruptions directly with requested mode
                    corruptions = sampler.corrupt(
                        chunk_queries_tensor,
                        num_negatives=K,
                        mode=mode,
                        device=device
                    )
                
                # Handle variable corruption counts (some may be filtered)
                valid_mask = corruptions.sum(dim=-1) != 0  # [Q, K]
                
                # Create candidates: positive + corruptions -> [Q, 1+K, 3]
                candidates = torch.zeros(Q, 1 + K, 3, dtype=torch.long, device=device)
                candidates[:, 0, :] = chunk_queries_tensor
                candidates[:, 1:, :] = corruptions
                
                # Flatten for batch evaluation: [Q*(1+K), 3]
                flat_candidates = candidates.view(-1, 3)
                total_candidates = flat_candidates.shape[0]
                
                # Process candidates in chunks of fixed_batch_size with padding
                all_log_probs = []
                all_success = []
                
                for cand_start in range(0, total_candidates, fixed_batch_size):
                    cand_end = min(cand_start + fixed_batch_size, total_candidates)
                    batch_candidates = flat_candidates[cand_start:cand_end]
                    actual_size = batch_candidates.shape[0]
                    
                    # Pad to fixed_batch_size
                    padded_candidates, _ = self._pad_queries(batch_candidates)
                    
                    # Evaluate
                    log_probs, success, depths, rewards = self.evaluate_policy(
                        queries=padded_candidates,
                        max_steps=self.env.max_depth,
                        deterministic=deterministic,
                    )
                    
                    # Trim to actual size
                    all_log_probs.append(log_probs[:actual_size])
                    all_success.append(success[:actual_size])
                
                # Concatenate results
                log_probs = torch.cat(all_log_probs, dim=0)  # [Q*(1+K)]
                success = torch.cat(all_success, dim=0)      # [Q*(1+K)]
                
                # Reshape results: [Q, 1+K]
                log_probs = log_probs.view(Q, 1 + K)
                success = success.view(Q, 1 + K)
                
                # Apply success penalty - failed proofs get -100 penalty
                log_probs = log_probs.clone()
                log_probs[~success.bool()] -= 100.0
                
                # Ranking with random tie-breaking
                pos_score = log_probs[:, 0:1]  # [Q, 1]
                neg_scores = log_probs[:, 1:]  # [Q, K]
                
                # Random keys for tie-breaking
                if parity_mode:
                    # Numpy RNG with seed=0 for exact parity with model_eval.py (slow)
                    rnd = torch.as_tensor(np.random.RandomState(0).rand(Q, 1 + K), device=device, dtype=torch.float32)
                else:
                    # Fast torch RNG (no seeding needed, just for tie-breaking)
                    rnd = torch.rand(Q, 1 + K, device=device)
                
                # Count negatives that beat positive (better score, or tied with higher random key)
                better = (neg_scores > pos_score) & valid_mask
                tied_wins = (neg_scores == pos_score) & (rnd[:, 1:] > rnd[:, 0:1]) & valid_mask
                ranks = 1 + better.sum(dim=1) + tied_wins.sum(dim=1)
                per_mode_ranks[mode].append(ranks.float())
        
        # Aggregate results
        results: Dict[str, Any] = {
            "MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0,
            "per_mode": {}
        }
        
        for mode in corruption_modes:
            if per_mode_ranks[mode]:
                all_ranks = torch.cat(per_mode_ranks[mode])
                results["per_mode"][mode] = compute_metrics_from_ranks(all_ranks)
            else:
                results["per_mode"][mode] = compute_metrics_from_ranks(torch.tensor([], device=device))
        
        # Average across modes
        for mode in corruption_modes:
            for k, v in results["per_mode"][mode].items():
                results[k] += v
        
        n_modes = len(corruption_modes)
        for k in ["MRR", "Hits@1", "Hits@3", "Hits@10"]:
            results[k] /= n_modes if n_modes > 0 else 1.0
            
        results["_mrr"] = results["MRR"]
        
        if verbose:
            print(f"\nResults:")
            print(f"  MRR: {results['MRR']:.4f}")
            print(f"  Hits@1: {results['Hits@1']:.4f}")
            print(f"  Hits@10: {results['Hits@10']:.4f}")
        
        return results
