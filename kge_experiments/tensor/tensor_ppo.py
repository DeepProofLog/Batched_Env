"""
PPO (Proximal Policy Optimization) Algorithm.

This module implements the PPO reinforcement learning algorithm with:
    - GAE (Generalized Advantage Estimation) for advantage computation
    - Clipped surrogate objective for stable policy updates
    - Value function loss with optional clipping
    - Entropy bonus for exploration

Key Components:
    - collect_rollouts(): Collects environment transitions into a buffer
    - train(): Performs PPO policy updates with multiple epochs
    - learn(): Main training loop orchestrating rollout and update cycles
"""

import time
from typing import Dict, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
# Enable compiled autograd to handle autograd.grad / backward() in compiled regions
if hasattr(torch._dynamo.config, 'compiled_autograd'):
    torch._dynamo.config.compiled_autograd = True
    
from tensordict import TensorDict

from tensor.tensor_rollout import RolloutBuffer
from utils.trace_utils import TraceRecorder
from tensor.tensor_callbacks import Display
try:
    from debug_training import analyze_logits, analyze_values_returns, analyze_advantages, print_training_health_report
    DEBUG_TRAINING_AVAILABLE = True
except ImportError:
    DEBUG_TRAINING_AVAILABLE = False


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Compute fraction of variance that y_pred explains about y_true.
    
    Returns 1 - Var[y_true - y_pred] / Var[y_true]
    
    Interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    
    Args:
        y_pred: Predicted values (values from rollout buffer)
        y_true: True values (returns from rollout buffer)
        
    Returns:
        Explained variance as a tensor (0-d)
    """
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    
    var_y = y_true.var()
    if var_y == 0:
        return torch.tensor(float('nan'), device=y_true.device)
    return 1.0 - (y_true - y_pred).var() / var_y




class PPOLossModule(nn.Module):
    """
    Fused module that wraps the policy and loss computation.
    
    This allows torch.compile to see the Policy Forward + Loss Computation
    as a single graph, enabling:
    1. Kernel fusion across the boundary (e.g. policy output -> loss).
    2. Elimination of overhead from launching multiple separate compiled functions.
    3. Minimized device synchronization.
    """
    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy
        
    def forward(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        clip_range: float,
        clip_range_vf: Optional[float],
        ent_coef: float,
        vf_coef: float,
    ) -> torch.Tensor:
        # 1. Run Policy Forward (was evaluate_actions)
        values, log_probs, entropy = self.policy(obs, actions=actions)
        values = values.flatten()
        
        # 2. Compute Loss Components
        # Ratio between old and new policy
        log_ratio = log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        
        # Clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(
            ratio,
            1.0 - clip_range,
            1.0 + clip_range,
        )
        # Use minimum for element-wise min
        policy_loss = -torch.minimum(policy_loss_1, policy_loss_2).mean()
        
        # Clip fraction
        clip_fraction_t = torch.mean((torch.abs(ratio - 1) > clip_range).float())
        
        # Value loss
        if clip_range_vf is None:
            values_pred = values
        else:
            values_pred = old_values + torch.clamp(
                values - old_values,
                -clip_range_vf,
                clip_range_vf,
            )
        
        # MSE Loss
        value_loss = F.mse_loss(returns, values_pred)
        
        # Entropy loss
        if entropy is None:
            # Fallback if policy doesn't return entropy (shouldn't happen with ActorCriticPolicy)
            entropy_loss = -torch.mean(-log_probs)
        else:
            entropy_loss = -torch.mean(entropy)
            
        # Total loss
        loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss
        
        # Approx KL
        ratio_minus_one = ratio - 1.0
        approx_kl_div_t = torch.mean(ratio_minus_one - log_ratio)
        
        # Pack attributes into a single tensor
        metrics_packed = torch.stack([
            loss,
            policy_loss,
            value_loss,
            entropy_loss,
            approx_kl_div_t,
            clip_fraction_t
        ])
        
        return metrics_packed


class PPO:
    """
    Proximal Policy Optimization (PPO) algorithm.
    
    PPO is an on-policy gradient method that alternates between sampling data
    through interaction with the environment and optimizing a "surrogate" objective
    function using stochastic gradient ascent.
    
    Key Features:
    - **Clipped Surrogate Objective**: Constrains the policy update step size to
      prevent destructive large updates.
    - **Generalized Advantage Estimation (GAE)**: Used for calculating advantages,
      balancing bias and variance.
    - **Value Function Clipping**: Optional clipping of the value function updates
      to match the policy trust region.
    - **Entropy Regularization**: Encourages exploration by adding an entropy bonus
      to the loss.
    
    Attributes:
        policy (nn.Module): Actor-Critic network.
        env (EnvBase): Batched TorchRL environment.
        n_steps (int): Steps per environment per rollout.
        learning_rate (float): Adam optimizer learning rate.
        n_epochs (int): Number of optimization epochs per rollout.
        batch_size (int): Minibatch size for gradient updates.
        gamma (float): Discount factor for future rewards.
        gae_lambda (float): GAE smoothing parameter.
        clip_range (float): Hyperparameter epsilon for clipping policy updates.
        clip_range_vf (Optional[float]): Hyperparameter for clipping value updates.
        normalize_advantage (bool): Whether to normalize advantages per minibatch.
        ent_coef (float): Entropy coefficient.
        vf_coef (float): Value function coefficient.
        max_grad_norm (float): Gradient clipping norm.
        target_kl (Optional[float]): KL divergence limit for early stopping.
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
        target_kl: Optional[float] = None,
        device: torch.device = None,
        verbose: bool = True,
        trace_dir: Optional[str] = None,
        trace_prefix: str = "batched",
        trace_recorder: Optional[TraceRecorder] = None,
        seed: Optional[int] = None,
        use_amp: bool = False,  # Enable AMP for mixed precision training
        parity: bool = False,   # Enable parity mode (e.g. numpy RNG for rollouts)
        total_timesteps: Optional[int] = None,  # Total training timesteps for schedule computation
        use_compile: bool = True,  # Enable torch.compile for policy optimization
        debug_ppo: bool = False,  # Enable detailed training diagnostics
    ):
        """
        Initialize the PPO algorithm.
        
        Args:
            policy (nn.Module): Actor-critic policy network.
            env: Environment instance (used to infer n_envs and device).
            n_steps (int): Number of steps to run for each environment per update.
            learning_rate (float): Learning rate.
            n_epochs (int): Number of epochs when optimizing the surrogate loss.
            batch_size (int): Minibatch size.
            gamma (float): Discount factor.
            gae_lambda (float): Factor for trade-off of bias vs variance for GAE.
            clip_range (float): Clipping parameter epsilon.
            clip_range_vf (Optional[float]): Clipping parameter for value function.
            normalize_advantage (bool): Whether to normalize or not the advantage.
            ent_coef (float): Entropy coefficient for the loss calculation.
            vf_coef (float): Value function coefficient for the loss calculation.
            max_grad_norm (float): The maximum value for the gradient clipping.
            target_kl (Optional[float]): Limit the KL divergence between updates.
            device (torch.device): Device to run on.
            verbose (bool): Verbosity flag.
            trace_dir (Optional[str]): Directory to save traces.
            trace_prefix (str): Prefix for trace files.
            trace_recorder (Optional[TraceRecorder]): Existing recorder instance.
            seed (Optional[int]): Random seed for RNG synchronization between rollouts.
            parity (bool): If True, use older/slower methods (e.g. numpy RNG) to match SB3 exactly.
            total_timesteps (Optional[int]): Total training timesteps for schedule computation.
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
        self.ent_coef_initial = ent_coef  # Store initial value
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.device = device if device is not None else torch.device('cpu')
        self.verbose = verbose
        self.seed = seed  # For RNG synchronization in parity testing
        self.trace_recorder = trace_recorder or (TraceRecorder(trace_dir, prefix=trace_prefix) if trace_dir else None)
        self._trace_episode_ids = None
        self._trace_lengths = None
        self.env_device = getattr(env, "_device", None) or getattr(env, "device", None) or torch.device("cpu")
        if not isinstance(self.env_device, torch.device):
            self.env_device = torch.device(self.env_device)
        # Optimization: skip .to() calls when devices match
        # Normalize devices to ensure "cuda" == "cuda:0" if index is 0
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        if self.env_device.type == 'cuda' and self.env_device.index is None:
            self.env_device = torch.device(f"cuda:{torch.cuda.current_device()}")
            
        self._same_device = (self.device == self.env_device) or \
                            (self.device.type == self.env_device.type and self.device.index == self.env_device.index)
        
        
        # Schedule configuration
        self.total_timesteps = total_timesteps or 100000  # Default if not specified
        
        # AMP (Automatic Mixed Precision) - configure via use_amp parameter
        self.use_amp = use_amp and (self.device.type == "cuda")
        if self.use_amp:
            # If BF16 is supported, we don't need GradScaler
            if torch.cuda.is_bf16_supported():
                self.scaler = None
            else:
                self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None        
        
        
        # Get number of environments
        self.n_envs = int(env.batch_size[0]) if isinstance(env.batch_size, torch.Size) else int(env.batch_size)
        
        # Create rollout buffer
        self.rollout_buffer = RolloutBuffer(
            buffer_size=n_steps,
            n_envs=self.n_envs,
            device=self.device,
            gamma=gamma,
            gae_lambda=gae_lambda,
            parity=parity,
        )
        
        # Persistent state for consecutive learn() calls
        self._last_obs = None
        self._last_episode_starts = None
        self._current_episode_reward = None
        self._current_episode_length = None
        self.num_timesteps = 0
        
        # Initialize optimizer
        # NOTE: eps=1e-5 matches SB3's default (not PyTorch's 1e-8) - critical for parity!
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate,
            eps=1e-5,
            fused=True  # Use fused kernel for performance
        )
        
        # Compile components (if enabled)
        self.use_compile = use_compile
        self.debug_ppo = debug_ppo  # Enable detailed diagnostics
        if use_compile:
            self._compile_components()
        else:
            print("[PPO] torch.compile disabled")
            self._setup_loss_module()

    def _compute_loss_components(
        self,
        advantages: torch.Tensor,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
        old_values: torch.Tensor,
        entropy: Optional[torch.Tensor],
        clip_range: float,
        clip_range_vf: Optional[float],
        ent_coef: float,
        vf_coef: float,
    ) -> tuple:
        """
        Compute PPO loss components in a compiled function to avoid eager op overhead.
        """
        # Ratio between old and new policy
        log_ratio = log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        
        # Clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(
            ratio,
            1.0 - clip_range,
            1.0 + clip_range,
        )
        # Use minimum for element-wise min (clearer intent and potentially better for compiler)
        policy_loss = -torch.minimum(policy_loss_1, policy_loss_2).mean()
        
        # Clip fraction
        clip_fraction_t = torch.mean((torch.abs(ratio - 1) > clip_range).float())
        
        # Value loss
        if clip_range_vf is None:
            values_pred = values
        else:
            values_pred = old_values + torch.clamp(
                values - old_values,
                -clip_range_vf,
                clip_range_vf,
            )
        
        # MSE Loss
        value_loss = F.mse_loss(returns, values_pred)
        
        # Entropy loss
        if entropy is None:
            entropy_loss = -torch.mean(-log_probs)
        else:
            entropy_loss = -torch.mean(entropy)
            
        # Total loss
        loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss
        
        # Approx KL
        # (exp(log_ratio) - 1) - log_ratio
        ratio_minus_one = ratio - 1.0
        approx_kl_div_t = torch.mean(ratio_minus_one - log_ratio)
        
        # Pack attributes into a single tensor for reduced graph overhead
        metrics_packed = torch.stack([
            loss,
            policy_loss,
            value_loss,
            entropy_loss,
            approx_kl_div_t,
            clip_fraction_t
        ])
        
        return metrics_packed


    # Define wrapper for compilation
    # Define wrapper for compilation
    def _compile_components(self):
        # Create the fused module
        self.loss_module = PPOLossModule(self.policy)
        
        # Compile the ENTIRE fused module
        # This creates a single graph for (Policy Forward + Loss), enabling global optimization
        self.loss_module = torch.compile(self.loss_module, mode='reduce-overhead', fullgraph=True)
        
        # Also compile policy separately for collect_rollouts (inference only)
        # We need this because collect_rollouts calls policy() directly
        self.policy = torch.compile(self.policy, mode='reduce-overhead', fullgraph=True)

    def _setup_loss_module(self):
        """Setup loss module without compilation (fallback)"""
        self.loss_module = PPOLossModule(self.policy)
    
    def collect_rollouts(
        self,
        current_obs,
        episode_starts: torch.Tensor,
        current_episode_reward: torch.Tensor,
        current_episode_length: torch.Tensor,
        episode_rewards: list,
        episode_lengths: list,
        iteration: int,
        return_traces: bool = False,
        on_step_callback: Optional[Callable] = None,
    ) -> tuple:
        """
        Collect experiences using the current policy and fill the rollout buffer.
        
        This method steps the environment `n_steps` times. It stores observations,
        actions, rewards, done flags, probabilities, and values in the buffer.
        It handles environment resets and tracking episode statistics.
        
        Args:
            current_obs (Any): Last observation from previous rollout (or reset).
            episode_starts (Tensor): [N_envs] Binary mask for episode starts.
            current_episode_reward (Tensor): [N_envs] Accumulator for rewards.
            current_episode_length (Tensor): [N_envs] Accumulator for lengths.
            episode_rewards (List): Global list to Append completed episode rewards.
            episode_lengths (List): Global list to Append completed episode lengths.
            iteration (int): Current global iteration number.
            return_traces (bool): If True, collect and return step-by-step traces.
            
        Returns:
            Tuple[Any, Tensor, Tensor, Tensor, int, Optional[List]]: 
                - next_obs: Latest observation.
                - episode_starts: Updated start masks.
                - current_episode_reward: Updated reward accumulators.
                - current_episode_length: Updated length accumulators.
                - total_steps: Total steps collected (n_steps * n_envs).
                - traces: Optional list of trace dictionaries (if return_traces=True).
        """
        from tensordict import TensorDict
        
        self.policy.eval()
        self.rollout_buffer.reset()
        # self.metrics_collector.reset()
        
        traces = [] if return_traces else None
        n_collected = 0
        dones = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        
        with torch.no_grad():
            # Keys that rollout buffer stores and model uses
            _obs_keys = ("sub_index", "derived_sub_indices", "action_mask")
            
            while n_collected < self.n_steps:
                log_interval = max(1, self.n_steps // 5)
                if n_collected % log_interval == 0:
                    print(f"Collecting rollouts: {n_collected}/{self.n_steps} steps")
                # Shallow clone - only clone tensors needed by buffer/model
                # This is faster than full TensorDict.clone()
                obs_snapshot = TensorDict(
                    {k: current_obs[k].clone() for k in _obs_keys if k in current_obs.keys()},
                    batch_size=current_obs.batch_size,
                    device=current_obs.device
                )
                # Skip .to() if devices already match
                obs_device = obs_snapshot if self._same_device else obs_snapshot.to(self.device)

                # Get action from policy - always sample (not deterministic)
                # With same model weights and RNG state, sampling produces identical results
                # Use bfloat16 for AMP if available to avoid GradScaler
                amp_dtype = torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported() else torch.float16
                with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=amp_dtype):
                    actions, values, log_probs = self.policy(obs_device, deterministic=False)
                
                dist = getattr(self.policy.action_dist, "distribution", None)
                if dist is not None and hasattr(dist, "logits"):
                    dist_logits = dist.logits.detach().clone()
            
                # Step environment - skip .to() if devices match
                actions_env = actions if self._same_device else actions.to(self.env_device)
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

                # Skip .to() if devices match
                rewards = rewards_env if self._same_device else rewards_env.to(self.device)
                dones = dones_env if self._same_device else dones_env.to(self.device)
                
                # Store transition
                self.rollout_buffer.add(
                    obs=obs_device,
                    action=actions,
                    reward=rewards,
                    episode_start=episode_starts,
                    value=values,
                    log_prob=log_probs
                )
                
                # Collect traces if requested
                if return_traces:
                    for idx in range(self.n_envs):
                        sub_index = obs_snapshot.get("sub_index")[idx]
                        derived_sub_indices = obs_snapshot.get("derived_sub_indices")[idx]
                        action_mask = obs_snapshot.get("action_mask")[idx]
                        trace_entry = {
                            "step": n_collected,
                            "env": idx,
                            "state_obs": {
                                "sub_index": sub_index.cpu().numpy().copy() if hasattr(sub_index, 'cpu') else sub_index,
                                "derived_sub_indices": derived_sub_indices.cpu().numpy().copy() if hasattr(derived_sub_indices, 'cpu') else derived_sub_indices,
                                "action_mask": action_mask.cpu().numpy().copy() if hasattr(action_mask, 'cpu') else action_mask,
                            },
                            "action": int(actions[idx]),
                            "reward": float(rewards[idx]),
                            "done": bool(dones[idx]),
                            "value": float(values[idx]),
                            "log_prob": float(log_probs[idx]),
                        }
                        traces.append(trace_entry)
                
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
                
                # Check for episode ends
                if dones.any():
                    done_indices = torch.where(dones)[0]
                    # Batch extract values to avoid per-element .item() calls
                    ep_rewards_batch = current_episode_reward[done_indices].tolist()
                    ep_lengths_batch = current_episode_length[done_indices].tolist()
                    
                    # Pre-extract optional values if available
                    # We use .tolist() once per batch to avoid N .item() calls
                    extra_keys = []
                    extra_batches = {}
                    
                    # Helper to safely batch-extract
                    def extract_batch(key, tensor_val):
                        if tensor_val is not None and tensor_val.shape[0] >= self.n_envs:
                            # Extract and flatten to 1D in a compile-friendly way
                            # Using view(-1) instead of while loop for static reshape
                            extracted = tensor_val[done_indices].view(-1)
                            return extracted.tolist() 
                        elif tensor_val is not None:
                            # It might be a scalar or weird shape - fallback to list of None (safe)
                            return None
                        return None

                    if step_info.get("label") is not None:
                        extra_keys.append("label")
                        extra_batches["label"] = extract_batch("label", step_info.get("label"))
                    if step_info.get("query_depth") is not None:
                        extra_keys.append("query_depth")
                        extra_batches["query_depth"] = extract_batch("query_depth", step_info.get("query_depth"))
                    if step_info.get("is_success") is not None:
                        extra_keys.append("is_success")
                        extra_batches["is_success"] = extract_batch("is_success", step_info.get("is_success"))

                    for i, idx in enumerate(done_indices):
                        ep_reward = ep_rewards_batch[i]
                        ep_length = int(ep_lengths_batch[i])
                        episode_rewards.append(ep_reward)
                        episode_lengths.append(ep_length)

                        # Collect episode info for rollout metrics
                        info_dict = {
                            "episode": {"r": ep_reward, "l": ep_length},
                        }
                        
                        # Apply extras
                        for key in extra_keys:
                            batch = extra_batches[key]
                            if batch is not None:
                                val = batch[i]
                                # Convert ints/bools safely
                                if key == "is_success":
                                    info_dict[key] = bool(val)
                                else:
                                    info_dict[key] = int(val)
                                    
                        infos_list = [info_dict]
                        # self.metrics_collector.accumulate(infos_list)
                        if on_step_callback:
                            on_step_callback(infos_list)
                        
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
            last_obs = current_obs if self._same_device else current_obs.to(self.device)
            last_values = self.policy.predict_values(last_obs)
        
        # Compute advantages and returns
        if dones.dim() > 1:
            dones = dones.squeeze(-1)
        self.rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)
        
        if return_traces:
            return (
                current_obs,
                episode_starts,
                current_episode_reward,
                current_episode_length,
                n_collected * self.n_envs,
                traces,
            )
        return (
            current_obs,
            episode_starts,
            current_episode_reward,
            current_episode_length,
            n_collected * self.n_envs,
        )

    def train(self, return_traces: bool = False) -> Dict[str, float]:
        """
        Update policy using the currently collected rollout buffer.
        
        Performs `n_epochs` of PPO updates:
        1. Iterates over minibatches of data.
        2. Computes the PPO clipped loss, value loss, and entropy loss.
        3. Takes an optimizer step.
        4. Monitors KL divergence for early stopping.
        
        Args:
            return_traces (bool): If True, return detailed per-batch training traces.
        
        Returns:
            Dict[str, float]: Dictionary containing average training metrics:
                - "policy_loss": Average clipped policy loss.
                - "value_loss": Average value function loss.
                - "entropy": Average entropy loss.
                - "clip_fraction": Average fraction of clipped updates.
                - "approx_kl": Average approximate KL divergence.
                - "traces": Optional list of per-batch details (if return_traces=True).
        """
        # Set policy to training mode
        self.policy.train()
        
        # Accumulators for logging (Tensors)
        # NOTE: pg_losses, value_losses, entropy_losses, clip_fractions accumulate across ALL epochs
        # But approx_kl_divs should only contain values from the LAST epoch (matching SB3)
        pg_losses_t = []
        value_losses_t = []
        entropy_losses_t = []
        clip_fractions_t = []
        
        # Training traces for detailed comparison
        train_traces = [] if return_traces else None
        
        if self.verbose:
            print(f"[PPO] Training for {self.n_epochs} epochs...")
        
        # --------------------
        # Epoch loop
        # --------------------
        continue_training = True
        for epoch in range(self.n_epochs):
            # Reset KL divergence list for each epoch (matching SB3 behavior)
            # SB3 declares approx_kl_divs inside the epoch loop, so only last epoch is reported
            approx_kl_divs_t = []
            # Minibatch loop
            for batch_data in self.rollout_buffer.get(batch_size=self.batch_size):
                obs, actions, old_values, old_log_probs, advantages, returns = batch_data
                
                # Flatten actions if needed
                actions = actions.squeeze(-1) if actions.dim() > 1 else actions
                # Normalize advantages per minibatch
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                
                # --------------------
                # --------------------
                # Training Step
                # --------------------
                # Ensure optimizer state is initialized for all params (needed for functional access)
                # We can just check if state is empty for the first param
                is_initialized = len(self.optimizer.state) > 0
                if not is_initialized:
                   # Fast manual init:
                   for group in self.optimizer.param_groups:
                        for p in group['params']:
                            if p not in self.optimizer.state:
                                self.optimizer.state[p] = {
                                    'step': torch.zeros(1, device=p.device), # Scalar tensor
                                    'exp_avg': torch.zeros_like(p),
                                    'exp_avg_sq': torch.zeros_like(p)
                                }

                # Compute forward pass and losses (compiled)
                # IMPORTANT: Must use same amp_dtype as rollout collection to avoid KL divergence
                amp_dtype = torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported() else torch.float16
                with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=amp_dtype):
                    # Fused execution: Policy Forward + Loss Computation in one graph
                    metrics_packed = self.loss_module(
                        obs, 
                        actions,
                        advantages, 
                        returns, 
                        old_log_probs, 
                        old_values,
                        self.clip_range, 
                        self.clip_range_vf, 
                        self.ent_coef, 
                        self.vf_coef
                    )
                    
                    # Unpack for usage (slicing is cheap)
                    loss = metrics_packed[0]
                    policy_loss = metrics_packed[1]
                    value_loss = metrics_packed[2]
                    entropy_loss = metrics_packed[3]
                    approx_kl_div_t = metrics_packed[4]
                    clip_fraction_t = metrics_packed[5]

                # Log approx KL for tracking
                # Clone to safe memory because graph output memory is recycled
                with torch.no_grad():
                     approx_kl_divs_t.append(approx_kl_div_t.detach().clone())
                
                # Log losses (Keep as tensors)
                # Must clone because these are graph outputs that will be overwritten
                pg_losses_t.append(policy_loss.detach().clone())
                value_losses_t.append(value_loss.detach().clone())
                entropy_losses_t.append(entropy_loss.detach().clone())
                clip_fractions_t.append(clip_fraction_t.detach().clone())
                
                # Collect training traces if requested
                if return_traces:
                    # Note: We need to re-compute simple stats if we want them exact for trace
                    # But for speed we just log the loss headers
                    train_traces.append({
                        "epoch": epoch,
                        "batch_size": len(actions),
                        "policy_loss": policy_loss.item(),
                        "value_loss": value_loss.item(),
                        "entropy_loss": entropy_loss.item(),
                        "clip_fraction": clip_fraction_t.item(),
                         # ratios etc are internal now, can't log without extracting
                    })
                
                # Check KL divergence for early stopping BEFORE optimizer step (matching SB3)
                # This prevents applying the update when KL exceeds threshold
                if self.target_kl is not None:
                    # Note: accessing .item() causes a synchronous device-to-host transfer. 
                    # If performance is critical, set target_kl=None to avoid this check.
                    approx_kl_div = approx_kl_div_t.item()
                    if approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break
                
                # Apply optimizer step only if KL check passed
                # NOTE: backward() is NOT traceable by Dynamo - it must remain eager
                # Only forward pass and loss computation use fullgraph=True
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping (eager)
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                
                # Fused Adam step (runs optimized CUDA kernel)
                self.optimizer.step()
            
            
            if not continue_training:
                break
            
            # Print epoch stats (Sync ONCE per epoch)
            print(f"Epoch {epoch+1}/{self.n_epochs}. ")
            if self.verbose: # and epoch == self.n_epochs - 1:
                # Compute means on GPU then sync
                   mean_pg = torch.stack(pg_losses_t).mean().item() if pg_losses_t else 0.0
                   mean_val = torch.stack(value_losses_t).mean().item() if value_losses_t else 0.0
                   mean_ent = torch.stack(entropy_losses_t).mean().item() if entropy_losses_t else 0.0
                   mean_kl = torch.stack(approx_kl_divs_t).mean().item() if approx_kl_divs_t else 0.0
                   mean_clip = torch.stack(clip_fractions_t).mean().item() if clip_fractions_t else 0.0
                   
                   print(f"Losses: total {loss.item():.5f}, "
                        f"value {mean_val:.5f}, "
                        f"policy {mean_pg:.5f}, "
                        f"entropy {mean_ent:.5f}, "
                        f"approx_kl {mean_kl:.5f} "
                        f"clip_fraction {mean_clip:.5f}. ")
        
        # NOTE: Policy is left in training mode to avoid unnecessary mode switching
        # The policy will be set to eval mode at the start of collect_rollouts()
        
        # Return average metrics (Sync ONCE at end)
        with torch.no_grad():
            # Compute explained variance from rollout buffer
            # This is done OUTSIDE the compiled training loop for efficiency
            ev_tensor = explained_variance(self.rollout_buffer.values, self.rollout_buffer.returns)
            
            # Debug: log value and return statistics
            if self.verbose:
                values_flat = self.rollout_buffer.values.flatten()
                returns_flat = self.rollout_buffer.returns.flatten()
                print(f"[PPO] Values: min={values_flat.min().item():.3f}, max={values_flat.max().item():.3f}, mean={values_flat.mean().item():.3f}, std={values_flat.std().item():.3f}")
                print(f"[PPO] Returns: min={returns_flat.min().item():.3f}, max={returns_flat.max().item():.3f}, mean={returns_flat.mean().item():.3f}, std={returns_flat.std().item():.3f}")
                print(f"[PPO] Explained variance: {ev_tensor.item():.4f}")
            
            metrics = {
                "policy_loss": torch.stack(pg_losses_t).mean().item() if pg_losses_t else 0.0,
                "value_loss": torch.stack(value_losses_t).mean().item() if value_losses_t else 0.0,
                "entropy": -torch.stack(entropy_losses_t).mean().item() if entropy_losses_t else 0.0, # entropy_loss is already negative of entropy
                "clip_fraction": torch.stack(clip_fractions_t).mean().item() if clip_fractions_t else 0.0,
                "approx_kl": torch.stack(approx_kl_divs_t).mean().item() if approx_kl_divs_t else 0.0,
                "explained_var": ev_tensor.item(),
            }
            
            # Detailed training diagnostics when debug_ppo is enabled
            if getattr(self, 'debug_ppo', False) and DEBUG_TRAINING_AVAILABLE:
                # Analyze values and returns
                values_flat = self.rollout_buffer.values.flatten()
                returns_flat = self.rollout_buffer.returns.flatten()
                advantages_flat = self.rollout_buffer.advantages.flatten()
                
                value_stats = analyze_values_returns(values_flat, returns_flat)
                advantage_stats = analyze_advantages(advantages_flat)
                
                print_training_health_report(
                    logit_stats={  # Empty for now - would need to capture from forward pass
                        "logits_valid_min": 0.0,
                        "logits_valid_max": 0.0,
                        "logits_valid_mean": 0.0,
                        "logits_valid_std": 0.0,
                        "probs_valid_min": 0.0,
                        "probs_valid_max": 0.0,
                        "entropy_mean": metrics.get("entropy", 0.0),
                        "entropy_min": metrics.get("entropy", 0.0),
                        "entropy_max": metrics.get("entropy", 0.0),
                        "relative_entropy_mean": 0.5,
                        "num_valid_actions_mean": 5.0,
                    },
                    value_stats=value_stats,
                    advantage_stats=advantage_stats,
                    loss_stats=metrics,
                )
        
        if return_traces:
            metrics["traces"] = train_traces
        
        return metrics

    def learn(
        self,
        total_timesteps: int,
        callback=None,
        reset_num_timesteps: bool = True,
        on_iteration_start_callback=None,
        on_step_callback=None,
    ) -> None:
        """
        Execute the PPO main loop: alternate between collecting rollouts and training.
        
        Args:
            total_timesteps (int): The total number of environment steps to train for.
            callback (Optional[Callable]): Callback called at every step.
            reset_num_timesteps (bool): If True, reset the timestep counter.
                                        Set False to continue training.
            on_iteration_start_callback (Optional[Callable]): Callback called at the start of each iteration.
        """
        from tensordict import TensorDict
        
        # Handle reset_num_timesteps
        if reset_num_timesteps:
            self.num_timesteps = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        
        total_steps_done = self.num_timesteps
        iteration = 0
        
        # Only reset environment if needed (first call or explicit reset)
        if reset_num_timesteps or self._last_obs is None:
            current_obs = self.env.reset()
            
            current_episode_reward = torch.zeros(self.n_envs, device=self.device)
            current_episode_length = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
            episode_starts = torch.ones(self.n_envs, dtype=torch.float32, device=self.device)
        else:
            # Resume from last state
            current_obs = self._last_obs
            current_episode_reward = self._current_episode_reward
            current_episode_length = self._current_episode_length
            episode_starts = self._last_episode_starts
        
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
                        
            
            # ============================================================
            # Callbacks (Start of Iteration)
            # ============================================================
            if on_iteration_start_callback is not None:
                on_iteration_start_callback(iteration, total_steps_done)
            
            # ============================================================
            # Logging
            # ============================================================
            if self.verbose:
                print(f"\n[PPO] ===== Iteration {iteration} ({total_steps_done}/{total_timesteps} steps) =====")

            # ============================================================
            # Collect rollouts
            # ============================================================
            rollout_start_time = time.time()
            
            (
                current_obs,
                episode_starts,
                current_episode_reward,
                current_episode_length,
                steps_collected,
            ) = self.collect_rollouts(
                current_obs=current_obs,
                episode_starts=episode_starts,
                current_episode_reward=current_episode_reward,
                current_episode_length=current_episode_length,
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                iteration=iteration,
                on_step_callback=on_step_callback,
            )
            
            total_steps_done += steps_collected
            rollout_time = time.time() - rollout_start_time
            print(f"[PPO] Rollout collected in {rollout_time:.2f}s")
            print(f"[PPO] FPS: {steps_collected/rollout_time:.2f}")
            if self.verbose:
                if episode_rewards:
                    recent_rewards = episode_rewards[-10:]
                    print(f"[PPO] Recent episodes: reward={sum(recent_rewards)/len(recent_rewards):.3f}, length={sum(episode_lengths[-10:])/len(episode_lengths[-10:]):.1f}")

            # ============================================================
            # Train policy
            # ============================================================
            if self.verbose:
                print("\n[PPO] ===== Training policy =====")
            train_start_time = time.time()
            train_metrics = self.train()
            train_time = time.time() - train_start_time
            train_extra = {**train_metrics, "total_timesteps": total_steps_done, "iterations": iteration}
            Display.print_formatted_metrics(metrics={}, prefix="train", extra_metrics=train_extra)
            
            # Store last training metrics for external access
            self.last_train_metrics = train_metrics
            
            print(f"[PPO] Training completed in {train_time:.2f}s")
            
            # ============================================================
            # Callback
            # ============================================================
            if callback is not None:
                callback_result = callback(locals(), globals())
                if callback_result is False:
                    if self.verbose:
                        print("[PPO] Training stopped by callback")
                    break

        if self.trace_recorder is not None:
            self.trace_recorder.flush()
        
        # Save state for consecutive learn() calls
        self._last_obs = current_obs
        self._last_episode_starts = episode_starts
        self._current_episode_reward = current_episode_reward
        self._current_episode_length = current_episode_length
        self.num_timesteps = total_steps_done
