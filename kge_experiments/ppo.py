"""
PPO (Proximal Policy Optimization) for Optimized Environment.

This module implements PPO for use with Env_vec which uses
EnvObs/EnvState instead of TensorDict.

Key Differences from ppo.py:
    - Uses step_functional() instead of step_and_maybe_reset()
    - Works with EnvObs NamedTuples
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
import time

from rollout import RolloutBuffer
from env import EnvVec, EnvObs, EnvState


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



def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Compute fraction of variance that y_pred explains about y_true.
    Returns 1 - Var[y_true - y_pred] / Var[y_true]
    """
    var_y = torch.var(y_true)
    return 1.0 - torch.var(y_true - y_pred) / (var_y + 1e-8)


# ============================================================================
# Loss Module (for torch.compile)
# ============================================================================

class PPOLossModule(nn.Module):
    """
    Fused module that wraps policy forward + loss computation.
    
    This allows torch.compile to see the entire computation as a single graph,
    enabling kernel fusion and eliminating overhead from separate compiled functions.
    
    Uses raw tensors (not TensorDict) for stable memory addresses with CUDA graphs.
    """
    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy
        
    def forward(
        self,
        sub_index: torch.Tensor,          # [B, 1, A, 3]
        derived_sub_indices: torch.Tensor, # [B, S, A, 3]
        action_mask: torch.Tensor,         # [B, S]
        actions: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        clip_range: float,
        clip_range_vf: float,  # Must be a float, use 0.0 for no clipping
        ent_coef: float,
        vf_coef: float,
    ) -> torch.Tensor:
        # Use raw tensor method to avoid TensorDict (CUDA graph compatible)
        # 1. Policy Forward
        values, log_probs, entropy = self.policy.evaluate_actions(
            sub_index, derived_sub_indices, action_mask, actions
        )
        values = values.flatten()
        
        # 2. PPO Loss Computation
        log_ratio = log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        
        # Clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        policy_loss = -torch.minimum(policy_loss_1, policy_loss_2).mean()
        
        # Clip fraction
        clip_fraction_t = torch.mean((torch.abs(ratio - 1) > clip_range).float())
        
        # Value loss - handle clip_range_vf correctly:
        # When clip_range_vf > 0: apply clipping (clamp value difference)
        # When clip_range_vf == 0: no clipping (use values directly)
        # Use torch.where for CUDA graph compatibility (avoids control flow)
        value_diff = values - old_values
        clipped_diff = torch.clamp(value_diff, -clip_range_vf, clip_range_vf)
        # If clip_range_vf == 0, use raw values; otherwise use clipped values
        # Equivalent to: values if clip_range_vf == 0 else old_values + clipped_diff
        values_pred = torch.where(
            torch.tensor(clip_range_vf > 0, device=values.device),
            old_values + clipped_diff,
            values
        )
        value_loss = F.mse_loss(returns, values_pred)
        
        # Entropy loss - always use entropy path (policy should always return entropy)
        entropy_loss = -torch.mean(entropy)
            
        # Total loss
        loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss
        
        # Approx KL
        approx_kl_div_t = torch.mean((ratio - 1.0) - log_ratio)
        
        # Pack metrics into single tensor for efficient return
        metrics_packed = torch.stack([
            loss,
            policy_loss,
            value_loss,
            entropy_loss,
            approx_kl_div_t,
            clip_fraction_t
        ])
        
        return metrics_packed


# ============================================================================
# PPO Implementation
# ============================================================================

class PPO:
    """
    Proximal Policy Optimization for Env_vec.
    
    This implementation works with the functional/immutable state approach
    of Env_vec rather than the TensorDict-based BatchedEnv.
    
    Key Features:
        - Fixed batch size for CUDA graph compatibility (reduce-overhead mode)
        - Integrated evaluation methods (evaluate_policy, evaluate_with_corruptions)
        - Single-step compilation for fast warmup
        - eval_only mode for memory-efficient evaluation (skips 8-16GB buffer allocation)
    
    Args:
        policy: Actor-critic policy network
        env: Env_vec instance
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
        eval_only: If True, skip rollout buffer allocation for evaluation-only use.
            This saves ~8-16GB VRAM but disables training methods.
    """
    
    
    def __init__(
        self,
        policy: nn.Module,
        env: EnvVec,
        config,  # Accept config object (KGEConfig or TrainConfig)
        **kwargs  # Optional overrides
    ):
        """
        Initialize PPO with config object.
        
        Args:
            policy: Actor-critic policy network
            env: EnvVec environment
            config: Configuration object (e.g., KGEConfig from builder)
            **kwargs: Optional parameter overrides
        """
        self.config = config
        self.policy = policy
        self.env = env
        
        # Extract parameters from config with optional kwargs overrides
        self.batch_size_env = kwargs.get('batch_size_env', config.n_envs)
        self.padding_atoms = kwargs.get('padding_atoms', config.padding_atoms)
        self.padding_states = kwargs.get('padding_states', config.padding_states)
        # Handle both max_steps (TrainConfig) and max_depth (test configs)
        self.max_depth = kwargs.get('max_depth', getattr(config, 'max_steps', getattr(config, 'max_depth', 20)))
        
        # PPO Hyperparameters
        self.n_steps = kwargs.get('n_steps', config.n_steps)
        self.learning_rate = kwargs.get('learning_rate', config.learning_rate)
        self.n_epochs = kwargs.get('n_epochs', config.n_epochs)
        self.batch_size = kwargs.get('batch_size', config.batch_size)
        self.gamma = kwargs.get('gamma', config.gamma)
        self.gae_lambda = kwargs.get('gae_lambda', getattr(config, 'gae_lambda', 0.95))
        self.clip_range = kwargs.get('clip_range', config.clip_range)
        self.clip_range_vf = kwargs.get('clip_range_vf', getattr(config, 'clip_range_vf', None))
        self.normalize_advantage = kwargs.get('normalize_advantage', getattr(config, 'normalize_advantage', True))
        self.ent_coef = kwargs.get('ent_coef', config.ent_coef)
        self.vf_coef = kwargs.get('vf_coef', getattr(config, 'vf_coef', 0.5))
        self.max_grad_norm = kwargs.get('max_grad_norm', getattr(config, 'max_grad_norm', 0.5))
        self.target_kl = kwargs.get('target_kl', getattr(config, 'target_kl', None))
        # Handle both KGEConfig (with _components) and TrainConfig (without)
        components = getattr(config, '_components', {})
        self.device = kwargs.get('device', components.get('device', getattr(config, 'device', torch.device('cpu'))))
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        self.verbose = kwargs.get('verbose', config.verbose)
        self.seed = kwargs.get('seed', config.seed)
        self.parity = kwargs.get('parity', config.parity)
        self.eval_only = kwargs.get('eval_only', getattr(config, 'eval_only', False))
        self._compile_policy = kwargs.get('compile_policy', not config.parity)
        self._compile_mode = kwargs.get('compile_mode', 'reduce-overhead')
        self._fixed_batch_size = kwargs.get('fixed_batch_size', None)
        
        # Metrics info (CPU-only to avoid synchronization)
        query_labels = kwargs.get('query_labels', None)
        query_depths = kwargs.get('query_depths', None)
        self.query_labels = query_labels.detach().cpu() if query_labels is not None else None
        self.query_depths = query_depths.detach().cpu() if query_depths is not None else None
        self.current_query_indices = None
        
        # AMP (Automatic Mixed Precision) configuration
        use_amp = kwargs.get('use_amp', True)
        self.use_amp = use_amp and (self.device.type == "cuda")
        if self.use_amp:
            self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.amp_dtype = torch.float32
        
        # Validation
        buffer_size = self.n_steps * self.batch_size_env
        if not self.eval_only and self.batch_size > buffer_size:
            raise ValueError(f"batch_size ({self.batch_size}) must be <= buffer_size ({buffer_size})")
        
        if not self.eval_only:
            # Pre-allocate rollout buffer and training tensors
            self.rollout_buffer = RolloutBuffer(
                buffer_size=self.n_steps,
                n_envs=self.batch_size_env,
                device=self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                padding_atoms=self.padding_atoms,
                padding_states=self.padding_states,
                parity=self.parity,
                batch_size=self.batch_size,
            )
            
            A, S = self.padding_atoms, self.padding_states
            self._train_sub_index = torch.zeros((self.batch_size, 1, A, 3), dtype=torch.long, device=self.device)
            self._train_derived_sub_indices = torch.zeros((self.batch_size, S, A, 3), dtype=torch.long, device=self.device)
            self._train_action_mask = torch.zeros((self.batch_size, S), dtype=torch.bool, device=self.device)
            self._train_actions = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
            self._train_advantages = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
            self._train_returns = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
            self._train_old_log_probs = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
            self._train_old_values = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
            
            self.optimizer = torch.optim.Adam(
                self.policy.parameters(),
                lr=self.learning_rate,
                eps=1e-5,
                fused=(self.device.type == 'cuda' if hasattr(self.device, 'type') else False),
            )
            self._compile_policy_network()
        else:
            self.rollout_buffer = None
            self.optimizer = None
            self.loss_module = None
            self._uncompiled_policy = self.policy
            self._compiled_policy_fn = torch.compile(
                self._uncompiled_policy.get_logits, 
                mode=self._compile_mode, 
                fullgraph=True
            )

        self._last_state = None
        self._last_obs = None
        self.num_timesteps = 0
        
        # Build callbacks if enabled
        if not self.eval_only and getattr(config, 'use_callbacks', True):
            self.callback = self._build_callbacks()
        else:
            self.callback = None
    
    def _build_callbacks(self):
        """Build callbacks based on config flags."""
        from callbacks import (
            TorchRLCallbackManager, MetricsCallback, RankingCallback,
            CheckpointCallback, ScalarAnnealingCallback, AnnealingTarget
        )
        from pathlib import Path
        
        callbacks = []
        config = self.config
        
        # MetricsCallback - always included if use_metrics_callback
        if getattr(config, 'use_metrics_callback', True):
            callbacks.append(MetricsCallback(
                log_interval=1,
                verbose=self.verbose,
                collect_detailed=True
            ))
        
        # Get components from config
        components = getattr(config, '_components', {})
        sampler = components.get('sampler')
        dh = components.get('data_handler')
        
        # RankingCallback for evaluation
        if getattr(config, 'use_ranking_callback', True) and getattr(config, 'eval_freq', 0) > 0:
            if sampler and dh:
                valid_split = dh.get_materialized_split('valid')
                valid_queries = valid_split.queries.squeeze(1)
                valid_depths = valid_split.depths
                n_eval = getattr(config, 'n_eval_queries', None)
                if n_eval:
                    valid_queries = valid_queries[:n_eval]
                    valid_depths = valid_depths[:n_eval]
                
                n_corruptions = getattr(config, 'eval_neg_samples', getattr(config, 'n_corruptions', 50))
                scheme = getattr(config, 'corruption_scheme', ('head', 'tail'))
                
                callbacks.append(RankingCallback(
                    eval_env=self.env,
                    policy=self.policy,
                    sampler=sampler,
                    eval_data=valid_queries,
                    eval_data_depths=valid_depths,
                    eval_freq=int(config.eval_freq),
                    n_corruptions=n_corruptions,
                    corruption_scheme=tuple(scheme),
                    ppo_agent=self
                ))
        
        # CheckpointCallback for model saving
        if getattr(config, 'use_checkpoint_callback', True) and getattr(config, 'save_model', False):
            models_path = getattr(config, 'models_path', './models/')
            run_sig = getattr(config, 'run_signature', getattr(config, 'dataset', 'run'))
            save_path = Path(models_path) / run_sig
            best_metric = getattr(config, 'eval_best_metric', 'mrr_mean')
            if best_metric == 'mrr':
                best_metric = 'mrr_mean'
            
            callbacks.append(CheckpointCallback(
                save_path=save_path,
                policy=self.policy,
                train_metric="ep_rew_mean",
                eval_metric=best_metric,
                verbose=True,
                date=None
            ))
        
        # ScalarAnnealingCallback for lr/entropy decay
        if getattr(config, 'use_annealing_callback', True):
            annealing_targets = []
            total_timesteps = getattr(config, 'total_timesteps', 0)
            
            if getattr(config, 'lr_decay', False):
                lr_init = getattr(config, 'lr_init_value', self.learning_rate)
                lr_final = getattr(config, 'lr_final_value', 1e-6)
                def _set_lr(v):
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = float(v)
                    self.learning_rate = float(v)
                annealing_targets.append(AnnealingTarget(
                    name='lr', setter=_set_lr, initial=float(lr_init), final=float(lr_final),
                    start_point=float(getattr(config, 'lr_start', 0.0)),
                    end_point=float(getattr(config, 'lr_end', 1.0)),
                    transform=getattr(config, 'lr_transform', 'linear'),
                    value_type='float',
                ))
            
            if getattr(config, 'ent_coef_decay', False):
                ent_init = getattr(config, 'ent_coef_init_value', self.ent_coef)
                ent_final = getattr(config, 'ent_coef_final_value', 0.01)
                def _set_ent(v):
                    self.ent_coef = float(v)
                annealing_targets.append(AnnealingTarget(
                    name='ent_coef', setter=_set_ent, initial=float(ent_init), final=float(ent_final),
                    start_point=float(getattr(config, 'ent_coef_start', 0.0)),
                    end_point=float(getattr(config, 'ent_coef_end', 1.0)),
                    transform=getattr(config, 'ent_coef_transform', 'linear'),
                    value_type='float',
                ))
            
            if annealing_targets:
                callbacks.append(ScalarAnnealingCallback(
                    total_timesteps=total_timesteps,
                    targets=annealing_targets,
                    verbose=1
                ))
        
        return TorchRLCallbackManager(callbacks=callbacks) if callbacks else None
    
    def _compile_policy_network(self):
        """Compile the policy network for faster training.
        
        Creates a PPOLossModule that fuses policy forward + loss computation,
        then compiles the entire module using torch.compile.
        
        IMPORTANT: We use 'default' mode for the loss module instead of 'reduce-overhead'
        because TensorDict creation inside the forward pass causes CUDA graph instability.
        The environment still uses 'reduce-overhead' since it has stable tensor patterns.
        """

        if self.verbose:
            print("Compiling policy network and loss module for training...")
        
        # Store reference to uncompiled policy for loss_module
        self._uncompiled_policy = self.policy
        
        # Create fused loss module using UNCOMPILED policy
        # This will be compiled as a single unit
        self.loss_module = PPOLossModule(self._uncompiled_policy)
        
        # Use reduce-overhead mode for maximum performance
        # This works because we use evaluate_actions_raw which bypasses TensorDict
        self.loss_module = torch.compile(
            self.loss_module,
            mode=self._compile_mode,  # reduce-overhead now works with raw tensors
            fullgraph=True,
        )
        
        # Pre-warm gradients to ensure stable memory addresses for CUDA graphs
        # Without this, the first few backward passes allocate new gradient tensors
        self._warmup_gradients()
        
        # For inference (collect_rollouts), compile the policy logits function separately
        # This creates a different CUDA graph than the training graph
        self._compiled_policy_fn = torch.compile(
            self._uncompiled_policy.get_logits,
            mode=self._compile_mode,
            fullgraph=True,
        )
        
        # Also compile the policy for other uses (predict_values, etc)
        self.policy = torch.compile(
            self._uncompiled_policy,
            mode=self._compile_mode,
            fullgraph=True,
        )
    
    def _warmup_gradients(self):
        """Pre-allocate gradient tensors for stable CUDA graph addresses.
        
        Running a dummy forward+backward pass ensures gradient tensors are allocated
        before CUDA graph recording. Using set_to_none=False in zero_grad() keeps
        these addresses stable across iterations.
        """
        # Create dummy inputs matching the training batch shape
        batch_size = self.batch_size
        A = self.padding_atoms
        S = self.padding_states
        
        dummy_sub_index = torch.zeros((batch_size, 1, A, 3), dtype=torch.long, device=self.device)
        dummy_derived = torch.zeros((batch_size, S, A, 3), dtype=torch.long, device=self.device)
        dummy_mask = torch.ones((batch_size, S), dtype=torch.bool, device=self.device)
        dummy_actions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        dummy_advantages = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        dummy_returns = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        dummy_old_log_probs = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        dummy_old_values = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        
        # Run forward pass through UNCOMPILED loss module to allocate gradients
        # This ensures gradient tensors exist before any CUDA graph recording
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
            # Use the uncompiled policy directly
            values, log_probs, entropy = self._uncompiled_policy.evaluate_actions(
                dummy_sub_index, dummy_derived, dummy_mask, dummy_actions
            )
            # Compute simple loss
            dummy_loss = values.mean() + log_probs.mean() + entropy.mean()
        
        # Backward to allocate gradient tensors
        self.optimizer.zero_grad(set_to_none=False)
        dummy_loss.backward()
        
        # Zero gradients but keep tensors allocated
        self.optimizer.zero_grad(set_to_none=False)
    
    @property
    def fixed_batch_size(self) -> int:
        """Consistent batch size for evaluation (prevents CUDA graph recompilation)."""
        return self._fixed_batch_size if self._fixed_batch_size is not None else self.batch_size_env
    
    def _obs_to_tensordict(self, obs: EnvObs) -> TensorDict:
        """Convert EnvObs to TensorDict for policy forward pass."""
        return obs
    
    def _handle_done_episodes(
        self,
        done_indices: Tensor,
        current_episode_reward: Tensor,
        current_episode_length: Tensor,
        state: EnvState,
        next_query_indices: Optional[Tensor],
        episode_rewards: List[float],
        episode_lengths: List[int],
        on_step_callback: Optional[Callable],
    ) -> None:
        """Handle logging and callbacks for completed episodes (vectorized)."""
        num_dones = done_indices.numel()
        done_idx_cpu = done_indices.cpu().numpy()
        
        # Batch fetch stats (GPU -> CPU) and update logs
        batch_rs = current_episode_reward[done_indices].float().cpu().numpy()
        batch_ls = current_episode_length[done_indices].cpu().numpy().astype(int)
        episode_rewards.extend(batch_rs.tolist())
        episode_lengths.extend(batch_ls.tolist())
        
        # Handle callbacks
        if on_step_callback is not None:
            batch_succ = state['success'][done_indices].cpu().numpy().astype(bool) if "success" in state.keys() else np.zeros(num_dones, dtype=bool)
            
            # Fetch meta info if available
            batch_q_idxs = self.current_query_indices[done_idx_cpu] if self.current_query_indices is not None else None
            batch_lbls = self.query_labels[torch.as_tensor(batch_q_idxs, dtype=torch.long)].numpy() if batch_q_idxs is not None and self.query_labels is not None else None
            batch_depths = self.query_depths[torch.as_tensor(batch_q_idxs, dtype=torch.long)].numpy() if batch_q_idxs is not None and self.query_depths is not None else None
            
            # Construct infos using efficient zipping
            iterators = [
                batch_rs, batch_ls, batch_succ,
                batch_q_idxs if batch_q_idxs is not None else [None] * num_dones,
                batch_lbls if batch_lbls is not None else [None] * num_dones,
                batch_depths if batch_depths is not None else [None] * num_dones
            ]
            
            batch_infos = [
                {
                    "episode": {"r": float(r), "l": int(l)},
                    "is_success": bool(s),
                    **({ "episode_idx": int(q) } if q is not None else {}),
                    **({ "label": int(lbl) } if lbl is not None else {}),
                    **({ "query_depth": int(d) } if d is not None else {})
                }
                for r, l, s, q, lbl, d in zip(*iterators)
            ]
            on_step_callback(batch_infos)
            
        # Update pointers
        if self.current_query_indices is not None and next_query_indices is not None:
             self.current_query_indices[done_idx_cpu] = next_query_indices[done_indices].cpu().numpy()
    
    def collect_rollouts(
        self,
        current_state: EnvState,
        current_obs: EnvObs,
        episode_starts: torch.Tensor,
        current_episode_reward: torch.Tensor,
        current_episode_length: torch.Tensor,
        episode_rewards: list,
        episode_lengths: list,
        iteration: int,
        return_traces: bool = False,
        on_step_callback: Optional[Callable] = None,
    ) -> Tuple[EnvState, EnvObs, torch.Tensor, torch.Tensor, torch.Tensor, int, Optional[List]]:
        """Collect experiences using the current policy."""
        self.policy.eval()
        self.rollout_buffer.reset()
        
        traces = [] if return_traces else None
        n_collected = 0
        last_dones = torch.zeros(self.batch_size_env, dtype=torch.bool, device=self.device)
        
        state = current_state
        obs = current_obs

        # Initialize query indices from state if needed
        if self.current_query_indices is None and 'per_env_ptrs' in state.keys():
            self.current_query_indices = state['per_env_ptrs'].cpu().numpy()

        with torch.no_grad():
            while n_collected < self.n_steps:
                if self.verbose and n_collected % max(1, self.n_steps // 5) == 0:
                    print(f"Rollout: {n_collected}/{self.n_steps}")
                
                obs_snap = {k: v.clone() for k, v in obs.items()}
                values = self.policy.predict_values(obs)
                
                # Step environment
                torch.compiler.cudagraph_mark_step_begin()
                logits = self._compiled_policy_fn(obs).clone()
                
                masked_logits = torch.where(obs['action_mask'], logits, torch.full_like(logits, float('-inf')))
                probs = torch.softmax(masked_logits, dim=-1)
                actions = torch.multinomial(probs, 1).squeeze(-1)
                
                log_probs = torch.log_softmax(masked_logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
                log_probs = torch.where(~state['done'], log_probs, torch.zeros_like(log_probs))
                
                new_obs, new_state = self.env.step_and_reset(state, actions)
                
                self.rollout_buffer.add(
                    sub_index=obs_snap['sub_index'],
                    derived_sub_indices=obs_snap['derived_sub_indices'],
                    action_mask=obs_snap['action_mask'],
                    action=actions,
                    reward=new_state['step_rewards'],
                    episode_start=episode_starts,
                    value=values.flatten(),
                    log_prob=log_probs,
                )
                
                if return_traces:
                    for idx in range(self.batch_size_env):
                        traces.append({
                            "step": n_collected, "env": idx,
                            "query_idx": int(self.current_query_indices[idx]),
                            "state_obs": {
                                "sub_index": obs['sub_index'][idx].cpu().numpy().copy(),
                                "derived_sub_indices": obs['derived_sub_indices'][idx].cpu().numpy().copy(),
                                "action_mask": obs['action_mask'][idx].cpu().numpy().copy(),
                            },
                            "value": float(values[idx]),
                            "log_prob": float(log_probs[idx]),
                            "action_probs": probs[idx].cpu().numpy().copy(),
                            "action": int(actions[idx]), 
                            "reward": float(new_state['step_rewards'][idx]),
                            "done": bool(new_state['step_dones'][idx]),
                        })
                
                current_episode_reward += new_state['step_rewards']
                current_episode_length += 1
                n_collected += 1
                
                # Handle callbacks for done episodes
                done_indices = torch.nonzero(new_state['step_dones']).squeeze(-1)
                if done_indices.numel() > 0:
                    self._handle_done_episodes(
                        done_indices, current_episode_reward, current_episode_length,
                        state, new_state['per_env_ptrs'], episode_rewards, episode_lengths,
                        on_step_callback
                    )
                    current_episode_reward.masked_fill_(new_state['step_dones'], 0.0)
                    current_episode_length.masked_fill_(new_state['step_dones'], 0)
                
                episode_starts = new_state['step_dones'].float()
                last_dones = new_state['step_dones']
                state, obs = new_state, new_obs

            last_values = self.policy.predict_values(obs)
        
        self.rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=last_dones.float())
        
        return (state, obs, episode_starts, current_episode_reward, current_episode_length, 
                n_collected * self.batch_size_env, traces)

        
    def train(self, return_traces: bool = False) -> Dict[str, float]:
        """
        Update policy using the currently collected rollout buffer.
        
        Args:
            return_traces: If True, include detailed traces in return dict.
            timeout_seconds: Maximum time in seconds for training. If None, no timeout.
                             Useful for debugging CUDA graph hangs in reduce-overhead mode.
        
        Returns:
            Dict containing average training metrics
        """
        self.policy.train()
        train_start_time = time.time()
        
        # Total number of iterations (batches per epoch * num epochs)
        buffer_size = self.n_steps * self.batch_size_env
        n_batches = buffer_size // self.batch_size
        total_batches = self.n_epochs * n_batches
        
        # Accumulators - pre-allocate tensors to avoid list appending and stacking
        pg_losses_t = torch.zeros(total_batches, device=self.device)
        value_losses_t = torch.zeros(total_batches, device=self.device)
        entropy_losses_t = torch.zeros(total_batches, device=self.device)
        clip_fractions_t = torch.zeros(total_batches, device=self.device)
        approx_kl_divs_t = torch.zeros(total_batches, device=self.device)
        
        train_traces = [] if return_traces else None
        
        continue_training = True
        batch_count = 0
        for epoch in range(self.n_epochs):
            for batch_data in self.rollout_buffer.get(batch_size=self.batch_size):
                # Mark start of new training step for CUDA graph trees
                # This tells PyTorch it's okay to reuse memory from previous iterations
                torch.compiler.cudagraph_mark_step_begin()
                
                # Yields raw tensors: (sub_index, derived_sub_indices, action_mask, 
                #                       actions, old_values, old_log_probs, advantages, returns)
                (sub_index, derived_sub_indices, action_mask, 
                 actions, old_values, old_log_probs, advantages, returns) = batch_data
                
                # Copy data into pre-allocated tensors (preserves memory addresses for CUDA graphs)
                # Since we enforce batch_size divides buffer_size evenly, all batches have consistent size
                self._train_sub_index.copy_(sub_index)
                self._train_derived_sub_indices.copy_(derived_sub_indices)
                self._train_action_mask.copy_(action_mask)
                self._train_actions.copy_(actions)
                self._train_old_values.copy_(old_values)
                self._train_old_log_probs.copy_(old_log_probs)
                self._train_returns.copy_(returns)
                
                # Normalize advantages into pre-allocated tensor
                if self.normalize_advantage and len(advantages) > 1:
                    adv_mean = advantages.mean()
                    adv_std = advantages.std() + 1e-8
                    self._train_advantages.copy_((advantages - adv_mean) / adv_std)
                else:
                    self._train_advantages.copy_(advantages)
                
                # Fused execution: Policy Forward + Loss Computation in one graph
                # Pass raw tensors directly - TensorDict is built inside the compiled module
                # This avoids CUDA graph recompilation due to varying TensorDict memory layouts
                with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                    metrics_packed = self.loss_module(
                        self._train_sub_index,
                        self._train_derived_sub_indices,
                        self._train_action_mask,
                        self._train_actions,
                        self._train_advantages,
                        self._train_returns,
                        self._train_old_log_probs,
                        self._train_old_values,
                        self.clip_range,
                        self.clip_range_vf if self.clip_range_vf is not None else 0.0,  # Must be float, not None
                        self.ent_coef,
                        self.vf_coef,
                    )
                
                # Unpack metrics (slicing is cheap)
                loss = metrics_packed[0]
                policy_loss = metrics_packed[1]
                value_loss = metrics_packed[2]
                entropy_loss = metrics_packed[3]
                approx_kl_div = metrics_packed[4].detach()
                clip_fraction = metrics_packed[5]
                
                # Store losses - MUST clone() for CUDA graph compatibility!
                # detach() alone keeps the same memory address which gets overwritten
                pg_losses_t[batch_count] = policy_loss.detach().clone()
                value_losses_t[batch_count] = value_loss.detach().clone()
                entropy_losses_t[batch_count] = entropy_loss.detach().clone()
                clip_fractions_t[batch_count] = clip_fraction.detach().clone()
                approx_kl_divs_t[batch_count] = approx_kl_div.clone()
                
                # Collect traces (defer .item() to end if needed)
                if return_traces:
                    train_traces.append({
                        "epoch": epoch,
                        "batch_size": sub_index.shape[0],
                        "policy_loss_idx": batch_count,
                        "value_loss_idx": batch_count,
                    })
                
                batch_count += 1
                
                # KL divergence early stopping - check BEFORE optimizer step
                # This prevents applying the update when KL exceeds threshold
                if self.target_kl is not None:
                    kl_val = approx_kl_div.item()
                    if kl_val > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose:
                            print(f"Early stopping at step {epoch} due to reaching max kl: {kl_val:.2f}")
                        break
                
                # Optimizer step - use set_to_none=False to keep gradient addresses stable
                # This is critical for CUDA graph stability in reduce-overhead mode
                self.optimizer.zero_grad(set_to_none=False)
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.max_grad_norm,
                        foreach=True,  # Use fused CUDA ops for 2-3x speedup
                    )
                self.optimizer.step()
            
            if not continue_training:
                break
            
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.n_epochs}")
        
        # Compute metrics - single sync point for all accumulated tensors
        with torch.no_grad():
            ev = explained_variance(self.rollout_buffer.values, self.rollout_buffer.returns)
            
            # Use only filled entries (in case of early stopping)
            filled_pg = pg_losses_t[:batch_count]
            filled_vl = value_losses_t[:batch_count]
            filled_ent = entropy_losses_t[:batch_count]
            filled_clip = clip_fractions_t[:batch_count]
            filled_kl = approx_kl_divs_t[:batch_count]
            
            pg_loss_mean = filled_pg.mean() if batch_count > 0 else torch.tensor(0.0)
            value_loss_mean = filled_vl.mean() if batch_count > 0 else torch.tensor(0.0)
            entropy_loss_mean = filled_ent.mean() if batch_count > 0 else torch.tensor(0.0)
            clip_frac_mean = filled_clip.mean() if batch_count > 0 else torch.tensor(0.0)
            kl_mean = filled_kl.mean() if batch_count > 0 else torch.tensor(0.0)
        
        metrics = {
            "policy_loss": pg_loss_mean.item(),
            "value_loss": value_loss_mean.item(),
            "entropy": -entropy_loss_mean.item(),
            "clip_fraction": clip_frac_mean.item(),
            "approx_kl": kl_mean.item(),
            "explained_var": ev.item(),
        }
        
        # Convert trace indices to actual values if needed
        if return_traces and train_traces:
            pg_vals = pg_losses_t[:batch_count].cpu().numpy()
            vl_vals = value_losses_t[:batch_count].cpu().numpy()
            for trace in train_traces:
                trace["policy_loss"] = float(pg_vals[trace.pop("policy_loss_idx")])
                trace["value_loss"] = float(vl_vals[trace.pop("value_loss_idx")])
            metrics["traces"] = train_traces
        
        return metrics
    
    def learn(
        self,
        total_timesteps: int,
        queries: Optional[torch.Tensor] = None, # Deprecated, env handles queries
        reset_num_timesteps: bool = True,
        on_iteration_start_callback=None,
        on_step_callback=None,
        return_traces: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute the PPO main loop: alternate between collecting rollouts and training.
        
        Args:
            total_timesteps: Total number of environment steps to train for
            queries: [Deprecated] Query tensor - env handles this now
            reset_num_timesteps: If True, reset the timestep counter
            on_iteration_start_callback: Optional callback called at start of each iteration
            on_step_callback: Optional callback called when episodes complete
            return_traces: If True, return detailed traces for debugging
            
        Returns:
            Dict containing training info and optionally traces
        """
        if reset_num_timesteps:
            self.num_timesteps = 0
        else:
            total_timesteps += self.num_timesteps
        
        iteration = 0
        
        # Set environment to train mode
        self.env.train()
        
        # No explicit current_query_indices initialization here; handled in collect_rollouts
        
        # Reset environment to get initial state and observation
        # with torch.inference_mode():
        #     obs, state = self.env.reset()
        obs, state = self.env.reset()
        
        episode_starts = torch.ones(self.batch_size_env, dtype=torch.float32, device=self.device)
        current_episode_reward = torch.zeros(self.batch_size_env, dtype=torch.float32, device=self.device)
        current_episode_length = torch.zeros(self.batch_size_env, dtype=torch.long, device=self.device)
        episode_rewards = []
        episode_lengths = []
        
        # Trace collection
        all_rollout_traces = [] if return_traces else None
        all_train_traces = [] if return_traces else None
        
        while self.num_timesteps < total_timesteps:
            iteration += 1
            
            # Callback: Start of iteration
            if on_iteration_start_callback is not None:
                on_iteration_start_callback(iteration, self.num_timesteps)
            
            # Collect rollouts using env's internal query cycling
            print("\nCollecting rollouts")
            start_time = time.time()
            result = self.collect_rollouts(
                current_state=state,
                current_obs=obs,
                episode_starts=episode_starts,
                current_episode_reward=current_episode_reward,
                current_episode_length=current_episode_length,
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                iteration=iteration,
                on_step_callback=on_step_callback,
                return_traces=return_traces,
            )
            state, obs, episode_starts, current_episode_reward, current_episode_length, n_steps, rollout_info = result
            
            if return_traces and rollout_info is not None:
                all_rollout_traces.append({
                    'iteration': iteration,
                    'traces': rollout_info if isinstance(rollout_info, list) else rollout_info.get('traces', []),
                })
            
            self.num_timesteps += n_steps
            end_time = time.time()-start_time
            print(f"Rollout collected in {end_time:.2f}s. FPS: {n_steps / end_time:.2f}\n")            
            
            # Train
            print("\nTraining")
            start_time = time.time()
            train_metrics = self.train(return_traces=return_traces)
            
            if return_traces and 'traces' in train_metrics:
                all_train_traces.append({
                    'iteration': iteration,
                    'traces': train_metrics['traces'],
                })
            
            self.last_train_metrics = train_metrics
            print(f"Training completed in {time.time() - start_time:.2f}s")
            
            if self.verbose:
                print(f"Iteration {iteration}, timesteps: {self.num_timesteps}/{total_timesteps}.  "
                    f"total loss: {train_metrics['policy_loss'] + train_metrics['value_loss']:.4f}, "
                    f"policy_loss: {train_metrics['policy_loss']:.4f}, "
                    f"value_loss: {train_metrics['value_loss']:.4f}, "
                    f"entropy_loss: {train_metrics['entropy']:.4f}, "
                    f"approx_kl: {train_metrics['approx_kl']:.4f}, "
                    f"clip_fraction: {train_metrics['clip_fraction']:.4f}, "
                    f"explained_var: {train_metrics['explained_var']:.4f}\n")
            
            # Callback: End of iteration (matching SB3/PPO interface)
            if self.callback is not None:
                locals_dict = {
                    'iteration': iteration,
                    'total_steps_done': self.num_timesteps,
                    'episode_rewards': episode_rewards,
                    'episode_lengths': episode_lengths,
                    'train_metrics': train_metrics,
                }
                callback_result = self.callback(locals_dict, globals())
                if callback_result is False:
                    if self.verbose:
                        print("[PPOOptimized] Training stopped by callback")
                    break
        
        # Return results
        result_dict = {
            'num_timesteps': self.num_timesteps,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'last_train_metrics': self.last_train_metrics,
        }
        
        if return_traces:
            result_dict['rollout_traces'] = all_rollout_traces
            result_dict['train_traces'] = all_train_traces
        
        return result_dict

    @torch.no_grad()
    def evaluate_policy(
        self,
        queries: Tensor,
        max_steps: Optional[int] = None,
        deterministic: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Evaluate policy on a batch of queries."""
        device = self.device
        max_steps = max_steps or self.max_depth
        
        self.env.set_eval_dataset(queries)
        obs, state = self.env.reset()
        
        B = state['current_states'].shape[0]
        total_log_probs = torch.zeros(B, device=device)
        total_rewards = torch.zeros(B, device=device)
        
        for _ in range(max_steps):
            if state['done'].all():
                break
            
            torch.compiler.cudagraph_mark_step_begin()
            logits = self._compiled_policy_fn(obs).clone()
            
            masked_logits = torch.where(obs['action_mask'], logits, torch.full_like(logits, float('-inf')))
            actions = masked_logits.argmax(dim=-1) if deterministic else \
                      torch.multinomial(torch.softmax(masked_logits, dim=-1), 1).squeeze(-1)
            
            step_log_probs = torch.log_softmax(masked_logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
            step_log_probs = torch.where(~state['done'], step_log_probs, torch.zeros_like(step_log_probs))
            
            new_obs, new_state = self.env.step(state, actions)
            total_log_probs += step_log_probs
            total_rewards += new_state['step_rewards']
            state, obs = new_state, new_obs
        
        return total_log_probs, state['success'], state['depths'], total_rewards
    
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
    
    @torch.no_grad()
    def evaluate(
        self,
        queries: Optional[Tensor] = None,
        sampler: Optional[Any] = None,
        *,
        n_corruptions: Optional[int] = None,
        corruption_modes: Optional[Sequence[str]] = None,
        chunk_queries: int = 50,
        verbose: bool = False,
        deterministic: bool = True,
        parity_mode: bool = False,
        query_depths: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate policy on queries with corruptions for ranking metrics (MRR, Hits@K).
        
        Auto-configures from config._components if parameters not provided.
        
        For each query, generates corruptions and evaluates all candidates 
        (positive + corruptions) to compute ranking metrics.
        
        This method handles ALL chunking and padding:
        1. Chunks positive queries into batches of chunk_queries
        2. For each chunk, generates corruptions per corruption mode
        3. Flattens candidates and pads to fixed_batch_size
        4. Calls evaluate_policy with padded batches
        5. Computes ranks and aggregates metrics
        
        Args:
            queries: [total_queries, 3] Tensor of test triples (auto-fetched if None)
            sampler: Sampler for generating corruptions (auto-fetched if None)
            n_corruptions: Number of corruptions per query (uses config if None)
            corruption_modes: Tuple of modes ('head', 'tail') (uses config if None)
            chunk_queries: Number of positive queries to process at once
            verbose: Print progress
            deterministic: Use deterministic action selection
            parity_mode: If True, use numpy RNG for tie-breaking
            query_depths: Optional [total_queries] Tensor of query depths for detailed metrics
            
        Returns:
            Dictionary with MRR and Hits@n_corruptions metrics
        """
        from callbacks import Display # Delayed import to avoid circular dependency
        
        # Auto-configure from config if parameters not provided
        if queries is None:
            if hasattr(self, 'config') and hasattr(self.config, '_components'):
                queries = self.config._components.get('test_queries')
                if queries is None:
                    raise ValueError("No queries provided and config._components['test_queries'] not found")
            else:
                raise ValueError("queries parameter is required when config is not available")
        
        if sampler is None:
            if hasattr(self, 'config') and hasattr(self.config, '_components'):
                sampler = self.config._components.get('sampler')
                if sampler is None:
                    raise ValueError("No sampler provided and config._components['sampler'] not found")
            else:
                raise ValueError("sampler parameter is required when config is not available")
        
        if n_corruptions is None:
            n_corruptions = getattr(self.config, 'n_corruptions', 50) if hasattr(self, 'config') else 50
        
        if corruption_modes is None:
            corruption_modes = tuple(getattr(self.config, 'corruption_scheme', ('head', 'tail'))) if hasattr(self, 'config') else ('head', 'tail')
        
        if self._compiled_policy_fn is None:
            raise RuntimeError("PPO must be compiled for evaluation (not in eval_only mode)")
        
        device = self.device
        total_queries = queries.shape[0]
        fixed_batch_size = self.fixed_batch_size
        per_mode_ranks = {
            mode: torch.zeros(total_queries, device=device) 
            for mode in corruption_modes
        }
        
        # Accumulators for aggregate stats
        acc_lengths = []
        acc_rewards = []
        acc_success = []
        acc_is_pos = []
        acc_depths = []
        
        # Initialize RNG for parity mode once (outside loop)
        rng = None
        if parity_mode:
            # Numpy RNG with seed=0 for parity with model_eval.py
            rng = np.random.RandomState(0)

        # Process positive queries in chunks
        for start in range(0, total_queries, chunk_queries):
            end = min(start + chunk_queries, total_queries)
            Q = end - start
            
            if verbose:
                print(f"Processing queries {start}-{end} / {total_queries}")
            
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
                        chunk_queries_tensor, num_negatives=n_corruptions, mode='head', device=device
                    )
                if sampler_mode in ('tail', 'both'):
                    tail_corruptions = sampler.corrupt(
                        chunk_queries_tensor, num_negatives=n_corruptions, mode='tail', device=device
                    )
            
            for mode in corruption_modes:
                if parity_mode:
                    if mode == 'head':
                        corruptions = head_corruptions if head_corruptions is not None else torch.zeros(Q, n_corruptions, 3, dtype=torch.long, device=device)
                    else:
                        corruptions = tail_corruptions if tail_corruptions is not None else torch.zeros(Q, n_corruptions, 3, dtype=torch.long, device=device)
                else:
                    # Fast path: generate corruptions directly with requested mode
                    corruptions = sampler.corrupt(
                        chunk_queries_tensor,
                        num_negatives=n_corruptions,
                        mode=mode,
                        device=device
                    )
                
                # Handle variable corruption counts
                valid_mask = corruptions.sum(dim=-1) != 0  # [Q, n_corruptions]
                
                # Create candidates: positive + corruptions -> [Q, 1+n_corruptions, 3]
                candidates = torch.zeros(Q, 1 + n_corruptions, 3, dtype=torch.long, device=device)
                candidates[:, 0, :] = chunk_queries_tensor
                candidates[:, 1:, :] = corruptions
                
                # Flatten for batch evaluation: [Q*(1+n_corruptions), 3]
                flat_candidates = candidates.view(-1, 3)
                total_candidates = flat_candidates.shape[0]
                
                # Process candidates in chunks of fixed_batch_size with padding
                all_log_probs = []
                all_success = []
                all_lengths = []
                all_rewards = []
                
                for cand_start in range(0, total_candidates, fixed_batch_size):
                    cand_end = min(cand_start + fixed_batch_size, total_candidates)
                    batch_candidates = flat_candidates[cand_start:cand_end]
                    actual_size = batch_candidates.shape[0]
                    
                    # Pad to fixed_batch_size
                    padded_candidates, _ = self._pad_queries(batch_candidates)
                    
                    # Evaluate
                    log_probs, success, depths, rewards = self.evaluate_policy(
                        queries=padded_candidates,
                        max_steps=self.max_depth,
                        deterministic=deterministic,
                    )
                    
                    # Trim to actual size
                    all_log_probs.append(log_probs[:actual_size])
                    all_success.append(success[:actual_size])
                    all_lengths.append(depths[:actual_size]) # depths output is actually episode length
                    all_rewards.append(rewards[:actual_size])
                
                # Concatenate results
                log_probs = torch.cat(all_log_probs, dim=0)  # [Q*(1+n_corruptions)]
                success = torch.cat(all_success, dim=0)      # [Q*(1+n_corruptions)]
                lengths = torch.cat(all_lengths, dim=0)      # [Q*(1+n_corruptions)]
                rewards = torch.cat(all_rewards, dim=0)      # [Q*(1+n_corruptions)]
                
                # Accumulate detailed stats (filtering invalid negatives)
                # Reshape masks to map back to candidates
                valid_mask_full = torch.zeros(Q, 1 + n_corruptions, dtype=torch.bool, device=device)
                valid_mask_full[:, 0] = True # Positives always valid
                valid_mask_full[:, 1:] = valid_mask # Negatives
                
                flat_valid = valid_mask_full.view(-1)
                
                # Stats for valid entries only
                acc_lengths.append(lengths[flat_valid])
                acc_rewards.append(rewards[flat_valid])
                acc_success.append(success[flat_valid].float())
                
                # Positive mask: index % (1+n_corruptions) == 0
                is_pos = torch.zeros(Q * (1 + n_corruptions), dtype=torch.bool, device=device)
                is_pos[torch.arange(0, Q * (1 + n_corruptions), 1 + n_corruptions, device=device)] = True
                acc_is_pos.append(is_pos[flat_valid])
                
                # Depths
                if query_depths is not None:
                    # Expand depths: [Q] -> [Q, 1+n_corruptions] -> flattened -> valid
                    chunk_depths = query_depths[start:end].to(device)
                    expanded_depths = chunk_depths.unsqueeze(1).expand(Q, 1 + n_corruptions).reshape(-1)
                    # For metrics, we use -1 for non-positives or if not specified
                    # Actually model_eval.py allows bucketting negatives by "unknown_neg".
                    # But typically we want specific depths for POSITIVE queries.
                    # model_eval.py fills negatives with -1 depth.
                    mask_pos_full = is_pos # [Q*(1+n_corruptions)]
                    expanded_depths = torch.where(mask_pos_full, expanded_depths, torch.full_like(expanded_depths, -1))
                    acc_depths.append(expanded_depths[flat_valid])


                # Reshape results: [Q, 1+n_corruptions]
                log_probs = log_probs.view(Q, 1 + n_corruptions)
                success = success.view(Q, 1 + n_corruptions)
                
                # Apply success penalty - failed proofs get -100 penalty
                log_probs = log_probs.clone()
                log_probs[~success.bool()] -= 100.0
                
                # Ranking with random tie-breaking
                pos_score = log_probs[:, 0:1]  # [Q, 1]
                neg_scores = log_probs[:, 1:]  # [Q, n_corruptions]
                
                # Random keys for tie-breaking
                if parity_mode:
                    # Use pre-initialized RNG for consistent consumption order
                    rnd = torch.as_tensor(rng.rand(Q, 1 + n_corruptions), device=device, dtype=torch.float32)
                else:
                    # Fast torch RNG (no seeding needed, just for tie-breaking)
                    rnd = torch.rand(Q, 1 + n_corruptions, device=device)
                
                # Count negatives that beat positive (better score, or tied with higher random key)
                better = (neg_scores > pos_score) & valid_mask
                tied_wins = (neg_scores == pos_score) & (rnd[:, 1:] > rnd[:, 0:1]) & valid_mask
                ranks = 1 + better.sum(dim=1) + tied_wins.sum(dim=1)
                per_mode_ranks[mode][start:end] = ranks.float()
        
        # Aggregate results
        results: Dict[str, Any] = {
            "MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0,
            "per_mode": {}
        }
        
        for mode in corruption_modes:
            all_ranks = per_mode_ranks[mode]
            results["per_mode"][mode] = compute_metrics_from_ranks(all_ranks)
        
        # Average across modes
        for mode in corruption_modes:
            for k, v in results["per_mode"][mode].items():
                results[k] += v
        
        n_modes = len(corruption_modes)
        for k in ["MRR", "Hits@1", "Hits@3", "Hits@10"]:
            results[k] /= n_modes if n_modes > 0 else 1.0
            
        results["_mrr"] = results["MRR"]
        
        # --- Stats Formatting (matching model_eval.py) ---
        if acc_lengths:
            all_lens = torch.cat(acc_lengths).float()
            all_rews = torch.cat(acc_rewards).float()
            all_succ = torch.cat(acc_success).float()
            all_pos = torch.cat(acc_is_pos)
            all_depths_t = torch.cat(acc_depths) if acc_depths else None
            
            def fmt(t):
                count = t.numel()
                if count == 0: return Display._format_stat_string(None, None, 0)
                if count == 1: return Display._format_stat_string(t.mean().item(), 0.0, 1)
                return Display._format_stat_string(t.mean().item(), t.std().item(), count)
            
            results["len"] = fmt(all_lens)
            results["ep_len_mean"] = getattr(all_lens.mean(), 'item', lambda: 0.0)()
            
            pos_idxs = torch.nonzero(all_pos).view(-1)
            neg_idxs = torch.nonzero(~all_pos).view(-1)
            
            if pos_idxs.numel() > 0:
                results["reward"] = fmt(all_rews[pos_idxs])
                results["ep_rew_mean"] = getattr(all_rews.mean(), 'item', lambda: 0.0)()
                results["success_rate"] = getattr(all_succ[pos_idxs].mean(), 'item', lambda: 0.0)()
            
            for lbl_key, idxs in [("pos", pos_idxs), ("neg", neg_idxs)]:
                if idxs.numel() > 0:
                    results[f"len_{lbl_key}"] = fmt(all_lens[idxs])
                    results[f"reward_{lbl_key}"] = fmt(all_rews[idxs])
                    results[f"proven_{lbl_key}"] = fmt(all_succ[idxs])
            
            # By Depth
            if all_depths_t is not None:
                unique_d = torch.unique(all_depths_t)
                for d in unique_d:
                    d_val = int(d.item())
                    mask_d = (all_depths_t == d)
                    for is_p, lbl in [(True, "pos"), (False, "neg")]:
                        mask_dp = mask_d & (all_pos if is_p else ~all_pos)
                        if mask_dp.any():
                            depth_key = Display._format_depth_key(d_val if is_p else -1)
                            results[f"len_d_{depth_key}_{lbl}"] = fmt(all_lens[mask_dp])
                            results[f"reward_d_{depth_key}_{lbl}"] = fmt(all_rews[mask_dp])
                            results[f"proven_d_{depth_key}_{lbl}"] = fmt(all_succ[mask_dp])

        if verbose:
            print(f"\nResults:")
            print(f"  MRR: {results['MRR']:.4f}")
            print(f"  Hits@1: {results['Hits@1']:.4f}")
            print(f"  Hits@10: {results['Hits@10']:.4f}")
        
        return results
