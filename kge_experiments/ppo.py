"""
Optimal PPO (Proximal Policy Optimization) with CUDA graph support.

Optimized implementation combining best practices from deep dive documents:
- Separate _compiled_rollout_step (stochastic) and _compiled_eval_step (deterministic)
- Uses _uncompiled_policy in fused functions to avoid nested compilation
- Buffer-copy pattern for CUDA graph stability
- V10-style evaluation with slot recycling

Training: collect_rollouts() + train()
Evaluation: evaluate_policy() + evaluate()
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch import Tensor
from sklearn.metrics import average_precision_score
from typing import Optional, Callable, Dict, List, Tuple, Any, Sequence
from tensordict import TensorDict

from rollout import RolloutBuffer
from env import EnvVec, EnvObs, EnvState
from utils import atom_to_str

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')


# =============================================================================
# HELPERS
# =============================================================================

def compute_metrics_from_ranks(ranks: Tensor) -> Dict[str, float]:
    if ranks.numel() == 0:
        return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}
    r = ranks.float()
    return {
        "MRR": float((1.0 / r).mean().item()),
        "Hits@1": float((r <= 1.0).float().mean().item()),
        "Hits@3": float((r <= 3.0).float().mean().item()),
        "Hits@10": float((r <= 10.0).float().mean().item()),
    }

def explained_variance(y_pred: Tensor, y_true: Tensor) -> Tensor:
    var_y = torch.var(y_true)
    return 1.0 - torch.var(y_true - y_pred) / (var_y + 1e-8)


# =============================================================================
# LOSS MODULE
# =============================================================================

class PPOLossModule(nn.Module):
    """Fused policy forward + loss computation."""
    
    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, sub_index, derived_sub_indices, action_mask, actions, advantages, returns, old_log_probs, old_values, clip_range, clip_range_vf, ent_coef, vf_coef):
        values, log_probs, entropy = self.policy.evaluate_actions(sub_index, derived_sub_indices, action_mask, actions)
        values = values.flatten()

        # PPO Loss
        log_ratio = log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        pg1 = advantages * ratio
        pg2 = advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        policy_loss = -torch.minimum(pg1, pg2).mean()
        clip_fraction = (torch.abs(ratio - 1) > clip_range).float().mean()

        # Value loss
        value_diff = values - old_values
        clipped_diff = torch.clamp(value_diff, -clip_range_vf, clip_range_vf)
        values_pred = torch.where(torch.tensor(clip_range_vf > 0, device=values.device), old_values + clipped_diff, values)
        value_loss = F.mse_loss(returns, values_pred)

        # Entropy
        entropy_loss = -entropy.mean()

        # Total
        loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss
        approx_kl = ((ratio - 1.0) - log_ratio).mean()

        return torch.stack([loss, policy_loss, value_loss, entropy_loss, approx_kl.detach(), clip_fraction])


# =============================================================================
# PPO OPTIMAL CLASS
# =============================================================================

class PPO:
    """Optimal PPO with CUDA graph support and separate train/eval compilation."""

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------

    def __init__(self, policy: nn.Module, env: EnvVec, config: Any, **kwargs: Any) -> None:
        self.config = config
        self.policy = policy
        self.env = env

        # Core params
        self.batch_size_env = kwargs.get('batch_size_env', config.n_envs)
        self.padding_atoms = kwargs.get('padding_atoms', config.padding_atoms)
        self.padding_states = kwargs.get('padding_states', config.padding_states)
        self.max_depth = kwargs.get('max_depth', getattr(config, 'max_steps', getattr(config, 'max_depth', 20)))

        # PPO hyperparams
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

        # Device
        components = getattr(config, '_components', {})
        self.device = kwargs.get('device', components.get('device', getattr(config, 'device', torch.device('cpu'))))
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        self.verbose = kwargs.get('verbose', config.verbose)
        self.seed = kwargs.get('seed', config.seed)
        self.eval_only = kwargs.get('eval_only', getattr(config, 'eval_only', False))
        self._compile_mode = kwargs.get('compile_mode', 'reduce-overhead')
        self.ranking_compile_mode = kwargs.get('ranking_compile_mode', getattr(config, 'ranking_compile_mode', 'default'))
        self._fixed_batch_size = kwargs.get('fixed_batch_size', getattr(config, 'fixed_batch_size', None))
        self.ranking_tie_seed = int(getattr(config, 'ranking_tie_seed', 0))

        # KGE inference (eval-time fusion)
        self.kge_inference_engine = kwargs.get(
            'kge_inference_engine', getattr(config, 'kge_inference_engine', None)
        )
        self.kge_index_manager = kwargs.get(
            'kge_index_manager', getattr(config, 'kge_index_manager', None)
        )
        self.kge_inference = bool(getattr(config, 'kge_inference', False))
        self.kge_inference_success = bool(getattr(config, 'kge_inference_success', True))
        self.kge_eval_kge_weight = float(getattr(config, 'kge_eval_kge_weight', 2.0))
        self.kge_eval_rl_weight = float(getattr(config, 'kge_eval_rl_weight', 1.0))
        self.kge_fail_penalty = float(getattr(config, 'kge_fail_penalty', 100.0))
        self.kge_only_eval = bool(getattr(config, 'kge_only_eval', False))
        self._kge_log_eps = 1e-9

        # Metrics info (CPU-only to avoid synchronization) - copied from PPOOld
        query_labels = kwargs.get('query_labels', None)
        query_depths = kwargs.get('query_depths', None)
        
        # Try to extract from components if not provided
        components = getattr(config, '_components', {}) 
        if query_depths is None and components.get('dh'):
            dh = components.get('dh')
            if hasattr(dh, 'get_materialized_split'):
                try:
                    train_split = dh.get_materialized_split('train')
                    query_depths = train_split.depths
                    # FIX: Extract labels or default to ones (all positives for training)
                    if hasattr(train_split, 'labels') and train_split.labels.numel() == query_depths.numel():
                        query_labels = train_split.labels
                except Exception as e:
                    if self.verbose:
                        print(f"[PPO] Warning: Could not extract training depths: {e}")

        self.query_labels = query_labels.detach().cpu() if query_labels is not None else None
        self.query_depths = query_depths.detach().cpu() if query_depths is not None else None

        # IMPORTANT: Keep uncompiled policy reference for fused steps
        self._uncompiled_policy = policy
        self.last_train_metrics = {}

        # Pre-allocated eval buffers
        self._eval_padded_buffer = torch.zeros(self.fixed_batch_size, 3, dtype=torch.long, device=self.device)
        max_cand = self.fixed_batch_size * 1001
        self._eval_acc_success = torch.zeros(max_cand, dtype=torch.bool, device=self.device)
        self._eval_result_success = torch.zeros(self.fixed_batch_size, dtype=torch.bool, device=self.device)
        self._eval_result_depths = torch.zeros(self.fixed_batch_size, dtype=torch.long, device=self.device)

        # AMP
        use_amp = kwargs.get('use_amp', True)
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.amp_dtype = torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported() else torch.float32

        # Training setup
        if not self.eval_only:
            self.rollout_buffer = RolloutBuffer(
                buffer_size=self.n_steps, n_envs=self.batch_size_env, device=self.device,
                gamma=self.gamma, gae_lambda=self.gae_lambda,
                padding_atoms=self.padding_atoms, padding_states=self.padding_states,
                batch_size=self.batch_size,
            )
            A, S = self.padding_atoms, self.padding_states
            self._train_sub_index = torch.zeros((self.batch_size, 1, A, 3), dtype=torch.long, device=self.device)
            self._train_derived = torch.zeros((self.batch_size, S, A, 3), dtype=torch.long, device=self.device)
            self._train_mask = torch.zeros((self.batch_size, S), dtype=torch.uint8, device=self.device)
            self._train_actions = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
            self._train_advantages = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
            self._train_returns = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
            self._train_old_log_probs = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
            self._train_old_values = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)

            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate, eps=1e-5, fused=(self.device.type == 'cuda'))
            self._compile_all()
        else:
            self.rollout_buffer = None
            self.optimizer = None
        
        # Ensure evaluation is also set up (buffers and kernels)
        self._setup_fused_eval_step()

        self.num_timesteps = 0
        self.callback = None  # Disabled for optimal version
        self.current_query_indices = None

    @property
    def fixed_batch_size(self) -> int:
        return self._fixed_batch_size or self.batch_size_env

    # -------------------------------------------------------------------------
    # COMPILATION
    # -------------------------------------------------------------------------

    def _compile_all(self):
        """Compile all functions using uncompiled policy in fused steps."""
        # Compile loss module
        self.loss_module = torch.compile(PPOLossModule(self._uncompiled_policy), mode=self._compile_mode, fullgraph=True)

        # Warmup gradients
        self._warmup_gradients()

        # Compile policy for standalone use
        self._compiled_policy_fn = torch.compile(self._uncompiled_policy.get_logits, mode=self._compile_mode, fullgraph=True)
        self.policy = torch.compile(self._uncompiled_policy, mode=self._compile_mode, fullgraph=True)

        # Setup fused steps (uses _uncompiled_policy!)
        if self.config.compile:
            self._setup_fused_rollout_step()

    def _warmup_gradients(self):
        """Pre-allocate gradients for CUDA graph stability."""
        B, A, S = self.batch_size, self.padding_atoms, self.padding_states
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
            v, lp, ent = self._uncompiled_policy.evaluate_actions(
                torch.zeros((B, 1, A, 3), dtype=torch.long, device=self.device),
                torch.zeros((B, S, A, 3), dtype=torch.long, device=self.device),
                torch.ones((B, S), dtype=torch.uint8, device=self.device),
                torch.zeros(B, dtype=torch.long, device=self.device),
            )
            loss = v.mean() + lp.mean() + ent.mean()
        self.optimizer.zero_grad(set_to_none=False)
        loss.backward()
        self.optimizer.zero_grad(set_to_none=False)

    def _setup_fused_rollout_step(self):
        """Fused policy + env step for training (stochastic, with log_probs).
        
        CRITICAL: Uses self._uncompiled_policy to avoid nested compilation!
        NOTE: log_probs are NOT masked for done states to match TensorPPO behavior.
              The PPO loss computation handles done states via episode_starts/advantages.
        """
        policy = self._uncompiled_policy  # Uncompiled!
        env = self.env.env if hasattr(self.env, 'env') and not isinstance(self.env, EnvVec) else self.env

        def fused_step(obs, state):
            # Policy forward (not compiled here - compiled as part of fused_step)
            logits = policy.get_logits(obs)
            masked = logits.masked_fill(obs['action_mask'] == 0, -3.4e38)
            
            # Stochastic action selection
            probs = torch.softmax(masked, dim=-1)
            actions = torch.multinomial(probs, 1).squeeze(-1)
            
            # Log probabilities for PPO (no masking for done states - matches TensorPPO)
            log_probs = torch.log_softmax(masked, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Environment step with auto-reset
            new_obs, new_state = env._step_and_reset_core(state, actions, env._query_pool, env._per_env_ptrs)
            
            return new_obs, new_state, actions, log_probs

        self._compiled_rollout_step = torch.compile(fused_step, mode=self._compile_mode, fullgraph=True)

    def _setup_fused_eval_step(self):
        """Consolidated setup for both standard and ranking evaluation.
        
        Initializes persistent buffers and compiles evaluation kernels (deterministic).
        """
        if hasattr(self, '_compiled_eval_step'):
            return
            
        # 1. Standard Step Setup
        policy = self._uncompiled_policy
        env = self.env.env if hasattr(self.env, 'env') and not isinstance(self.env, EnvVec) else self.env
        device = self.device
        _compile_mode = self._compile_mode

        def fused_eval(obs, state):
            """Core deterministic step used by evaluate_policy."""
            # IMPORTANT: Use self._uncompiled_policy directly, not the captured 'policy' variable,
            # to ensure torch.compile sees the current weights rather than caching stale weights
            logits = self._uncompiled_policy.get_logits(obs)
            masked = logits.masked_fill(obs['action_mask'] == 0, -3.4e38)
            actions = masked.argmax(dim=-1)
            new_obs, new_state = env._step_core(state, actions)
            return new_obs, new_state

        # Use 'default' mode instead of 'reduce-overhead' to avoid CUDA graph caching of weights
        self._compiled_eval_step = torch.compile(fused_eval, mode='default', fullgraph=True)

        # 2. Ranking Evaluation Setup (Buffers + Compilation)
        B, S, A = self.fixed_batch_size, self.padding_states, self.padding_atoms
        H = self.max_depth + 1
        pad = self.env.padding_idx
        
        # Persistent state buffers for optimized ranking
        self._ranking_current = torch.full((B, A, 3), pad, dtype=torch.long, device=device)
        self._ranking_derived = torch.full((B, S, A, 3), pad, dtype=torch.long, device=device)
        self._ranking_counts = torch.zeros(B, dtype=torch.long, device=device)
        self._ranking_mask = torch.zeros(B, S, dtype=torch.bool, device=device)
        self._ranking_depths = torch.zeros(B, dtype=torch.long, device=device)
        self._ranking_done = torch.zeros(B, dtype=torch.bool, device=device)
        self._ranking_pool_ptr = torch.zeros(B, dtype=torch.long, device=device)

        # Original queries for excluded_queries computation
        self._ranking_original_queries = torch.full((B, A, 3), pad, dtype=torch.long, device=device)

        # History tracking for memory pruning
        self._ranking_history_hashes = torch.zeros(B, H, dtype=torch.long, device=device)
        self._ranking_history_count = torch.zeros(B, dtype=torch.long, device=device)
        self._ranking_max_history = H
        
        # Variable index tracking for proper unification
        self._ranking_next_var = torch.zeros(B, dtype=torch.long, device=device)
        self._runtime_var_start_index = env.runtime_var_start_index
        
        # Global ranking pool
        max_pool = 4_000_000
        self._ranking_max_pool = max_pool
        self._ranking_pool = torch.zeros(max_pool, 3, dtype=torch.long, device=device)
        self._ranking_pool_size = torch.tensor(0, dtype=torch.long, device=device)
        self._ranking_result_buf = torch.zeros(max_pool, dtype=torch.bool, device=device)

        # Log probability tracking buffers
        self._ranking_ep_logprob = torch.zeros(B, dtype=torch.float32, device=device)  # Per-slot accumulator
        self._ranking_ep_logprob = torch.zeros(B, dtype=torch.float32, device=device)  # Per-slot accumulator
        self._ranking_result_logprob = torch.zeros(max_pool, dtype=torch.float32, device=device)  # Per-query final log prob
        self._ranking_result_depths = torch.zeros(max_pool, dtype=torch.long, device=device)
        self._ranking_result_success = torch.zeros(max_pool, dtype=torch.bool, device=device)
        self._ranking_result_rewards = torch.zeros(max_pool, dtype=torch.float32, device=device)

        # Helper buffers
        self._ranking_stride = torch.tensor(B, dtype=torch.long, device=device)
        self._ranking_arange_B = torch.arange(B, device=device)
        self._ranking_arange_S = torch.arange(S, device=device)

        # Pre-allocated obs dict for ranking_step (avoids dict creation overhead)
        self._ranking_obs = {
            'sub_index': torch.zeros((B, 1, A, 3), dtype=torch.long, device=device),
            'derived_sub_indices': torch.zeros((B, S, A, 3), dtype=torch.long, device=device),
            'action_mask': torch.zeros((B, S), dtype=torch.uint8, device=device),
        }

        # Pre-allocated state dict for _step_core (avoids TensorDict creation overhead)
        self._ranking_state_dict = TensorDict({
            "current_states": torch.zeros((B, A, 3), dtype=torch.long, device=device),
            "derived_states": torch.zeros((B, S, A, 3), dtype=torch.long, device=device),
            "derived_counts": torch.zeros(B, dtype=torch.long, device=device),
            "original_queries": torch.zeros((B, A, 3), dtype=torch.long, device=device),
            "next_var_indices": torch.zeros(B, dtype=torch.long, device=device),
            "depths": torch.zeros(B, dtype=torch.long, device=device),
            "done": torch.zeros(B, dtype=torch.uint8, device=device),
            "success": torch.zeros(B, dtype=torch.uint8, device=device),
            "current_labels": torch.ones(B, dtype=torch.long, device=device),
            "history_hashes": torch.zeros((B, H), dtype=torch.long, device=device),
            "history_count": torch.zeros(B, dtype=torch.long, device=device),
            "step_rewards": torch.zeros(B, dtype=torch.float32, device=device),
            "step_dones": torch.zeros(B, dtype=torch.uint8, device=device),
            "step_successes": torch.zeros(B, dtype=torch.uint8, device=device),
            "step_labels": torch.zeros(B, dtype=torch.long, device=device),
            "cumulative_rewards": torch.zeros(B, dtype=torch.float32, device=device),
            "per_env_ptrs": torch.zeros(B, dtype=torch.long, device=device),
            "neg_counters": torch.zeros(B, dtype=torch.int64, device=device),
        }, batch_size=[B], device=device)

        # Capture predicate indices as constants for torch.compile compatibility
        true_pred_idx = env.true_pred_idx
        false_pred_idx = env.false_pred_idx
        end_pred_idx = env.end_pred_idx

        # Ranking Loop Kernel
        def ranking_step(pool: Tensor, pool_size: Tensor):
            """Fused step with slot recycling, history tracking and memory pruning."""
            # Create obs dict with views to current ranking state (no copies needed)
            obs = {
                'sub_index': self._ranking_current.unsqueeze(1),
                'derived_sub_indices': self._ranking_derived,
                'action_mask': self._ranking_mask.to(torch.uint8),
            }
            # Policy forward
            logits = self._uncompiled_policy.get_logits(obs)
            masked_logits = logits.masked_fill(~self._ranking_mask, -3.4e38)
            actions = masked_logits.argmax(dim=-1)

            # Compute log probabilities for selected actions (detach to avoid gradient tracking)
            log_probs = torch.log_softmax(masked_logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1).detach()

            # Create state dict with views to current ranking state (no copies for basic fields)
            current_state_dict = TensorDict({
                "current_states": self._ranking_current,
                "derived_states": self._ranking_derived,
                "derived_counts": self._ranking_counts,
                "original_queries": self._ranking_original_queries,
                "next_var_indices": self._ranking_next_var,
                "depths": self._ranking_depths,
                "done": self._ranking_done.to(torch.uint8),
                "success": torch.zeros(B, dtype=torch.uint8, device=device),
                "current_labels": torch.ones(B, dtype=torch.long, device=device),
                "history_hashes": self._ranking_history_hashes,
                "history_count": self._ranking_history_count,
                "step_rewards": torch.zeros(B, dtype=torch.float32, device=device),
                "step_dones": torch.zeros(B, dtype=torch.uint8, device=device),
                "step_successes": torch.zeros(B, dtype=torch.uint8, device=device),
                "step_labels": torch.zeros(B, dtype=torch.long, device=device),
                "cumulative_rewards": torch.zeros(B, dtype=torch.float32, device=device),
                "per_env_ptrs": self._ranking_pool_ptr,
                "neg_counters": torch.zeros(B, dtype=torch.int64, device=device),
            }, batch_size=[B], device=device)

            # Call shared logic
            # Note: We pass stride as slot_lengths/offsets is not exactly what we need for linear scan with reset
            # But the ranking logic relies on strided pool buffer access.
            # Actually, `ranking_step` logic in `_step_and_reset_core` supports `per_env_ptrs` as index into pool.
            # We just need to handle the `finished_idx` logic externally?
            # Wait, `_step_and_reset_core` automatically advances `per_env_ptrs`.
            # In our case, `self._ranking_stride` logic was: `new_ptr = ptr + stride`.
            # `_step_and_reset_core` does `next_ptrs = per_env_ptrs + 1` (linear) or `+ stride`?
            # It assumes `per_env_ptrs` behaves like an index. 
            # In `ranking_step` (original), we used `_ranking_stride` (B). So ptrs were: 0, B, 2B...
            # This is "Bucket Scheduling".
            # `_step_and_reset_core` supports "Slot Scheduling" via `slot_lengths` and `slot_offsets`.
            # If we pass `slot_lengths=1` (effectively) and proper `slot_offsets`, we can replicate it.
            # Or simpler: we update pointers manually if `_step_and_reset_core` doesn't do what we want.
            # Actually `_step_and_reset_core` updates `per_env_ptrs`.
            # Let's check `_step_and_reset_core`: 
            # if slot_lengths/offsets: query_idx = offset + (ptr % len). ptr++.
            # if order=True (eval): query_idx = ptr % size. ptr++.
            
            # The original `ranking_step` logic was: `ptr += stride`.
            # This suggests we are treating the pool as a flat list, but jumping by B?
            # No, `ptr` was initialized to `arange(B)`. Then `ptr += B`.
            # So slot 0 gets 0, B, 2B...
            # Slot 1 gets 1, B+1, 2B+1...
            # This is INTERLEAVED scheduling.
            # `_step_and_reset_core` with `order=True` does `ptr++`.
            # `ptr` starts at `arange(B)`.
            # Next step `ptr` becomes `arange(B) + 1`.
            # Slot 0 gets 0, 1, 2...
            # Slot 1 gets 1, 2, 3...
            # This means multiple slots would work on the SAME query (redundant work!).
            
            # We MUST maintain the strided behavior: Slot i works on queries {i, i+B, i+2B...}.
            # `_step_and_reset_core` with `slot_lengths` and `slot_offsets` can do this.
            # Slot Length = Total Queries / B (approx).
            # Slot Offset = i * Slot Length? No.
            
            # To restore "Interleaved" behavior with `_step_and_reset_core`:
            # We can rely on `per_env_ptrs` tracking the ACTUAL index.
            # But `_step_and_reset_core` increments by 1.
            # WE NEED TO INCREMENT BY B.
            
            # Option 1: Pass `slot_lengths=None` (default). `_step_and_reset_core` increments by 1.
            # This breaks interleaving.
            
            # Option 2: Use `_step_core` -> Manual Reset.
            # This basically brings us back to `ranking_step` logic, but using `_step_core` for the step part.
            
            active = ~self._ranking_done
            
            # Accumulate log probs for active slots
            new_ep_logprob = torch.where(active, self._ranking_ep_logprob + log_probs, self._ranking_ep_logprob)

            # 1. Step Core
            new_obs, new_state = env._step_core(current_state_dict, actions)
            
            new_current = new_state['current_states']
            new_derived = new_state['derived_states']
            new_counts = new_state['derived_counts']
            new_depths = new_state['depths']
            new_done = new_state['done'].bool()
            new_success = new_state['success'].bool()
            new_history = new_state['history_hashes']
            new_h_count = new_state['history_count']
            new_original = new_state['original_queries'] # Just passed through
            new_next_var = new_state['next_var_indices']
            
            rewards = new_state['step_rewards']
            step_dones = new_state['step_dones'].bool()
            
            # 2. Reset / Recycle Logic (Custom for Ranking Step)
            # Use step_dones from env (handles max_dist, terminal)
            finished_idx = torch.where(step_dones, self._ranking_pool_ptr, torch.full_like(self._ranking_pool_ptr, -1))
            
            # Stride update: ptr += B
            new_ptr = torch.where(step_dones, self._ranking_pool_ptr + self._ranking_stride, self._ranking_pool_ptr)
            needs_reset = step_dones & (new_ptr < pool_size)
            
            # Prepare log probs for finished episodes
            valid_finish = step_dones & (finished_idx >= 0) & (finished_idx < pool_size)
            ep_logprob_to_store = torch.where(valid_finish, new_ep_logprob, torch.zeros_like(new_ep_logprob))
            
            # Perform Reset using Env's reset logic
            safe_idx = new_ptr.clamp(0, max_pool - 1)
            reset_queries_raw = pool[safe_idx]
            reset_labels = torch.ones(B, dtype=torch.long, device=device)
            reset_state = env._reset_from_queries(reset_queries_raw, reset_labels)
            
            # Merge logic (needs_reset mask)
            # Inlined merge since _step_and_reset_core_merge is not exposed
            
            m1 = needs_reset.view(B, 1, 1)
            m3 = needs_reset.view(B, 1, 1, 1)
            mH = needs_reset.view(B, 1).expand(B, H)
            
            final_current = torch.where(m1, reset_state['current_states'], new_current)
            final_derived = torch.where(m3, reset_state['derived_states'], new_derived)
            final_counts = torch.where(needs_reset, reset_state['derived_counts'], new_counts)
            final_depths = torch.where(needs_reset, reset_state['depths'], new_depths)
            final_done = torch.where(needs_reset, reset_state['done'], new_done)
            final_history = torch.where(mH, reset_state['history_hashes'], new_history)
            final_h_count = torch.where(needs_reset, reset_state['history_count'], new_h_count)
            final_original = torch.where(m1, reset_state['original_queries'], new_original)
            final_next_var = torch.where(needs_reset, reset_state['next_var_indices'], new_next_var)
            
            # Reset ep_logprob
            final_ep_logprob = torch.where(needs_reset, torch.zeros_like(new_ep_logprob), new_ep_logprob)
            
            # Handle exhausted
            exhausted = (new_ptr >= pool_size) & step_dones
            final_done = torch.where(exhausted, torch.ones_like(final_done), final_done)
            final_ep_logprob = torch.where(exhausted, torch.zeros_like(final_ep_logprob), final_ep_logprob)
            
            new_mask = self._ranking_arange_S.unsqueeze(0) < final_counts.unsqueeze(1)

            # Return both final_depths (for state) and new_depths (for recording)
            # new_depths is the depth at step completion BEFORE reset merge
            return (final_current, final_derived, final_counts, new_mask,
                    final_depths, final_done, new_ptr, final_history, final_h_count,
                    step_dones, new_success, finished_idx, final_ep_logprob, ep_logprob_to_store, final_original, final_next_var, new_depths)

        # Use configurable mode. 'default' is safer for interleaved training/eval.
        # 'reduce-overhead' makes profile_eval faster but risks stale weights if not careful.
        mode = getattr(self, 'ranking_compile_mode', 'default')
        self._compiled_ranking_step = torch.compile(ranking_step, mode=mode, fullgraph=True)
        print(f"[PPO] Compiled ranking_step with mode={mode}")

    def _create_ranking_step_fn(self):
        """Factory method to create ranking_step function with fresh policy weight references.

        This allows recompilation of the ranking step with current policy weights,
        avoiding stale weight caching issues with torch.compile fullgraph mode.
        """
        # Capture necessary variables for the closure
        B = self.fixed_batch_size
        H = self.max_depth + 1
        device = self.device
        max_pool = self._ranking_max_pool
        env = self.env.env if hasattr(self.env, 'env') and not isinstance(self.env, EnvVec) else self.env

        def ranking_step(pool: Tensor, pool_size: Tensor):
            """Fused step with slot recycling, history tracking and memory pruning."""
            # Policy - use self._uncompiled_policy for fresh weight access
            obs = {
                'sub_index': self._ranking_current.unsqueeze(1),
                'derived_sub_indices': self._ranking_derived,
                'action_mask': self._ranking_mask.to(torch.uint8),
            }
            logits = self._uncompiled_policy.get_logits(obs)
            masked_logits = logits.masked_fill(~self._ranking_mask, -3.4e38)
            actions = masked_logits.argmax(dim=-1)

            # Compute log probabilities for selected actions
            log_probs = torch.log_softmax(masked_logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1).detach()

            # Build current state dict
            current_state_dict = TensorDict({
                "current_states": self._ranking_current,
                "derived_states": self._ranking_derived,
                "derived_counts": self._ranking_counts,
                "original_queries": self._ranking_original_queries,
                "next_var_indices": self._ranking_next_var,
                "depths": self._ranking_depths,
                "done": self._ranking_done.to(torch.uint8),
                "success": torch.zeros(B, dtype=torch.uint8, device=device),
                "current_labels": torch.ones(B, dtype=torch.long, device=device),
                "history_hashes": self._ranking_history_hashes,
                "history_count": self._ranking_history_count,
                "step_rewards": torch.zeros(B, dtype=torch.float32, device=device),
                "step_dones": torch.zeros(B, dtype=torch.uint8, device=device),
                "step_successes": torch.zeros(B, dtype=torch.uint8, device=device),
                "step_labels": torch.zeros(B, dtype=torch.long, device=device),
                "cumulative_rewards": torch.zeros(B, dtype=torch.float32, device=device),
                "per_env_ptrs": self._ranking_pool_ptr,
                "neg_counters": torch.zeros(B, dtype=torch.int64, device=device),
            }, batch_size=[B], device=device)

            active = ~self._ranking_done
            new_ep_logprob = torch.where(active, self._ranking_ep_logprob + log_probs, self._ranking_ep_logprob)

            # Step Core
            new_obs, new_state = env._step_core(current_state_dict, actions)

            new_current = new_state['current_states']
            new_derived = new_state['derived_states']
            new_counts = new_state['derived_counts']
            new_depths = new_state['depths']
            new_done = new_state['done'].bool()
            new_success = new_state['success'].bool()
            new_history = new_state['history_hashes']
            new_h_count = new_state['history_count']
            new_original = new_state['original_queries']
            new_next_var = new_state['next_var_indices']
            step_dones = new_state['step_dones'].bool()

            # Reset / Recycle Logic
            finished_idx = torch.where(step_dones, self._ranking_pool_ptr, torch.full_like(self._ranking_pool_ptr, -1))
            new_ptr = torch.where(step_dones, self._ranking_pool_ptr + self._ranking_stride, self._ranking_pool_ptr)
            needs_reset = step_dones & (new_ptr < pool_size)

            valid_finish = step_dones & (finished_idx >= 0) & (finished_idx < pool_size)
            ep_logprob_to_store = torch.where(valid_finish, new_ep_logprob, torch.zeros_like(new_ep_logprob))

            # Reset
            safe_idx = new_ptr.clamp(0, max_pool - 1)
            reset_queries_raw = pool[safe_idx]
            reset_labels = torch.ones(B, dtype=torch.long, device=device)
            reset_state = env._reset_from_queries(reset_queries_raw, reset_labels)

            # Merge
            m1 = needs_reset.view(B, 1, 1)
            m3 = needs_reset.view(B, 1, 1, 1)
            mH = needs_reset.view(B, 1).expand(B, H)

            final_current = torch.where(m1, reset_state['current_states'], new_current)
            final_derived = torch.where(m3, reset_state['derived_states'], new_derived)
            final_counts = torch.where(needs_reset, reset_state['derived_counts'], new_counts)
            final_depths = torch.where(needs_reset, reset_state['depths'], new_depths)
            final_done = torch.where(needs_reset, reset_state['done'], new_done)
            final_history = torch.where(mH, reset_state['history_hashes'], new_history)
            final_h_count = torch.where(needs_reset, reset_state['history_count'], new_h_count)
            final_original = torch.where(m1, reset_state['original_queries'], new_original)
            final_next_var = torch.where(needs_reset, reset_state['next_var_indices'], new_next_var)

            final_ep_logprob = torch.where(needs_reset, torch.zeros_like(new_ep_logprob), new_ep_logprob)

            exhausted = (new_ptr >= pool_size) & step_dones
            final_done = torch.where(exhausted, torch.ones_like(final_done), final_done)
            final_ep_logprob = torch.where(exhausted, torch.zeros_like(final_ep_logprob), final_ep_logprob)

            new_mask = self._ranking_arange_S.unsqueeze(0) < final_counts.unsqueeze(1)

            return (final_current, final_derived, final_counts, new_mask,
                    final_depths, final_done, new_ptr, final_history, final_h_count,
                    step_dones, new_success, finished_idx, final_ep_logprob, ep_logprob_to_store, final_original, final_next_var, new_depths)

        return ranking_step

    # -------------------------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------------------------

    def collect_rollouts(self, current_state, current_obs, episode_starts, current_episode_reward, current_episode_length,
                         episode_rewards, episode_lengths, iteration, return_traces=False, on_step_callback=None):
        """Collect experiences using compiled fused step (parity-compatible)."""
        self.policy.eval()
        self.rollout_buffer.reset()
        n_collected, traces = 0, [] if return_traces else None
        
        # NOTE: The standard path below is parity-compatible as shown by test_compiled_rollouts.py
        # The key is proper initialization in learn() with sample_negatives + _reset_from_queries
        # See test_compiled_rollouts.py lines 440-480 for the working pattern
        
        # Standard collect_rollouts path
        state, obs = current_state, current_obs

        # Initialize query indices tracking (per_env_ptrs always present in EnvOptimal state)
        if self.current_query_indices is None and hasattr(self.env, '_per_env_ptrs') and self.env._per_env_ptrs is not None:
            self.current_query_indices = self.env._per_env_ptrs.cpu().numpy()

        with torch.no_grad():
            while n_collected < self.n_steps:
                # Snapshot obs BEFORE policy forward (matches TensorPPO)
                obs_snap = {k: v.clone() for k, v in obs.items()}
                
                if hasattr(self, '_compiled_rollout_step') and self._compiled_rollout_step is not None:
                    torch.compiler.cudagraph_mark_step_begin()
                    obs_in, state_in = {k: v.clone() for k, v in obs.items()}, state.clone()
                    new_obs, new_state, actions, log_probs = self._compiled_rollout_step(obs_in, state_in)
                    values = self._uncompiled_policy.predict_values(obs_snap)
                else:
                    # Uncompiled fallback
                    actions, values, log_probs = self._uncompiled_policy(obs_snap, deterministic=False)
                    new_obs, new_state = self.env._step_and_reset_core(state, actions, self.env._query_pool, self.env._per_env_ptrs)

                self.rollout_buffer.add(
                    sub_index=obs_snap['sub_index'], derived_sub_indices=obs_snap['derived_sub_indices'],
                    action_mask=obs_snap['action_mask'], action=actions, reward=new_state['step_rewards'],
                    episode_start=episode_starts, value=values.flatten(), log_prob=log_probs,
                )

                # Collect traces if requested (for parity testing)
                if return_traces:
                    for idx in range(self.batch_size_env):
                        trace_entry = {
                            "step": n_collected,
                            "env": idx,
                            "pointer": int(self.env._per_env_ptrs[idx]) if hasattr(self.env, '_per_env_ptrs') else None,
                            "query_idx": int(self.current_query_indices[idx]) if self.current_query_indices is not None else None,
                            "state_obs": {
                                "sub_index": obs_snap['sub_index'][idx].cpu().numpy().copy(),
                                "derived_sub_indices": obs_snap['derived_sub_indices'][idx].cpu().numpy().copy(),
                                "action_mask": obs_snap['action_mask'][idx].cpu().numpy().copy(),
                            },
                            "action": int(actions[idx]),
                            "reward": float(new_state['step_rewards'][idx]),
                            "done": bool(new_state['step_dones'][idx]),
                            "value": float(values[idx]),
                            "log_prob": float(log_probs[idx]),
                        }
                        traces.append(trace_entry)

                current_episode_reward += new_state['step_rewards']
                current_episode_length += 1
                n_collected += 1

                # Update per-env pointers from state (critical for query cycling parity)
                self.env._per_env_ptrs.copy_(new_state['per_env_ptrs'])

                # Handle callbacks for done episodes
                done_indices = torch.nonzero(new_state['step_dones']).flatten()
                num_dones = done_indices.numel()
                if num_dones > 0:
                    done_idx_cpu = done_indices.cpu().numpy()
                    batch_rs = current_episode_reward[done_indices].float().cpu().numpy()
                    batch_ls = current_episode_length[done_indices].cpu().numpy().astype(int)
                    episode_rewards.extend(batch_rs.tolist())
                    episode_lengths.extend(batch_ls.tolist())

                    if on_step_callback is not None:
                        # Extract success and labels from step (before reset overwrites them)
                        batch_successes = new_state['step_successes'][done_indices].bool().cpu().numpy()
                        batch_labels = new_state['step_labels'][done_indices].cpu().numpy()
                        on_step_callback(
                            rewards=batch_rs,
                            lengths=batch_ls,
                            done_idx_cpu=done_idx_cpu,
                            current_query_indices=self.current_query_indices,
                            query_labels=self.query_labels,
                            query_depths=self.query_depths,
                            successes=batch_successes,
                            step_labels=batch_labels,  # Actual labels with negative sampling
                        )

                    # Update pointers
                    if self.current_query_indices is not None:
                         # Use per_env_ptrs from new_state (which is the next query index)
                         self.current_query_indices[done_idx_cpu] = new_state['per_env_ptrs'][done_indices].cpu().numpy()
                    
                    # Manual masking is still needed for accumulators
                    current_episode_reward.masked_fill_(new_state['step_dones'].bool(), 0.0)
                    current_episode_length.masked_fill_(new_state['step_dones'].bool(), 0)


                episode_starts = new_state['step_dones'].float()
                state, obs = new_state, new_obs

            last_values = self._uncompiled_policy.predict_values(obs)

        self.rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=new_state['step_dones'].float())
        return state, obs, episode_starts, current_episode_reward, current_episode_length, n_collected * self.batch_size_env, traces


    def train(self, return_traces=False):
        """Update policy from rollout buffer using compiled loss."""
        self.policy.train()
        buffer_size = self.n_steps * self.batch_size_env
        n_batches = buffer_size // self.batch_size
        total = self.n_epochs * n_batches

        pg_losses = torch.zeros(total, device=self.device)
        vl_losses = torch.zeros(total, device=self.device)
        ent_losses = torch.zeros(total, device=self.device)
        clips = torch.zeros(total, device=self.device)
        kls = torch.zeros(total, device=self.device)
        
        # Training traces for parity testing
        train_traces = [] if return_traces else None

        batch_count = 0
        continue_training = True
                
        last_epoch_start = 0
        for epoch in range(self.n_epochs):
            last_epoch_start = batch_count
            for batch in self.rollout_buffer.get(batch_size=self.batch_size):
                torch.compiler.cudagraph_mark_step_begin()
                sub_idx, derived, mask, actions, old_values, old_lp, advantages, returns = batch

                self._train_sub_index.copy_(sub_idx)
                self._train_derived.copy_(derived)
                self._train_mask.copy_(mask)
                self._train_actions.copy_(actions)
                self._train_old_values.copy_(old_values)
                self._train_old_log_probs.copy_(old_lp)
                self._train_returns.copy_(returns)
                if self.normalize_advantage and len(advantages) > 1:
                    self._train_advantages.copy_((advantages - advantages.mean()) / (advantages.std() + 1e-8))
                else:
                    self._train_advantages.copy_(advantages)

                with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                    metrics = self.loss_module(
                        self._train_sub_index, self._train_derived, self._train_mask,
                        self._train_actions, self._train_advantages, self._train_returns,
                        self._train_old_log_probs, self._train_old_values,
                        self.clip_range, self.clip_range_vf or 0.0, self.ent_coef, self.vf_coef,
                    )

                loss = metrics[0]
                policy_loss = metrics[1]
                value_loss = metrics[2]
                entropy_loss = metrics[3]
                approx_kl_div = metrics[4]
                clip_fraction = metrics[5]
                
                pg_losses[batch_count] = policy_loss.detach().clone()
                vl_losses[batch_count] = value_loss.detach().clone()
                ent_losses[batch_count] = entropy_loss.detach().clone()
                kls[batch_count] = approx_kl_div.detach().clone()
                clips[batch_count] = clip_fraction.detach().clone()
                
                # Collect training traces if requested
                if return_traces:
                    train_traces.append({
                        "epoch": epoch,
                        "batch_size": len(actions),
                        "policy_loss": policy_loss.item(),
                        "value_loss": value_loss.item(),
                        "entropy_loss": entropy_loss.item(),
                        "clip_fraction": clip_fraction.item(),
                    })
                
                batch_count += 1

                # Check KL divergence BEFORE optimizer step (matching TensorPPO/SB3)
                if self.target_kl and approx_kl_div.item() > 1.5 * self.target_kl:
                    if self.verbose:
                        print(f"[PPO] Early stopping at step {epoch} due to reaching max kl: {approx_kl_div.item():.2f}")
                    continue_training = False
                    break
                
                # Apply optimizer step only if KL check passed
                self.optimizer.zero_grad(set_to_none=False)
                loss.backward()
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm, foreach=True)
                self.optimizer.step()

            if not continue_training:
                break
            
            # Print epoch stats (matching TensorPPO)
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.n_epochs}. ")
                mean_pg = pg_losses[:batch_count].mean().item()
                mean_val = vl_losses[:batch_count].mean().item()
                mean_ent = ent_losses[:batch_count].mean().item()
                mean_kl = kls[:batch_count].mean().item()
                mean_clip = clips[:batch_count].mean().item()
                print(f"Losses: total {loss.item():.5f}, value {mean_val:.5f}, "
                      f"policy {mean_pg:.5f}, entropy {mean_ent:.5f}, "
                      f"approx_kl {mean_kl:.5f} clip_fraction {mean_clip:.5f}. ")

        with torch.no_grad():
            values = self.rollout_buffer.values.flatten()
            returns = self.rollout_buffer.returns.flatten()
            ev = explained_variance(values, returns)
        
        result = {
            "policy_loss": pg_losses[:batch_count].mean().item(),
            "value_loss": vl_losses[:batch_count].mean().item(),
            "entropy": -ent_losses[:batch_count].mean().item(),
            "clip_fraction": clips[:batch_count].mean().item(),
            "approx_kl": kls[last_epoch_start:batch_count].mean().item(),
            "explained_var": ev.item(),
        }
        
        if return_traces:
            result["traces"] = train_traces
        
        return result

    def learn(self, total_timesteps, queries=None, reset_num_timesteps=True, on_iteration_start_callback=None, on_step_callback=None, return_traces=False):
        """Main PPO training loop."""
        if reset_num_timesteps:
            self.num_timesteps = 0
        else:
            total_timesteps += self.num_timesteps

        if self.callback and hasattr(self.callback, 'on_training_start'):
            self.callback.on_training_start(total_timesteps=total_timesteps)

        self.env.train()
        # Handle flexible reset return (EnvOptimal returns Tuple, Wrapper returns TensorDict)
        reset_res = self.env.reset()
        if isinstance(reset_res, tuple):
            obs, state = reset_res
        else:
            obs, state = reset_res, getattr(self.env, "_state", None)
        
        ep_starts = torch.ones(self.batch_size_env, dtype=torch.float32, device=self.device)
        curr_ep_rew = torch.zeros(self.batch_size_env, dtype=torch.float32, device=self.device)
        curr_ep_len = torch.zeros(self.batch_size_env, dtype=torch.long, device=self.device)
        ep_rews, ep_lens = [], []
        iteration = 0

        while self.num_timesteps < total_timesteps:
            iteration += 1
            if on_iteration_start_callback:
                on_iteration_start_callback(iteration, self.num_timesteps)

            step_cb = on_step_callback
            if self.callback is not None:
                if hasattr(self.callback, 'prepare_batch_infos'):
                    step_cb = self.callback.prepare_batch_infos
                else:
                    step_cb = self.callback.on_step

            # Call invocation of on_iteration_start on the callback manager if it exists
            if self.callback and hasattr(self.callback, 'on_iteration_start'):
                 self.callback.on_iteration_start(iteration, self.num_timesteps)

            # Collect rollouts
            rollout_start_time = time.time()
            result = self.collect_rollouts(state, obs, ep_starts, curr_ep_rew, curr_ep_len, ep_rews, ep_lens, iteration, return_traces, step_cb)
            state, obs, ep_starts, curr_ep_rew, curr_ep_len, n_steps, _ = result
            state = state.clone()
            obs = {k: v.clone() for k, v in obs.items()}
            self.num_timesteps += n_steps

            rollout_time = time.time() - rollout_start_time
            if self.verbose:
                print(f"[PPO] Rollout collected in {rollout_time:.2f}s. FPS: {n_steps/rollout_time:.2f}")

            train_start_time = time.time()
            train_metrics = self.train(return_traces)
            self.last_train_metrics = train_metrics
            train_time = time.time() - train_start_time
            if self.verbose:
                print(f"[PPO] Training completed in {train_time:.2f}s")
            
            # Callback: End of iteration (logs metrics)
            if self.callback:
                # Prepare locals for callback manager
                # Manager expects 'iteration', 'total_steps_done', 'train_metrics'
                total_steps_done = self.num_timesteps
                self.callback(locals(), globals())
        
        if self.callback and hasattr(self.callback, 'on_training_end'):
            self.callback.on_training_end()

        # Retrieve final metrics from callback manager if available, else fallback to instance variable
        final_metrics = getattr(self.callback, 'last_metrics', self.last_train_metrics) if self.callback else self.last_train_metrics

        return {'num_timesteps': self.num_timesteps, 'episode_rewards': ep_rews, 'episode_lengths': ep_lens, 'last_train_metrics': final_metrics}

    # -------------------------------------------------------------------------
    # EVALUATION
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_policy(self, queries: Tensor, max_steps: Optional[int] = None, deterministic: bool = True):
        """Standard deterministic evaluation loop for a batch of queries."""
        max_steps = max_steps or self.max_depth
        
        torch.compiler.cudagraph_mark_step_begin()
        env = self.env.env if hasattr(self.env, 'env') and not isinstance(self.env, EnvVec) else self.env
        obs, state = env.reset(queries)
        
        # Buffer-copy pattern for CUDA graph stability
        obs, state = obs.clone(), state.clone()

        for _ in range(max_steps):
            torch.compiler.cudagraph_mark_step_begin()
            obs, state = self._compiled_eval_step(obs, state)
            obs, state = obs.clone(), state.clone()

        success = state.get('success', state.get('is_success'))
        return success, state['depths']

    def _pad_queries(self, queries: Tensor) -> Tuple[Tensor, int]:
        B = queries.shape[0]
        if B >= self.fixed_batch_size:
            return queries[:self.fixed_batch_size], min(B, self.fixed_batch_size)
        self._eval_padded_buffer[:B] = queries
        if B > 0:
            self._eval_padded_buffer[B:] = queries[-1]
        return self._eval_padded_buffer, B

    def _setup_ranking_pool(self, queries: Tensor, sampler, n_corruptions, modes: Sequence[str]):
        """Consolidated pool setup for corruption ranking."""
        N, device = queries.shape[0], self.device
        B, H = self.fixed_batch_size, self._ranking_max_history
        pad = self.env.padding_idx
        
        pools = []
        for mode in modes:
            neg = sampler.corrupt(queries, num_negatives=n_corruptions, mode=mode, device=device)
            cands = torch.cat([queries.unsqueeze(1), neg], dim=1)
            pools.append(cands.transpose(0, 1).contiguous().view(-1, 3))
        
        new_pool = torch.cat(pools, dim=0)
        new_size = new_pool.size(0)
        
        if new_size > self._ranking_max_pool:
            raise ValueError(f"Pool size {new_size} exceeds max {self._ranking_max_pool}")
        
        self._ranking_pool[:new_size].copy_(new_pool)
        self._ranking_pool[new_size:].fill_(pad)
        self._ranking_pool_size.fill_(new_size)
        self._ranking_result_buf.zero_()
        self._ranking_result_logprob.zero_()
        self._ranking_result_depths.zero_()
        self._ranking_result_success.zero_()
        self._ranking_result_rewards.zero_()
        self._ranking_stride.fill_(N)
        
        # Initialize loop buffers using env._reset_from_queries for proper derived computation
        init_idx = torch.arange(B, device=device).clamp(max=max(0, N - 1))
        initial_queries_raw = self._ranking_pool[init_idx]  # [B, 3]

        # Use env's reset logic to get proper initial state with correct derived states
        env = self.env.env if hasattr(self.env, 'env') and not isinstance(self.env, EnvVec) else self.env
        init_state = env._reset_from_queries(initial_queries_raw, torch.ones(B, dtype=torch.long, device=device))

        self._ranking_current.copy_(init_state['current_states'])
        self._ranking_derived.copy_(init_state['derived_states'])
        self._ranking_counts.copy_(init_state['derived_counts'])
        self._ranking_original_queries.copy_(init_state['original_queries'])
        self._ranking_mask.copy_(self._ranking_arange_S.unsqueeze(0) < init_state['derived_counts'].unsqueeze(1))
        self._ranking_depths.zero_()
        self._ranking_done.zero_()
        if N < B: self._ranking_done[N:].fill_(True)
        self._ranking_pool_ptr.copy_(torch.arange(B, device=device))
        self._ranking_history_hashes.copy_(init_state['history_hashes'])
        self._ranking_history_count.copy_(init_state['history_count'])
        self._ranking_next_var.copy_(init_state['next_var_indices'])
        self._ranking_ep_logprob.zero_()
        self._ranking_result_logprob.zero_()

    def _collect_chunk_stats(self, pool_size, corruption_modes, device, all_stats, CQ, K):
        """Helper to collect detailed stats for a chunk (GPU-only, defer CPU transfer)."""
        # Keep all data on GPU until final aggregation to avoid sync overhead
        chunk_depths = self._ranking_result_depths[:pool_size].clone()
        chunk_proven = self._ranking_result_buf[:pool_size].clone()
        # Rewards = proven.float() (binary success converted to float)

        off = 0
        for mode_i, _ in enumerate(corruption_modes):
            pos_start, pos_end = off, off + CQ
            neg_start, neg_end = off + CQ, off + CQ * K

            # Keep tensors on GPU (defer .cpu() to aggregation)
            # Note: rewards = proven.float() since success is binary (0 or 1)
            if CQ > 0:
                proven_pos = chunk_proven[pos_start:pos_end].float()
                all_stats['len_pos'].append(chunk_depths[pos_start:pos_end].float())
                all_stats['rew_pos'].append(proven_pos)
                all_stats['pvn_pos'].append(proven_pos)

            if neg_end > neg_start:
                proven_neg = chunk_proven[neg_start:neg_end].float()
                all_stats['len_neg'].append(chunk_depths[neg_start:neg_end].float())
                all_stats['rew_neg'].append(proven_neg)
                all_stats['pvn_neg'].append(proven_neg)

            off += CQ * K

    def _aggregate_metrics(self, all_stats, query_depths, N, chunk_queries, corruption_modes, all_ranks=None):
        """Helper to aggregate all collected metrics."""
        res = {}

        # Concat detailed stats
        full_stats = {k: torch.cat(v) if v else torch.tensor([]) for k, v in all_stats.items()}

        # DEBUG

        # Global means
        global_means = {}
        for k, v in full_stats.items():
            global_means[k] = v.mean().item() if v.numel() > 0 else 0.0
        res['proven_pos'] = global_means.get('pvn_pos', 0)
        res['proven_neg'] = global_means.get('pvn_neg', 0)
        res['len_pos']    = global_means.get('len_pos', 0)
        res['len_neg']    = global_means.get('len_neg', 0)
        res['reward_label_pos'] = global_means.get('rew_pos', 0)
        res['reward_label_neg'] = global_means.get('rew_neg', 0)

        # Depth-based stats for POSITIVES
        if query_depths is not None:
            qd = query_depths.cpu()
            expanded_depths_list = []
            for start in range(0, N, chunk_queries):
                end = min(start + chunk_queries, N)
                chunk_d = qd[start:end]
                for _ in corruption_modes:
                    expanded_depths_list.append(chunk_d)

            if expanded_depths_list:
                expanded_depths = torch.cat(expanded_depths_list)
                unique_d = torch.unique(qd).tolist()

                for d in unique_d:
                    mask = (expanded_depths == d)
                    if mask.any():
                        res[f"len_d_{d}_pos"] = full_stats['len_pos'][mask].mean().item()
                        res[f"proven_d_{d}_pos"] = full_stats['pvn_pos'][mask].mean().item()
                        res[f"reward_d_{d}_pos"] = full_stats['rew_pos'][mask].mean().item()

        # Compute Ranking Metrics (MRR, Hits@K)
        if all_ranks is not None:
            def compute_stats(r):
                if r.numel() == 0: return {}
                return {
                    'MRR': (1.0 / r.float()).mean().item(),
                    'Hits@1': (r <= 1).float().mean().item(),
                    'Hits@3': (r <= 3).float().mean().item(),
                    'Hits@10': (r <= 10).float().mean().item(),
                }

            global_metrics = {'MRR': [], 'Hits@1': [], 'Hits@3': [], 'Hits@10': []}
            for m in corruption_modes:
                if all_ranks[m]:
                    ranks = torch.cat(all_ranks[m])
                    m_stats = compute_stats(ranks)
                    for k, v in m_stats.items():
                        global_metrics[k].append(v)

            for k in global_metrics.keys():
                if global_metrics[k]:
                    res[k] = np.mean(global_metrics[k])
                else:
                    res[k] = 0.0
                        
        return res

    def _format_kge_atoms(self, atoms: Tensor, index_manager: Any) -> List[str]:
        """Convert atom indices to canonical strings for KGE scoring."""
        idx2pred = index_manager.idx2predicate
        idx2const = index_manager.idx2constant
        n_constants = index_manager.constant_no
        pad_idx = index_manager.padding_idx
        return [
            atom_to_str(atom, idx2pred, idx2const, n_constants, padding_idx=pad_idx)
            for atom in atoms
        ]

    def _score_kge_candidates(
        self,
        queries: Tensor,
        valid_mask: Tensor,
        *,
        device: torch.device,
    ) -> Tensor:
        """Score candidate triples with the KGE engine.

        Args:
            queries: [Q, K, 3] candidate triples.
            valid_mask: [Q, K] validity mask for candidates.
            device: target device for output tensor.

        Returns:
            Tensor of shape [Q, K] with KGE scores in [0, 1].
        """
        # [Q, K, 3] -> [Q*K, 3]
        flat_queries = queries.reshape(-1, 3)
        # [Q, K] -> [Q*K]
        flat_valid = valid_mask.reshape(-1)
        flat_valid_cpu = flat_valid.detach().cpu()
        if not flat_valid_cpu.any():
            return torch.zeros(queries.shape[0], queries.shape[1], device=device)

        flat_queries_cpu = flat_queries.detach().cpu()[flat_valid_cpu]
        atom_strs = self._format_kge_atoms(flat_queries_cpu, self.kge_index_manager)
        scores_list = self.kge_inference_engine.predict_batch(atom_strs)
        scores_cpu = torch.zeros(flat_queries.shape[0], dtype=torch.float32)
        scores_cpu[flat_valid_cpu] = torch.as_tensor(scores_list, dtype=torch.float32)
        return scores_cpu.view(queries.shape[0], queries.shape[1]).to(device=device)

    def evaluate(self, queries: Tensor, sampler, n_corruptions: int = 50, corruption_modes: Sequence[str] = ('head', 'tail'), *,
                 chunk_queries: int = None, verbose: bool = False, deterministic: bool = True,
                 query_depths: Optional[Tensor] = None, debug: bool = False):
        """Fast corruption ranking evaluation with slot recycling (production use)."""
        self._uncompiled_policy.eval()
        N, device = queries.shape[0], self.device
        # If n_corruptions is None (exhaustive), use sampler info for safer K estimate
        # Use a better K estimate for exhaustive ranking to guide chunking
        if n_corruptions is None:
             # Exhaustive: pool size per query is roughly num_entities (or max_pool_len) * nm
             pool_per_query = (sampler.max_pool_len if sampler.max_pool_len > 0 else sampler.num_entities)
             K_est = pool_per_query + 1
        else:
             K_est = 1 + n_corruptions
             
        K, nm = K_est, len(corruption_modes)

        # Auto-chunk to fit pool AND batch size
        # CRITICAL: chunk_queries MUST be <= fixed_batch_size, otherwise only first B queries get processed!
        max_q_pool = self._ranking_max_pool // (K * nm)
        max_q_batch = self.fixed_batch_size  # Each env processes one query per chunk
        max_q = min(max_q_pool, max_q_batch)
        chunk_queries = min(chunk_queries or max_q, max_q)
        
        if chunk_queries < 1: raise ValueError("Queries/modes too large for pool.")
        
        all_ranks = {mode: [] for mode in corruption_modes}
        
        # Accumulators for detailed stats
        all_stats = {
            'len_pos': [], 'len_neg': [], 
            'rew_pos': [], 'rew_neg': [], 
            'pvn_pos': [], 'pvn_neg': []
        }
        tie_generator = torch.Generator(device=device)
        tie_generator.manual_seed(self.ranking_tie_seed)

        for start in range(0, N, chunk_queries):
            t_start = time.time()
            if verbose:
                print(f"Chunk {start}-{min(start + chunk_queries, N)}")
            end = min(start + chunk_queries, N)
            chunk = queries[start:end]
            CQ = end - start
            
            self._setup_ranking_pool(chunk, sampler, n_corruptions, corruption_modes)
            # Compute pool_size without GPU sync - it's always CQ * K * nm
            pool_size = CQ * K * nm  # Known from inputs, no .item() needed

            if debug:
                print(f"\n[DEBUG] Chunk {start}-{end}, pool_size={pool_size}")
                print(f"[DEBUG] Initial: counts={self._ranking_counts[:8].tolist()}, done={self._ranking_done[:8].tolist()}")
                print(f"[DEBUG] Initial mask sum (first 8 envs): {self._ranking_mask[:8].sum(1).tolist()}")

            # Ensure fresh weights before ranking loop (signal to torch.compile)
            torch.compiler.cudagraph_mark_step_begin()

            max_steps = (pool_size // self.fixed_batch_size + 2) * self.max_depth
            debug_step = 0

            # Check termination only at 50%, 100%, and 70% of max_steps to minimize GPU syncs
            check_steps = {max_steps // 2, max_steps - 1, (max_steps * 7) // 10}
            for step_i in range(max_steps):
                torch.compiler.cudagraph_mark_step_begin()

                # Always use original compiled step (logits inside)
                (new_cur, new_der, new_cnt, new_mask, new_dep, new_done, new_ptr,
                    new_h, new_hc, _, success, indices, new_ep_logprob, logprob_to_store, new_orig, new_next_var, step_depths) = self._compiled_ranking_step(self._ranking_pool, self._ranking_pool_size)

                if debug and debug_step < 5:
                    print(f"[DEBUG] Step {debug_step}: new_cnt[:8]={new_cnt[:8].tolist()}, new_done[:8]={new_done[:8].tolist()}")
                    print(f"[DEBUG]   success[:8]={success[:8].tolist()}, step_depths[:8]={step_depths[:8].tolist()}")
                debug_step += 1

                # Use copy_ for in-place update (required for CUDA graphs) but batch 4 tensors at a time
                self._ranking_current.copy_(new_cur)
                self._ranking_derived.copy_(new_der)
                self._ranking_counts.copy_(new_cnt)
                self._ranking_mask.copy_(new_mask)
                self._ranking_depths.copy_(new_dep)
                self._ranking_done.copy_(new_done)
                self._ranking_pool_ptr.copy_(new_ptr)
                self._ranking_history_hashes.copy_(new_h)
                self._ranking_history_count.copy_(new_hc)
                self._ranking_ep_logprob.copy_(new_ep_logprob)
                self._ranking_original_queries.copy_(new_orig)
                self._ranking_next_var.copy_(new_next_var)

                # Result record - only update valid indices
                v_idx = (indices >= 0) & (indices < pool_size)
                clamped_indices = indices.clamp(0, pool_size-1)

                # Scatter with masked values
                final_buf = torch.where(v_idx, success, self._ranking_result_buf[clamped_indices])
                final_logprob = torch.where(v_idx, logprob_to_store, self._ranking_result_logprob[clamped_indices])
                final_depths = torch.where(v_idx, step_depths, self._ranking_result_depths[clamped_indices])

                self._ranking_result_buf.scatter_(0, clamped_indices, final_buf)
                self._ranking_result_logprob.scatter_(0, clamped_indices, final_logprob)
                self._ranking_result_depths.scatter_(0, clamped_indices, final_depths)

                # Check termination only at strategic points
                if step_i in check_steps and self._ranking_done.all():
                    break

            # --- Collect Detailed Stats for this Chunk ---
            self._collect_chunk_stats(pool_size, corruption_modes, device, all_stats, CQ, K)

            # Ranking computation per mode (using accumulated log probs)
            offset = 0
            for mode in corruption_modes:
                # Get success mask and log probs for this mode's queries
                success_t = self._ranking_result_buf[offset:offset + CQ * K].view(K, CQ)
                success = success_t.t().contiguous()  # [CQ, K]

                logprob_t = self._ranking_result_logprob[offset:offset + CQ * K].view(K, CQ)
                logprobs = logprob_t.t().contiguous()  # [CQ, K]
                
                # Identify valid candidates (not padding)
                # Check queries in pool: if (r,h,t) has 0, it's padding
                # query layout: [r, h, t] or similar. DataHandler says (r, h, t). Padding is 0.
                queries_t = self._ranking_pool[offset:offset + CQ * K].view(K, CQ, 3) 
                # [K, CQ, 3] -> [CQ, K, 3]
                queries_batch = queries_t.permute(1, 0, 2)
                # Valid candidates: predicate/head/tail are non-padding
                is_valid = (queries_batch[:, :, 0] > 0) & (queries_batch[:, :, 1] > 0) & (queries_batch[:, :, 2] > 0)

                use_kge = (
                    self.kge_inference
                    and self.kge_inference_engine is not None
                    and self.kge_index_manager is not None
                )

                if use_kge:
                    kge_scores = self._score_kge_candidates(
                        queries_batch,
                        is_valid,
                        device=device,
                    )
                    # kge_scores: [CQ, K]
                    kge_log_scores = torch.log(kge_scores.clamp(min=self._kge_log_eps))
                    
                    if self.kge_only_eval:
                        # KGE-only mode: use pure KGE scores (matches paper evaluation)
                        scores = kge_log_scores
                    else:
                        # Hybrid mode: combine KGE with RL binary success (not logprobs)
                        # Using binary success (+1 for proven) instead of negative logprobs
                        scores = self.kge_eval_kge_weight * kge_log_scores
                        if self.kge_inference_success:
                            # Add bonus only for proven queries (binary: 1 if proven, 0 if not)
                            scores = torch.where(
                                success,
                                scores + self.kge_eval_rl_weight,  # Binary +1 bonus for proven
                                scores - self.kge_fail_penalty,    # Penalty for failed
                            )
                        else:
                            scores = scores + self.kge_eval_rl_weight * logprobs
                            scores = torch.where(success, scores, scores - self.kge_fail_penalty)
                else:
                    # Successful proofs keep their log prob, failed get penalty
                    scores = torch.where(success, logprobs, logprobs - self.kge_fail_penalty)

                # Mask out invalid padding candidates with a very low score so they rank last
                scores = scores.masked_fill(~is_valid, -1e9)

                pos, neg = scores[:, 0:1], scores[:, 1:]
                rnd = torch.rand((CQ, K), generator=tie_generator, device=device)  # [CQ, K]
                better = (neg > pos)
                tied = (neg == pos) & (rnd[:, 1:] > rnd[:, 0:1])
                all_ranks[mode].append(1 + better.sum(1) + tied.sum(1))
                offset += CQ * K
            if verbose:
                elapsed = time.time() - t_start
                print(f" Took {elapsed:.2f} seconds. ms/cand: {1000 * elapsed / (CQ * K)}")

        # Metric Aggregation
        res = self._aggregate_metrics(all_stats, query_depths, N, chunk_queries, corruption_modes, all_ranks=all_ranks)
        
        return res
