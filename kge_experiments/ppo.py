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

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

from rollout import RolloutBuffer
from env import EnvOptimal, EnvObs, EnvState, TensorDictEnvWrapper


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

class PPOOptimal:
    """Optimal PPO with CUDA graph support and separate train/eval compilation."""

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------

    def __init__(self, policy: nn.Module, env: EnvOptimal, config, **kwargs):
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
        self.parity = kwargs.get('parity', config.parity)
        self.eval_only = kwargs.get('eval_only', getattr(config, 'eval_only', False))
        self._compile_mode = kwargs.get('compile_mode', 'reduce-overhead')
        self._fixed_batch_size = kwargs.get('fixed_batch_size', getattr(config, 'fixed_batch_size', None))

        # IMPORTANT: Keep uncompiled policy reference for fused steps
        self._uncompiled_policy = policy

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
                parity=self.parity, batch_size=self.batch_size,
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
            self._setup_fused_eval_step()

        self.num_timesteps = 0
        self.callback = None  # Disabled for optimal version
        self.current_query_indices = None
        print(f"[PPOOptimal] Initialized with device={self.device}, batch_size_env={self.batch_size_env}")

    @property
    def fixed_batch_size(self) -> int:
        return self._fixed_batch_size or self.batch_size_env

    # -------------------------------------------------------------------------
    # COMPILATION
    # -------------------------------------------------------------------------

    def _compile_all(self):
        """Compile all functions using uncompiled policy in fused steps."""
        # In parity mode, use uncompiled versions for exact reproducibility
        if self.parity:
            self.loss_module = PPOLossModule(self._uncompiled_policy)
            self._compiled_policy_fn = self._uncompiled_policy.get_logits
            # Keep self.policy as the uncompiled policy
            # (already set: self.policy = policy in __init__)
            print("[PPOOptimal] Parity mode - skipping compilation")
            return

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

        print("[PPOOptimal] Compilation complete")

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
        env = self.env

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
        print("[PPOOptimal] Fused rollout step compiled")

    def _setup_fused_eval_step(self):
        """Fused policy + env step for eval (deterministic, no log_probs).
        
        CRITICAL: Uses self._uncompiled_policy to avoid nested compilation!
        """
        if hasattr(self, '_compiled_eval_step'):
            return
            
        policy = self._uncompiled_policy  # Uncompiled!
        env = self.env

        def fused_eval(obs, state):
            # Policy forward
            logits = policy.get_logits(obs)
            masked = logits.masked_fill(obs['action_mask'] == 0, -3.4e38)
            
            # Deterministic action (argmax)
            actions = masked.argmax(dim=-1)
            
            # Environment step (no auto-reset for eval)
            new_obs, new_state = env._step_core(state, actions)
            
            return new_obs, new_state

        self._compiled_eval_step = torch.compile(fused_eval, mode=self._compile_mode, fullgraph=True)
        print("[PPOOptimal] Fused eval step compiled")

    # -------------------------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------------------------

    def collect_rollouts(self, current_state, current_obs, episode_starts, current_episode_reward, current_episode_length,
                         episode_rewards, episode_lengths, iteration, return_traces=False, on_step_callback=None):
        """Collect experiences using compiled fused step."""
        self.policy.eval()
        self.rollout_buffer.reset()
        n_collected, traces = 0, [] if return_traces else None
        state, obs = current_state, current_obs

        # Initialize query indices tracking (per_env_ptrs always present in EnvOptimal state)
        if self.current_query_indices is None and hasattr(self.env, '_per_env_ptrs') and self.env._per_env_ptrs is not None:
            self.current_query_indices = self.env._per_env_ptrs.cpu().numpy()

        with torch.no_grad():
            while n_collected < self.n_steps:
                obs_snap = {k: v.clone() for k, v in obs.items()}
                values = self._uncompiled_policy.predict_values(obs)
                
                torch.compiler.cudagraph_mark_step_begin()
                obs_in, state_in = {k: v.clone() for k, v in obs.items()}, state.clone()
                
                # Use compiled step if available, otherwise fallback to uncompiled
                if hasattr(self, '_compiled_rollout_step') and self._compiled_rollout_step is not None:
                    new_obs, new_state, actions, log_probs = self._compiled_rollout_step(obs_in, state_in)
                else:
                    # Uncompiled fallback for parity tests
                    # NOTE: log_probs NOT masked for done states to match TensorPPO
                    logits = self._uncompiled_policy.get_logits(obs_in)
                    masked = logits.masked_fill(obs_in['action_mask'] == 0, -3.4e38)
                    probs = torch.softmax(masked, dim=-1)
                    actions = torch.multinomial(probs, 1).squeeze(-1)
                    log_probs = torch.log_softmax(masked, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
                    new_obs, new_state = self.env._step_and_reset_core(state_in, actions, self.env._query_pool, self.env._per_env_ptrs)

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
                # Use copy_() to preserve CUDA graph compatibility
                self.env._per_env_ptrs.copy_(new_state['per_env_ptrs'])

                done_mask = new_state['step_dones'].bool()
                if done_mask.any():
                    # Use masked_select for efficiency (avoids nonzero call)
                    rews = current_episode_reward[done_mask].float().cpu().numpy()
                    lens = current_episode_length[done_mask].cpu().numpy().astype(int)
                    episode_rewards.extend(rews.tolist())
                    episode_lengths.extend(lens.tolist())
                    current_episode_reward.masked_fill_(done_mask, 0.0)
                    current_episode_length.masked_fill_(done_mask, 0)
                    if self.current_query_indices is not None:
                        done_idx_cpu = done_mask.nonzero(as_tuple=True)[0].cpu().numpy()
                        self.current_query_indices[done_idx_cpu] = new_state['per_env_ptrs'][done_mask].cpu().numpy()

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

        batch_count = 0
        for epoch in range(self.n_epochs):
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
                pg_losses[batch_count] = metrics[1].detach().clone()
                vl_losses[batch_count] = metrics[2].detach().clone()
                ent_losses[batch_count] = metrics[3].detach().clone()
                kls[batch_count] = metrics[4].clone()
                clips[batch_count] = metrics[5].detach().clone()
                batch_count += 1

                if self.target_kl and kls[batch_count - 1].item() > 1.5 * self.target_kl:
                    break

                self.optimizer.zero_grad(set_to_none=False)
                loss.backward()
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm, foreach=True)
                self.optimizer.step()

        with torch.no_grad():
            ev = explained_variance(self.rollout_buffer.values, self.rollout_buffer.returns)
        return {
            "policy_loss": pg_losses[:batch_count].mean().item(),
            "value_loss": vl_losses[:batch_count].mean().item(),
            "entropy": -ent_losses[:batch_count].mean().item(),
            "clip_fraction": clips[:batch_count].mean().item(),
            "approx_kl": kls[:batch_count].mean().item(),
            "explained_var": ev.item(),
        }

    def learn(self, total_timesteps, queries=None, reset_num_timesteps=True, on_iteration_start_callback=None, on_step_callback=None, return_traces=False):
        """Main PPO training loop."""
        if reset_num_timesteps:
            self.num_timesteps = 0
        else:
            total_timesteps += self.num_timesteps

        self.env.train()
        obs, state = self.env.reset()
        ep_starts = torch.ones(self.batch_size_env, dtype=torch.float32, device=self.device)
        curr_ep_rew = torch.zeros(self.batch_size_env, dtype=torch.float32, device=self.device)
        curr_ep_len = torch.zeros(self.batch_size_env, dtype=torch.long, device=self.device)
        ep_rews, ep_lens = [], []
        iteration = 0

        while self.num_timesteps < total_timesteps:
            iteration += 1
            if on_iteration_start_callback:
                on_iteration_start_callback(iteration, self.num_timesteps)

            result = self.collect_rollouts(state, obs, ep_starts, curr_ep_rew, curr_ep_len, ep_rews, ep_lens, iteration, return_traces, on_step_callback)
            state, obs, ep_starts, curr_ep_rew, curr_ep_len, n_steps, _ = result
            state = state.clone()
            self.num_timesteps += n_steps

            train_metrics = self.train(return_traces)
            self.last_train_metrics = train_metrics

        return {'num_timesteps': self.num_timesteps, 'episode_rewards': ep_rews, 'episode_lengths': ep_lens, 'last_train_metrics': self.last_train_metrics}

    # -------------------------------------------------------------------------
    # EVALUATION
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_policy(self, queries: Tensor, max_steps: Optional[int] = None, deterministic: bool = True):
        """Evaluate queries with compiled step loop."""
        max_steps = max_steps or self.max_depth
        self._setup_fused_eval_step()

        torch.compiler.cudagraph_mark_step_begin()
        obs, state = self.env.reset(queries)
        obs, state = obs.clone(), state.clone()

        for _ in range(max_steps):
            torch.compiler.cudagraph_mark_step_begin()
            obs, state = self._compiled_eval_step(obs, state)
            obs, state = obs.clone(), state.clone()

        self._eval_result_success[:] = state['success']
        self._eval_result_depths[:] = state['depths']
        return self._eval_result_success, self._eval_result_depths

    def _pad_queries(self, queries: Tensor) -> Tuple[Tensor, int]:
        B = queries.shape[0]
        if B >= self.fixed_batch_size:
            return queries[:self.fixed_batch_size], min(B, self.fixed_batch_size)
        self._eval_padded_buffer[:B] = queries
        if B > 0:
            self._eval_padded_buffer[B:] = queries[-1]
        return self._eval_padded_buffer, B

    # -------------------------------------------------------------------------
    # V10-STYLE EVALUATION WITH SLOT RECYCLING
    # -------------------------------------------------------------------------

    def _setup_v10_eval_buffers(self):
        """Initialize V10-style evaluation buffers including history for memory pruning."""
        if hasattr(self, '_v10_current'):
            return  # Already initialized
            
        B = self.fixed_batch_size
        A = self.padding_atoms
        S = self.padding_states
        H = self.max_depth + 1  # History size matches env.max_history_size
        device = self.device
        pad = self.env.padding_idx
        
        # Persistent state buffers
        self._v10_current = torch.full((B, A, 3), pad, dtype=torch.long, device=device)
        self._v10_derived = torch.full((B, S, A, 3), pad, dtype=torch.long, device=device)
        self._v10_counts = torch.zeros(B, dtype=torch.long, device=device)
        self._v10_mask = torch.zeros(B, S, dtype=torch.bool, device=device)
        self._v10_depths = torch.zeros(B, dtype=torch.long, device=device)
        self._v10_done = torch.zeros(B, dtype=torch.bool, device=device)
        self._v10_pool_ptr = torch.zeros(B, dtype=torch.long, device=device)
        
        # History tracking buffers for memory pruning
        self._v10_history_hashes = torch.zeros(B, H, dtype=torch.long, device=device)
        self._v10_history_count = torch.zeros(B, dtype=torch.long, device=device)
        self._v10_max_history = H
        
        # Pool buffers - 1M entries = ~24MB, handles large eval sets
        max_pool = 1_000_000
        self._v10_max_pool = max_pool
        self._v10_pool = torch.zeros(max_pool, 3, dtype=torch.long, device=device)
        self._v10_pool_size = torch.tensor(0, dtype=torch.long, device=device)
        self._v10_result_buf = torch.zeros(max_pool, dtype=torch.bool, device=device)
        
        # Constants
        self._v10_stride = torch.tensor(B, dtype=torch.long, device=device)
        self._v10_arange_B = torch.arange(B, device=device)
        self._v10_arange_S = torch.arange(S, device=device)
        self._v10_N = 0
        self._v10_K = 0

    def _compile_v10_eval_step(self):
        """Compile V10-style evaluation step with full history tracking and memory pruning."""
        if hasattr(self, '_compiled_v10_step'):
            return
            
        self._setup_v10_eval_buffers()
        
        policy = self._uncompiled_policy
        env = self.env  # Use env for get_derived_simple and hash computation
        B, S, A = self.fixed_batch_size, self.padding_states, self.padding_atoms
        H = self._v10_max_history
        pad = self.env.padding_idx
        max_depth = self.max_depth
        device = self.device
        max_pool = self._v10_max_pool
        true_pred_idx = self.env.true_pred_idx
        false_pred_idx = self.env.false_pred_idx
        end_pred_idx = self.env.end_pred_idx
        mask_fill_val = -3.4e38
        
        # References to persistent buffers
        current = self._v10_current
        derived = self._v10_derived
        counts = self._v10_counts
        mask = self._v10_mask
        depths = self._v10_depths
        done = self._v10_done
        pool_ptr = self._v10_pool_ptr
        history_hashes = self._v10_history_hashes
        history_count = self._v10_history_count
        arange_B = self._v10_arange_B
        arange_S = self._v10_arange_S
        stride_tensor = self._v10_stride
        
        def step_fn(pool: Tensor, pool_size: Tensor):
            """V10-style step with full history tracking and memory pruning."""
            # 1. Policy forward
            obs = {
                'sub_index': current.unsqueeze(1),
                'derived_sub_indices': derived,
                'action_mask': mask.to(torch.uint8),
            }
            logits = policy.get_logits(obs)
            masked_logits = logits.masked_fill(~mask, mask_fill_val)
            actions = masked_logits.argmax(dim=-1)
            
            # 2. Take step
            active = ~done
            next_states = derived[arange_B, actions]
            new_current = torch.where(active.view(B, 1, 1), next_states, current)
            new_depths = torch.where(active, depths + 1, depths)
            
            # 3. Update history (compute hash of new state and add to history)
            new_hash = env._compute_hash(new_current)
            write_pos = history_count.clamp(max=H - 1)
            update_val = torch.where(active, new_hash, history_hashes[arange_B, write_pos])
            new_history = history_hashes.clone()
            new_history.scatter_(1, write_pos.unsqueeze(1), update_val.unsqueeze(1))
            new_h_count = torch.where(active, (history_count + 1).clamp(max=H), history_count)
            
            # 4. Compute derived with memory pruning for active slots
            still_active = active  # Compute for all active
            new_derived, new_counts = env.get_derived_simple(new_current, new_history, new_h_count)
            # Keep old derived for done slots
            new_derived = torch.where(still_active.view(B, 1, 1, 1), new_derived, derived)
            new_counts = torch.where(still_active, new_counts, counts)
            
            # 5. Terminal check
            first_pred = new_current[:, 0, 0]
            is_true = (first_pred == true_pred_idx) if true_pred_idx is not None else torch.zeros(B, dtype=torch.bool, device=device)
            is_false_pred = (first_pred == false_pred_idx) if false_pred_idx is not None else torch.zeros(B, dtype=torch.bool, device=device)
            is_pad = (first_pred == pad)
            is_end = (first_pred == end_pred_idx) if end_pred_idx is not None else torch.zeros(B, dtype=torch.bool, device=device)
            is_terminal = is_true | is_false_pred | is_pad | is_end
            is_success = is_true & active
            truncated = (new_depths >= max_depth) & active
            newly_done = active & (is_terminal | truncated)
            
            # 6. Slot recycling
            finished_idx = torch.where(newly_done, pool_ptr, torch.full_like(pool_ptr, -1))
            new_ptr = torch.where(newly_done, pool_ptr + stride_tensor, pool_ptr)
            needs_reset = newly_done & (new_ptr < pool_size)
            
            # 7. Get new queries for reset slots
            safe_idx = new_ptr.clamp(0, max_pool - 1)
            new_queries_raw = pool[safe_idx]
            reset_queries = torch.full((B, A, 3), pad, dtype=torch.long, device=device)
            reset_queries[:, 0, :] = new_queries_raw
            
            # Reset history for recycled slots
            reset_history = torch.zeros_like(new_history)
            reset_h_count = torch.zeros_like(new_h_count)
            
            # Compute derived for reset queries (with empty history)
            reset_derived, reset_counts = env.get_derived_simple(reset_queries, reset_history, reset_h_count)
            
            # 8. Merge with torch.where
            m1 = needs_reset.view(B, 1, 1)
            m3 = needs_reset.view(B, 1, 1, 1)
            mH = needs_reset.view(B, 1).expand(B, H)
            final_current = torch.where(m1, reset_queries, new_current)
            final_derived = torch.where(m3, reset_derived, new_derived)
            final_counts = torch.where(needs_reset, reset_counts, new_counts)
            final_depths = torch.where(needs_reset, torch.zeros_like(new_depths), new_depths)
            final_done = torch.where(needs_reset, torch.zeros_like(done), done | newly_done)
            final_history = torch.where(mH, reset_history, new_history)
            final_h_count = torch.where(needs_reset, reset_h_count, new_h_count)
            
            # Handle exhausted pool
            exhausted = new_ptr >= pool_size
            final_done = torch.where(exhausted & newly_done, torch.ones_like(final_done), final_done)
            new_mask = arange_S.unsqueeze(0) < final_counts.unsqueeze(1)
            
            return (final_current, final_derived, final_counts, new_mask, 
                    final_depths, final_done, new_ptr, final_history, final_h_count,
                    newly_done, is_success, finished_idx)
        
        self._compiled_v10_step = torch.compile(step_fn, mode=self._compile_mode, fullgraph=True, dynamic=False)
        print("[PPOOptimal] V10-style evaluation step compiled (with history tracking)")

    def _setup_v10_pool(self, queries: Tensor, sampler, n_corruptions: int, modes: Sequence[str]):
        """Setup pool with transposed layout for V10 evaluation."""
        N = queries.shape[0]
        K = 1 + n_corruptions
        B = self.fixed_batch_size
        H = self._v10_max_history
        device = self.device
        pad = self.env.padding_idx
        
        pools = []
        for mode in modes:
            neg = sampler.corrupt(queries, num_negatives=n_corruptions, mode=mode, device=device)
            cands = torch.cat([queries.unsqueeze(1), neg], dim=1)
            # Transposed layout: K x N instead of N x K
            cands_t = cands.transpose(0, 1).contiguous()
            pools.append(cands_t.view(-1, 3))
        
        new_pool = torch.cat(pools, dim=0)
        new_size = new_pool.size(0)
        
        if new_size > self._v10_max_pool:
            raise ValueError(f"Pool size {new_size} exceeds max {self._v10_max_pool}")
        
        self._v10_pool[:new_size].copy_(new_pool)
        self._v10_pool[new_size:].fill_(pad)
        self._v10_pool_size.fill_(new_size)
        self._v10_result_buf.zero_()
        
        self._v10_N = N
        self._v10_K = K
        self._v10_stride.fill_(N)
        
        # Initialize from pool
        init_idx = torch.arange(B, device=device).clamp(max=max(0, N - 1))
        queries_raw = self._v10_pool[init_idx]
        
        init_queries = torch.full((B, self.padding_atoms, 3), pad, dtype=torch.long, device=device)
        init_queries[:, 0, :] = queries_raw
        
        # Initialize with empty history (start of new episode)
        init_history = torch.zeros(B, H, dtype=torch.long, device=device)
        init_h_count = torch.zeros(B, dtype=torch.long, device=device)
        
        # Get derived with empty history (no memory pruning on first step)
        init_derived, init_counts = self.env.get_derived_simple(init_queries, init_history, init_h_count)
        
        self._v10_current.copy_(init_queries)
        self._v10_derived.copy_(init_derived)
        self._v10_counts.copy_(init_counts)
        self._v10_mask.copy_(self._v10_arange_S.unsqueeze(0) < init_counts.unsqueeze(1))
        self._v10_depths.zero_()
        self._v10_history_hashes.zero_()
        self._v10_history_count.zero_()
        
        self._v10_done.zero_()
        if N < B:
            self._v10_done[N:] = True
        
        self._v10_pool_ptr.copy_(torch.arange(B, device=device))

    @torch.no_grad()
    def evaluate(self, queries: Tensor, sampler, n_corruptions: int = 50, corruption_modes: Sequence[str] = ('head', 'tail'), *,
                 chunk_queries: int = None, verbose: bool = False, deterministic: bool = True, parity_mode: bool = False,
                 query_depths: Optional[Tensor] = None):
        """V10-style evaluation with slot recycling for maximum throughput.

        Args:
            queries: [N, 3] Query triples to evaluate
            sampler: Negative sampler for corruption generation
            n_corruptions: Number of negative samples per query (default 50)
            corruption_modes: Tuple of corruption modes, e.g. ('head', 'tail')
            chunk_queries: Max queries per chunk (auto-computed if None to fit pool)
            verbose: Print progress information
            deterministic: Use deterministic action selection
            parity_mode: If True, use eval_corruptions for exact ranking parity with tensor implementation.
            query_depths: Optional tensor of query depths for evaluation metrics by depth.
        """
        # If parity mode, delegate to eval_corruptions for exact parity
        if parity_mode:
            from tensor.tensor_model_eval import eval_corruptions

            # Create uncompiled environment for eval_corruptions (it needs dynamic batch sizes)
            # The compiled environment has fixed batch sizes baked in
            from env import EnvOptimal as EnvVec, TensorDictEnvWrapper

            uncompiled_env = EnvVec(
                vec_engine=self.env.engine,
                batch_size=self.env.batch_size,
                padding_atoms=self.env.padding_atoms,
                padding_states=self.env.padding_states,
                max_depth=self.env.max_depth,
                end_proof_action=self.env.end_proof_action,
                runtime_var_start_index=self.env.runtime_var_start_index,
                device=self.device,
                memory_pruning=self.env.memory_pruning,
                valid_queries=self.env.valid_queries,
                compile=False,  # Do not compile - eval_corruptions needs dynamic batching
            )

            wrapped_env = TensorDictEnvWrapper(uncompiled_env)
            return eval_corruptions(
                actor=self.policy,
                env=wrapped_env,
                queries=queries,
                sampler=sampler,
                n_corruptions=n_corruptions,
                corruption_modes=tuple(corruption_modes),
                deterministic=deterministic,
                verbose=verbose,
                query_depths=query_depths,
            )

        # Otherwise, use optimized V10 evaluation
        device = self.device
        N = queries.shape[0]
        K = 1 + n_corruptions
        num_modes = len(corruption_modes)
        
        # Setup V10 evaluation buffers (needed to get max_pool size)
        self._setup_v10_eval_buffers()
        self._compile_v10_eval_step()
        
        # Auto-compute chunk size to fit within pool limit
        # Pool size = N_chunk * K * num_modes
        # So N_chunk = max_pool // (K * num_modes)
        max_queries_per_chunk = self._v10_max_pool // (K * num_modes)
        if chunk_queries is None:
            chunk_queries = max_queries_per_chunk
        else:
            chunk_queries = min(chunk_queries, max_queries_per_chunk)
        
        if chunk_queries < 1:
            raise ValueError(f"Too many corruptions ({n_corruptions}) or modes ({num_modes}) to fit in pool. "
                           f"Max pool: {self._v10_max_pool}, needed per query: {K * num_modes}")
        
        # If all queries fit in one chunk, use single-pass evaluation
        if N <= chunk_queries:
            return self._evaluate_chunk(queries, sampler, n_corruptions, corruption_modes, verbose)
        
        # Otherwise, process in chunks and aggregate results
        if verbose:
            print(f"Processing {N} queries in chunks of {chunk_queries} (pool limit: {self._v10_max_pool})")
        
        # Accumulate per-mode results
        all_ranks = {mode: [] for mode in corruption_modes}
        
        for start_idx in range(0, N, chunk_queries):
            end_idx = min(start_idx + chunk_queries, N)
            chunk = queries[start_idx:end_idx]
            
            if verbose:
                print(f"  Chunk {start_idx//chunk_queries + 1}: queries {start_idx}-{end_idx-1}")
            
            # Get ranks for this chunk (only run once per chunk!)
            chunk_ranks = self._evaluate_chunk_ranks(chunk, sampler, n_corruptions, corruption_modes)
            for mode in corruption_modes:
                all_ranks[mode].append(chunk_ranks[mode])
        
        # Aggregate ranks across all chunks
        results = {}
        for mode in corruption_modes:
            ranks = torch.cat(all_ranks[mode], dim=0)  # [N_total]
            results[f'{mode}_mrr'] = (1.0 / ranks.float()).mean().item()
            results[f'{mode}_hits1'] = (ranks <= 1).float().mean().item()
            results[f'{mode}_hits3'] = (ranks <= 3).float().mean().item()
            results[f'{mode}_hits10'] = (ranks <= 10).float().mean().item()
        
        # Average across modes
        nm = len(corruption_modes)
        results['MRR'] = sum(results[f'{m}_mrr'] for m in corruption_modes) / nm
        results['Hits@1'] = sum(results[f'{m}_hits1'] for m in corruption_modes) / nm
        results['Hits@3'] = sum(results[f'{m}_hits3'] for m in corruption_modes) / nm
        results['Hits@10'] = sum(results[f'{m}_hits10'] for m in corruption_modes) / nm
        
        return results

    def _evaluate_chunk_ranks(self, queries: Tensor, sampler, n_corruptions: int, 
                               corruption_modes: Sequence[str]) -> Dict[str, Tensor]:
        """Evaluate a chunk and return raw ranks per mode (for aggregation)."""
        device = self.device
        N = queries.shape[0]
        K = 1 + n_corruptions
        
        # Setup pool for this chunk
        self._setup_v10_pool(queries, sampler, n_corruptions, corruption_modes)
        pool_size_int = int(self._v10_pool_size.item())
        
        # Run evaluation loop
        max_steps = (pool_size_int // self.fixed_batch_size + 2) * self.max_depth
        steps = 0
        
        while steps < max_steps:
            torch.compiler.cudagraph_mark_step_begin()
            
            (new_cur, new_der, new_cnt, new_mask, new_dep, new_done, new_ptr,
             new_history, new_h_count, newly_done, success, indices) = self._compiled_v10_step(self._v10_pool, self._v10_pool_size)
            
            torch._foreach_copy_(
                [self._v10_current, self._v10_derived, self._v10_counts, self._v10_mask, 
                 self._v10_depths, self._v10_done, self._v10_pool_ptr,
                 self._v10_history_hashes, self._v10_history_count],
                [new_cur, new_der, new_cnt, new_mask, new_dep, new_done, new_ptr,
                 new_history, new_h_count]
            )
            
            safe_idx = indices.clamp(min=0, max=pool_size_int - 1)
            valid_mask = (indices >= 0) & (indices < pool_size_int)
            safe_val = success & valid_mask
            self._v10_result_buf.scatter_(0, safe_idx, safe_val)
            
            steps += 1
            if self._v10_done.all():
                break
        
        # Compute ranks per mode
        ranks_per_mode = {}
        offset = 0
        for mode in corruption_modes:
            ms_t = self._v10_result_buf[offset:offset + N * K].view(K, N)
            ms = ms_t.t().contiguous()
            
            scores = torch.where(ms, torch.zeros(N, K, device=device),
                               torch.full((N, K), -100.0, device=device))
            
            pos, neg = scores[:, 0:1], scores[:, 1:]
            rnd = torch.rand(N, K, device=device)
            better = neg > pos
            tied = (neg == pos) & (rnd[:, 1:] > rnd[:, 0:1])
            ranks = 1 + better.sum(1) + tied.sum(1)
            
            ranks_per_mode[mode] = ranks
            offset += N * K
        
        return ranks_per_mode

    def _evaluate_chunk(self, queries: Tensor, sampler, n_corruptions: int, 
                        corruption_modes: Sequence[str], verbose: bool = False) -> Dict[str, float]:
        """Evaluate a single chunk of queries (must fit in pool)."""
        device = self.device
        N = queries.shape[0]
        K = 1 + n_corruptions
        
        # Setup pool with transposed layout
        self._setup_v10_pool(queries, sampler, n_corruptions, corruption_modes)
        pool_size_int = int(self._v10_pool_size.item())
        
        if verbose:
            print(f"V10 Pool: {pool_size_int}, batch: {self.fixed_batch_size}")
        
        # Run evaluation loop
        max_steps = (pool_size_int // self.fixed_batch_size + 2) * self.max_depth
        steps = 0
        
        while steps < max_steps:
            torch.compiler.cudagraph_mark_step_begin()
            
            (new_cur, new_der, new_cnt, new_mask, new_dep, new_done, new_ptr,
             new_history, new_h_count, newly_done, success, indices) = self._compiled_v10_step(self._v10_pool, self._v10_pool_size)
            
            # Copy back to persistent buffers (including history)
            torch._foreach_copy_(
                [self._v10_current, self._v10_derived, self._v10_counts, self._v10_mask, 
                 self._v10_depths, self._v10_done, self._v10_pool_ptr,
                 self._v10_history_hashes, self._v10_history_count],
                [new_cur, new_der, new_cnt, new_mask, new_dep, new_done, new_ptr,
                 new_history, new_h_count]
            )
            
            # Record results
            safe_idx = indices.clamp(min=0, max=pool_size_int - 1)
            valid_mask = (indices >= 0) & (indices < pool_size_int)
            safe_val = success & valid_mask
            self._v10_result_buf.scatter_(0, safe_idx, safe_val)
            
            steps += 1
            
            if self._v10_done.all():
                break
        
        if verbose:
            print(f"V10 Steps: {steps}")
        
        # Compute metrics
        results = {}
        offset = 0
        for mode in corruption_modes:
            # Untranspose: K x N -> N x K
            ms_t = self._v10_result_buf[offset:offset + N * K].view(K, N)
            ms = ms_t.t().contiguous()
            
            scores = torch.where(ms, torch.zeros(N, K, device=device),
                               torch.full((N, K), -100.0, device=device))
            
            pos, neg = scores[:, 0:1], scores[:, 1:]
            rnd = torch.rand(N, K, device=device)
            better = neg > pos
            tied = (neg == pos) & (rnd[:, 1:] > rnd[:, 0:1])
            ranks = 1 + better.sum(1) + tied.sum(1)
            
            results[f'{mode}_mrr'] = (1.0 / ranks.float()).mean().item()
            results[f'{mode}_hits1'] = (ranks <= 1).float().mean().item()
            results[f'{mode}_hits3'] = (ranks <= 3).float().mean().item()
            results[f'{mode}_hits10'] = (ranks <= 10).float().mean().item()
            offset += N * K
        
        nm = len(corruption_modes)
        results['MRR'] = sum(results[f'{m}_mrr'] for m in corruption_modes) / nm
        results['Hits@1'] = sum(results[f'{m}_hits1'] for m in corruption_modes) / nm
        results['Hits@3'] = sum(results[f'{m}_hits3'] for m in corruption_modes) / nm
        results['Hits@10'] = sum(results[f'{m}_hits10'] for m in corruption_modes) / nm
        
        return results

# Backward compatibility alias
PPO = PPOOptimal
