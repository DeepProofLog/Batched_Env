"""
PPO (Proximal Policy Optimization) with CUDA graph support.

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
from env import EnvVec, EnvObs, EnvState


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

def compute_auc_pr(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    try:
        if len(y_true) == 0 or y_true.sum() == 0 or y_true.sum() == len(y_true):
            return 1.0 if y_true.sum() == len(y_true) else 0.0
        return float(average_precision_score(y_true, y_scores))
    except Exception:
        return 0.0

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
# PPO CLASS
# =============================================================================

class PPO:
    """PPO with CUDA graph support."""

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------

    def __init__(self, policy: nn.Module, env: EnvVec, config, **kwargs):
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

        # Pre-allocated buffers
        self._eval_padded_buffer = torch.zeros(self.fixed_batch_size, 3, dtype=torch.long, device=self.device)
        max_cand = self.fixed_batch_size * 1001
        self._eval_acc_log_probs = torch.zeros(max_cand, device=self.device)
        self._eval_acc_success = torch.zeros(max_cand, dtype=torch.bool, device=self.device)
        self._eval_acc_depths = torch.zeros(max_cand, dtype=torch.long, device=self.device)
        self._eval_acc_rewards = torch.zeros(max_cand, device=self.device)
        self._eval_result_log_probs = torch.zeros(self.fixed_batch_size, device=self.device)
        self._eval_result_success = torch.zeros(self.fixed_batch_size, dtype=torch.bool, device=self.device)
        self._eval_result_depths = torch.zeros(self.fixed_batch_size, dtype=torch.long, device=self.device)
        self._eval_result_rewards = torch.zeros(self.fixed_batch_size, device=self.device)

        # Pre-computed constant for mask filling (avoids tensor creation in hot loop)
        self._mask_fill_value = -3.4e38

        # Metrics
        query_labels = kwargs.get('query_labels', None)
        query_depths = kwargs.get('query_depths', None)
        if query_depths is None and components.get('dh'):
            try:
                train_split = components['dh'].get_materialized_split('train')
                query_depths = train_split.depths
                query_labels = getattr(train_split, 'labels', torch.ones_like(query_depths))
            except Exception:
                pass
        self.query_labels = query_labels.detach().cpu() if query_labels is not None else None
        self.query_depths = query_depths.detach().cpu() if query_depths is not None else None
        self.current_query_indices = None

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
            self._uncompiled_policy = policy
            self._compiled_policy_fn = torch.compile(policy.get_logits, mode=self._compile_mode, fullgraph=True)

        self.num_timesteps = 0
        self.callback = self._build_callbacks() if not self.eval_only and getattr(config, 'use_callbacks', True) else None
        print(f"[DEBUG] PPO.__init__ finished setup.")

    @property
    def fixed_batch_size(self) -> int:
        return self._fixed_batch_size or self.batch_size_env

    # -------------------------------------------------------------------------
    # COMPILATION
    # -------------------------------------------------------------------------

    def _compile_all(self):
        """Compile all policy and env functions."""
        self._uncompiled_policy = self.policy
        self.loss_module = torch.compile(PPOLossModule(self._uncompiled_policy), mode=self._compile_mode, fullgraph=True)
        self._warmup_gradients()
        self._compiled_policy_fn = torch.compile(self._uncompiled_policy.get_logits, mode=self._compile_mode, fullgraph=True)
        self.policy = torch.compile(self._uncompiled_policy, mode=self._compile_mode, fullgraph=True)
        if self.config.compile:
            self._setup_fused_rollout_step()
        print("[DEBUG] PPO.__init__ reached compiled_evaluate_loop init block")

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
        """Fused policy + env step for training."""
        policy, env = self._uncompiled_policy, self.env

        def fused_step(obs, state):
            logits = policy.get_logits(obs)
            masked = logits.masked_fill(obs['action_mask'] == 0, -3.4e38)
            probs = torch.softmax(masked, dim=-1)
            actions = torch.multinomial(probs, 1).squeeze(-1)
            log_probs = torch.log_softmax(masked, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
            log_probs = log_probs.masked_fill(state['done'].bool(), 0.0)
            new_obs, new_state = env._step_and_reset_core(state, actions, env._query_pool, env._per_env_ptrs)
            return new_obs, new_state, actions, log_probs

        self._compiled_rollout_step = torch.compile(fused_step, mode=self._compile_mode, fullgraph=True)

    def _setup_fused_eval_step(self):
        """Fused policy + env step for eval (deterministic)."""
        if hasattr(self, '_compiled_eval_step'):
            return
        policy, env = self._uncompiled_policy, self.env
        mask_val = self._mask_fill_value

        def fused_eval(obs, state):
            logits = policy.get_logits(obs)
            # Fuse mask application and argmax - use pre-computed mask value
            masked_logits = torch.where(obs['action_mask'].bool(), logits, mask_val)
            actions = masked_logits.argmax(dim=-1)
            new_obs, new_state = env._step_core(state, actions)
            return new_obs, new_state

        self._compiled_eval_step = torch.compile(fused_eval, mode=self._compile_mode, fullgraph=True)
        print("[PPO] Eval compilation complete.")

    # -------------------------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------------------------

    def collect_rollouts(self, current_state, current_obs, episode_starts, current_episode_reward, current_episode_length,
                         episode_rewards, episode_lengths, iteration, return_traces=False, on_step_callback=None):
        """Collect experiences."""
        self.policy.eval()
        self.rollout_buffer.reset()
        n_collected, traces = 0, [] if return_traces else None
        state, obs = current_state, current_obs

        if self.current_query_indices is None and 'per_env_ptrs' in state.keys():
            self.current_query_indices = state['per_env_ptrs'].cpu().numpy()

        with torch.no_grad():
            while n_collected < self.n_steps:
                obs_snap = {k: v.clone() for k, v in obs.items()}
                values = self.policy.predict_values(obs)
                torch.compiler.cudagraph_mark_step_begin()
                obs_in, state_in = {k: v.clone() for k, v in obs.items()}, state.clone()
                new_obs, new_state, actions, log_probs = self._compiled_rollout_step(obs_in, state_in)

                self.rollout_buffer.add(
                    sub_index=obs_snap['sub_index'], derived_sub_indices=obs_snap['derived_sub_indices'],
                    action_mask=obs_snap['action_mask'], action=actions, reward=new_state['step_rewards'],
                    episode_start=episode_starts, value=values.flatten(), log_prob=log_probs,
                )

                current_episode_reward += new_state['step_rewards']
                current_episode_length += 1
                n_collected += 1

                done_idx = torch.nonzero(new_state['step_dones']).squeeze(-1)
                if done_idx.numel() > 0:
                    self._handle_done_episodes(done_idx, current_episode_reward, current_episode_length, state, new_state['per_env_ptrs'], episode_rewards, episode_lengths, on_step_callback)
                    current_episode_reward.masked_fill_(new_state['step_dones'].bool(), 0.0)
                    current_episode_length.masked_fill_(new_state['step_dones'].bool(), 0)

                episode_starts = new_state['step_dones'].float()
                state, obs = new_state, new_obs

            last_values = self.policy.predict_values(obs)

        self.rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=new_state['step_dones'].float())
        return state, obs, episode_starts, current_episode_reward, current_episode_length, n_collected * self.batch_size_env, traces

    def _handle_done_episodes(self, done_idx, ep_rew, ep_len, state, next_ptrs, episode_rewards, episode_lengths, callback):
        idx_cpu = done_idx.cpu().numpy()
        rews = ep_rew[done_idx].float().cpu().numpy()
        lens = ep_len[done_idx].cpu().numpy().astype(int)
        episode_rewards.extend(rews.tolist())
        episode_lengths.extend(lens.tolist())
        if callback and self.current_query_indices is not None:
            batch_infos = [{"episode": {"r": float(r), "l": int(l)}} for r, l in zip(rews, lens)]
            callback(batch_infos)
            self.current_query_indices[idx_cpu] = next_ptrs[done_idx].cpu().numpy()

    def train(self, return_traces=False):
        """Update policy from rollout buffer."""
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
        """Main PPO loop."""
        if reset_num_timesteps:
            self.num_timesteps = 0
        else:
            total_timesteps += self.num_timesteps

        if self.callback:
            self.callback.on_training_start()

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

            step_cb = self.callback.on_step if self.callback else on_step_callback
            result = self.collect_rollouts(state, obs, ep_starts, curr_ep_rew, curr_ep_len, ep_rews, ep_lens, iteration, return_traces, step_cb)
            state, obs, ep_starts, curr_ep_rew, curr_ep_len, n_steps, _ = result
            state = state.clone()
            self.num_timesteps += n_steps

            train_metrics = self.train(return_traces)
            self.last_train_metrics = train_metrics

            if self.callback:
                if self.callback({'iteration': iteration, 'total_steps_done': self.num_timesteps, 'episode_rewards': ep_rews, 'episode_lengths': ep_lens, 'train_metrics': train_metrics}, None) is False:
                    break

        return {'num_timesteps': self.num_timesteps, 'episode_rewards': ep_rews, 'episode_lengths': ep_lens, 'last_train_metrics': self.last_train_metrics}

    # -------------------------------------------------------------------------
    # EVALUATION
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_policy(self, queries: Tensor, max_steps: Optional[int] = None, deterministic: bool = True):
        """Evaluate queries with compiled step loop."""
        assert queries.shape[0] == self.env.batch_size
        max_steps = max_steps or self.max_depth
        self._setup_fused_eval_step()

        torch.compiler.cudagraph_mark_step_begin()
        obs, state = self.env.reset(queries)
        obs, state = obs.clone(), state.clone()

        for _ in range(max_steps):
            torch.compiler.cudagraph_mark_step_begin()
            obs, state = self._compiled_eval_step(obs, state)
            obs, state = obs.clone(), state.clone()

        self._eval_result_log_probs.zero_()
        self._eval_result_success[:] = state['success']
        self._eval_result_depths[:] = state['depths']
        self._eval_result_rewards[:] = state['cumulative_rewards']
        return self._eval_result_log_probs, self._eval_result_success, self._eval_result_depths, self._eval_result_rewards

    def _pad_queries(self, queries: Tensor) -> Tuple[Tensor, int]:
        B = queries.shape[0]
        if B >= self.fixed_batch_size:
            return queries[:self.fixed_batch_size], min(B, self.fixed_batch_size)
        # Use copy_ for in-place updates (more efficient than indexing assignment)
        self._eval_padded_buffer[:B].copy_(queries)
        if B > 0 and B < self.fixed_batch_size:
            # Replicate last query to fill the buffer (avoids creating new tensors)
            self._eval_padded_buffer[B:].copy_(queries[-1].unsqueeze(0).expand(self.fixed_batch_size - B, -1))
        return self._eval_padded_buffer, B

    @torch.no_grad()
    def evaluate(self, queries: Tensor, sampler, n_corruptions: int = 50, corruption_modes: Sequence[str] = ('head', 'tail'), *,
                 chunk_queries: int = 50, verbose: bool = False, deterministic: bool = True, parity_mode: bool = False, query_depths: Optional[Tensor] = None):
        """Compute MRR/Hits@K metrics with mega-batching to reduce kernel overhead."""
        from callbacks import Display
        device = self.device
        total = queries.shape[0]
        per_mode_ranks = {m: torch.zeros(total, device=device) for m in corruption_modes}
        rng = np.random.RandomState(0) if parity_mode else None

        for start in range(0, total, chunk_queries):
            end = min(start + chunk_queries, total)
            Q = end - start
            chunk = queries[start:end]

            for mode in corruption_modes:
                corruptions = sampler.corrupt(chunk, num_negatives=n_corruptions, mode=mode, device=device)
                valid_mask = corruptions.sum(dim=-1) != 0
                candidates = torch.zeros(Q, 1 + n_corruptions, 3, dtype=torch.long, device=device)
                candidates[:, 0, :] = chunk
                candidates[:, 1:, :] = corruptions
                flat = candidates.view(-1, 3)
                total_cand = flat.shape[0]

                acc_success = self._eval_acc_success[:total_cand]
                acc_success.fill_(False)

                # MEGA-BATCHING: Process multiple batches in single evaluate_policy_mega call
                # This dramatically reduces kernel launch overhead (Command Buffer saturation)
                mega_batch_size = self.fixed_batch_size * 8  # Process 8 batches at once
                n_mega_batches = total_cand // mega_batch_size

                for i in range(n_mega_batches):
                    mega_start = i * mega_batch_size
                    mega_end = mega_start + mega_batch_size
                    mega_batch = flat[mega_start:mega_end]
                    success = self.evaluate_policy_mega(mega_batch, self.max_depth, deterministic)
                    acc_success[mega_start:mega_end] = success

                # Handle remaining candidates with standard batching
                remaining_start = n_mega_batches * mega_batch_size
                if remaining_start < total_cand:
                    flat_remainder = flat[remaining_start:]
                    total_remaining = flat_remainder.shape[0]

                    # Process full batches
                    n_full_batches = total_remaining // self.fixed_batch_size
                    for i in range(n_full_batches):
                        batch_start = i * self.fixed_batch_size
                        batch_end = batch_start + self.fixed_batch_size
                        batch = flat_remainder[batch_start:batch_end]
                        _, success, _, _ = self.evaluate_policy(batch, self.max_depth, deterministic)
                        acc_success[remaining_start + batch_start:remaining_start + batch_end] = success

                    # Handle final remainder with padding
                    remainder = total_remaining % self.fixed_batch_size
                    if remainder > 0:
                        batch_start = n_full_batches * self.fixed_batch_size
                        batch = flat_remainder[batch_start:]
                        padded, _ = self._pad_queries(batch)
                        _, success, _, _ = self.evaluate_policy(padded, self.max_depth, deterministic)
                        acc_success[remaining_start + batch_start:] = success[:remainder]

                success = acc_success.view(Q, 1 + n_corruptions)
                log_probs = torch.zeros_like(success, dtype=torch.float32)
                log_probs[~success] -= 100.0

                pos_score = log_probs[:, 0:1]
                neg_scores = log_probs[:, 1:]
                rnd = torch.as_tensor(rng.rand(Q, 1 + n_corruptions), device=device, dtype=torch.float32) if parity_mode else torch.rand(Q, 1 + n_corruptions, device=device)
                better = (neg_scores > pos_score) & valid_mask
                tied = (neg_scores == pos_score) & (rnd[:, 1:] > rnd[:, 0:1]) & valid_mask
                ranks = 1 + better.sum(dim=1) + tied.sum(dim=1)
                per_mode_ranks[mode][start:end] = ranks.float()

        results = {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0, "per_mode": {}}
        for mode in corruption_modes:
            results["per_mode"][mode] = compute_metrics_from_ranks(per_mode_ranks[mode])
        for mode in corruption_modes:
            for k, v in results["per_mode"][mode].items():
                results[k] += v
        n = len(corruption_modes)
        for k in ["MRR", "Hits@1", "Hits@3", "Hits@10"]:
            results[k] /= n if n > 0 else 1.0
        return results

    # -------------------------------------------------------------------------
    # CALLBACKS
    # -------------------------------------------------------------------------

    def _build_callbacks(self):
        from callbacks import TorchRLCallbackManager, MetricsCallback, RankingCallback, CheckpointCallback, ScalarAnnealingCallback, AnnealingTarget
        from pathlib import Path
        callbacks = []
        config = self.config
        components = getattr(config, '_components', {})

        if getattr(config, 'use_metrics_callback', True):
            callbacks.append(MetricsCallback(log_interval=1, verbose=self.verbose, collect_detailed=True))

        sampler = components.get('sampler')
        dh = components.get('dh')
        if getattr(config, 'use_ranking_callback', True) and getattr(config, 'eval_freq', 0) > 0 and sampler and dh:
            valid_split = dh.get_materialized_split('valid')
            valid_queries = valid_split.queries.squeeze(1)
            valid_depths = valid_split.depths
            n_eval = getattr(config, 'n_eval_queries', None)
            if n_eval:
                valid_queries, valid_depths = valid_queries[:n_eval], valid_depths[:n_eval]
            callbacks.append(RankingCallback(
                eval_env=self.env, policy=self.policy, sampler=sampler, eval_data=valid_queries,
                eval_data_depths=valid_depths, eval_freq=int(config.eval_freq),
                n_corruptions=getattr(config, 'eval_neg_samples', 50),
                corruption_scheme=tuple(getattr(config, 'corruption_scheme', ('head', 'tail'))),
                ppo_agent=self
            ))

        if getattr(config, 'use_checkpoint_callback', True) and getattr(config, 'save_model', False):
            save_path = Path(getattr(config, 'models_path', './models/')) / getattr(config, 'run_signature', 'run')
            callbacks.append(CheckpointCallback(save_path=save_path, policy=self.policy, train_metric="ep_rew_mean", eval_metric=getattr(config, 'eval_best_metric', 'mrr_mean'), verbose=True))

        if getattr(config, 'use_annealing_callback', True):
            targets = []
            total = getattr(config, 'total_timesteps', 0)
            if getattr(config, 'lr_decay', False):
                def _set_lr(v):
                    for pg in self.optimizer.param_groups: pg['lr'] = float(v)
                    self.learning_rate = float(v)
                targets.append(AnnealingTarget(name='lr', setter=_set_lr, initial=float(getattr(config, 'lr_init_value', self.learning_rate)), final=float(getattr(config, 'lr_final_value', 1e-6)), start_point=0.0, end_point=1.0, transform='linear', value_type='float'))
            if getattr(config, 'ent_coef_decay', False):
                def _set_ent(v): self.ent_coef = float(v)
                targets.append(AnnealingTarget(name='ent_coef', setter=_set_ent, initial=float(getattr(config, 'ent_coef_init_value', self.ent_coef)), final=float(getattr(config, 'ent_coef_final_value', 0.01)), start_point=0.0, end_point=1.0, transform='linear', value_type='float'))
            if targets:
                callbacks.append(ScalarAnnealingCallback(total_timesteps=total, targets=targets, verbose=1))

        return TorchRLCallbackManager(callbacks=callbacks) if callbacks else None
