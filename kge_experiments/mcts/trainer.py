"""
MuZero Trainer for DeepProofLog.

Main training loop combining MCTS search with neural network training.
Uses real environment dynamics instead of learned model.

Training loop:
1. Collect episodes using MCTS for action selection
2. Store trajectories with MCTS statistics in replay buffer
3. Sample batches and train network on MCTS targets

Key differences from PPO:
- Off-policy: reuses old experiences from replay buffer
- Action selection: MCTS search instead of direct policy sampling
- Training targets: MCTS visit counts instead of advantages
- Value targets: N-step returns instead of GAE

Optimization notes (CUDA graph compatible batched MCTS):
- MCTSTreeTensors replaces Python dicts for [B, D, A] tensor storage
- Vectorized PUCT selection processes all B environments simultaneously
- Compile single simulation step, not entire search loop
- Slot recycling for efficient batched evaluation
- Avoid .item() calls in hot paths - use tensor operations

Reference: Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by
Planning with a Learned Model" (MuZero, 2020)
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tensordict import TensorDict

from .config import MCTSConfig
from .tree import MCTS, Node, MCTSTreeTensors, MCTSBatched
from .networks import MCTSPolicy
from .replay_buffer import MCTSReplayBuffer, MCTSTrajectory, MCTSTransition


class MuZeroTrainer:
    """MuZero-style trainer using MCTS with real environment dynamics.

    Combines MCTS search for action selection with neural network training
    on MCTS-generated targets. Unlike the original MuZero, this implementation
    uses real environment dynamics (env.step()) instead of a learned model.

    Training loop:
    1. collect_episodes(): Run MCTS to collect trajectories
    2. train_step(): Update network on sampled trajectories
    3. Repeat until total_timesteps reached

    Attributes:
        config: MCTSConfig with hyperparameters.
        env: Environment instance.
        policy: Policy/Value network.
        mcts: MCTS search algorithm.
        replay_buffer: Experience storage.
        optimizer: Network optimizer.
    """

    def __init__(
        self,
        policy: MCTSPolicy,
        env: Any,
        config: MCTSConfig,
        callback: Optional[Any] = None,
        **kwargs: Any,
    ):
        """Initialize MuZero trainer.

        Args:
            policy: Policy/Value network.
            env: Environment with step(), reset() methods.
            config: MCTSConfig with hyperparameters.
            callback: Optional callback for logging.
            **kwargs: Additional configuration overrides.
        """
        self.policy = policy
        self.env = env
        self.config = config
        self.callback = callback

        # Device setup
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)

        # Initialize original MCTS (for non-batched fallback)
        self.mcts = MCTS(config)

        # Initialize replay buffer
        self.replay_buffer = MCTSReplayBuffer(
            max_size=config.replay_buffer_size,
            discount=config.discount,
            n_step=config.value_nstep,
            device=self.device,
        )

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Training state
        self.num_timesteps = 0
        self.num_episodes = 0
        self.iteration = 0
        self.last_train_metrics = {}

        # Gradient clipping
        self.max_grad_norm = config.max_grad_norm

        # Loss weights
        self.policy_loss_weight = config.policy_loss_weight
        self.value_loss_weight = config.value_loss_weight

        # Logging
        self.verbose = config.verbose
        self._uncompiled_policy = policy  # For compatibility with PPO interface

        # =====================================================================
        # BATCHED MCTS SETUP (CUDA Graph Compatible)
        # =====================================================================
        self.use_batched_mcts = config.use_batched_mcts

        if self.use_batched_mcts:
            # Batch dimensions
            self.batch_size = config.mcts_batch_size
            self.fixed_batch_size = config.fixed_batch_size
            self.max_depth = config.max_tree_depth
            self.max_actions = config.max_actions

            # Initialize batched MCTS
            self.mcts_batched = MCTSBatched(
                config=config,
                batch_size=self.batch_size,
                max_actions=self.max_actions,
                device=self.device,
            )

            # Pre-allocate buffers for batched operations
            self._alloc_batched_buffers()

            # Setup compiled simulation step
            if config.compile:
                self._setup_compiled_sim_step()
            else:
                self._compiled_sim_step = None
                self._compiled_policy_forward = None

            # Setup evaluation buffers (slot recycling pattern)
            self._alloc_eval_buffers()

    # =========================================================================
    # BATCHED MCTS BUFFER ALLOCATION
    # =========================================================================

    def _alloc_batched_buffers(self) -> None:
        """Pre-allocate buffers for batched MCTS operations.

        Allocates fixed-size tensors for:
        - Simulation state tracking
        - Path recording for backpropagation
        - Rollout collection
        """
        B = self.batch_size
        D = self.max_depth
        A = self.max_actions
        device = self.device
        pad_atoms = getattr(self.env, 'padding_atoms', 6)

        # Simulation buffers - state clones for tree traversal
        self._sim_current = torch.zeros((B, pad_atoms, 3), dtype=torch.long, device=device)
        self._sim_derived = torch.zeros((B, A, pad_atoms, 3), dtype=torch.long, device=device)
        self._sim_counts = torch.zeros(B, dtype=torch.long, device=device)
        self._sim_mask = torch.zeros((B, A), dtype=torch.bool, device=device)
        self._sim_done = torch.zeros(B, dtype=torch.bool, device=device)
        self._sim_depths = torch.zeros(B, dtype=torch.long, device=device)

        # Path buffers for backpropagation - track (depth, action) pairs
        self._path_depths = torch.zeros((B, D), dtype=torch.long, device=device)
        self._path_actions = torch.zeros((B, D), dtype=torch.long, device=device)
        self._path_length = torch.zeros(B, dtype=torch.long, device=device)

        # Rollout buffers - store transitions for B parallel environments
        rollout_size = self.config.episodes_per_iteration * self.config.max_episode_steps
        self._rollout_obs_sub = torch.zeros((rollout_size, 1, pad_atoms, 3), dtype=torch.long, device=device)
        self._rollout_obs_derived = torch.zeros((rollout_size, A, pad_atoms, 3), dtype=torch.long, device=device)
        self._rollout_obs_mask = torch.zeros((rollout_size, A), dtype=torch.bool, device=device)
        self._rollout_actions = torch.zeros(rollout_size, dtype=torch.long, device=device)
        self._rollout_rewards = torch.zeros(rollout_size, dtype=torch.float32, device=device)
        self._rollout_dones = torch.zeros(rollout_size, dtype=torch.bool, device=device)
        self._rollout_values = torch.zeros(rollout_size, dtype=torch.float32, device=device)
        self._rollout_visit_counts = torch.zeros((rollout_size, A), dtype=torch.float32, device=device)

        # Helper tensors
        self._arange_B = torch.arange(B, device=device)
        self._arange_A = torch.arange(A, device=device)
        self._zeros_B_float = torch.zeros(B, dtype=torch.float32, device=device)
        self._zeros_B_long = torch.zeros(B, dtype=torch.long, device=device)
        self._ones_B_long = torch.ones(B, dtype=torch.long, device=device)
        # Pre-allocated tensors for terminal values (avoid allocation in hot loop)
        self._terminal_pos = torch.ones(B, dtype=torch.float32, device=device)
        self._terminal_neg = -torch.ones(B, dtype=torch.float32, device=device)

        # Mark static addresses for CUDA graph stability
        if hasattr(torch, "_dynamo"):
            static_buffers = [
                self._sim_current, self._sim_derived, self._sim_counts,
                self._sim_mask, self._sim_done, self._sim_depths,
                self._path_depths, self._path_actions, self._path_length,
                self._arange_B, self._zeros_B_float, self._zeros_B_long, self._ones_B_long,
            ]
            for buf in static_buffers:
                torch._dynamo.mark_static_address(buf)

    def _alloc_eval_buffers(self) -> None:
        """Pre-allocate buffers for batched evaluation with slot recycling.

        Follows PPO's _ranking_buffers pattern for efficient evaluation.
        Uses double-buffering for pipeline-friendly updates.

        Pool layout (strided for slot recycling):
        - Stride = N (number of queries)
        - Pool[i] = candidate i % N, type i // N
        - Types: [pos_0..pos_N-1, neg0_0..neg0_N-1, neg1_0..neg1_N-1, ...]
        - Each env advances by stride when done to get next candidate
        """
        B = self.fixed_batch_size
        D = self.max_depth
        A = self.max_actions
        H = getattr(self.env, 'max_history_size', D + 1)  # Match env's history size
        device = self.device
        pad_atoms = getattr(self.env, 'padding_atoms', 6)
        pad_idx = getattr(self.env, 'padding_idx', 0)
        S = A  # Max derived states

        def _alloc_buffer():
            return {
                "current": torch.full((B, pad_atoms, 3), pad_idx, dtype=torch.long, device=device),
                "derived": torch.full((B, S, pad_atoms, 3), pad_idx, dtype=torch.long, device=device),
                "counts": torch.zeros(B, dtype=torch.long, device=device),
                "mask": torch.zeros((B, S), dtype=torch.bool, device=device),
                "depths": torch.zeros(B, dtype=torch.long, device=device),
                "done": torch.zeros(B, dtype=torch.bool, device=device),
                "success": torch.zeros(B, dtype=torch.bool, device=device),
                "pool_ptr": torch.zeros(B, dtype=torch.long, device=device),
                "root_value": torch.zeros(B, dtype=torch.float32, device=device),
                "first_step": torch.ones(B, dtype=torch.bool, device=device),
                "history_hashes": torch.zeros((B, H), dtype=torch.int64, device=device),
                "history_count": torch.zeros(B, dtype=torch.long, device=device),
                "original_queries": torch.full((B, pad_atoms, 3), pad_idx, dtype=torch.long, device=device),
                "next_var": torch.zeros(B, dtype=torch.long, device=device),
            }

        # Double buffer pattern
        self._eval_buffers = (_alloc_buffer(), _alloc_buffer())
        self._eval_buf_idx = 0

        # Result buffers - sized for max expected pool
        max_pool = 500_000  # Max queries + corruptions
        self._eval_max_pool = max_pool
        self._eval_pool = torch.zeros((max_pool, 3), dtype=torch.long, device=device)
        self._eval_pool_size = torch.tensor(0, dtype=torch.long, device=device)
        self._eval_results_success = torch.zeros(max_pool, dtype=torch.bool, device=device)
        self._eval_results_value = torch.zeros(max_pool, dtype=torch.float32, device=device)
        self._eval_results_depth = torch.zeros(max_pool, dtype=torch.long, device=device)

        # Helper constants (follow PPO naming for consistency)
        self._eval_stride = torch.tensor(0, dtype=torch.long, device=device)  # Set per chunk
        self._eval_arange_B = torch.arange(B, device=device)
        self._eval_arange_S = torch.arange(S, device=device)
        self._eval_reset_labels = torch.ones(B, dtype=torch.long, device=device)
        self._eval_zero_float = torch.zeros(B, dtype=torch.float32, device=device)
        self._eval_zero_long = torch.zeros(B, dtype=torch.long, device=device)
        self._eval_zero_uint8 = torch.zeros(B, dtype=torch.uint8, device=device)
        self._eval_zero_int64 = torch.zeros(B, dtype=torch.int64, device=device)
        self._eval_true_bool = torch.ones(B, dtype=torch.bool, device=device)
        self._eval_false_bool = torch.zeros(B, dtype=torch.bool, device=device)
        self._eval_minus_one = torch.full((B,), -1, dtype=torch.long, device=device)

        # Mark static addresses for CUDA graph stability
        if hasattr(torch, "_dynamo"):
            static_buffers = [
                self._eval_pool, self._eval_pool_size,
                self._eval_results_success, self._eval_results_value, self._eval_results_depth,
                self._eval_stride, self._eval_arange_B, self._eval_arange_S, self._eval_reset_labels,
                self._eval_zero_float, self._eval_zero_long, self._eval_zero_uint8, self._eval_zero_int64,
                self._eval_true_bool, self._eval_false_bool, self._eval_minus_one,
            ]
            for buf in self._eval_buffers:
                static_buffers.extend(buf.values())
            for buf in static_buffers:
                if isinstance(buf, Tensor):
                    torch._dynamo.mark_static_address(buf)

    def _setup_compiled_sim_step(self) -> None:
        """Setup compiled single simulation step (policy + env forward).

        Compiles policy forward pass and environment step together.
        Uses _uncompiled_policy to avoid nested compilation issues.
        """
        policy = self._uncompiled_policy
        env = self.env
        device = self.device
        B = self.batch_size
        A = self.max_actions

        def sim_step_fn(
            obs_sub: Tensor,      # [B, 1, A_atoms, 3]
            obs_derived: Tensor,  # [B, S, A_atoms, 3]
            obs_mask: Tensor,     # [B, S]
            state: TensorDict,
            actions: Tensor,      # [B]
        ) -> Tuple[Tensor, Tensor, Tensor, TensorDict]:
            """Single simulation step: env.step with given actions.

            Note: Policy forward is separate to allow PUCT-based action selection.

            Args:
                obs_sub: Current state observations [B, 1, A_atoms, 3].
                obs_derived: Derived state observations [B, S, A_atoms, 3].
                obs_mask: Action mask [B, S].
                state: Environment state TensorDict.
                actions: Actions to take [B].

            Returns:
                new_obs_sub: New current states [B, 1, A_atoms, 3].
                new_obs_derived: New derived states [B, S, A_atoms, 3].
                new_obs_mask: New action mask [B, S].
                new_state: New environment state.
            """
            # Environment step
            new_obs, new_state = env.step(state, actions)

            new_obs_sub = new_obs['sub_index']
            new_obs_derived = new_obs['derived_sub_indices']
            new_obs_mask = new_obs['action_mask']

            return new_obs_sub, new_obs_derived, new_obs_mask, new_state

        def policy_forward_fn(
            obs_sub: Tensor,
            obs_derived: Tensor,
            obs_mask: Tensor,
        ) -> Tuple[Tensor, Tensor]:
            """Policy forward pass for priors and values.

            Args:
                obs_sub: [B, 1, A_atoms, 3]
                obs_derived: [B, S, A_atoms, 3]
                obs_mask: [B, S]

            Returns:
                priors: [B, A] softmax probabilities over valid actions.
                values: [B] value estimates.
            """
            obs = {
                'sub_index': obs_sub,
                'derived_sub_indices': obs_derived,
                'action_mask': obs_mask,
            }
            logits = policy.get_logits(obs)  # [B, A]
            values = policy.predict_values(obs)  # [B]

            # Mask invalid actions
            masked_logits = logits.masked_fill(obs_mask == 0, -3.4e38)
            priors = torch.softmax(masked_logits, dim=-1)

            return priors, values.flatten()

        # Compile with reduce-overhead mode
        mode = self.config.compile_mode
        self._compiled_sim_step = torch.compile(sim_step_fn, mode=mode, fullgraph=True)
        self._compiled_policy_forward = torch.compile(policy_forward_fn, mode=mode, fullgraph=True)

    # =========================================================================
    # BATCHED MCTS SEARCH
    # =========================================================================

    def search_batched(
        self,
        env_state: TensorDict,
        obs: Dict[str, Tensor],
        action_mask: Tensor,
        add_noise: bool = True,
        num_simulations: Optional[int] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Run MCTS search for B environments in parallel.

        Outer loop stays in Python (data-dependent branching).
        Inner operations (PUCT selection, backprop) are vectorized.

        Args:
            env_state: Batched environment state [B, ...].
            obs: Batched observations dict.
            action_mask: [B, A] valid actions per environment.
            add_noise: Whether to add exploration noise at root.
            num_simulations: Override number of simulations (uses config if None).

        Returns:
            actions: [B] selected actions.
            stats: Dict with search statistics (root_value, visit_counts).
        """
        B = self.batch_size
        tree = self.mcts_batched.tree
        device = self.device

        # Reset tree tensors for new search
        tree.reset()

        # Convert action mask to bool if needed
        if action_mask.dtype != torch.bool:
            action_mask = action_mask.bool()

        # Initialize root priors from policy
        with torch.no_grad():
            obs_tuple = (obs['sub_index'], obs['derived_sub_indices'], obs['action_mask'])
            if self._compiled_policy_forward is not None:
                priors, root_values = self._compiled_policy_forward(*obs_tuple)
            else:
                logits = self._uncompiled_policy.get_logits(obs)
                root_values = self._uncompiled_policy.predict_values(obs).flatten()
                masked_logits = logits.masked_fill(~action_mask, -3.4e38)
                priors = torch.softmax(masked_logits, dim=-1)

        # Store root priors
        tree.priors[:, 0, :priors.shape[1]] = priors
        tree.expanded_mask[:, 0, :] = action_mask

        # Add exploration noise at root
        if add_noise and self.config.add_exploration_noise:
            self.mcts_batched._add_dirichlet_noise_batched(action_mask)

        # Run N simulations
        num_sims = num_simulations if num_simulations is not None else self.config.num_simulations

        for sim_idx in range(num_sims):
            # Mark step for CUDA graph
            torch.compiler.cudagraph_mark_step_begin()

            # Clone state for simulation
            sim_state = env_state.clone()
            sim_obs = {k: v.clone() for k, v in obs.items()}
            sim_mask = action_mask.clone()

            # Reset path tracking
            self._path_length.zero_()

            # Selection phase: traverse tree to leaf
            depth = torch.zeros(B, dtype=torch.long, device=device)

            for d in range(self.config.max_episode_steps):
                # Check which envs have expanded nodes at this depth
                expanded_at_depth = tree.expanded_mask[self._arange_B, depth]  # [B, A]
                has_expanded = expanded_at_depth.any(dim=1)  # [B]

                if not has_expanded.any():
                    break  # All envs at leaf

                # Select action using PUCT (vectorized)
                actions = self.mcts_batched._select_child_batched(depth, sim_mask)

                # Record path for backprop
                self._path_depths[:, d] = depth
                self._path_actions[:, d] = actions
                self._path_length += has_expanded.long()

                # Environment step (compiled or eager)
                with torch.no_grad():
                    if self._compiled_sim_step is not None:
                        sim_obs_sub, sim_obs_derived, sim_obs_mask, sim_state = self._compiled_sim_step(
                            sim_obs['sub_index'],
                            sim_obs['derived_sub_indices'],
                            sim_obs['action_mask'],
                            sim_state,
                            actions,
                        )
                        sim_obs = {
                            'sub_index': sim_obs_sub,
                            'derived_sub_indices': sim_obs_derived,
                            'action_mask': sim_obs_mask,
                        }
                        sim_mask = sim_obs_mask.bool()
                    else:
                        new_obs, sim_state = self.env.step(sim_state, actions)
                        sim_obs = {k: v.clone() for k, v in new_obs.items()}
                        sim_mask = sim_obs['action_mask'].bool() if 'action_mask' in sim_obs else sim_state['derived_counts'] > 0

                # Check if done
                done = sim_state.get('done', sim_state.get('step_dones', torch.zeros(B, dtype=torch.bool, device=device)))
                if done.bool().all():
                    break

                depth = depth + 1

            # Expansion: get policy priors and values for leaf nodes
            with torch.no_grad():
                obs_tuple = (sim_obs['sub_index'], sim_obs['derived_sub_indices'], sim_obs['action_mask'])
                if self._compiled_policy_forward is not None:
                    leaf_priors, leaf_values = self._compiled_policy_forward(*obs_tuple)
                else:
                    leaf_logits = self._uncompiled_policy.get_logits(sim_obs)
                    leaf_values = self._uncompiled_policy.predict_values(sim_obs).flatten()
                    masked_leaf = leaf_logits.masked_fill(~sim_mask, -3.4e38)
                    leaf_priors = torch.softmax(masked_leaf, dim=-1)

            # Handle terminal states - use success/failure reward
            done_mask = sim_state.get('done', sim_state.get('step_dones', self._zeros_B_long)).bool()
            success = sim_state.get('success', sim_state.get('step_successes', self._zeros_B_long)).bool()
            terminal_values = torch.where(success, self._terminal_pos, self._terminal_neg)
            leaf_values = torch.where(done_mask, terminal_values, leaf_values)

            # Store leaf priors in tree (expand leaf nodes)
            # Vectorized: only expand if not already expanded
            leaf_depth = depth.clamp(max=self.max_depth - 1)

            # Check which envs need expansion (not already expanded at leaf depth)
            # expanded_mask shape: [B, D, A]
            already_expanded = tree.expanded_mask[self._arange_B, leaf_depth].any(dim=1)  # [B]
            needs_expand = ~already_expanded  # [B]

            if needs_expand.any():
                # Use advanced indexing to expand all at once
                expand_idx = needs_expand.nonzero(as_tuple=True)[0]  # indices of envs to expand
                expand_depths = leaf_depth[expand_idx]  # [num_expand]

                # Scatter priors and masks for envs that need expansion
                A = leaf_priors.shape[1]
                tree.priors[expand_idx.unsqueeze(1), expand_depths.unsqueeze(1), self._arange_A[:A]] = leaf_priors[expand_idx]
                tree.expanded_mask[expand_idx.unsqueeze(1), expand_depths.unsqueeze(1), :] = sim_mask[expand_idx].unsqueeze(1)

            # Backpropagation
            self.mcts_batched._backpropagate_batched(
                self._path_depths,
                self._path_actions,
                self._path_length,
                leaf_values,
            )

        # Select final action based on visit counts
        temperature = self.config.get_temperature(self.num_timesteps)
        final_actions = self.mcts_batched._select_action_batched(temperature)

        # Collect statistics
        root_visits = tree.visit_counts[:, 0, :].float()
        root_values_mean = torch.where(
            root_visits.sum(dim=1) > 0,
            tree.value_sums[:, 0, :].sum(dim=1) / root_visits.sum(dim=1).clamp(min=1),
            self._zeros_B_float
        )

        stats = {
            "root_value": root_values_mean,
            "visit_counts": root_visits,
            "root_visits_total": root_visits.sum(dim=1),
        }

        return final_actions, stats

    # =========================================================================
    # BATCHED EPISODE COLLECTION
    # =========================================================================

    def collect_episodes_batched(
        self,
        num_steps: int,
        add_noise: bool = True,
    ) -> Dict[str, float]:
        """Collect episodes from B parallel environments.

        Vectorized episode collection using batched MCTS search.
        Avoids .item() calls in hot paths - uses tensor operations.

        Args:
            num_steps: Total steps to collect across all envs.
            add_noise: Whether to add exploration noise.

        Returns:
            Dict with collection statistics.
        """
        self.policy.eval()
        B = self.batch_size
        device = self.device

        # Reset all B environments
        obs, state = self.env.reset()

        # Accumulators (stay on GPU)
        steps_collected = 0
        total_rewards = torch.zeros(B, dtype=torch.float32, device=device)
        episode_lengths = torch.zeros(B, dtype=torch.long, device=device)
        episode_count = torch.zeros(1, dtype=torch.long, device=device)
        success_count = torch.zeros(1, dtype=torch.long, device=device)

        while steps_collected < num_steps:
            torch.compiler.cudagraph_mark_step_begin()

            # Get action mask
            action_mask = obs.get('action_mask', state['derived_counts'] > 0)
            if action_mask.dtype != torch.bool:
                action_mask = action_mask.bool()

            # Run MCTS search for all B environments
            with torch.no_grad():
                actions, search_stats = self.search_batched(state, obs, action_mask, add_noise=add_noise)

            # Store transition data (GPU tensors, no .item())
            idx_start = steps_collected
            idx_end = min(steps_collected + B, len(self._rollout_actions))
            actual_B = idx_end - idx_start

            if actual_B > 0:
                self._rollout_obs_sub[idx_start:idx_end] = obs['sub_index'][:actual_B]
                self._rollout_obs_derived[idx_start:idx_end] = obs['derived_sub_indices'][:actual_B]
                self._rollout_obs_mask[idx_start:idx_end] = obs['action_mask'][:actual_B].bool()
                self._rollout_actions[idx_start:idx_end] = actions[:actual_B]
                self._rollout_values[idx_start:idx_end] = search_stats['root_value'][:actual_B]
                self._rollout_visit_counts[idx_start:idx_end] = search_stats['visit_counts'][:actual_B]

            # Environment step with auto-reset
            if hasattr(self.env, 'step_and_reset'):
                new_obs, new_state = self.env.step_and_reset(
                    state, actions, self.env.query_pool, self.env.per_env_ptrs
                )
            else:
                new_obs, new_state = self.env.step(state, actions)

            # Track rewards and dones (stay on GPU)
            step_rewards = new_state.get('step_rewards', torch.zeros(B, device=device))
            step_dones = new_state.get('step_dones', new_state.get('done', torch.zeros(B, dtype=torch.bool, device=device))).bool()

            if actual_B > 0:
                self._rollout_rewards[idx_start:idx_end] = step_rewards[:actual_B]
                self._rollout_dones[idx_start:idx_end] = step_dones[:actual_B]

            # Accumulate episode stats
            total_rewards += step_rewards
            episode_lengths += 1

            # Handle done episodes
            done_mask = step_dones
            if done_mask.any():
                step_success = new_state.get('step_successes', new_state.get('success', torch.zeros(B, dtype=torch.bool, device=device))).bool()
                episode_count += done_mask.sum()
                success_count += (done_mask & step_success).sum()

                # Reset accumulators for done envs
                total_rewards.masked_fill_(done_mask, 0)
                episode_lengths.masked_fill_(done_mask, 0)

            steps_collected += B
            obs, state = new_obs, new_state

        # Single GPU->CPU sync at end for logging
        ep_count = episode_count.item()
        succ_count = success_count.item()

        return {
            "steps_collected": steps_collected,
            "episodes_completed": ep_count,
            "success_rate": succ_count / max(1, ep_count),
            "mean_episode_length": steps_collected / max(1, ep_count),
        }

    def collect_episodes(
        self,
        num_episodes: int,
        add_noise: bool = True,
    ) -> Dict[str, float]:
        """Collect episodes using MCTS for action selection.

        Runs MCTS search at each step to select actions, collecting
        trajectories with search statistics for training.

        Args:
            num_episodes: Number of episodes to collect.
            add_noise: Whether to add exploration noise at root.

        Returns:
            Dict with collection statistics.
        """
        self.policy.eval()
        total_rewards = []
        total_lengths = []
        total_success = []

        for ep_idx in range(num_episodes):
            trajectory = self._collect_single_episode(add_noise=add_noise)
            self.replay_buffer.add_trajectory(trajectory)
            self.num_episodes += 1

            total_rewards.append(trajectory.total_reward)
            total_lengths.append(trajectory.length)
            # Check if episode ended successfully
            if trajectory.transitions:
                total_success.append(trajectory.transitions[-1].reward > 0)

        self.num_timesteps += sum(total_lengths)

        return {
            "episodes_collected": num_episodes,
            "mean_reward": sum(total_rewards) / len(total_rewards) if total_rewards else 0.0,
            "mean_length": sum(total_lengths) / len(total_lengths) if total_lengths else 0.0,
            "success_rate": sum(total_success) / len(total_success) if total_success else 0.0,
            "total_transitions": sum(total_lengths),
        }

    def _collect_single_episode(
        self,
        add_noise: bool = True,
    ) -> MCTSTrajectory:
        """Collect single episode trajectory using MCTS.

        Args:
            add_noise: Whether to add exploration noise at root.

        Returns:
            Completed trajectory with MCTS statistics.
        """
        trajectory = MCTSTrajectory()

        # Ensure environment is in training mode
        if hasattr(self.env, 'train'):
            self.env.train()

        # Reset environment
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            obs, state = reset_result
        else:
            obs = reset_result
            state = getattr(self.env, "_state", None)

        # Clone observation tensors
        obs = {k: v.clone() if isinstance(v, Tensor) else v for k, v in obs.items()}

        done = False
        step = 0

        while not done and step < self.config.max_episode_steps:
            # Get action mask
            action_mask = obs.get("action_mask", None)
            if action_mask is None and state is not None:
                action_mask = state.get("derived_counts", None)
            if action_mask is None:
                break

            if action_mask.dim() == 2:
                action_mask = action_mask.squeeze(0)

            # Run MCTS search
            # Adjust noise based on temperature schedule
            temperature = self.config.get_temperature(self.num_timesteps)
            use_noise = add_noise and temperature > 0.1

            action, search_stats = self.mcts.search(
                env=self.env,
                env_state=state,
                networks=self.policy,
                obs=obs,
                action_mask=action_mask,
                add_noise=use_noise,
            )

            # Store transition (before taking action)
            trans = MCTSTransition(
                obs={k: v.clone() if isinstance(v, Tensor) else v for k, v in obs.items()},
                action=action,
                reward=0.0,  # Updated after step
                done=False,
                visit_counts=search_stats.get("visit_counts", {}),
                root_value=search_stats.get("root_value", 0.0),
            )

            # Take action in environment
            action_tensor = torch.tensor([action], device=self.device)
            step_result = self.env.step(state, action_tensor)

            if isinstance(step_result, tuple):
                new_obs, new_state = step_result
            else:
                new_obs = step_result
                new_state = getattr(self.env, "_state", state)

            # Extract reward and done from state
            reward_tensor = new_state.get("step_rewards", new_state.get("rewards", torch.zeros(1)))
            done_tensor = new_state.get("done", new_state.get("step_dones", torch.zeros(1, dtype=torch.bool)))

            reward = reward_tensor.item() if isinstance(reward_tensor, Tensor) and reward_tensor.numel() == 1 else float(reward_tensor[0] if isinstance(reward_tensor, Tensor) else reward_tensor)
            done = done_tensor.item() if isinstance(done_tensor, Tensor) and done_tensor.numel() == 1 else bool(done_tensor[0] if isinstance(done_tensor, Tensor) else done_tensor)

            # Update transition with reward and done
            trans.reward = reward
            trans.done = done

            trajectory.add(trans)

            # Update for next iteration
            obs = {k: v.clone() if isinstance(v, Tensor) else v for k, v in new_obs.items()}
            state = new_state.clone() if hasattr(new_state, 'clone') else new_state
            step += 1

        return trajectory

    def train_step(self) -> Dict[str, float]:
        """Perform single training step on sampled batch.

        Samples batch from replay buffer and updates network
        using cross-entropy loss for policy and MSE for value.

        Returns:
            Dict with loss metrics.
        """
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return {"train_step_skipped": True}

        self.policy.train()

        # Sample batch from replay buffer
        batch = self.replay_buffer.sample_batch(
            batch_size=self.config.batch_size,
            sequence_length=1,  # Single-step for now
        )

        obs = batch["obs"]
        actions = batch["actions"]
        policy_targets = batch["policy_targets"]
        value_targets = batch["value_targets"]

        # Forward pass
        logits = self.policy.get_logits(obs)  # [B, S]
        values = self.policy.predict_values(obs)  # [B]

        # Policy loss: cross-entropy with MCTS visit distribution
        # Mask logits to valid actions (where policy_targets > 0 or from action_mask)
        action_mask = obs.get("action_mask", None)
        if action_mask is not None:
            # Ensure same shape
            if logits.shape[-1] != policy_targets.shape[-1]:
                # Pad policy targets to match logits
                if policy_targets.shape[-1] < logits.shape[-1]:
                    padding = torch.zeros(
                        policy_targets.shape[0],
                        logits.shape[-1] - policy_targets.shape[-1],
                        device=self.device,
                    )
                    policy_targets = torch.cat([policy_targets, padding], dim=-1)
                else:
                    policy_targets = policy_targets[:, :logits.shape[-1]]

            masked_logits = logits.masked_fill(~action_mask.bool(), float("-inf"))
        else:
            masked_logits = logits

        # Compute log probabilities with numerical stability
        log_probs = F.log_softmax(masked_logits, dim=-1)

        # Replace -inf with large negative value to avoid NaN
        log_probs = torch.where(
            torch.isinf(log_probs),
            torch.full_like(log_probs, -100.0),
            log_probs
        )

        # Cross-entropy loss: -sum(target * log_prob)
        # Only compute where policy_targets > 0
        policy_loss = -(policy_targets * log_probs).sum(dim=-1).mean()

        # Handle NaN case
        if torch.isnan(policy_loss):
            policy_loss = torch.tensor(0.0, device=self.device)

        # Value loss: MSE
        # Ensure values and targets have same shape
        values_flat = values.flatten()
        value_loss = F.mse_loss(values_flat, value_targets)

        # Total loss
        total_loss = (
            self.policy_loss_weight * policy_loss
            + self.value_loss_weight * value_loss
        )

        # Backward and optimize
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        if self.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm
            )
        else:
            grad_norm = torch.tensor(0.0)

        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm,
        }

    def learn(
        self,
        total_timesteps: int,
        reset_num_timesteps: bool = True,
        on_iteration_start: Optional[Callable] = None,
        on_iteration_end: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Main training loop.

        Alternates between:
        1. Collecting episodes with MCTS
        2. Training network on collected data

        Args:
            total_timesteps: Total environment steps to train.
            reset_num_timesteps: Whether to reset timestep counter.
            on_iteration_start: Callback at start of each iteration.
            on_iteration_end: Callback at end of each iteration.

        Returns:
            Dict with training results.
        """
        if reset_num_timesteps:
            self.num_timesteps = 0
            self.iteration = 0

        episode_rewards = []
        episode_lengths = []
        train_losses = []

        if self.callback and hasattr(self.callback, 'on_training_start'):
            self.callback.on_training_start(total_timesteps=total_timesteps)

        while self.num_timesteps < total_timesteps:
            self.iteration += 1
            iter_start_time = time.time()

            if on_iteration_start:
                on_iteration_start(self.iteration, self.num_timesteps)

            if self.callback and hasattr(self.callback, 'on_iteration_start'):
                self.callback.on_iteration_start(self.iteration, self.num_timesteps)

            # Collect episodes with MCTS
            collect_stats = self.collect_episodes(
                num_episodes=self.config.episodes_per_iteration,
                add_noise=True,
            )

            episode_rewards.append(collect_stats["mean_reward"])
            episode_lengths.append(collect_stats["mean_length"])

            # Train on collected data
            train_metrics = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}
            num_train_steps = max(1, self.config.episodes_per_iteration // 4)

            for _ in range(num_train_steps):
                step_metrics = self.train_step()
                if "train_step_skipped" not in step_metrics:
                    for k, v in step_metrics.items():
                        if k in train_metrics:
                            train_metrics[k] += v / num_train_steps

            train_losses.append(train_metrics["total_loss"])
            self.last_train_metrics = train_metrics

            iter_time = time.time() - iter_start_time

            # Logging
            if self.verbose and self.iteration % self.config.log_interval == 0:
                print(
                    f"[MCTS] Iter {self.iteration} | "
                    f"Timesteps: {self.num_timesteps}/{total_timesteps} | "
                    f"Episodes: {self.num_episodes} | "
                    f"Mean Reward: {collect_stats['mean_reward']:.3f} | "
                    f"Success: {collect_stats['success_rate']:.1%} | "
                    f"Loss: {train_metrics['total_loss']:.4f} | "
                    f"Time: {iter_time:.2f}s"
                )

            # Callbacks
            if on_iteration_end:
                on_iteration_end(self.iteration, self.num_timesteps, collect_stats, train_metrics)

            if self.callback:
                callback_locals = {
                    "iteration": self.iteration,
                    "total_steps_done": self.num_timesteps,
                    "train_metrics": train_metrics,
                    "collect_stats": collect_stats,
                }
                self.callback(callback_locals, globals())

        if self.callback and hasattr(self.callback, 'on_training_end'):
            self.callback.on_training_end()

        return {
            "num_timesteps": self.num_timesteps,
            "num_episodes": self.num_episodes,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "train_losses": train_losses,
            "last_train_metrics": self.last_train_metrics,
            "buffer_stats": self.replay_buffer.get_statistics(),
        }

    def _evaluate_single_query(
        self,
        query: Tensor,
        device: torch.device,
    ) -> Tuple[bool, float, int]:
        """Evaluate a single query using greedy MCTS.

        Args:
            query: Query tensor [1, 3].
            device: Device for tensor operations.

        Returns:
            Tuple of (success, root_value, depth).
        """
        # Create initial state
        reset_result = self.env.reset(query)
        if isinstance(reset_result, tuple):
            obs, state = reset_result
        else:
            obs = reset_result
            state = getattr(self.env, "_state", None)

        obs = {k: v.clone() if isinstance(v, Tensor) else v for k, v in obs.items()}

        done = False
        step = 0
        initial_root_value = 0.0

        while not done and step < self.config.max_episode_steps:
            action_mask = obs.get("action_mask", state.get("derived_counts", None) if state is not None else None)
            if action_mask is None:
                break

            if action_mask.dim() == 2:
                action_mask = action_mask.squeeze(0)

            # Greedy MCTS (no noise)
            action, search_stats = self.mcts.search(
                env=self.env,
                env_state=state,
                networks=self.policy,
                obs=obs,
                action_mask=action_mask,
                add_noise=False,
            )

            # Capture initial root value for scoring
            if step == 0:
                initial_root_value = search_stats.get("root_value", 0.0)

            action_tensor = torch.tensor([action], device=device)
            step_result = self.env.step(state, action_tensor)

            if isinstance(step_result, tuple):
                obs, state = step_result
            else:
                obs = step_result
                state = getattr(self.env, "_state", state)

            obs = {k: v.clone() if isinstance(v, Tensor) else v for k, v in obs.items()}

            done_tensor = state.get("done", state.get("step_dones", torch.zeros(1, dtype=torch.bool)))
            done = done_tensor.item() if isinstance(done_tensor, Tensor) and done_tensor.numel() == 1 else bool(done_tensor[0])
            step += 1

        # Check success
        success_tensor = state.get("success", torch.zeros(1, dtype=torch.bool))
        success = success_tensor.item() if isinstance(success_tensor, Tensor) and success_tensor.numel() == 1 else bool(success_tensor[0])

        return success, initial_root_value, step

    def evaluate(
        self,
        queries: Tensor,
        sampler: Any,
        n_corruptions: int = 50,
        corruption_modes: Tuple[str, ...] = ('head', 'tail'),
        verbose: bool = False,
        **kwargs,
    ) -> Dict[str, float]:
        """Evaluate model on ranking task with proper MRR computation.

        Uses greedy MCTS (no noise) for evaluation. Generates corruptions
        using the sampler and ranks candidates by proof success (primary)
        and root value (secondary for tie-breaking).

        Compatible with PPO.evaluate() interface.

        Args:
            queries: Test queries [N, 3].
            sampler: Corruption sampler with corrupt() method.
            n_corruptions: Number of negative samples per query.
            corruption_modes: Corruption modes to use ('head', 'tail').
            verbose: Whether to print progress.
            **kwargs: Additional arguments (for compatibility).

        Returns:
            Dict with MRR, Hits@K metrics and detailed statistics.
        """
        self.policy.eval()
        device = self.device
        N = queries.shape[0]

        # Accumulators for ranking
        all_ranks = {mode: [] for mode in corruption_modes}

        # Accumulators for detailed stats
        pos_successes = []
        pos_depths = []
        neg_successes = []
        neg_depths = []

        if verbose:
            print(f"[MCTS] Evaluating {N} queries with {n_corruptions} corruptions per mode")

        with torch.no_grad():
            for q_idx in range(N):
                if verbose and (q_idx + 1) % 10 == 0:
                    print(f"[MCTS] Processing query {q_idx + 1}/{N}")

                query = queries[q_idx:q_idx+1]  # [1, 3]

                # Evaluate positive query
                pos_success, pos_value, pos_depth = self._evaluate_single_query(query, device)
                pos_successes.append(pos_success)
                pos_depths.append(pos_depth)

                # Process each corruption mode
                for mode in corruption_modes:
                    # Generate corruptions for this query
                    corruptions = sampler.corrupt(
                        query,
                        num_negatives=n_corruptions,
                        mode=mode,
                        device=device,
                    )  # [1, n_corruptions, 3] or [n_corruptions, 3]

                    if corruptions.dim() == 3:
                        corruptions = corruptions.squeeze(0)  # [n_corruptions, 3]

                    # Evaluate each corruption
                    neg_results = []
                    for c_idx in range(corruptions.shape[0]):
                        corruption = corruptions[c_idx:c_idx+1]  # [1, 3]
                        neg_success, neg_value, neg_depth = self._evaluate_single_query(corruption, device)
                        neg_results.append((neg_success, neg_value, neg_depth))
                        neg_successes.append(neg_success)
                        neg_depths.append(neg_depth)

                    # Compute rank for this query in this mode
                    # Primary ranking: proof success (proven > not proven)
                    # Secondary: root value (higher is better) for tie-breaking
                    rank = 1
                    for neg_success, neg_value, _ in neg_results:
                        # Negative beats positive if:
                        # 1. Negative proved and positive didn't, OR
                        # 2. Both have same success status but negative has higher value
                        if neg_success and not pos_success:
                            rank += 1
                        elif neg_success == pos_success and neg_value > pos_value:
                            rank += 1
                        elif neg_success == pos_success and neg_value == pos_value:
                            # Random tie-breaking: 50% chance negative wins
                            if torch.rand(1).item() > 0.5:
                                rank += 1

                    all_ranks[mode].append(rank)

        # Compute metrics
        proven_pos = sum(pos_successes) / len(pos_successes) if pos_successes else 0.0
        proven_neg = sum(neg_successes) / len(neg_successes) if neg_successes else 0.0
        mean_depth_pos = sum(pos_depths) / len(pos_depths) if pos_depths else 0.0
        mean_depth_neg = sum(neg_depths) / len(neg_depths) if neg_depths else 0.0

        # Aggregate ranking metrics across modes
        all_ranks_combined = []
        for mode in corruption_modes:
            all_ranks_combined.extend(all_ranks[mode])

        if all_ranks_combined:
            ranks_tensor = torch.tensor(all_ranks_combined, dtype=torch.float32)
            mrr = (1.0 / ranks_tensor).mean().item()
            hits_1 = (ranks_tensor <= 1.0).float().mean().item()
            hits_3 = (ranks_tensor <= 3.0).float().mean().item()
            hits_10 = (ranks_tensor <= 10.0).float().mean().item()
        else:
            mrr, hits_1, hits_3, hits_10 = 0.0, 0.0, 0.0, 0.0

        # Per-mode metrics
        result = {
            "MRR": mrr,
            "Hits@1": hits_1,
            "Hits@3": hits_3,
            "Hits@10": hits_10,
            "proven_pos": proven_pos,
            "proven_neg": proven_neg,
            "len_pos": mean_depth_pos,
            "len_neg": mean_depth_neg,
        }

        # Add per-mode MRR
        for mode in corruption_modes:
            if all_ranks[mode]:
                mode_ranks = torch.tensor(all_ranks[mode], dtype=torch.float32)
                result[f"MRR_{mode}"] = (1.0 / mode_ranks).mean().item()
                result[f"Hits@1_{mode}"] = (mode_ranks <= 1.0).float().mean().item()

        if verbose:
            print(f"[MCTS] Evaluation complete: MRR={mrr:.4f}, Hits@1={hits_1:.4f}, "
                  f"proven_pos={proven_pos:.4f}, proven_neg={proven_neg:.4f}")

        return result

    # =========================================================================
    # BATCHED EVALUATION WITH SLOT RECYCLING (PPO Pattern)
    # =========================================================================

    def _setup_eval_pool(
        self,
        queries: Tensor,
        sampler: Any,
        n_corruptions: int,
        modes: Tuple[str, ...],
    ) -> int:
        """Setup evaluation pool with strided layout for slot recycling.

        Pool layout (strided, matches PPO _setup_ranking_pool):
        - Stride = N (number of queries)
        - Pool = [pos_q0, pos_q1, ..., pos_qN-1,
                  neg0_q0, neg0_q1, ..., neg0_qN-1,
                  neg1_q0, neg1_q1, ..., neg1_qN-1, ...]
        - Each env starts at position 0..B-1
        - When done, advances by stride N to get next candidate for same query

        Args:
            queries: Test queries [N, 3].
            sampler: Corruption sampler.
            n_corruptions: Number of negatives per query.
            modes: Corruption modes.

        Returns:
            Pool size (total candidates).
        """
        N = queries.shape[0]
        device = self.device
        B = self.fixed_batch_size
        H = getattr(self.env, 'max_history_size', self.max_depth + 1)
        pad_idx = getattr(self.env, 'padding_idx', 0)

        pool_offset = 0
        for mode in modes:
            # Generate corruptions for all queries at once
            neg = sampler.corrupt(queries, num_negatives=n_corruptions, mode=mode, device=device)
            neg_count = neg.shape[1] if neg.dim() > 1 else 0

            K = 1 + neg_count  # Positive + negatives
            pool_end = pool_offset + N * K

            if pool_end > self._eval_max_pool:
                raise ValueError(f"Pool size {pool_end} exceeds max {self._eval_max_pool}")

            # Layout: [K, N, 3] then reshape to [K*N, 3]
            pool_slice = self._eval_pool[pool_offset:pool_end].view(K, N, 3)
            pool_slice[0].copy_(queries)  # Positives first
            if neg_count > 0:
                pool_slice[1:].copy_(neg.transpose(0, 1))  # Negatives

            pool_offset = pool_end

        # Set pool size and stride
        pool_size = pool_offset
        self._eval_pool[pool_size:].fill_(pad_idx)
        self._eval_pool_size.fill_(pool_size)
        self._eval_stride.fill_(N)  # Stride = N queries

        # Reset result buffers
        self._eval_results_success[:pool_size].zero_()
        self._eval_results_value[:pool_size].zero_()
        self._eval_results_depth[:pool_size].zero_()

        # Initialize first B slots using strided indices
        init_idx = self._eval_arange_B.clamp(max=max(0, N - 1))
        initial_queries = self._eval_pool[init_idx]  # [B, 3]

        # Reset env with initial queries
        init_state = self.env.reset_from_queries(initial_queries, self._eval_reset_labels)

        # Copy to buffer 0
        buf0 = self._eval_buffers[0]
        self._eval_buf_idx = 0

        torch._foreach_copy_(
            [
                buf0["current"],
                buf0["derived"],
                buf0["counts"],
                buf0["original_queries"],
                buf0["history_hashes"],
                buf0["history_count"],
                buf0["next_var"],
            ],
            [
                init_state['current_states'],
                init_state['derived_states'],
                init_state['derived_counts'],
                init_state['original_queries'],
                init_state['history_hashes'],
                init_state['history_count'],
                init_state['next_var_indices'],
            ],
        )

        # Update mask based on counts
        buf0["mask"].copy_(self._eval_arange_S.unsqueeze(0) < init_state['derived_counts'].unsqueeze(1))
        buf0["depths"].zero_()
        buf0["done"].zero_()
        buf0["success"].zero_()
        buf0["root_value"].zero_()
        buf0["first_step"].fill_(True)

        # Mark slots beyond N as done (padding)
        if N < B:
            buf0["done"][N:].fill_(True)

        # Pool pointers: env i starts at position i
        buf0["pool_ptr"].copy_(self._eval_arange_B)

        return pool_size

    def _build_eval_step(self, cur_buf: Dict, next_buf: Dict, use_mcts_search: bool = False):
        """Build evaluation step function for slot recycling.

        Similar to PPO's _build_step but can use MCTS search for action selection.

        Args:
            cur_buf: Current buffer dict.
            next_buf: Next buffer dict (for double-buffering).
            use_mcts_search: If True, use full MCTS search for action selection.
                            If False, use direct policy forward (faster, like PPO).

        Returns:
            Step function that returns done mask.
        """
        B = self.fixed_batch_size
        S = self.max_actions
        H = getattr(self.env, 'max_history_size', self.max_depth + 1)
        device = self.device
        env = self.env
        max_pool = self._eval_max_pool

        # Unpack buffer references (for closure)
        cur_current = cur_buf["current"]
        cur_derived = cur_buf["derived"]
        cur_counts = cur_buf["counts"]
        cur_mask = cur_buf["mask"]
        cur_depths = cur_buf["depths"]
        cur_done = cur_buf["done"]
        cur_pool_ptr = cur_buf["pool_ptr"]
        cur_history = cur_buf["history_hashes"]
        cur_h_count = cur_buf["history_count"]
        cur_original = cur_buf["original_queries"]
        cur_next_var = cur_buf["next_var"]
        cur_root_value = cur_buf["root_value"]
        cur_first_step = cur_buf["first_step"]

        next_current = next_buf["current"]
        next_derived = next_buf["derived"]
        next_counts = next_buf["counts"]
        next_mask = next_buf["mask"]
        next_depths = next_buf["depths"]
        next_done = next_buf["done"]
        next_pool_ptr = next_buf["pool_ptr"]
        next_history = next_buf["history_hashes"]
        next_h_count = next_buf["history_count"]
        next_original = next_buf["original_queries"]
        next_next_var = next_buf["next_var"]
        next_root_value = next_buf["root_value"]
        next_first_step = next_buf["first_step"]

        def step():
            # Build current state TensorDict
            cur_state = TensorDict({
                "current_states": cur_current,
                "derived_states": cur_derived,
                "derived_counts": cur_counts,
                "original_queries": cur_original,
                "next_var_indices": cur_next_var,
                "depths": cur_depths,
                "done": cur_done,
                "success": cur_buf["success"],
                "current_labels": self._eval_reset_labels,
                "history_hashes": cur_history,
                "history_count": cur_h_count,
                "step_rewards": self._eval_zero_float,
                "step_dones": self._eval_zero_uint8,
                "step_successes": self._eval_zero_uint8,
                "step_labels": self._eval_zero_long,
                "cumulative_rewards": self._eval_zero_float,
                "per_env_ptrs": cur_pool_ptr,
                "neg_counters": self._eval_zero_int64,
                "corruption_mode_counters": self._eval_zero_int64,
            }, batch_size=[B], device=device)

            # Build observation for policy
            cur_obs = {
                'sub_index': cur_current.unsqueeze(1),
                'derived_sub_indices': cur_derived,
                'action_mask': cur_mask,
            }

            active = ~cur_done

            if use_mcts_search:
                # Use full MCTS search for action selection (more accurate, slower)
                # Skip search entirely if all environments are done
                if active.any():
                    # Use eval_num_simulations for faster evaluation
                    eval_sims = self.config.eval_num_simulations or self.config.num_simulations
                    actions, search_stats = self.search_batched(
                        cur_state, cur_obs, cur_mask, add_noise=False,
                        num_simulations=eval_sims
                    )
                    values = search_stats['root_value']
                else:
                    # All done - use dummy actions (won't affect results)
                    actions = self._eval_zero_long[:B]
                    values = self._eval_zero_float[:B]
            else:
                # Use direct policy forward (faster, like PPO)
                logits = self._uncompiled_policy.get_logits(cur_obs)
                masked_logits = logits.masked_fill(~cur_mask, -3.4e38)
                actions = masked_logits.argmax(dim=-1)
                values = self._uncompiled_policy.predict_values(cur_obs).flatten()

            # Track root value on first step
            new_root_value = torch.where(cur_first_step & active, values, cur_root_value)

            # Environment step
            _, new_state = env.step(cur_state, actions)

            # Extract new state components
            new_current = new_state['current_states']
            new_derived = new_state['derived_states']
            new_counts = new_state['derived_counts']
            new_depths = new_state['depths']
            new_done_state = new_state['done'].bool()
            new_success = new_state['success'].bool()
            new_history = new_state['history_hashes']
            new_h_count = new_state['history_count']
            new_original = new_state['original_queries']
            new_next_var = new_state['next_var_indices']
            step_dones = new_state['step_dones'].bool()

            # Slot recycling logic (matches PPO pattern)
            finished_idx = torch.where(step_dones, cur_pool_ptr, self._eval_minus_one)
            new_ptr = torch.where(step_dones, cur_pool_ptr + self._eval_stride, cur_pool_ptr)
            needs_reset = step_dones & (new_ptr < self._eval_pool_size)

            # Get reset queries for recycled slots
            safe_idx = new_ptr.clamp(0, max_pool - 1)
            reset_queries = self._eval_pool[safe_idx]
            reset_state = env.reset_from_queries(reset_queries, self._eval_reset_labels)
            reset_done = reset_state['done'].bool()

            # Merge reset state with continuing state
            m1 = needs_reset.view(B, 1, 1)
            m3 = needs_reset.view(B, 1, 1, 1)
            mH = needs_reset.view(B, 1).expand(B, H)

            torch.where(m1, reset_state['current_states'], new_current, out=next_current)
            torch.where(m3, reset_state['derived_states'], new_derived, out=next_derived)
            torch.where(needs_reset, reset_state['derived_counts'], new_counts, out=next_counts)
            torch.where(needs_reset, reset_state['depths'], new_depths, out=next_depths)
            torch.where(needs_reset, reset_done, new_done_state, out=next_done)
            torch.where(mH, reset_state['history_hashes'], new_history, out=next_history)
            torch.where(needs_reset, reset_state['history_count'], new_h_count, out=next_h_count)
            torch.where(m1, reset_state['original_queries'], new_original, out=next_original)
            torch.where(needs_reset, reset_state['next_var_indices'], new_next_var, out=next_next_var)
            torch.where(needs_reset, self._eval_zero_float, new_root_value, out=next_root_value)
            torch.where(needs_reset, self._eval_true_bool, self._eval_false_bool, out=next_first_step)

            # Handle exhausted slots (no more candidates in pool)
            exhausted = (new_ptr >= self._eval_pool_size) & step_dones
            torch.where(exhausted, self._eval_true_bool, next_done, out=next_done)

            # Update pool pointers
            next_pool_ptr.copy_(new_ptr)

            # Update mask based on counts
            torch.lt(self._eval_arange_S.unsqueeze(0), next_counts.unsqueeze(1), out=next_mask)

            # Store results for finished episodes
            valid_finish = step_dones & (finished_idx >= 0) & (finished_idx < self._eval_pool_size)
            clamped_indices = torch.minimum(
                torch.maximum(finished_idx, self._eval_zero_long),
                self._eval_pool_size - 1,
            )

            final_success = torch.where(valid_finish, new_success, self._eval_results_success[clamped_indices])
            final_value = torch.where(valid_finish, new_root_value, self._eval_results_value[clamped_indices])
            final_depth = torch.where(valid_finish, new_depths, self._eval_results_depth[clamped_indices])

            self._eval_results_success.scatter_(0, clamped_indices, final_success)
            self._eval_results_value.scatter_(0, clamped_indices, final_value)
            self._eval_results_depth.scatter_(0, clamped_indices, final_depth)

            # Copy success to next buffer
            next_buf["success"].copy_(torch.where(needs_reset, self._eval_false_bool, new_success))

            return next_done

        return step

    def evaluate_batched(
        self,
        queries: Tensor,
        sampler: Any,
        n_corruptions: int = 50,
        corruption_modes: Tuple[str, ...] = ('head', 'tail'),
        verbose: bool = False,
        use_mcts_search: bool = False,
        **kwargs,
    ) -> Dict[str, float]:
        """Batched evaluation with slot recycling for CUDA graph efficiency.

        Follows PPO's evaluate() pattern:
        1. Setup pool with strided layout
        2. Initialize first B slots
        3. Run step loop with slot recycling
        4. Compute ranking metrics from results

        Args:
            queries: Test queries [N, 3].
            sampler: Corruption sampler with corrupt() method.
            n_corruptions: Number of negative samples per query.
            corruption_modes: Corruption modes to use ('head', 'tail').
            verbose: Whether to print progress.
            use_mcts_search: If True, use full MCTS search for action selection.
                            If False, use direct policy forward (faster, like PPO).
            **kwargs: Additional arguments (for compatibility).

        Returns:
            Dict with MRR, Hits@K metrics and detailed statistics.
        """
        if not self.use_batched_mcts:
            return self.evaluate(queries, sampler, n_corruptions, corruption_modes, verbose, **kwargs)

        self.policy.eval()
        device = self.device
        N = queries.shape[0]
        K = 1 + n_corruptions
        B = self.fixed_batch_size
        nm = len(corruption_modes)

        if verbose:
            search_mode = "MCTS search" if use_mcts_search else "direct policy"
            print(f"[MCTS] Evaluating {N} queries with {n_corruptions} corruptions per mode ({search_mode})")

        # Setup pool with strided layout
        pool_size = self._setup_eval_pool(queries, sampler, n_corruptions, corruption_modes)

        if verbose:
            print(f"[MCTS] Pool size: {pool_size}, stride: {N}")

        # Build step functions for double-buffering
        buf0, buf1 = self._eval_buffers
        step_ab = self._build_eval_step(buf0, buf1, use_mcts_search=use_mcts_search)
        step_ba = self._build_eval_step(buf1, buf0, use_mcts_search=use_mcts_search)
        cur_idx = self._eval_buf_idx

        # Main evaluation loop
        max_steps = (pool_size // B + 2) * self.max_depth

        with torch.no_grad():
            for step_idx in range(max_steps):
                torch.compiler.cudagraph_mark_step_begin()

                # Alternate between buffers
                done = step_ab() if cur_idx == 0 else step_ba()
                cur_idx ^= 1

                # Early termination check (periodic to minimize syncs)
                if step_idx % 50 == 49 and done.all():
                    break

        self._eval_buf_idx = cur_idx

        # Compute ranking metrics
        return self._compute_ranking_metrics_batched(
            N, K, corruption_modes, pool_size, verbose
        )

    def _compute_ranking_metrics_batched(
        self,
        N: int,
        K: int,
        corruption_modes: Tuple[str, ...],
        pool_size: int,
        verbose: bool,
    ) -> Dict[str, float]:
        """Compute MRR and Hits@K from batched evaluation results.

        Pool layout is strided: [pos_0..pos_N-1, neg0_0..neg0_N-1, ...]
        So for query q_idx:
        - Positive at index q_idx
        - Negative k at index (k+1)*N + q_idx

        Args:
            N: Number of original queries.
            K: Candidates per query (1 positive + n_corruptions).
            corruption_modes: Corruption modes used.
            pool_size: Total candidates in pool.
            verbose: Whether to print progress.

        Returns:
            Dict with MRR, Hits@K metrics.
        """
        device = self.device
        nm = len(corruption_modes)

        # Extract results
        success = self._eval_results_success[:pool_size]
        values = self._eval_results_value[:pool_size]
        depths = self._eval_results_depth[:pool_size]

        # Compute ranks for each (query, mode) pair
        all_ranks = {mode: [] for mode in corruption_modes}
        pos_successes = []
        pos_depths = []
        neg_successes = []
        neg_depths = []

        # Process each mode's results
        # Pool layout per mode: [K, N] candidates, stored as K*N contiguous
        candidates_per_mode = N * K

        for mode_idx, mode in enumerate(corruption_modes):
            mode_offset = mode_idx * candidates_per_mode

            # Reshape to [K, N] layout
            mode_success = success[mode_offset:mode_offset + candidates_per_mode].view(K, N)
            mode_values = values[mode_offset:mode_offset + candidates_per_mode].view(K, N)
            mode_depths = depths[mode_offset:mode_offset + candidates_per_mode].view(K, N)

            # Transpose to [N, K] for per-query processing
            mode_success = mode_success.t()  # [N, K]
            mode_values = mode_values.t()    # [N, K]
            mode_depths = mode_depths.t()    # [N, K]

            # First column is positive, rest are negatives
            pos_success = mode_success[:, 0]   # [N]
            pos_value = mode_values[:, 0]      # [N]
            pos_depth = mode_depths[:, 0]      # [N]

            neg_success = mode_success[:, 1:]  # [N, K-1]
            neg_values = mode_values[:, 1:]    # [N, K-1]
            neg_depths_t = mode_depths[:, 1:]  # [N, K-1]

            # Accumulate stats
            pos_successes.extend(pos_success.tolist())
            pos_depths.extend(pos_depth.tolist())
            neg_successes.extend(neg_success.flatten().tolist())
            neg_depths.extend(neg_depths_t.flatten().tolist())

            # Vectorized rank computation for all N queries
            # Negative beats positive if:
            # 1. neg proved and pos didn't, OR
            # 2. same proof status but neg has higher value
            pos_success_exp = pos_success.unsqueeze(1)  # [N, 1]
            pos_value_exp = pos_value.unsqueeze(1)      # [N, 1]

            neg_wins = (
                (neg_success & ~pos_success_exp) |
                ((neg_success == pos_success_exp) & (neg_values > pos_value_exp))
            )  # [N, K-1]

            # Random tie-breaking
            ties = (neg_success == pos_success_exp) & (neg_values == pos_value_exp)
            tie_wins = ties & (torch.rand(N, K-1, device=device) > 0.5)

            # Rank = 1 + number of negatives that beat positive
            ranks = 1 + neg_wins.sum(dim=1) + tie_wins.sum(dim=1)  # [N]
            all_ranks[mode].append(ranks)

        # Aggregate ranks across modes
        ranks_combined = []
        for mode in corruption_modes:
            if all_ranks[mode]:
                ranks_combined.append(torch.cat(all_ranks[mode]))

        if ranks_combined:
            ranks_tensor = torch.cat(ranks_combined).float().cpu()
            mrr = (1.0 / ranks_tensor).mean().item()
            hits_1 = (ranks_tensor <= 1.0).float().mean().item()
            hits_3 = (ranks_tensor <= 3.0).float().mean().item()
            hits_10 = (ranks_tensor <= 10.0).float().mean().item()
        else:
            mrr, hits_1, hits_3, hits_10 = 0.0, 0.0, 0.0, 0.0

        proven_pos = sum(pos_successes) / len(pos_successes) if pos_successes else 0.0
        proven_neg = sum(neg_successes) / len(neg_successes) if neg_successes else 0.0
        mean_depth_pos = sum(pos_depths) / len(pos_depths) if pos_depths else 0.0
        mean_depth_neg = sum(neg_depths) / len(neg_depths) if neg_depths else 0.0

        result = {
            "MRR": mrr,
            "Hits@1": hits_1,
            "Hits@3": hits_3,
            "Hits@10": hits_10,
            "proven_pos": proven_pos,
            "proven_neg": proven_neg,
            "len_pos": mean_depth_pos,
            "len_neg": mean_depth_neg,
        }

        # Per-mode metrics
        for mode_idx, mode in enumerate(corruption_modes):
            if all_ranks[mode]:
                mode_ranks = torch.cat(all_ranks[mode]).float().cpu()
                result[f"MRR_{mode}"] = (1.0 / mode_ranks).mean().item()
                result[f"Hits@1_{mode}"] = (mode_ranks <= 1.0).float().mean().item()

        if verbose:
            print(f"[MCTS] Evaluation complete: MRR={mrr:.4f}, Hits@1={hits_1:.4f}, "
                  f"proven_pos={proven_pos:.4f}, proven_neg={proven_neg:.4f}")

        return result

    def save(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_timesteps": self.num_timesteps,
            "num_episodes": self.num_episodes,
            "iteration": self.iteration,
            "config": self.config,
        }, path)

    def load(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to load checkpoint from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.num_timesteps = checkpoint.get("num_timesteps", 0)
        self.num_episodes = checkpoint.get("num_episodes", 0)
        self.iteration = checkpoint.get("iteration", 0)
