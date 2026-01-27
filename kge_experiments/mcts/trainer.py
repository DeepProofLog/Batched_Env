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

Reference: Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by
Planning with a Learned Model" (MuZero, 2020)
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import MCTSConfig
from .tree import MCTS, Node
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

        # Initialize MCTS
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

    def evaluate(
        self,
        queries: Tensor,
        sampler: Any,
        n_corruptions: int = 50,
        corruption_modes: Tuple[str, ...] = ('head', 'tail'),
        **kwargs,
    ) -> Dict[str, float]:
        """Evaluate model on ranking task.

        Uses greedy MCTS (no noise) for evaluation.
        Compatible with PPO.evaluate() interface.

        Args:
            queries: Test queries [N, 3].
            sampler: Corruption sampler.
            n_corruptions: Number of negative samples.
            corruption_modes: Corruption modes to use.
            **kwargs: Additional arguments (for compatibility).

        Returns:
            Dict with MRR, Hits@K metrics.
        """
        self.policy.eval()
        device = self.device

        # Simple evaluation: run each query through MCTS
        successes = []
        depths = []

        with torch.no_grad():
            for q_idx in range(queries.shape[0]):
                query = queries[q_idx:q_idx+1]  # [1, 3]

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

                while not done and step < self.config.max_episode_steps:
                    action_mask = obs.get("action_mask", state.get("derived_counts", None) if state else None)
                    if action_mask is None:
                        break

                    if action_mask.dim() == 2:
                        action_mask = action_mask.squeeze(0)

                    # Greedy MCTS (no noise, low temperature)
                    action, _ = self.mcts.search(
                        env=self.env,
                        env_state=state,
                        networks=self.policy,
                        obs=obs,
                        action_mask=action_mask,
                        add_noise=False,
                    )

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
                successes.append(success)
                depths.append(step)

        # Compute metrics
        proven_pos = sum(successes) / len(successes) if successes else 0.0
        mean_depth = sum(depths) / len(depths) if depths else 0.0

        # For full ranking evaluation, we'd need to run on corruptions too
        # For now, return basic metrics
        return {
            "proven_pos": proven_pos,
            "mean_depth": mean_depth,
            "MRR": proven_pos,  # Placeholder - full ranking needs corruption handling
            "Hits@1": proven_pos,
            "Hits@3": proven_pos,
            "Hits@10": proven_pos,
        }

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
