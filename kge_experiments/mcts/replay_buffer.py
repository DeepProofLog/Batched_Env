"""
MCTS Replay Buffer.

Stores trajectories with MCTS search statistics for off-policy training.
Unlike PPO's on-policy buffer, this maintains a history of experiences
that can be sampled multiple times for training.

Key features:
- Stores observations, actions, rewards, MCTS visit counts, and values
- Computes n-step returns for value targets
- Supports trajectory-based and position-based sampling
- Efficient GPU storage for fast sampling

Reference: MuZero paper (Schrittwieser et al., 2020)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import random

import torch
from torch import Tensor


@dataclass
class MCTSTransition:
    """Single transition with MCTS statistics.

    Attributes:
        obs: Observation dict at this step.
        action: Action taken.
        reward: Reward received.
        done: Whether episode ended.
        visit_counts: MCTS visit counts for each action.
        root_value: MCTS root value estimate.
        policy_target: Normalized visit counts (training target).
        value_target: N-step return (computed after trajectory completion).
    """
    obs: Dict[str, Tensor]
    action: int
    reward: float
    done: bool
    visit_counts: Dict[int, int]
    root_value: float
    policy_target: Optional[Tensor] = None
    value_target: Optional[float] = None


@dataclass
class MCTSTrajectory:
    """Complete episode trajectory with MCTS statistics.

    Stores all transitions from one episode along with metadata
    needed for training target computation.

    Attributes:
        transitions: List of transitions in the episode.
        total_reward: Cumulative episode reward.
        length: Number of steps in episode.
    """
    transitions: List[MCTSTransition] = field(default_factory=list)
    total_reward: float = 0.0
    length: int = 0

    def add(self, transition: MCTSTransition) -> None:
        """Add transition to trajectory."""
        self.transitions.append(transition)
        self.total_reward += transition.reward
        self.length += 1

    def compute_targets(
        self,
        discount: float = 0.99,
        n_step: int = 5,
    ) -> None:
        """Compute policy and value targets for all transitions.

        Policy targets: Normalized visit counts from MCTS search.
        Value targets: N-step bootstrapped returns.

        Args:
            discount: Discount factor for returns.
            n_step: Number of steps for n-step returns.
        """
        n = len(self.transitions)

        for i, trans in enumerate(self.transitions):
            # Policy target: normalize visit counts to probability distribution
            if trans.visit_counts:
                total_visits = sum(trans.visit_counts.values())
                if total_visits > 0:
                    # Create dense tensor of visit probs
                    max_action = max(trans.visit_counts.keys()) + 1
                    policy_target = torch.zeros(max_action, dtype=torch.float32)
                    for action, count in trans.visit_counts.items():
                        policy_target[action] = count / total_visits
                    trans.policy_target = policy_target

            # Value target: n-step return
            value_target = 0.0
            for j in range(i, min(i + n_step, n)):
                value_target += (discount ** (j - i)) * self.transitions[j].reward

            # Bootstrap with root value if not at end of trajectory
            bootstrap_idx = i + n_step
            if bootstrap_idx < n:
                value_target += (discount ** n_step) * self.transitions[bootstrap_idx].root_value
            elif not self.transitions[n - 1].done:
                # Episode truncated, use last root value for bootstrap
                value_target += (discount ** (n - 1 - i)) * self.transitions[n - 1].root_value

            trans.value_target = value_target


class MCTSReplayBuffer:
    """Replay buffer for MCTS trajectories.

    Stores complete trajectories with MCTS statistics for off-policy training.
    Supports both trajectory-level and position-level sampling.

    Attributes:
        max_size: Maximum number of trajectories to store.
        trajectories: List of stored trajectories.
        discount: Discount factor for return computation.
        n_step: N-step return horizon.
    """

    def __init__(
        self,
        max_size: int = 100000,
        discount: float = 0.99,
        n_step: int = 5,
        device: torch.device = None,
    ):
        """Initialize replay buffer.

        Args:
            max_size: Maximum number of trajectories.
            discount: Discount factor for returns.
            n_step: N-step return horizon.
            device: Device for tensor storage.
        """
        self.max_size = max_size
        self.discount = discount
        self.n_step = n_step
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.trajectories: List[MCTSTrajectory] = []
        self._total_transitions = 0

    def __len__(self) -> int:
        """Number of trajectories in buffer."""
        return len(self.trajectories)

    @property
    def total_transitions(self) -> int:
        """Total number of transitions across all trajectories."""
        return self._total_transitions

    def add_trajectory(self, trajectory: MCTSTrajectory) -> None:
        """Add completed trajectory to buffer.

        Computes targets and stores trajectory. Removes oldest trajectories
        if buffer exceeds max_size.

        Args:
            trajectory: Completed trajectory with MCTS statistics.
        """
        # Compute training targets
        trajectory.compute_targets(discount=self.discount, n_step=self.n_step)

        # Add to buffer
        self.trajectories.append(trajectory)
        self._total_transitions += trajectory.length

        # Remove oldest if over capacity
        while len(self.trajectories) > self.max_size:
            removed = self.trajectories.pop(0)
            self._total_transitions -= removed.length

    def add(
        self,
        observations: List[Dict[str, Tensor]],
        actions: List[int],
        rewards: List[float],
        dones: List[bool],
        visit_counts: List[Dict[int, int]],
        root_values: List[float],
    ) -> None:
        """Add trajectory from lists (convenience method).

        Args:
            observations: List of observation dicts.
            actions: List of actions taken.
            rewards: List of rewards received.
            dones: List of done flags.
            visit_counts: List of MCTS visit count dicts.
            root_values: List of MCTS root values.
        """
        trajectory = MCTSTrajectory()

        for obs, action, reward, done, vc, rv in zip(
            observations, actions, rewards, dones, visit_counts, root_values
        ):
            trans = MCTSTransition(
                obs=obs,
                action=action,
                reward=reward,
                done=done,
                visit_counts=vc,
                root_value=rv,
            )
            trajectory.add(trans)

        self.add_trajectory(trajectory)

    def sample_batch(
        self,
        batch_size: int,
        sequence_length: int = 1,
    ) -> Dict[str, Any]:
        """Sample batch of transitions for training.

        Args:
            batch_size: Number of sequences to sample.
            sequence_length: Length of each sequence (for unrolled training).

        Returns:
            Dict with batched observations, actions, policy targets, value targets.
        """
        if self.total_transitions < batch_size:
            raise ValueError(
                f"Not enough transitions ({self.total_transitions}) for batch_size={batch_size}"
            )

        # Sample positions
        samples = []
        for _ in range(batch_size):
            # Uniform sampling over all positions
            traj_idx = random.randint(0, len(self.trajectories) - 1)
            traj = self.trajectories[traj_idx]
            max_start = max(0, traj.length - sequence_length)
            pos = random.randint(0, max_start)
            samples.append((traj, pos))

        # Batch the samples
        batch_obs = []
        batch_actions = []
        batch_policy_targets = []
        batch_value_targets = []
        batch_rewards = []

        # Determine max action dimension from samples
        max_action_dim = 0
        for traj, pos in samples:
            for i in range(sequence_length):
                if pos + i < traj.length:
                    trans = traj.transitions[pos + i]
                    if trans.policy_target is not None:
                        max_action_dim = max(max_action_dim, trans.policy_target.shape[0])

        for traj, pos in samples:
            seq_obs = []
            seq_actions = []
            seq_policy = []
            seq_values = []
            seq_rewards = []

            for i in range(sequence_length):
                if pos + i < traj.length:
                    trans = traj.transitions[pos + i]

                    # Move observation tensors to device
                    obs = {
                        k: v.to(self.device) if isinstance(v, Tensor) else v
                        for k, v in trans.obs.items()
                    }
                    seq_obs.append(obs)
                    seq_actions.append(trans.action)
                    seq_rewards.append(trans.reward)

                    # Pad policy target to max_action_dim
                    if trans.policy_target is not None:
                        pt = torch.zeros(max_action_dim, dtype=torch.float32, device=self.device)
                        pt[: trans.policy_target.shape[0]] = trans.policy_target.to(self.device)
                        seq_policy.append(pt)
                    else:
                        seq_policy.append(torch.zeros(max_action_dim, dtype=torch.float32, device=self.device))

                    seq_values.append(trans.value_target if trans.value_target is not None else 0.0)
                else:
                    # Pad with last transition if sequence extends beyond trajectory
                    trans = traj.transitions[-1]
                    obs = {
                        k: v.to(self.device) if isinstance(v, Tensor) else v
                        for k, v in trans.obs.items()
                    }
                    seq_obs.append(obs)
                    seq_actions.append(trans.action)
                    seq_rewards.append(0.0)  # No reward for padding
                    seq_policy.append(torch.zeros(max_action_dim, dtype=torch.float32, device=self.device))
                    seq_values.append(0.0)

            batch_obs.append(seq_obs)
            batch_actions.append(seq_actions)
            batch_policy_targets.append(torch.stack(seq_policy))
            batch_value_targets.append(seq_values)
            batch_rewards.append(seq_rewards)

        # Collate observations
        # For sequence_length=1, simplify to single-step batch
        if sequence_length == 1:
            collated_obs = self._collate_obs([s[0] for s in batch_obs])
            return {
                "obs": collated_obs,
                "actions": torch.tensor([a[0] for a in batch_actions], dtype=torch.long, device=self.device),
                "policy_targets": torch.stack([pt[0] for pt in batch_policy_targets]),
                "value_targets": torch.tensor([v[0] for v in batch_value_targets], dtype=torch.float32, device=self.device),
                "rewards": torch.tensor([r[0] for r in batch_rewards], dtype=torch.float32, device=self.device),
            }

        # Multi-step batch
        return {
            "obs_sequences": batch_obs,
            "actions": torch.tensor(batch_actions, dtype=torch.long, device=self.device),
            "policy_targets": torch.stack(batch_policy_targets),
            "value_targets": torch.tensor(batch_value_targets, dtype=torch.float32, device=self.device),
            "rewards": torch.tensor(batch_rewards, dtype=torch.float32, device=self.device),
        }

    def _collate_obs(self, obs_list: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """Collate list of observation dicts into batched dict.

        Args:
            obs_list: List of observation dicts.

        Returns:
            Batched observation dict.
        """
        if not obs_list:
            return {}

        keys = obs_list[0].keys()
        collated = {}

        for key in keys:
            tensors = [obs[key] for obs in obs_list]
            if isinstance(tensors[0], Tensor):
                collated[key] = torch.stack(tensors)
            else:
                collated[key] = tensors

        return collated

    def clear(self) -> None:
        """Clear all trajectories from buffer."""
        self.trajectories.clear()
        self._total_transitions = 0

    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics.

        Returns:
            Dict with buffer statistics.
        """
        if not self.trajectories:
            return {
                "num_trajectories": 0,
                "total_transitions": 0,
                "avg_trajectory_length": 0.0,
                "avg_trajectory_reward": 0.0,
            }

        lengths = [t.length for t in self.trajectories]
        rewards = [t.total_reward for t in self.trajectories]

        return {
            "num_trajectories": len(self.trajectories),
            "total_transitions": self._total_transitions,
            "avg_trajectory_length": sum(lengths) / len(lengths),
            "avg_trajectory_reward": sum(rewards) / len(rewards),
            "min_trajectory_length": min(lengths),
            "max_trajectory_length": max(lengths),
        }


class PrioritizedMCTSReplayBuffer(MCTSReplayBuffer):
    """Prioritized replay buffer for MCTS.

    Samples transitions based on priority (e.g., TD error or loss).
    Higher priority transitions are sampled more frequently.

    Currently implements uniform sampling (TODO: add proper prioritization).
    """

    def __init__(
        self,
        max_size: int = 100000,
        discount: float = 0.99,
        n_step: int = 5,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        device: torch.device = None,
    ):
        """Initialize prioritized replay buffer.

        Args:
            max_size: Maximum number of trajectories.
            discount: Discount factor.
            n_step: N-step return horizon.
            alpha: Priority exponent (0 = uniform, 1 = full priority).
            beta: Importance sampling exponent.
            beta_increment: Beta increase per sample call.
            device: Device for tensor storage.
        """
        super().__init__(max_size, discount, n_step, device)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        # Priority storage (per trajectory for simplicity)
        self.priorities: List[float] = []

    def add_trajectory(self, trajectory: MCTSTrajectory, priority: float = 1.0) -> None:
        """Add trajectory with priority.

        Args:
            trajectory: Trajectory to add.
            priority: Initial priority (default: max priority).
        """
        super().add_trajectory(trajectory)
        self.priorities.append(priority ** self.alpha)

        # Remove oldest priority if over capacity
        while len(self.priorities) > len(self.trajectories):
            self.priorities.pop(0)

    def sample_batch(
        self,
        batch_size: int,
        sequence_length: int = 1,
    ) -> Dict[str, Any]:
        """Sample batch with prioritized sampling.

        Currently falls back to uniform sampling.
        TODO: Implement proper prioritized sampling with importance weights.
        """
        # For now, use uniform sampling
        batch = super().sample_batch(batch_size, sequence_length)

        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return batch

    def update_priorities(
        self,
        indices: List[int],
        priorities: List[float],
    ) -> None:
        """Update priorities for sampled trajectories.

        Args:
            indices: Trajectory indices to update.
            priorities: New priority values.
        """
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority ** self.alpha
