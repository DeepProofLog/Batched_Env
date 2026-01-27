"""
MCTS Tree Data Structures and Search Algorithm.

Implements MuZero-style Monte Carlo Tree Search with:
- Node: Tree node storing visit statistics and children (original implementation)
- MCTSTreeTensors: Tensor-based tree for batched CUDA graph compatible operations
- MinMaxStats: Q-value normalization tracker
- MCTS: Main search algorithm with PUCT selection
- MCTSBatched: Batched search algorithm for CUDA graph compatibility

Key differences from AlphaZero:
- Uses real environment dynamics instead of learned dynamics model
- Supports action masking for invalid actions
- Designed for variable action spaces (knowledge graph reasoning)

Optimization notes:
- MCTSTreeTensors replaces Python dicts with tensors [B, D, A] for CUDA graphs
- Vectorized PUCT selection processes all B environments simultaneously
- Compile single simulation step, not entire search loop (data-dependent branching)

Reference: Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by
Planning with a Learned Model" (MuZero, 2020)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import MCTSConfig


# =============================================================================
# TENSOR-BASED TREE STRUCTURE (CUDA Graph Compatible)
# =============================================================================

@dataclass
class MCTSTreeTensors:
    """Tensor-based MCTS tree for B parallel environments.

    All tensors have shape [B, max_depth, max_actions] or [B, max_depth].
    This structure replaces Python dicts/Node objects for CUDA graph compatibility.

    Tensor shape conventions:
    - B: batch size (number of parallel environments)
    - D: max_depth (maximum tree depth)
    - A: max_actions (padding_states from env, typically 120)

    Attributes:
        visit_counts: [B, D, A] int32 - Visit count N(s,a) per (env, depth, action).
        value_sums: [B, D, A] float32 - Sum of backed-up values W(s,a).
        priors: [B, D, A] float32 - Prior probabilities P(a|s) from policy network.
        expanded_mask: [B, D, A] bool - Which actions are expanded at each depth.
        parent_visits: [B, D] int32 - Total visits to parent at each depth (for PUCT).
        rewards: [B, D] float32 - Rewards received at each depth.
        current_depth: [B] int32 - Current tree depth per environment.
        min_values: [B] float32 - MinMax stats: minimum Q-value observed per env.
        max_values: [B] float32 - MinMax stats: maximum Q-value observed per env.
        device: torch.device - Device for all tensors.
    """
    visit_counts: Tensor      # [B, D, A] int32
    value_sums: Tensor        # [B, D, A] float32
    priors: Tensor            # [B, D, A] float32
    expanded_mask: Tensor     # [B, D, A] bool
    parent_visits: Tensor     # [B, D] int32
    rewards: Tensor           # [B, D] float32
    current_depth: Tensor     # [B] int32
    min_values: Tensor        # [B] float32
    max_values: Tensor        # [B] float32
    device: torch.device = field(default_factory=lambda: torch.device('cuda'))

    @classmethod
    def allocate(
        cls,
        batch_size: int,
        max_depth: int,
        max_actions: int,
        device: torch.device,
    ) -> "MCTSTreeTensors":
        """Allocate pre-sized tree tensors for CUDA graph stability.

        Args:
            batch_size: Number of parallel environments (B).
            max_depth: Maximum tree depth (D), typically max_episode_steps + 1.
            max_actions: Maximum actions per state (A), typically padding_states.
            device: Device for tensor allocation.

        Returns:
            MCTSTreeTensors with pre-allocated zeroed tensors.
        """
        B, D, A = batch_size, max_depth, max_actions
        return cls(
            visit_counts=torch.zeros((B, D, A), dtype=torch.int32, device=device),
            value_sums=torch.zeros((B, D, A), dtype=torch.float32, device=device),
            priors=torch.zeros((B, D, A), dtype=torch.float32, device=device),
            expanded_mask=torch.zeros((B, D, A), dtype=torch.bool, device=device),
            parent_visits=torch.zeros((B, D), dtype=torch.int32, device=device),
            rewards=torch.zeros((B, D), dtype=torch.float32, device=device),
            current_depth=torch.zeros(B, dtype=torch.int32, device=device),
            min_values=torch.full((B,), float('inf'), dtype=torch.float32, device=device),
            max_values=torch.full((B,), float('-inf'), dtype=torch.float32, device=device),
            device=device,
        )

    def reset(self) -> None:
        """Reset all tree tensors for new search (in-place, no allocation)."""
        self.visit_counts.zero_()
        self.value_sums.zero_()
        self.priors.zero_()
        self.expanded_mask.zero_()
        self.parent_visits.zero_()
        self.rewards.zero_()
        self.current_depth.zero_()
        self.min_values.fill_(float('inf'))
        self.max_values.fill_(float('-inf'))

    def mark_static_addresses(self) -> None:
        """Mark all tensors as static addresses for CUDA graph stability."""
        if hasattr(torch, "_dynamo"):
            torch._dynamo.mark_static_address(self.visit_counts)
            torch._dynamo.mark_static_address(self.value_sums)
            torch._dynamo.mark_static_address(self.priors)
            torch._dynamo.mark_static_address(self.expanded_mask)
            torch._dynamo.mark_static_address(self.parent_visits)
            torch._dynamo.mark_static_address(self.rewards)
            torch._dynamo.mark_static_address(self.current_depth)
            torch._dynamo.mark_static_address(self.min_values)
            torch._dynamo.mark_static_address(self.max_values)


class MinMaxStatsBatched:
    """Batched min/max Q-value tracking for B environments.

    Vectorized version of MinMaxStats for CUDA graph compatibility.
    Tracks separate min/max bounds per environment for Q-value normalization.

    Attributes:
        min_values: [B] float32 - Minimum Q-value observed per environment.
        max_values: [B] float32 - Maximum Q-value observed per environment.
    """

    def __init__(self, batch_size: int, device: torch.device):
        """Initialize batched min/max stats.

        Args:
            batch_size: Number of parallel environments.
            device: Device for tensor allocation.
        """
        self.min_values = torch.full((batch_size,), float('inf'), dtype=torch.float32, device=device)
        self.max_values = torch.full((batch_size,), float('-inf'), dtype=torch.float32, device=device)
        self.device = device

    def reset(self) -> None:
        """Reset min/max bounds for new search."""
        self.min_values.fill_(float('inf'))
        self.max_values.fill_(float('-inf'))

    def update(self, values: Tensor) -> None:
        """Update min/max bounds with new values (vectorized).

        Args:
            values: [B] float32 - New Q-values to incorporate.
        """
        torch.minimum(self.min_values, values, out=self.min_values)
        torch.maximum(self.max_values, values, out=self.max_values)

    def normalize(self, values: Tensor) -> Tensor:
        """Normalize values to [0, 1] using observed bounds (vectorized).

        Args:
            values: [B] or [B, A] float32 - Values to normalize.

        Returns:
            Normalized values in [0, 1], or 0 where bounds are invalid.
        """
        range_vals = self.max_values - self.min_values
        valid_range = range_vals > 0

        if values.dim() == 1:
            # [B] case
            normalized = torch.where(
                valid_range,
                (values - self.min_values) / range_vals.clamp(min=1e-8),
                torch.zeros_like(values)
            )
        else:
            # [B, A] case - broadcast along action dimension
            valid_range_expanded = valid_range.unsqueeze(1)
            min_expanded = self.min_values.unsqueeze(1)
            range_expanded = range_vals.unsqueeze(1).clamp(min=1e-8)
            normalized = torch.where(
                valid_range_expanded,
                (values - min_expanded) / range_expanded,
                torch.zeros_like(values)
            )

        return normalized


# =============================================================================
# BATCHED MCTS OPERATIONS (CUDA Graph Compatible)
# =============================================================================

class MCTSBatched:
    """Batched Monte Carlo Tree Search for CUDA graph compatibility.

    Processes B environments in parallel using tensor operations.
    Designed for torch.compile with mode='reduce-overhead'.

    Key design decisions:
    - Outer search loop stays in Python (data-dependent branching)
    - Inner operations (PUCT selection, backprop) are fully vectorized
    - Uses MCTSTreeTensors instead of Python dicts/Node objects
    - Avoids .item() calls in hot paths

    Usage:
        mcts = MCTSBatched(config, batch_size=100, max_actions=120, device=device)
        actions, stats = mcts.search_batched(env, state, policy, obs, action_mask)

    Attributes:
        config: MCTSConfig with search hyperparameters.
        tree: MCTSTreeTensors for storing search statistics.
        batch_size: Number of parallel environments.
        max_actions: Maximum actions per state.
        device: Computation device.
    """

    def __init__(
        self,
        config: MCTSConfig,
        batch_size: int,
        max_actions: int,
        device: torch.device,
    ):
        """Initialize batched MCTS.

        Args:
            config: MCTSConfig with search hyperparameters.
            batch_size: Number of parallel environments (B).
            max_actions: Maximum actions per state (A), typically padding_states.
            device: Device for tensor operations.
        """
        self.config = config
        self.batch_size = batch_size
        self.max_actions = max_actions
        self.device = device

        # Tree depth = max_episode_steps + 1 (for root)
        self.max_depth = config.max_episode_steps + 1

        # Pre-allocate tree tensors
        self.tree = MCTSTreeTensors.allocate(
            batch_size=batch_size,
            max_depth=self.max_depth,
            max_actions=max_actions,
            device=device,
        )
        self.tree.mark_static_addresses()

        # Pre-allocate helper tensors
        self._arange_B = torch.arange(batch_size, device=device)
        self._arange_A = torch.arange(max_actions, device=device)
        self._zeros_B = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self._ones_B = torch.ones(batch_size, dtype=torch.float32, device=device)

        # PUCT constants (computed from config)
        self._pb_c_base = config.pb_c_base
        self._pb_c_init = config.pb_c_init

        # Mark static addresses
        if hasattr(torch, "_dynamo"):
            torch._dynamo.mark_static_address(self._arange_B)
            torch._dynamo.mark_static_address(self._arange_A)
            torch._dynamo.mark_static_address(self._zeros_B)
            torch._dynamo.mark_static_address(self._ones_B)

    def _get_puct_constant_batched(self, parent_visits: Tensor) -> Tensor:
        """Compute PUCT exploration constant for all environments.

        c(s) = log((1 + N(s) + pb_c_base) / pb_c_base) + pb_c_init

        Args:
            parent_visits: [B] int32 - Parent visit counts per environment.

        Returns:
            [B] float32 - PUCT constants per environment.
        """
        # Vectorized PUCT constant computation
        return (
            torch.log((1 + parent_visits.float() + self._pb_c_base) / self._pb_c_base)
            + self._pb_c_init
        )

    def _select_child_batched(
        self,
        depth: Tensor,
        action_mask: Tensor,
    ) -> Tensor:
        """Select best action for all B environments using PUCT (vectorized).

        UCB(s,a) = Q_norm(s,a) + c(s) * P(a|s) * sqrt(N(s)) / (1 + N(s,a))

        Args:
            depth: [B] int64 - Current depth per environment.
            action_mask: [B, A] bool - Valid actions per environment.

        Returns:
            [B] int64 - Selected action per environment.
        """
        B = self.batch_size
        tree = self.tree

        # Gather values at current depth: [B, A]
        # Use advanced indexing: tree.tensor[batch_idx, depth, :]
        batch_idx = self._arange_B

        # Gather per-depth slices: [B, A]
        visit_counts = tree.visit_counts[batch_idx, depth]  # [B, A]
        value_sums = tree.value_sums[batch_idx, depth]      # [B, A]
        priors = tree.priors[batch_idx, depth]              # [B, A]
        parent_visits = tree.parent_visits[batch_idx, depth]  # [B]

        # Q-values: [B, A] - mean value where visited, 0 otherwise
        visit_counts_float = visit_counts.float()
        q_values = torch.where(
            visit_counts > 0,
            value_sums / visit_counts_float.clamp(min=1),
            self._zeros_B.unsqueeze(1).expand(B, self.max_actions)
        )

        # Normalize Q-values to [0, 1] using MinMaxStats
        q_norm = self._normalize_q_batched(q_values)

        # PUCT exploration bonus: [B, A]
        # c * P(a|s) * sqrt(N(s)) / (1 + N(s,a))
        c = self._get_puct_constant_batched(parent_visits)  # [B]
        sqrt_parent = torch.sqrt(parent_visits.float()).clamp(min=1.0)  # [B]
        exploration = (
            c.unsqueeze(1) * priors * sqrt_parent.unsqueeze(1) / (1.0 + visit_counts_float)
        )  # [B, A]

        # UCB scores: [B, A]
        ucb = q_norm + exploration

        # Mask invalid actions with large negative value
        ucb = ucb.masked_fill(~action_mask, -1e9)

        # Select argmax per environment
        return ucb.argmax(dim=1)  # [B]

    def _normalize_q_batched(self, q_values: Tensor) -> Tensor:
        """Normalize Q-values to [0, 1] using tree's MinMax stats.

        Args:
            q_values: [B, A] float32 - Raw Q-values.

        Returns:
            [B, A] float32 - Normalized Q-values in [0, 1].
        """
        tree = self.tree

        # Compute range per environment
        range_vals = tree.max_values - tree.min_values  # [B]
        valid_range = range_vals > 0  # [B]

        # Broadcast for [B, A] computation
        valid_range_expanded = valid_range.unsqueeze(1)
        min_expanded = tree.min_values.unsqueeze(1)
        range_expanded = range_vals.unsqueeze(1).clamp(min=1e-8)

        normalized = torch.where(
            valid_range_expanded,
            (q_values - min_expanded) / range_expanded,
            torch.zeros_like(q_values)
        )

        return normalized

    def _backpropagate_batched(
        self,
        path_depths: Tensor,
        path_actions: Tensor,
        path_length: Tensor,
        leaf_values: Tensor,
    ) -> None:
        """Backpropagate values through search paths for all B environments.

        Updates visit counts and value sums along paths from leaf to root.
        Also updates MinMax stats with observed Q-values.

        Optimized version using scatter_add_ for efficient updates.

        Args:
            path_depths: [B, max_depth] int64 - Depths visited in each path.
            path_actions: [B, max_depth] int64 - Actions taken at each depth.
            path_length: [B] int64 - Length of each path (number of nodes).
            leaf_values: [B] float32 - Value estimates at leaf nodes.
        """
        B = self.batch_size
        D = self.max_depth
        A = self.max_actions
        tree = self.tree
        discount = self.config.discount
        device = self.device

        # Use the actual path tensor dimension as max_len
        # This avoids GPU-CPU sync from .item() while respecting tensor bounds
        max_len = min(path_depths.shape[1], path_actions.shape[1])

        # Process backprop in reverse order (leaf to root)
        values = leaf_values.clone()  # [B]

        # Pre-compute step indices [max_len]
        step_range = torch.arange(max_len, device=device)

        for step in range(max_len - 1, -1, -1):
            # Check which environments have this step in their path
            # Note: We don't use active.any() to avoid GPU-CPU sync
            # The scatter_add with zero updates is a no-op
            active = step < path_length  # [B]

            # Get depth and action for this step
            d = path_depths[:, step]  # [B]
            a = path_actions[:, step]  # [B]

            # Compute flat indices for 3D tensor: idx = b * (D * A) + d * A + a
            flat_idx = self._arange_B * (D * A) + d * A + a  # [B]

            # Update visit counts using scatter_add_
            visit_update = active.int().to(tree.visit_counts.dtype)
            tree.visit_counts.view(-1).scatter_add_(0, flat_idx, visit_update)

            # Update value sums using scatter_add_
            value_update = torch.where(active, values, self._zeros_B)
            tree.value_sums.view(-1).scatter_add_(0, flat_idx, value_update)

            # Update parent visits (2D tensor: idx = b * D + d)
            parent_flat_idx = self._arange_B * D + d
            parent_update = active.int().to(tree.parent_visits.dtype)
            tree.parent_visits.view(-1).scatter_add_(0, parent_flat_idx, parent_update)

            # Update MinMax stats with Q-value at this node
            # Q = value_sums / visit_counts (after update)
            new_visit = tree.visit_counts.view(-1)[flat_idx].float()
            new_sum = tree.value_sums.view(-1)[flat_idx]
            q_value = torch.where(
                new_visit > 0,
                new_sum / new_visit.clamp(min=1),
                self._zeros_B
            )

            # Update min/max where active
            active_q = torch.where(active, q_value, tree.min_values)
            torch.minimum(tree.min_values, active_q, out=tree.min_values)

            active_q_max = torch.where(active, q_value, tree.max_values)
            torch.maximum(tree.max_values, active_q_max, out=tree.max_values)

            # Discount value for parent level
            reward = tree.rewards.view(-1)[parent_flat_idx]  # [B]
            values = torch.where(active, reward + discount * values, values)

    def _add_dirichlet_noise_batched(self, action_mask: Tensor) -> None:
        """Add Dirichlet noise to root priors for exploration.

        Args:
            action_mask: [B, A] bool - Valid actions at root.
        """
        tree = self.tree
        alpha = self.config.root_dirichlet_alpha
        frac = self.config.root_exploration_fraction

        # Get root priors
        root_priors = tree.priors[:, 0, :]  # [B, A]

        # Generate Dirichlet noise for all actions, mask handles invalid ones
        # Removed .item() call to avoid GPU-CPU sync

        # Generate noise using Dirichlet distribution
        # Note: We need different-sized noise vectors per env, so we generate max_valid
        # and mask out invalid actions
        noise_alpha = torch.full((self.batch_size, self.max_actions), alpha, device=self.device)

        # Sample from Dirichlet (approximation: sample gamma and normalize)
        gamma_samples = torch.distributions.Gamma(noise_alpha, torch.ones_like(noise_alpha)).sample()
        gamma_samples = gamma_samples.masked_fill(~action_mask, 0)
        noise_sum = gamma_samples.sum(dim=1, keepdim=True).clamp(min=1e-8)
        noise = gamma_samples / noise_sum  # [B, A]

        # Mix original priors with noise
        new_priors = (1 - frac) * root_priors + frac * noise

        # Re-normalize to sum to 1 over valid actions
        new_priors = new_priors.masked_fill(~action_mask, 0)
        prior_sum = new_priors.sum(dim=1, keepdim=True).clamp(min=1e-8)
        tree.priors[:, 0, :] = new_priors / prior_sum

    def _select_action_batched(self, temperature: float) -> Tensor:
        """Select final action based on root visit counts.

        Args:
            temperature: Temperature for visit count -> probability conversion.

        Returns:
            [B] int64 - Selected action per environment.
        """
        tree = self.tree
        root_visits = tree.visit_counts[:, 0, :].float()  # [B, A]

        if temperature < 1e-8:
            # Greedy: select most visited action
            return root_visits.argmax(dim=1)

        # Sample proportionally to visit counts^(1/temperature)
        visit_probs = root_visits.pow(1.0 / temperature)
        visit_probs = visit_probs / visit_probs.sum(dim=1, keepdim=True).clamp(min=1e-8)

        # Clamp probabilities for numerical stability
        visit_probs = visit_probs.clamp(min=0, max=1)

        return torch.multinomial(visit_probs, 1).squeeze(1)


# =============================================================================
# ORIGINAL NODE-BASED IMPLEMENTATION (Reference)
# =============================================================================


@dataclass
class Node:
    """MCTS tree node storing search statistics.

    Each node represents a state in the search tree. Nodes track:
    - Visit count for UCB calculation
    - Value sum for mean value estimation
    - Prior probability from policy network
    - Children for expanded actions

    Attributes:
        prior: Prior probability P(a|s) from policy network.
        visit_count: Number of times this node was visited N(s,a).
        value_sum: Sum of backed-up values W(s,a).
        reward: Immediate reward r(s,a) from taking action to reach this node.
        children: Dict mapping action indices to child nodes.
        hidden_state: Optional state representation for dynamics model.
        to_play: Player to move (always 0 for single-player KG reasoning).
    """

    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    reward: float = 0.0
    children: Dict[int, "Node"] = field(default_factory=dict)
    hidden_state: Optional[Tensor] = None
    to_play: int = 0

    def is_expanded(self) -> bool:
        """Check if node has been expanded (has children)."""
        return len(self.children) > 0

    def value(self) -> float:
        """Mean value Q(s,a) = W(s,a) / N(s,a)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(
        self,
        actions: List[int],
        priors: Tensor,
        reward: float = 0.0,
        hidden_state: Optional[Tensor] = None,
    ) -> None:
        """Expand node with child nodes for each valid action.

        Args:
            actions: List of valid action indices.
            priors: Tensor of prior probabilities for each action.
            reward: Reward received transitioning to this node.
            hidden_state: State representation (optional).
        """
        self.reward = reward
        self.hidden_state = hidden_state

        for action in actions:
            # Get prior for this action (handle both 1D and indexed tensors)
            if priors.dim() == 1:
                prior_val = priors[action].item()
            else:
                prior_val = priors[action].item()

            self.children[action] = Node(prior=prior_val)

    def add_exploration_noise(
        self,
        dirichlet_alpha: float,
        exploration_fraction: float,
    ) -> None:
        """Add Dirichlet noise to prior probabilities at root.

        This encourages exploration of actions not favored by the policy.

        Args:
            dirichlet_alpha: Alpha parameter for Dirichlet distribution.
            exploration_fraction: Fraction of prior to replace with noise.
        """
        if not self.children:
            return

        actions = list(self.children.keys())
        noise = torch.distributions.Dirichlet(
            torch.full((len(actions),), dirichlet_alpha)
        ).sample()

        for i, action in enumerate(actions):
            child = self.children[action]
            child.prior = (
                child.prior * (1 - exploration_fraction)
                + noise[i].item() * exploration_fraction
            )


class MinMaxStats:
    """Tracks min/max Q-values for normalization.

    MuZero normalizes Q-values to [0, 1] using the min and max values
    observed in the current tree. This helps PUCT work consistently
    regardless of the reward scale.

    Attributes:
        min_value: Minimum Q-value observed.
        max_value: Maximum Q-value observed.
    """

    def __init__(self):
        self.min_value: float = float("inf")
        self.max_value: float = float("-inf")

    def update(self, value: float) -> None:
        """Update min/max bounds with new value."""
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)

    def normalize(self, value: float) -> float:
        """Normalize value to [0, 1] using observed bounds.

        Returns 0.0 if bounds are invalid (no values observed yet).
        """
        if self.max_value <= self.min_value:
            return 0.0
        return (value - self.min_value) / (self.max_value - self.min_value)


class MCTS:
    """Monte Carlo Tree Search algorithm.

    Implements MuZero-style MCTS with real environment dynamics.
    The search uses:
    - PUCT for action selection during tree traversal
    - Real env.step() for state transitions (no dynamics model)
    - Policy network for prior probabilities and value estimates

    Usage:
        mcts = MCTS(config)
        action, search_stats = mcts.search(
            env=env,
            env_state=state,
            networks=policy_network,
            action_mask=mask,
        )

    Attributes:
        config: MCTSConfig with search hyperparameters.
    """

    def __init__(self, config: MCTSConfig):
        """Initialize MCTS with configuration.

        Args:
            config: MCTSConfig containing search hyperparameters.
        """
        self.config = config

    def search(
        self,
        env: Any,
        env_state: Any,
        networks: torch.nn.Module,
        obs: Dict[str, Tensor],
        action_mask: Tensor,
        add_noise: bool = True,
    ) -> Tuple[int, Dict[str, Any]]:
        """Run MCTS search from current state.

        Performs num_simulations iterations of:
        1. Selection: Traverse tree using UCB until leaf
        2. Expansion: Expand leaf using policy network
        3. Backpropagation: Update value statistics along path

        Args:
            env: Environment instance with step() method.
            env_state: Current environment state (TensorDict or similar).
            networks: Policy network with forward(obs) -> (logits, value).
            obs: Current observation dict.
            action_mask: Boolean mask of valid actions [num_actions].
            add_noise: Whether to add exploration noise at root.

        Returns:
            Tuple of:
            - Selected action index
            - Search statistics dict with 'visit_counts', 'values', etc.
        """
        device = action_mask.device

        # Get initial policy and value from network
        with torch.no_grad():
            logits = networks.get_logits(obs)  # [1, num_actions] or [num_actions]
            if logits.dim() == 2:
                logits = logits.squeeze(0)
            value = networks.predict_values(obs)  # [1] or scalar
            if isinstance(value, Tensor):
                value = value.item()

        # Mask invalid actions
        masked_logits = logits.masked_fill(~action_mask.bool(), float("-inf"))
        priors = F.softmax(masked_logits, dim=-1)

        # Get valid actions
        valid_actions = action_mask.nonzero(as_tuple=True)[0].tolist()
        if not valid_actions:
            # No valid actions - return dummy action
            return 0, {"visit_counts": {}, "values": {}, "root_value": 0.0}

        # Initialize root node
        root = Node()
        root.expand(valid_actions, priors, reward=0.0, hidden_state=None)

        # Add exploration noise at root
        if add_noise and self.config.add_exploration_noise:
            root.add_exploration_noise(
                self.config.root_dirichlet_alpha,
                self.config.root_exploration_fraction,
            )

        # Initialize Q-value normalization tracker
        min_max_stats = MinMaxStats()

        # Run simulations
        for _ in range(self.config.num_simulations):
            # Create copy of environment state for simulation
            # Note: We assume env_state can be cloned
            sim_state = self._clone_state(env_state)
            sim_obs = {k: v.clone() for k, v in obs.items()}

            # Selection and expansion
            node = root
            search_path = [node]
            current_action_mask = action_mask.clone()

            # Traverse tree until we reach a leaf
            while node.is_expanded():
                action, child = self._select_child(node, min_max_stats)
                if action is None:
                    break
                search_path.append(child)
                node = child

                # Take action in simulated environment
                sim_obs, sim_state = env.step(sim_state, torch.tensor([action], device=device))
                sim_obs = {k: v.clone() for k, v in sim_obs.items()}

                # Get new action mask
                current_action_mask = sim_obs.get("action_mask", sim_state.get("derived_counts", None))
                if current_action_mask is None:
                    break

                # Check if episode is done
                done = sim_state.get("done", sim_state.get("step_dones", None))
                if done is not None and done.bool().any():
                    break

            # Expansion: get policy and value for leaf node
            if not node.is_expanded():
                with torch.no_grad():
                    leaf_logits = networks.get_logits(sim_obs)
                    if leaf_logits.dim() == 2:
                        leaf_logits = leaf_logits.squeeze(0)
                    leaf_value = networks.predict_values(sim_obs)
                    if isinstance(leaf_value, Tensor):
                        leaf_value = leaf_value.item()

                # Check if terminal
                done = sim_state.get("done", torch.zeros(1, dtype=torch.bool))
                if done.bool().any():
                    # Terminal node: use actual reward
                    success = sim_state.get("success", torch.zeros(1, dtype=torch.bool))
                    leaf_value = 1.0 if success.bool().any() else -1.0
                else:
                    # Non-terminal: expand with policy
                    if current_action_mask is not None and current_action_mask.numel() > 0:
                        if current_action_mask.dim() == 2:
                            current_action_mask = current_action_mask.squeeze(0)
                        current_action_mask = current_action_mask.bool()
                        masked_leaf_logits = leaf_logits.masked_fill(
                            ~current_action_mask, float("-inf")
                        )
                        leaf_priors = F.softmax(masked_leaf_logits, dim=-1)
                        leaf_valid_actions = current_action_mask.nonzero(as_tuple=True)[0].tolist()
                        if leaf_valid_actions:
                            # Get reward from state if available
                            reward = sim_state.get("step_rewards", torch.zeros(1))
                            if isinstance(reward, Tensor):
                                reward = reward.item() if reward.numel() == 1 else 0.0
                            node.expand(leaf_valid_actions, leaf_priors, reward=reward)

            # Backpropagation
            self._backpropagate(search_path, leaf_value, min_max_stats)

        # Collect search statistics
        visit_counts = {
            action: child.visit_count for action, child in root.children.items()
        }
        values = {action: child.value() for action, child in root.children.items()}

        # Select action based on visit counts
        action = self._select_action(root, temperature=self.config.temperature_init)

        return action, {
            "visit_counts": visit_counts,
            "values": values,
            "root_value": root.value(),
            "root_visit_count": root.visit_count,
        }

    def _select_child(
        self,
        node: Node,
        min_max_stats: MinMaxStats,
    ) -> Tuple[Optional[int], Optional[Node]]:
        """Select child node using PUCT formula.

        UCB(s,a) = Q(s,a) + c * P(a|s) * sqrt(N(s)) / (1 + N(s,a))

        where Q is normalized to [0, 1] using MinMaxStats.

        Args:
            node: Parent node to select from.
            min_max_stats: Q-value normalization tracker.

        Returns:
            Tuple of (selected action, selected child node).
        """
        if not node.children:
            return None, None

        # Compute UCB scores
        pb_c = self.config.get_puct_constant(node.visit_count)
        sqrt_visit = math.sqrt(node.visit_count) if node.visit_count > 0 else 1.0

        best_action = None
        best_child = None
        best_score = float("-inf")

        for action, child in node.children.items():
            # Normalized Q-value
            q_value = min_max_stats.normalize(child.value()) if child.visit_count > 0 else 0.0

            # Prior score (exploration bonus)
            prior_score = (
                pb_c * child.prior * sqrt_visit / (1 + child.visit_count)
            )

            ucb_score = q_value + prior_score

            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
                best_child = child

        return best_action, best_child

    def _backpropagate(
        self,
        search_path: List[Node],
        value: float,
        min_max_stats: MinMaxStats,
    ) -> None:
        """Backpropagate value through search path.

        Updates visit counts and value sums along the path from leaf to root.
        Also updates MinMaxStats with observed values.

        Args:
            search_path: List of nodes from root to leaf.
            value: Value estimate at leaf node.
            min_max_stats: Q-value normalization tracker.
        """
        discount = self.config.discount

        # Backpropagate from leaf to root
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            min_max_stats.update(node.value())

            # Discount value for parent
            value = node.reward + discount * value

    def _select_action(
        self,
        root: Node,
        temperature: float,
    ) -> int:
        """Select action from root based on visit counts.

        When temperature > 0, samples action proportionally to visit counts.
        When temperature ~= 0, selects most visited action (greedy).

        Args:
            root: Root node after search.
            temperature: Temperature for softmax over visit counts.

        Returns:
            Selected action index.
        """
        if not root.children:
            return 0

        actions = list(root.children.keys())
        visit_counts = torch.tensor(
            [root.children[a].visit_count for a in actions],
            dtype=torch.float32,
        )

        if temperature < 1e-8:
            # Greedy selection
            return actions[visit_counts.argmax().item()]

        # Sample from visit count distribution
        visit_probs = visit_counts ** (1.0 / temperature)
        visit_probs = visit_probs / visit_probs.sum()
        action_idx = torch.multinomial(visit_probs, 1).item()

        return actions[action_idx]

    def _clone_state(self, state: Any) -> Any:
        """Clone environment state for simulation.

        Args:
            state: Environment state (TensorDict or similar).

        Returns:
            Deep copy of state.
        """
        if hasattr(state, "clone"):
            return state.clone()
        elif isinstance(state, dict):
            return {k: v.clone() if isinstance(v, Tensor) else v for k, v in state.items()}
        else:
            return state

    def get_action_probs(
        self,
        root: Node,
        temperature: float,
    ) -> Dict[int, float]:
        """Get action probability distribution from visit counts.

        Args:
            root: Root node after search.
            temperature: Temperature for softmax.

        Returns:
            Dict mapping action to probability.
        """
        if not root.children:
            return {}

        actions = list(root.children.keys())
        visit_counts = torch.tensor(
            [root.children[a].visit_count for a in actions],
            dtype=torch.float32,
        )

        if temperature < 1e-8:
            # One-hot on most visited
            probs = torch.zeros_like(visit_counts)
            probs[visit_counts.argmax()] = 1.0
        else:
            visit_probs = visit_counts ** (1.0 / temperature)
            probs = visit_probs / visit_probs.sum()

        return {a: probs[i].item() for i, a in enumerate(actions)}
