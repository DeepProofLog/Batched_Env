"""
MCTS Tree Data Structures and Search Algorithm.

Implements MuZero-style Monte Carlo Tree Search with:
- Node: Tree node storing visit statistics and children
- MinMaxStats: Q-value normalization tracker
- MCTS: Main search algorithm with PUCT selection

Key differences from AlphaZero:
- Uses real environment dynamics instead of learned dynamics model
- Supports action masking for invalid actions
- Designed for variable action spaces (knowledge graph reasoning)

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
