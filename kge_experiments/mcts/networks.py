"""
MCTS Neural Networks.

Policy and Value networks for MuZero-style MCTS. Since we use real environment
dynamics (env.step()) during search, we don't need a dynamics/prediction network.

The MCTSPolicy provides:
- Policy logits for action selection and MCTS priors
- Value estimates for leaf evaluation and bootstrapping

Architecture is similar to ActorCriticPolicy but designed for MCTS:
- Shared embedding and backbone
- Separate policy and value heads
- Support for observation dicts with action masks
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tensordict import TensorDict


class MCTSEmbedder(nn.Module):
    """Embedding layer for MCTS observations.

    Converts index-based observations [B, S, A, 3] into dense embeddings [B, S, E].
    Can wrap an existing embedder or create a simple one.

    Tensor shape conventions:
        B: Batch size
        S: Number of states (action space)
        A: Number of atoms per state
        E: Embedding dimension
    """

    def __init__(
        self,
        n_constants: int,
        n_predicates: int,
        n_vars: int = 1000,
        embedding_dim: int = 256,
        padding_idx: int = 0,
        base_embedder: Optional[nn.Module] = None,
    ):
        """Initialize embedder.

        Args:
            n_constants: Number of constants in knowledge base.
            n_predicates: Number of predicates.
            n_vars: Number of runtime variables.
            embedding_dim: Output embedding dimension.
            padding_idx: Index used for padding.
            base_embedder: Optional existing embedder to wrap.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        if base_embedder is not None:
            self.embedder = base_embedder
            self.embedding_dim = getattr(base_embedder, "embedding_dim", embedding_dim)
        else:
            # Create simple embedding layers
            total_vocab = n_constants + n_predicates + n_vars + 10  # +10 for special tokens
            self.pred_embedding = nn.Embedding(
                n_predicates + 1, embedding_dim, padding_idx=padding_idx
            )
            self.entity_embedding = nn.Embedding(
                n_constants + n_vars + 1, embedding_dim, padding_idx=padding_idx
            )
            self.embedder = None

            # Projection to combine atom components
            self.atom_projection = nn.Linear(3 * embedding_dim, embedding_dim)

    def forward(self, indices: Tensor) -> Tensor:
        """Embed observation indices.

        Args:
            indices: Index tensor with variable shapes, last dim is (predicate, arg1, arg2).
                     Supports: [B, S, A, 3], [B, A, 3], [B, 1, A, 3], [B, S, 1, A, 3], etc.

        Returns:
            Embeddings with atoms aggregated:
            - [B, S, A, 3] -> [B, S, E]
            - [B, A, 3] -> [B, E]
            - [B, 1, A, 3] -> [B, 1, E]
        """
        if self.embedder is not None:
            return self.embedder.get_embeddings_batch(indices)

        # Simple embedding: embed each component and aggregate
        # Last dimension is always 3 (predicate, arg1, arg2)
        original_shape = indices.shape[:-1]
        flat_indices = indices.reshape(-1, 3)  # [N, 3]

        # Embed predicate and arguments
        pred_emb = self.pred_embedding(flat_indices[:, 0])  # [N, E]
        arg1_emb = self.entity_embedding(flat_indices[:, 1])  # [N, E]
        arg2_emb = self.entity_embedding(flat_indices[:, 2])  # [N, E]

        # Combine with projection
        atom_emb = torch.cat([pred_emb, arg1_emb, arg2_emb], dim=-1)  # [N, 3E]
        atom_emb = self.atom_projection(atom_emb)  # [N, E]

        # Reshape back to original shape (without last dim)
        atom_emb = atom_emb.view(*original_shape, -1)  # [..., E]

        # Aggregate over atoms dimension
        # Find the atoms dimension (typically the second-to-last before the embedding)
        # For [B, S, A, E] -> aggregate over A to get [B, S, E]
        # For [B, A, E] -> aggregate over A to get [B, E]
        # For [B, 1, A, E] -> aggregate over A to get [B, 1, E]
        ndim = atom_emb.dim()

        if ndim >= 3:
            # Aggregate over the second-to-last dimension (atoms)
            atom_dim = -2
            # Create padding mask
            mask = (indices[..., 0] != self.padding_idx).unsqueeze(-1).float()  # [..., 1]
            masked_emb = atom_emb * mask
            state_emb = masked_emb.sum(dim=atom_dim) / (mask.sum(dim=atom_dim).clamp(min=1))
        else:
            # [B, E] - just return as is
            state_emb = atom_emb

        return state_emb


class MCTSBackbone(nn.Module):
    """Shared backbone network for policy and value heads.

    Residual MLP that transforms embeddings into hidden features.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout_prob: float = 0.0,
    ):
        """Initialize backbone.

        Args:
            input_dim: Input embedding dimension.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of residual blocks.
            dropout_prob: Dropout probability.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_prob),
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through backbone.

        Args:
            x: Input embeddings [..., input_dim].

        Returns:
            Hidden features [..., hidden_dim].
        """
        x = self.input_transform(x)
        for block in self.res_blocks:
            x = block(x) + x  # Residual connection
        return x


class MCTSPolicy(nn.Module):
    """Policy/Value network for MCTS.

    Provides policy logits and value estimates for MCTS search.
    Compatible with the existing PPO policy interface for easy integration.

    Architecture:
        Embedder: [B, S, A, 3] -> [B, S, E]
        Backbone: [B, S, E] -> [B, S, H]
        Policy Head: [B, S, H] -> [B, S] (dot-product attention)
        Value Head: [B, H] -> [B, 1]

    Attributes:
        embedder: Observation embedder.
        backbone: Shared feature extractor.
        policy_head: Policy head for action logits.
        value_head: Value head for state value.
    """

    def __init__(
        self,
        embedder: nn.Module,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout_prob: float = 0.0,
        device: torch.device = None,
        use_l2_norm: bool = True,
        temperature: float = 0.1,
    ):
        """Initialize MCTS policy.

        Args:
            embedder: Observation embedder module.
            embed_dim: Embedding dimension.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of backbone layers.
            dropout_prob: Dropout probability.
            device: Target device.
            use_l2_norm: Whether to L2 normalize for dot-product attention.
            temperature: Temperature for logit scaling.
        """
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.use_l2_norm = use_l2_norm
        self.temperature = temperature

        # Embedder (can be MCTSEmbedder or wrapped existing embedder)
        self.embedder = embedder

        # Shared backbone
        self.backbone = MCTSBackbone(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_prob=dropout_prob,
        )

        # Policy head: Projects to embed_dim for dot-product with action embeddings
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Value head: Scalar value prediction
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.to(self.device)

    def forward(
        self,
        obs: Union[Dict[str, Tensor], TensorDict],
        deterministic: bool = False,
        actions: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass for training.

        Args:
            obs: Observation dict with 'sub_index', 'derived_sub_indices', 'action_mask'.
            deterministic: Whether to use greedy action selection.
            actions: Optional actions for log prob computation.

        Returns:
            If actions is None: (actions, values, log_probs)
            If actions is given: (values, log_probs, entropy)
        """
        logits, values = self._forward_impl(obs)

        # Mask invalid actions
        action_mask = obs.get("action_mask", None)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), float("-inf"))

        # Compute distribution
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        if actions is not None:
            # Evaluation mode: compute log probs and entropy for given actions
            action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            entropy = -(probs * log_probs).sum(dim=-1)
            entropy = entropy.masked_fill(probs.sum(dim=-1) == 0, 0.0)
            return values, action_log_probs, entropy

        # Action selection mode
        if deterministic:
            selected_actions = logits.argmax(dim=-1)
        else:
            # Handle potential all-inf rows by replacing with uniform
            safe_probs = torch.where(
                probs.sum(dim=-1, keepdim=True) > 0,
                probs,
                torch.ones_like(probs) / probs.shape[-1],
            )
            selected_actions = torch.multinomial(safe_probs, 1).squeeze(-1)

        action_log_probs = log_probs.gather(1, selected_actions.unsqueeze(-1)).squeeze(-1)

        return selected_actions, values, action_log_probs

    def _forward_impl(
        self,
        obs: Union[Dict[str, Tensor], TensorDict],
    ) -> Tuple[Tensor, Tensor]:
        """Internal forward pass computing logits and values.

        Args:
            obs: Observation dict.

        Returns:
            Tuple of (logits [B, S], values [B]).
        """
        # Extract observation components
        sub_index = obs.get("sub_index")  # [B, 1, A, 3]
        derived_sub_indices = obs.get("derived_sub_indices")  # [B, S, A, 3]

        # Embed current state and derived states
        if hasattr(self.embedder, "get_embeddings_batch"):
            obs_emb = self.embedder.get_embeddings_batch(sub_index)  # [B, 1, E]
            action_emb = self.embedder.get_embeddings_batch(derived_sub_indices)  # [B, S, E]
        else:
            obs_emb = self.embedder(sub_index)  # [B, 1, E]
            action_emb = self.embedder(derived_sub_indices)  # [B, S, E]

        # Pass through backbone
        obs_features = self.backbone(obs_emb)  # [B, 1, H]
        action_features = self.backbone(action_emb)  # [B, S, H]

        # Policy head: compute logits via dot-product attention
        obs_policy = self.policy_head(obs_features)  # [B, 1, E]
        action_policy = self.policy_head(action_features)  # [B, S, E]

        if self.use_l2_norm:
            obs_policy = F.normalize(obs_policy, dim=-1)
            action_policy = F.normalize(action_policy, dim=-1)

        logits = torch.matmul(obs_policy, action_policy.transpose(-2, -1)).squeeze(-2)  # [B, S]
        logits = logits / (self.embed_dim ** 0.5)

        if self.temperature is not None and self.temperature > 0:
            logits = logits / self.temperature

        # Value head: from observation features
        values = self.value_head(obs_features).squeeze(-1).squeeze(-1)  # [B]

        return logits, values

    def get_logits(
        self,
        obs: Union[Dict[str, Tensor], TensorDict, Tuple[Tensor, Tensor, Tensor]],
    ) -> Tensor:
        """Get policy logits (compatible with PPO interface).

        Args:
            obs: Observation dict or tuple of (obs_emb, action_emb, mask).

        Returns:
            Logits [B, S].
        """
        if isinstance(obs, tuple):
            # Direct embeddings passed (for compiled eval)
            obs_emb, action_emb, action_mask = obs
            obs_features = self.backbone(obs_emb)
            action_features = self.backbone(action_emb)

            obs_policy = self.policy_head(obs_features)
            action_policy = self.policy_head(action_features)

            if self.use_l2_norm:
                obs_policy = F.normalize(obs_policy, dim=-1)
                action_policy = F.normalize(action_policy, dim=-1)

            logits = torch.matmul(obs_policy, action_policy.transpose(-2, -1)).squeeze(-2)
            logits = logits / (self.embed_dim ** 0.5)

            if self.temperature is not None and self.temperature > 0:
                logits = logits / self.temperature

            # Mask invalid actions
            if action_mask is not None:
                logits = logits.masked_fill(~action_mask.bool(), float("-inf"))

            return logits

        # Dict-based observation
        logits, _ = self._forward_impl(obs)

        action_mask = obs.get("action_mask", None)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), float("-inf"))

        return logits

    def predict_values(
        self,
        obs: Union[Dict[str, Tensor], TensorDict],
    ) -> Tensor:
        """Get value estimates (compatible with PPO interface).

        Args:
            obs: Observation dict.

        Returns:
            Values [B].
        """
        _, values = self._forward_impl(obs)
        return values

    def evaluate_actions(
        self,
        sub_index: Tensor,
        derived_sub_indices: Tensor,
        action_mask: Tensor,
        actions: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Evaluate actions for training (compatible with PPO interface).

        Args:
            sub_index: [B, 1, A, 3]
            derived_sub_indices: [B, S, A, 3]
            action_mask: [B, S]
            actions: [B]

        Returns:
            Tuple of (values [B], log_probs [B], entropy [B]).
        """
        obs = {
            "sub_index": sub_index,
            "derived_sub_indices": derived_sub_indices,
            "action_mask": action_mask,
        }
        return self.forward(obs, actions=actions)


def create_mcts_policy(
    embedder: nn.Module,
    embed_dim: int = 256,
    hidden_dim: int = 256,
    num_layers: int = 4,
    device: torch.device = None,
    **kwargs,
) -> MCTSPolicy:
    """Factory function for MCTSPolicy.

    Args:
        embedder: Observation embedder.
        embed_dim: Embedding dimension.
        hidden_dim: Hidden layer dimension.
        num_layers: Number of backbone layers.
        device: Target device.
        **kwargs: Additional arguments passed to MCTSPolicy.

    Returns:
        MCTSPolicy instance.
    """
    return MCTSPolicy(
        embedder=embedder,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=device,
        **kwargs,
    )
