"""
GPU-based Policy and Value Networks for PPO using TensorDict.

This module provides actor-critic networks that mimic the SB3 models
but work entirely on GPU with torch tensors and tensordicts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from tensordict import TensorDict
from stable_baselines3.common.distributions import CategoricalDistribution


class PolicyNetwork(nn.Module):
    """
    Residual MLP that produces policy logits from observation embeddings.
    
    This network processes observation embeddings and action embeddings to
    produce logits for action selection. It uses residual connections for
    better gradient flow.
    
    Args:
        embed_dim: Dimension of input embeddings
        hidden_dim: Dimension of hidden layers
        num_layers: Number of residual blocks
        dropout_prob: Dropout probability
    """
    
    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 8,
        dropout_prob: float = 0.2
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Initial transformation from embedding to hidden representation
        self.obs_transform = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_prob)
        )
        
        # Residual blocks for deep feature extraction
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_prob)
            ) for _ in range(num_layers)
        ])
        
        # Final transformation back to embedding space
        self.out_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def _encode_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply shared residual stack to embeddings."""
        original_shape = embeddings.shape
        flat_embeddings = embeddings.reshape(-1, original_shape[-1])
        
        x = self.obs_transform(flat_embeddings)
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = x + residual
        
        encoded = self.out_transform(x)
        return encoded.view(*original_shape[:-1], -1)
    
    def forward(
        self,
        obs_embeddings: torch.Tensor,
        action_embeddings: torch.Tensor,
        action_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute action logits.
        
        Args:
            obs_embeddings: Observation embeddings (batch, embed_dim) or (batch, 1, embed_dim)
            action_embeddings: Action embeddings (batch, n_actions, embed_dim)
            action_mask: Valid action mask (batch, n_actions)
        
        Returns:
            Masked logits (batch, n_actions)
        """
        # Process embeddings through residual network
        encoded_obs = self._encode_embeddings(obs_embeddings)
        encoded_actions = self._encode_embeddings(action_embeddings)
        
        # If obs has an extra dimension, squeeze it
        if encoded_obs.dim() == 3 and encoded_obs.shape[1] == 1:
            encoded_obs = encoded_obs.squeeze(1)
        
        # Compute similarity (dot product) between observation and action embeddings
        # Use bmm for efficiency when possible
        logits = torch.bmm(
            encoded_obs.unsqueeze(1),
            encoded_actions.transpose(1, 2)
        ).squeeze(1)
        
        # Scale logits by 1/sqrt(embed_dim) like in scaled dot-product attention
        # This prevents the dot products from growing too large and causing
        # the softmax to become too peaked (low entropy)
        # logits = logits / (self.embed_dim ** 0.5)
        
        # Apply action mask
        logits = logits.masked_fill(~action_mask.bool(), float("-inf"))
        
        return logits


class ValueNetwork(nn.Module):
    """
    Residual MLP that maps observation embeddings to scalar value estimates.
    
    Args:
        embed_dim: Dimension of input embeddings
        hidden_dim: Dimension of hidden layers
        num_layers: Number of residual blocks
        dropout_prob: Dropout probability
    """
    
    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 8,
        dropout_prob: float = 0.2
    ):
        super().__init__()
        
        # Initial transformation
        self.input_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_prob)
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_prob)
            ) for _ in range(num_layers)
        ])
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute value estimates.
        
        Args:
            embeddings: Observation embeddings (batch, embed_dim) or (batch, 1, embed_dim)
        
        Returns:
            Value estimates (batch,)
        """
        # Handle shape
        if embeddings.dim() == 3 and embeddings.shape[1] == 1:
            embeddings = embeddings.squeeze(1)
        
        # Process through network
        x = self.input_layer(embeddings)
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = x + residual
        
        value = self.output_layer(x)
        return value.squeeze(-1)


class ActorCriticPolicy(nn.Module):
    """
    Combined actor-critic policy that processes observations and produces
    actions and value estimates.
    
    This class combines the policy and value networks with an embedder
    to create a complete actor-critic model.
    
    Args:
        embedder: Embedder to convert observations to embeddings
        embed_dim: Dimension of embeddings
        hidden_dim: Dimension of hidden layers
        num_layers: Number of residual blocks
        dropout_prob: Dropout probability
        device: PyTorch device
    """
    
    def __init__(
        self,
        embedder,
        embed_dim: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout_prob: float = 0.0,
        device: torch.device = None,
    ):
        super().__init__()
        
        self.embedder = embedder
        self.embed_dim = embed_dim
        self.device = device if device is not None else torch.device('cpu')
        
        # Create policy and value networks
        self.policy_net = PolicyNetwork(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_prob=dropout_prob
        )
        
        self.value_net = ValueNetwork(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_prob=dropout_prob
        )
        
        # Create action distribution (more efficient to create once)
        # Note: action_space.n is not available here, but CategoricalDistribution
        # will determine the number of actions from the logits shape at runtime
        self.action_dist = CategoricalDistribution(action_dim=-1)
        
        # Move to device
        self.to(self.device)
    
    def _extract_features(self, obs: TensorDict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract embeddings and action mask from observations.
        
        Args:
            obs: TensorDict containing 'sub_index', 'derived_sub_indices', 'action_mask'
        
        Returns:
            Tuple of (obs_embeddings, action_embeddings, action_mask)
        """
        # Get tensors from TensorDict
        obs_sub_indices = obs.get("sub_index")  # (batch, 1, pad_atoms, 3)
        action_sub_indices = obs.get("derived_sub_indices")  # (batch, pad_states, pad_atoms, 3)
        action_mask = obs.get("action_mask")  # (batch, pad_states)
        
        # Ensure correct dtype
        if obs_sub_indices.dtype != torch.int32:
            obs_sub_indices = obs_sub_indices.to(torch.int32)
        if action_sub_indices.dtype != torch.int32:
            action_sub_indices = action_sub_indices.to(torch.int32)
        
        # Get embeddings
        obs_embeddings = self.embedder.get_embeddings_batch(obs_sub_indices)  # (batch, 1, embed_dim)
        action_embeddings = self.embedder.get_embeddings_batch(action_sub_indices)  # (batch, pad_states, embed_dim)
        
        return obs_embeddings, action_embeddings, action_mask
    
    def forward(
        self,
        obs: TensorDict,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: compute actions, values, and log probabilities.
        
        Args:
            obs: Observations as TensorDict
            deterministic: If True, select argmax action instead of sampling
        
        Returns:
            Tuple of (actions, values, log_probs)
        """
        # Extract features
        obs_embeddings, action_embeddings, action_mask = self._extract_features(obs)
        
        # Get logits and values
        logits = self.policy_net(obs_embeddings, action_embeddings, action_mask)
        values = self.value_net(obs_embeddings)
        
        # Sample actions using the pre-created distribution
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()
        
        # Compute log probabilities
        action_log_probs = distribution.log_prob(actions)
        
        return actions, values, action_log_probs
    
    def evaluate_actions(
        self,
        obs: TensorDict,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training (compute values, log probs, and entropy).
        
        Args:
            obs: Observations as TensorDict
            actions: Actions to evaluate (batch,)
        
        Returns:
            Tuple of (values, log_probs, entropy)
        """
        # Extract features
        obs_embeddings, action_embeddings, action_mask = self._extract_features(obs)
        
        # Get logits and values
        logits = self.policy_net(obs_embeddings, action_embeddings, action_mask)
        values = self.value_net(obs_embeddings)
        
        # Use pre-created distribution for computing log probabilities and entropy
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        action_log_probs = distribution.log_prob(actions)
        
        # Compute entropy - the distribution handles masked actions correctly
        entropy = distribution.entropy()
        
        # Check for NaN/Inf in results (after distribution computation)
        if torch.isnan(action_log_probs).any() or torch.isinf(action_log_probs).any():
            raise RuntimeError("NaN or Inf detected in action_log_probs in evaluate_actions()." \
                                "Check action masks and logits."
                               f"\n action_log_probs: {action_log_probs}"
                               f"\n logits: {logits}"
                               f"\n action_mask: {action_mask}"
                               f"\n actions: {actions}")
        
        return values, action_log_probs, entropy
    
    def predict_values(self, obs: TensorDict) -> torch.Tensor:
        """
        Predict values for observations.
        
        Args:
            obs: Observations as TensorDict
        
        Returns:
            Value estimates (batch,)
        """
        obs_embeddings, _, _ = self._extract_features(obs)
        return self.value_net(obs_embeddings)


def create_actor_critic(
    embedder,
    embed_dim: int = 100,
    hidden_dim: int = 256,
    num_layers: int = 4,
    dropout_prob: float = 0.0,
    device: torch.device = None,
) -> ActorCriticPolicy:
    """
    Factory function to create an actor-critic policy.
    
    Args:
        embedder: Embedder to convert observations to embeddings
        embed_dim: Dimension of embeddings
        hidden_dim: Dimension of hidden layers
        num_layers: Number of residual blocks
        dropout_prob: Dropout probability
        device: PyTorch device
    
    Returns:
        ActorCriticPolicy instance
    """
    return ActorCriticPolicy(
        embedder=embedder,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout_prob=dropout_prob,
        device=device,
    )
