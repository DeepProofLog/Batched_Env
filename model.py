"""
model.py - Clean implementation that exactly mimics SB3's model logic.

This file is a simplified version of model.py that removes all batched-specific
optimizations and matches the SB3 implementation as closely as possible.

Architecture matches sb3/sb3_model.py:
- SharedBody: Shared residual MLP for both policy and value
- PolicyHead: Projects shared features to embedding space
- ValueHead: Produces scalar value estimates
- SharedPolicyValueNetwork: Combines shared body with separate heads
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Tuple, Optional, Dict
import gymnasium as gym
import numpy as np
from tensordict import TensorDict
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import MlpExtractor


class CustomCombinedExtractor(nn.Module):
    """
    Feature extractor that converts index-based observations into embeddings.
    Matches SB3's BaseFeaturesExtractor pattern exactly.
    """
    
    def __init__(self, embedder):
        super().__init__()
        self.embedder = embedder
        self.embed_dim = embedder.embedding_dim
        # Match SB3 CombinedExtractor contract
        self._features_dim = 1
        self.features_dim = 1
    
    def forward(self, obs: TensorDict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract embeddings from observations.
        
        Args:
            obs: TensorDict containing 'sub_index', 'derived_sub_indices', 'action_mask'
        
        Returns:
            Tuple of (obs_embeddings, action_embeddings, action_mask)
        """
        # Get tensors from TensorDict - exactly as in SB3
        obs_sub_indices = obs.get("sub_index")  # (batch, 1, pad_atoms, 3)
        action_sub_indices = obs.get("derived_sub_indices")  # (batch, pad_states, pad_atoms, 3)
        action_mask = obs.get("action_mask")  # (batch, pad_states)
        
        # Ensure correct dtype
        if obs_sub_indices.dtype != torch.int32:
            obs_sub_indices = obs_sub_indices.to(torch.int32)
        if action_sub_indices.dtype != torch.int32:
            action_sub_indices = action_sub_indices.to(torch.int32)
        
        # Get embeddings from embedder
        obs_embeddings = self.embedder.get_embeddings_batch(obs_sub_indices)
        action_embeddings = self.embedder.get_embeddings_batch(action_sub_indices)
        
        return obs_embeddings, action_embeddings, action_mask


class SharedBody(nn.Module):
    """Shared residual MLP body for both policy and value networks.
    Matches sb3/sb3_model.py SharedBody exactly."""
    def __init__(self, embed_dim=64, hidden_dim=256, num_layers=8, dropout_prob=0.0):
        super().__init__()
        # Initial transformation from embedding to hidden representation
        # NOTE: dropout_prob set to 0.0 by default to avoid train/eval mode inconsistencies
        # that cause issues with PPO's log probability computation and value function learning
        self.input_transform = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_prob)
        )
        
        # Residual blocks for deep feature extraction (shared between policy and value)
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_prob)
            ) for _ in range(num_layers)
        ])
        
        self.hidden_dim = hidden_dim

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply shared residual stack to embeddings."""
        original_shape = embeddings.shape
        flat_embeddings = embeddings.reshape(-1, original_shape[-1])
        x = self.input_transform(flat_embeddings)
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = x + residual
        return x.view(*original_shape[:-1], -1)


class PolicyHead(nn.Module):
    """Policy head that takes shared body output and produces action logits.
    Matches sb3/sb3_model.py PolicyHead exactly."""
    def __init__(self, hidden_dim=256, embed_dim=64):
        super().__init__()
        # Final transformation that projects the processed representation to embedding space
        self.out_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, shared_features: torch.Tensor) -> torch.Tensor:
        """Project shared features back to embedding dimension."""
        return self.out_transform(shared_features)


class ValueHead(nn.Module):
    """Value head that takes shared body output and produces scalar value estimates.
    Matches sb3/sb3_model.py ValueHead exactly."""
    def __init__(self, hidden_dim=256):
        super().__init__()
        # Output layers to produce a single scalar value estimate
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, shared_features: torch.Tensor) -> torch.Tensor:
        """Get value prediction from shared features."""
        value = self.output_layer(shared_features)
        return value.squeeze(-1)


class SharedPolicyValueNetwork(nn.Module):
    """Combined network with shared body for policy and value function.
    Matches sb3/sb3_model.py SharedPolicyValueNetwork exactly."""
    def __init__(self, embed_dim=64, hidden_dim=256, num_layers=8, dropout_prob=0.0):
        super().__init__()
        # Shared body for both policy and value
        # NOTE: dropout_prob=0.0 by default to avoid train/eval mode issues with PPO
        self.shared_body = SharedBody(embed_dim, hidden_dim, num_layers, dropout_prob)
        
        # Separate heads for policy and value
        self.policy_head = PolicyHead(hidden_dim, embed_dim)
        self.value_head = ValueHead(hidden_dim)

    def _encode_with_shared_body(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply shared body to embeddings."""
        return self.shared_body(embeddings)

    def forward_policy(self, obs_embeddings: torch.Tensor, action_embeddings: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for policy network."""
        # Process observation embeddings through shared body and policy head
        shared_obs = self._encode_with_shared_body(obs_embeddings)
        encoded_obs = self.policy_head(shared_obs)
        
        # Process action embeddings through shared body and policy head
        shared_actions = self._encode_with_shared_body(action_embeddings)
        encoded_actions = self.policy_head(shared_actions)
        
        # Compute similarity (dot product) between observation and action embeddings
        logits = torch.matmul(encoded_obs, encoded_actions.transpose(-2, -1)).squeeze(-2)
        # Apply logit scaling exactly as in sb3 - CRITICAL for parity
        logits = logits / (obs_embeddings.shape[-1] ** 0.5)
        
        # action_mask: (batch, pad_states) with 1 for valid slots
        logits = logits.masked_fill(~action_mask.bool(), float("-inf"))
        
        return logits

    def forward_value(self, obs_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass for value network."""
        # Process observation embeddings through shared body and value head
        shared_obs = self._encode_with_shared_body(obs_embeddings)
        value = self.value_head(shared_obs)
        return value


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function with shared body.
    It receives as input the features extracted by the feature extractor.
    Matches sb3/sb3_model.py CustomNetwork exactly.

    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    :param embed_dim: dimension of embeddings
    """
    def __init__(
        self,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 1, 
        embed_dim: int = 200
    ):
        super(CustomNetwork, self).__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Use shared policy-value network instead of separate networks
        self.shared_network = SharedPolicyValueNetwork(embed_dim)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward method for SB3 compatibility.
        Accepts features (which can include sub_indices for embedding) and outputs:
        - latent policy representation
        - latent value representation
        """
        # Assuming `features` is observation sub_indices passed here for embedding
        obs_embeddings, action_embeddings, action_mask = features
        probs = self.shared_network.forward_policy(obs_embeddings, action_embeddings, action_mask) # (batch_size=n_envs,pad_states)
        value = self.shared_network.forward_value(obs_embeddings).squeeze(-1) # (batch_size=n_envs)
        return probs, value
    
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward method for the actor network.
        Accepts features (which can include sub_indices for embedding) and outputs the latent policy representation.
        """
        obs_embeddings, action_embeddings, action_mask = features
        return self.shared_network.forward_policy(obs_embeddings, action_embeddings, action_mask)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward method for the critic network.
        Accepts features (which can include sub_indices for embedding) and outputs the latent value representation.
        """
        obs_embeddings, _, _ = features
        return self.shared_network.forward_value(obs_embeddings).squeeze(-1)

    # Expose value_network for compatibility with top-k filtering
    @property
    def value_network(self):
        """Property to expose value computation for top-k filtering compatibility."""
        return self._value_network_wrapper

    def _value_network_wrapper(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Wrapper to compute values from embeddings for top-k filtering."""
        return self.shared_network.forward_value(embeddings)


class ActorCriticPolicy(nn.Module):
    """
    Combined actor-critic policy that exactly matches SB3's ActorCriticPolicy.
    
    This is a clean implementation that removes all batched-specific optimizations
    and matches the SB3 logic line-by-line.
    """
    
    def __init__(
        self,
        embedder,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout_prob: float,
        device: torch.device,
        share_features_extractor: bool = True,
        init_seed: Optional[int] = None,
        action_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.device = device if device is not None else torch.device('cpu')
        self.share_features_extractor = share_features_extractor

        # Features extractor(s) – mirror SB3 CombinedExtractor contract and ordering
        self.features_extractor = CustomCombinedExtractor(embedder)
        if share_features_extractor:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.features_extractor
        else:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = CustomCombinedExtractor(embedder)

        # --- Base SB3-style scaffold (consumes RNG in the same order) ---
        # Build CustomNetwork as SB3 does inside _build_mlp_extractor (defaults)
        self.mlp_extractor = CustomNetwork(
            last_layer_dim_pi=self.features_extractor._features_dim,
            last_layer_dim_vf=1,
            embed_dim=getattr(self.features_extractor, "embed_dim", embed_dim),
        )
        # Dummy action/value heads created during _build in SB3
        latent_pi = getattr(self.mlp_extractor, "latent_dim_pi", 1)
        latent_vf = getattr(self.mlp_extractor, "latent_dim_vf", 1)
        action_dim = int(action_dim) if action_dim is not None else -1
        self.action_net = nn.Linear(latent_pi, action_dim if action_dim > 0 else 1)
        self.value_net = nn.Linear(latent_vf, 1)

        # SB3-style orthogonal initialization sequence (match ActorCriticPolicy._build)
        module_gains = {
            self.features_extractor: math.sqrt(2.0),
            self.mlp_extractor: math.sqrt(2.0),
            self.action_net: 0.01,
            self.value_net: 1.0,
        }
        if not self.share_features_extractor:
            del module_gains[self.features_extractor]
            module_gains[self.pi_features_extractor] = math.sqrt(2.0)
            module_gains[self.vf_features_extractor] = math.sqrt(2.0)
        for module, gain in module_gains.items():
            module.apply(partial(BasePolicy.init_weights, gain=gain))

        # Action distribution - now aligned with SB3 action_dim
        self.action_dist = CategoricalDistribution(action_dim)

        # Move everything to device
        self.to(self.device)
    
    def extract_features(
        self, 
        obs: TensorDict,
        features_extractor: Optional[CustomCombinedExtractor] = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        """
        Extract features from observations - matches SB3 exactly.
        """
        if self.share_features_extractor:
            extractor = self.features_extractor if features_extractor is None else features_extractor
            return extractor(obs)
        else:
            # Extract features separately for policy and value
            pi_features = self.pi_features_extractor(obs)
            vf_features = self.vf_features_extractor(obs)
            return pi_features, vf_features
    
    def forward(
        self,
        obs: TensorDict,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass - exactly matches SB3's forward.
        
        Args:
            obs: Observations as TensorDict
            deterministic: If True, select argmax action instead of sampling
        
        Returns:
            Tuple of (actions, values, log_probs)
        """
        # Extract features following sb3 pattern
        features = self.extract_features(obs)
        if self.share_features_extractor:
            logits, values = self.mlp_extractor(features)
        else:
            (obs_embeddings, action_embeddings, action_mask), (obs_embeddings_vf, _, _) = features
            logits = self.mlp_extractor.forward_actor((obs_embeddings, action_embeddings, action_mask))
            values = self.mlp_extractor.forward_critic((obs_embeddings_vf, None, None))

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
        Evaluate actions according to current policy - matches SB3 exactly.
        
        Args:
            obs: Observations as TensorDict
            actions: Actions to evaluate
        
        Returns:
            Tuple of (values, log_probs, entropy)
        """
        # Extract features
        features = self.extract_features(obs)

        if self.share_features_extractor:
            logits, values = self.mlp_extractor(features)
        else:
            (obs_embeddings, action_embeddings, action_mask), (obs_embeddings_vf, _, _) = features
            logits = self.mlp_extractor.forward_actor((obs_embeddings, action_embeddings, action_mask))
            values = self.mlp_extractor.forward_critic((obs_embeddings_vf, None, None))

        # Create distribution and evaluate - exactly as in SB3
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        action_log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        return values, action_log_probs, entropy
    
    def predict_values(self, obs: TensorDict) -> torch.Tensor:
        """
        Predict values for observations - matches SB3.
        """
        features = self.extract_features(obs, self.vf_features_extractor)
        if self.share_features_extractor:
            _, values = self.mlp_extractor(features)
        else:
            (_, _, _), (obs_embeddings_vf, _, _) = features
            values = self.mlp_extractor.forward_critic((obs_embeddings_vf, None, None))
        return values


def create_actor_critic(
    embedder,
    embed_dim: int = 100,
    hidden_dim: int = 256,
    num_layers: int = 4,
    dropout_prob: float = 0.0,
    device: torch.device = None,
    padding_atoms: Optional[int] = None,
    padding_states: Optional[int] = None,
    max_arity: Optional[int] = None,
    total_vocab_size: Optional[int] = None,
    match_sb3_init: bool = False,
    init_seed: Optional[int] = None,
    **kwargs
) -> ActorCriticPolicy:
    """
    Factory function to create an ActorCriticPolicy.
    Matches the interface used in train.py.
    """
    policy = ActorCriticPolicy(
        embedder=embedder,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout_prob=dropout_prob,
        device=device,
        share_features_extractor=kwargs.get('share_features_extractor', True),
        init_seed=init_seed,
        action_dim=int(padding_states) if padding_states is not None else None,
    )
    return policy

