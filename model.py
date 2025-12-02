"""
model_new.py - Clean implementation that exactly mimics SB3's model logic.

This file is a simplified version of model.py that removes all batched-specific
optimizations and matches the SB3 implementation as closely as possible.
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


class PolicyNetwork(nn.Module):
    """
    Policy network that produces logits from observation and action embeddings.
    Matches SB3's implementation exactly - NO SCALING, simple processing.
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
        
        # Residual blocks - match SB3 exactly
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
    
    def _forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        x = self.obs_transform(x)
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = x + residual
        return self.out_transform(x)

    def _encode_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Process embeddings through residual network.
        Matches SB3 exactly - process all embeddings, no masking optimization.
        """
        original_shape = embeddings.shape
        flat_embeddings = embeddings.reshape(-1, original_shape[-1])
        encoded = self._forward_mlp(flat_embeddings)
        return encoded.view(*original_shape[:-1], -1)
    
    def forward(
        self,
        obs_embeddings: torch.Tensor,
        action_embeddings: torch.Tensor,
        action_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute action logits - exactly as in SB3.
        
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
        
        # Compute similarity (dot product) between observation and action embeddings
        # Exactly as in SB3: matmul + squeeze, NO SCALING
        logits = torch.matmul(encoded_obs, encoded_actions.transpose(-2, -1)).squeeze(-2)
        
        # Apply action mask
        logits = logits.masked_fill(~action_mask.bool(), float("-inf"))
        
        return logits


class ValueNetwork(nn.Module):
    """
    Value network that maps observation embeddings to scalar value estimates.
    Matches SB3 implementation exactly.
    """
    
    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 8,
        dropout_prob: float = 0.2
    ):
        super().__init__()
        
        # Initial transformation - match SB3 exactly
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
        Compute value estimate from observation embeddings.
        Matches SB3 exactly.
        """
        # Process observation embeddings through the input layer
        x = self.input_layer(embeddings)
        # Pass through residual blocks
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = x + residual
        # Get final value prediction
        value = self.output_layer(x)
        return value.squeeze(-1)


class CustomNetwork(nn.Module):
    """
    Minimal wrapper matching SB3's CustomNetwork: holds policy/value towers and
    exposes forward_actor/forward_critic helpers while tracking latent dims.
    """

    def __init__(
        self,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 1,
        embed_dim: int = 200,
    ):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.policy_network = PolicyNetwork(embed_dim)
        self.value_network = ValueNetwork(embed_dim)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_embeddings, action_embeddings, action_mask = features
        probs = self.policy_network(obs_embeddings, action_embeddings, action_mask)
        value = self.value_network(obs_embeddings).squeeze(-1)
        return probs, value

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        obs_embeddings, action_embeddings, action_mask = features
        return self.policy_network(obs_embeddings, action_embeddings, action_mask)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        obs_embeddings, _, _ = features
        return self.value_network(obs_embeddings).squeeze(-1)


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

        # Separate custom_network mirror (unused in forward but present in sb3 state_dict)
        self.custom_network = CustomNetwork(
            last_layer_dim_pi=hidden_dim,
            last_layer_dim_vf=1,
            embed_dim=getattr(self.features_extractor, "embed_dim", embed_dim),
        )
        self.custom_network.policy_network = PolicyNetwork(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_prob=dropout_prob,
        )
        self.custom_network.value_network = ValueNetwork(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_prob=dropout_prob,
        )

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

